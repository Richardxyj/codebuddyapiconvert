"""
responses_projection — /v1/responses 的后端投影层。

目标
----
Codex CLI 会把大量运行时提示、完整工具 schema、长历史、以及工具输出一并塞进
/v1/responses 请求里。腾讯后端对这类 agentic payload 很容易触发内容审核，或者
因为上下文过长而表现不稳定。

本模块在保持外部 OpenAI Responses 兼容的前提下，只对发往后端的 Chat body 做
"最小语义闭包"投影：

- 固定短 system 摘要替换 Codex/Claude Code harness
- 保留最新用户意图
- 保留最近一段真实 assistant/tool 链路
- 把更早历史压缩成规则摘要
- 把 tool schema 收敛成结构字段
- 把超长 tool output / tool arguments 压缩成可继续推理的摘要
"""

from __future__ import annotations

import json
import re
from typing import Any


AGENTIC_TOOL_NAMES = {
    "exec_command",
    "write_stdin",
    "update_plan",
    "request_user_input",
    "view_image",
    "get_goal",
    "create_goal",
    "update_goal",
    "apply_patch",
    "tool_search_tool",
}

HARNESS_USER_MARKERS = (
    "# AGENTS.md instructions",
    "<environment_context>",
    "<permissions instructions>",
    "<collaboration_mode>",
    "<skills_instructions>",
    "<system-reminder>",
    "# claudeMd",
)

HARNESS_SYSTEM_MARKERS = (
    "You are a coding agent running in the Codex CLI",
    "Within this context, Codex refers to",
    "# AGENTS.md spec",
    "<permissions instructions>",
    "<collaboration_mode>",
    "<skills_instructions>",
    "The following deferred tools are now available via ToolSearch.",
    "### Available skills",
    "## request_user_input availability",
    "You are Claude Code",
)

BASE_SYSTEM_PROMPT = (
    "You are a coding assistant serving an OpenAI-compatible CLI. "
    "Be precise, concise, safe, and action-oriented. "
    "Use available tools when needed, follow repository instructions and durable user context, "
    "and continue from the preserved recent context. "
    "If earlier history was condensed, rely on the preserved recent messages and rerun tools when exact old details are required."
)

HISTORY_PREFIX = "Earlier conversation summary (condensed):"

# 本地补丁 #6（2026-09-18）：尺寸感知投影
# 原版对所有 agentic 请求一律激进压缩（历史压成 ~2200 字符摘要 + 尾部 8 条），
# 长会话的 agent 工作记忆被毁（反复重读文件、死循环）。
# 新策略：payload 能装进模型上下文时原样直通（保真）；装不下才走 aggressive 压缩。
# 补丁 #13（2026-09-20）：直通预算对齐 Codex 原生几何 —— 详见 docs/21
#   Codex 原生算术（codex-rs/core/src/session/context_window.rs + protocol/openai_models.rs）：
#     resolved_ctx = context_window ?? max_context_window
#     可用窗口     = resolved_ctx × effective_context_window_percent / 100   (默认 95)
#     压缩阈值     = auto_compact_token_limit 为空时 = resolved_ctx × 9 / 10  (90%)
#     token_limit_reached = used >= 阈值 || used >= 可用窗口
#   关键不变量：**上游能吞下的输入 ≥ Codex 压缩阈值 + 单轮增长**。
#   违反它就会出现「已越过阈值、却被投影先肢解」的静默失忆窗口（docs/20 实测占生产 24.8%）。
#   本表的数字 = 目录 model-catalogs/relay-mu9j3868.json 声明的真实窗口（一把尺）；
#   预算 = 窗口 − 5% headroom = 原生 usable 窗口口径。
#   实测（2026-09-20 天花板探针 probe_ceiling.py）：verbatim 250,051 token → HTTP 200，3/3。
MODEL_CONTEXT_TOKENS = {
    "kimi-k3-1": 256000,          # 原 1,000,000 → 对齐目录声明
    "kimi-k2.8-preview": 256000,  # 已一致
    "glm-5.3": 256000,            # 原 128,000（失忆带主因）
    "deepseek-v4-flash": 256000,  # 原 128,000
}
DEFAULT_MODEL_CONTEXT_TOKENS = 128000  # 未登记模型的保守兜底（故意不抬）
OUTPUT_HEADROOM_TOKENS = 12800  # 原 32,000；改 5% of 256K，对齐原生 usable 窗口(95%)口径

# 修复（2026-09-19 审查）：固定 2.2 chars/token 的预算对纯中文负载严重低估 token 数
# （CJK ≈ 1 token/字 → 210 万中文字符可能 ≈ 200 万 token，直通会超出 kimi 1M 上下文）。
# 改为分文种估算：CJK≈1 token/字，其它≈1 token/3 字符。纯 ASCII 比旧算法略宽松（3.0）。
_CJK_RE = re.compile(r"[\u2e80-\u9fff\uf900-\ufaff\uff00-\uffef\u3000-\u303f]")


def _context_budget_tokens(model: str | None) -> int:
    tokens = MODEL_CONTEXT_TOKENS.get(model or "", DEFAULT_MODEL_CONTEXT_TOKENS)
    return max(tokens - OUTPUT_HEADROOM_TOKENS, 24000)


def _estimate_tokens(text: str) -> float:
    if not text:
        return 0.0
    cjk = len(_CJK_RE.findall(text))
    return cjk + (len(text) - cjk) / 3.0


def _messages_token_estimate(messages: list) -> float:
    total = 0.0
    for msg in messages:
        if isinstance(msg, dict):
            total += _estimate_tokens(_content_to_text(msg.get("content", "")))
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    fn = tc.get("function") or {}
                    total += _estimate_tokens(str(fn.get("arguments", "")))
    return total


def _tools_size(tools: list) -> int:
    if not tools:
        return 0
    try:
        return len(json.dumps(tools, ensure_ascii=False))
    except Exception:
        return 0


def pair_token_estimate(messages: list, tools: list) -> int:
    """messages + tools 的 CJK 感知 token 估算（补丁 #12，投影前/后通用）。"""
    try:
        tools_json = json.dumps(tools, ensure_ascii=False) if tools else ""
    except Exception:
        tools_json = ""
    return int(round(_messages_token_estimate(messages) + _estimate_tokens(tools_json)))

MAX_SYSTEM_GUIDANCE_CHARS = 1200
MAX_USER_CHARS = 3200
MAX_ASSISTANT_CHARS = 1800
# 修订 2（2026-09-20）：1600 会把文件读取类 tool output 肢解成 head10+[omitted]+tail6，
# 模型看到的就是"输出被从中间截断"→ 同一文件反复重读（实证：单会话 156 次读同一
# SKILL.md、4 小时死循环）。≤12000 字符的工具输出原样直通，更大的仍走 head/tail 摘要。
MAX_TOOL_OUTPUT_CHARS = 12000
MAX_TOOL_ARGS_CHARS = 900
MAX_HISTORY_SUMMARY_CHARS = 2200
MAX_HISTORY_ITEMS = 10
MAX_TAIL_MESSAGES = 8
# 修订 2（2026-09-20）：与 MAX_TOOL_OUTPUT_CHARS 联动——tail 必须装得下"一条
# 12K 工具输出 + 其调用链 + 最近对话"，否则最近 tool 链会被挤出 tail
# （tests/test_responses_adapter.py 的 recent-chain 不变量守的就是这条）。
MAX_TAIL_CHARS = 20000

SCHEMA_KEEP_KEYS = {
    "type",
    "properties",
    "required",
    "items",
    "enum",
    "oneOf",
    "anyOf",
    "allOf",
    "additionalProperties",
    "format",
    "minimum",
    "maximum",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "nullable",
}


def project_responses_chat_body(body: dict) -> tuple[dict, dict]:
    """把 Responses 转出来的 Chat body 投影成更适合腾讯后端的最小上下文。"""
    projected = dict(body)
    messages = list(body.get("messages") or [])
    tools = list(body.get("tools") or [])

    # 补丁 #6（2026-09-19 审查修订）：尺寸感知——token 估算 ≤ 预算才直通（CJK 感知）
    total_chars = _messages_size(messages) + _tools_size(tools)
    tools_json = json.dumps(tools, ensure_ascii=False) if tools else ""
    token_estimate = _messages_token_estimate(messages) + _estimate_tokens(tools_json)
    budget = _context_budget_tokens(body.get("model"))

    # 补丁 #12（2026-09-20）：对外暴露"投影前真实体积"。
    # 动机：Codex 用响应 usage.input_tokens 管理上下文预算与 auto-compact。投影把发往后端
    # 的 payload 从数百万字符压到几万字符，后端回报的 prompt_tokens 只剩几千 → 远端
    # 200K 压缩阈值永远够不着，会话无限膨胀（实测单会话 52MB ≈ 440 万 token）。
    # 这里在投影前算一次真实估算值，随 stats 一并返回，供 converter 按真实体积上报。
    original_token_estimate = int(round(token_estimate))
    if token_estimate <= budget:
        return projected, {
            "mode": "passthrough",
            "aggressive": False,
            "context_budget_tokens": budget,
            "token_estimate": round(token_estimate),
            "original_token_estimate": original_token_estimate,
            "projected_token_estimate": original_token_estimate,
            "original_messages": len(messages),
            "projected_messages": len(messages),
            "original_message_chars": total_chars,
            "projected_message_chars": total_chars,
        }

    projected_tools, tool_stats = _project_tools(tools)
    if projected_tools:
        projected["tools"] = projected_tools
    elif "tools" in projected:
        projected["tools"] = []

    aggressive = _looks_like_agentic_cli(messages, tools)
    if not aggressive:
        projected["messages"] = _project_messages_conservative(messages)
        return projected, {
            "mode": "conservative",
            "aggressive": False,
            "original_token_estimate": original_token_estimate,
            "projected_token_estimate": pair_token_estimate(
                projected["messages"], projected.get("tools") or []
            ),
            "original_messages": len(messages),
            "projected_messages": len(projected["messages"]),
            "original_message_chars": _messages_size(messages),
            "projected_message_chars": _messages_size(projected["messages"]),
            **tool_stats,
        }

    tool_name_by_call_id = _build_tool_call_name_map(messages)
    preserved_guidance: list[str] = []
    conversation: list[dict] = []
    dropped_harness_messages = 0

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role")
        text = _content_to_text(msg.get("content", ""))

        if role == "system":
            if _looks_like_harness_system(text):
                dropped_harness_messages += 1
                continue
            guidance = _truncate_text(text, MAX_SYSTEM_GUIDANCE_CHARS)
            if guidance:
                preserved_guidance.append(guidance)
            continue

        if role == "user" and _looks_like_harness_user(text):
            dropped_harness_messages += 1
            continue

        projected_msg = _project_conversation_message(msg)
        if projected_msg is not None:
            conversation.append(projected_msg)

    if not conversation:
        conversation = _project_messages_conservative(messages)

    tail_start = _choose_tail_start(conversation)
    tail_start = _expand_tail_for_tool_context(conversation, tail_start)
    latest_user_idx = _latest_user_index(conversation)

    anchor_user = None
    if latest_user_idx is not None and latest_user_idx < tail_start:
        anchor_user = dict(conversation[latest_user_idx])

    omitted: list[dict] = []
    for idx, msg in enumerate(conversation):
        if idx >= tail_start:
            break
        if latest_user_idx is not None and idx == latest_user_idx and anchor_user is not None:
            continue
        omitted.append(msg)

    final_messages: list[dict] = [{"role": "system", "content": BASE_SYSTEM_PROMPT}]
    guidance_message = _merge_guidance_messages(preserved_guidance)
    if guidance_message:
        final_messages.append({"role": "system", "content": guidance_message})

    history_summary = _build_history_summary(omitted, tool_name_by_call_id)
    if history_summary:
        final_messages.append({"role": "system", "content": history_summary})

    if anchor_user is not None:
        final_messages.append(anchor_user)

    final_messages.extend(conversation[tail_start:])
    projected["messages"] = final_messages

    return projected, {
        "mode": "aggressive",
        "aggressive": True,
        "context_budget_tokens": budget,
        "original_token_estimate": original_token_estimate,
        "projected_token_estimate": pair_token_estimate(
            final_messages, projected.get("tools") or []
        ),
        "dropped_harness_messages": dropped_harness_messages,
        "preserved_guidance_messages": len(preserved_guidance),
        "summarized_history_messages": len(omitted),
        "anchor_user_preserved": anchor_user is not None,
        "tail_messages": len(conversation[tail_start:]),
        "original_messages": len(messages),
        "projected_messages": len(final_messages),
        "original_message_chars": _messages_size(messages),
        "projected_message_chars": _messages_size(final_messages),
        **tool_stats,
    }


def _looks_like_agentic_cli(messages: list[dict], tools: list[dict]) -> bool:
    tool_names = {
        _tool_name(tool)
        for tool in tools
        if _tool_name(tool)
    }
    if tool_names & AGENTIC_TOOL_NAMES:
        return True

    for msg in messages:
        if not isinstance(msg, dict):
            continue
        text = _content_to_text(msg.get("content", ""))
        if _looks_like_harness_user(text) or _looks_like_harness_system(text):
            return True
    return False


def _project_messages_conservative(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for msg in messages:
        projected = _project_conversation_message(msg, conservative=True)
        if projected is not None:
            out.append(projected)
    return out


def _project_conversation_message(msg: dict, conservative: bool = False) -> dict | None:
    if not isinstance(msg, dict):
        return None

    role = msg.get("role")
    out = dict(msg)

    if role == "system":
        text = _content_to_text(msg.get("content", ""))
        out["content"] = _truncate_text(text, MAX_SYSTEM_GUIDANCE_CHARS)
        return out

    if role == "user":
        text = _content_to_text(msg.get("content", ""))
        out["content"] = _truncate_text(text, MAX_USER_CHARS)
        return out

    if role == "assistant":
        text = _content_to_text(msg.get("content", ""))
        out["content"] = _summarize_free_text(text, MAX_ASSISTANT_CHARS)
        tool_calls = []
        for tool_call in msg.get("tool_calls") or []:
            projected_call = _project_tool_call(tool_call)
            if projected_call is not None:
                tool_calls.append(projected_call)
        if tool_calls:
            out["tool_calls"] = tool_calls
        elif "tool_calls" in out:
            out.pop("tool_calls", None)
        return out

    if role == "tool":
        out["content"] = _summarize_tool_output(_content_to_text(msg.get("content", "")))
        return out

    if conservative:
        text = _content_to_text(msg.get("content", ""))
        out["content"] = _truncate_text(text, MAX_ASSISTANT_CHARS)
        return out

    return None


def _project_tool_call(tool_call: dict) -> dict | None:
    if not isinstance(tool_call, dict):
        return None

    function = tool_call.get("function") or {}
    name = function.get("name", "")
    arguments = function.get("arguments", "")

    return {
        "id": tool_call.get("id"),
        "type": tool_call.get("type", "function"),
        "function": {
            "name": name,
            "arguments": _summarize_tool_arguments(name, arguments),
        },
    }


def _summarize_tool_arguments(name: str, arguments: Any) -> str:
    if not isinstance(arguments, str):
        try:
            return json.dumps(arguments, ensure_ascii=False)
        except Exception:
            return json.dumps({"summary": _truncate_text(str(arguments), 240)}, ensure_ascii=False)

    if len(arguments) <= MAX_TOOL_ARGS_CHARS:
        return arguments

    if name == "apply_patch":
        return json.dumps(
            {"summary": "Large apply_patch payload omitted; a patch was prepared or applied in a previous step."},
            ensure_ascii=False,
        )

    try:
        parsed = json.loads(arguments)
    except Exception:
        return json.dumps({"summary": _truncate_text(arguments, 320)}, ensure_ascii=False)

    return json.dumps(_shrink_json_value(parsed), ensure_ascii=False)


def _shrink_json_value(value: Any, depth: int = 0, key: str = "") -> Any:
    if depth >= 4:
        return "<omitted>"

    if isinstance(value, dict):
        out = {}
        items = list(value.items())
        for idx, (item_key, item_value) in enumerate(items):
            if idx >= 12:
                out["_omitted_keys"] = len(items) - idx
                break
            out[item_key] = _shrink_json_value(item_value, depth + 1, item_key)
        return out

    if isinstance(value, list):
        trimmed = [_shrink_json_value(item, depth + 1, key) for item in value[:6]]
        if len(value) > 6:
            trimmed.append(f"<omitted {len(value) - 6} items>")
        return trimmed

    if isinstance(value, str):
        limit = 240 if key in {"cmd", "chars", "patch", "content", "text", "question"} else 120
        return _truncate_text(value, limit)

    return value


def _project_tools(tools: list[dict]) -> tuple[list[dict], dict]:
    projected = []
    original_chars = _tools_size(tools)

    for tool in tools:
        if not isinstance(tool, dict):
            continue

        if tool.get("type") != "function":
            # 补丁 #14（2026-09-21，docs/25 R1·A1）：非 function 工具不再静默丢弃。
            # 投影无法为其瘦身 schema，但守恒优先于瘦身——原样保留进出。
            # （Responses 主链路上游已由 _convert_tools_for_chat 统一包装为 function，
            #  此分支是防其他入口直接喂 Chat 格式时的回归。）
            projected.append(tool)
            continue

        function = tool.get("function") or tool
        name = function.get("name")
        if not name:
            continue

        projected_function: dict[str, Any] = {"name": name}
        if "parameters" in function:
            projected_function["parameters"] = _project_schema(function.get("parameters"))
        if "strict" in function:
            projected_function["strict"] = function.get("strict")

        projected.append({"type": "function", "function": projected_function})

    return projected, {
        "original_tools": len(tools),
        "projected_tools": len(projected),
        "original_tool_chars": original_chars,
        "projected_tool_chars": _tools_size(projected),
    }


def _project_schema(schema: Any, depth: int = 0) -> Any:
    if depth >= 6:
        return {"type": "object"}

    if isinstance(schema, dict):
        out: dict[str, Any] = {}
        for key, value in schema.items():
            if key not in SCHEMA_KEEP_KEYS:
                continue
            if key == "properties" and isinstance(value, dict):
                out["properties"] = {
                    prop: _project_schema(prop_schema, depth + 1)
                    for prop, prop_schema in value.items()
                }
            elif key == "items":
                out["items"] = _project_schema(value, depth + 1)
            elif key in {"oneOf", "anyOf", "allOf"} and isinstance(value, list):
                out[key] = [_project_schema(item, depth + 1) for item in value[:6]]
            elif key == "additionalProperties" and isinstance(value, dict):
                out[key] = _project_schema(value, depth + 1)
            else:
                out[key] = value
        return out or {"type": "object"}

    if isinstance(schema, list):
        return [_project_schema(item, depth + 1) for item in schema[:6]]

    return schema


def _choose_tail_start(messages: list[dict]) -> int:
    if not messages:
        return 0

    start = len(messages) - 1
    total_chars = 0
    kept = 0

    for idx in range(len(messages) - 1, -1, -1):
        cost = _message_cost(messages[idx])
        if kept > 0 and (kept >= MAX_TAIL_MESSAGES or total_chars + cost > MAX_TAIL_CHARS):
            break
        start = idx
        total_chars += cost
        kept += 1
    return start


def _expand_tail_for_tool_context(messages: list[dict], start: int) -> int:
    if start <= 0 or not messages:
        return start

    needed_call_ids = {
        msg.get("tool_call_id")
        for msg in messages[start:]
        if isinstance(msg, dict) and msg.get("role") == "tool" and msg.get("tool_call_id")
    }
    if not needed_call_ids:
        return start

    expanded = start
    for idx in range(start - 1, -1, -1):
        msg = messages[idx]
        if msg.get("role") != "assistant":
            continue
        call_ids = {
            tool_call.get("id")
            for tool_call in msg.get("tool_calls") or []
            if isinstance(tool_call, dict)
        }
        if call_ids & needed_call_ids:
            expanded = idx
            needed_call_ids -= call_ids
            if not needed_call_ids:
                break
    return expanded


def _latest_user_index(messages: list[dict]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        if messages[idx].get("role") == "user":
            return idx
    return None


def _build_history_summary(messages: list[dict], tool_name_by_call_id: dict[str, str]) -> str:
    lines: list[str] = []
    total_chars = 0
    summarized = 0

    for msg in messages:
        line = _history_line(msg, tool_name_by_call_id)
        if not line:
            continue
        if summarized >= MAX_HISTORY_ITEMS or total_chars + len(line) > MAX_HISTORY_SUMMARY_CHARS:
            break
        lines.append(f"- {line}")
        total_chars += len(line)
        summarized += 1

    remaining = len(messages) - summarized
    if remaining > 0:
        lines.append(f"- {remaining} earlier messages or tool results were further condensed.")

    if not lines:
        return ""
    return HISTORY_PREFIX + "\n" + "\n".join(lines)


def _history_line(msg: dict, tool_name_by_call_id: dict[str, str]) -> str:
    role = msg.get("role")
    text = _content_to_text(msg.get("content", ""))

    if role == "user":
        return f"User asked: {_truncate_text(text, 220)}"

    if role == "assistant":
        tool_names = [
            (tool_call.get("function") or {}).get("name")
            for tool_call in msg.get("tool_calls") or []
            if isinstance(tool_call, dict)
        ]
        tool_names = [name for name in tool_names if name]
        if text and tool_names:
            return f"Assistant replied: {_truncate_text(text, 160)} Then called tools: {', '.join(tool_names[:4])}."
        if tool_names:
            return f"Assistant called tools: {', '.join(tool_names[:4])}."
        if text:
            return f"Assistant replied: {_truncate_text(text, 180)}"
        return ""

    if role == "tool":
        tool_name = tool_name_by_call_id.get(msg.get("tool_call_id", ""), "tool")
        summary = _tool_output_inline_summary(text)
        return f"Tool {tool_name} returned: {summary}"

    if role == "system":
        return f"System guidance: {_truncate_text(text, 180)}"

    return ""


def _build_tool_call_name_map(messages: list[dict]) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for msg in messages:
        if not isinstance(msg, dict) or msg.get("role") != "assistant":
            continue
        for tool_call in msg.get("tool_calls") or []:
            if not isinstance(tool_call, dict):
                continue
            call_id = tool_call.get("id")
            name = (tool_call.get("function") or {}).get("name")
            if call_id and name:
                mapping[call_id] = name
    return mapping


def _merge_guidance_messages(messages: list[str]) -> str:
    merged: list[str] = []
    total = 0
    for message in messages[:2]:
        text = message.strip()
        if not text:
            continue
        if total + len(text) > MAX_SYSTEM_GUIDANCE_CHARS:
            text = _truncate_text(text, MAX_SYSTEM_GUIDANCE_CHARS - total)
        merged.append(text)
        total += len(text)
        if total >= MAX_SYSTEM_GUIDANCE_CHARS:
            break
    if not merged:
        return ""
    if len(merged) == 1:
        return merged[0]
    return "Additional instructions:\n" + "\n\n".join(merged)


def _summarize_tool_output(text: str) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    # 修订 2（2026-09-20）：去掉 ≤24 行门槛——行数多但体量小的文件读取（如 58 行
    # SKILL.md ≈ 4.3KB）曾被强制走 head/tail 摘要，造成"中间截断"假象。
    if len(text) <= MAX_TOOL_OUTPUT_CHARS:
        return text

    lines = text.splitlines()
    exit_line = next((line.strip() for line in lines if "Process exited with code" in line), "")
    useful_lines = []
    saw_output = False
    for line in lines:
        stripped = line.rstrip()
        if stripped == "Output:":
            saw_output = True
            continue
        if (
            stripped.startswith("Chunk ID:")
            or stripped.startswith("Wall time:")
            or stripped.startswith("Original token count:")
            or stripped.startswith("Process exited with code")
        ):
            continue
        useful_lines.append(stripped)

    body_lines = useful_lines

    head = body_lines[:10]
    tail = body_lines[-6:] if len(body_lines) > 16 else []
    omitted = max(len(body_lines) - len(head) - len(tail), 0)

    parts: list[str] = []
    if exit_line:
        parts.append(exit_line)
    if head:
        parts.append("Key output:")
        parts.extend(head)
    if omitted:
        parts.append(f"... [omitted {omitted} lines] ...")
    if tail:
        parts.append("Recent tail:")
        parts.extend(tail)

    summary = "\n".join(part for part in parts if part).strip()
    return _truncate_text(summary or text, MAX_TOOL_OUTPUT_CHARS)


def _tool_output_inline_summary(text: str) -> str:
    summarized = _summarize_tool_output(text)
    summarized = summarized.replace("\n", " | ")
    return _truncate_text(summarized, 220)


def _summarize_free_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text

    head = text[: limit // 2].rstrip()
    tail = text[-(limit // 3):].lstrip()
    omitted = len(text) - len(head) - len(tail)
    return f"{head}\n... [{omitted} chars omitted] ...\n{tail}"


def _truncate_text(text: str, limit: int) -> str:
    text = (text or "").strip()
    if not text:
        return ""
    if len(text) <= limit:
        return text
    return text[: max(limit - 24, 0)].rstrip() + f" ... [truncated {len(text) - max(limit - 24, 0)} chars]"


def _content_to_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if "text" in block:
                    parts.append(str(block.get("text", "")))
                elif "output" in block:
                    parts.append(str(block.get("output", "")))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def _looks_like_harness_user(text: str) -> bool:
    return any(marker in text for marker in HARNESS_USER_MARKERS)


def _looks_like_harness_system(text: str) -> bool:
    return any(marker in text for marker in HARNESS_SYSTEM_MARKERS)


def _message_cost(msg: dict) -> int:
    cost = len(_content_to_text(msg.get("content", "")))
    for tool_call in msg.get("tool_calls") or []:
        if not isinstance(tool_call, dict):
            continue
        function = tool_call.get("function") or {}
        cost += len(function.get("name", ""))
        cost += len(function.get("arguments", ""))
    return cost


def _messages_size(messages: list[dict]) -> int:
    total = 0
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        total += _message_cost(msg)
        total += len(msg.get("role", ""))
    return total


def _tool_name(tool: dict) -> str:
    if not isinstance(tool, dict):
        return ""
    function = tool.get("function") or tool
    return str(function.get("name", "") or "")


def _tools_size(tools: list[dict]) -> int:
    try:
        return len(json.dumps(tools, ensure_ascii=False))
    except Exception:
        return 0
