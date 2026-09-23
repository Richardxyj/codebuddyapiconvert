"""
responses_adapter.py — OpenAI Responses API ↔ Chat Completions API 适配层。

Codex CLI 使用 Responses API（POST /v1/responses），而 CodeBuddy 后端只支持
Chat Completions 协议。本模块做双向转换：
  请求：Responses input/instructions/tools → Chat messages/tools
  响应：Chat SSE delta → Responses 语义事件流（response.created / output_text.delta / …）

事件类型参考：https://developers.openai.com/api/docs/guides/streaming-responses
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Any

# ---------------------------------------------------------------------------
# ID 生成
# ---------------------------------------------------------------------------

def _rand_id(prefix: str = "resp_") -> str:
    return prefix + os.urandom(12).hex()


def _warn_dropped_line(line: str) -> None:
    """修复 F2（2026-09-19 审查）：畸形/非 dict SSE 行不再静默丢弃，留 ⚠ 日志便于排障。

    优先写入 converter 的日志文件；包上下文不可用时退化为 stderr 打印。
    """
    msg = f"⚠ 丢弃无法解析的 SSE 行: {line[:200]!r}"
    try:
        from .converter import _log  # 延迟导入，避免模块级循环依赖

        _log(msg)
    except Exception:
        print(msg, file=sys.stderr)

# ---------------------------------------------------------------------------
# 请求转换：Responses → Chat
# ---------------------------------------------------------------------------

def _convert_images_for_chat(obj):
    """本地补丁 #10 v2（2026-09-18）：Responses 图片部件 → Chat image_url 格式。

    实测后端全部模型支持 Chat 格式图片输入（image_url 对象，ZEBRA-42 测试
    5 模型全命中）。原版把 Responses 的 input_image 原样透传给 Chat 后端，
    触发 400 "unsupported content type: input_image"。此函数深度遍历做部件
    格式翻译。
    """
    if isinstance(obj, list):
        return [_convert_images_for_chat(x) for x in obj]
    if isinstance(obj, dict):
        t = obj.get("type")
        if t == "input_image":
            url = obj.get("image_url")
            if isinstance(url, str):
                return {"type": "image_url", "image_url": {"url": url}}
            if isinstance(url, dict):
                return {"type": "image_url", "image_url": url}
            return {"type": "text", "text": "[图片无效：缺少 image_url]"}
        return {k: _convert_images_for_chat(v) for k, v in obj.items()}
    return obj


def _convert_user_content(content, norm_log: list | None = None):
    """本地补丁 #10 v2：user 消息 content 转换。

    含非纯文本部件时保留数组结构（Chat 多模态格式），纯文本仍返回字符串
    （与原版行为一致）。图片部件需先经 _convert_images_for_chat 翻译；
    Chat 不支持的 Responses 部件显式降级为 JSON 文本并记入 norm_log。
    """
    if not isinstance(content, list):
        return _extract_content(content)

    chat_legal_types = {"text", "image_url", "input_text", "output_text"}
    has_non_text = any(
        not isinstance(p, dict) or p.get("type") not in chat_legal_types
        for p in content
    )
    if not has_non_text:
        return _extract_content(content)

    parts = []
    degraded_types = []
    for p in content:
        if not isinstance(p, dict):
            parts.append({"type": "text", "text": str(p)})
            continue
        t = p.get("type")
        if t in ("text", "input_text", "output_text"):
            parts.append({"type": "text", "text": p.get("text", "")})
        elif t == "image_url":
            parts.append(p)  # 已翻译为 Chat 格式
        else:
            # 未知部件类型：降级为文本承载，禁止静默丢弃（docs/25 R1/G-10）
            parts.append({"type": "text", "text": json.dumps(p, ensure_ascii=False)})
            degraded_types.append(str(t) if t else "<missing>")

    if norm_log is not None and degraded_types:
        norm_log.append({
            "shape": "user_content",
            "parts": len(parts),
            "texts": sum(1 for p in parts if p.get("type") == "text"),
            "images": sum(1 for p in parts if p.get("type") == "image_url"),
            "degraded": degraded_types,
        })
    return parts


# 本地补丁 #16（2026-09-21）：工具输出内容形态归一化（docs/29 / docs/25 R1 补充条款）
# ---------------------------------------------------------------------------
# 入站 `function_call_output.output` / `custom_tool_call_output.output` 有两种
# 合法形态：纯字符串，或「内容部件数组」。实测 CodeBuddy 的 cua_repl 子工具 `js`
# 返回 [{"type":"input_text",…}, {"type":"input_text",…}]（1 个 Wall-time 头 +
# 1 个工具文档体）。
#
# Chat 的 tool.content 只认 `text` / `image_url` 部件；Responses 专有的
# `input_text` 一旦漏到上游即 400：
#   {"code":11101,"msg":"Parse message failed: unsupported content type at index 0: input_text"}
# 且该 item 会**永久留在会话历史**里 → 该会话之后每一轮都失败（模型无关，
# glm-5.3 / deepseek-v4-flash 同样中招）。修复前 `function_call_output` 分支直接
# `item.get("output","")` 原样透传，属 R1 补充条款禁止的"形态伪装成透传"。
#
# 归一化规则（R1 出路②：内容零丢失 + 由调用方记日志）：
#   str           → 原样返回
#   list[部件]     → text 类统一为 {"type":"text"}；图片保持 image_url；
#                   未知部件降级为 JSON 文本承载（不丢弃）
#   其它          → json.dumps 字符串化
# 两条路径（function / custom）共用本函数，避免第四条路径再漏（docs/25 §3.2）。
_CHAT_TEXT_PART_TYPES = ("text", "input_text", "output_text")


def _normalize_tool_output(output, norm_log: list | None = None):
    """把工具输出归一化为 Chat 合法形态（见本段上方注释）。"""
    if isinstance(output, str):
        return output

    if isinstance(output, list):
        parts: list[dict] = []
        degraded: list[str] = []
        for p in output:
            if isinstance(p, str):
                parts.append({"type": "text", "text": p})
                continue
            if not isinstance(p, dict):
                parts.append({"type": "text", "text": str(p)})
                degraded.append(type(p).__name__)
                continue
            t = p.get("type")
            if t in _CHAT_TEXT_PART_TYPES:
                parts.append({"type": "text", "text": p.get("text", "")})
            elif t == "image_url":
                parts.append(p)  # 已由 _convert_images_for_chat 翻译
            else:
                # 未知部件类型：R1 禁止静默丢弃 → 降级为文本承载并记日志
                parts.append({"type": "text",
                              "text": json.dumps(p, ensure_ascii=False)})
                degraded.append(str(t))
        if norm_log is not None:
            norm_log.append({
                "shape": "parts",
                "parts": len(parts),
                "texts": sum(1 for x in parts if x.get("type") == "text"),
                "images": sum(1 for x in parts if x.get("type") == "image_url"),
                "degraded": degraded,
            })
        if not parts:
            return ""
        return parts

    # 其它类型（dict / int / None…）→ 字符串化，绝不把非 Chat 结构透传
    if norm_log is not None:
        norm_log.append({"shape": type(output).__name__, "parts": 1,
                         "texts": 1, "images": 0, "degraded": []})
    return json.dumps(output, ensure_ascii=False)


def responses_request_to_chat(body: dict, norm_log: list | None = None) -> dict:
    """将 Responses API 请求体转换为 Chat Completions 请求体。

    关键映射：
      input → messages
      instructions → system message（置顶）
      max_output_tokens → max_tokens
      tools 格式微调（Responses 用 name，Chat 用 function.name）

    `norm_log`（补丁 #16，可选）：收集工具输出形态归一化记录，供调用方落
    `── RESPONSES OUTPUT NORM ──` 观测日志（R2）。为 None 时不做收集，
    对既有调用方完全向后兼容。
    """
    # 本地补丁 #10 v2：图片部件格式翻译（Responses input_image → Chat image_url）
    if isinstance(body.get("input"), list):
        body = dict(body)
        body["input"] = _convert_images_for_chat(body["input"])

    messages: list[dict] = []

    # instructions → system message
    instructions = body.get("instructions")
    if instructions:
        messages.append({"role": "system", "content": instructions})

    # input → messages
    inp = body.get("input", [])
    if isinstance(inp, str):
        messages.append({"role": "user", "content": inp})
    elif isinstance(inp, list):
        messages.extend(_convert_input_items(inp, norm_log))

    # 构造 Chat body
    chat: dict[str, Any] = {"messages": messages, "stream": True}

    # model
    if "model" in body:
        chat["model"] = body["model"]

    # tools — Responses 和 Chat 的 function tool 格式略有不同
    tools = body.get("tools")
    if tools:
        chat["tools"] = _convert_tools_for_chat(tools)
    if "tool_choice" in body:
        chat["tool_choice"] = body["tool_choice"]

    # 透传常见参数
    for key in ("temperature", "top_p", "stop", "seed",
                "presence_penalty", "frequency_penalty",
                "response_format", "reasoning_effort"):
        if key in body:
            chat[key] = body[key]

    # 本地补丁（2026-09-17）：Responses 结构化 reasoning {"effort": ...} → 扁平 reasoning_effort
    # 否则 codex 的 model_reasoning_effort 会在协议转换时被丢弃
    reasoning = body.get("reasoning")
    if isinstance(reasoning, dict) and reasoning.get("effort"):
        chat.setdefault("reasoning_effort", reasoning["effort"])

    # max_output_tokens → max_tokens
    if "max_output_tokens" in body:
        chat["max_tokens"] = body["max_output_tokens"]
    elif "max_tokens" in body:
        chat["max_tokens"] = body["max_tokens"]

    return chat


def _convert_input_items(items: list, norm_log: list | None = None) -> list[dict]:
    """将 Responses API 的 input 数组转换为 Chat messages。

    input 里可能包含：
      - {"role": "user/developer", "content": ...}   → 直接映射
      - {"type": "message", ...}                      → 助手消息
      - {"type": "function_call", ...}                → 需合并到前面的助手消息
      - {"type": "function_call_output", ...}         → tool 角色

    `norm_log`（补丁 #16）：透传给 `_normalize_tool_output` 收集形态归一化记录。
    """
    messages: list[dict] = []
    # 临时缓存：合并相邻的 assistant message 和 function_call
    pending_assistant_content: str | None = None
    pending_tool_calls: list[dict] = []

    def _flush_assistant():
        nonlocal pending_assistant_content, pending_tool_calls
        if pending_assistant_content is not None or pending_tool_calls:
            msg: dict[str, Any] = {"role": "assistant",
                                   "content": pending_assistant_content or ""}
            if pending_tool_calls:
                msg["tool_calls"] = pending_tool_calls[:]
            messages.append(msg)
            pending_assistant_content = None
            pending_tool_calls.clear()

    for item in items:
        if not isinstance(item, dict):
            continue

        item_type = item.get("type")
        role = item.get("role", "")

        # 简单消息 {"role": "user", "content": "..."}
        if item_type is None and role in ("user", "system", "developer"):
            _flush_assistant()
            mapped_role = "system" if role == "developer" else role
            content = _convert_user_content(item.get("content", ""), norm_log) if role == "user" else _extract_content(item.get("content", ""))
            messages.append({"role": mapped_role, "content": content})
            continue

        # typed message（Responses 里常见）
        if item_type == "message" and role in ("user", "system", "developer"):
            _flush_assistant()
            mapped_role = "system" if role == "developer" else role
            content = _convert_user_content(item.get("content", ""), norm_log) if role == "user" else _extract_content(item.get("content", ""))
            messages.append({"role": mapped_role, "content": content})
            continue

        # assistant 消息（来自前一轮输出）
        if item_type == "message" and role == "assistant":
            _flush_assistant()
            content_parts = item.get("content", [])
            text = _extract_output_text(content_parts) if isinstance(content_parts, list) else str(content_parts)
            pending_assistant_content = text
            continue

        # 简单 role=assistant（无 type 标记）
        if item_type is None and role == "assistant":
            _flush_assistant()
            content = _extract_content(item.get("content", ""))
            pending_assistant_content = content
            continue

        # function_call — 合并到前面的 assistant 消息
        if item_type == "function_call":
            if pending_assistant_content is None:
                pending_assistant_content = ""
            pending_tool_calls.append({
                "id": item.get("call_id", item.get("id", _rand_id("call_"))),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "{}"),
                },
            })
            continue

        # function_call_output → tool 消息
        # 补丁 #16：output 可能是「内容部件数组」（cua_repl 的 js 工具实测返回
        # [input_text, input_text]）——必须归一化，否则 11101（见 _normalize_tool_output）
        if item_type == "function_call_output":
            _flush_assistant()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": _normalize_tool_output(item.get("output", ""), norm_log),
            })
            continue

        # 补丁 #14（2026-09-21）：custom_tool_call（freeform 工具的历史调用）→
        # 还原为降级包装后的 function 调用（arguments = {"input": <freeform 文本>}，
        # 与 _convert_tools_for_chat 的包装 schema 对齐，保证往返一致）
        if item_type == "custom_tool_call":
            if pending_assistant_content is None:
                pending_assistant_content = ""
            input_text = item.get("input", "")
            if not isinstance(input_text, str):
                input_text = str(input_text)
            pending_tool_calls.append({
                "id": item.get("call_id", item.get("id", _rand_id("call_"))),
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": json.dumps({"input": input_text}, ensure_ascii=False),
                },
            })
            continue

        # 补丁 #14：custom_tool_call_output → tool 消息（freeform 工具的执行结果）
        # 补丁 #16：与 function_call_output 共用同一个归一化函数（不再各自实现，
        # 避免第四条路径再漏；#14 原先的 json.dumps 会把图片部件压成 JSON 文本）
        if item_type == "custom_tool_call_output":
            _flush_assistant()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": _normalize_tool_output(item.get("output", ""), norm_log),
            })
            continue

        # 其他未知类型 — 尝试当作普通消息
        if role:
            _flush_assistant()
            content = _extract_content(item.get("content", ""))
            messages.append({"role": role, "content": content})

    _flush_assistant()
    return messages


def _extract_content(content) -> str:
    """提取 content（可能是 str / list[{type,text}]）。"""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for p in content:
            if isinstance(p, dict):
                if p.get("type") in ("input_text", "text"):
                    parts.append(p.get("text", ""))
                elif p.get("type") == "output_text":
                    parts.append(p.get("text", ""))
            elif isinstance(p, str):
                parts.append(p)
        return "".join(parts) or str(content)
    return str(content)


def _extract_output_text(content_parts: list) -> str:
    """从 Responses output content parts 提取纯文本。"""
    texts = []
    for part in content_parts:
        if isinstance(part, dict) and part.get("type") == "output_text":
            texts.append(part.get("text", ""))
    return "".join(texts)


def _custom_input_from_args(args: str) -> str:
    """补丁 #14（2026-09-21）：从降级包装的 function arguments 还原 freeform input。

    请求侧把 custom/freeform 工具包装成单参数 input:string 的 function；
    模型回呼时 arguments 形如 '{"input": "*** Begin Patch…"}'。这里解出内层
    文本还原为原生 custom_tool_call.input。解析失败 / 形状不符时原样返回
    arguments（宁可让上层看到原始内容，也不静默吞掉）。
    """
    if not args:
        return ""
    try:
        parsed = json.loads(args)
        if isinstance(parsed, dict) and isinstance(parsed.get("input"), str):
            return parsed["input"]
    except (json.JSONDecodeError, ValueError):
        pass
    return args


def _convert_tools_for_chat(tools: list) -> list:
    """将 Responses 格式的 tools 转为 Chat 格式。

    Responses:  {"type": "function", "name": "shell", "description": ..., "parameters": ...}
    Chat:       {"type": "function", "function": {"name": "shell", "description": ..., "parameters": ...}}

    补丁 #14（2026-09-21，docs/24 B / docs/25 G-01·R1·A1）：非 function 工具不再静默丢弃。
    后端是 chat/completions 协议，只认 {"type":"function"}；对 custom（freeform，如
    apply_patch）/ web_search / local_shell / 未知类型走 R1 出路②「显式降级」：
    包装成 function（custom → 单参数 input:string 承载 freeform 文本；其余 → 空 object），
    模型至少能看到工具存在并可调用。降级清单由 converter 层记结构化日志（R2）。
    """
    result = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        # 已经是 Chat 格式（有 "function" key）
        if "function" in t:
            result.append(t)
            continue
        if t.get("type") == "namespace":
            namespace = t.get("name") or ""
            namespace_tools = t.get("tools")
            if not namespace or not isinstance(namespace_tools, list):
                continue
            for subtool in namespace_tools:
                if not isinstance(subtool, dict):
                    continue
                subtool = dict(subtool)
                subtool_name = subtool.get("name") or ""
                if not subtool_name:
                    continue
                subtool["name"] = f"{namespace}__{subtool_name}"
                if not subtool.get("description"):
                    subtool["description"] = t.get("description") or f"Tool '{subtool_name}' in namespace '{namespace}'"
                result.extend(_convert_tools_for_chat([subtool]))
            continue
        if t.get("type") == "function":
            # Responses 扁平格式 → Chat 嵌套格式
            fn: dict[str, Any] = {"name": t.get("name", "")}
            if "description" in t:
                fn["description"] = t["description"]
            if "parameters" in t:
                fn["parameters"] = t["parameters"]
            if "strict" in t:
                fn["strict"] = t["strict"]
            result.append({"type": "function", "function": fn})
            continue
        # 补丁 #14：非 function 类型 → 降级包装为 function（R1 出路②，禁止 continue 丢弃）
        ttype = t.get("type")
        name = t.get("name") or (ttype if isinstance(ttype, str) else "")
        if not name:
            # 连可寻址的名字都没有，包装无从谈起；converter 层 TOOLS IN 日志兜底观测
            continue
        fn = {"name": name}
        desc = t.get("description") or f"Tool '{name}' (relayed from Responses tool type '{ttype}')"
        fn["description"] = desc
        if ttype == "custom":
            # freeform：调用语义是模型发原始文本 → 单参数 input:string 承载
            fn["parameters"] = {
                "type": "object",
                "properties": {
                    "input": {"type": "string", "description": "Raw text input for this tool"}
                },
                "required": ["input"],
            }
        else:
            # web_search / local_shell / 未知类型：Responses 里本就无参数 schema 可投影
            fn["parameters"] = {"type": "object", "properties": {}}
        result.append({"type": "function", "function": fn})
    return result


# ---------------------------------------------------------------------------
# 响应转换：Chat → Responses
# ---------------------------------------------------------------------------

class ResponsesStreamConverter:
    """将 Chat SSE 流实时转换为 Responses API 语义事件流。

    用法：
      converter = ResponsesStreamConverter(model="glm-5.2")
      # 对后端返回的每个 SSE 行调 feed_line()
      # feed_line 返回要发送给客户端的 Responses 事件字符串（可能多行）
      for line in backend_sse:
          events = converter.feed_line(line)
          if events:
              yield events.encode()
      # 流结束后调 finish() 获取收尾事件
      yield converter.finish().encode()
    """

    def __init__(self, model: str = "unknown", real_input_tokens: int = 0,
                 custom_tool_names: list[str] | set[str] | None = None,
                 namespace_tool_names: dict[str, str] | None = None):
        """
        real_input_tokens: 本地补丁 #12（2026-09-20）——投影前的真实上下文 token 估算。

        默认 0 表示沿用后端回报的 prompt_tokens（原行为）。非 0 时，
        `usage.input_tokens` / `total_tokens` 按该真实值上报，使客户端（Codex）
        的上下文预算与 auto-compact 判定基于真实体积，而非投影后的残缺体积。

        custom_tool_names: 本地补丁 #14（2026-09-21）——入站声明为 custom/freeform
        的工具名（如 apply_patch）。模型经降级包装回呼这些名字时，输出还原为原生
        custom_tool_call item（codex.exe 实证：ResponseItem::CustomToolCall 7 字段 +
        response.custom_tool_call_input.delta/done 事件均在其解析表内）。
        """
        self.resp_id = _rand_id("resp_")
        self.msg_id = _rand_id("msg_")
        self.model = model
        self.created_at = int(time.time())
        self.real_input_tokens = int(real_input_tokens) if real_input_tokens else 0
        # 补丁 #14（2026-09-21）：入站声明为 custom/freeform 的工具名集合。
        # 这些工具在请求侧被降级包装成 function；模型回呼时在此还原为原生
        # custom_tool_call item + custom_tool_call_input.* 事件（往返一致，docs/25 §3.2）。
        self._custom_tool_names: set[str] = set(custom_tool_names or [])
        # Codex 发送 namespace tools，但 Chat 后端只见展平后的别名。模型回呼
        # 别名时需拆回 name=<子工具名> + namespace=<namespace> 两字段。
        self._namespace_tool_names: dict[str, str] = dict(namespace_tool_names or {})
        # 补丁 #18（2026-09-22，docs/30 / docs/25 G-11）：模型有时会照抄系统提示
        # 里的裸子工具名（如 spawn_agent）。裸名在所有 namespace 中唯一时按该
        # namespace 还原；冲突时保持原样转发，禁止猜测目标 namespace。
        self._namespace_fallbacks: dict[str, str] = {}
        namespaces_by_subtool: dict[str, set[str]] = {}
        for alias, subtool_name in self._namespace_tool_names.items():
            namespace = alias.rsplit("__", 1)[0]
            namespaces_by_subtool.setdefault(subtool_name, set()).add(namespace)
        self._namespace_fallbacks = {
            subtool_name: next(iter(namespaces))
            for subtool_name, namespaces in namespaces_by_subtool.items()
            if len(namespaces) == 1
        }

        # 状态标记
        self._emitted_created = False
        self._emitted_msg_item = False
        self._emitted_content_part = False

        # 本地补丁（2026-09-17）：后端思考流 reasoning_content 状态
        self._reasoning = ""
        self._reasoning_id = _rand_id("rs_")
        self._emitted_reasoning = False
        self._reasoning_closed = False

        # 累积内容
        self._content = ""
        self._tool_calls: dict[int, dict] = {}  # index → {id, name, args, fc_id, output_idx, emitted}
        # 补丁 #18 观测：记录被裸名兜底还原的调用次数，由 converter 层统一落日志。
        self.namespace_fallback_hits: dict[str, int] = {}
        self._finish_reason: str | None = None
        self._usage: dict | None = None

    # ---- 公开接口 ----

    def feed_line(self, line: str) -> str:
        """处理一行 SSE（如 'data: {...}'），返回转换后的 Responses 事件字符串。"""
        line = line.strip()
        if not line or not line.startswith("data:"):
            return ""
        data = line[5:].strip()
        if data == "[DONE]":
            return ""
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            # 修复 F2（2026-09-19 审查）：留痕，避免思考片段无声丢失无法排障
            _warn_dropped_line(line)
            return ""
        # 修复 F1（2026-09-19 审查）：顶层非 dict 的合法 JSON（如 data: 123）跳过
        if not isinstance(chunk, dict):
            _warn_dropped_line(line)
            return ""
        return self._process_chunk(chunk)

    def finish(self) -> str:
        """流结束后，发出收尾事件（done + completed）。"""
        events: list[str] = []

        # 本地补丁：先关闭 reasoning item（若未关闭）
        self._close_reasoning(events)

        # 关闭 text content
        if self._emitted_content_part:
            moi = self._msg_output_index()
            events.append(self._evt("response.output_text.done", {
                "output_index": moi, "content_index": 0, "text": self._content
            }))
            events.append(self._evt("response.content_part.done", {
                "output_index": moi, "content_index": 0,
                "part": {"type": "output_text", "text": self._content, "annotations": []}
            }))

        if self._emitted_msg_item:
            events.append(self._evt("response.output_item.done", {
                "output_index": self._msg_output_index(),
                "item": self._msg_item("completed")
            }))

        # 关闭 function calls
        for idx in sorted(self._tool_calls):
            tc = self._tool_calls[idx]
            if tc.get("emitted"):
                oi = tc["output_idx"]
                # 补丁 #14：custom/freeform 工具 → custom_tool_call_input.done
                if self._is_custom_tool(tc["name"]):
                    events.append(self._evt("response.custom_tool_call_input.done", {
                        "output_index": oi, "input": _custom_input_from_args(tc["args"])
                    }))
                else:
                    events.append(self._evt("response.function_call_arguments.done", {
                        "output_index": oi, "arguments": tc["args"]
                    }))
                events.append(self._evt("response.output_item.done", {
                    "output_index": oi, "item": self._fc_item(tc, "completed")
                }))

        # 修复 V-1（2026-09-19 对抗审查）：后端小 max_tokens 只产 reasoning 不产正文时，
        # 原实现硬编码 completed——客户端无法区分"正常空回复"与"被截断"。
        # 对齐 OpenAI Responses 语义：finish=length 且无正文/工具调用 → response.incomplete。
        truncated_no_content = (
            self._finish_reason in ("length", "max_tokens")
            and not self._content
            and not self._tool_calls
        )
        if truncated_no_content:
            events.append(self._evt("response.incomplete", {
                "response": self._response_obj("incomplete")
            }))
        else:
            events.append(self._evt("response.completed", {
                "response": self._response_obj("completed")
            }))
        return "".join(events)

    def get_nonstream_response(self) -> dict:
        """流结束后获取完整的非流式 Response 对象。"""
        # 修复 V-1：与 finish() 同款 incomplete 语义
        if (
            self._finish_reason in ("length", "max_tokens")
            and not self._content
            and not self._tool_calls
        ):
            return self._response_obj("incomplete")
        return self._response_obj("completed")

    # ---- 内部 ----

    def _process_chunk(self, chunk: dict) -> str:
        events: list[str] = []

        # 模型名
        if chunk.get("model"):
            self.model = chunk["model"]

        # 首次 → 发 created + in_progress
        if not self._emitted_created:
            resp = self._response_obj("in_progress")
            events.append(self._evt("response.created", {"response": resp}))
            events.append(self._evt("response.in_progress", {"response": resp}))
            self._emitted_created = True

        # usage
        if chunk.get("usage"):
            self._usage = chunk["usage"]

        for choice in chunk.get("choices") or []:
            # 修复 F1（2026-09-19 审查）：choices 元素非 dict 时跳过
            if not isinstance(choice, dict):
                continue
            # F1：delta 为 null / 非 dict 时按空 delta 处理（不丢 finish_reason）
            delta = choice.get("delta") or {}
            if not isinstance(delta, dict):
                delta = {}
            finish = choice.get("finish_reason")

            # ---- reasoning delta（本地补丁：思考内容 → Responses reasoning 事件）----
            reasoning = delta.get("reasoning_content")
            if reasoning:
                if not self._emitted_reasoning:
                    events.append(self._evt("response.output_item.added", {
                        "output_index": 0,
                        "item": {"type": "reasoning", "id": self._reasoning_id, "summary": []}
                    }))
                    events.append(self._evt("response.reasoning_summary_part.added", {
                        "item_id": self._reasoning_id,
                        "output_index": 0, "summary_index": 0,
                        "part": {"type": "summary_text", "text": ""}
                    }))
                    self._emitted_reasoning = True
                self._reasoning += reasoning
                events.append(self._evt("response.reasoning_summary_text.delta", {
                    "item_id": self._reasoning_id,
                    "output_index": 0, "summary_index": 0, "delta": reasoning
                }))

            # ---- content delta ----
            content = delta.get("content")
            if content:
                self._close_reasoning(events)
                moi = self._msg_output_index()
                if not self._emitted_msg_item:
                    events.append(self._evt("response.output_item.added", {
                        "output_index": moi,
                        "item": self._msg_item("in_progress", empty=True)
                    }))
                    self._emitted_msg_item = True

                if not self._emitted_content_part:
                    events.append(self._evt("response.content_part.added", {
                        "output_index": moi, "content_index": 0,
                        "part": {"type": "output_text", "text": "", "annotations": []}
                    }))
                    self._emitted_content_part = True

                self._content += content
                events.append(self._evt("response.output_text.delta", {
                    "output_index": moi, "content_index": 0, "delta": content
                }))

            # ---- tool_calls delta ----
            if delta.get("tool_calls"):
                self._close_reasoning(events)
            for tc in delta.get("tool_calls", []):
                idx = tc.get("index", 0)
                if idx not in self._tool_calls:
                    # 修复（2026-09-19 审查）：始终为 message item 预留一个槽位，
                    # 避免"思考→工具→正文"交错序列下工具与正文撞 output_index
                    # （原逻辑仅在 msg 已发出时 +1，正文晚于工具到达时会重叠）。
                    base = (1 if self._emitted_reasoning else 0) + 1  # +1 = 预留 msg 槽
                    oi = base + len(self._tool_calls)
                    self._tool_calls[idx] = {
                        "id": tc.get("id", ""),
                        "name": "",
                        "args": "",
                        "fc_id": _rand_id("fc_"),
                        "output_idx": oi,
                        "emitted": False,
                    }
                slot = self._tool_calls[idx]
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function", {})
                if fn.get("name"):
                    slot["name"] = fn["name"]

                if not slot["emitted"]:
                    events.append(self._evt("response.output_item.added", {
                        "output_index": slot["output_idx"],
                        "item": self._fc_item(slot, "in_progress")
                    }))
                    slot["emitted"] = True

                if fn.get("arguments"):
                    slot["args"] += fn["arguments"]
                    # 补丁 #14：custom/freeform 工具 → 原生 custom_tool_call_input.delta
                    # （片段为 JSON 参数碎片，最终 input 以 output_item.done 为准）
                    if self._is_custom_tool(slot["name"]):
                        events.append(self._evt("response.custom_tool_call_input.delta", {
                            "output_index": slot["output_idx"],
                            "delta": fn["arguments"]
                        }))
                    else:
                        events.append(self._evt("response.function_call_arguments.delta", {
                            "output_index": slot["output_idx"],
                            "delta": fn["arguments"]
                        }))

            if finish:
                self._finish_reason = finish

        return "".join(events)

    def _evt(self, event_type: str, data: dict) -> str:
        """格式化一个 SSE 事件。"""
        payload = {"type": event_type, **data}
        return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"

    # ---- 本地补丁（2026-09-17）：reasoning 辅助 ----

    def _msg_output_index(self) -> int:
        """message item 的 output_index（reasoning item 已占用 0 时后移到 1）。"""
        return 1 if self._emitted_reasoning else 0

    def _close_reasoning(self, events: list[str]) -> None:
        """关闭 reasoning item（发出 summary done 事件）。幂等。"""
        if self._emitted_reasoning and not self._reasoning_closed:
            events.append(self._evt("response.reasoning_summary_text.done", {
                "item_id": self._reasoning_id,
                "output_index": 0, "summary_index": 0, "text": self._reasoning
            }))
            events.append(self._evt("response.reasoning_summary_part.done", {
                "item_id": self._reasoning_id,
                "output_index": 0, "summary_index": 0,
                "part": {"type": "summary_text", "text": self._reasoning}
            }))
            events.append(self._evt("response.output_item.done", {
                "output_index": 0,
                "item": {"type": "reasoning", "id": self._reasoning_id,
                         "summary": [{"type": "summary_text", "text": self._reasoning}]}
            }))
            self._reasoning_closed = True

    def _msg_item(self, status: str = "in_progress", empty: bool = False) -> dict:
        content = [] if empty else [
            {"type": "output_text", "text": self._content, "annotations": []}
        ]
        return {
            "type": "message",
            "id": self.msg_id,
            "status": status,
            "role": "assistant",
            "content": content,
        }

    def _fc_item(self, tc: dict, status: str) -> dict:
        # 补丁 #14：custom/freeform 工具 → 原生 custom_tool_call item
        if self._is_custom_tool(tc["name"]):
            return {
                "type": "custom_tool_call",
                "id": tc["fc_id"],
                "call_id": tc["id"],
                "name": tc["name"],
                "input": _custom_input_from_args(tc["args"]),
                "status": status,
            }
        item = {
            "type": "function_call",
            "id": tc["fc_id"],
            "call_id": tc["id"],
            "name": tc["name"],
            "arguments": tc["args"],
            "status": status,
        }
        subtool_name = self._namespace_tool_names.get(tc["name"])
        if subtool_name:
            item["name"] = subtool_name
            item["namespace"] = tc["name"].rsplit("__", 1)[0]
            return item
        namespace = self._namespace_fallbacks.get(tc["name"])
        if namespace:
            item["namespace"] = namespace
            hits = self.namespace_fallback_hits.get(tc["name"], 0)
            self.namespace_fallback_hits[tc["name"]] = hits + 1
        return item

    def _is_custom_tool(self, name: str) -> bool:
        return bool(name) and name in self._custom_tool_names

    def _response_obj(self, status: str) -> dict:
        output = []
        # 本地补丁：reasoning item 排在最前
        if self._emitted_reasoning:
            output.append({
                "type": "reasoning",
                "id": self._reasoning_id,
                "summary": [{"type": "summary_text", "text": self._reasoning}],
            })
        if self._emitted_msg_item or self._content:
            output.append(self._msg_item(status))
        for idx in sorted(self._tool_calls):
            tc = self._tool_calls[idx]
            if tc.get("emitted"):
                output.append(self._fc_item(tc, status))

        usage = None
        if self._usage:
            u = self._usage
            input_tokens = u.get("prompt_tokens", u.get("input_tokens", 0)) or 0
            output_tokens = u.get("completion_tokens", u.get("output_tokens", 0)) or 0
            # 本地补丁 #12（2026-09-20）：按投影前的真实体积上报输入的上下文大小。
            # 不这么做时，客户端只看到"投影后 payload"的 token 数（几千），
            # 远端 auto-compact 阈值（200K）永远够不着，会话无限膨胀。
            if self.real_input_tokens > 0:
                input_tokens = self.real_input_tokens
            details = u.get("prompt_tokens_details") or u.get("input_tokens_details") or {}
            cached_tokens = details.get("cached_tokens", 0) or 0
            usage = {
                "input_tokens": input_tokens,
                "input_tokens_details": {"cached_tokens": cached_tokens},
                "output_tokens": output_tokens,
                "output_tokens_details": {"reasoning_tokens": 0},
                "total_tokens": input_tokens + output_tokens,
            }

        resp = {
            "id": self.resp_id,
            "object": "response",
            "created_at": self.created_at,
            "status": status,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": True,
            "usage": usage,
        }
        if status == "incomplete":
            resp["incomplete_details"] = {"reason": "max_output_tokens"}
        return resp
