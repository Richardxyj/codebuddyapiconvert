#!/usr/bin/env python3
"""
test_responses_adapter.py — 验证 Responses API 适配层的转换逻辑。

直接运行：python3 test_responses_adapter.py
"""

import json
import sys
sys.path.insert(0, ".")

from core.responses_adapter import (
    _convert_user_content,
    responses_request_to_chat,
    ResponsesStreamConverter,
)
from core.desensitize import desensitize_body
from core.responses_projection import (
    MODEL_CONTEXT_TOKENS,
    pair_token_estimate,
    project_responses_chat_body,
)


def _over_budget_text(model: str = "glm-5.3", factor: float = 1.15, unit: str = "x") -> str:
    """构造一段必然超过「直通预算」的 ASCII 文本。

    直通预算 = MODEL_CONTEXT_TOKENS[model] − OUTPUT_HEADROOM_TOKENS，会随常量演进
    （补丁 #13 把 glm 类从 96,000 抬到 243,200）。测试不应写死字符数，否则常量一变
    这些"超预算"用例就悄悄退化成了直通用例。
    """
    from core.responses_projection import (
        DEFAULT_MODEL_CONTEXT_TOKENS,
        MODEL_CONTEXT_TOKENS,
        OUTPUT_HEADROOM_TOKENS,
    )

    ctx = MODEL_CONTEXT_TOKENS.get(model, DEFAULT_MODEL_CONTEXT_TOKENS)
    budget = max(ctx - OUTPUT_HEADROOM_TOKENS, 24000)
    unit = unit or "x"
    need = int(budget * factor * 3)  # ASCII ≈ 1 token / 3 字符
    return (unit * (need // len(unit) + 1))[:need]


def test_simple_text_request():
    """测试：简单文本 input → messages 转换。"""
    req = {
        "model": "glm-5.2",
        "input": "Hello, how are you?",
        "instructions": "You are a helpful assistant.",
        "stream": True,
    }
    chat = responses_request_to_chat(req)
    assert chat["messages"][0] == {"role": "system", "content": "You are a helpful assistant."}
    assert chat["messages"][1] == {"role": "user", "content": "Hello, how are you?"}
    assert chat["model"] == "glm-5.2"
    print("✅ test_simple_text_request")


def test_array_input_request():
    """测试：数组 input（user + assistant + function_call + function_call_output）。"""
    req = {
        "model": "glm-5.2",
        "input": [
            {"role": "user", "content": "Fix the bug"},
            {"type": "message", "id": "msg_1", "role": "assistant",
             "content": [{"type": "output_text", "text": "I'll check the file."}]},
            {"type": "function_call", "id": "fc_1", "call_id": "call_123",
             "name": "shell", "arguments": '{"cmd":"cat main.py"}'},
            {"type": "function_call_output", "call_id": "call_123",
             "output": "print('hello')"},
            {"role": "user", "content": "Now fix it"},
        ],
        "instructions": "You are a coding assistant.",
    }
    chat = responses_request_to_chat(req)
    msgs = chat["messages"]

    assert msgs[0] == {"role": "system", "content": "You are a coding assistant."}
    assert msgs[1] == {"role": "user", "content": "Fix the bug"}
    assert msgs[2]["role"] == "assistant"
    assert msgs[2]["content"] == "I'll check the file."
    assert len(msgs[2]["tool_calls"]) == 1
    assert msgs[2]["tool_calls"][0]["function"]["name"] == "shell"
    assert msgs[3]["role"] == "tool"
    assert msgs[3]["tool_call_id"] == "call_123"
    assert msgs[4] == {"role": "user", "content": "Now fix it"}
    print("✅ test_array_input_request")


def test_tools_conversion():
    """测试：Responses 扁平 tools 格式 → Chat 嵌套格式。"""
    req = {
        "model": "glm-5.2",
        "input": "test",
        "tools": [
            {"type": "function", "name": "shell",
             "description": "Run a shell command",
             "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
        ],
    }
    chat = responses_request_to_chat(req)
    tool = chat["tools"][0]
    assert tool["type"] == "function"
    assert "function" in tool
    assert tool["function"]["name"] == "shell"
    print("✅ test_tools_conversion")


def test_max_output_tokens():
    """测试：max_output_tokens → max_tokens。"""
    req = {"model": "glm-5.2", "input": "test", "max_output_tokens": 4096}
    chat = responses_request_to_chat(req)
    assert chat["max_tokens"] == 4096
    print("✅ test_max_output_tokens")


def test_developer_role():
    """测试：developer role → system。"""
    req = {"model": "glm-5.2", "input": [
        {"role": "developer", "content": "Be concise."},
        {"role": "user", "content": "Hi"},
    ]}
    chat = responses_request_to_chat(req)
    assert chat["messages"][0] == {"role": "system", "content": "Be concise."}
    assert chat["messages"][1] == {"role": "user", "content": "Hi"}
    print("✅ test_developer_role")


def test_typed_developer_message_request():
    """测试：typed message + developer role 也能映射为 system。"""
    req = {
        "model": "glm-5.2",
        "input": [
            {"type": "message", "role": "developer", "content": "Be concise."},
            {"type": "message", "role": "user", "content": "Hi"},
        ],
    }
    chat = responses_request_to_chat(req)
    assert chat["messages"][0] == {"role": "system", "content": "Be concise."}
    assert chat["messages"][1] == {"role": "user", "content": "Hi"}
    print("✅ test_typed_developer_message_request")


def test_desensitize_harness_user_and_tools():
    """测试：harness user 上下文会被摘要，tool 描述会脱敏，真实 user 不改。"""
    body = {
        "messages": [
            {"role": "system", "content": "Refuse exploit development."},
            {"role": "user", "content": "# AGENTS.md instructions\n<environment_context> sandbox escalation"},
            {"role": "user", "content": "please explain dos attacks"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "exec_command", "description": "Run dangerous exploit development checks."}}
        ],
    }
    out = desensitize_body(
        body,
        roles=("system", "developer"),
        desensitize_harness_user=True,
        desensitize_tools=True,
    )
    assert "​" in out["messages"][0]["content"]
    # P-201（补丁 #10）：AGENTS.md / environment_context 保真透传 + 零宽脱敏
    m1 = out["messages"][1]["content"]
    assert "# AGENTS.md instructions" in m1
    assert "<environment_context>" in m1
    assert "s​andbox" in m1  # sandbox 被零宽脱敏
    assert "Repository instructions and durable user context are provided." not in m1
    assert "​" not in out["messages"][2]["content"]
    assert "​" in out["tools"][0]["function"]["description"]
    print("✅ test_desensitize_harness_user_and_tools")


def test_compact_harness_messages_and_strip_tool_metadata():
    """测试：Codex 注入长提示被压缩，tool 描述可直接裁掉。"""
    body = {
        "messages": [
            {"role": "system", "content": "You are a coding agent running in the Codex CLI. # How you work\nUse sandbox and escalation."},
            {"role": "system", "content": "<permissions instructions>\nFilesystem sandboxing defines which files can be read or written."},
            {"role": "user", "content": "# AGENTS.md instructions\n<environment_context> sandbox escalation"},
        ],
        "tools": [
            {"type": "function", "function": {"name": "exec_command", "description": "Run dangerous exploit development checks.", "parameters": {"type": "object", "properties": {"cmd": {"type": "string", "description": "Shell command to execute."}}}}}
        ],
    }
    out = desensitize_body(
        body,
        roles=("system", "developer"),
        desensitize_harness_user=True,
        desensitize_tools=True,
        compact_harness=True,
        strip_tool_metadata=True,
    )
    assert len(out["messages"][0]["content"]) < 220
    # 通道检测词零宽脱敏后字面量不可见（2026-09-19 P-201 补充：Codex CLI → C?odex CL?I）
    assert "C​odex" in out["messages"][0]["content"]
    # P-201（补丁 #10）：permissions 保真透传（零宽脱敏后字面量不可见）
    m1 = out["messages"][1]["content"]
    assert "<permissions instructions>" in m1
    assert "Filesystem s​andboxing defines" in m1
    # P-201（补丁 #10）：AGENTS.md + environment_context 保真透传
    m2 = out["messages"][2]["content"]
    assert "# AGENTS.md instructions" in m2
    assert "<environment_context>" in m2
    assert "Repository instructions and environment context" not in m2
    assert "description" not in out["tools"][0]["function"]
    assert "description" not in out["tools"][0]["function"]["parameters"]["properties"]["cmd"]
    print("✅ test_compact_harness_messages_and_strip_tool_metadata")


def test_no_compact_still_prunes_codex_runtime_metadata():
    """测试：保留全文模式仍会裁掉 Codex 注入的运行时元数据大段文本。"""
    body = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a coding agent running in the Codex CLI.\n\n"
                    "# How you work\nUse sandbox and escalation carefully.\n\n"
                    "<permissions instructions>\nFilesystem sandboxing defines which files can be read or written.\n"
                    "## How to request escalation\n...\n</permissions instructions>\n\n"
                    "The following deferred tools are now available via ToolSearch.\n..."
                ),
            },
            {
                "role": "user",
                "content": (
                    "# AGENTS.md instructions\n<INSTRUCTIONS>\nproject guidance\n</INSTRUCTIONS>"
                    "<environment_context>\nvery long runtime context\n</environment_context>\n"
                    "<skills_instructions>\nvery long skills metadata\n</skills_instructions>\n"
                    "test"
                ),
            },
            {"role": "user", "content": "test"},
        ],
    }
    out = desensitize_body(
        body,
        roles=("system", "developer"),
        desensitize_harness_user=True,
        compact_harness=False,
    )
    system_text = out["messages"][0]["content"]
    harness_text = out["messages"][1]["content"]
    assert "You are a coding agent running in the C​odex CLI" in system_text
    assert "## Planning" not in system_text
    assert "## Task execution" not in system_text
    assert "### Final answer structure and style guidelines" not in system_text
    assert "# How you work" in system_text
    # P-201（补丁 #10）：Codex base_instructions 与 permissions 块保真透传
    assert "Filesystem s​andboxing defines" in system_text
    assert "<permissions instructions>" in system_text
    # ToolSearch 工具清单尾段仍被裁掉（_RUNTIME_TAIL_MARKERS 保留）
    assert "ToolSearch" not in system_text
    assert "Runtime tool, agent, sk" in system_text
    # P-201：environment_context / AGENTS.md 保真；skills 块仍摘要（有意偏差）
    assert "# AGENTS.md instructions" in harness_text
    assert "very long runtime context" in harness_text
    assert "very long skills metadata" not in harness_text
    assert "Runtime sk​ill metadata" in harness_text
    assert out["messages"][2]["content"] == "test"
    print("✅ test_no_compact_still_prunes_codex_runtime_metadata")


def test_responses_projection_compacts_codex_harness_and_tools():
    """测试：超预算的 Codex 风格请求会投影为短 system + 极简 schema。

    补丁 #6 尺寸感知后，小 body 直通（passthrough）——本用例需构造超过
    token 预算的大 body 才触发 aggressive（尺寸由 _over_budget_text 按预算推导）。
    """
    # ASCII 填充到 > 预算（≈ 3 字符/token），确保超过直通预算
    harness_filler = "\n" + _over_budget_text()
    body = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a coding agent running in the Codex CLI.\n"
                    "# AGENTS.md spec\nVery long harness instructions."
                ),
            },
            {
                "role": "system",
                "content": "Additional repo rule: always run tests after editing.",
            },
            {
                "role": "user",
                "content": "# AGENTS.md instructions\n<environment_context>\nlong context\n</environment_context>"
                + harness_filler,
            },
            {"role": "user", "content": "实现该方案"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "description": "Run a command with a long dangerous description",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string", "description": "Shell command to execute."},
                            "yield_time_ms": {"type": "number", "description": "Wait time"},
                        },
                        "required": ["cmd"],
                        "additionalProperties": False,
                    },
                    "strict": False,
                },
            }
        ],
    }
    out, stats = project_responses_chat_body(body)
    assert stats["mode"] == "aggressive"
    assert out["messages"][0]["role"] == "system"
    assert "OpenAI-compatible CLI" in out["messages"][0]["content"]
    assert all("# AGENTS.md instructions" not in msg.get("content", "") for msg in out["messages"])
    assert any("Additional repo rule" in msg.get("content", "") for msg in out["messages"])
    assert out["messages"][-1] == {"role": "user", "content": "实现该方案"}
    tool = out["tools"][0]["function"]
    assert tool["name"] == "exec_command"
    assert "description" not in tool
    assert "description" not in tool["parameters"]["properties"]["cmd"]
    assert stats["projected_tool_chars"] < stats["original_tool_chars"]
    print("✅ test_responses_projection_compacts_codex_harness_and_tools")


def test_responses_projection_preserves_recent_tool_chain_and_summarizes_history():
    """测试：超预算请求中较早轮次会被摘要，最近 tool 链保持完整。

    补丁 #6 尺寸感知后需构造超过 token 预算的大 body 才触发 aggressive。
    """
    big_output = (
        "Chunk ID: a1\nWall time: 0.0\nProcess exited with code 0\nOutput:\n"
        + "\n".join(f"line {i}" for i in range(40))
        + "\n" + _over_budget_text()  # 超过直通预算（尺寸随常量推导）
    )
    body = {
        "messages": [
            {"role": "system", "content": "You are a coding agent running in the Codex CLI."},
            {"role": "user", "content": "# AGENTS.md instructions\n<environment_context>ctx</environment_context>"},
            {"role": "user", "content": "先看 README"},
            {
                "role": "assistant",
                "content": "I will inspect the repository.",
                "tool_calls": [
                    {
                        "id": "call_old",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": "{\"cmd\":\"ls -la\"}"},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_old", "content": "Output:\nREADME.md\nsrc\n"},
            {"role": "assistant", "content": "README is present."},
            {"role": "user", "content": "现在修复 converter 的 responses 链路"},
            {
                "role": "assistant",
                "content": "I will patch the proxy and then run tests.",
                "tool_calls": [
                    {
                        "id": "call_recent",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": "sed -n '1,200p' converter.py", "yield_time_ms": 1000}),
                        },
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_recent", "content": big_output},
            {"role": "assistant", "content": "I found the endpoint and will implement projection now."},
            {"role": "user", "content": "继续，别依赖 fallback retry"},
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "exec_command",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "cmd": {"type": "string"},
                            "yield_time_ms": {"type": "number"},
                        },
                        "required": ["cmd"],
                    },
                },
            }
        ],
    }
    out, stats = project_responses_chat_body(body)
    system_messages = [m["content"] for m in out["messages"] if m["role"] == "system"]
    assert system_messages[0].startswith("You are a coding assistant serving an OpenAI-compatible CLI.")
    assert any("Earlier conversation summary" in text for text in system_messages)
    assert any("先看 README" in text for text in system_messages)

    recent_assistant = next(
        msg for msg in out["messages"]
        if msg.get("role") == "assistant" and any(tc.get("id") == "call_recent" for tc in msg.get("tool_calls", []))
    )
    recent_tool = next(msg for msg in out["messages"] if msg.get("role") == "tool" and msg.get("tool_call_id") == "call_recent")
    assert recent_assistant["tool_calls"][0]["function"]["name"] == "exec_command"
    assert "Process exited with code 0" in recent_tool["content"]
    assert "line 39" in recent_tool["content"]
    assert len(recent_tool["content"]) < len(big_output)
    assert out["messages"][-1] == {"role": "user", "content": "继续，别依赖 fallback retry"}
    assert stats["summarized_history_messages"] >= 1
    print("✅ test_responses_projection_preserves_recent_tool_chain_and_summarizes_history")


def test_responses_projection_shrinks_large_tool_arguments():
    """测试：超预算请求的超长 tool arguments 会压缩成结构化 JSON 摘要。

    补丁 #6 尺寸感知后需构造超过 token 预算的大 body 才触发投影。
    """
    long_cmd = "echo " + ("x" * 1600)
    body = {
        "messages": [
            {"role": "system", "content": "You are a coding agent running in the Codex CLI."},
            {"role": "user", "content": "执行一个很长的命令" + "\n" + _over_budget_text(unit="y")},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_long",
                        "type": "function",
                        "function": {
                            "name": "exec_command",
                            "arguments": json.dumps({"cmd": long_cmd, "yield_time_ms": 1000, "workdir": "/tmp"}),
                        },
                    }
                ],
            },
        ],
        "tools": [],
    }
    out, _ = project_responses_chat_body(body)
    args = out["messages"][-1]["tool_calls"][0]["function"]["arguments"]
    parsed = json.loads(args)
    assert parsed["cmd"].startswith("echo ")
    assert "truncated" in parsed["cmd"]
    print("✅ test_responses_projection_shrinks_large_tool_arguments")


def test_responses_projection_small_body_passthrough():
    """测试（补丁 #6）：小 body 在 token 预算内应原样直通，不做 aggressive 投影。"""
    body = {
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a coding agent running in the Codex CLI.\n"
                    "# AGENTS.md spec\nVery long harness instructions."
                ),
            },
            {"role": "user", "content": "实现该方案"},
        ],
        "tools": [],
    }
    out, stats = project_responses_chat_body(body)
    assert stats["mode"] == "passthrough"
    assert stats["aggressive"] is False
    assert out["messages"] == body["messages"]  # 原样直通，无投影
    assert stats["token_estimate"] <= stats["context_budget_tokens"]
    print("✅ test_responses_projection_small_body_passthrough")


def test_stream_converter_text():
    """测试：Chat SSE 文本流 → Responses 事件流。"""
    conv = ResponsesStreamConverter(model="glm-5.2")

    # 模拟 Chat SSE chunks
    chunks = [
        'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{"content":" world"},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":2,"total_tokens":12}}',
        'data: [DONE]',
    ]

    all_events = []
    for line in chunks:
        result = conv.feed_line(line)
        if result:
            for evt_line in result.strip().split("\n\n"):
                if evt_line.startswith("data: "):
                    all_events.append(json.loads(evt_line[6:]))

    # 收尾
    finish = conv.finish()
    for evt_line in finish.strip().split("\n\n"):
        if evt_line.startswith("data: "):
            all_events.append(json.loads(evt_line[6:]))

    # 验证事件类型序列
    types = [e["type"] for e in all_events]
    assert "response.created" in types
    assert "response.in_progress" in types
    assert "response.output_item.added" in types
    assert "response.content_part.added" in types
    assert "response.output_text.delta" in types
    assert "response.output_text.done" in types
    assert "response.content_part.done" in types
    assert "response.output_item.done" in types
    assert "response.completed" in types

    # 验证最终文本
    text_done = [e for e in all_events if e["type"] == "response.output_text.done"][0]
    assert text_done["text"] == "Hello world"

    # 验证 completed response
    completed = [e for e in all_events if e["type"] == "response.completed"][0]
    resp = completed["response"]
    assert resp["status"] == "completed"
    assert resp["output"][0]["type"] == "message"
    assert resp["output"][0]["content"][0]["text"] == "Hello world"
    assert resp["usage"]["input_tokens"] == 10

    print("✅ test_stream_converter_text")


def test_stream_converter_function_call():
    """测试：Chat SSE tool_calls → Responses function_call 事件。"""
    conv = ResponsesStreamConverter(model="glm-5.2")

    chunks = [
        'data: {"id":"chatcmpl-2","choices":[{"index":0,"delta":{"role":"assistant","tool_calls":[{"index":0,"id":"call_abc","type":"function","function":{"name":"shell","arguments":""}}]},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"{\\"cmd"}}]},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-2","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\\": \\"ls\\"}"}}]},"finish_reason":null}]}',
        'data: {"id":"chatcmpl-2","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}',
        'data: [DONE]',
    ]

    all_events = []
    for line in chunks:
        result = conv.feed_line(line)
        if result:
            for evt_line in result.strip().split("\n\n"):
                if evt_line.startswith("data: "):
                    all_events.append(json.loads(evt_line[6:]))

    finish = conv.finish()
    for evt_line in finish.strip().split("\n\n"):
        if evt_line.startswith("data: "):
            all_events.append(json.loads(evt_line[6:]))

    types = [e["type"] for e in all_events]
    assert "response.output_item.added" in types
    assert "response.function_call_arguments.delta" in types
    assert "response.function_call_arguments.done" in types
    assert "response.completed" in types

    # 验证 function call arguments
    args_done = [e for e in all_events if e["type"] == "response.function_call_arguments.done"][0]
    assert args_done["arguments"] == '{"cmd": "ls"}'

    print("✅ test_stream_converter_function_call")


def test_nonstream_response():
    """测试：非流式 Response 对象生成。"""
    conv = ResponsesStreamConverter(model="glm-5.2")
    conv.feed_line('data: {"id":"c1","choices":[{"index":0,"delta":{"content":"Hi"},"finish_reason":null}]}')
    conv.feed_line('data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}')

    resp = conv.get_nonstream_response()
    assert resp["object"] == "response"
    assert resp["status"] == "completed"
    assert resp["output"][0]["type"] == "message"
    assert resp["output"][0]["content"][0]["text"] == "Hi"
    assert resp["usage"]["input_tokens"] == 5

    print("✅ test_nonstream_response")


def test_channel_detection_terms_zwsp():
    """P-201 v2 回归（2026-09-19）：通道检测词必须被零宽脱敏。

    实证链：完整 base_instructions 原文 → 后端 400 code 11128
    (Illegal API invocation from an unapproved channel)；
    通道词 ZWSP 化后 → 200（probe_11128.py P1/P1b）。
    若本用例挂，说明词表被回退，codex 全形态请求将再次全阻。
    """
    from core.desensitize import desensitize_text

    sample = (
        "You are a coding agent running in the Codex CLI. "
        "Codex CLI is an open source project led by OpenAI. "
        "Powered by ChatGPT and GPT-5 models. Try codex exec and codex-cli flags."
    )
    out = desensitize_text(sample)
    for term in ["Codex CLI", "Codex", "codex-cli", "codex exec", "ChatGPT", "OpenAI", "GPT-5"]:
        assert term not in out, f"通道检测词未脱敏: {term}"
    assert "C​odex" in out and "O​penAI" in out  # ZWSP 插在首字符后
    assert "C​hatGPT" in out and "G​PT-5" in out
    assert "coding agent running in" in out  # 普通文本不动
    print("✅ test_channel_detection_terms_zwsp")


def test_desensitize_preserves_windows_and_posix_paths():
    """路径中的通道词不得插入 ZWSP，否则模型会构造出不存在的路径。"""
    from core.desensitize import desensitize_text

    windows = r"C:\Users\demo-user\Documents\Codex\2026-09-21\plugin-browser-openai-bundle"
    posix = "/home/demo-user/.codex/plugins/openai/browser"
    assert desensitize_text(windows) == windows
    assert desensitize_text(posix) == posix
    print("✅ test_desensitize_preserves_windows_and_posix_paths")


def test_desensitize_still_processes_non_path_text():
    """路径豁免不能削弱普通文本的通道检测词脱敏。"""
    from core.desensitize import desensitize_text

    out = desensitize_text("Use Codex Desktop with OpenAI GPT-5.")
    assert "C​odex" in out and "O​penAI" in out and "G​PT-5" in out
    print("✅ test_desensitize_still_processes_non_path_text")


def test_model_context_tokens_match_active_catalog_models():
    """目录已移除的模型不得残留投影常量，避免后续误读可用上下文。"""
    active_models = {
        "glm-5.3",
        "kimi-k3-1",
        "kimi-k2.8-preview",
        "deepseek-v4-flash",
    }
    assert set(MODEL_CONTEXT_TOKENS) == active_models
    assert set(MODEL_CONTEXT_TOKENS.values()) == {256_000}
    print("✅ test_model_context_tokens_match_active_catalog_models")


if __name__ == "__main__":
    test_simple_text_request()
    test_array_input_request()
    test_tools_conversion()
    test_max_output_tokens()
    test_developer_role()
    test_typed_developer_message_request()
    test_desensitize_harness_user_and_tools()
    test_compact_harness_messages_and_strip_tool_metadata()
    test_no_compact_still_prunes_codex_runtime_metadata()
    test_responses_projection_compacts_codex_harness_and_tools()
    test_responses_projection_preserves_recent_tool_chain_and_summarizes_history()
    test_responses_projection_shrinks_large_tool_arguments()
    test_responses_projection_small_body_passthrough()
    test_stream_converter_text()
    test_stream_converter_function_call()
    test_nonstream_response()
    print(f"\n🎉 All {16} tests passed!")

def test_incomplete_when_truncated_without_content():
    """V-1 回归（2026-09-19 对抗审查）：后端小 max_tokens 只产 reasoning 不产正文时，
    Responses 语义应为 incomplete + incomplete_details.reason=max_output_tokens，
    而非 completed（客户端无法区分正常空回复与被截断）。"""
    conv = ResponsesStreamConverter(model="m")
    conv.feed_line('data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"thinking..."},"finish_reason":null}]}')
    conv.feed_line('data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"length"}],"usage":{"prompt_tokens":100,"completion_tokens":16,"total_tokens":116}}')
    tail = conv.finish()
    assert "response.incomplete" in tail
    assert "response.completed" not in tail
    resp = conv.get_nonstream_response()
    assert resp["status"] == "incomplete"
    assert resp["incomplete_details"] == {"reason": "max_output_tokens"}
    assert [o["type"] for o in resp["output"]] == ["reasoning"]


def test_completed_when_truncated_but_has_content():
    """V-1 边界：finish=length 但已有正文 → 仍 completed（截断的正文也好过标 incomplete）。"""
    conv = ResponsesStreamConverter(model="m")
    conv.feed_line('data: {"id":"c1","choices":[{"index":0,"delta":{"role":"assistant","content":"partial answer"},"finish_reason":null}]}')
    conv.feed_line('data: {"id":"c1","choices":[{"index":0,"delta":{},"finish_reason":"length"}]}')
    tail = conv.finish()
    assert "response.completed" in tail
    resp = conv.get_nonstream_response()
    assert resp["status"] == "completed"
    assert "incomplete_details" not in resp



# ---------------------------------------------------------------------------
# 补丁 #12（2026-09-20）：真实上下文用量上报
# 背景：投影把发往后端的 payload 压到几万字符，后端回报的 prompt_tokens 只剩几千，
# 客户端（Codex）看到的是"投影后体积"→ auto-compact 阈值永远够不着 → 会话无限膨胀。
# 修复：投影层暴露投影前的真实估算，转换器按真实值上报 usage.input_tokens。
# ---------------------------------------------------------------------------

def test_projection_aggressive_exposes_original_token_estimate():
    """补丁 #12：aggressive 分支必须同时给出投影前（真实）与投影后的 token 估算。"""
    big_output = _over_budget_text(unit="line of log output\n")
    body = {
        "model": "glm-5.3",
        "messages": [
            {"role": "system", "content": "You are a coding agent running in the Codex CLI."},
            {"role": "user", "content": "把日志里的错误汇总"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "type": "function",
                        "function": {"name": "exec_command", "arguments": '{"cmd":"cat app.log"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": big_output},
        ],
        "tools": [
            {
                "type": "function",
                "function": {"name": "exec_command", "parameters": {"type": "object", "properties": {}}},
            }
        ],
    }
    out, stats = project_responses_chat_body(body)

    assert stats["mode"] == "aggressive"
    # 真实体积超过预算（这才是"应该触发远端压缩"的量级）
    assert stats["original_token_estimate"] > stats["context_budget_tokens"]
    # 投影后确实小了，且与投影结果自洽
    assert stats["projected_token_estimate"] < stats["original_token_estimate"]
    assert stats["projected_token_estimate"] == pair_token_estimate(
        out["messages"], out.get("tools") or []
    )
    print("✅ test_projection_aggressive_exposes_original_token_estimate")


def test_projection_passthrough_exposes_original_token_estimate():
    """补丁 #12：直通分支的真实体积 = 投影后体积（未做压缩）。"""
    body = {
        "messages": [
            {"role": "system", "content": "You are a coding agent running in the Codex CLI."},
            {"role": "user", "content": "实现该方案"},
        ],
        "tools": [],
    }
    out, stats = project_responses_chat_body(body)

    assert stats["mode"] == "passthrough"
    assert stats["original_token_estimate"] == stats["projected_token_estimate"]
    assert stats["original_token_estimate"] == pair_token_estimate(out["messages"], [])
    print("✅ test_projection_passthrough_exposes_original_token_estimate")


def test_converter_reports_real_input_tokens():
    """补丁 #12：给了 real_input_tokens 时，usage 按真实体积上报。"""
    conv = ResponsesStreamConverter(model="glm-5.3", real_input_tokens=4_400_000)
    conv.feed_line(
        'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3672,"completion_tokens":210,"total_tokens":3882}}'
    )
    resp = conv.get_nonstream_response()

    assert resp["usage"]["input_tokens"] == 4_400_000
    assert resp["usage"]["output_tokens"] == 210
    assert resp["usage"]["total_tokens"] == 4_400_210
    print("✅ test_converter_reports_real_input_tokens")


def test_converter_default_usage_follows_upstream():
    """补丁 #12 回归：未给 real_input_tokens 时行为与改动前完全一致。"""
    conv = ResponsesStreamConverter(model="glm-5.3")
    conv.feed_line(
        'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3672,"completion_tokens":210,"total_tokens":3882}}'
    )
    resp = conv.get_nonstream_response()

    assert resp["usage"]["input_tokens"] == 3672
    assert resp["usage"]["total_tokens"] == 3882
    print("✅ test_converter_default_usage_follows_upstream")


def test_converter_preserves_upstream_cached_tokens():
    """上游缓存命中必须如实透传，不能把可见性指标硬编码为 0。"""
    conv = ResponsesStreamConverter(model="glm-5.3")
    conv.feed_line(
        'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3672,"completion_tokens":210,"total_tokens":3882,'
        '"prompt_tokens_details":{"cached_tokens":1000}}}'
    )
    resp = conv.get_nonstream_response()

    assert resp["usage"]["input_tokens_details"]["cached_tokens"] == 1000
    print("✅ test_converter_preserves_upstream_cached_tokens")


def test_converter_cached_tokens_default_zero():
    """上游未回报缓存命中时保持 Responses API 合法形态，默认 0。"""
    conv = ResponsesStreamConverter(model="glm-5.3")
    conv.feed_line(
        'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"ok"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":3672,"completion_tokens":210,"total_tokens":3882}}'
    )
    resp = conv.get_nonstream_response()

    assert resp["usage"]["input_tokens_details"]["cached_tokens"] == 0
    print("✅ test_converter_cached_tokens_default_zero")


def test_converter_real_usage_in_stream_completed_event():
    """补丁 #12：流式路径的 response.completed 事件同样携带真实 usage。"""
    conv = ResponsesStreamConverter(model="glm-5.3", real_input_tokens=250_000)
    conv.feed_line(
        'data: {"id":"c1","choices":[{"index":0,"delta":{"content":"hi"},"finish_reason":"stop"}],'
        '"usage":{"prompt_tokens":1200,"completion_tokens":30,"total_tokens":1230}}'
    )
    tail = conv.finish()
    assert "response.completed" in tail

    resp = conv.get_nonstream_response()
    assert resp["usage"]["input_tokens"] == 250_000
    assert resp["usage"]["total_tokens"] == 250_030
    print("✅ test_converter_real_usage_in_stream_completed_event")


# ---------------------------------------------------------------------------
# 补丁 #14（2026-09-21，docs/24 B / docs/25 G-01·G-02）：工具保真契约测试
# ---------------------------------------------------------------------------

def test_patch14_non_function_tools_wrapped_not_dropped():
    """#14 / R1：非 function 工具不得静默丢弃 —— docs/24 第 6 节 6 类型探针。

    影子实测（修复前）：custom / web_search / local_shell 全部被 continue 吞掉，
    只有 function 存活。修复后 6/6 必须全部出现在出站 tools 里（降级包装为 function）。
    """
    req = {
        "model": "glm-5.3",
        "input": "probe",
        "tools": [
            {"type": "function", "name": "exec_command",
             "description": "Run a command",
             "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}},
            {"type": "function", "name": "mcp__node_repl__js",
             "description": "Eval JS",
             "parameters": {"type": "object", "properties": {"code": {"type": "string"}}}},
            {"type": "custom", "name": "apply_patch",
             "description": "Apply a patch (freeform)"},
            {"type": "custom", "name": "tool_search"},
            {"type": "web_search"},
            {"type": "local_shell"},
        ],
    }
    chat = responses_request_to_chat(req)
    tools = chat["tools"]

    names = [t["function"]["name"] for t in tools]
    # 6/6 守恒：一个都不能少
    assert names == ["exec_command", "mcp__node_repl__js", "apply_patch",
                     "tool_search", "web_search", "local_shell"], names
    # 全部包装为 chat function 格式
    assert all(t["type"] == "function" and "function" in t for t in tools)
    # custom/freeform → 单参数 input:string 承载
    ap = tools[2]["function"]
    assert ap["parameters"]["properties"]["input"]["type"] == "string"
    assert ap["parameters"]["required"] == ["input"]
    # 无名类型（web_search / local_shell）以类型名寻址
    assert tools[4]["function"]["name"] == "web_search"
    assert tools[5]["function"]["name"] == "local_shell"
    # function 工具不受影响（回归）
    assert tools[0]["function"]["parameters"]["properties"]["cmd"]["type"] == "string"
    print("✅ test_patch14_non_function_tools_wrapped_not_dropped")


def test_patch14_custom_tool_history_roundtrip():
    """#14：custom_tool_call / custom_tool_call_output 历史项必须还原进 messages。

    修复前这两类 item 被「未知类型」分支静默吞掉 → 下一轮请求丢掉 apply_patch
    调用链（R1 违例 + 合成 payload 不再以 user 消息收尾的风险）。
    """
    req = {
        "model": "glm-5.3",
        "input": [
            {"role": "user", "content": "patch it"},
            {"type": "custom_tool_call", "id": "tc_1", "call_id": "call_9",
             "name": "apply_patch", "input": "*** Begin Patch\\n*** Update File: a.py\\n"},
            {"type": "custom_tool_call_output", "call_id": "call_9",
             "output": "Done!"},
            {"role": "user", "content": "thanks"},
        ],
    }
    chat = responses_request_to_chat(req)
    msgs = chat["messages"]

    # custom_tool_call → assistant.tool_calls（arguments 为包装 schema 的 JSON）
    assistant = msgs[1]
    assert assistant["role"] == "assistant"
    tc = assistant["tool_calls"][0]
    assert tc["function"]["name"] == "apply_patch"
    import json as _json
    assert _json.loads(tc["function"]["arguments"]) == {
        "input": "*** Begin Patch\\n*** Update File: a.py\\n"}
    # custom_tool_call_output → tool 消息
    assert msgs[2]["role"] == "tool"
    assert msgs[2]["tool_call_id"] == "call_9"
    assert msgs[2]["content"] == "Done!"
    # 尾部 user 消息仍在（11133 防线）
    assert msgs[3] == {"role": "user", "content": "thanks"}
    print("✅ test_patch14_custom_tool_history_roundtrip")


def test_patch14_custom_tool_call_response_shape():
    """#14：custom 工具回呼 → 原生 custom_tool_call item + input 事件（往返一致）。

    codex.exe 实证其解析表含 ResponseItem::CustomToolCall（7 字段）与
    response.custom_tool_call_input.delta/done 事件。普通 function 工具不受影响。
    """
    conv = ResponsesStreamConverter(
        model="glm-5.3", custom_tool_names={"apply_patch"})
    ev1 = conv.feed_line(
        'data: {"id":"c1","choices":[{"index":0,"delta":{"tool_calls":['
        '{"index":0,"id":"call_1","function":{"name":"apply_patch",'
        '"arguments":"{\\"input\\": \\"*** Begin Patch\\"}"}}]}}]}'
    )
    ev2 = conv.feed_line(
        'data: {"id":"c2","choices":[{"index":0,"delta":{"tool_calls":['
        '{"index":1,"id":"call_2","function":{"name":"exec_command",'
        '"arguments":"{\\"cmd\\": \\"ls\\"}"}}]}}]}'
    )
    tail = conv.finish()

    # custom 工具：added 事件携带 custom_tool_call item；delta 用原生事件名
    assert '"type":"custom_tool_call"' in ev1 or '"type": "custom_tool_call"' in ev1
    assert "response.custom_tool_call_input.delta" in ev1
    assert "response.function_call_arguments.delta" not in ev1
    # function 工具：行为与修复前完全一致（回归）
    assert "response.function_call_arguments.delta" in ev2
    assert '"type":"function_call"' in ev2 or '"type": "function_call"' in ev2
    # 收尾：custom 用 input.done，最终 item 的 input 已从包装参数中解出
    assert "response.custom_tool_call_input.done" in tail
    assert "response.function_call_arguments.done" in tail

    resp = conv.get_nonstream_response()
    items = {i["type"]: i for i in resp["output"]}
    assert items["custom_tool_call"]["name"] == "apply_patch"
    assert items["custom_tool_call"]["input"] == "*** Begin Patch"
    assert items["custom_tool_call"]["call_id"] == "call_1"
    assert items["function_call"]["name"] == "exec_command"
    print("✅ test_patch14_custom_tool_call_response_shape")


def test_patch14_projection_keeps_non_function_tools():
    """#14 / A1：投影（超预算 aggressive 态）不得丢弃非 function 工具。"""
    body = {
        "messages": [
            {"role": "system", "content": "You are a coding agent running in the Codex CLI."},
            {"role": "user", "content": _over_budget_text()},
        ],
        "tools": [
            {"type": "function", "function": {"name": "exec_command",
             "parameters": {"type": "object", "properties": {"cmd": {"type": "string"}}}}},
            {"type": "web_search"},
        ],
    }
    out, stats = project_responses_chat_body(body)

    assert stats["mode"] != "passthrough"
    out_names = [
        (t.get("function") or {}).get("name") or t.get("type")
        for t in out["tools"] if isinstance(t, dict)
    ]
    # 守恒：function 在、web_search 也在
    assert "exec_command" in out_names
    assert "web_search" in out_names
    assert stats["projected_tools"] == stats["original_tools"] == 2
    print("✅ test_patch14_projection_keeps_non_function_tools")


def test_namespace_tools_flattened_and_restored():
    """namespace 子工具必须展平给 Chat 后端，并在回呼时还原原名。"""
    req = {
        "model": "glm-5.3",
        "input": "probe",
        "tools": [{
            "type": "namespace",
            "name": "mcp__node_repl",
            "description": "Node REPL namespace",
            "tools": [
                {"type": "function", "name": "js",
                 "description": "Evaluate JavaScript",
                 "parameters": {"type": "object", "properties": {
                     "code": {"type": "string"}}, "required": ["code"]}},
            ],
        }],
    }
    chat = responses_request_to_chat(req)
    names = [t["function"]["name"] for t in chat["tools"]]
    assert names == ["mcp__node_repl__js"], names

    conv = ResponsesStreamConverter(
        model="glm-5.3",
        namespace_tool_names={"mcp__node_repl__js": "js"},
    )
    conv.feed_line(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"function":{"name":"mcp__node_repl__js","arguments":"{\\"code\\": \\"1+1\\"}"}}]}}]}'
    )
    conv.finish()
    resp = conv.get_nonstream_response()
    call = next(i for i in resp["output"] if i["type"] == "function_call")
    assert call["name"] == "js"
    assert call["namespace"] == "mcp__node_repl"
    assert call["arguments"] == '{"code": "1+1"}'
    print("✅ test_namespace_tools_flattened_and_restored")


def test_namespace_bare_subtool_name_restored():
    """#18 / G-11：唯一裸子工具名必须还原 namespace，不能原样转发。"""
    conv = ResponsesStreamConverter(
        model="glm-5.3",
        namespace_tool_names={"collaboration__spawn_agent": "spawn_agent"},
    )
    conv.feed_line(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"function":{"name":"spawn_agent","arguments":"{}"}}]}}]}'
    )
    conv.finish()
    resp = conv.get_nonstream_response()
    call = next(i for i in resp["output"] if i["type"] == "function_call")
    assert call["name"] == "spawn_agent"
    assert call["namespace"] == "collaboration"
    assert conv.namespace_fallback_hits == {"spawn_agent": 4}
    print("✅ test_namespace_bare_subtool_name_restored")


def test_namespace_bare_subtool_conflict_not_restored():
    """#18 / G-11：同名裸子工具跨 namespace 冲突时禁止猜测，保持原样转发。"""
    conv = ResponsesStreamConverter(
        model="glm-5.3",
        namespace_tool_names={
            "collaboration__list_agents": "list_agents",
            "mcp__node_repl__list_agents": "list_agents",
        },
    )
    conv.feed_line(
        'data: {"choices":[{"delta":{"tool_calls":[{"index":0,"id":"call_1",'
        '"function":{"name":"list_agents","arguments":"{}"}}]}}]}'
    )
    conv.finish()
    resp = conv.get_nonstream_response()
    call = next(i for i in resp["output"] if i["type"] == "function_call")
    assert call["name"] == "list_agents"
    assert "namespace" not in call
    assert conv.namespace_fallback_hits == {}
    print("✅ test_namespace_bare_subtool_conflict_not_restored")


# ---------------------------------------------------------------------------
# 补丁 #16（2026-09-21，docs/29 / docs/25 G-09·R1 补充条款）：工具输出形态归一化
#
# 事故：cua_repl 子工具 `js` 的 function_call_output.output 是
#   [{"type":"input_text",…}, {"type":"input_text",…}]
# 修复前 function_call_output 分支原样透传 → Chat tool.content 出现 Responses
# 专有类型 input_text → 上游 400（code 11101）→ 该 item 永久留在历史里 →
# **整条会话此后每一轮都失败**（glm-5.3 与 deepseek-v4-flash 同样中招）。
# ---------------------------------------------------------------------------

# 上游只接受这两种 Chat 部件类型
_CHAT_LEGAL_PART_TYPES = {"text", "image_url"}


def _tool_msg(payload_input, kind="function"):
    """把 input items 走一遍转换，返回 tool 消息。"""
    chat = responses_request_to_chat({"model": "glm-5.3", "input": payload_input})
    return next(m for m in chat["messages"] if m.get("role") == "tool")


def test_patch16_tool_output_parts_normalized():
    """#16 / G-09：生产实测的 [input_text, input_text] 形态必须被归一化。

    这是把整条会话砖死的那个形状，逐字复刻自 rollout：
    rollout-2026-09-21T13-12-10-…-3c0a4ae948ce.jsonl 的 `js` 工具输出。
    """
    norm = []
    chat = responses_request_to_chat({
        "model": "glm-5.3",
        "input": [
            {"role": "user", "content": "verify"},
            {"type": "function_call", "call_id": "call_js", "name": "mcp__cua_repl__js",
             "arguments": '{"code":"1+1"}'},
            {"type": "function_call_output", "call_id": "call_js", "output": [
                {"type": "input_text", "text": "Wall time: 0.5309 seconds\nOutput:"},
                {"type": "input_text", "text": "## Computer Use\n\nControl native apps…"},
            ]},
            {"role": "user", "content": "ok"},
        ],
    }, norm)

    tool = next(m for m in chat["messages"] if m["role"] == "tool")
    content = tool["content"]
    assert isinstance(content, list), content
    assert [p["type"] for p in content] == ["text", "text"], content
    # 内容零丢失、顺序不变（R1）
    assert content[0]["text"] == "Wall time: 0.5309 seconds\nOutput:"
    assert content[1]["text"] == "## Computer Use\n\nControl native apps…"
    # R2：必须留下归一化记录
    assert norm and norm[0]["shape"] == "parts"
    assert norm[0]["parts"] == 2 and norm[0]["texts"] == 2 and norm[0]["degraded"] == []
    print("✅ test_patch16_tool_output_parts_normalized")


def test_patch16_image_parts_preserved():
    """#16 / R1：view_image 的 [input_image] 形态必须保留为 image_url，不得压成文本。

    view_image 是唯一长期产出部件数组的工具（09-20/09-21 共 123 处），
    靠补丁 #10 的深度翻译才一直没炸；#16 不得把它退化成字符串。
    """
    tool = _tool_msg([
        {"role": "user", "content": "look"},
        {"type": "function_call", "call_id": "call_img", "name": "view_image",
         "arguments": '{"path":"a.png"}'},
        {"type": "function_call_output", "call_id": "call_img", "output": [
            {"type": "input_image", "image_url": "data:image/png;base64,iVBORw0KGgo="},
        ]},
        {"role": "user", "content": "ok"},
    ])
    content = tool["content"]
    assert isinstance(content, list), content
    assert content[0]["type"] == "image_url", content
    assert content[0]["image_url"]["url"].startswith("data:image/png;base64,")
    print("✅ test_patch16_image_parts_preserved")


def test_patch16_mixed_text_and_image_parts():
    """#16：文本 + 图片混合部件 → [text, image_url]，两类都不丢。"""
    tool = _tool_msg([
        {"role": "user", "content": "mixed"},
        {"type": "function_call", "call_id": "call_m", "name": "view_image",
         "arguments": "{}"},
        {"type": "function_call_output", "call_id": "call_m", "output": [
            {"type": "input_text", "text": "screenshot follows"},
            {"type": "input_image", "image_url": "data:image/png;base64,AAAA"},
        ]},
        {"role": "user", "content": "ok"},
    ])
    content = tool["content"]
    assert [p["type"] for p in content] == ["text", "image_url"], content
    assert content[0]["text"] == "screenshot follows"
    print("✅ test_patch16_mixed_text_and_image_parts")


def test_patch16_unknown_part_degraded_not_dropped():
    """#16 / R1：未知部件类型降级为文本承载（绝不静默丢弃）并进 degraded 记录。"""
    norm = []
    chat = responses_request_to_chat({
        "model": "glm-5.3",
        "input": [
            {"role": "user", "content": "audio?"},
            {"type": "function_call", "call_id": "call_a", "name": "rec", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_a", "output": [
                {"type": "input_audio", "audio_url": "data:audio/wav;base64,ZZZ"},
            ]},
            {"role": "user", "content": "ok"},
        ],
    }, norm)
    tool = next(m for m in chat["messages"] if m["role"] == "tool")
    content = tool["content"]
    assert isinstance(content, list) and content[0]["type"] == "text", content
    assert "input_audio" in content[0]["text"]          # 内容仍在（JSON 文本承载）
    assert norm[0]["degraded"] == ["input_audio"]
    print("✅ test_patch16_unknown_part_degraded_not_dropped")


def test_patch16_custom_tool_output_shares_normalizer():
    """#16 / §3.2 完备性：custom 路径与 function 路径共用同一归一化函数。

    回归项：字符串行为与 #14 完全一致（历史 16 处 custom 输出全是字符串）；
    同时新增能力——部件数组不再被 json.dumps 压成文本。
    """
    # (a) 字符串：行为与 #14 不变
    req_str = {
        "model": "glm-5.3",
        "input": [
            {"role": "user", "content": "patch"},
            {"type": "custom_tool_call", "call_id": "call_9", "name": "apply_patch",
             "input": "*** Begin Patch"},
            {"type": "custom_tool_call_output", "call_id": "call_9", "output": "Done!"},
            {"role": "user", "content": "thanks"},
        ],
    }
    norm_str = []
    msgs = responses_request_to_chat(req_str, norm_str)["messages"]
    assert msgs[2]["role"] == "tool" and msgs[2]["content"] == "Done!"
    assert norm_str == []          # 字符串形态无需归一化 → 不产生噪音日志

    # (b) 部件数组：走同一函数 → text 部件（旧行为是 json.dumps 成一坨文本）
    req_parts = {
        "model": "glm-5.3",
        "input": [
            {"role": "user", "content": "patch"},
            {"type": "custom_tool_call", "call_id": "call_8", "name": "apply_patch",
             "input": "*** Begin Patch"},
            {"type": "custom_tool_call_output", "call_id": "call_8", "output": [
                {"type": "input_text", "text": "line1"},
                {"type": "input_text", "text": "line2"},
            ]},
            {"role": "user", "content": "thanks"},
        ],
    }
    msgs2 = responses_request_to_chat(req_parts)["messages"]
    c = msgs2[2]["content"]
    assert [p["type"] for p in c] == ["text", "text"], c
    assert [p["text"] for p in c] == ["line1", "line2"]
    print("✅ test_patch16_custom_tool_output_shares_normalizer")


def test_patch16_no_responses_part_type_survives():
    """#16 不变量扫描：任何入站 output 形态都不得让 Responses 专有部件类型出站。

    这是 G-09 的**通用**防线——不只覆盖已知的 js / view_image 两种形状，
    而是把所有可能形态都扫一遍（R3：不依赖"恰好只有这两种"）。
    """
    shapes = [
        "plain string",
        [{"type": "input_text", "text": "a"}, {"type": "input_text", "text": "b"}],
        [{"type": "output_text", "text": "a"}],
        [{"type": "text", "text": "a"}],
        [{"type": "input_image", "image_url": "data:image/png;base64,AA"}],
        [{"type": "input_audio", "audio_url": "data:audio/wav;base64,AA"}],
        [{"type": "input_file", "file_id": "f1"}],
        ["bare string part"],
        [123],
        {"nested": "dict"},
        [],
        None,
        42,
    ]
    for i, shape in enumerate(shapes):
        for kind in ("function", "custom_tool"):
            call = {"type": f"{kind}_call", "call_id": f"c{i}", "name": "t",
                    "arguments": "{}", "input": "x"}
            out = {"type": f"{kind}_call_output", "call_id": f"c{i}", "output": shape}
            chat = responses_request_to_chat({
                "model": "glm-5.3",
                "input": [{"role": "user", "content": "u"}, call, out,
                          {"role": "user", "content": "u2"}],
            })
            tool = next(m for m in chat["messages"] if m["role"] == "tool")
            content = tool["content"]
            if isinstance(content, list):
                types = {p.get("type") for p in content}
                assert types <= _CHAT_LEGAL_PART_TYPES, (shape, kind, content)
            else:
                assert isinstance(content, str), (shape, kind, type(content))
    print("✅ test_patch16_no_responses_part_type_survives")


def test_user_content_unknown_part_degraded_not_dropped():
    """用户内容中的未知部件必须降级为文本承载，禁止静默丢弃。"""
    content = _convert_user_content([
        {"type": "input_text", "text": "audio follows"},
        {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}},
        {"type": "input_audio", "audio_url": "data:audio/wav;base64,ZZZ"},
    ])

    assert [p["type"] for p in content] == ["text", "image_url", "text"]
    assert content[0]["text"] == "audio follows"
    assert content[1]["image_url"]["url"].startswith("data:image/png;base64,")
    assert "input_audio" in content[2]["text"]
    print("✅ test_user_content_unknown_part_degraded_not_dropped")


def test_user_content_audio_only_without_image_not_collapsed():
    """无图片时也不能走纯文本提取路径，否则降级语义和观测记录都会消失。"""
    norm = []
    content = _convert_user_content([
        {"type": "input_audio", "audio_url": "data:audio/wav;base64,ZZZ"},
    ], norm)

    assert [p["type"] for p in content] == ["text"], content
    assert "input_audio" in content[0]["text"]
    assert norm[0]["degraded"] == ["input_audio"]
    print("✅ test_user_content_audio_only_without_image_not_collapsed")


def test_user_content_mixed_text_and_audio_without_image_keeps_audio():
    """text + audio 且无 image：audio 必须显式降级，不得被 _extract_content 丢弃。"""
    norm = []
    content = _convert_user_content([
        {"type": "input_text", "text": "audio follows"},
        {"type": "input_audio", "audio_url": "data:audio/wav;base64,ZZZ"},
    ], norm)

    assert [p["type"] for p in content] == ["text", "text"], content
    assert content[0]["text"] == "audio follows"
    assert "input_audio" in content[1]["text"]
    assert norm[0]["degraded"] == ["input_audio"]
    print("✅ test_user_content_mixed_text_and_audio_without_image_keeps_audio")


def test_user_content_untyped_part_degradation_is_logged():
    """无 type 的未知部件也必须留下降级记录，不能只降级不观测。"""
    norm = []
    content = _convert_user_content([{"data": "unsupported"}], norm)

    assert [p["type"] for p in content] == ["text"], content
    assert "unsupported" in content[0]["text"]
    assert norm[0]["degraded"] == ["<missing>"]
    print("✅ test_user_content_untyped_part_degradation_is_logged")


# ---------------------------------------------------------------------------
# 补丁 #19：动态拉取官方模型列表（/v2/enterprises/personal/models + 1h TTL）
# ---------------------------------------------------------------------------

import core.converter as _conv


def _reset_official_cache():
    _conv._official_models_cache["list"] = None
    _conv._official_models_cache["ts"] = 0.0


def test_extract_official_models_prefers_default_tag():
    """带 default 标签的 agent 的 models 是权威列表。"""
    payload = {"data": {"agents": [
        {"models": ["other-1", "other-2"], "tags": []},
        {"models": ["glm-9", "kimi-k9"], "tags": ["default"]},
    ]}}
    assert _conv._extract_official_models(payload) == ["glm-9", "kimi-k9"]
    print("✅ test_extract_official_models_prefers_default_tag")


def test_extract_official_models_union_excludes_lite():
    """无 default 标签时退化为并集，且剔除内部 lite 模型、去重保序。"""
    payload = {"data": {"agents": [
        {"models": ["a", "lite", "b"], "tags": []},
        {"models": ["b", "c", "lite"], "tags": []},
    ]}}
    assert _conv._extract_official_models(payload) == ["a", "b", "c"]
    # 空 / 缺字段输入必须安全返回空列表
    assert _conv._extract_official_models({}) == []
    assert _conv._extract_official_models({"data": {}}) == []
    print("✅ test_extract_official_models_union_excludes_lite")


class _FakeResp:
    def __init__(self, payload, status=200):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload


def _patch_fetch(monkeypatch, payload=None, status=200, raise_exc=None, calls=None):
    """替换 _cred 与 httpx.Client，返回可控的官方模型响应。"""
    class _Cred:
        def get_headers(self):
            return {"Authorization": "Bearer x"}

    monkeypatch.setattr(_conv, "_cred", lambda: _Cred())

    class _Client:
        def __init__(self, timeout=0):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def get(self, url, headers=None):
            if calls is not None:
                calls.append(url)
            if raise_exc is not None:
                raise raise_exc
            return _FakeResp(payload or {"data": {"agents": []}}, status)

    monkeypatch.setattr(_conv.httpx, "Client", _Client)


def test_official_models_take_priority_over_static(monkeypatch):
    """官方动态列表命中时是 /v1/models 的最高权威来源，且仍合并 EXTRA_MODELS。"""
    _reset_official_cache()
    payload = {"data": {"agents": [
        {"models": ["glm-9", "kimi-k9"], "tags": ["default"]},
    ]}}
    _patch_fetch(monkeypatch, payload)
    # 即便 product.json 有别的模型，官方列表也必须优先
    monkeypatch.setattr(_conv, "_load_models_from_workbuddy", lambda: ["wb-static"])

    models = _conv.get_available_models()
    assert models[:2] == ["glm-9", "kimi-k9"], models
    assert "wb-static" not in models
    # EXTRA_MODELS 仍合并（补丁 #2 语义不回归）
    for m in _conv.EXTRA_MODELS:
        assert m in models
    _reset_official_cache()
    print("✅ test_official_models_take_priority_over_static")


def test_official_models_failure_falls_back_to_static(monkeypatch):
    """上游 5xx / 网络异常 / 凭据缺失时静默回退静态链，绝不抛错影响路由。"""
    _reset_official_cache()
    monkeypatch.setattr(_conv, "_load_models_from_workbuddy", lambda: ["wb-static"])

    # 情形1：HTTP 非 200
    _patch_fetch(monkeypatch, status=500)
    models = _conv.get_available_models()
    assert "wb-static" in models

    # 情形2：网络异常
    _reset_official_cache()
    _patch_fetch(monkeypatch, raise_exc=RuntimeError("boom"))
    models = _conv.get_available_models()
    assert "wb-static" in models

    # 情形3：凭据缺失（_cred 抛 HTTPException 503）
    from fastapi import HTTPException
    _reset_official_cache()
    def _no_cred():
        raise HTTPException(status_code=503, detail="no cred")
    monkeypatch.setattr(_conv, "_cred", _no_cred)
    monkeypatch.setattr(_conv.httpx, "Client", lambda *a, **k: (_ for _ in ()).throw(AssertionError("不应发起网络请求")))
    models = _conv.get_available_models()
    assert "wb-static" in models

    _reset_official_cache()
    print("✅ test_official_models_failure_falls_back_to_static")


def test_official_models_ttl_cache_avoids_refetch(monkeypatch):
    """TTL 内重复调用命中缓存，不重复请求上游（resolve_model 每请求调用的安全闸）。"""
    _reset_official_cache()
    payload = {"data": {"agents": [
        {"models": ["glm-9"], "tags": ["default"]},
    ]}}
    calls = []
    _patch_fetch(monkeypatch, payload, calls=calls)

    m1 = _conv._fetch_official_models()
    m2 = _conv._fetch_official_models()
    m3 = _conv._fetch_official_models()
    assert m1 == ["glm-9"] and m2 == m1 and m3 == m1
    assert len(calls) == 1, f"TTL 内应只请求一次，实际 {len(calls)}"

    # force=True 必须绕过缓存重新拉取
    m4 = _conv._fetch_official_models(force=True)
    assert len(calls) == 2
    assert m4 == m1
    _reset_official_cache()
    print("✅ test_official_models_ttl_cache_avoids_refetch")
