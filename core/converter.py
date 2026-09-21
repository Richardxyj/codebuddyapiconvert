#!/usr/bin/env python3
"""
codebuddy2openai — 把 CodeBuddy / WorkBuddy 的订阅暴露成标准 OpenAI 兼容 API。

原理（直连后端，原生 function calling）：
  - 读取本机已登录的 CodeBuddy 桌面端凭据（auth 文件里的 token / uid / enterpriseId）。
  - 直接转发到 CodeBuddy 后端 `https://copilot.tencent.com/v2/chat/completions`。
    该后端本身就是标准 OpenAI chat/completions 协议（含原生 tools / tool_calls / SSE 流式）。
  - 转换器只做两件事：①注入鉴权 header（Authorization / X-User-Id 等）
    ②在本地 /v1/* 与后端 /v2/* 之间做路径映射与透传（含 Anthropic / Chat / Responses 三种协议）。
  - token 过期时自动调 `/v2/plugin/auth/token/refresh` 刷新，并回写 auth 文件。

跨平台：自动定位 auth 目录（macOS / Windows / Linux）。
依赖：fastapi + uvicorn + httpx（pip install fastapi "uvicorn[standard]" httpx）。

用法：
  python3 converter.py                       # 默认 127.0.0.1:8787
  python3 converter.py --port 9000
  python3 converter.py --api-key mysecret    # 启用客户端鉴权
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import threading
import traceback
from collections import deque
import time
from pathlib import Path

import httpx
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

try:
    from .desensitize import desensitize_body
except ImportError:  # 模块缺失时降级为不脱敏

    def desensitize_body(
        body,
        roles=("system",),
        desensitize_harness_user=False,
        desensitize_tools=False,
        compact_harness=False,
        strip_tool_metadata=False,
    ):
        return body


from .anthropic_adapter import (
    AnthropicStreamConverter,
    anthropic_request_to_chat,
)
from .responses_adapter import (
    ResponsesStreamConverter,
    responses_request_to_chat,
)
from .responses_projection import project_responses_chat_body

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

BACKEND = "https://copilot.tencent.com"
DEFAULT_DOMAIN = "www.codebuddy.cn"
USER_AGENT = "codebuddy2openai/2.0"

# ---------------------------------------------------------------------------
# 平台相关：定位 auth 目录
# ---------------------------------------------------------------------------


def auth_dirs() -> list[Path]:
    env_dir = os.environ.get("CODEBUDDY_AUTH_DIR")
    if env_dir:
        return [Path(env_dir)]
    home = Path.home()
    plat = sys.platform
    if plat == "darwin":
        return [
            home
            / "Library"
            / "Application Support"
            / "CodeBuddyExtension"
            / "Data"
            / "Public"
            / "auth"
        ]
    if plat == "win32":
        local = Path(os.environ.get("LOCALAPPDATA", home / "AppData" / "Local"))
        return [local / "CodeBuddyExtension" / "Data" / "Public" / "auth"]
    xdg = Path(os.environ.get("XDG_DATA_HOME", home / ".local" / "share"))
    return [xdg / "CodeBuddyExtension" / "Data" / "Public" / "auth"]


def find_auth_file() -> Path | None:
    for d in auth_dirs():
        if d.is_dir():
            for f in sorted(d.glob("*.info")):
                return f
    return None


# ---------------------------------------------------------------------------
# Auth 凭据管理（读 + 自动刷新 + 回写）
# ---------------------------------------------------------------------------


class CredentialManager:
    """从 auth 文件读取凭据；token 临近过期时自动刷新并回写。"""

    def __init__(self, path: Path):
        self.path = path
        self._lock = threading.Lock()
        self._cached: dict | None = None
        self._mtime: float = 0.0

    def _read_raw(self) -> dict:
        with open(self.path, encoding="utf-8") as f:
            return json.load(f)

    def _load_if_stale(self):
        """若文件 mtime 变了（外部刷新过），重新加载缓存。"""
        try:
            mt = self.path.stat().st_mtime
        except OSError:
            return
        if self._cached is None or mt != self._mtime:
            self._cached = self._read_raw()
            self._mtime = mt

    def _session(self) -> dict:
        self._load_if_stale()
        if self._cached is None:
            raise RuntimeError(f"无法读取 auth 文件：{self.path}")
        return self._cached

    def _is_expired(self) -> bool:
        s = self._session()
        expires_at = (s.get("auth") or {}).get("expiresAt") or 0
        # 提前 60s 判定过期
        return time.time() * 1000 >= (expires_at - 60_000)

    def _refresh(self):
        """调后端刷新 token，写回 auth 文件与缓存。"""
        s = self._session()
        auth = s.get("auth") or {}
        headers = self._build_headers_from(auth, s.get("account") or {})
        headers["X-Refresh-Token"] = auth.get("refreshToken", "")
        headers["X-Auth-Refresh-Source"] = "plugin"
        url = f"{BACKEND}/v2/plugin/auth/token/refresh"
        try:
            with httpx.Client(timeout=15) as c:
                r = c.post(url, headers=headers, json={})
            data = r.json()
        except Exception as e:
            raise RuntimeError(f"刷新 token 网络失败：{e}")
        if data.get("code") != 0 or not data.get("data"):
            raise RuntimeError(f"刷新 token 失败：{data.get('msg', data)}")
        new_auth = data["data"]
        if not isinstance(new_auth, dict) or not new_auth.get("accessToken"):
            raise RuntimeError("刷新响应缺少访问令牌")
        new_auth["refreshToken"] = new_auth.get("refreshToken") or auth.get("refreshToken", "")
        # 继承部分字段
        new_auth["domain"] = new_auth.get("domain") or auth.get("domain")
        new_auth["lastRefreshTime"] = int(time.time() * 1000)
        # 计算 expiresAt（若后端没直接给）
        if not new_auth.get("expiresAt") and new_auth.get("expiresIn"):
            new_auth["expiresAt"] = (
                int(time.time() * 1000) + new_auth["expiresIn"] * 1000
            )
        if not new_auth.get("refreshExpiresAt") and new_auth.get("refreshExpiresIn"):
            new_auth["refreshExpiresAt"] = (
                int(time.time() * 1000) + new_auth["refreshExpiresIn"] * 1000
            )
        s["auth"] = new_auth
        # 原子写回
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(s, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.path)
        self._cached = s
        self._mtime = self.path.stat().st_mtime

    def _build_headers_from(self, auth: dict, account: dict) -> dict:
        domain = auth.get("domain") or DEFAULT_DOMAIN
        h = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "Authorization": f"Bearer {auth.get('accessToken', '')}",
            "X-User-Id": account.get("uid", ""),
            "X-Enterprise-Id": account.get("enterpriseId", ""),
            "X-Tenant-Id": account.get("enterpriseId", ""),
            "X-Domain": domain,
            "User-Agent": USER_AGENT,
        }
        return h

    def get_headers(self) -> dict:
        """返回带最新 token 的后端请求 header；必要时先刷新。"""
        with self._lock:
            if self._is_expired():
                self._refresh()
            s = self._session()
            return self._build_headers_from(s.get("auth") or {}, s.get("account") or {})

    def summary(self) -> dict:
        s = self._session()
        auth = s.get("auth") or {}
        acct = s.get("account") or {}
        exp = auth.get("expiresAt", 0)
        return {
            "uid": acct.get("uid"),
            "nickname": acct.get("nickname"),
            "enterpriseName": acct.get("enterpriseName"),
            "token_expires_at": exp,
            "token_expired": self._is_expired(),
        }


# ---------------------------------------------------------------------------
# 模型列表
# ---------------------------------------------------------------------------

DEFAULT_MODELS = [
    "hy3",
    "hy4-preview",
    "kimi-k3",
    "kimi-k2.8-preview",
    "glm-5.3",
    "glm-5.3-flash",
    "glm-5.2",
    "glm-5.1",
    "glm-5v-turbo",
    "kimi-k2.7",
    "kimi-k2.6",
    "kimi-k2.5",
    "deepseek-v4-pro",
    "deepseek-v4.1-flash",
    "deepseek-v4-flash",
    "minimax-m3-pay",
    "hy3-preview-agent",
    "auto",
]

# 标识非聊天模型的 tag（需要过滤掉）
NON_CHAT_MODEL_TAGS = {
    "text-to-image",
    "image-to-image",
    "text-to-video",
}


def _find_workbuddy_product_json() -> Path | None:
    """
    查找本机 WorkBuddy 应用的 product.json 配置文件。

    WorkBuddy 在安装时会自动解压 asar 到 app.asar.unpacked 目录，
    因此无需用户手动提取。

    macOS: /Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/cli/product.json
    Windows: %LOCALAPPDATA%\\Programs\\WorkBuddy\\resources\\app.asar.unpacked\\cli\\product.json
    Linux: /opt/WorkBuddy/resources/app.asar.unpacked/cli/product.json

    Returns:
        Path 对象如果找到配置文件，否则 None
    """
    possible_paths = []

    if sys.platform == "darwin":  # macOS
        possible_paths.extend(
            [
                # 标准安装路径（WorkBuddy 自动解压）
                Path(
                    "/Applications/WorkBuddy.app/Contents/Resources/app.asar.unpacked/cli/product.json"
                ),
                # 开发/调试：本地提取的目录
                Path.home()
                / "Desktop/workspace/opensource/codebuddy2api/workbuddy_extracted/cli/product.json",
            ]
        )
    elif sys.platform == "win32":  # Windows
        local_app_data = Path(os.environ.get("LOCALAPPDATA", ""))
        possible_paths.extend(
            [
                local_app_data
                / "Programs/WorkBuddy/resources/app.asar.unpacked/cli/product.json",
                Path(
                    "C:/Program Files/WorkBuddy/resources/app.asar.unpacked/cli/product.json"
                ),
            ]
        )
    else:  # Linux
        possible_paths.extend(
            [
                Path("/opt/WorkBuddy/resources/app.asar.unpacked/cli/product.json"),
                Path.home() / ".local/share/WorkBuddy/cli/product.json",
            ]
        )

    for path in possible_paths:
        if path.exists() and path.is_file():
            return path

    return None


def _load_models_from_workbuddy() -> list[str]:
    """
    从本机 WorkBuddy product.json 读取模型列表。

    过滤规则：
    1. 只保留聊天模型（排除 text-to-image, text-to-video 等）
    2. 排除 vendor 为 "tencent" 的内部模型（通常是补全/内部专用）
    3. 返回模型 ID 列表

    Returns:
        模型 ID 列表，如果加载失败返回空列表
    """
    product_json_path = _find_workbuddy_product_json()

    if product_json_path is None:
        return []

    try:
        with open(product_json_path, encoding="utf-8") as f:
            data = json.load(f)

        models = data.get("models", [])
        chat_models = []

        for model in models:
            model_id = model.get("id")
            if not model_id:
                continue

            # 过滤掉非聊天模型
            tags = model.get("tags", [])
            if any(tag in NON_CHAT_MODEL_TAGS for tag in tags):
                continue

            # 过滤掉内部模型（vendor 为 tencent 的通常是补全/跳转等内部功能）
            vendor = model.get("vendor", "")
            if vendor == "tencent":
                continue

            # 过滤掉名称中明显是补全/内部功能的模型
            name_lower = model_id.lower()
            if any(
                keyword in name_lower
                for keyword in ["completion", "rewrite", "jump", "codewise"]
            ):
                continue

            chat_models.append(model_id)

        return chat_models

    except Exception as e:
        # 解析失败时静默降级，不影响服务启动
        print(
            f"Warning: Failed to load models from WorkBuddy product.json: {e}",
            file=sys.stderr,
        )
        return []


# 本地补丁（2026-09-17）：后端实际可用但 product.json 动态列表未包含的模型
# （实测：glm-5.3 / kimi-k2.8-preview 后端可正常调用，但不在 WorkBuddy product.json 列表中）
EXTRA_MODELS = ["glm-5.3", "kimi-k2.8-preview"]


def get_available_models() -> list[str]:
    """
    获取可用的模型列表。

    优先从 WorkBuddy product.json 读取，如果失败则使用 DEFAULT_MODELS。
    无论来源，都会合并 EXTRA_MODELS（本地补丁）。

    Returns:
        模型 ID 列表
    """
    workbuddy_models = _load_models_from_workbuddy()

    if workbuddy_models:
        # 成功从 WorkBuddy 加载，使用动态列表
        base = workbuddy_models
    else:
        # 降级到硬编码列表
        base = DEFAULT_MODELS

    merged = list(base)
    for m in EXTRA_MODELS:
        if m not in merged:
            merged.append(m)
    return merged


# 本地补丁 #4（2026-09-17）：Codex 桌面版模型选择器只列 OpenAI 自家目录，
# 选择器里的名字（gpt-*）在后端无权限（400 "only available for authorized users"）。
# 别名映射：选择器点到的名字 → WorkBuddy 真实模型；未知名字回退 MODEL_FALLBACK。
# （未在映射表里的 slug 回退到默认模型，保证选择器里点任何项都不会 400）
MODEL_ALIAS = {
    "gpt-5.6-sol": "kimi-k3-1",            # 选择器 "5.6 Sol"
    "gpt-6-astra": "glm-5.3",              # 选择器 "6 Astra"
    "gpt-5.6-terra": "kimi-k2.8-preview",  # 选择器 "5.6 Terra"
    "gpt-5.6-luna": "glm-5.2",             # 选择器 "5.6 Luna"
    "gpt-5.5": "deepseek-v4-pro",          # 选择器 "5.5"
}
MODEL_FALLBACK = "kimi-k3-1"

# 本地补丁 #5（2026-09-18）：新版 Codex runtime 在"所选思考档 == 模型目录默认档"时会
# 省略 reasoning 参数；后端只有收到显式 effort 才输出思考文本（reasoning_content）。
# 请求未携带 effort 时补默认值，保证思考流始终可见。可用环境变量覆盖。
DEFAULT_REASONING_EFFORT = os.environ.get("CODEBUDDY_DEFAULT_REASONING_EFFORT", "max")


def resolve_model(name: str | None) -> str:
    """模型名解析：WorkBuddy 真实模型名原样通过；OpenAI 目录名按表映射；未知回退。"""
    if not name:
        return MODEL_FALLBACK
    if name in MODEL_ALIAS:
        return MODEL_ALIAS[name]
    if name in set(get_available_models()):
        return name
    return MODEL_FALLBACK


# 后端请求体里出现过的额外字段（透传时若客户端给了就保留）
PASSTHROUGH_BODY_KEYS = {
    "model",
    "messages",
    "tools",
    "tool_choice",
    "temperature",
    "max_tokens",
    "max_completion_tokens",
    "top_p",
    "stream",
    "stream_options",
    "stop",
    "presence_penalty",
    "frequency_penalty",
    "n",
    "response_format",
    "seed",
    "user",
    "reasoning_effort",
    "verbosity",
    "reasoning_summary",
}

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(title="codebuddy2openai", version="2.0")
CONFIG: dict = {
    "api_key": "",
    "cred": None,
    "log_path": None,
    "desensitize": False,
    "no_compact": False,
    # 本地补丁 #12（2026-09-20）：把投影前的真实上下文 token 估算作为
    # usage.input_tokens 上报，让客户端（Codex）的 auto-compact 基于真实体积判定。
    "report_real_usage": True,
}  # cred: CredentialManager | None


# ---------------------------------------------------------------------------
# 日志（写文件）
# ---------------------------------------------------------------------------

_LOG_LOCK = threading.Lock()


_LOG_ROTATE_BYTES = 200 * 1024 * 1024  # 本地补丁 #8（2026-09-18）：日志超过 200MB 自动轮转
_LOG_ROTATE_CHECK = 0  # 上次轮转检查的时间戳（秒），避免每条日志都 stat


def _log(msg: str):
    """写一行日志到 CONFIG['log_path'] 指定的文件（追加，带时间戳）。未设置则丢弃。"""
    global _LOG_ROTATE_CHECK
    path = CONFIG.get("log_path")
    if not path:
        return
    line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n"
    try:
        with _LOG_LOCK:
            # 补丁 #8：每 60 秒最多检查一次体积，超限则轮转（log -> log.1，覆盖旧 log.1）
            now = time.time()
            if now - _LOG_ROTATE_CHECK > 60:
                _LOG_ROTATE_CHECK = now
                try:
                    if os.path.exists(path) and os.path.getsize(path) > _LOG_ROTATE_BYTES:
                        rotated = path + ".1"
                        if os.path.exists(rotated):
                            os.remove(rotated)
                        os.replace(path, rotated)
                except OSError:
                    pass
            with open(path, "a", encoding="utf-8") as f:
                f.write(line)
    except OSError:
        pass  # 日志失败不应影响主流程


def _truncate(s: str, n: int = 80) -> str:
    s = str(s).replace("\n", " ").strip()
    return s[:n] + ("…" if len(s) > n else "")


def _log_body_dump(tag: str, body: dict) -> None:
    """本地补丁 #9（2026-09-18）：转储请求体到日志；超大 body 只保留头尾。

    尺寸感知直通（补丁 #6）后，单请求体可达数 MB，全文转储会让 converter.log
    每轮涨数 MB。超过 6 万字符只保留头 4 万 + 尾 1 万，附总量标注。
    """
    try:
        text = json.dumps(body, ensure_ascii=False, indent=2)
    except Exception:
        text = repr(body)
    if len(text) <= 60_000:
        _log(f"{tag}\n{text}")
        return
    head, tail = text[:40_000], text[-10_000:]
    omitted = len(text) - len(head) - len(tail)
    _log(
        f"{tag}\n{head}\n"
        f"... [TRUNCATED {omitted} chars; total {len(text)} chars - patch #9] ...\n{tail}"
    )


def _check_auth(authorization: str | None, x_api_key: str | None):
    key = CONFIG["api_key"]
    if not key:
        return
    token = ""
    if authorization and authorization.startswith("Bearer "):
        token = authorization[7:].strip()
    if not token and x_api_key:
        token = x_api_key
    if token != key:
        raise HTTPException(
            status_code=401,
            detail={"error": {"message": "invalid api key", "type": "auth_error"}},
        )


def _cred() -> CredentialManager:
    # The management deployment binds one credential to each ASGI request.
    # Standalone converter usage keeps the original single-account behavior.
    try:
        from admin.pool import REQUEST_CREDENTIAL
        selected = REQUEST_CREDENTIAL.get()
        if selected is not None:
            return selected
    except ImportError:
        pass
    if CONFIG["cred"] is None:
        raise HTTPException(
            status_code=503,
            detail={
                "error": {
                    "message": "未找到登录凭据，请先在桌面端登录 CodeBuddy/WorkBuddy",
                    "type": "auth_error",
                }
            },
        )
    return CONFIG["cred"]


@app.get("/health")
def health():
    cred = CONFIG["cred"]
    info: dict = {
        "status": "ok",
        "platform": sys.platform,
        "python": sys.version.split()[0],
        "auth_file": str(find_auth_file() or "(未找到)"),
        "mode": "direct-proxy (native function calling)",
    }
    if cred is not None:
        try:
            info["credential"] = cred.summary()
        except Exception as e:
            info["credential_error"] = str(e)
    return info


@app.get("/v1/models")
def list_models(
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
):
    _check_auth(authorization, x_api_key)
    models = get_available_models()
    data = [
        {"id": m, "object": "model", "created": 1700000000, "owned_by": "codebuddy"}
        for m in models
    ]
    return {"object": "list", "data": data}


@app.post("/v1/chat/completions")
async def chat_completions(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
):
    _check_auth(authorization, x_api_key)
    cred = _cred()

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {"message": f"bad json: {e}", "type": "invalid_request_error"}
            },
        )

    messages = payload.get("messages") or []
    if not messages:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "messages is required",
                    "type": "invalid_request_error",
                }
            },
        )

    # 构造后端 body：只透传已知的合法字段
    client_wants_stream = bool(payload.get("stream"))
    body = {k: payload[k] for k in PASSTHROUGH_BODY_KEYS if k in payload}
    body["model"] = resolve_model(body.get("model"))
    # 补丁 #5：无 effort 时补默认，保证后端输出思考流
    body.setdefault("reasoning_effort", DEFAULT_REASONING_EFFORT)
    # 后端只支持流式：始终以 stream=True 调后端，非流式由转换器聚合
    body["stream"] = True
    if "stream_options" not in body:
        body["stream_options"] = {"include_usage": True}

    # 腾讯后端不支持 developer role，遇到会触发安全策略拦截（11128），统一映射为 system
    if "messages" in body and isinstance(body["messages"], list):
        body["messages"] = [
            dict(m, role="system")
            if isinstance(m, dict) and m.get("role") == "developer"
            else m
            for m in body["messages"]
        ]

    # 可选：脱敏。缓解客户端合规模板（如 Codex CLI / ZCode 注入的说明文字）被后端误判为敏感词。
    # 处理 system / developer 消息、Codex 注入的上下文 user 消息，以及 tools 的 description。
    if CONFIG.get("desensitize"):
        body = desensitize_body(
            body,
            roles=("system", "developer"),
            desensitize_harness_user=True,
            desensitize_tools=True,
            compact_harness=not CONFIG.get("no_compact"),
            strip_tool_metadata=True,
        )

    # 日志：请求摘要
    model_name = payload.get("model", "auto")
    tool_names = [
        t.get("function", {}).get("name")
        for t in (payload.get("tools") or [])
        if isinstance(t, dict)
    ]
    last_user = _last_user_text(messages)
    rid = os.urandom(4).hex()
    _log(
        f"[{rid}] ▶ REQUEST {model_name} | stream={client_wants_stream} | msgs={len(messages)}"
        + (f" | tools={tool_names}" if tool_names else "")
        + (f" | last_user={_truncate(last_user, 60)!r}" if last_user else "")
    )
    # 完整请求体（发往后端的实际内容；若启用脱敏，这里已是脱敏后）
    # 修复（2026-09-19 审查）：补丁 #9 漏改此端点——大 body 仍全文转储，统一走截断转储
    _log_body_dump(f"[{rid}] ── REQUEST BODY (发往后端) ──", body)

    headers = await asyncio.to_thread(cred.get_headers)
    url = f"{BACKEND}/v2/chat/completions"
    t0 = time.time()

    if client_wants_stream:
        return StreamingResponse(
            _stream_upstream(url, headers, body, model_name, t0, rid),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 非流式：后端只支持流式，这里把后端 SSE 聚合成单个 chat.completion 响应
    try:
        async with httpx.AsyncClient(timeout=300) as c:
            async with c.stream("POST", url, headers=headers, json=body) as r:
                if r.status_code != 200:
                    raw = await r.aread()
                    _log(
                        f"[{rid}] ✗ HTTP {r.status_code} | {model_name} | {_truncate(raw.decode('utf-8', 'replace'), 200)}"
                    )
                    _log(f"[{rid}] ── ERROR BODY ──\n{raw.decode('utf-8', 'replace')}")
                    raise HTTPException(
                        status_code=r.status_code,
                        detail=_safe_err_raw(raw, r.status_code),
                    )
                collected = await _collect_stream(r)
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        _log(f"[{rid}] ✗ 网络错误 | {model_name} | {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "error": {"message": f"upstream error: {e}", "type": "upstream_error"}
            },
        )
    _log_finish(model_name, t0, collected, rid)
    return JSONResponse(content=collected)


def _last_user_text(messages: list) -> str:
    """取最后一条 user 消息的文本，用于日志预览。"""
    for m in reversed(messages):
        if m.get("role") != "user":
            continue
        content = m.get("content", "")
        if isinstance(content, list):
            for blk in content:
                if isinstance(blk, dict) and blk.get("type") == "text":
                    return str(blk.get("text", ""))
            return ""
        return str(content)
    return ""


def _log_finish(model_name: str, t0: float, result: dict, rid: str = ""):
    """记录一次完成的请求：耗时 / finish_reason / usage / 工具调用 / 审核拦截 + 完整响应。"""
    elapsed = time.time() - t0
    prefix = f"[{rid}] " if rid else ""
    choice = (result.get("choices") or [{}])[0]
    finish = choice.get("finish_reason")
    msg = choice.get("message") or {}
    tcs = msg.get("tool_calls") or []
    usage = result.get("usage") or {}
    tag = ""
    if finish == "content-filter":
        tag = " ⚠️内容审核拦截"
    tc_names = [t.get("function", {}).get("name") for t in tcs]
    _log(
        f"{prefix}◀ RESPONSE {model_name} | {elapsed:.1f}s | finish={finish}{tag}"
        + (f" | tool_calls={tc_names}" if tc_names else "")
        + f" | tokens={usage.get('total_tokens', '?')}"
    )
    # 完整响应体
    _log(
        f"{prefix}── RESPONSE BODY ──\n{json.dumps(result, ensure_ascii=False, indent=2)}"
    )


async def _collect_stream(response: httpx.Response) -> dict:
    """消费后端的 OpenAI SSE 流，聚合成单个非流式 chat.completion 对象。

    合并所有 chunk 的 delta（content / tool_calls），并取 usage / finish_reason。
    """
    content_parts: list[str] = []
    # tool_calls: index -> {id, name, arguments(分片拼接)}
    tool_calls: dict[int, dict] = {}
    model: str | None = None
    finish_reason: str | None = None
    usage: dict | None = None

    async for line in response.aiter_lines():
        line = line.strip()
        if not line or not line.startswith("data:"):
            continue
        data = line[5:].strip()
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        model = chunk.get("model") or model
        if chunk.get("usage"):
            usage = chunk["usage"]
        for choice in chunk.get("choices") or []:
            if choice.get("finish_reason"):
                finish_reason = choice["finish_reason"]
            delta = choice.get("delta") or {}
            if delta.get("content"):
                content_parts.append(delta["content"])
            for tc in delta.get("tool_calls") or []:
                idx = tc.get("index", 0)
                slot = tool_calls.setdefault(
                    idx, {"id": None, "name": None, "arguments": ""}
                )
                if tc.get("id"):
                    slot["id"] = tc["id"]
                fn = tc.get("function") or {}
                if fn.get("name"):
                    slot["name"] = fn["name"]
                if fn.get("arguments"):
                    slot["arguments"] += fn["arguments"]

    tcs = None
    if tool_calls:
        tcs = [
            {
                "id": v["id"],
                "type": "function",
                "function": {"name": v["name"], "arguments": v["arguments"]},
            }
            for _, v in sorted(tool_calls.items())
        ]
        finish_reason = finish_reason or "tool_calls"

    message = {"role": "assistant", "content": "".join(content_parts) or None}
    if tcs:
        message["tool_calls"] = tcs
    return {
        "id": "chatcmpl-" + os.urandom(12).hex(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model or "unknown",
        "choices": [
            {"index": 0, "message": message, "finish_reason": finish_reason or "stop"}
        ],
        "usage": usage
        or {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def _safe_err_raw(raw: bytes, status: int) -> dict:
    try:
        return json.loads(raw.decode("utf-8", "replace"))
    except Exception:
        return {
            "error": {
                "message": raw.decode("utf-8", "replace")[:500],
                "type": "upstream_error",
                "code": status,
            }
        }


async def _stream_upstream(
    url: str,
    headers: dict,
    body: dict,
    model_name: str = "?",
    t0: float = 0.0,
    rid: str = "",
):
    """把后端 SSE 原样转发给客户端（后端已是标准 OpenAI SSE，含 tool_calls）。

    同时轻量解析流，统计 finish_reason / tool_calls / usage 用于日志，不阻塞转发。
    完整原始 SSE 累积后落盘到日志（调试用）。
    """
    finish_reason = None
    tool_names: list[str] = []
    usage: dict = {}
    saw_filter = False
    buf = b""
    # 修复 B2（2026-09-19 审查）：原 list 全量累积整条流（多 MB × 并发 = 内存尖峰），
    # 与 responses 路径对齐——只为日志保留最后 30 块
    raw_parts: deque = deque(maxlen=30)
    prefix = f"[{rid}] " if rid else ""

    def _feed(chunk: bytes):
        nonlocal finish_reason, saw_filter, buf
        # 行缓冲解析：把累计的 chunk 按 data: 行切出来统计
        buf += chunk
        while b"\n" in buf:
            line, buf = buf.split(b"\n", 1)
            line = line.strip()
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                continue
            try:
                obj = json.loads(data)
            except Exception:
                continue
            # 修复 F1（2026-09-19 审查）：顶层非 dict 的合法 JSON（如 data: 123）直接跳过
            if not isinstance(obj, dict):
                continue
            if obj.get("usage"):
                usage.update(obj["usage"])
            for ch in obj.get("choices") or []:
                # F1：choices 元素非 dict 时跳过
                if not isinstance(ch, dict):
                    continue
                if ch.get("finish_reason"):
                    finish_reason = ch["finish_reason"]
                # F1：delta 为 null / 非 dict 时按空 delta 处理
                delta = ch.get("delta") or {}
                if not isinstance(delta, dict):
                    delta = {}
                for tc in delta.get("tool_calls") or []:
                    if not isinstance(tc, dict):
                        continue
                    nm = (tc.get("function") or {}).get("name")
                    if nm:
                        tool_names.append(nm)
            # 内容审核拦截常以 content-filter 或特殊中文文案返回
            try:
                text_repr = data.decode("utf-8", "replace")
            except Exception:
                text_repr = ""
            if (
                "content-filter" in text_repr
                or "敏感" in text_repr
                or "审核" in text_repr
            ):
                saw_filter = True

    try:
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", url, headers=headers, json=body) as r:
                if r.status_code != 200:
                    err = await r.aread()
                    _log(
                        f"{prefix}✗ HTTP {r.status_code} | {model_name} | {_truncate(err.decode('utf-8', 'replace'), 200)}"
                    )
                    _log(f"{prefix}── ERROR BODY ──\n{err.decode('utf-8', 'replace')}")
                    yield _err_event(err, r.status_code)
                    return
                async for chunk in r.aiter_bytes():
                    if chunk:
                        raw_parts.append(chunk)
                        _feed(chunk)
                        yield chunk
    except httpx.HTTPError as e:
        _log(f"{prefix}✗ 网络错误 | {model_name} | {e}")
        yield _err_event(str(e).encode(), 502)

    # 流结束：输出完成日志
    elapsed = time.time() - t0 if t0 else 0
    tag = " ⚠️内容审核拦截" if (saw_filter or finish_reason == "content-filter") else ""
    _log(
        f"{prefix}◀ RESPONSE {model_name} | {elapsed:.1f}s | stream finish={finish_reason}{tag}"
        + (f" | tool_calls={tool_names}" if tool_names else "")
        + f" | tokens={usage.get('total_tokens', '?')}"
    )
    # 原始 SSE 尾部（B2：仅最后 30 块，deque 直接 join）
    _log(
        f"{prefix}── RESPONSE RAW SSE (tail) ──\n{b''.join(raw_parts).decode('utf-8', 'replace')}"
    )


def _safe_err(r: httpx.Response) -> dict:
    try:
        return {"error": r.json()}
    except Exception:
        return {
            "error": {
                "message": r.text[:500],
                "type": "upstream_error",
                "code": r.status_code,
            }
        }


def _err_event(msg: bytes, status: int) -> bytes:
    # 以 OpenAI SSE 错误 chunk 形式返回
    import json as _json

    chunk = {
        "error": {
            "message": msg.decode("utf-8", "replace")[:500],
            "type": "upstream_error",
            "code": status,
        },
    }
    return f"data: {_json.dumps(chunk, ensure_ascii=False)}\n\n".encode()


def _looks_like_content_filter_text(text: str) -> bool:
    text = (text or "").lower()
    return (
        "content-filter" in text
        or "content_filter" in text
        or "敏感内容" in text
        or "内容审核" in text
        or "无法响应您的请求" in text
    )


def _chat_body_desensitize(body: dict, *, force_compact: bool = False) -> dict:
    if not CONFIG.get("desensitize"):
        return body
    return desensitize_body(
        body,
        roles=("system", "developer"),
        desensitize_harness_user=True,
        desensitize_tools=True,
        compact_harness=(force_compact or not CONFIG.get("no_compact")),
        strip_tool_metadata=True,
    )


# ---------------------------------------------------------------------------
# 本地补丁 #11（2026-09-20）：后端限流/瞬时故障 建连阶段指数退避重试
#
# 背景：kimi-k3-1 高峰期被后端限流（HTTP 429 code 14003 "too many requests"，
# 峰值 ~5 次/分钟），每次 429 直接穿透为 Codex 端 "stream disconnected →
# 正在重新连接" 风暴；另有 Modern Standby 唤醒瞬间 DNS 未就绪导致的
# getaddrinfo 连接失败（09:30 唤醒后 100 秒内 10 次）。
# 方案：仅在"尚未向客户端输出任何字节"的建连阶段重试（无输出重复风险），
# 429/502/503/504 与连接类网络错误按 2/4/8/16s（封顶 20s）指数退避 + 抖动，
# 并尊重 Retry-After 响应头。重试耗尽后按原有错误路径返回。
# ---------------------------------------------------------------------------
_RATE_LIMIT_RETRY_MAX = 4      # 首次尝试之外的最大重试次数
_RATE_LIMIT_RETRY_BASE = 2.0   # 退避基数（秒）
_RETRYABLE_STATUS = (429, 502, 503, 504)


def _retry_backoff_seconds(attempt: int, retry_after: str | None = None) -> float:
    """指数退避 + 尊重 Retry-After + 0~1s 抖动（错开并发会话同时重试）。"""
    delay = min(_RATE_LIMIT_RETRY_BASE * (2 ** attempt), 20.0)
    if retry_after:
        try:
            delay = max(delay, float(retry_after))
        except ValueError:
            pass
    return delay + (os.urandom(1)[0] / 255.0)


async def _post_backend_once(
    url: str, headers: dict, body: dict, rid: str = "", model_name: str = "?"
) -> tuple[int, bytes]:
    """单次后端 POST（整体缓冲，非流式路径）。

    补丁 #11：非流式请求完整缓冲、天然幂等——对 429/502/503/504 做指数
    退避重试后再返回，与流式路径 _open_backend_stream 行为对齐。
    """
    prefix = f"[{rid}] " if rid else ""
    status, raw = 0, b""
    for attempt in range(_RATE_LIMIT_RETRY_MAX + 1):
        retry_after: str | None = None
        async with httpx.AsyncClient(timeout=120) as c:
            async with c.stream("POST", url, headers=headers, json=body) as r:
                chunks: list[bytes] = []
                async for chunk in r.aiter_bytes():
                    if chunk:
                        chunks.append(chunk)
                status, raw = r.status_code, b"".join(chunks)
                retry_after = r.headers.get("retry-after")
        if status in _RETRYABLE_STATUS and attempt < _RATE_LIMIT_RETRY_MAX:
            delay = _retry_backoff_seconds(attempt, retry_after)
            _log(
                f"{prefix}↻ {model_name} | HTTP {status}，{delay:.1f}s 后重试"
                f"（{attempt + 1}/{_RATE_LIMIT_RETRY_MAX}）"
            )
            await asyncio.sleep(delay)
            continue
        return status, raw
    return status, raw


async def _open_backend_stream(
    url: str, headers: dict, body: dict, model_name: str = "?", prefix: str = ""
):
    """打开后端流式连接（补丁 #11：建连阶段自带退避重试）。

    返回 (client, ctx, response)；调用方必须在 finally 中先后关闭 ctx 与
    client。重试仅发生在建连阶段（零输出），因此不会造成输出重复：
    - 429/502/503/504：读尽错误体 → 关闭 → 退避 → 重连
    - ConnectError/ConnectTimeout（DNS 未就绪/瞬时抖动）：退避 → 重连
    重试耗尽后返回最后一次结果，由调用方按原有错误路径处理。
    """
    for attempt in range(_RATE_LIMIT_RETRY_MAX + 1):
        client = httpx.AsyncClient(timeout=httpx.Timeout(600.0, connect=30.0))
        try:
            ctx = client.stream("POST", url, headers=headers, json=body)
            resp = await ctx.__aenter__()
        except (httpx.ConnectError, httpx.ConnectTimeout) as e:
            await client.aclose()
            if attempt < _RATE_LIMIT_RETRY_MAX:
                delay = _retry_backoff_seconds(attempt)
                _log(
                    f"{prefix}↻ {model_name} | 建连失败（{type(e).__name__}: {e}），"
                    f"{delay:.1f}s 后重试（{attempt + 1}/{_RATE_LIMIT_RETRY_MAX}）"
                )
                await asyncio.sleep(delay)
                continue
            raise
        except Exception:
            await client.aclose()
            raise
        if resp.status_code in _RETRYABLE_STATUS and attempt < _RATE_LIMIT_RETRY_MAX:
            retry_after = resp.headers.get("retry-after")
            raw = b""
            async for chunk in resp.aiter_bytes():
                if chunk:
                    raw += chunk
            _log(
                f"{prefix}↻ {model_name} | HTTP {resp.status_code} 限流/过载，"
                f"退避重试（{attempt + 1}/{_RATE_LIMIT_RETRY_MAX}） | "
                f"{_truncate(raw.decode('utf-8', 'replace'), 120)}"
            )
            await ctx.__aexit__(None, None, None)
            await client.aclose()
            await asyncio.sleep(_retry_backoff_seconds(attempt, retry_after))
            continue
        return client, ctx, resp
    raise httpx.ConnectError("backend stream: retry loop exhausted")


async def _post_backend_with_filter_retry(
    url: str, headers: dict, body: dict, rid: str = "", model_name: str = "?"
) -> tuple[int, bytes, dict]:
    prefix = f"[{rid}] " if rid else ""
    status, raw = await _post_backend_once(url, headers, body, rid, model_name)
    text = raw.decode("utf-8", "replace")
    if (
        status == 200
        and _looks_like_content_filter_text(text)
        and CONFIG.get("desensitize")
        and CONFIG.get("no_compact")
    ):
        retry_body = _chat_body_desensitize(body, force_compact=True)
        _log(
            f"{prefix}↻ RESPONSES {model_name} | content filter detected, retry with compact harness"
        )
        # 修复（2026-09-19 审查）：补丁 #9 漏改——重试 body 也走截断转储
        _log_body_dump(f"{prefix}── RESPONSES RETRY CHAT BODY ──", retry_body)
        retry_status, retry_raw = await _post_backend_once(
            url, headers, retry_body, rid, model_name
        )
        retry_text = retry_raw.decode("utf-8", "replace")
        if retry_status == 200 and not _looks_like_content_filter_text(retry_text):
            return retry_status, retry_raw, retry_body
    return status, raw, body


# ---------------------------------------------------------------------------
# Responses API 端点（Codex CLI 兼容）
# ---------------------------------------------------------------------------


@app.post("/v1/responses")
async def create_response(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
):
    """OpenAI Responses API 兼容端点。

    Codex CLI 使用 Responses API（wire_api = "responses"）而非 Chat Completions。
    本端点接收 Responses 格式请求，转换为 Chat 格式发往后端，再将后端的 Chat SSE
    转换为 Responses 语义事件流返回。
    """
    _check_auth(authorization, x_api_key)
    cred = _cred()

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {"message": f"bad json: {e}", "type": "invalid_request_error"}
            },
        )

    # 补丁 #14（2026-09-21，docs/24 G-02 / docs/25 §5 方法①）：入站工具观测。
    # 此前 converter 只 dump 出站 body，无法区分「Codex 没发」/「发了但非 function」/
    # 「发了是 function 仍被丢」三种假设。本行日志把入站 tools[] 的 name+type
    # 一次性记全（纯新增观测，不改任何行为）。
    rid = os.urandom(4).hex()
    inbound_tools = payload.get("tools") or []
    if isinstance(inbound_tools, list) and inbound_tools:
        in_digest = ", ".join(
            f"{t.get('name') or '?'}:{t.get('type') or '?'}"
            for t in inbound_tools if isinstance(t, dict)
        ) or "-"
        _log(f"[{rid}] ── RESPONSES TOOLS IN ── n={len(inbound_tools)} | {in_digest}")
        # R2：降级动作留痕 —— 非 function 类型已被 #14 包装为 function
        degraded = [
            f"{t.get('name') or '?'}({t.get('type')})→function"
            for t in inbound_tools
            if isinstance(t, dict) and t.get("type") != "function"
        ]
        if degraded:
            _log(
                f"[{rid}] ── RESPONSES TOOLS DEGRADE ── action=wrap "
                f"| count={len(degraded)} | " + ", ".join(degraded)
                + " | reason=chat-completions-backend | reversible=yes"
            )
    # 补丁 #14：入站声明为 custom/freeform 的工具名 → 响应侧还原为原生 custom_tool_call
    custom_tool_names = {
        t.get("name") for t in (inbound_tools if isinstance(inbound_tools, list) else [])
        if isinstance(t, dict) and t.get("type") == "custom" and t.get("name")
    }
    # namespace 工具在请求侧展平为 <namespace>__<subtool>；响应侧需还原子工具名。
    namespace_tool_names: dict[str, str] = {}
    for t in inbound_tools if isinstance(inbound_tools, list) else []:
        if not isinstance(t, dict) or t.get("type") != "namespace":
            continue
        namespace = t.get("name")
        for subtool in t.get("tools") or []:
            if isinstance(subtool, dict) and namespace and subtool.get("name"):
                namespace_tool_names[f"{namespace}__{subtool['name']}"] = subtool["name"]

    # 转换请求：Responses → Chat
    # 补丁 #16：norm_log 收集工具输出形态归一化记录（R2：降级必须留痕）
    norm_log: list = []
    try:
        chat_body = responses_request_to_chat(payload, norm_log)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"request conversion error: {e}",
                    "type": "invalid_request_error",
                }
            },
        )

    if norm_log:
        shapes = ", ".join(
            f"{r['shape']}(parts={r['parts']},text={r['texts']},img={r['images']}"
            + (f",degraded={len(r['degraded'])}" if r["degraded"] else "")
            + ")"
            for r in norm_log
        )
        degraded_total = sum(len(r["degraded"]) for r in norm_log)
        _log(
            f"[{rid}] ── RESPONSES OUTPUT NORM ── action=normalize | "
            f"count={len(norm_log)} | {shapes} | "
            f"reason=chat-rejects-responses-part-types | reversible=no"
            + (f" | degraded_types={sorted({d for r in norm_log for d in r['degraded']})}"
               if degraded_total else "")
        )

    chat_body["model"] = resolve_model(chat_body.get("model"))
    # 先解析模型名再投影（投影需按模型上下文决定直通/压缩）
    chat_body, projection_stats = project_responses_chat_body(chat_body)
    # 补丁 #12（2026-09-20）：投影前的真实上下文体积，用于按真实值上报 usage
    real_input_tokens = 0
    if CONFIG.get("report_real_usage", True):
        real_input_tokens = int(projection_stats.get("original_token_estimate") or 0)
    # 补丁 #5：无 effort 时补默认，保证后端输出思考流
    chat_body.setdefault("reasoning_effort", DEFAULT_REASONING_EFFORT)
    chat_body["stream"] = True
    if "stream_options" not in chat_body:
        chat_body["stream_options"] = {"include_usage": True}

    chat_body = _chat_body_desensitize(chat_body)

    client_wants_stream = payload.get("stream", True)  # Codex CLI 默认 stream
    model_name = payload.get("model", "auto")
    _log(
        f"[{rid}] ▶ RESPONSES {model_name} | stream={client_wants_stream} | input_items={len(payload.get('input', []))}"
    )
    _log(
        f"[{rid}] ── RESPONSES PROJECTION ── "
        f"mode={projection_stats.get('mode')} "
        f"| msgs {projection_stats.get('original_messages')}→{projection_stats.get('projected_messages')} "
        f"| chars {projection_stats.get('original_message_chars')}→{projection_stats.get('projected_message_chars')} "
        f"| tools {projection_stats.get('original_tools')}→{projection_stats.get('projected_tools')} "
        f"| tool_chars {projection_stats.get('original_tool_chars')}→{projection_stats.get('projected_tool_chars')} "
        f"| summarized_history={projection_stats.get('summarized_history_messages', 0)} "
        f"| dropped_harness={projection_stats.get('dropped_harness_messages', 0)} "
        f"| anchor_user={projection_stats.get('anchor_user_preserved', False)} "
        f"| tokens {projection_stats.get('original_token_estimate')}→"
        f"{projection_stats.get('projected_token_estimate')} "
        f"| reported_input_tokens={real_input_tokens or 'upstream'}"
    )
    # 补丁 #14：出站工具清单（与 TOOLS IN 同格式，in/out 差分可直接 grep）
    out_tools = chat_body.get("tools") or []
    if out_tools:
        out_digest = ", ".join(
            ((t.get("function") or {}).get("name") or t.get("name") or "?")
            for t in out_tools if isinstance(t, dict)
        ) or "-"
        _log(f"[{rid}] ── RESPONSES TOOLS OUT ── n={len(out_tools)} | {out_digest}")
    _log_body_dump(f"[{rid}] ── RESPONSES → CHAT BODY ──", chat_body)

    headers = await asyncio.to_thread(cred.get_headers)
    url = f"{BACKEND}/v2/chat/completions"
    t0 = time.time()

    if client_wants_stream:
        return StreamingResponse(
            _stream_responses(
                url, headers, chat_body, model_name, t0, rid,
                real_input_tokens=real_input_tokens,
                custom_tool_names=custom_tool_names,
                namespace_tool_names=namespace_tool_names,
            ),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 非流式：聚合后端 SSE → 非流式 Response 对象
    try:
        status_code, raw, final_body = await _post_backend_with_filter_retry(
            url, headers, chat_body, rid, model_name
        )
        if status_code != 200:
            _log(
                f"[{rid}] ✗ HTTP {status_code} | {model_name} | {_truncate(raw.decode('utf-8', 'replace'), 200)}"
            )
            raise HTTPException(
                status_code=status_code, detail=_safe_err_raw(raw, status_code)
            )
        converter = ResponsesStreamConverter(
            model=model_name, real_input_tokens=real_input_tokens,
            custom_tool_names=custom_tool_names,
            namespace_tool_names=namespace_tool_names,
        )
        for line in raw.decode("utf-8", "replace").splitlines():
            converter.feed_line(line)
        chat_body = final_body
    except HTTPException:
        raise
    except httpx.HTTPError as e:
        _log(f"[{rid}] ✗ 网络错误 | {model_name} | {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "error": {"message": f"upstream error: {e}", "type": "upstream_error"}
            },
        )

    result = converter.get_nonstream_response()
    elapsed = time.time() - t0
    _log(f"[{rid}] ◀ RESPONSES {model_name} | {elapsed:.1f}s")
    _log(
        f"[{rid}] ── RESPONSE OBJ ──\n{json.dumps(result, ensure_ascii=False, indent=2)}"
    )
    return JSONResponse(content=result)


async def _stream_responses(
    url: str,
    headers: dict,
    body: dict,
    model_name: str = "?",
    t0: float = 0.0,
    rid: str = "",
    real_input_tokens: int = 0,
    custom_tool_names: set[str] | None = None,
    namespace_tool_names: dict[str, str] | None = None,
):
    """消费后端 Chat SSE，实时转换为 Responses API 事件流输出。

    本地补丁 #7（2026-09-18）：真流式。
    原版先 await 完整后端响应（_post_backend_with_filter_retry 缓冲全部字节）再逐行
    "回放"，导致 Codex 端思考全程黑盒等待（后端实际 ~2s 就开始流式输出思考）。
    现改为 httpx 流式读取 + 逐行即时转发。
    注：content-filter 重试仅在看板 no_compact 模式下有意义，流式路径不再支持
    中途重试（过滤事件会原样透传给客户端）。

    本地补丁 #11（2026-09-20）：建连阶段经 _open_backend_stream 做 429/5xx 与
    连接类错误的指数退避重试——后端限流尖峰不再直接穿透为 Codex 端
    "stream disconnected → 正在重新连接" 风暴。
    """
    converter = ResponsesStreamConverter(
        model=model_name, real_input_tokens=real_input_tokens,
        custom_tool_names=custom_tool_names,
        namespace_tool_names=namespace_tool_names,
    )
    prefix = f"[{rid}] " if rid else ""

    # 补丁 #11：建连（含退避重试）。此阶段尚未向客户端输出任何字节。
    try:
        client, ctx, r = await _open_backend_stream(url, headers, body, model_name, prefix)
    except httpx.HTTPError as e:
        _log(f"{prefix}✗ 网络错误 | {model_name} | {e}")
        error_evt = {"type": "error", "error": {"message": str(e)[:500], "code": 502}}
        yield f"data: {json.dumps(error_evt, ensure_ascii=False)}\n\n".encode()
        return

    try:
        if r.status_code != 200:
            raw = b""
            async for chunk in r.aiter_bytes():
                if chunk:
                    raw += chunk
            _log(
                f"{prefix}✗ HTTP {r.status_code} | {model_name} | "
                f"{_truncate(raw.decode('utf-8', 'replace'), 200)}"
            )
            error_evt = {
                "type": "error",
                "error": {
                    "message": raw.decode("utf-8", "replace")[:500],
                    "code": r.status_code,
                },
            }
            yield f"data: {json.dumps(error_evt, ensure_ascii=False)}\n\n".encode()
            return

        # 修复（2026-09-19 审查）：原 list 无限累积整个流（多 MB 响应 × 并发 = 内存尖峰），
        # 只为日志保留最后 30 行——改用有界 deque
        raw_sse_lines: deque = deque(maxlen=30)
        async for line in r.aiter_lines():
            if line.strip():
                raw_sse_lines.append(line)
            events = converter.feed_line(line)
            if events:
                yield events.encode("utf-8")
    except httpx.HTTPError as e:
        _log(f"{prefix}✗ 网络错误 | {model_name} | {e}")
        error_evt = {"type": "error", "error": {"message": str(e)[:500], "code": 502}}
        yield f"data: {json.dumps(error_evt, ensure_ascii=False)}\n\n".encode()
        return
    except Exception as e:
        # 修复 F1（2026-09-19 审查）：兜底——未预期异常不再穿透生成器硬截断 SSE 流，
        # 对齐 httpx.HTTPError 分支：yield error 事件 + 记录 traceback
        _log(
            f"{prefix}✗ 流处理异常 | {model_name} | {type(e).__name__}: {e}\n"
            + traceback.format_exc()
        )
        error_evt = {
            "type": "error",
            "error": {"message": f"{type(e).__name__}: {e}"[:500], "code": 502},
        }
        yield f"data: {json.dumps(error_evt, ensure_ascii=False)}\n\n".encode()
        return
    finally:
        # 补丁 #11：ctx（后端响应）与 client 均由本生成器负责关闭
        try:
            await ctx.__aexit__(None, None, None)
        except Exception:
            pass
        await client.aclose()

    # 发送收尾事件
    finish_events = converter.finish()
    if finish_events:
        yield finish_events.encode("utf-8")

    elapsed = time.time() - t0 if t0 else 0
    _log(f"{prefix}◀ RESPONSES {model_name} | {elapsed:.1f}s | stream done")
    _log(f"{prefix}── RESPONSES RAW SSE ──\n" + "\n".join(raw_sse_lines))


# ---------------------------------------------------------------------------
# Anthropic Messages API 端点（Claude Code / CC Switch 兼容）
# ---------------------------------------------------------------------------


@app.post("/v1/messages")
async def create_message(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
):
    """Anthropic Messages API 兼容端点。

    Claude Code / CC Switch 使用 Anthropic Messages API（POST /v1/messages）。
    本端点接收 Anthropic 格式请求，转换为 Chat 格式发往后端，再将后端的 Chat SSE
    转换为 Anthropic SSE 事件流返回。
    """
    _check_auth(authorization, x_api_key)
    cred = _cred()

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {"message": f"bad json: {e}", "type": "invalid_request_error"}
            },
        )

    # 将 Anthropic 格式消息、工具规范在进入后端前统一转换为 OpenAI Chat 格式。
    messages = payload.get("messages") or []
    if not messages:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "messages is required",
                    "type": "invalid_request_error",
                }
            },
        )

    try:
        chat_body = anthropic_request_to_chat(payload)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"request conversion error: {e}",
                    "type": "invalid_request_error",
                }
            },
        )

    chat_body["model"] = resolve_model(chat_body.get("model"))
    # 补丁 #5：无 effort 时补默认，保证后端输出思考流
    chat_body.setdefault("reasoning_effort", DEFAULT_REASONING_EFFORT)
    # 读取用户的 stream 参数，如果未提供则默认为 True
    user_stream = payload.get("stream", True)
    # 无论用户如何设置，都向后端请求流式响应（后端只支持流式）
    chat_body["stream"] = True
    if "stream_options" not in chat_body:
        chat_body["stream_options"] = {"include_usage": True}

    if CONFIG.get("desensitize"):
        chat_body = desensitize_body(
            chat_body,
            roles=("system", "developer"),
            desensitize_harness_user=True,
            desensitize_tools=True,
            compact_harness=not CONFIG.get("no_compact"),
            strip_tool_metadata=True,
        )

    model_name = payload.get("model", "auto")
    chat_messages = chat_body.get("messages", [])
    rid = os.urandom(4).hex()
    _log(
        f"[{rid}] ▶ ANTHROPIC {model_name} | msgs={len(chat_messages)} | anthropic_msgs={len(messages)} | user_stream={user_stream}"
    )
    _log_body_dump(f"[{rid}] ── ANTHROPIC → CHAT BODY ──", chat_body)

    headers = await asyncio.to_thread(cred.get_headers)
    url = f"{BACKEND}/v2/chat/completions"
    t0 = time.time()

    # 如果用户请求流式响应，直接返回流式
    if user_stream:
        return StreamingResponse(
            _stream_anthropic(url, headers, chat_body, model_name, t0, rid),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    # 否则，收集完整响应并返回 JSON
    from fastapi.responses import JSONResponse

    response_data = await _collect_anthropic_nonstream(
        url, headers, chat_body, model_name, t0, rid
    )
    return JSONResponse(content=response_data)


async def _collect_anthropic_nonstream(
    url: str,
    headers: dict,
    body: dict,
    model_name: str = "?",
    t0: float = 0.0,
    rid: str = "",
) -> dict:
    """收集完整的流式响应并返回非流式 Anthropic Message 对象。"""
    converter = AnthropicStreamConverter(model=model_name)
    prefix = f"[{rid}] " if rid else ""

    try:
        async with httpx.AsyncClient(timeout=120.0) as c:
            async with c.stream("POST", url, headers=headers, json=body) as r:
                if r.status_code != 200:
                    err = await r.aread()
                    _log(
                        f"{prefix}✗ HTTP {r.status_code} | {model_name} | {_truncate(err.decode('utf-8', 'replace'), 200)}"
                    )
                    raise HTTPException(
                        status_code=r.status_code,
                        detail={
                            "error": {
                                "message": err.decode("utf-8", "replace")[:500],
                                "type": "api_error",
                                "code": r.status_code,
                            }
                        },
                    )
                async for line in r.aiter_lines():
                    converter.feed_line(line)
    except httpx.HTTPError as e:
        _log(f"{prefix}✗ 网络错误 | {model_name} | {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "error": {"message": str(e)[:500], "type": "api_error", "code": 502}
            },
        ) from None

    elapsed = time.time() - t0 if t0 else 0
    _log(f"{prefix}◀ ANTHROPIC {model_name} | {elapsed:.1f}s | nonstream done")
    return converter.get_nonstream_response()


async def _stream_anthropic(
    url: str,
    headers: dict,
    body: dict,
    model_name: str = "?",
    t0: float = 0.0,
    rid: str = "",
):
    """消费后端 OpenAI Chat SSE，实时转换为 Anthropic Messages SSE 事件流。"""
    converter = AnthropicStreamConverter(model=model_name)
    prefix = f"[{rid}] " if rid else ""

    try:
        async with httpx.AsyncClient(timeout=None) as c:
            async with c.stream("POST", url, headers=headers, json=body) as r:
                if r.status_code != 200:
                    err = await r.aread()
                    _log(
                        f"{prefix}✗ HTTP {r.status_code} | {model_name} | {_truncate(err.decode('utf-8', 'replace'), 200)}"
                    )
                    error_evt = {
                        "type": "error",
                        "error": {
                            "message": err.decode("utf-8", "replace")[:500],
                            "type": "api_error",
                            "code": r.status_code,
                        },
                    }
                    yield f"event: error\ndata: {json.dumps(error_evt, ensure_ascii=False)}\n\n".encode()
                    return
                async for line in r.aiter_lines():
                    events = converter.feed_line(line)
                    if events:
                        yield events.encode("utf-8")
    except httpx.HTTPError as e:
        _log(f"{prefix}✗ 网络错误 | {model_name} | {e}")
        error_evt = {
            "type": "error",
            "error": {"message": str(e)[:500], "type": "api_error", "code": 502},
        }
        yield f"event: error\ndata: {json.dumps(error_evt, ensure_ascii=False)}\n\n".encode()
        return
    except Exception as e:
        # 修复 F1（2026-09-19 审查）：兜底——未预期异常不再穿透生成器硬截断 SSE 流，
        # 对齐 httpx.HTTPError 分支：yield error 事件 + 记录 traceback
        _log(
            f"{prefix}✗ 流处理异常 | {model_name} | {type(e).__name__}: {e}\n"
            + traceback.format_exc()
        )
        error_evt = {
            "type": "error",
            "error": {
                "message": f"{type(e).__name__}: {e}"[:500],
                "type": "api_error",
                "code": 502,
            },
        }
        yield f"event: error\ndata: {json.dumps(error_evt, ensure_ascii=False)}\n\n".encode()
        return

    finish_events = converter.finish()
    if finish_events:
        yield finish_events.encode("utf-8")

    elapsed = time.time() - t0 if t0 else 0
    _log(f"{prefix}◀ ANTHROPIC {model_name} | {elapsed:.1f}s | stream done")


@app.post("/v1/messages/count_tokens")
async def count_tokens(
    request: Request,
    authorization: str | None = Header(default=None),
    x_api_key: str | None = Header(default=None, alias="X-Api-Key"),
):
    """Anthropic token 计数端点。

    Claude Code 在发送消息前调用此端点获取 token 计数。
    后端只支持流式请求，所以我们发送流式请求并从中提取 usage。
    """
    _check_auth(authorization, x_api_key)
    cred = _cred()

    try:
        payload = await request.json()
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {"message": f"bad json: {e}", "type": "invalid_request_error"}
            },
        )

    messages = payload.get("messages") or []
    if not messages:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "messages is required",
                    "type": "invalid_request_error",
                }
            },
        )

    try:
        chat_body = anthropic_request_to_chat(payload)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": f"request conversion error: {e}",
                    "type": "invalid_request_error",
                }
            },
        )

    # 最小化实际生成：只需要 usage 统计
    chat_body["model"] = resolve_model(chat_body.get("model"))
    # 补丁 #5：无 effort 时补默认，保证后端输出思考流
    chat_body.setdefault("reasoning_effort", DEFAULT_REASONING_EFFORT)
    chat_body["max_tokens"] = 1
    chat_body["stream"] = True  # 后端只支持流式
    chat_body["stream_options"] = {"include_usage": True}

    if CONFIG.get("desensitize"):
        chat_body = desensitize_body(
            chat_body,
            roles=("system", "developer"),
            desensitize_harness_user=True,
            desensitize_tools=True,
            compact_harness=not CONFIG.get("no_compact"),
            strip_tool_metadata=True,
        )

    headers = await asyncio.to_thread(cred.get_headers)
    url = f"{BACKEND}/v2/chat/completions"

    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            async with client.stream(
                "POST", url, headers=headers, json=chat_body
            ) as resp:
                if resp.status_code != 200:
                    err = await resp.aread()
                    _log(
                        f"✗ count_tokens HTTP {resp.status_code}: {_truncate(err.decode('utf-8', 'replace'), 200)}"
                    )
                    raise HTTPException(
                        status_code=resp.status_code,
                        detail={
                            "error": {
                                "message": err.decode("utf-8", "replace")[:500],
                                "type": "api_error",
                                "code": resp.status_code,
                            }
                        },
                    )

                # 解析 SSE 流，查找 usage 信息
                # message_start 包含初始 usage（0），message_delta 包含真实 usage
                input_tokens = 0
                async for line in resp.aiter_lines():
                    if not line or line.startswith(":"):
                        continue
                    if line.startswith("data: "):
                        data_str = line[6:]
                        if data_str.strip() == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data_str)
                            # message_delta 事件直接包含 usage
                            if "usage" in chunk:
                                usage = chunk.get("usage") or {}
                                tokens = usage.get("prompt_tokens", 0) or usage.get(
                                    "input_tokens", 0
                                )
                                if tokens > 0:
                                    input_tokens = tokens
                            # message_start 事件在 message 对象中包含 usage
                            elif "message" in chunk and "usage" in chunk["message"]:
                                usage = chunk["message"].get("usage") or {}
                                tokens = usage.get("prompt_tokens", 0) or usage.get(
                                    "input_tokens", 0
                                )
                                if tokens > 0:
                                    input_tokens = tokens
                        except json.JSONDecodeError:
                            continue

                return {"input_tokens": input_tokens}

    except httpx.HTTPError as e:
        _log(f"✗ count_tokens network error: {e}")
        raise HTTPException(
            status_code=502,
            detail={
                "error": {"message": str(e)[:500], "type": "api_error", "code": 502}
            },
        ) from None


# ---------------------------------------------------------------------------
# 启动
# ---------------------------------------------------------------------------


def preflight() -> bool:
    af = find_auth_file()
    sys.stderr.write("==== 预检 ====\n")
    sys.stderr.write(f"平台      : {sys.platform}\n")
    sys.stderr.write(f"Python    : {sys.version.split()[0]}\n")
    sys.stderr.write(f"后端      : {BACKEND} (直连，原生 function calling)\n")
    sys.stderr.write(f"登录文件  : {af or '(未找到)'}\n")
    if auth_dirs():
        sys.stderr.write(f"已查目录  : {', '.join(str(d) for d in auth_dirs())}\n")
    ok = True
    if af is None:
        sys.stderr.write(
            "\n[警告] 未找到登录文件。请在桌面端完成登录（CodeBuddy/WorkBuddy）。\n"
        )
        ok = False
    else:
        try:
            cm = CredentialManager(af)
            info = cm.summary()
            sys.stderr.write(
                f"账号      : {info.get('nickname')} / {info.get('enterpriseName')}\n"
            )
            sys.stderr.write(
                f"token过期 : {'是(将自动刷新)' if info['token_expired'] else '否'}\n"
            )
        except Exception as e:
            sys.stderr.write(f"[警告] 读取凭据失败：{e}\n")
            ok = False
    sys.stderr.write("================\n")
    return ok


def main():
    ap = argparse.ArgumentParser(
        description="CodeBuddy -> OpenAI 兼容转换器（直连后端）"
    )
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8787)
    ap.add_argument(
        "--api-key",
        default=os.environ.get("CODEBUDDY2OPENAI_KEY", ""),
        help="可选：要求客户端携带的 API key（默认不校验）",
    )
    ap.add_argument(
        "--log",
        default=None,
        metavar="PATH",
        help="开启日志并写到该文件（如 --log converter.log 或 --log /tmp/cb.log）。"
        "不传则不记日志。",
    )
    ap.add_argument(
        "--desensitize",
        action="store_true",
        help="启用脱敏：对 system 消息里的合规模板敏感词（DoS/exploit/credential 等）"
        "插入零宽空格，缓解被后端内容审核误拦。默认关闭。",
    )
    ap.add_argument(
        "--no-compact",
        action="store_true",
        help="配合 --desensitize 使用：跳过 system/harness 压缩，仅做零宽脱敏。"
        "保留原始 system prompt 完整内容（如 Claude Code 的行为指令），"
        "但审核误拦风险略高于默认压缩模式。",
    )
    ap.add_argument(
        "--no-real-usage",
        action="store_true",
        help="补丁 #12 回退开关：不把投影前的真实上下文 token 数作为 usage.input_tokens "
        "上报，改回沿用后端回报（投影后）的值。默认上报真实值。"
        "也可用环境变量 CODEBUDDY2OPENAI_REAL_USAGE=0 关闭。",
    )
    ap.add_argument("--skip-check", action="store_true", help="跳过启动预检")
    args = ap.parse_args()

    CONFIG["api_key"] = args.api_key
    CONFIG["desensitize"] = args.desensitize
    CONFIG["no_compact"] = args.no_compact
    env_real = os.environ.get("CODEBUDDY2OPENAI_REAL_USAGE", "1").strip().lower()
    CONFIG["report_real_usage"] = not args.no_real_usage and env_real not in ("0", "false", "no", "off")
    # --log 直接指定文件路径即开启；不传则不记
    CONFIG["log_path"] = (
        args.log if args.log else os.environ.get("CODEBUDDY2OPENAI_LOG")
    )
    af = find_auth_file()
    CONFIG["cred"] = CredentialManager(af) if af else None

    if not args.skip_check:
        preflight()

    sys.stderr.write(
        f"\n✅ 监听 http://{args.host}:{args.port}（直连后端，原生 function calling）\n"
    )
    sys.stderr.write("   GET  /v1/models\n")
    sys.stderr.write(
        "   POST /v1/chat/completions   (原生 tools/tool_calls，支持流式)\n"
    )
    sys.stderr.write("   POST /v1/responses          (Responses API，Codex CLI 兼容)\n")
    sys.stderr.write(
        "   POST /v1/messages           (Anthropic API，Claude Code / CC Switch 兼容)\n"
    )
    sys.stderr.write("   GET  /health\n")
    if args.api_key:
        sys.stderr.write("   鉴权已启用（API key 已设置）\n")
    if CONFIG["log_path"]:
        sys.stderr.write(f"   日志      : {CONFIG['log_path']}\n")
    if args.desensitize:
        mode = "零宽脱敏 + 保留全文" if args.no_compact else "零宽脱敏 + 压缩摘要"
        sys.stderr.write(f"   脱敏      : 已启用（{mode}）\n")
    sys.stderr.write(
        "   真实用量  : "
        + ("已启用（usage.input_tokens 按投影前真实体积上报）\n"
           if CONFIG["report_real_usage"] else "已关闭（沿用后端回报值）\n")
    )
    sys.stderr.write("按 Ctrl+C 退出。\n\n")

    # 启动时写一条标记
    _log("==== converter 启动 ====")

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
