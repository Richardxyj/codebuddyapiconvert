"""本机配置探针 —— 档 B「本地控制中心」只读数据层（docs/33 §3）。

红线（docs/33 §2）：
  C3  本模块全部只读 —— 不修改 config.toml / catalog / auth / 日志，
      只读取并聚合，健康判定复用既有守卫口径（service_run._catalog_is_healthy）。
  C4  零新增依赖 —— 仅用标准库（tomllib 为 py3.11+ 内置）。

所有函数绝不抛出：本机文件缺失/损坏时返回带 ok=False 与 reason 的字典，
让前端呈现"未知/待修"而非 500（R1：不许静默，必须显式标注）。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # ~/codebuddy2api
CODEX_HOME = Path.home() / ".codex"
CONFIG_TOML = CODEX_HOME / "config.toml"
CATALOG = CODEX_HOME / "model-catalogs" / "relay-mu9j3868.json"
CONVERTER_LOG = ROOT / "converter.log"

# 与 service_run.py 守卫 #21 完全一致的安全几何（docs/13）
EXPECTED_CTX = 256000
EXPECTED_COMPACT = 204800
MIN_MODELS = 10

# config.toml 关键键（docs/33 §3.1）：缺失会直接影响链路能力
REQUIRED_TOP_KEYS = ["model", "model_provider", "model_catalog_json"]


def _ok(**kw):
    out = {"ok": True}
    out.update(kw)
    return out


def _bad(reason, **kw):
    out = {"ok": False, "reason": reason}
    out.update(kw)
    return out


def api_key_status() -> dict:
    """客户端 API Key 是否已设置（绝不回显值本身）。"""
    present = bool(os.environ.get("CODEBUDDY2OPENAI_KEY"))
    return _ok(present=present)


def config_status() -> dict:
    """config.toml 关键键完整性（只读解析）。"""
    if not CONFIG_TOML.is_file():
        return _bad("config.toml 不存在", path=str(CONFIG_TOML), missing=REQUIRED_TOP_KEYS)
    try:
        data = tomllib.loads(CONFIG_TOML.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as e:
        return _bad(f"config.toml 解析失败: {e}", path=str(CONFIG_TOML))
    missing = [k for k in REQUIRED_TOP_KEYS if k not in data]
    provider = data.get("model_providers", {}).get("workbuddy", {})
    detail = {
        "model": data.get("model"),
        "model_provider": data.get("model_provider"),
        "model_catalog_json": data.get("model_catalog_json"),
        "provider_base_url": provider.get("base_url"),
        "wire_api": provider.get("wire_api"),
        "stream_idle_timeout_ms": provider.get("stream_idle_timeout_ms"),
        "stream_max_retries": provider.get("stream_max_retries"),
        "multi_agent_v2": data.get("features", {}).get("multi_agent_v2"),
    }
    # 关键派生键：provider 表与流式耐心参数缺失同样视为不健康
    if not provider:
        missing.append("model_providers.workbuddy")
    for k in ("stream_idle_timeout_ms", "stream_max_retries"):
        if provider.get(k) is None:
            missing.append(k)
    if data.get("features", {}).get("multi_agent_v2") is not True:
        missing.append("features.multi_agent_v2")
    if missing:
        return _bad("缺少关键键", path=str(CONFIG_TOML), missing=missing, detail=detail)
    return _ok(path=str(CONFIG_TOML), missing=[], detail=detail)


def _catalog_geometry(models: list) -> bool:
    return bool(models) and all(
        m.get("context_window") == EXPECTED_CTX
        and m.get("auto_compact_token_limit") == EXPECTED_COMPACT
        for m in models
    )


def catalog_status() -> dict:
    """catalog 条目数与几何（口径 = 守卫 #21 的 _catalog_is_healthy）。"""
    if not CATALOG.is_file():
        return _bad("catalog 不存在", path=str(CATALOG), count=0)
    try:
        models = json.loads(CATALOG.read_text(encoding="utf-8")).get("models") or []
    except (OSError, ValueError) as e:
        return _bad(f"catalog 解析失败: {e}", path=str(CATALOG), count=0)
    healthy = len(models) >= MIN_MODELS and _catalog_geometry(models)
    detail = {
        "count": len(models),
        "geometry_ok": _catalog_geometry(models),
        "expected_ctx": EXPECTED_CTX,
        "expected_compact": EXPECTED_COMPACT,
        "slugs": [m.get("slug") for m in models],
    }
    if not healthy:
        return _bad("条目数不足或几何异常", path=str(CATALOG), **detail)
    return _ok(path=str(CATALOG), **detail)


def port_status(port: int = 8787) -> dict:
    """8787 监听状态与 PID（best-effort，拿不到 PID 不视为失败）。"""
    listening = False
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(("127.0.0.1", port))
        listening = True
    except OSError:
        listening = False
    finally:
        s.close()
    pid = None
    if listening:
        try:
            out = subprocess.run(
                ["netstat", "-ano"], capture_output=True, text=True, timeout=10
            ).stdout
            for line in out.splitlines():
                if f"127.0.0.1:{port}" in line and "LISTENING" in line:
                    parts = line.split()
                    pid = int(parts[-1])
                    break
        except (OSError, subprocess.SubprocessError, ValueError, IndexError):
            pid = None
    if not listening:
        return _bad("8787 未监听", port=port)
    return _ok(port=port, pid=pid)


def task_status(name: str = "Codebuddy2Api") -> dict:
    """计划任务存在性与最近结果（PowerShell cmdlet，只读查询）。"""
    ps = (
        "$t = Get-ScheduledTask -TaskName '%s' -ErrorAction SilentlyContinue; "
        "if (-not $t) { Write-Output 'MISSING'; exit }; "
        "$i = $t | Get-ScheduledTaskInfo; "
        "Write-Output ($t.State.ToString() + '|' + $i.LastTaskResult + '|' + $i.LastRunTime)"
        % name
    )
    try:
        r = subprocess.run(
            ["powershell", "-NoProfile", "-Command", ps],
            capture_output=True, text=True, timeout=20,
        )
        text = (r.stdout or "").strip()
    except (OSError, subprocess.SubprocessError) as e:
        return _bad(f"查询失败: {e}", task=name)
    if text == "MISSING" or not text:
        return _bad("计划任务不存在", task=name)
    parts = text.split("|")
    return _ok(
        task=name,
        state=parts[0] if len(parts) > 0 else "unknown",
        last_result=parts[1] if len(parts) > 1 else "unknown",
        last_run=parts[2] if len(parts) > 2 else "unknown",
    )


def local_overview() -> dict:
    """配置总览页聚合（§3.1）。各项独立判定，单项失败不拖垮整体。"""
    sections = {
        "api_key": api_key_status(),
        "config": config_status(),
        "catalog": catalog_status(),
        "port": port_status(),
        "task": task_status(),
    }
    all_ok = all(v.get("ok") for v in sections.values())
    return {"ok": all_ok, "sections": sections}


# 错误高亮关键字（§3.3）
ERROR_MARK = "✗ HTTP"
PROJECTION_MARK = "PROJECTION"


def read_log_tail(lines: int = 120, max_bytes: int = 262144) -> dict:
    """converter.log 尾部读取（§3.3）：只 seek 尾部，绝不整文件载入。

    返回带 annotated 标记的行，便于前端高亮 ✗ HTTP 与 PROJECTION。
    """
    if not CONVERTER_LOG.is_file():
        return _bad("converter.log 不存在", path=str(CONVERTER_LOG), lines=[])
    try:
        size = CONVERTER_LOG.stat().st_size
        with open(CONVERTER_LOG, "rb") as f:
            f.seek(max(0, size - max_bytes))
            raw = f.read()
        text = raw.decode("utf-8", errors="replace")
        # 丢弃首行可能因 seek 落在多字节字符中间造成的半行
        if size > max_bytes and "\n" in text:
            text = text.split("\n", 1)[1]
        rows = text.splitlines()[-lines:]
    except OSError as e:
        return _bad(f"读取失败: {e}", path=str(CONVERTER_LOG), lines=[])
    annotated = [
        {
            "text": ln,
            "kind": (
                "error" if ERROR_MARK in ln
                else "projection" if PROJECTION_MARK in ln
                else "normal"
            ),
        }
        for ln in rows
    ]
    return _ok(
        path=str(CONVERTER_LOG), size=size, lines=annotated,
        truncated=size > max_bytes,
    )
