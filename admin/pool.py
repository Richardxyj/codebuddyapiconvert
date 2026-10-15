"""Account-scoped billing, refresh, daily check-in and request routing."""
import asyncio
import json
import math
import re
import threading
import time
from contextvars import ContextVar
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation

import httpx
from fastapi import HTTPException
from fastapi.responses import JSONResponse

REQUEST_CREDENTIAL = ContextVar("pool_credential", default=None)
CN = timezone(timedelta(hours=8))
BILLING = "https://www.codebuddy.cn/v2/billing/meter/"


def _pool_log(msg: str):
    """把池事件写进 converter.log（复用 converter 的日志通道；失败静默——日志绝不能打断请求）。"""
    try:
        from core import converter
        converter._log(msg)
    except Exception:
        pass


_RESET_DATETIME = re.compile(
    r"(20\d{2})[-/年.](\d{1,2})[-/月.](\d{1,2})[日T\s]*(\d{1,2}):(\d{2})(?::(\d{2}))?"
)
# (?<![\d.])：拒绝小数（"1.5小时"不得截断成"5小时"，对抗审计 P2-5）
_RESET_RELATIVE = re.compile(r"(?<![\d.])(\d+)\s*(秒|分钟|个小时|小时|天)")
# 解析结果边界（对抗审计 P1-1：相对与绝对同限，防"30天"月配额文案把唯一账号冷一月）
_RETRY_AT_MAX_AHEAD = 7 * 86400


def parse_retry_at(text, now):
    """从上游 429/402 错误文本解析重置时刻（epoch 秒）；解析不到返回 None（调用方兜底，不编造）。

    语义与 dsh-connect-workbuddy 的"不编造倒计时"对齐（docs/35 §2.2）：
    - 绝对时间（如"将在 2026-10-15 14:30:00 重置"）按 CN 时区（UTC+8）解释；
    - 相对时间（"N 分钟/小时/天后"）按当前时钟换算；小数（"1.5小时"）不截断、返回 None；
    - 解析结果在过去或超过 7 天视为不可信，返回 None（远期多为解析伪影/月配额文案）。
    """
    if not text:
        return None
    ts = None
    m = _RESET_DATETIME.search(str(text))
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)),
                          int(m.group(4)), int(m.group(5)), int(m.group(6) or 0), tzinfo=CN)
            ts = dt.timestamp()
        except ValueError:
            return None
    else:
        m = _RESET_RELATIVE.search(str(text))
        if not m:
            return None
        try:
            n = int(m.group(1))
        except ValueError:
            return None
        unit = {"秒": 1, "分钟": 60, "小时": 3600, "个小时": 3600, "天": 86400}[m.group(2)]
        ts = now + n * unit
    if now < ts <= now + _RETRY_AT_MAX_AHEAD:
        return ts
    return None


def number(value):
    if isinstance(value, bool):
        raise ValueError("invalid number")
    try:
        n = Decimal(str(value))
    except InvalidOperation:
        raise ValueError("invalid number")
    if not n.is_finite():
        raise ValueError("invalid number")
    return max(Decimal(0), n)


def summarize_packages(packages):
    remain = Decimal(0)
    size = Decimal(0)
    for p in packages:
        # Precise values carry fractional credit balances; cycle values are spendable now.
        prefix = "Cycle" if any(k in p for k in ["CycleCapacityRemainPrecise", "CycleCapacityRemain"]) else ""
        def field(name, default=None):
            return p.get(prefix + name + "Precise", p.get(prefix + name, default))
        r = number(field("CapacityRemain"))
        capacity = number(field("CapacitySize", r))
        remain += min(r, capacity) if capacity else r
        size += capacity
    return {"remaining": float(remain), "capacity": float(size), "used": float(max(Decimal(0), size - remain)), "packages": len(packages)}


class BillingError(Exception):
    def __init__(self, message, status=502, code=None):
        super().__init__(message)
        self.status, self.code = status, code


class AccountPool:
    def __init__(self, store, clock=time.time, client_factory=None):
        self.store, self.clock = store, clock
        self.client_factory = client_factory or (lambda: httpx.Client(timeout=20, follow_redirects=False))
        self.locks = {}
        self.cursor = 0
        self.jobs = asyncio.Lock()
        self.last_sync = 0
        with store.lock:
            store.data.setdefault("pool", {"routing": "round_robin", "auto_checkin": True, "checkin_time": "09:00"})
            store.data.setdefault("account_status", {})
            store.save()

    def operation_lock(self, aid):
        with self.store.lock:
            return self.locks.setdefault(aid, threading.Lock())

    def snapshot(self, aid):
        with self.store.lock:
            item = self.store.data["accounts"].get(aid)
            if not item:
                raise HTTPException(404, "账号不存在")
            return dict(item), dict(self.store.data["account_status"].get(aid, {}))

    def update(self, aid, **values):
        with self.store.lock:
            if aid in self.store.data["accounts"]:
                self.store.data["account_status"].setdefault(aid, {}).update(values)
                self.store.save()

    def state(self, row):
        status = self.store.data["account_status"].get(row["id"], {})
        if not row["enabled"]:
            return "paused"
        if row["status"] == "invalid" or status.get("auth_invalid"):
            return "invalid"
        if status.get("cooldown_until", 0) > self.clock():
            return "cooling"
        if status.get("credits_updated", 0) > self.clock() - 600 and status.get("remaining") == 0 and status.get("packages", 0) > 0:
            return "exhausted"
        return "available"

    def rows(self, rows):
        today = datetime.fromtimestamp(self.clock(), CN).strftime("%Y-%m-%d")
        for row in rows:
            status = self.store.data["account_status"].get(row["id"], {})
            row.update({"pool_state": self.state(row), "remaining": status.get("remaining"), "credits_updated": status.get("credits_updated"),
                        "credits_stale": status.get("credits_updated", 0) < self.clock() - 600,
                        "today_checked_in": status.get("checkin_date") == today, "cooldown_until": status.get("cooldown_until", 0),
                        "last_error": status.get("last_error"), "token_refreshed": status.get("token_refreshed"),
                        "request_count": status.get("request_count", 0)})
        return rows

    def select(self):
        with self.store.lock:
            candidates = [row for row in self.store.account_rows() if self.state(row) == "available"]
            if self.store.data["pool"]["routing"] == "manual":
                candidates = [row for row in candidates if row["id"] == self.store.data["active"]]
            if not candidates:
                raise HTTPException(503, "暂无可用账号：请检查暂停、积分、冷却或登录状态")
            aid = candidates[self.cursor % len(candidates)]["id"]
            self.cursor += 1
            item = self.store.data["accounts"][aid]
            return aid, self.store.manager_for(aid, item)

    def states_summary(self):
        """池内各账号一行摘要，供 503/异常日志使用（R2：拒绝零日志降级）。全程持锁（对抗审计 P3-11）。"""
        with self.store.lock:
            rows = self.store.account_rows()
            return ", ".join(f"{row.get('name') or row['id']}:{self.state(row)}" for row in rows) or "(无账号)"

    def record(self, aid, status=200, retry_at=None):
        with self.store.lock:
            if aid not in self.store.data["accounts"]:
                return
            data = self.store.data["account_status"].setdefault(aid, {})
            data["request_count"] = data.get("request_count", 0) + 1
            data["last_request"] = int(self.clock())
            if status in (401, 403):
                data.update(cooldown_until=self.clock() + 300, last_error="上游认证或访问被拒绝，已冷却 5 分钟")
            elif status in (402, 429):
                # 上游声明的重置时刻优先（P1a：解析响应体，不编造倒计时）；
                # 解析不到才用 30 分钟兜底。文案区分两种来源并带恢复时刻（对抗审计 P2-6 深化）。
                if retry_at is not None and retry_at > self.clock():
                    until = retry_at
                    resume = datetime.fromtimestamp(retry_at, CN).strftime("%m-%d %H:%M")
                    message = f"上游额度或频率限制，已按上游声明重置时间冷却至 {resume}"
                else:
                    until = self.clock() + 1800
                    message = "上游额度或频率限制，上游未声明重置时间，默认冷却 30 分钟"
                data.update(cooldown_until=until, last_error=message)
            self.store.save()

    def billing(self, client, headers, path, body=None):
        response = client.post(BILLING + path, headers=headers, json=body or {})
        if response.status_code != 200:
            raise BillingError("积分服务请求失败", response.status_code)
        try:
            d = response.json()
        except ValueError:
            raise BillingError("积分服务返回格式异常")
        if not isinstance(d, dict) or d.get("code") != 0:
            raise BillingError("上游未接受该操作，请检查账号或稍后重试", code=d.get("code") if isinstance(d, dict) else None)
        return d.get("data")

    def credits(self, client, headers):
        packages = []
        now = datetime.fromtimestamp(self.clock(), CN)
        for page in range(1, 101):
            d = self.billing(client, headers, "get-user-resource", {"PageNumber": page, "PageSize": 100, "ProductCode": "p_tcaca", "Status": [0, 3], "PackageEndTimeRangeBegin": now.strftime("%Y-%m-%d %H:%M:%S"), "PackageEndTimeRangeEnd": "2126-01-01 00:00:00"})
            try:
                data = d["Response"]["Data"]
                current = data["Accounts"]
                total = int(data["TotalCount"])
                if not isinstance(current, list) or total < 0:
                    raise ValueError()
            except (KeyError, TypeError, ValueError):
                raise BillingError("积分数据格式异常，上次余额已保留")
            packages.extend(current)
            if len(packages) >= total:
                return summarize_packages(packages)
            if not current:
                break
        raise BillingError("积分包分页不完整，上次余额已保留")

    def checkin_status(self, client, headers):
        try:
            d = self.billing(client, headers, "checkin-activity-status")
        except BillingError as e:
            if e.status not in (404, 405):
                raise
            d = self.billing(client, headers, "checkin-status")
        if not isinstance(d, dict):
            raise BillingError("签到状态格式异常")
        def flag(snake, camel):
            v = d.get(snake, d.get(camel))
            if v not in (True, False, 0, 1, "true", "false", "1", "0"):
                raise BillingError("签到状态缺少必要字段")
            return v in (True, 1, "true", "1")
        return {"active": flag("active", "active"), "checked": flag("today_checked_in", "todayCheckedIn")}

    def operate(self, aid, action):
        with self.operation_lock(aid):
            item, previous = self.snapshot(aid)
            if not item["enabled"]:
                return {"id": aid, "ok": False, "message": "账号已暂停，未执行操作"}
            manager = self.store.manager_for(aid, item)
            try:
                if action == "refresh":
                    with manager._lock:
                        manager._refresh()
                    self.update(aid, token_refreshed=int(self.clock()), auth_invalid=False, cooldown_until=0, last_error=None)
                    return {"id": aid, "ok": True, "message": "登录凭据已刷新"}
                headers = manager.get_headers()
                if "workbuddy.ai" in str(headers.get("X-Domain", "")).lower():
                    return {"id": aid, "ok": False, "message": "当前仅支持国内账号积分与签到"}
                today = datetime.fromtimestamp(self.clock(), CN).strftime("%Y-%m-%d")
                with self.client_factory() as client:
                    message = "积分状态已更新"
                    checkin_error = None
                    try:
                        ci = self.checkin_status(client, headers)
                        if ci["checked"]:
                            self.update(aid, checkin_date=today)
                        if action == "checkin":
                            if ci["checked"]:
                                message = "今日已签到，无需重复签到"
                            elif not ci["active"]:
                                message = "当前没有可参与的签到活动"
                            else:
                                self.billing(client, headers, "daily-checkin")
                                ci = self.checkin_status(client, headers)
                                if not ci["checked"]:
                                    raise BillingError("签到已提交，但上游尚未确认；请刷新状态")
                                self.update(aid, checkin_date=today)
                                message = "签到成功"
                    except (BillingError, httpx.HTTPError) as e:
                        if action == "checkin":
                            raise
                        checkin_error = "签到状态暂时无法查询"
                    data = self.credits(client, headers)
                    # A successful balance lookup is proof of current authentication.
                    self.update(aid, **data, credits_updated=int(self.clock()), auth_invalid=False, last_error=checkin_error)
                    if data["remaining"] > 0 and action == "checkin":
                        self.update(aid, cooldown_until=0)
                    return {"id": aid, "ok": True, "message": message, "remaining": data["remaining"]}
            except BillingError as e:
                self.update(aid, last_error=str(e))
                if e.status in (401, 403):
                    self.update(aid, cooldown_until=self.clock()+300)
                return {"id": aid, "ok": False, "message": str(e)}
            except (httpx.HTTPError, OSError, ValueError, RuntimeError, KeyError):
                # Do not expose raw refresh responses or bearer credentials.
                message = "操作失败：请检查登录凭据或网络，必要时重新授权"
                self.update(aid, last_error=message)
                return {"id": aid, "ok": False, "message": message}

    async def batch(self, action):
        if self.jobs.locked():
            raise HTTPException(409, "已有批量任务进行中，请稍后刷新")
        async with self.jobs:
            with self.store.lock:
                ids = [aid for aid, item in self.store.data["accounts"].items() if item["enabled"]]
            semaphore = asyncio.Semaphore(3)
            async def run(aid):
                async with semaphore:
                    try:
                        return await asyncio.to_thread(self.operate, aid, action)
                    except HTTPException:
                        return {"id": aid, "ok": False, "message": "账号已移除"}
            return {"results": await asyncio.gather(*(run(aid) for aid in ids))}

    async def tick(self):
        now = datetime.fromtimestamp(self.clock(), CN)
        with self.store.lock:
            cfg = dict(self.store.data["pool"])
            due = [aid for aid, item in self.store.data["accounts"].items() if item["enabled"] and self.store.data["account_status"].get(aid, {}).get("checkin_date") != now.strftime("%Y-%m-%d") and self.store.data["account_status"].get(aid, {}).get("auto_attempt", 0) < self.clock()-1800]
        if cfg["auto_checkin"] and now.strftime("%H:%M") >= cfg["checkin_time"]:
            for aid in due:
                self.update(aid, auto_attempt=int(self.clock()))
                try:
                    await asyncio.to_thread(self.operate, aid, "checkin")
                except HTTPException:
                    pass
        if self.clock() - self.last_sync > 300 and not self.jobs.locked():
            await self.batch("status")
            self.last_sync = self.clock()


class PoolMiddleware:
    def __init__(self, app, pool):
        self.app, self.pool = app, pool

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] != "POST" or scope["path"] not in {"/v1/chat/completions", "/v1/responses", "/v1/messages", "/v1/messages/count_tokens"}:
            return await self.app(scope, receive, send)
        headers = dict(scope.get("headers", []))
        try:
            self.pool.store.check_api(headers.get(b"authorization", b"").decode(), headers.get(b"x-api-key", b"").decode())
            aid, manager = self.pool.select()
            # Refresh before entering a stream; retry another eligible account only
            # if credential preparation fails, never replay a partially emitted response.
            try:
                await asyncio.to_thread(manager.get_headers)
            except Exception:
                self.pool.update(aid, cooldown_until=self.pool.clock()+300, last_error="凭据暂不可用，已冷却 5 分钟")
                aid, manager = self.pool.select()
                await asyncio.to_thread(manager.get_headers)
        except HTTPException as e:
            # P0（docs/35 §3）：池空/无可用账号 503 必须留结构化日志——09-30 事故根因就是零日志。
            _pool_log(f"✗ POOL {e.status_code} {scope.get('path')}：{e.detail} | 账号状态: {self.pool.states_summary()}")
            return await JSONResponse({"detail": e.detail}, e.status_code)(scope, receive, send)
        except Exception:
            # 广义失败（凭据准备崩溃等）同样不能零日志
            _pool_log(f"✗ POOL 500 {scope.get('path')}：账号凭据异常 | 账号状态: {self.pool.states_summary()}")
            return await JSONResponse({"detail": "账号凭据暂不可用，请刷新凭据或重新授权"}, 503)(scope, receive, send)
        token = REQUEST_CREDENTIAL.set(manager)
        status = 200
        reset_hint = None
        buffer = b""
        async def observe(message):
            nonlocal status, buffer, reset_hint
            if message["type"] == "http.response.start":
                status = message["status"]
            if message["type"] == "http.response.body":
                buffer += message.get("body", b"")
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line.startswith(b"data:"):
                        try:
                            obj = json.loads(line[5:])
                            err = obj.get("error") or {}
                            code = err.get("code") if isinstance(err, dict) else None
                            if code in (401, 402, 403, 429):
                                status = code
                                if code in (402, 429):
                                    # P1a（docs/35 §2.2）：从错误文本提取上游声明的重置时刻
                                    text = err.get("message") or err.get("msg") or obj.get("msg")
                                    hint = parse_retry_at(text, self.pool.clock())
                                    if hint is not None:
                                        reset_hint = max(reset_hint or 0, hint)
                        except (ValueError, AttributeError):
                            pass
                if len(buffer) > 65536:
                    buffer = b""
            await send(message)
        try:
            await self.app(scope, receive, observe)
        finally:
            REQUEST_CREDENTIAL.reset(token)
            self.pool.record(aid, status, retry_at=reset_hint)
