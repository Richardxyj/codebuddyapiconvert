"""service_run.py — 转换器后台服务入口（计划任务直接调用，无 PowerShell / 无窗口）

职责：
1. config 守卫：Codex App 启动/更新会重写 ~/.codex/config.toml 并抹掉第三方键
   （model_catalog_json 已两次被抹）——每分钟 tick 检查，缺失则原子补回（补丁 #20；
   S1 起（docs/35 §6.2，2026-10-15）补回内容为 managed 块，修复动作记 guard.log）
2. catalog 守卫：Codex++ 的 catalog 生成会把 relay-mu9j3868.json 覆盖成单模型
   （kimi-k3-1, ctx 272000/compact 244800）——每分钟 tick 校验条目数与几何，
   异常则从 model-catalog-canonical.json 原子恢复（补丁 #21）
3. auth 守卫：auth/<账号>.info 被外删（10-08 实证 auth 目录整体消失、服务靠内存
   凭据撑到重启即断）而 C:\\data\\auth 权威副本仍在时，从副本恢复（补丁 #22）
4. 端口守卫：8787 已被占用则静默退出（任务下个周期再试），避免绑定崩溃循环
5. 输出接管：pythonw 无控制台（sys.stdout/stderr 为 None，print 会崩），
   重定向到 converter-stdout.log（横幅 / uvicorn 日志 / traceback 全落盘）
6. 进程内启动复合应用（档 B「本地控制中心」，docs/33）：import core.converter 取模块级
   FastAPI app（不触发其 main()），经 admin.server.create_app() 把 converter 挂到根路径、
   admin 管理后台挂到 /admin/，再以 uvicorn 单进程运行。本机模式 secure_cookie=False、
   绑定 127.0.0.1；create_app 会覆盖 converter.CONFIG 的 log_path，须在之后补回。

生命周期 CLI（S1，docs/35 §6.2）：
    python service_run.py              # 无参（计划任务）：守卫 + 服务，行为不变
    python service_run.py setup        # 幂等安装：备份 config → 修复 → 报告
    python service_run.py doctor       # 体检（只读报告，exit 0=全部 OK / 1=有 FAIL）
    python service_run.py doctor --fix # 体检 + 修复（修复动作记 guard.log）
    python service_run.py uninstall    # 移除 config 中的 managed 块（不动非我方写入的键）

任务动作："<venv>\\Scripts\\pythonw.exe" "C:\\Users\\richard.xu\\codebuddy2api\\service_run.py"
工作目录：C:\\Users\\richard.xu\\codebuddy2api
触发器：登录 + 每 1 分钟周期（守卫使其健康时空转）＝ 自愈看门狗
"""
import json
import os
import shutil
import socket
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_LOG = os.path.join(ROOT, "converter-stdout.log")
# S1：守卫修复动作日志（R2 禁无日志降级——谁、何时、为什么改了 config 必须可审计）
GUARD_LOG = os.path.join(ROOT, "guard.log")
# S1：managed 块标记——setup/doctor 只认自己的块；用户手写的同名键不误判、不误删
MANAGED_BEGIN = "# >>> codebuddy2api managed >>>"
MANAGED_END = "# <<< codebuddy2api managed <<<"
# S1：卸载戳记（对抗审计 P1-2）——uninstall 落此文件后，看门狗的 config 键守卫
# 停止补回（否则每分钟 tick 会把 managed 块自动重装回去，卸载形同虚设）；
# setup 会清除该戳记。doctor 会提示戳记在位但计划任务仍在运行。
UNINSTALLED_STAMP = os.path.join(ROOT, "uninstalled.stamp")
CONFIG_RELPATH = r"~\.codex\config.toml"
# catalog 恢复源（canonical）：由 sync_model_catalog.py --write 成功后自动刷新
CANONICAL_CATALOG = os.path.join(ROOT, "model-catalog-canonical.json")
CATALOG_RELPATH = r"~\.codex\model-catalogs\relay-mu9j3868.json"
# 与转换器 #13 投影预算 243,200 匹配的安全几何（docs/13）；
# 272K 档会让 Codex 以为窗口更大、原生压缩永不触发
EXPECTED_CTX = 256000
EXPECTED_COMPACT = 204800
# 健康下限：正常同步后 16 个模型；上游列表波动也远高于此
MIN_MODELS = 10
# auth 守卫（#22）：auth/ 是转换器运行凭据目录，C:\data\auth 是权威副本（docs/33）
AUTH_DIR = os.path.join(ROOT, "auth")
AUTH_MASTER_DIR = r"C:\data\auth"
MANAGEMENT_STATE = r"C:\data\management\state.json"


def _guard_log(msg: str):
    """守卫修复动作日志（追加一行，带时间戳；失败静默——日志绝不能打断守卫）。"""
    try:
        with open(GUARD_LOG, "a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {msg}\n")
    except OSError:
        pass


def run_guards() -> list[str]:
    """跑三个守卫（#20/#21/#22）；返回修复项列表，修复动作记 guard.log（S1，R2）。"""
    fixes = []
    try:
        if ensure_catalog_key():
            fixes.append("config")
            _guard_log("FIX config: model_catalog_json 缺失或为裸键，已写入/规范化 managed 块")
    except Exception as e:
        _guard_log(f"ERR  config 守卫异常：{type(e).__name__}")
    try:
        if ensure_catalog_models():
            fixes.append("catalog")
            _guard_log("FIX catalog: relay-mu9j3868.json 异常（单模型/几何污染/损坏），已从 canonical 原子恢复")
    except Exception as e:
        _guard_log(f"ERR  catalog 守卫异常：{type(e).__name__}")
    try:
        if ensure_auth_file():
            fixes.append("auth")
            _guard_log("FIX auth: auth/ 凭据缺失，已从 C:\\data\\auth 权威副本恢复")
    except Exception as e:
        _guard_log(f"ERR  auth 守卫异常：{type(e).__name__}")
    return fixes


def ensure_catalog_key() -> bool:
    """补丁 #20（2026-09-29）+ S1（2026-10-15）：config.toml 的 model_catalog_json 键守卫。

    Codex App 每次启动/更新重写 config.toml 时只保留自家键，第三方键
    `model_catalog_json`（桌面选择器读真名模型的唯一入口，docs/04）被抹后
    选择器回落 OpenAI 内置名。看门狗每分钟 tick 执行本函数。

    三种健康路径 + 一种收敛路径（对抗审计 P2-9 深化 + 15:10 生产残缺块实证）：
    1. 完整 managed 块在位（BEGIN+键+END）→ 健康，不动；
    2. 裸键在位且值精确匹配我方 catalog → **规范化**：原键行包进 BEGIN/END；
    3. 无键无标记 → 在首个表头前插入 managed 块；
    4. **残缺块**（标记在、键不在——多写者竞态/Codex++ 重写的真实产物）→
       先剔除所有孤立标记行，再按 2/3 收敛——绝不允许残缺形态存活。

    值不匹配的裸键（用户指向别的 catalog）不动——那不是我方的键。
    卸载戳记在位时直接跳过（uninstall 语义）。
    任何失败静默返回 False——本函数在 stdout 接管前运行，绝不可 print（pythonw 会崩）。
    """
    key_line = 'model_catalog_json = "model-catalogs/relay-mu9j3868.json"'
    cfg = os.path.expanduser(CONFIG_RELPATH)
    try:
        # 卸载戳记在位：用户已 uninstall，看门狗不得把键补回（对抗审计 P1-2）
        if os.path.isfile(UNINSTALLED_STAMP):
            return False
        with open(cfg, encoding="utf-8") as f:
            text = f.read()
        has_key = key_line in text
        has_begin = MANAGED_BEGIN in text
        if has_key and has_begin:
            return False  # 完整 managed 形态
        lines = text.splitlines(keepends=True)
        # 收敛预处理：剔除所有孤立的标记行（残缺块），无论后续走规范化还是插入
        if has_begin or MANAGED_END in text:
            lines = [ln for ln in lines
                     if ln.strip() not in (MANAGED_BEGIN, MANAGED_END)]
        # 规范化：裸键且值精确匹配 → 包进 managed 块（原地，位置不变）
        foreign_key = False
        for i, ln in enumerate(lines):
            stripped = ln.strip()
            if stripped == key_line:
                lines[i:i + 1] = [MANAGED_BEGIN + "\n", key_line + "\n", MANAGED_END + "\n"]
                tmp = cfg + ".guard.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    f.write("".join(lines))
                os.replace(tmp, cfg)
                return True
            if stripped.startswith("model_catalog_json") and "=" in stripped:
                foreign_key = True  # 用户指向别的 catalog——不碰、不插（避免 TOML 重复键）
        if foreign_key:
            return False
        # 无键：TOML 顶层键必须位于第一个表头之前；无表头则追加到末尾
        insert_at = next(
            (i for i, ln in enumerate(lines) if ln.lstrip().startswith("[")),
            len(lines),
        )
        lines[insert_at:insert_at] = [MANAGED_BEGIN + "\n", key_line + "\n", MANAGED_END + "\n", "\n"]
        tmp = cfg + ".guard.tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write("".join(lines))
        os.replace(tmp, cfg)
        return True
    except OSError:
        return False


def _catalog_is_healthy(path: str) -> bool:
    """catalog 健康判定：可解析、条目数达下限、全条目几何等于安全值。"""
    try:
        with open(path, encoding="utf-8") as f:
            models = json.load(f).get("models") or []
    except (OSError, ValueError):
        return False
    if len(models) < MIN_MODELS:
        return False
    return all(
        m.get("context_window") == EXPECTED_CTX
        and m.get("auto_compact_token_limit") == EXPECTED_COMPACT
        for m in models
    )


def ensure_catalog_models() -> bool:
    """补丁 #21（2026-09-29）：model-catalogs/relay-mu9j3868.json 内容守卫。

    Codex++ 的 catalog 生成功能会把本文件覆盖成单模型（kimi-k3-1，
    ctx 272000/compact 244800）——选择器只剩 1 个模型，且 244800 大于转换器
    #13 投影预算 243200，原生压缩永不触发。看门狗每分钟 tick 校验：
    条目数 < MIN_MODELS 或几何异常 → 从 CANONICAL_CATALOG 原子恢复。
    canonical 自身缺失/损坏时绝不动作（坏恢复源不得覆盖好文件）。
    任何失败静默返回 False——本函数在 stdout 接管前运行，绝不可 print。
    """
    catalog = os.path.expanduser(CATALOG_RELPATH)
    try:
        if _catalog_is_healthy(catalog):
            return False
        if not _catalog_is_healthy(CANONICAL_CATALOG):
            return False
        tmp = catalog + ".guard.tmp"
        with open(CANONICAL_CATALOG, "rb") as src, open(tmp, "wb") as dst:
            dst.write(src.read())
        os.replace(tmp, catalog)
        return True
    except OSError:
        return False


def ensure_auth_file() -> bool:
    """补丁 #22（2026-10-08）：auth 凭据文件守卫。

    10-08 实证 auth/ 目录被外删（state.json accounts 仍指向 0-converter-plugin.info），
    服务靠内存凭据维持、重启即断。本函数读 MANAGEMENT_STATE 的 accounts[].file，
    对 auth/ 中缺失而 AUTH_MASTER_DIR 权威副本仍存在的凭据逐个恢复。
    只恢复文件、不改 state.json / admin 逻辑；state 缺失/损坏时静默不动作。
    任何失败静默返回 False——本函数在 stdout 接管前运行，绝不可 print。
    """
    try:
        with open(MANAGEMENT_STATE, encoding="utf-8") as f:
            accounts = json.load(f).get("accounts") or {}
        restored = False
        for acct in accounts.values():
            name = (acct or {}).get("file") or ""
            # 仅允许纯文件名（防 state 被篡改后路径穿越）
            if not name or "/" in name or "\\" in name or ".." in name:
                continue
            dst = os.path.join(AUTH_DIR, name)
            if os.path.exists(dst):
                continue
            src = os.path.join(AUTH_MASTER_DIR, name)
            if not os.path.isfile(src):
                continue
            os.makedirs(AUTH_DIR, exist_ok=True)
            tmp = dst + ".guard.tmp"
            with open(src, "rb") as s, open(tmp, "wb") as d:
                d.write(s.read())
            os.replace(tmp, dst)
            restored = True
        return restored
    except (OSError, ValueError):
        return False


def port_busy(port: int = 8787) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    try:
        s.bind(("127.0.0.1", port))
        return False
    except OSError:
        return True
    finally:
        s.close()


# ---------------------------------------------------------------------------
# S1：生命周期 CLI（docs/35 §6.2）—— setup / doctor / uninstall
# ---------------------------------------------------------------------------


def doctor(fix: bool = False) -> int:
    """体检报告（只读）；fix=True 时顺手跑守卫修复（记 guard.log）。

    返回 0=全部 OK，1=存在 FAIL。供人工排障与 CI 用；打印到 stdout
    （仅交互 CLI 调用，计划任务的无参路径不会走到这里）。
    """
    problems = 0

    def report(ok: bool, name: str, detail: str, fix_note: str = ""):
        nonlocal problems
        if ok:
            print(f"  OK   {name}: {detail}")
        else:
            problems += 1
            suffix = f"（--fix 可修复：{fix_note}）" if fix_note else ""
            print(f"  FAIL {name}: {detail}{suffix}")

    print("■ codebuddy2api 转换器体检")
    # 0. 卸载戳记（对抗审计 P1-2）：在位时 config 键守卫停摆，这是卸载语义而非故障
    if os.path.isfile(UNINSTALLED_STAMP):
        print("  WARN 已卸载（uninstalled.stamp 在位）：config 键守卫停摆。")
        print("        计划任务 Codebuddy2Api 仍在运行（服务/其他守卫不受影响）——")
        print("        如需彻底停用请 Disable-ScheduledTask -TaskName Codebuddy2Api")
    # 1. config.toml：存在 / TOML 可解析 / catalog 键在位
    cfg = os.path.expanduser(CONFIG_RELPATH)
    try:
        with open(cfg, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        report(False, "config.toml", "文件不存在（Codex 未初始化或路径异常）", "无法自动创建")
    else:
        try:
            import tomllib
            tomllib.loads(text)
            report(True, "config.toml 语法", "TOML 可解析")
        except Exception as e:
            report(False, "config.toml 语法", f"解析失败：{e}", "手工修复（守卫不覆盖语法损坏）")
        has_key = "model_catalog_json" in text
        has_block = MANAGED_BEGIN in text
        if os.path.isfile(UNINSTALLED_STAMP):
            # 已卸载：键在位与否都算预期状态，不报 FAIL
            detail = "已卸载（键已随 managed 块移除）" if not has_key else "已卸载（键仍在：手写或 App 洗脱残留，不自动删）"
            report(True, "model_catalog_json", detail)
        elif has_key:
            form = "managed 块" if has_block else "裸键（值匹配，下个守卫周期自动规范化为 managed 块）"
            report(True, "model_catalog_json", f"在位（{form}）")
        else:
            report(False, "model_catalog_json", "缺失（Codex App 重写会抹掉，docs/04）", "补回 managed 块")
    # 2. relay catalog 与 canonical
    catalog = os.path.expanduser(CATALOG_RELPATH)
    report(_catalog_is_healthy(catalog), "relay catalog", catalog, "从 canonical 原子恢复")
    report(_catalog_is_healthy(CANONICAL_CATALOG), "canonical catalog", CANONICAL_CATALOG, "重跑 scripts/sync_model_catalog.py --write")
    # 3. auth 凭据
    try:
        with open(MANAGEMENT_STATE, encoding="utf-8") as f:
            accounts = json.load(f).get("accounts") or {}
        missing = [a.get("file") for a in accounts.values()
                   if a.get("file") and not os.path.isfile(os.path.join(AUTH_DIR, a["file"]))]
        report(not missing, "auth 凭据", f"{len(accounts)} 账号，缺失 {missing or '无'}", "从 C:\\data\\auth 权威副本恢复")
    except (OSError, ValueError):
        report(False, "auth 凭据", f"无法读取管理状态 {MANAGEMENT_STATE}", "检查 state.json")
    # 4. 服务端口
    print(f"  INFO 端口 8787: {'被占用（服务运行中）' if port_busy() else '空闲（服务未运行；计划任务下个周期拉起）'}")
    if fix:
        fixes = run_guards()
        print(f"  FIX  守卫修复: {fixes or '无需修复'}（详情见 guard.log）")
    print(f"■ 结论: {'全部 OK' if problems == 0 else f'{problems} 项 FAIL'}")
    return 0 if problems == 0 else 1


def setup() -> int:
    """幂等安装：清除卸载戳记 → 备份 config.toml → doctor --fix → 报告（S1，docs/35 §6.2）。"""
    print("■ codebuddy2api 安装（幂等，可重复执行）")
    if os.path.isfile(UNINSTALLED_STAMP):
        try:
            os.remove(UNINSTALLED_STAMP)
            _guard_log("SETUP: 已清除卸载戳记，config 键守卫恢复")
            print("  OK   已清除卸载戳记（uninstalled.stamp），守卫恢复")
        except OSError:
            print("  WARN 卸载戳记删除失败，config 键守卫仍停摆")
    cfg = os.path.expanduser(CONFIG_RELPATH)
    if os.path.isfile(cfg):
        backup_dir = os.path.join(ROOT, "backup-config")
        os.makedirs(backup_dir, exist_ok=True)
        stamp = time.strftime("%Y%m%d-%H%M%S")
        target = os.path.join(backup_dir, f"config.toml.{stamp}")
        shutil.copy2(cfg, target)
        print(f"  OK   已备份 config.toml → {target}")
        # 只保留最近 10 份备份
        backups = sorted(b for b in os.listdir(backup_dir) if b.startswith("config.toml."))
        for old in backups[:-10]:
            try:
                os.remove(os.path.join(backup_dir, old))
            except OSError:
                pass
    else:
        print("  WARN config.toml 不存在，跳过备份（Codex 未初始化？）")
    return doctor(fix=True)


# 生命周期 CLI 管理的计划任务名（uninstall --full 用）
TASK_NAME = "Codebuddy2Api"


def _disable_task() -> str:
    """uninstall --full：禁用计划任务（schtasks，原生可用）。返回报告行。"""
    q = subprocess.run(["schtasks", "/Query", "/TN", TASK_NAME],
                       capture_output=True, timeout=15)
    if q.returncode != 0:
        return f"  INFO 计划任务 {TASK_NAME} 不存在（未注册或已移除）"
    r = subprocess.run(["schtasks", "/Change", "/TN", TASK_NAME, "/Disable"],
                       capture_output=True, timeout=15)
    if r.returncode == 0:
        _guard_log(f"UNINSTALL --full: 已禁用计划任务 {TASK_NAME}")
        return f"  OK   已禁用计划任务 {TASK_NAME}（重新启用：setup，或 schtasks /Change /TN {TASK_NAME} /Enable）"
    _guard_log(f"ERR  UNINSTALL --full: 禁用计划任务失败（exit {r.returncode}）")
    return f"  FAIL 禁用计划任务失败（exit {r.returncode}）；请手工 schtasks /Change /TN {TASK_NAME} /Disable"


def uninstall(full: bool = False) -> int:
    """卸载：只移除 config.toml 中我方的 managed 块；不动任何非我方写入的键（S1）。

    full=True（uninstall --full）：除 managed 块外，同时禁用计划任务
    Codebuddy2Api——否则戳记虽停了 config 键守卫，服务进程仍会被每分钟
    tick 拉起，「卸载」名不副实。
    """
    print(f"■ codebuddy2api 卸载（{'移除 managed 块 + 禁用计划任务' if full else '仅移除 managed 块'}；备份保留在 backup-config/）")
    cfg = os.path.expanduser(CONFIG_RELPATH)
    try:
        with open(cfg, encoding="utf-8") as f:
            text = f.read()
    except OSError:
        print("  INFO config.toml 不存在，无事可做")
        return 0
    if MANAGED_BEGIN not in text:
        print("  INFO 无 managed 块（从未 setup，或为旧版手写键）")
        if "model_catalog_json" in text:
            print("  WARN 发现非 managed 的 model_catalog_json 键，未改动（不删除非我方写入的内容）")
        # 即使无块也落戳记：阻止看门狗后续补装（对抗审计 P1-2）
        try:
            with open(UNINSTALLED_STAMP, "w", encoding="utf-8") as f:
                f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " uninstalled (no managed block)\n")
            print("  OK   已落卸载戳记（config 键守卫停摆，setup 可恢复）")
        except OSError:
            print("  WARN 卸载戳记写入失败，config 键守卫仍会补装缺失的键")
        if full:
            print(_disable_task())
        return 0
    lines = text.splitlines(keepends=True)
    out, skipping, removed = [], False, 0
    for ln in lines:
        stripped = ln.strip()
        if stripped == MANAGED_BEGIN:
            skipping = True
            removed += 1
            continue
        if stripped == MANAGED_END:
            skipping = False
            removed += 1
            continue
        if not skipping:
            out.append(ln)
    if skipping:
        # 残缺 managed 块（有 BEGIN 无 END，文件截断/部分重写）：宁可不写也不吞内容（对抗审计 P2-10）
        _guard_log("ERR  uninstall: config.toml 存在残缺 managed 块（缺 END 标记），已放弃写入")
        print("  FAIL 残缺 managed 块（缺 END 标记），已放弃写入以保护 config.toml；请手工检查")
        return 1
    tmp = cfg + ".uninstall.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write("".join(out))
    os.replace(tmp, cfg)
    # 落卸载戳记：阻止看门狗在 1 分钟内把 managed 块补回（对抗审计 P1-2）
    try:
        with open(UNINSTALLED_STAMP, "w", encoding="utf-8") as f:
            f.write(time.strftime("%Y-%m-%d %H:%M:%S") + " uninstalled\n")
        stamp_note = "；已落卸载戳记（守卫停摆，setup 可恢复）"
    except OSError:
        stamp_note = "；⚠ 卸载戳记写入失败——看门狗将在 1 分钟内补回 managed 块！"
    _guard_log(f"UNINSTALL: 已移除 config.toml 中的 managed 块（{removed} 行）{stamp_note}")
    print(f"  OK   已移除 managed 块（{removed} 行）{stamp_note}")
    if full:
        print(_disable_task())
        print("  INFO 当前服务进程仍在运行；下个周期计划任务已禁用，不会再拉起。")
    else:
        print(f"  INFO 计划任务 {TASK_NAME} 未改动（服务仍可用）；彻底停用：uninstall --full")
    return 0


def main() -> None:
    # config/catalog/auth 守卫必须在最前：每分钟 tick 都执行（与端口状态无关），任何异常不阻断启动
    # S1：经 run_guards() 执行——修复动作记 guard.log（R2），行为与 #20/#21/#22 等价
    run_guards()

    if port_busy():
        # 已有实例在服务（或端口被外部占用）——静默退出，等下个周期
        return

    # pythonw：stdout/stderr 为 None，必须接管，否则任何 print/traceback 直接崩
    try:
        stream = open(OUT_LOG, "a", buffering=1, encoding="utf-8", errors="replace")
        sys.stdout = stream
        sys.stderr = stream
    except OSError:
        stream = None  # 日志文件不可写也继续跑（转换器自身写 converter.log）

    # 管理密钥：本机走用户环境变量 CODEBUDDY_ADMIN_KEY（docs/33 §2 D3），映射到 admin 读取的
    # ADMIN_KEY。绝不打印其值（AGENTS.md 安全红线）。未设置则 create_app 会抛错——静默退出，
    # 等下个周期（避免无管理密钥时把后台开成无鉴权）。
    if os.environ.get("CODEBUDDY_ADMIN_KEY") and not os.environ.get("ADMIN_KEY"):
        os.environ["ADMIN_KEY"] = os.environ["CODEBUDDY_ADMIN_KEY"]

    # 让 converter 的模块级配置与命令行等价（--desensitize --log converter.log）。
    # create_app() 内部会 converter.CONFIG.update({desensitize:True, no_compact:False, log_path:None})，
    # 故在 create_app 之后再补回 log_path / 真实用量等运行参数（顺序敏感）。
    from core import converter
    from admin.server import create_app

    try:
        app = create_app(secure_cookie=False)
    except Exception:
        # 管理密钥缺失/过短等启动失败：静默退出等下周期，不影响端口守卫语义
        return

    # 补回本机运行参数（create_app 已置 log_path=None；desensitize 已为 True）
    converter.CONFIG["log_path"] = os.path.join(ROOT, "converter.log")
    converter.CONFIG["desensitize"] = True
    converter.CONFIG["no_compact"] = False
    env_real = os.environ.get("CODEBUDDY2OPENAI_REAL_USAGE", "1").strip().lower()
    converter.CONFIG["report_real_usage"] = env_real not in ("0", "false", "no", "off")
    # 凭据：converter 单账号回退（无 pool 选中时），与管理面多凭据并存
    if converter.CONFIG.get("cred") is None:
        af = converter.find_auth_file()
        converter.CONFIG["cred"] = converter.CredentialManager(af) if af else None

    sys.stderr.write("\n✅ 监听 http://127.0.0.1:8787（转换器 + 本地控制中心 /admin/）\n")
    sys.stderr.write("   管理后台  : http://127.0.0.1:8787/admin/\n")
    sys.stderr.write(f"   日志      : {converter.CONFIG['log_path']}\n")
    converter._log("==== converter + admin 启动（档 B 本地控制中心） ====")

    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8787, log_level="warning")


if __name__ == "__main__":
    # S1 生命周期 CLI（docs/35 §6.2）：无参 = 计划任务语义（守卫 + 服务），行为不变
    _args = sys.argv[1:]
    if _args and (sys.stdout is None or sys.stderr is None):
        # pythonw 跑子命令（无控制台）：输出接管到 OUT_LOG，否则 doctor/setup/uninstall
        # 的报告静默丢失（print 在 stdout=None 时不崩但全丢，对抗审计 P3-f 深化）
        try:
            _cli_stream = open(OUT_LOG, "a", buffering=1, encoding="utf-8", errors="replace")
            if sys.stdout is None:
                sys.stdout = _cli_stream
            if sys.stderr is None:
                sys.stderr = _cli_stream
        except OSError:
            pass
    if not _args:
        main()
    elif _args[0] == "setup":
        sys.exit(setup())
    elif _args[0] == "doctor":
        sys.exit(doctor(fix="--fix" in _args[1:]))
    elif _args[0] == "uninstall":
        sys.exit(uninstall(full="--full" in _args[1:]))
    else:
        print("用法: python service_run.py [setup | doctor [--fix] | uninstall [--full]]", file=sys.stderr)
        sys.exit(2)
