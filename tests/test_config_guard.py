"""service_run 守卫测试（#20 config 键 + #21 catalog 内容，2026-09-29）。

#20 覆盖：缺键补回（插在首个表头前 / 无表头追加末尾）、已键不动、文件缺失静默、
原子替换不留临时文件。
#21 覆盖：catalog 被截断/几何污染后从 canonical 恢复、健康幂等、canonical 缺失不动。
"""
import json
import os

import pytest

import service_run


@pytest.fixture
def guard_home(tmp_path, monkeypatch):
    """把 ~ 指到 tmp_path，使 ensure_catalog_key 读写 tmp 下的 config.toml。

    戳记路径也指向 tmp（测试隔离：真实 ROOT 落过 uninstall 戳记后本组测试不会全红）。
    """
    monkeypatch.setattr(service_run, "UNINSTALLED_STAMP", str(tmp_path / "uninstalled.stamp"))
    monkeypatch.setattr(os.path, "expanduser", lambda p: str(tmp_path / ".codex" / "config.toml"))
    cfg_dir = tmp_path / ".codex"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    return cfg_dir / "config.toml"


def _fake_expanduser_for(tmp_path):
    """通用 expanduser 替身：~\\.codex\\... 映射到 tmp_path/.codex/...，其余原样返回。"""

    def fake(p):
        if p.startswith("~"):
            rel = p[1:].lstrip("\\/").replace("\\", "/")
            return str(tmp_path.joinpath(*rel.split("/")))
        return p

    return fake


def _make_catalog(n_models=service_run.MIN_MODELS + 6, ctx=256000, compact=204800):
    """生成健康 catalog 结构；n/ctx/compact 可注入异常值。"""
    return {
        "models": [
            {
                "slug": f"model-{i}",
                "display_name": f"model-{i}",
                "context_window": ctx,
                "auto_compact_token_limit": compact,
            }
            for i in range(n_models)
        ]
    }


@pytest.fixture
def catalog_env(tmp_path, monkeypatch):
    """#21 环境：~ 指向 tmp，canonical 指向 tmp 内文件，返回 (catalog, canonical) 路径。"""
    monkeypatch.setattr(os.path, "expanduser", _fake_expanduser_for(tmp_path))
    canonical = tmp_path / "canonical-catalog.json"
    monkeypatch.setattr(service_run, "CANONICAL_CATALOG", str(canonical))
    catalog = tmp_path / ".codex" / "model-catalogs" / "relay-mu9j3868.json"
    catalog.parent.mkdir(parents=True, exist_ok=True)
    return catalog, canonical


def _read(p):
    with open(p, encoding="utf-8") as f:
        return f.read()


def test_inserts_key_before_first_table(guard_home):
    guard_home.write_text(
        'model = "glm-5.3-flash"\nnotify = ["x"]\n[model_providers.workbuddy]\nname = "w"\n',
        encoding="utf-8",
    )
    assert service_run.ensure_catalog_key() is True
    text = _read(guard_home)
    assert 'model_catalog_json = "model-catalogs/relay-mu9j3868.json"' in text
    # 顶层键必须落在第一个表头之前
    assert text.index("model_catalog_json") < text.index("[model_providers.workbuddy]")
    # 其余内容保留
    assert 'model = "glm-5.3-flash"' in text
    assert 'notify = ["x"]' in text
    assert 'name = "w"' in text
    # 原子替换无残留临时文件
    assert not os.path.exists(str(guard_home) + ".guard.tmp")


def test_bare_key_normalized_into_managed_block(guard_home):
    """S1 深化：值匹配的裸键 → 规范化为 managed 块（键行原样、位置不变、幂等）。"""
    original = 'model = "m"\nmodel_catalog_json = "model-catalogs/relay-mu9j3868.json"\n[desktop]\n'
    guard_home.write_text(original, encoding="utf-8")
    assert service_run.ensure_catalog_key() is True
    text = _read(guard_home)
    assert service_run.MANAGED_BEGIN in text and service_run.MANAGED_END in text
    lines = text.splitlines()
    # 键行原文与位置不变（第 2 行），前后内容不动
    key_line = 'model_catalog_json = "model-catalogs/relay-mu9j3868.json"'
    assert lines[0] == 'model = "m"'
    assert lines[2] == key_line
    assert lines[-1] == "[desktop]"
    assert lines.index(service_run.MANAGED_BEGIN) < lines.index(key_line) < lines.index(service_run.MANAGED_END)
    # 幂等：已是 managed 形态，再跑不动
    assert service_run.ensure_catalog_key() is False
    assert _read(guard_home) == text


def test_foreign_catalog_key_untouched(guard_home):
    """外值裸键（用户自己的 catalog）：不规范化、不插入——防 TOML 重复键。"""
    original = 'model_catalog_json = "my-own-catalog.json"\n[model_providers.workbuddy]\nname = "w"\n'
    guard_home.write_text(original, encoding="utf-8")
    assert service_run.ensure_catalog_key() is False
    assert _read(guard_home) == original


def test_orphan_markers_converge_to_full_block(guard_home):
    """15:10 生产实证形态：孤立 BEGIN/END（键行被多写者竞态弄丢）→ 必须收敛为完整块。

    深优化首版的误判 bug 回归锚：`MANAGED_BEGIN in text → return False` 会把
    残缺块当健康放行——键永远丢、选择器永远回落内置名。
    """
    guard_home.write_text(
        f'model = "m"\n{service_run.MANAGED_BEGIN}\n{service_run.MANAGED_END}\n[desktop]\n',
        encoding="utf-8",
    )
    assert service_run.ensure_catalog_key() is True
    text = _read(guard_home)
    key_line = 'model_catalog_json = "model-catalogs/relay-mu9j3868.json"'
    lines = text.splitlines()
    # 完整块唯一，键在块内，TOML 有效
    assert text.count(service_run.MANAGED_BEGIN) == 1
    assert lines.index(service_run.MANAGED_BEGIN) < lines.index(key_line) < lines.index(service_run.MANAGED_END)
    import tomllib
    tomllib.loads(text)
    # 幂等：收敛后不再动
    assert service_run.ensure_catalog_key() is False
    assert _read(guard_home) == text


def test_silent_when_file_missing(guard_home):
    assert service_run.ensure_catalog_key() is False
    assert not guard_home.exists()


def test_appends_when_no_table(guard_home):
    guard_home.write_text('model = "glm-5.3-flash"\n', encoding="utf-8")
    assert service_run.ensure_catalog_key() is True
    text = _read(guard_home)
    assert "model_catalog_json" in text
    assert 'model = "glm-5.3-flash"' in text


# ----- 补丁 #21：catalog 内容守卫 -----


def _write_json(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False)


def _read_json(path):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def test_catalog_restored_when_truncated(catalog_env):
    catalog, canonical = catalog_env
    _write_json(canonical, _make_catalog())
    _write_json(catalog, _make_catalog(n_models=1))  # Codex++ 单模型覆盖场景
    assert service_run.ensure_catalog_models() is True
    restored = _read_json(catalog)["models"]
    assert len(restored) == service_run.MIN_MODELS + 6
    assert all(
        m["context_window"] == 256000 and m["auto_compact_token_limit"] == 204800
        for m in restored
    )
    assert not os.path.exists(str(catalog) + ".guard.tmp")


def test_catalog_noop_when_healthy(catalog_env):
    catalog, canonical = catalog_env
    healthy = _make_catalog()
    _write_json(canonical, _make_catalog(n_models=1))  # canonical 即使是坏的也不该被用到
    _write_json(catalog, healthy)
    assert service_run.ensure_catalog_models() is False
    assert _read_json(catalog) == healthy


def test_catalog_silent_when_canonical_missing(catalog_env):
    catalog, _ = catalog_env
    broken = _make_catalog(n_models=1)
    _write_json(catalog, broken)
    assert service_run.ensure_catalog_models() is False  # canonical 缺失 → 不动作
    assert _read_json(catalog) == broken


def test_catalog_restored_when_geometry_polluted(catalog_env):
    catalog, canonical = catalog_env
    _write_json(canonical, _make_catalog())
    # 16 个模型但 ctx=272000/compact=244800（Codex++ 模板几何 → 压缩永不触发）
    _write_json(catalog, _make_catalog(n_models=16, ctx=272000, compact=244800))
    assert service_run.ensure_catalog_models() is True
    restored = _read_json(catalog)["models"]
    assert all(
        m["context_window"] == 256000 and m["auto_compact_token_limit"] == 204800
        for m in restored
    )


def test_catalog_silent_when_both_unparseable(catalog_env):
    catalog, canonical = catalog_env
    catalog.write_text("not json{{", encoding="utf-8")
    canonical.write_text("also bad[[", encoding="utf-8")
    assert service_run.ensure_catalog_models() is False  # 坏恢复源不得覆盖
    assert catalog.read_text(encoding="utf-8") == "not json{{"


# ----- 补丁 #22：auth 凭据文件守卫 -----


@pytest.fixture
def auth_env(tmp_path, monkeypatch):
    """#22 环境：auth 目录/权威副本/state.json 全部指向 tmp。"""
    auth_dir = tmp_path / "auth"
    master = tmp_path / "data-auth"
    state = tmp_path / "state.json"
    monkeypatch.setattr(service_run, "AUTH_DIR", str(auth_dir))
    monkeypatch.setattr(service_run, "AUTH_MASTER_DIR", str(master))
    monkeypatch.setattr(service_run, "MANAGEMENT_STATE", str(state))
    return auth_dir, master, state


def _write_state(state_path, files):
    accounts = {f"acct{i}": {"name": f"u{i}", "file": name, "enabled": True} for i, name in enumerate(files)}
    with open(state_path, "w", encoding="utf-8") as f:
        json.dump({"accounts": accounts, "active": "acct0" if files else None}, f)


def test_auth_restored_from_master(auth_env):
    auth_dir, master, state = auth_env
    master.mkdir(parents=True)
    (master / "0-converter-plugin.info").write_text('{"accessToken":"x"}', encoding="utf-8")
    _write_state(state, ["0-converter-plugin.info"])
    assert service_run.ensure_auth_file() is True
    restored = auth_dir / "0-converter-plugin.info"
    assert restored.read_text(encoding="utf-8") == '{"accessToken":"x"}'
    assert not os.path.exists(str(restored) + ".guard.tmp")


def test_auth_noop_when_present(auth_env):
    auth_dir, master, state = auth_env
    auth_dir.mkdir(parents=True)
    (auth_dir / "0-converter-plugin.info").write_text("live", encoding="utf-8")
    _write_state(state, ["0-converter-plugin.info"])
    assert service_run.ensure_auth_file() is False
    assert (auth_dir / "0-converter-plugin.info").read_text(encoding="utf-8") == "live"


def test_auth_silent_when_state_missing(auth_env):
    auth_dir, master, state = auth_env
    assert service_run.ensure_auth_file() is False
    assert not auth_dir.exists()


def test_auth_silent_when_master_missing(auth_env):
    auth_dir, master, state = auth_env
    _write_state(state, ["0-converter-plugin.info"])  # state 指向的副本不存在
    assert service_run.ensure_auth_file() is False
    assert not (auth_dir / "0-converter-plugin.info").exists()


def test_auth_skips_path_traversal(auth_env):
    auth_dir, master, state = auth_env
    master.mkdir(parents=True)
    evil = master.parent / "evil.txt"
    evil.write_text("pwn", encoding="utf-8")
    _write_state(state, ["..\\evil.txt"])
    assert service_run.ensure_auth_file() is False  # 恶意路径被跳过
    assert not auth_dir.exists() or not any(auth_dir.iterdir())


# ----- S1（docs/35 §6.2）：生命周期 CLI —— managed 块 / doctor / setup / uninstall -----


@pytest.fixture
def full_env(tmp_path, monkeypatch):
    """S1 全环境：config/catalog/canonical/auth/master/state/guard.log/ROOT 全在 tmp。"""
    monkeypatch.setattr(os.path, "expanduser", _fake_expanduser_for(tmp_path))
    canonical = tmp_path / "canonical.json"
    monkeypatch.setattr(service_run, "CANONICAL_CATALOG", str(canonical))
    auth_dir = tmp_path / "auth"
    master = tmp_path / "master"
    state = tmp_path / "state.json"
    monkeypatch.setattr(service_run, "AUTH_DIR", str(auth_dir))
    monkeypatch.setattr(service_run, "AUTH_MASTER_DIR", str(master))
    monkeypatch.setattr(service_run, "MANAGEMENT_STATE", str(state))
    guard_log = tmp_path / "guard.log"
    monkeypatch.setattr(service_run, "GUARD_LOG", str(guard_log))
    root = tmp_path / "root"
    root.mkdir()
    monkeypatch.setattr(service_run, "ROOT", str(root))
    # 卸载戳记路径也必须指向 tmp（对抗审计 P1-2 测试前提：不碰真实 ROOT）
    monkeypatch.setattr(service_run, "UNINSTALLED_STAMP", str(tmp_path / "uninstalled.stamp"))
    cfg_dir = tmp_path / ".codex"
    catalog_dir = cfg_dir / "model-catalogs"
    catalog_dir.mkdir(parents=True, exist_ok=True)
    return {
        "cfg": cfg_dir / "config.toml",
        "catalog": catalog_dir / "relay-mu9j3868.json",
        "canonical": canonical,
        "master": master,
        "state": state,
        "guard_log": guard_log,
        "root": root,
    }


def _seed_full_env(env, with_key=False):
    """铺一个「三处全坏」的环境：config 缺键 / catalog 单模型 / auth 缺文件。"""
    text = 'model = "glm-5.3-flash"\n[model_providers.workbuddy]\nname = "w"\n'
    if with_key:
        text = 'model_catalog_json = "model-catalogs/relay-mu9j3868.json"\n' + text
    env["cfg"].write_text(text, encoding="utf-8")
    _write_json(env["canonical"], _make_catalog())
    _write_json(env["catalog"], _make_catalog(n_models=1))
    env["master"].mkdir(parents=True, exist_ok=True)
    (env["master"] / "0-converter-plugin.info").write_text('{"accessToken":"x"}', encoding="utf-8")
    _write_state(env["state"], ["0-converter-plugin.info"])


def test_managed_block_markers_written(guard_home):
    """#20+S1：补回的键必须包在 managed 块标记里（uninstall 依此识别我方内容）。"""
    guard_home.write_text('[model_providers.workbuddy]\nname = "w"\n', encoding="utf-8")
    assert service_run.ensure_catalog_key() is True
    text = _read(guard_home)
    assert service_run.MANAGED_BEGIN in text and service_run.MANAGED_END in text
    assert text.index(service_run.MANAGED_BEGIN) < text.index("model_catalog_json") < text.index(service_run.MANAGED_END)
    # 块在首个表头之前
    assert text.index("model_catalog_json") < text.index("[model_providers.workbuddy]")


def test_run_guards_fixes_all_and_logs(full_env):
    """S1：三处全坏 → run_guards 全修 + guard.log 三条 FIX 记录（R2）。"""
    _seed_full_env(env := full_env)
    fixes = service_run.run_guards()
    assert sorted(fixes) == ["auth", "catalog", "config"]
    log = _read(full_env["guard_log"])
    for frag in ("FIX config", "FIX catalog", "FIX auth"):
        assert frag in log, log
    # 修复结果可验证
    assert "model_catalog_json" in _read(full_env["cfg"])
    assert len(_read_json(full_env["catalog"])["models"]) == service_run.MIN_MODELS + 6
    assert (full_env["master"].parent / "auth" / "0-converter-plugin.info").exists()
    # 健康后重跑：零修复、零新增日志
    before = _read(full_env["guard_log"])
    assert service_run.run_guards() == []
    assert _read(full_env["guard_log"]) == before


def test_doctor_reports_and_fixes(full_env, capsys):
    """S1：doctor 只读发现 FAIL；doctor --fix 修复后复查全 OK。"""
    _seed_full_env(full_env)
    assert service_run.doctor() == 1
    out = capsys.readouterr().out
    assert "FAIL" in out and "model_catalog_json" in out
    assert service_run.doctor(fix=True) == 1  # 修复本次仍报 FAIL（检查先于修复）
    capsys.readouterr()
    assert service_run.doctor() == 0
    out = capsys.readouterr().out
    assert "FAIL" not in out and "全部 OK" in out


def test_setup_backs_up_config(full_env, capsys):
    """S1：setup 幂等安装——备份 config 到 backup-config/ 后执行修复。"""
    _seed_full_env(full_env)
    assert service_run.setup() == 1  # 检查阶段仍有 FAIL（修复已执行）
    backups_dir = full_env["root"] / "backup-config"
    backups = list(backups_dir.glob("config.toml.*"))
    assert len(backups) == 1
    assert service_run.doctor() == 0  # setup 已修复全部


def test_uninstall_removes_managed_block_only(full_env, capsys):
    """S1：uninstall 只移除 managed 块，其余内容原样；legacy 裸键绝不动。"""
    _seed_full_env(full_env)
    service_run.run_guards()  # 补回 managed 块
    assert service_run.uninstall() == 0
    text = _read(full_env["cfg"])
    assert "model_catalog_json" not in text
    assert service_run.MANAGED_BEGIN not in text
    assert 'model = "glm-5.3-flash"' in text and 'name = "w"' in text
    assert "UNINSTALL" in _read(full_env["guard_log"])
    # 二次 uninstall：无块可删，幂等退出
    assert service_run.uninstall() == 0
    # legacy 裸键（非我方写入）：uninstall 不删除
    _seed_full_env(full_env, with_key=True)
    assert service_run.uninstall() == 0
    assert 'model_catalog_json = "model-catalogs/relay-mu9j3868.json"' in _read(full_env["cfg"])


def test_uninstall_blocks_watchdog_reinstall(full_env, capsys):
    """对抗审计 P1-2：uninstall 落戳记后，看门狗不得把 managed 块补回；setup 清戳记恢复。"""
    _seed_full_env(full_env)
    service_run.run_guards()
    assert "model_catalog_json" in _read(full_env["cfg"])
    assert service_run.uninstall() == 0
    assert "model_catalog_json" not in _read(full_env["cfg"])
    # 看门狗 tick（ensure_catalog_key 直调 + run_guards 整体）：均不得重装
    assert service_run.ensure_catalog_key() is False
    assert service_run.run_guards() == []
    assert "model_catalog_json" not in _read(full_env["cfg"])
    assert os.path.isfile(service_run.UNINSTALLED_STAMP)
    # setup 清除戳记 → 守卫恢复，键补回
    capsys.readouterr()
    service_run.setup()
    assert not os.path.isfile(service_run.UNINSTALLED_STAMP)
    assert "model_catalog_json" in _read(full_env["cfg"])


def test_uninstall_aborts_on_unterminated_block(full_env, capsys):
    """对抗审计 P2-10：残缺 managed 块（缺 END）必须放弃写入，不得吞掉其后内容。"""
    _seed_full_env(full_env)
    service_run.run_guards()
    broken = _read(full_env["cfg"]).replace(service_run.MANAGED_END + "\n", "")
    full_env["cfg"].write_text(broken, encoding="utf-8")
    assert service_run.uninstall() == 1
    text = _read(full_env["cfg"])
    # 文件原样保留：BEGIN 还在、后续内容未被吞
    assert service_run.MANAGED_BEGIN in text
    assert 'name = "w"' in text and 'model = "glm-5.3-flash"' in text
    assert "ERR" in _read(full_env["guard_log"])


class _FakeProc:
    """schtasks 替身：记录调用并按脚本返回 exit code。"""

    def __init__(self, script):
        self.script = script  # dict[cmd-keyword] -> returncode
        self.calls = []

    def __call__(self, args, capture_output=False, timeout=None):
        self.calls.append(args)
        verb = args[1]  # /Query | /Change
        return type("R", (), {"returncode": self.script.get(verb, 0)})()


def test_uninstall_full_disables_task(full_env, capsys, monkeypatch):
    """uninstall --full：除 managed 块外，必须禁用计划任务（否则戳记停了守卫但服务仍被拉起）。"""
    _seed_full_env(full_env)
    service_run.run_guards()
    fake = _FakeProc({"/Query": 0, "/Change": 0})
    monkeypatch.setattr(service_run.subprocess, "run", fake)
    assert service_run.uninstall(full=True) == 0
    out = capsys.readouterr().out
    assert "已禁用计划任务" in out
    # /Query 在先（探测存在性），/Change /Disable 在后
    verbs = [c[1] for c in fake.calls]
    assert verbs == ["/Query", "/Change"]
    assert "/Disable" in fake.calls[1]
    assert "UNINSTALL --full" in _read(full_env["guard_log"])


def test_uninstall_full_task_missing_is_benign(full_env, capsys, monkeypatch):
    """uninstall --full 且计划任务不存在：INFO 而非 FAIL（卸载语义不依赖任务存在）。"""
    _seed_full_env(full_env)
    service_run.run_guards()
    fake = _FakeProc({"/Query": 1})  # 任务不存在
    monkeypatch.setattr(service_run.subprocess, "run", fake)
    assert service_run.uninstall(full=True) == 0
    out = capsys.readouterr().out
    assert "不存在" in out
    assert "FAIL" not in out
    # /Query 失败即返回，不发 /Change
    assert len(fake.calls) == 1


def test_uninstall_full_disable_failure_reported(full_env, capsys, monkeypatch):
    """uninstall --full 禁用失败：FAIL 报告 + guard.log 记 ERR；managed 块与戳记仍已完成。"""
    _seed_full_env(full_env)
    service_run.run_guards()
    fake = _FakeProc({"/Query": 0, "/Change": 5})
    monkeypatch.setattr(service_run.subprocess, "run", fake)
    assert service_run.uninstall(full=True) == 0  # 卸载本身成功；任务失败只影响报告
    out = capsys.readouterr().out
    assert "FAIL 禁用计划任务失败" in out
    assert "ERR" in _read(full_env["guard_log"])
    assert os.path.isfile(service_run.UNINSTALLED_STAMP)
