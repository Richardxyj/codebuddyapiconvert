"""档 B 本机配置探针单测（docs/33 §3）。

探针全部只读且绝不抛出：用临时目录替换模块路径常量，验证
健康/缺失/损坏三种形态的返回契约（ok / reason / detail）。
"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from admin import localprobe


def _catalog(models):
    return {"models": models}


def _model(slug, ctx=256000, compact=204800):
    return {"slug": slug, "context_window": ctx, "auto_compact_token_limit": compact}


GOOD_TOML = """
model = "kimi-k3-1"
model_provider = "workbuddy"
model_catalog_json = "model-catalogs/relay-mu9j3868.json"

[model_providers.workbuddy]
base_url = "http://127.0.0.1:8787/v1"
wire_api = "responses"
stream_idle_timeout_ms = 900000
stream_max_retries = 10

[features]
multi_agent_v2 = true
"""


class ApiKeyTests(unittest.TestCase):
    def test_present_and_absent(self):
        with mock.patch.dict("os.environ", {"CODEBUDDY2OPENAI_KEY": "x"}):
            self.assertTrue(localprobe.api_key_status()["present"])
        with mock.patch.dict("os.environ", {}, clear=True):
            self.assertFalse(localprobe.api_key_status()["present"])

    def test_never_returns_value(self):
        with mock.patch.dict("os.environ", {"CODEBUDDY2OPENAI_KEY": "secret-value"}):
            out = localprobe.api_key_status()
        self.assertNotIn("secret-value", json.dumps(out, ensure_ascii=False))


class ConfigStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cfg = Path(self.tmp.name) / "config.toml"

    def tearDown(self):
        self.tmp.cleanup()

    def _patch(self):
        return mock.patch.object(localprobe, "CONFIG_TOML", self.cfg)

    def test_missing_file(self):
        with self._patch():
            out = localprobe.config_status()
        self.assertFalse(out["ok"])
        self.assertIn("不存在", out["reason"])

    def test_healthy(self):
        self.cfg.write_text(GOOD_TOML, encoding="utf-8")
        with self._patch():
            out = localprobe.config_status()
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["missing"], [])
        self.assertEqual(out["detail"]["model"], "kimi-k3-1")

    def test_missing_keys(self):
        self.cfg.write_text('model = "kimi-k3-1"\n', encoding="utf-8")
        with self._patch():
            out = localprobe.config_status()
        self.assertFalse(out["ok"])
        self.assertIn("model_provider", out["missing"])
        self.assertIn("model_catalog_json", out["missing"])
        self.assertIn("model_providers.workbuddy", out["missing"])

    def test_malformed(self):
        self.cfg.write_text("model = [unclosed\n", encoding="utf-8")
        with self._patch():
            out = localprobe.config_status()
        self.assertFalse(out["ok"])
        self.assertIn("解析失败", out["reason"])


class CatalogStatusTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.cat = Path(self.tmp.name) / "relay.json"

    def tearDown(self):
        self.tmp.cleanup()

    def _patch(self):
        return mock.patch.object(localprobe, "CATALOG", self.cat)

    def test_missing(self):
        with self._patch():
            out = localprobe.catalog_status()
        self.assertFalse(out["ok"])
        self.assertEqual(out["count"], 0)

    def test_healthy_16(self):
        models = [_model(f"m{i}") for i in range(16)]
        self.cat.write_text(json.dumps(_catalog(models)), encoding="utf-8")
        with self._patch():
            out = localprobe.catalog_status()
        self.assertTrue(out["ok"], out)
        self.assertEqual(out["count"], 16)
        self.assertTrue(out["geometry_ok"])

    def test_too_few(self):
        self.cat.write_text(json.dumps(_catalog([_model("only")])), encoding="utf-8")
        with self._patch():
            out = localprobe.catalog_status()
        self.assertFalse(out["ok"])

    def test_bad_geometry(self):
        models = [_model(f"m{i}", ctx=272000, compact=244800) for i in range(12)]
        self.cat.write_text(json.dumps(_catalog(models)), encoding="utf-8")
        with self._patch():
            out = localprobe.catalog_status()
        self.assertFalse(out["ok"])
        self.assertFalse(out["geometry_ok"])


class LogTailTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.log = Path(self.tmp.name) / "converter.log"

    def tearDown(self):
        self.tmp.cleanup()

    def _patch(self):
        return mock.patch.object(localprobe, "CONVERTER_LOG", self.log)

    def test_missing(self):
        with self._patch():
            out = localprobe.read_log_tail()
        self.assertFalse(out["ok"])
        self.assertEqual(out["lines"], [])

    def test_annotation(self):
        self.log.write_text(
            "normal line\n✗ HTTP 400 bad\nRESPONSES PROJECTION aggressive\n",
            encoding="utf-8",
        )
        with self._patch():
            out = localprobe.read_log_tail(lines=10)
        self.assertTrue(out["ok"])
        kinds = [l["kind"] for l in out["lines"]]
        self.assertEqual(kinds, ["normal", "error", "projection"])

    def test_tail_limit(self):
        self.log.write_text("\n".join(f"line{i}" for i in range(500)), encoding="utf-8")
        with self._patch():
            out = localprobe.read_log_tail(lines=50)
        self.assertLessEqual(len(out["lines"]), 50)


class LocalOverviewTests(unittest.TestCase):
    def test_aggregates_and_never_raises(self):
        with tempfile.TemporaryDirectory() as d:
            missing = Path(d) / "nope"
            with mock.patch.object(localprobe, "CONFIG_TOML", missing), \
                 mock.patch.object(localprobe, "CATALOG", missing), \
                 mock.patch.object(localprobe, "CONVERTER_LOG", missing), \
                 mock.patch.object(localprobe, "task_status", lambda *a, **k: {"ok": True}), \
                 mock.patch.object(localprobe, "port_status", lambda *a, **k: {"ok": True}):
                out = localprobe.local_overview()
        self.assertIn("sections", out)
        for key in ("api_key", "config", "catalog", "port", "task"):
            self.assertIn(key, out["sections"])
        self.assertFalse(out["sections"]["config"]["ok"])


if __name__ == "__main__":
    unittest.main()
