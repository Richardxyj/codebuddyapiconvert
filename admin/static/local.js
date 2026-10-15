/* 档 B「本地控制中心」本机三页面（docs/33 §3）。
   复用 app.js 暴露的全局工具：api() / esc() / goPage()。
   所有数据来自 /admin/api/local/* 只读端点；本脚本不发起任何写操作。 */
"use strict";
(function () {
  const $ = id => document.getElementById(id);

  // ---- 配置总览（§3.1）----
  function pill(ok) {
    return `<span class="pill ${ok ? "green" : "red"}">${ok ? "正常" : "异常"}</span>`;
  }
  function card(title, ok, rowsHtml, reason) {
    return `<article class="surface lo-card ${ok ? "" : "lo-bad"}">`
      + `<div class="lo-card-head"><h3>${esc(title)}</h3>${pill(ok)}</div>`
      + (ok ? "" : `<p class="lo-reason">${esc(reason || "未知异常")}</p>`)
      + `<dl class="lo-kv">${rowsHtml}</dl></article>`;
  }
  const kv = (k, v) => `<div><dt>${esc(k)}</dt><dd>${esc(v === null || v === undefined || v === "" ? "—" : v)}</dd></div>`;

  function renderOverview(data) {
    const s = (data && data.sections) || {};
    const out = [];

    const ak = s.api_key || {};
    out.push(card("客户端 API Key", !!ak.ok && ak.present, [
      kv("环境变量", "CODEBUDDY2OPENAI_KEY"),
      kv("已设置", ak.present ? "是" : "否"),
    ], ak.present ? "" : "未设置客户端密钥环境变量"));

    const cf = s.config || {};
    const cd = cf.detail || {};
    out.push(card("Codex 配置 (config.toml)", !!cf.ok, [
      kv("默认模型", cd.model),
      kv("提供方", cd.model_provider),
      kv("catalog 指针", cd.model_catalog_json),
      kv("base_url", cd.provider_base_url),
      kv("流式空闲超时", cd.stream_idle_timeout_ms != null ? cd.stream_idle_timeout_ms + " ms" : "—"),
      kv("多智能体 v2", cd.multi_agent_v2 === true ? "开" : "关"),
      kv("缺失键", (cf.missing && cf.missing.length) ? cf.missing.join(", ") : "无"),
    ], cf.reason + ((cf.missing && cf.missing.length) ? "：" + cf.missing.join(", ") : "")));

    const cat = s.catalog || {};
    out.push(card("模型目录 (catalog)", !!cat.ok, [
      kv("条目数", cat.count != null ? String(cat.count) : "—"),
      kv("几何 256000/204800", cat.geometry_ok ? "符合" : "异常"),
      kv("模型", (cat.slugs || []).slice(0, 6).join(", ") + ((cat.slugs || []).length > 6 ? " …" : "")),
    ], cat.reason));

    const pt = s.port || {};
    out.push(card("转换器端口 :8787", !!pt.ok, [
      kv("监听", pt.ok ? "是" : "否"),
      kv("PID", pt.pid != null ? String(pt.pid) : "—"),
    ], pt.reason));

    const tk = s.task || {};
    out.push(card("守护任务 Codebuddy2Api", !!tk.ok, [
      kv("状态", tk.state),
      kv("最近结果", tk.last_result),
      kv("最近运行", tk.last_run),
    ], tk.reason));

    $("lo-cards").innerHTML = out.join("");
  }

  async function loadOverview() {
    try {
      const data = await api("local/overview");
      renderOverview(data);
    } catch (e) {
      $("lo-cards").innerHTML = `<p class="history-empty">${esc(e.message)}</p>`;
    }
  }

  // ---- 凭据管理（§3.2）----
  async function loadCredentials() {
    try {
      const data = await api("overview");
      const accounts = data.accounts || [];
      const active = accounts.find(a => a.active && a.enabled);
      $("lc-current").innerHTML = active
        ? `<div class="lo-cred-active"><strong>${esc(active.name)}</strong>`
          + `<div class="muted mono">${esc(active.uid || active.nickname || "")}</div>`
          + `<div class="muted">剩余积分：${active.remaining === null || active.remaining === undefined ? "待查询" : esc(active.remaining)}</div></div>`
        : `<p class="history-empty">当前没有启用的调用账号，请到"账号池"启用一个。</p>`;
    } catch (e) {
      $("lc-current").innerHTML = `<p class="history-empty">${esc(e.message)}</p>`;
    }
  }

  // ---- 日志诊断（§3.3）----
  function renderLogs(data) {
    if (!data.ok) {
      $("ll-body").textContent = data.reason || "读取失败";
      $("ll-meta").textContent = data.path || "";
      return;
    }
    const html = (data.lines || []).map(l => {
      const cls = l.kind === "error" ? "lg-err" : l.kind === "projection" ? "lg-proj" : "";
      return `<span class="${cls}">${esc(l.text)}</span>`;
    }).join("\n");
    $("ll-body").innerHTML = html || "（日志为空）";
    const kb = data.size != null ? (data.size / 1024 / 1024).toFixed(1) + " MB" : "";
    $("ll-meta").textContent = `${data.path} · 共 ${kb}${data.truncated ? " · 仅显示尾部" : ""}`;
    $("ll-body").scrollTop = $("ll-body").scrollHeight;
  }

  async function loadLogs() {
    const lines = $("ll-lines") ? $("ll-lines").value : 120;
    try {
      const data = await api("local/logs?lines=" + encodeURIComponent(lines));
      renderLogs(data);
    } catch (e) {
      $("ll-body").textContent = e.message;
    }
  }

  // ---- 路由分发（app.js goPage 调用）----
  function render(page) {
    if (page === "local-overview") loadOverview();
    else if (page === "local-credentials") loadCredentials();
    else if (page === "local-logs") loadLogs();
  }

  function bind() {
    if ($("lo-refresh")) $("lo-refresh").addEventListener("click", loadOverview);
    if ($("ll-refresh")) $("ll-refresh").addEventListener("click", loadLogs);
    if ($("ll-lines")) $("ll-lines").addEventListener("change", loadLogs);
    if ($("lc-goto-accounts")) $("lc-goto-accounts").addEventListener("click", () => goPage("accounts"));
  }

  window.LocalPages = { render };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", bind);
  else bind();
})();
