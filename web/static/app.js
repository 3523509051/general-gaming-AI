/* NitroGen 评估工作台 - 前端交互逻辑 */

// ---------------- 全局状态 ----------------
const state = {
  games: [],
  videos: [],
  game: "",
  video: "",
  frames: [],        // 测试集帧列表 [{absolute_frame, second}]
  frameIdx: -1,      // 当前帧在 frames 中的下标
  lastFrameData: null,
  charts: {},        // ECharts 实例
  loadedTabs: {},    // 懒加载标记
};

// ---------------- 工具 ----------------
function fmtDuration(sec) {
  sec = Math.max(0, Math.round(sec || 0));
  const h = Math.floor(sec / 3600), m = Math.round((sec % 3600) / 60);
  if (h >= 10) return `${h}小时`;
  if (h >= 1) return `${h}小时${m}分`;
  if (sec >= 90) return `${m}分`;
  return `${sec}秒`;
}

async function api(path) {
  const r = await fetch(path);
  let j;
  try { j = await r.json(); } catch { throw new Error(`响应解析失败 (${r.status})`); }
  if (!j.ok) throw new Error(j.error || `请求失败 (${r.status})`);
  return j.data;
}

function showError(msg) {
  const b = document.getElementById("errorBanner");
  b.textContent = "⚠ " + msg;
  b.classList.remove("hidden");
  clearTimeout(b._t);
  b._t = setTimeout(() => b.classList.add("hidden"), 6000);
}

function setLineage(text, cls) {
  document.getElementById("lineageText").textContent = text;
  const badge = document.getElementById("lineageBadge");
  badge.textContent = cls === "ok" ? "血缘校验 ✅ 通过" : (cls === "warn" ? "未就绪" : "—");
  badge.className = "badge " + (cls || "");
}

function syncURL() {
  const p = new URLSearchParams();
  if (state.game) p.set("game", state.game);
  if (state.video) p.set("video", state.video);
  if (state.frames[state.frameIdx]) p.set("frame", state.frames[state.frameIdx].absolute_frame);
  const tab = document.querySelector(".tab.active")?.dataset.tab || "";
  if (tab) p.set("tab", tab.replace("tab-", ""));
  history.replaceState(null, "", "?" + p.toString());
}

// ---------------- 初始化 ----------------
window.addEventListener("DOMContentLoaded", async () => {
  await loadGames();
  // 若后台仍有探测/下载任务在跑，恢复进度轮询
  try {
    const st = await api("/api/rescan/status");
    if (st.running) pollRescan();
  } catch { /* ignore */ }
  try {
    const dl = await api("/api/download/status");
    if (dl.running) pollDownload();
  } catch { /* ignore */ }
  try {
    const ev = await api("/api/evaluate/status");
    if (ev.running) pollEvaluate();
  } catch { /* ignore */ }
  try {
    const gp = await api("/api/genplots/status");
    if (gp.running) pollGenplots();
  } catch { /* ignore */ }
  const p = new URLSearchParams(location.search);
  if (p.get("game")) {
    await onGameChange(p.get("game"), p.get("video"));
    if (p.get("tab")) switchTab("tab-" + p.get("tab"));
    if (p.get("frame") && state.frames.length) {
      const idx = state.frames.findIndex(f => f.absolute_frame === parseInt(p.get("frame")));
      if (idx >= 0) await selectFrame(idx);
    }
  }
});

async function loadGames() {
  try {
    const d = await api("/api/games");
    state.games = d.games;
    const sel = document.getElementById("gameSelect");
    sel.innerHTML = "";
    for (const g of d.games) {
      const o = document.createElement("option");
      o.value = g.game;
      const flags = [];
      if (g.tested) flags.push("已测");
      else if (g.ready_videos > 0) flags.push("可测");
      o.textContent = `${g.game}（${g.videos} 视频 / 录像约 ${fmtDuration(g.video_seconds)}${flags.length ? " · " + flags.join("·") : ""}）`;
      sel.appendChild(o);
    }
    sel.disabled = false;
  } catch (e) { showError("加载游戏列表失败: " + e.message); }
}

// ---------------- 游戏 / 视频联动 ----------------
async function onGameChange(game, presetVideo) {
  state.game = game;
  state.video = "";
  state.frames = [];
  state.frameIdx = -1;
  state.loadedTabs = {};
  state.gameInfo = state.games.find(g => g.game === game) || null;
  _estShown = false;
  document.getElementById("videoSelect").disabled = true;
  document.getElementById("videoSelect").innerHTML = '<option value="">加载中...</option>';
  setLineage(`${game} / （未选视频）`, "warn");
  syncURL();
  updateExtractBanner();
  // 页面刷新时若该游戏提取任务进行中，恢复轮询
  try {
    const st = await api("/api/extract/status");
    if (st.running && st.game === game) pollExtract();
  } catch { /* ignore */ }
  await loadVideos(presetVideo);
}

async function loadVideos(presetVideo) {
  try {
    const d = await api(`/api/games/${encodeURIComponent(state.game)}/videos`);
    state.videos = d.videos;
    const sel = document.getElementById("videoSelect");
    sel.innerHTML = "";
    for (const v of d.videos) {
      const o = document.createElement("option");
      o.value = v.video;
      const st = { downloaded: "已下载✅", available: "可下载", dead: "失效❌", unknown: "状态未知" }[v.status] || v.status;
      const t = v.tested ? ` · 已测 acc17=${(v.acc_17keys * 100).toFixed(1)}%` : "";
      o.textContent = `${v.video}（${st}${t}）`;
      o.title = v.status === "unknown" ? (v.error || "未探测，可点「探测链接」验证") : "";
      sel.appendChild(o);
    }
    sel.disabled = false;
    if (presetVideo && d.videos.some(v => v.video === presetVideo)) {
      sel.value = presetVideo;
      await onVideoChange(presetVideo);
    } else {
      const dl = d.videos.find(v => v.status === "downloaded");
      if (dl) { sel.value = dl.video; await onVideoChange(dl.video); }
    }
  } catch (e) {
    showError(e.message);
    document.getElementById("videoSelect").innerHTML = '<option value="">加载失败</option>';
  }
}

async function onVideoChange(video) {
  if (!video) return;
  state.video = video;
  state.loadedTabs = {};
  setLineage(`${state.game} / ${video}`, "ok");
  syncURL();
  // 加载测试集帧列表
  try {
    const d = await api(`/api/testset?game=${encodeURIComponent(state.game)}&video=${encodeURIComponent(video)}`);
    state.frames = d.frames;
    const slider = document.getElementById("frameSlider");
    slider.max = Math.max(0, d.frames.length - 1);
    if (d.frames.length) await selectFrame(0);
  } catch (e) {
    setLineage(`${state.game} / ${video}（无测试集，请先运行评估脚本）`, "warn");
    showError(e.message);
  }
  // 核心指标对比条
  loadMetricsBanner();
}

// ---------------- 核心指标对比条（zero-shot 结果 vs 参考水平，随 shift 实时变化） ----------------
async function loadMetricsBanner() {
  const banner = document.getElementById("metricsBanner");
  try {
    // 跟随当前 shift 选择器（auto=最优；具体 k 则实时返回该步指标）
    const shift = (document.getElementById("shiftSelect") || {}).value || "auto";
    const d = await api(`/api/metrics?game=${encodeURIComponent(state.game)}&video=${encodeURIComponent(state.video)}&shift=${shift}`);
    banner.classList.remove("hidden");
    const shiftLabel = shift === "auto" ? `shift k=${d.best_shift}(最优)` : `shift k=${shift}`;
    document.getElementById("metricsScope").textContent =
      `${d.game} / ${d.video} · ${d.metrics.test_frames ?? d.test_frames} 帧 · ${shiftLabel}`;
    const m = d.metrics, ref = d.reference, v = d.verdicts;
    const chips = [
      { label: "按键一致率(逐帧全对)", value: (m.acc_17keys * 100).toFixed(1) + "%",
        vs: `参考 ${(ref.acc_17keys * 100).toFixed(0)}%（17键全一致帧占比）`, good: v.acc_17keys, neutral: false },
      { label: "按键召回率", value: (m.btn_recall * 100).toFixed(1) + "%",
        vs: "—", good: null, neutral: true },
      { label: "摇杆相关系数", value: m.corr_jl.toFixed(2),
        vs: `参考 ${ref.corr_jl.toFixed(2)}`, good: v.corr_jl, neutral: false },
      { label: "摇杆 MSE (x)", value: m.mse_jl != null ? m.mse_jl.toFixed(3) : "—",
        vs: "越小越好", good: null, neutral: true },
    ];
    document.getElementById("metricsChips").innerHTML = chips.map(c => {
      const cls = c.neutral ? "neutral" : (c.good ? "good" : "bad");
      const flag = c.neutral ? "" : (c.good ? " ✅" : " ❌");
      return `<div class="metric-chip ${cls}">
        <div class="mc-label">${c.label}</div>
        <div class="mc-value">${c.value}${flag}</div>
        <div class="mc-vs">${c.vs}</div>
      </div>`;
    }).join("");
  } catch (e) {
    banner.classList.add("hidden");  // 未评估或无测试集时隐藏
  }
}

// ---------------- Tab 切换 ----------------
function switchTab(id) {
  document.querySelectorAll(".tab").forEach(t => t.classList.toggle("active", t.dataset.tab === id));
  document.querySelectorAll(".tab-panel").forEach(p => p.classList.toggle("active", p.id === id));
  syncURL();
  // 懒加载
  if (id === "tab-stats" && !state.loadedTabs.stats) { state.loadedTabs.stats = true; loadStats(); }
  if (id === "tab-seq" && !state.loadedTabs.seq) { state.loadedTabs.seq = true; loadSequences(); }
  // resize 已渲染图表
  Object.values(state.charts).forEach(c => c && c.resize());
}

// ---------------- Tab1: 帧识别 ----------------
async function selectFrame(idx, shiftOverride) {
  if (idx < 0 || idx >= state.frames.length) return;
  state.frameIdx = idx;
  const f = state.frames[idx];
  document.getElementById("frameSlider").value = idx;
  document.getElementById("frameInput").value = f.absolute_frame;
  document.getElementById("frameSecond").textContent = `≈ ${f.second.toFixed(2)}s`;
  await fetchFrame(f.absolute_frame, shiftOverride);
  syncURL();
}

function onFrameSlider(val) { selectFrame(parseInt(val)); }
function onFrameInput(val) {
  const v = parseInt(val);
  const idx = state.frames.findIndex(f => f.absolute_frame === v);
  if (idx >= 0) selectFrame(idx);
  else showError(`帧 ${v} 不在测试集内（抽样帧）`);
}
async function stepFrame(dir) { await selectFrame(state.frameIdx + dir); }
function onShiftChange(val) {
  if (state.frames[state.frameIdx]) fetchFrame(state.frames[state.frameIdx].absolute_frame, val);
  loadMetricsBanner();  // 顶部指标条随 shift 实时刷新
}

async function fetchFrame(frame, shiftOverride) {
  const overlay = document.getElementById("inferenceOverlay");
  const img = document.getElementById("framePreview");
  img.src = `/files/${state.game}/test_frames/${state.video}_f${String(frame).padStart(5, "0")}.jpg`;
  overlay.classList.remove("hidden");
  overlay.textContent = state._modelLoaded ? "推理中（约 0.3s）…" : "模型首次加载中（约 1 分钟）…";
  try {
    const shift = shiftOverride || document.getElementById("shiftSelect").value;
    const d = await api(`/api/frame?game=${encodeURIComponent(state.game)}&video=${encodeURIComponent(state.video)}&frame=${frame}&fresh=1&shift=${shift}`);
    state._modelLoaded = true;
    state.lastFrameData = d;
    renderFrameResult(d);
  } catch (e) {
    showError("识别失败: " + e.message);
    document.getElementById("gamepad").innerHTML = `<div class="loading-block">${e.message}</div>`;
  } finally {
    overlay.classList.add("hidden");
  }
}

// ---------------- 手柄渲染 ----------------
function renderFrameResult(d) {
  // 真值摘要
  const gt = d.ground_truth, pred = d.prediction;
  const gtBtns = Object.entries(gt.buttons).filter(([, v]) => v).map(([k]) => k);
  const predBtns = Object.entries(pred.buttons).filter(([, v]) => v).map(([k]) => k);
  document.getElementById("gtSummary").innerHTML = `
    <div><span class="k">真值按键</span> ${gtBtns.length ? gtBtns.join(", ") : "（无）"}</div>
    <div><span class="k">预测按键</span> ${predBtns.length ? predBtns.join(", ") : "（无）"}</div>
    <div><span class="k">真值摇杆 L</span> (${gt.j_left[0].toFixed(2)}, ${gt.j_left[1].toFixed(2)})　
        <span class="k" style="min-width:auto">预测</span> (${pred.j_left[0].toFixed(2)}, ${pred.j_left[1].toFixed(2)})</div>
    <div><span class="k">帧定位</span> chunk=${d.frame.chunk}, local=${d.frame.frame_idx}, ${d.frame.second.toFixed(2)}s</div>`;

  // 手柄
  renderGamepad(gt, pred);

  // 摇杆轨迹（18 步动作块）
  renderStickChart(d);

  // 指标
  const m = d.metrics;
  const acc = 1 - m.n_mismatch / 17;
  document.getElementById("frameMetrics").innerHTML = `
    ${metricCard((acc * 100).toFixed(1) + "%", "本帧17键一致率", acc > 0.9 ? "good" : "")}
    ${metricCard(m.n_mismatch, "差异键数", m.n_mismatch === 0 ? "good" : "bad")}
    ${metricCard(m.gt_n_press + " → " + m.pred_n_press, "按键数 (真值→预测)")}
    ${metricCard(d.shift, "对齐 shift k")}
    ${metricCard(m.inference_ms != null ? m.inference_ms + " ms" : "缓存", "推理耗时")}
    ${m.mismatch_keys.length ? `<div class="metric-card bad"><div class="val" style="font-size:14px">${m.mismatch_keys.join(", ")}</div><div class="name">差异键</div></div>` : ""}`;

  // shift 下拉（auto + 0~17，保持当前选择）
  const sel = document.getElementById("shiftSelect");
  const cur = sel.value;
  sel.innerHTML = "";
  for (const [v, t] of [["auto", "auto（最优）"], ...Array.from({ length: 18 }, (_, i) => [String(i), "k=" + i])]) {
    const o = document.createElement("option");
    o.value = v; o.textContent = t;
    sel.appendChild(o);
  }
  sel.value = cur || "auto";
}

function metricCard(val, name, cls = "") {
  return `<div class="metric-card ${cls}"><div class="val">${val}</div><div class="name">${name}</div></div>`;
}

// ---- 17 键 + 2 摇杆的 SVG 坐标（viewBox 720×360，按 Xbox 手柄布局） ----
// 圆 = 摇杆区（半径 38）；ABXY 菱形 + dpad 十字 + 中排三键 + 肩键/扳机
const GP = {
  sticks: {
    left:  { cx: 230, cy: 150, r: 60 },
    right: { cx: 540, cy: 230, r: 60 },
  },
  // 17 个键：标签（xbox 风） + 描述 + 坐标 + 半径
  keys: [
    // ABXY 菱形（右上）
    { id: "north",   label: "Y", desc: "north (Y)",     x: 540, y: 100, r: 22 },
    { id: "west",    label: "X", desc: "west (X)",      x: 490, y: 130, r: 22 },
    { id: "east",    label: "B", desc: "east (B)",      x: 590, y: 130, r: 22 },
    { id: "south",   label: "A", desc: "south (A)",     x: 540, y: 160, r: 22 },
    // dpad 十字（左下）
    { id: "dpad_up",    label: "▲", desc: "dpad_up",    x: 130, y: 230, r: 18 },
    { id: "dpad_left",  label: "◀", desc: "dpad_left",  x:  90, y: 260, r: 18 },
    { id: "dpad_right", label: "▶", desc: "dpad_right", x: 170, y: 260, r: 18 },
    { id: "dpad_down",  label: "▼", desc: "dpad_down",  x: 130, y: 290, r: 18 },
    // 中排三键
    { id: "back",  label: "BACK", desc: "back (SELECT)", x: 320, y: 110, r: 16 },
    { id: "guide", label: "◎",    desc: "guide (HOME)",  x: 360, y: 110, r: 16 },
    { id: "start", label: "START", desc: "start",         x: 400, y: 110, r: 16 },
    // 肩键（顶部后方）
    { id: "left_shoulder",  label: "LB", desc: "left_shoulder (L1)", x: 180, y:  35, r: 20 },
    { id: "right_shoulder", label: "RB", desc: "right_shoulder (R1)", x: 540, y:  35, r: 20 },
    // 扳机（顶部最外侧）
    { id: "left_trigger",   label: "LT", desc: "left_trigger (L2)",   x:  90, y:  35, r: 20 },
    { id: "right_trigger",  label: "RT", desc: "right_trigger (R2)",  x: 630, y:  35, r: 20 },
    // 摇杆按键
    { id: "left_thumb",  label: "L3", desc: "left_thumb", x: 230, y: 150, r: 18 },
    { id: "right_thumb", label: "R3", desc: "right_thumb", x: 540, y: 230, r: 18 },
  ],
};

function _stateOf(g, p) {
  if (g && p) return "hit";
  if (g && !p) return "miss";
  if (!g && p) return "false";
  return "off";
}

function renderGamepad(gt, pred) {
  const svg = document.querySelector("#gamepad svg");
  if (!svg) return;
  const g_pred = pred.buttons, g_gt = gt.buttons;

  // 极坐标转换：[-1,1] 坐标 → SVG 圆内坐标
  function stickDot(cx, cy, r, v) {
    return { x: cx + v[0] * r * 0.85, y: cy - v[1] * r * 0.85 };
  }

  const parts = [
    // 渐变定义（命中发光）
    `<defs><radialGradient id="bodyGrad" cx="50%" cy="30%" r="80%">
       <stop offset="0%" stop-color="#f1f5f9"/>
       <stop offset="100%" stop-color="#cbd5e1"/>
     </radialGradient>
     <filter id="glow"><feGaussianBlur stdDeviation="3"/></filter></defs>`,
    // 手柄本体（双握把 + 中央连接）
    `<path class="gp-body" d="
      M ${GP.sticks.left.cx-110},70
      Q ${GP.sticks.left.cx-130},200 ${GP.sticks.left.cx-50},320
      Q 220,340 360,340 Q 500,340 ${GP.sticks.right.cx+50},320
      Q ${GP.sticks.right.cx+130},200 ${GP.sticks.right.cx+110},70
      Q 600,40 540,30 L 180,30 Q 120,40 ${GP.sticks.left.cx-110},70 Z" />`,
    // 摇杆区域（虚线圆 + 标签）
    ...Object.entries(GP.sticks).flatMap(([name, s]) => [
      `<circle class="gp-stick-ring" cx="${s.cx}" cy="${s.cy}" r="${s.r}" />`,
      `<text class="gp-stick-zone-label" x="${s.cx}" y="${s.cy + s.r + 14}">${name.toUpperCase()} 摇杆</text>`,
      `<title>${name} 摇杆真值 (${(name==='left'?gt:pred).j_left?.[0]?.toFixed?.(2) ?? 'n/a'}, ...)</title>`,
    ]),
    // 摇杆点（真值蓝 / 预测橙）
    ...["left", "right"].map(name => {
      const s = GP.sticks[name];
      const dG = stickDot(s.cx, s.cy, s.r, name==="left" ? gt.j_left : gt.j_right);
      const dP = stickDot(s.cx, s.cy, s.r, name==="left" ? pred.j_left : pred.j_right);
      return [
        `<circle class="gp-stick-dot gp-stick-gt"   cx="${dG.x.toFixed(1)}" cy="${dG.y.toFixed(1)}" r="7" />`,
        `<circle class="gp-stick-dot gp-stick-pred" cx="${dP.x.toFixed(1)}" cy="${dP.y.toFixed(1)}" r="7" />`,
      ];
    }).flat(),
    // 17 个键（命中四态上色）
    ...GP.keys.map(k => {
      const g = g_gt[k.id] ? 1 : 0, p = g_pred[k.id] ? 1 : 0;
      const cls = _stateOf(g, p);
      return `<g class="gp-btn gp-${cls}">
        <circle class="gp-btn-glow" cx="${k.x}" cy="${k.y}" r="${k.r+6}" filter="url(#glow)" />
        <circle class="gp-btn-bg" cx="${k.x}" cy="${k.y}" r="${k.r}" />
        <text class="gp-btn-label" x="${k.x}" y="${k.y+5}">${k.label}</text>
        <title>${k.desc}  真值=${g}  预测=${p}</title>
      </g>`;
    }),
  ];
  svg.innerHTML = parts.join("");

  // 下方 17 键文字列表（兜底：确保 17 个键都能看到 + 名称/状态）
  const list = document.getElementById("gamepadList");
  list.innerHTML = GP.keys.map(k => {
    const g = g_gt[k.id] ? 1 : 0, p = g_pred[k.id] ? 1 : 0;
    return `<div class="gb-item ${_stateOf(g, p)}">
      <span class="gb-dot"></span>
      <span class="gb-name">${k.label}</span>
      <span class="gb-name" style="color:var(--text-dim);font-weight:400">${k.id}</span>
      <span class="gb-vals">${g}/${p}</span>
    </div>`;
  }).join("");
}

// ---------------- 摇杆轨迹图（18 步动作块） ----------------
function renderStickChart(d) {
  const el = document.getElementById("stickChart");
  if (!state.charts.stick) state.charts.stick = echarts.init(el);
  const chart = state.charts.stick;
  const block = d.action_block;
  const series = [];

  if (block) {
    // 18 步预测轨迹（左摇杆）
    series.push({
      name: "预测轨迹(左摇杆)",
      type: "line", symbol: "circle", symbolSize: 6,
      data: block.j_left.map(p => [p[0], p[1]]),
      lineStyle: { color: "#2563eb", width: 2 }, itemStyle: { color: "#2563eb" },
      emphasis: { focus: "series" },
    });
    // 标注 shift 步
    series.push({
      name: `第 ${d.shift} 步(对齐)`,
      type: "scatter", symbolSize: 18,
      data: [[block.j_left[d.shift][0], block.j_left[d.shift][1]]],
      itemStyle: { color: "rgba(37,99,235,0.35)", borderColor: "#1e40af", borderWidth: 2 },
    });
  }
  // 真值点
  series.push({
    name: "真值(左摇杆)",
    type: "scatter", symbolSize: 16,
    data: [[d.ground_truth.j_left[0], d.ground_truth.j_left[1]]],
    itemStyle: { color: "#16a34a" },
  });
  if (block) {
    series.push({
      name: "预测(右摇杆)",
      type: "line", symbol: "diamond", symbolSize: 5,
      data: block.j_right.map(p => [p[0], p[1]]),
      lineStyle: { color: "#ea580c", width: 1.5, type: "dashed" }, itemStyle: { color: "#ea580c" },
    });
  }
  series.push({
    name: "真值(右摇杆)",
    type: "scatter", symbolSize: 14, symbol: "diamond",
    data: [[d.ground_truth.j_right[0], d.ground_truth.j_right[1]]],
    itemStyle: { color: "#9a3412" },
  });

  chart.setOption({
    tooltip: { trigger: "item" },
    legend: { top: 0, textStyle: { fontSize: 11 } },
    grid: { left: 40, right: 20, top: 30, bottom: 30 },
    xAxis: { name: "x", min: -1.1, max: 1.1, splitLine: { lineStyle: { color: "#eef2f7" } } },
    yAxis: { name: "y", min: -1.1, max: 1.1, splitLine: { lineStyle: { color: "#eef2f7" } } },
    series,
  }, true);
}

// ---------------- Tab2: 统计分布 ----------------
async function loadStats() {
  document.getElementById("buttonChart").innerHTML = `<div class="loading-block"><div class="spinner"></div>统计计算中（全量标注帧）...</div>`;
  try {
    const d = await api(`/api/stats?game=${encodeURIComponent(state.game)}&video=${encodeURIComponent(state.video)}`);
    renderButtonChart(d.buttons);
    renderStickDist(d.joystick_samples);
    renderStatsSummary(d);
    renderStaticPlots(d.plots, d.metrics_row);
  } catch (e) {
    showError("统计加载失败: " + e.message);
    document.getElementById("buttonChart").innerHTML = `<div class="loading-block">${e.message}</div>`;
  }
}

function renderButtonChart(buttons) {
  const el = document.getElementById("buttonChart");
  el.innerHTML = "";
  if (!state.charts.button) state.charts.button = echarts.init(el);
  state.charts.button.setOption({
    tooltip: { trigger: "axis", valueFormatter: v => (v * 100).toFixed(2) + "%" },
    grid: { left: 50, right: 20, top: 20, bottom: 60 },
    xAxis: { type: "category", data: buttons.map(b => b.button), axisLabel: { rotate: 45, fontSize: 11 } },
    yAxis: { type: "value", axisLabel: { formatter: v => (v * 100).toFixed(0) + "%" } },
    series: [{
      type: "bar", data: buttons.map(b => b.press_rate),
      itemStyle: { color: "#2563eb", borderRadius: [4, 4, 0, 0] },
      label: { show: true, position: "top", fontSize: 9, formatter: p => (p.value * 100).toFixed(1) },
    }],
  }, true);
}

function renderStickDist(samples) {
  // 左 / 右摇杆各一张分布图（同等地位）
  renderStickDistOne("stickDistLeftChart", "distLeft", "左摇杆采样", samples.j_left,
                     "rgba(37,99,235,0.28)");
  renderStickDistOne("stickDistRightChart", "distRight", "右摇杆采样", samples.j_right,
                     "rgba(234,88,12,0.28)");
}

function renderStickDistOne(elId, chartKey, name, data, color) {
  const el = document.getElementById(elId);
  el.innerHTML = "";
  if (!state.charts[chartKey]) state.charts[chartKey] = echarts.init(el);
  state.charts[chartKey].setOption({
    tooltip: { formatter: p => `(${p.data[0].toFixed(2)}, ${p.data[1].toFixed(2)})` },
    grid: { left: 45, right: 15, top: 12, bottom: 25 },
    xAxis: { min: -1.1, max: 1.1, name: "x", splitLine: { lineStyle: { color: "#eef2f7" } } },
    yAxis: { min: -1.1, max: 1.1, name: "y", splitLine: { lineStyle: { color: "#eef2f7" } } },
    series: [{
      name, type: "scatter", symbolSize: 2.5,
      data, itemStyle: { color },
      large: true, largeThreshold: 1000,
    }],
  }, true);
}

function renderStatsSummary(d) {
  const s = d.summary;
  document.getElementById("statsSummary").innerHTML = `
    ${sumItem("标注帧数（该视频）", d.stats_frames.toLocaleString())}
    ${sumItem("最高频按键", s.top_button)}
    ${sumItem("最高频按键触发率", (s.top_press_rate * 100).toFixed(1) + "%")}
    ${sumItem("IDLE 帧占比", (s.idle_rate * 100).toFixed(1) + "%")}
    ${sumItem("左摇杆移动率", (s.left_stick_move_rate * 100).toFixed(1) + "%")}
    ${sumItem("右摇杆移动率", (s.right_stick_move_rate * 100).toFixed(1) + "%")}
    ${d.metrics_row ? sumItem("测试集 acc17", (d.metrics_row.acc_17keys * 100).toFixed(1) + "%") : ""}`;
}

function sumItem(k, v) {
  return `<div class="stats-summary-item"><span>${k}</span><span class="v">${v}</span></div>`;
}

function renderStaticPlots(plots, metricsRow) {
  const el = document.getElementById("staticPlots");
  const imgs = Object.entries(plots).filter(([, url]) => url).map(([name, url]) =>
    `<img src="${url}" title="${name}" onclick="window.open('${url}','_blank')">`).join("");
  if (imgs) {
    el.innerHTML = imgs;
    return;
  }
  // 无静态图：引导生成（统计图仅需读分片；shift 图需先评估）
  const shiftNote = metricsRow ? "" : "<br><span style='color:var(--text-dim);font-size:12px'>shift 扫描图需先完成评估（Tab③）</span>";
  el.innerHTML = `
    <p class='text-dim' style='margin-bottom:8px'>统计图尚未生成（用 matplotlib 生成 PNG，可离线存档/演示）${shiftNote}</p>
    <button class="primary-btn" onclick="triggerGenPlots()">生成静态图</button>`;
}

let genplotsTimer = null;

async function triggerGenPlots() {
  if (!state.game) return;
  try {
    const r = await fetch(`/api/genplots?game=${encodeURIComponent(state.game)}&video=${encodeURIComponent(state.video)}`, { method: "POST" });
    const j = await r.json();
    if (!j.ok) { showError(j.error || "生成失败"); return; }
    document.getElementById("staticPlots").innerHTML =
      `<div class="loading-block"><div class="spinner"></div>正在生成统计图（约 10~30 秒）…</div>`;
    pollGenplots();
  } catch (e) { showError("生成失败: " + e.message); }
}

function pollGenplots() {
  clearInterval(genplotsTimer);
  genplotsTimer = setInterval(async () => {
    let st;
    try { st = await api("/api/genplots/status"); }
    catch { return; }
    if (!st.running) {
      clearInterval(genplotsTimer);
      if (st.error) showError("静态图生成失败: " + st.error);
      else loadStats();  // 重新加载统计 Tab，静态图 URL 自动出现
    }
  }, 1500);
}

// ---------------- Tab3: 序列对比（扩展 C） ----------------
let seqData = null;
let evalTimer = null;

function renderDiffDefinition(def) {
  const el = document.getElementById("diffDefinition");
  if (!def) { document.getElementById("diffDefinitionCard").classList.add("hidden"); return; }
  document.getElementById("diffDefinitionCard").classList.remove("hidden");
  el.innerHTML = `
    <div class="dd-formula"><b>D = ${def.text}</b></div>
    <div class="dd-row"><span class="dd-k">按键分量</span>${def.button_component}</div>
    <div class="dd-row"><span class="dd-k">摇杆分量</span>${def.stick_component}</div>
    <div class="dd-row"><span class="dd-k">Top-5</span>${def.top5}</div>
    <div class="dd-row"><span class="dd-k">20 段曲线</span>${def.segments}</div>`;
}

function renderSegments(segments) {
  const el = document.getElementById("segmentsChart");
  if (!segments || !segments.length) { el.innerHTML = ""; return; }
  if (!state.charts.segments) state.charts.segments = echarts.init(el);
  // 20 段均值差异柱状图
  state.charts.segments.setOption({
    tooltip: { trigger: "axis",
      formatter: ps => {
        const s = segments[ps[0].dataIndex];
        return `段 ${s.id}: 帧 ${s.start_frame}~${s.end_frame}<br>均值差异 ${s.mean_diff}<br>最大差异帧 ${s.max_diff_frame}${s.max_diff_frame === s.points.reduce((a,b)=>a.diff_score>b.diff_score?b:a).absolute_frame ? "" : ""}`;
      } },
    grid: { left: 50, right: 20, top: 30, bottom: 30 },
    xAxis: { type: "category", name: "段 (每10帧)", data: segments.map(s => `#${s.id}`) },
    yAxis: { type: "value", name: "均值差异D" },
    series: [{
      type: "bar", data: segments.map(s => s.mean_diff),
      itemStyle: { color: (p) => p.dataIndex === 0 ? "#2563eb" : "#93c5fd", borderRadius: [3,3,0,0] },
      label: { show: false },
    }],
  }, true);
}

async function loadSequences() {
  const box = document.getElementById("seqChart");
  box.innerHTML = `<div class="loading-block"><div class="spinner"></div>加载序列数据...</div>`;
  document.querySelector("#top5Table tbody").innerHTML = "";
  try {
    seqData = await api(`/api/sequences?game=${encodeURIComponent(state.game)}&video=${encodeURIComponent(state.video)}`);
    renderDiffDefinition(seqData.diff_definition);
    renderSegments(seqData.segments);
    renderSeqChart();
    renderTop5();
  } catch (e) {
    // 未评估（predictions.csv 缺失）→ 引导运行评估；其它错误照常提示
    if (/predictions\.csv|评估/.test(e.message)) {
      renderEvalGuide();
    } else {
      showError("序列加载失败: " + e.message);
      box.innerHTML = `<div class="loading-block">${e.message}</div>`;
    }
  }
}

function renderEvalGuide() {
  const box = document.getElementById("seqChart");
  const cur = state.videos.find(v => v.video === state.video);
  const downloaded = cur && cur.status === "downloaded";
  box.innerHTML = `
    <div class="eval-guide">
      <div class="eval-guide-icon">🧪</div>
      <div class="eval-guide-body">
        <h4>该视频尚未评估，序列对比需要评估产物（predictions.csv）</h4>
        <p>评估流程：抽 200 帧测试集 → 加载 NitroGen 模型逐帧推理 → shift 扫描对齐 → 生成对比数据并入库。
           预计耗时 3~6 分钟（含模型加载），后台运行。</p>
        ${downloaded ? `
          <p class="eval-guide-warn">前置条件已满足（视频已下载）。</p>
          <button class="primary-btn" onclick="triggerEvaluate()">运行评估（约 3~6 分钟）</button>`
        : `
          <p class="eval-guide-warn">⚠ 前置条件：视频尚未下载，评估需要本地视频文件。
             请先在顶部栏点「⬇ 下载视频」下载后再回来评估。</p>`}
      </div>
    </div>`;
}

async function triggerEvaluate() {
  if (!state.game || !state.video) return;
  try {
    const r = await fetch(`/api/evaluate?game=${encodeURIComponent(state.game)}&video=${encodeURIComponent(state.video)}`, { method: "POST" });
    const j = await r.json();
    if (!j.ok) { showError(j.error || "评估启动失败"); return; }
    document.getElementById("seqChart").innerHTML =
      `<div class="loading-block"><div class="spinner"></div>评估已启动，正在后台运行…</div>`;
    showEvalLog(true);
    pollEvaluate();
  } catch (e) { showError("评估启动失败: " + e.message); }
}

function showEvalLog(show) {
  const b = document.getElementById("evalBanner");
  if (show) b.classList.remove("hidden"); else b.classList.add("hidden");
}

function pollEvaluate() {
  clearInterval(evalTimer);
  evalTimer = setInterval(async () => {
    let st;
    try { st = await api("/api/evaluate/status"); }
    catch { return; }
    // 日志尾部（评估进度）
    const logEl = document.getElementById("evalLog");
    if (st.log_tail) { logEl.textContent = st.log_tail; logEl.classList.remove("hidden"); }
    else logEl.classList.add("hidden");
    const txt = document.getElementById("evalText");
    if (st.running) {
      txt.textContent = `评估运行中（${st.game}/${st.video}）… 模型加载 + 推理约 3~6 分钟`;
    } else {
      clearInterval(evalTimer);
      if (st.stage === "done") {
        txt.textContent = "评估完成！序列对比与统计已生成并入库。";
        showEvalLog(true);
        // 刷新：序列数据 + 视频列表（tested 徽标）
        await loadGames();
        document.getElementById("gameSelect").value = state.game;
        state.gameInfo = state.games.find(g => g.game === state.game);
        await loadVideos(state.video);
        await loadSequences();
        setTimeout(() => showEvalLog(false), 6000);
      } else if (st.stage === "failed") {
        txt.textContent = "评估失败：" + (st.error || "未知错误");
        showEvalLog(true);
      }
    }
  }, 3000);
}

function renderSeqChart() {
  if (!seqData) return;
  const el = document.getElementById("seqChart");
  if (!state.charts.seq) state.charts.seq = echarts.init(el);
  const metric = document.getElementById("seqMetric").value;
  const frames = seqData.sequence.map(s => s.absolute_frame);
  let series, name;
  if (metric === "diff") {
    // 综合差异分 D：单线 + 标记 Top-5 差异帧
    const top5 = seqData.top5_mismatch.map(t => t.absolute_frame);
    name = "综合差异分 D";
    series = [{
      name: "D", type: "line", data: seqData.sequence.map(s => s.diff_score),
      symbol: "none", lineStyle: { color: "#7c3aed", width: 2 },
      markPoint: {
        data: top5.map(f => {
          const i = frames.indexOf(f);
          return { coord: [i, seqData.sequence[i].diff_score], value: "Top-5", itemStyle: { color: "#dc2626" } };
        }),
        symbolSize: 40, label: { fontSize: 8 },
      },
    }];
  } else {
    const [gk, pk] = {
      press: ["gt_n_press", "pred_n_press"],
      jlx: ["gt_jl_x", "pred_jl_x"],
      jly: ["gt_jl_y", "pred_jl_y"],
      jrx: ["gt_jr_x", "pred_jr_x"],
      jry: ["gt_jr_y", "pred_jr_y"],
    }[metric];
    name = { press: "按键数", jlx: "左摇杆 x", jly: "左摇杆 y",
             jrx: "右摇杆 x", jry: "右摇杆 y" }[metric];
    series = [
      { name: "标注真值", type: "line", data: seqData.sequence.map(s => s[gk]), symbol: "none",
        lineStyle: { color: "#2563eb", width: 2 } },
      { name: "模型预测", type: "line", data: seqData.sequence.map(s => s[pk]), symbol: "none",
        lineStyle: { color: "#ea580c", width: 1.5, type: "dashed" } },
    ];
  }

  state.charts.seq.setOption({
    tooltip: { trigger: "axis" },
    legend: { top: 0 },
    grid: { left: 50, right: 20, top: 34, bottom: 50 },
    dataZoom: [{ type: "inside" }, { type: "slider", height: 18, bottom: 8 }],
    xAxis: { type: "category", data: frames, name: "absolute_frame" },
    yAxis: { type: "value", name },
    series,
  }, true);
}

function renderTop5() {
  const tbody = document.querySelector("#top5Table tbody");
  if (!seqData || !seqData.top5_mismatch.length) {
    tbody.innerHTML = `<tr><td colspan="10" style="text-align:center;color:var(--text-dim)">无差异帧数据</td></tr>`;
    return;
  }
  tbody.innerHTML = seqData.top5_mismatch.map(t => {
    const diffTags = Object.entries(t.diff).map(([k, v]) =>
      `<span class="diff-tag ${v.gt ? "gt1" : "gt0"}">${k}:${v.gt ? "按→没按" : "没按→按"}</span>`).join("");
    // 秒数取该帧对应 sequence 里的 second（接口已返回），避免依赖未赋值的 state._fps
    const row = seqData.sequence.find(s => s.absolute_frame === t.absolute_frame);
    const sec = row ? row.second.toFixed(1) : "—";
    return `<tr>
      <td class="num">${t.absolute_frame}</td>
      <td class="num">${sec}s</td>
      <td class="num"><b style="color:var(--red)">${t.n_mismatch}</b></td>
      <td class="num" title="左/右摇杆 L2">${t.jl_l2 != null ? t.jl_l2.toFixed(3) : "—"} / ${t.jr_l2 != null ? t.jr_l2.toFixed(3) : "—"}</td>
      <td class="num"><b style="color:var(--orange)">${t.diff_score != null ? t.diff_score.toFixed(3) : "—"}</b></td>
      <td class="num">${t.gt_n_press}</td>
      <td class="num">${t.pred_n_press}</td>
      <td>${diffTags || "—"}</td>
      <td>${t.is_idle ? "IDLE" : ""}</td>
      <td><button class="jump-btn" onclick="jumpToFrame(${t.absolute_frame})">查看 ▶</button></td>
    </tr>`;
  }).join("");
}

async function jumpToFrame(frame) {
  switchTab("tab-recog");
  const idx = state.frames.findIndex(f => f.absolute_frame === frame);
  if (idx >= 0) await selectFrame(idx);
  else showError(`帧 ${frame} 不在测试集帧列表中`);
}

// ---------------- 自动提取流水线（选未提取游戏 -> 确认耗时 -> extract + probe） ----------------
let extractTimer = null;
let _estShown = false;

function updateExtractBanner(st) {
  const banner = document.getElementById("extractBanner");
  const info = document.getElementById("extractInfo");
  const btn = document.getElementById("extractBtn");
  const logEl = document.getElementById("extractLog");
  const icon = document.getElementById("extractIcon");
  if (!state.game || !state.gameInfo) { banner.classList.add("hidden"); return; }

  if (state.gameInfo.extracted && !st) { banner.classList.add("hidden"); return; }

  banner.classList.remove("hidden");
  banner.className = "extract-banner";

  if (!st) {  // 未提取：先展示耗时估算，用户确认后才开始（仅读取本地分片）
    info.textContent = `「${state.game}」在切片清单中，但标注尚未提取，数据分析不可用。正在估算耗时…`;
    btn.textContent = "读取本地分片";
    btn.disabled = true; btn.style.display = "";
    logEl.classList.add("hidden");
    icon.textContent = "📦";
    if (!_estShown) {
      _estShown = true;
      fetch(`/api/extract/estimate?game=${encodeURIComponent(state.game)}`)
        .then(r => r.json())
        .then(j => {
          if (j.ok && j.data.game === state.game) {
            info.textContent = `「${state.game}」标注未提取。读取本地分片需处理 ${j.data.chunks} 个 chunk，预计耗时 ${j.data.est_text}（纯本地、不联网）。确定开始吗？`;
            btn.disabled = false;
          }
        })
        .catch(() => { btn.disabled = false; });
    }
  } else {
    _estShown = true;
    const stageText = { extracting: "正在读取本地分片（extract_game.py，不联网）…",
                        done: "分片读取完成，标注已就绪，可进行数据分析。",
                        cancelled: "已停止（产物可能不完整，可重新读取）。",
                        failed: "读取失败" }[st.stage] || st.stage;
    info.textContent = `「${st.game}」${stageText}`;
    if (st.running) {
      banner.classList.add("running");
      btn.textContent = "停止";
      btn.disabled = false; btn.style.display = "";
      icon.textContent = "⏳";
    } else {
      banner.classList.add(st.stage === "done" ? "done" : st.stage === "failed" ? "failed" : "");
      btn.textContent = "重新读取";
      btn.style.display = (st.stage === "done" && state.gameInfo.extracted) ? "none" : "";
      btn.disabled = false;
      icon.textContent = st.stage === "done" ? "✅" : "⛔";
      if (st.stage === "failed") info.textContent += `  ${st.error || ""}`;
    }
    if (st.log_tail) { logEl.textContent = st.log_tail; logEl.classList.remove("hidden"); }
    else logEl.classList.add("hidden");
  }
}

async function triggerExtract() {
  if (!state.game) return;
  const btn = document.getElementById("extractBtn");
  const stopping = btn.textContent === "停止";
  try {
    if (stopping) {
      const r = await fetch("/api/extract/cancel", { method: "POST" });
      const j = await r.json();
      if (!j.ok) showError(j.error || "停止失败");
      else { clearInterval(extractTimer); await refreshBanner("cancelled"); }
      return;
    }
    const r = await fetch(`/api/extract?game=${encodeURIComponent(state.game)}`, { method: "POST" });
    const j = await r.json();
    if (!j.ok) { showError(j.error || "启动失败"); return; }
    btn.disabled = true; btn.textContent = "启动中…";
    pollExtract();
  } catch (e) { showError("操作失败: " + e.message); }
}

async function refreshBanner(stage) {
  const st = { game: state.game, stage, running: false, log_tail: "", error: "" };
  updateExtractBanner(st);
  await loadGames();
  document.getElementById("gameSelect").value = state.game;
  state.gameInfo = state.games.find(g => g.game === state.game);
  updateExtractBanner(st);
}

function pollExtract() {
  clearInterval(extractTimer);
  extractTimer = setInterval(async () => {
    let st;
    try { st = await api("/api/extract/status"); }
    catch { clearInterval(extractTimer); return; }
    updateExtractBanner(st);
    if (!st.running) {
      clearInterval(extractTimer);
      if (st.stage === "done" || st.stage === "failed") {
        // 刷新游戏列表（extracted 标记）+ 该游戏视频列表
        await loadGames();
        document.getElementById("gameSelect").value = state.game;
        state.gameInfo = state.games.find(g => g.game === state.game);
        await loadVideos();
        updateExtractBanner(st);
      } else if (st.stage === "cancelled") {
        await refreshBanner("cancelled");
      }
    }
  }, 2500);
}

// ---------------- 视频下载 ----------------
let downloadTimer = null;

function showDownloadBanner(st) {
  const b = document.getElementById("downloadBanner");
  const txt = document.getElementById("downloadText");
  const cbtn = document.getElementById("downloadCancelBtn");
  if (!st) { b.classList.add("hidden"); cbtn.style.display = "none"; return; }
  b.classList.remove("hidden");
  b.className = "download-banner" + (st.running ? " running" : st.pct === 100 ? " done" : "");
  cbtn.style.display = st.running ? "" : "none";
  document.getElementById("downloadBar").style.width = (st.pct || 0) + "%";
  if (st.running) txt.textContent = `下载中：${(st.pct || 0).toFixed(1)}%  ${st.msg || ""}`;
  else if (st.pct === 100) txt.textContent = `下载完成：data/videos/${st.game}_${st.video}.mp4`;
  else if (st.error) txt.textContent = `下载失败：${st.error}`;
}

async function triggerDownload() {
  if (!state.game || !state.video) { showError("请先选择视频"); return; }
  // 已下载的视频提示即可，不再重复下载
  const cur = state.videos.find(v => v.video === state.video);
  if (cur && cur.status === "downloaded") {
    showError(`该视频已下载（data/videos/${state.game}_${state.video}.mp4），可直接用于模型识别与评估。`);
    return;
  }
  const ok = confirm(`确认下载 ${state.game}/${state.video} ？视频文件较大，下载需几分钟到几十分钟（后台进行，可继续浏览）。`);
  if (!ok) return;
  try {
    const r = await fetch(`/api/download?game=${encodeURIComponent(state.game)}&video=${encodeURIComponent(state.video)}`, { method: "POST" });
    const j = await r.json();
    if (!j.ok) { showError(j.error || "下载启动失败"); return; }
    pollDownload();
  } catch (e) { showError("下载启动失败: " + e.message); }
}

async function cancelDownload() {
  try {
    await fetch("/api/download/cancel", { method: "POST" });
    showError("已请求停止下载（稍后自动清理未完成文件）。");
    clearInterval(downloadTimer);
    showDownloadBanner(null);
  } catch (e) { showError("停止失败: " + e.message); }
}

function pollDownload() {
  clearInterval(downloadTimer);
  downloadTimer = setInterval(async () => {
    let st;
    try { st = await api("/api/download/status"); }
    catch { clearInterval(downloadTimer); return; }
    showDownloadBanner(st);
    if (!st.running) {
      clearInterval(downloadTimer);
      if (st.pct === 100) await loadVideos(state.video);
      else if (!st.video) showDownloadBanner(null);
    }
  }, 2000);
}

// ---------------- 视频探测 ----------------
async function triggerRescan() {
  const isFull = !state.game;
  if (isFull) {
    const ok = confirm(
      "将全量探测切片清单中 311 个视频链接，预计 10~20 分钟（后台运行，期间可继续使用）。\n\n" +
      "提示：先选中某个游戏再点「探测链接」，会只探测该游戏的视频（快）。\n\n确定继续吗？");
    if (!ok) return;
  }
  const scope = state.game ? `game=${state.game}` : "full";
  try {
    const r = await fetch(`/api/rescan?scope=${encodeURIComponent(scope)}`, { method: "POST" });
    const j = await r.json();
    if (!j.ok) { showError(j.error || "启动失败"); return; }
    showRescanBanner(true, { text: "探测已启动，正在扫描视频链接…", pct: 2, running: true });
    pollRescan();
  } catch (e) { showError("探测启动失败: " + e.message); }
}

function showRescanBanner(show, { text, pct, running, done } = {}) {
  const b = document.getElementById("rescanBanner");
  const cbtn = document.getElementById("rescanCancelBtn");
  if (!show) { b.classList.add("hidden"); cbtn.style.display = "none"; return; }
  b.classList.remove("hidden");
  b.className = "rescan-banner" + (running ? " running" : done ? " done" : "");
  document.getElementById("rescanText").textContent = text;
  document.getElementById("rescanBar").style.width = (pct || 0) + "%";
  cbtn.textContent = done ? "关闭" : "停止";
  cbtn.style.display = (running || done) ? "" : "none";
}

async function cancelRescan() {
  // 已完成态下按钮是「关闭」：直接收起横幅
  const btn = document.getElementById("rescanCancelBtn");
  if (btn.textContent === "关闭") { showRescanBanner(false); return; }
  try {
    const r = await fetch("/api/rescan/cancel", { method: "POST" });
    const j = await r.json();
    if (!j.ok) { showError(j.error || "停止失败"); return; }
    showRescanBanner(false);
    showError("已停止探测。");
    if (state.game) await loadVideos(state.video);
  } catch (e) { showError("停止失败: " + e.message); }
}

function pollRescan() {
  const t = setInterval(async () => {
    let d;
    try { d = await api("/api/rescan/status"); }
    catch (e) {
      // 轮询失败不静默：明确提示，避免用户以为没结果
      clearInterval(t);
      showError("探测状态获取失败: " + (e.message || "网络错误"));
      showRescanBanner(false);
      return;
    }

    if (d.running) {
      // 进度条：已探测 / 总数
      const pct = (d.total && d.done) ? Math.min(98, Math.round(d.done / d.total * 100)) : 5;
      const scopeText = d.scope === "full" ? "全量" : (d.scope ? d.scope.replace("game=", "") : "");
      showRescanBanner(true, {
        text: `正在探测${scopeText ? `「${scopeText}」` : ""}视频链接：已探测 ${d.done ?? "?"}/${d.total ?? "?"} 个` +
              `（${d.status_counts.downloaded ?? 0} 已下载 / ${d.status_counts.available ?? 0} 可下载 / ${d.status_counts.dead ?? 0} 失效），后台运行中…`,
        pct, running: true,
      });
      return;
    }

    // 探测结束：绿色横幅常驻显示结果，不再用 6 秒自动消失的错误横幅
    clearInterval(t);
    const counts = d.status_counts || {};
    const nUnknown = counts.unknown || 0;
    const warn = nUnknown ? `（${nUnknown} 个状态未知，多为网络不通，非视频失效）` : "";
    showRescanBanner(true, {
      text: `探测完成：共 ${d.done ?? 0} 个链接（已下载 ${counts.downloaded ?? 0} / 可下载 ${counts.available ?? 0} / 失效 ${counts.dead ?? 0} / 未知 ${nUnknown}）${warn}`,
      pct: 100, running: false, done: true,
    });
    if (state.game) await loadVideos(state.video);
  }, 3000);
}

window.addEventListener("resize", () => Object.values(state.charts).forEach(c => c && c.resize()));
