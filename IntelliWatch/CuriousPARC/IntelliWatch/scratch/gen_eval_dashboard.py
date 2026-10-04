"""
Generates frontend/eval-dashboard.html — IntelliWatch AI Performance Validation Dashboard
"""
import os

html = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>IntelliWatch — AI Performance Validation Dashboard</title>
  <meta name="description" content="Real measured performance metrics for the IntelliWatch AI pipeline, evaluated on annotated ground-truth data."/>
  <link rel="preconnect" href="https://fonts.googleapis.com"/>
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin/>
  <link href="https://fonts.googleapis.com/css2?family=Barlow+Condensed:wght@400;600;700;800&family=DM+Sans:wght@300;400;500;600&family=JetBrains+Mono:wght@400;500&display=swap" rel="stylesheet"/>
  <link rel="stylesheet" href="/static/css/dashboard.css"/>
  <style>
    /* ── Eval Dashboard — page-level overrides ─────────────────────────── */
    :root {
      --eval-accent: #6c63ff;
      --eval-green: #22c55e;
      --eval-amber: #f59e0b;
      --eval-red: #ef4444;
      --eval-blue: #3b82f6;
      --eval-bg-card: rgba(255,255,255,0.035);
      --eval-border: rgba(255,255,255,0.08);
    }

    * { box-sizing: border-box; margin: 0; padding: 0; }

    body {
      background: #080c14;
      color: #e2e8f0;
      font-family: 'DM Sans', sans-serif;
      min-height: 100vh;
    }

    /* top bar */
    .eval-topbar {
      display: flex;
      align-items: center;
      gap: 1rem;
      padding: 1rem 2rem;
      border-bottom: 1px solid var(--eval-border);
      background: rgba(8,12,20,0.95);
      position: sticky;
      top: 0;
      z-index: 100;
      backdrop-filter: blur(12px);
    }
    .eval-topbar .logo {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.4rem;
      font-weight: 800;
      background: linear-gradient(135deg,#6c63ff,#22d3ee);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      background-clip: text;
      letter-spacing: .04em;
    }
    .eval-topbar .sep { color: rgba(255,255,255,.2); font-size: 1.2rem; }
    .eval-topbar .page-title {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.15rem;
      font-weight: 600;
      color: rgba(255,255,255,.7);
      letter-spacing: .04em;
    }
    .eval-topbar .spacer { flex: 1; }
    .eval-topbar a.back-btn {
      font-size: .8rem;
      font-family: 'DM Sans', sans-serif;
      color: rgba(255,255,255,.5);
      text-decoration: none;
      padding: .4rem .9rem;
      border: 1px solid rgba(255,255,255,.12);
      border-radius: 6px;
      transition: all .2s;
    }
    .eval-topbar a.back-btn:hover { color:#fff; border-color:rgba(255,255,255,.3); }

    /* page layout */
    .eval-page {
      max-width: 1280px;
      margin: 0 auto;
      padding: 2rem 1.5rem 4rem;
    }

    /* section headers */
    .section-header {
      margin: 2.5rem 0 1.25rem;
      display: flex;
      align-items: center;
      gap: .75rem;
    }
    .section-header h2 {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 1.3rem;
      font-weight: 700;
      letter-spacing: .06em;
      text-transform: uppercase;
      color: #fff;
    }
    .section-header .pill {
      font-family: 'JetBrains Mono', monospace;
      font-size: .65rem;
      padding: .2rem .55rem;
      border-radius: 4px;
      background: rgba(108,99,255,.15);
      color: var(--eval-accent);
      border: 1px solid rgba(108,99,255,.3);
    }
    .section-divider {
      flex: 1;
      height: 1px;
      background: var(--eval-border);
    }

    /* integrity banner */
    .integrity-banner {
      display: flex;
      align-items: flex-start;
      gap: 1rem;
      padding: 1rem 1.25rem;
      border-radius: 10px;
      background: rgba(34,197,94,.07);
      border: 1px solid rgba(34,197,94,.25);
      margin-bottom: 2rem;
    }
    .integrity-banner .icon { font-size: 1.4rem; line-height: 1; }
    .integrity-banner .text p:first-child {
      font-weight: 600;
      color: var(--eval-green);
      font-size: .9rem;
      margin-bottom: .2rem;
    }
    .integrity-banner .text p:last-child {
      font-size: .78rem;
      color: rgba(255,255,255,.5);
      line-height: 1.5;
    }

    /* stat cards */
    .stat-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px,1fr));
      gap: 1rem;
    }
    .stat-card {
      background: var(--eval-bg-card);
      border: 1px solid var(--eval-border);
      border-radius: 12px;
      padding: 1.25rem 1.5rem;
      position: relative;
      overflow: hidden;
      transition: border-color .2s, transform .2s;
    }
    .stat-card:hover { border-color: rgba(108,99,255,.35); transform: translateY(-2px); }
    .stat-card::before {
      content: '';
      position: absolute;
      inset: 0;
      background: linear-gradient(135deg, rgba(108,99,255,.06), transparent 60%);
      pointer-events: none;
    }
    .stat-card .label {
      font-size: .72rem;
      text-transform: uppercase;
      letter-spacing: .08em;
      color: rgba(255,255,255,.4);
      font-family: 'Barlow Condensed', sans-serif;
      font-weight: 600;
      margin-bottom: .5rem;
    }
    .stat-card .value {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: 2.4rem;
      font-weight: 800;
      line-height: 1;
      margin-bottom: .3rem;
    }
    .stat-card .value.green { color: var(--eval-green); }
    .stat-card .value.amber { color: var(--eval-amber); }
    .stat-card .value.blue  { color: var(--eval-blue); }
    .stat-card .value.purple { color: var(--eval-accent); }
    .stat-card .sub {
      font-size: .75rem;
      color: rgba(255,255,255,.35);
      font-family: 'JetBrains Mono', monospace;
    }
    .stat-card .loading-shimmer {
      height: 2.4rem;
      background: linear-gradient(90deg,rgba(255,255,255,.05) 25%,rgba(255,255,255,.1) 50%,rgba(255,255,255,.05) 75%);
      background-size: 200% 100%;
      animation: shimmer 1.5s infinite;
      border-radius: 6px;
    }
    @keyframes shimmer { to { background-position: -200% 0; } }

    /* class table */
    .class-table-wrap {
      background: var(--eval-bg-card);
      border: 1px solid var(--eval-border);
      border-radius: 12px;
      overflow: hidden;
    }
    .class-table {
      width: 100%;
      border-collapse: collapse;
      font-size: .82rem;
    }
    .class-table thead tr {
      background: rgba(108,99,255,.08);
      border-bottom: 1px solid var(--eval-border);
    }
    .class-table th {
      text-align: left;
      padding: .75rem 1rem;
      font-family: 'Barlow Condensed', sans-serif;
      font-size: .78rem;
      font-weight: 700;
      text-transform: uppercase;
      letter-spacing: .06em;
      color: rgba(255,255,255,.5);
    }
    .class-table td {
      padding: .7rem 1rem;
      border-bottom: 1px solid rgba(255,255,255,.04);
      font-family: 'JetBrains Mono', monospace;
      font-size: .8rem;
    }
    .class-table tr:last-child td { border-bottom: none; }
    .class-table tr:hover td { background: rgba(255,255,255,.02); }
    .class-table .bar-cell { width: 30%; }
    .perf-bar-bg {
      height: 6px;
      background: rgba(255,255,255,.06);
      border-radius: 3px;
      overflow: hidden;
    }
    .perf-bar {
      height: 100%;
      border-radius: 3px;
      transition: width 1s ease;
    }
    .badge {
      display: inline-block;
      padding: .2rem .5rem;
      border-radius: 4px;
      font-size: .65rem;
      font-family: 'Barlow Condensed', sans-serif;
      font-weight: 700;
      letter-spacing: .04em;
    }
    .badge.high { background:rgba(34,197,94,.15); color:var(--eval-green); }
    .badge.mid  { background:rgba(245,158,11,.15); color:var(--eval-amber); }
    .badge.low  { background:rgba(239,68,68,.15);  color:var(--eval-red);   }

    /* confusion matrix */
    .cm-grid-wrap {
      background: var(--eval-bg-card);
      border: 1px solid var(--eval-border);
      border-radius: 12px;
      padding: 1.5rem;
    }
    .cm-grid {
      display: grid;
      grid-template-columns: auto repeat(var(--cols),1fr);
      gap: 3px;
      max-width: 640px;
    }
    .cm-cell {
      display: flex;
      align-items: center;
      justify-content: center;
      padding: .55rem .4rem;
      border-radius: 4px;
      font-family: 'JetBrains Mono', monospace;
      font-size: .72rem;
      min-width: 52px;
      min-height: 38px;
    }
    .cm-cell.header { color:rgba(255,255,255,.4); font-size:.65rem; font-family:'Barlow Condensed',sans-serif; font-weight:600; background:transparent; }
    .cm-cell.row-label { color:rgba(255,255,255,.4); font-size:.65rem; font-family:'Barlow Condensed',sans-serif; font-weight:600; background:transparent; justify-content:flex-end; padding-right:.6rem; }
    .cm-legend { margin-top: 1rem; font-size: .72rem; color: rgba(255,255,255,.35); }

    /* benchmark table */
    .bench-table-wrap {
      background: var(--eval-bg-card);
      border: 1px solid var(--eval-border);
      border-radius: 12px;
      overflow: hidden;
    }
    .bench-table { width: 100%; border-collapse: collapse; font-size: .82rem; }
    .bench-table thead tr { background: rgba(108,99,255,.08); border-bottom: 1px solid var(--eval-border); }
    .bench-table th {
      text-align: left; padding: .75rem 1rem;
      font-family: 'Barlow Condensed', sans-serif; font-size: .78rem;
      font-weight: 700; text-transform: uppercase; letter-spacing: .06em;
      color: rgba(255,255,255,.5);
    }
    .bench-table td {
      padding: .7rem 1rem; border-bottom: 1px solid rgba(255,255,255,.04);
      font-family: 'JetBrains Mono', monospace; font-size: .8rem;
    }
    .bench-table tr:last-child td { border-bottom: none; }

    /* diagnostics grid */
    .diag-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px,1fr));
      gap: 1rem;
    }
    .diag-card {
      background: var(--eval-bg-card);
      border: 1px solid var(--eval-border);
      border-radius: 12px;
      padding: 1.25rem;
    }
    .diag-card h3 {
      font-family: 'Barlow Condensed', sans-serif;
      font-size: .95rem; font-weight: 700; text-transform: uppercase;
      letter-spacing: .06em; color: rgba(255,255,255,.7); margin-bottom: .75rem;
    }
    .kv-list { list-style: none; }
    .kv-list li {
      display: flex; justify-content: space-between; align-items: center;
      padding: .35rem 0; border-bottom: 1px solid rgba(255,255,255,.04);
      font-size: .78rem;
    }
    .kv-list li:last-child { border-bottom: none; }
    .kv-list .k { color: rgba(255,255,255,.4); font-family: 'DM Sans', sans-serif; }
    .kv-list .v { font-family: 'JetBrains Mono', monospace; color: #e2e8f0; }

    /* unavail state */
    .unavail-notice {
      display: flex; align-items: center; gap: .75rem; padding: .9rem 1.1rem;
      border-radius: 8px; background: rgba(245,158,11,.07);
      border: 1px solid rgba(245,158,11,.2); margin: .5rem 0;
      font-size: .78rem; color: rgba(255,255,255,.45);
    }
    .unavail-notice .icon { font-size: 1rem; color: var(--eval-amber); flex-shrink: 0; }

    /* refresh row */
    .refresh-row {
      display: flex; align-items: center; gap: 1rem;
      margin-bottom: 1.5rem;
    }
    .refresh-btn {
      display: flex; align-items: center; gap: .5rem;
      background: rgba(108,99,255,.12); border: 1px solid rgba(108,99,255,.3);
      color: var(--eval-accent); padding: .5rem 1rem; border-radius: 7px;
      font-family: 'DM Sans', sans-serif; font-size: .82rem; font-weight: 500;
      cursor: pointer; transition: all .2s;
    }
    .refresh-btn:hover { background: rgba(108,99,255,.22); }
    .refresh-btn.spinning svg { animation: spin .8s linear infinite; }
    @keyframes spin { to { transform: rotate(360deg); } }
    .last-updated {
      font-family: 'JetBrains Mono', monospace; font-size: .72rem;
      color: rgba(255,255,255,.3);
    }

    /* source badge */
    .source-tag {
      font-family: 'JetBrains Mono', monospace; font-size: .65rem;
      color: rgba(255,255,255,.25); margin-top: .25rem;
    }
  </style>
</head>
<body>

<!-- Top Bar -->
<nav class="eval-topbar">
  <span class="logo">IntelliWatch</span>
  <span class="sep">›</span>
  <span class="page-title">AI Performance Validation</span>
  <div class="spacer"></div>
  <a href="/" class="back-btn">← Back to Dashboard</a>
</nav>

<main class="eval-page">

  <!-- Integrity Banner -->
  <div class="integrity-banner">
    <div class="icon">🔬</div>
    <div class="text">
      <p>All metrics are sourced from actual evaluated inference runs</p>
      <p>Numbers displayed below come exclusively from <code>reports/person_ppe_evaluation.json</code> and <code>reports/model_benchmark.json</code> via <code>/api/v1/model/evaluation</code> and <code>/api/v1/model/benchmark</code>. No values are hardcoded, estimated, or fabricated.</p>
    </div>
  </div>

  <!-- Refresh Control -->
  <div class="refresh-row">
    <button class="refresh-btn" id="refreshBtn" onclick="loadAllData()">
      <svg id="refreshIcon" width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5">
        <path d="M23 4v6h-6M1 20v-6h6"/>
        <path d="M3.51 9a9 9 0 0114.85-3.36L23 10M1 14l4.64 4.36A9 9 0 0020.49 15"/>
      </svg>
      Refresh from API
    </button>
    <span class="last-updated" id="lastUpdated">Fetching data…</span>
  </div>

  <!-- SECTION 1: Core Performance Summary -->
  <div class="section-header">
    <h2>Core Performance</h2>
    <span class="pill">evaluation</span>
    <div class="section-divider"></div>
  </div>

  <div class="stat-grid" id="coreStats">
    <div class="stat-card"><div class="label">mAP@0.5</div><div class="loading-shimmer"></div></div>
    <div class="stat-card"><div class="label">Precision</div><div class="loading-shimmer"></div></div>
    <div class="stat-card"><div class="label">Recall</div><div class="loading-shimmer"></div></div>
    <div class="stat-card"><div class="label">F1 Score</div><div class="loading-shimmer"></div></div>
    <div class="stat-card"><div class="label">Images Evaluated</div><div class="loading-shimmer"></div></div>
  </div>

  <!-- SECTION 2: Per-Class Metrics -->
  <div class="section-header">
    <h2>Per-Class Metrics</h2>
    <span class="pill">per-class</span>
    <div class="section-divider"></div>
  </div>

  <div class="class-table-wrap">
    <table class="class-table">
      <thead>
        <tr>
          <th>Class</th>
          <th>Precision</th>
          <th>Recall</th>
          <th>mAP@0.5</th>
          <th>F1</th>
          <th class="bar-cell">AP Bar</th>
          <th>Grade</th>
        </tr>
      </thead>
      <tbody id="classTableBody">
        <tr><td colspan="7" style="text-align:center;padding:1.5rem;color:rgba(255,255,255,.3);font-size:.8rem;">Loading per-class data…</td></tr>
      </tbody>
    </table>
  </div>

  <!-- SECTION 3: Confusion Matrix -->
  <div class="section-header">
    <h2>Confusion Matrix</h2>
    <span class="pill">evaluation</span>
    <div class="section-divider"></div>
  </div>

  <div class="cm-grid-wrap" id="cmSection">
    <div class="unavail-notice">
      <span class="icon">⏳</span>
      <span>Loading confusion matrix from evaluation report…</span>
    </div>
  </div>

  <!-- SECTION 4: Benchmark -->
  <div class="section-header">
    <h2>Inference Benchmark</h2>
    <span class="pill">benchmark</span>
    <div class="section-divider"></div>
  </div>

  <div class="bench-table-wrap">
    <table class="bench-table">
      <thead>
        <tr>
          <th>Input Mode</th>
          <th>Avg Latency</th>
          <th>Throughput</th>
          <th>Preprocess</th>
          <th>Inference</th>
          <th>Postprocess</th>
          <th>Device</th>
        </tr>
      </thead>
      <tbody id="benchTableBody">
        <tr><td colspan="7" style="text-align:center;padding:1.5rem;color:rgba(255,255,255,.3);font-size:.8rem;">Loading benchmark data…</td></tr>
      </tbody>
    </table>
  </div>

  <!-- SECTION 5: Model Diagnostics -->
  <div class="section-header">
    <h2>Model Diagnostics</h2>
    <span class="pill">live</span>
    <div class="section-divider"></div>
  </div>

  <div class="diag-grid" id="diagGrid">
    <div class="diag-card">
      <h3>Loading…</h3>
      <div class="loading-shimmer" style="height:120px;border-radius:8px;"></div>
    </div>
  </div>

</main>

<script>
// ── Helpers ──────────────────────────────────────────────────────────────────

function pct(v, decimals = 1) {
  if (v == null || isNaN(v)) return 'N/A';
  return (v * 100).toFixed(decimals) + '%';
}

function ms(v) {
  if (v == null || isNaN(v)) return 'N/A';
  return v.toFixed(1) + ' ms';
}

function grade(ap) {
  if (ap == null) return '<span class="badge mid">N/A</span>';
  if (ap >= 0.75) return '<span class="badge high">HIGH</span>';
  if (ap >= 0.45) return '<span class="badge mid">MID</span>';
  return '<span class="badge low">LOW</span>';
}

function barColor(ap) {
  if (ap == null) return '#555';
  if (ap >= 0.75) return 'var(--eval-green)';
  if (ap >= 0.45) return 'var(--eval-amber)';
  return 'var(--eval-red)';
}

function colorValue(v, threshHigh = 0.75, threshMid = 0.45) {
  if (v == null) return '';
  if (v >= threshHigh) return 'green';
  if (v >= threshMid)  return 'amber';
  return 'low';
}

function setStatCard(id, label, value, cssClass, sub) {
  const el = document.getElementById(id);
  if (!el) return;
  el.innerHTML = `
    <div class="label">${label}</div>
    <div class="value ${cssClass}">${value}</div>
    <div class="sub">${sub || ''}</div>
  `;
}

// ── Core Stats ───────────────────────────────────────────────────────────────

function renderCoreStats(data) {
  const metrics = data.metrics || data;
  const summary = metrics.summary || metrics;

  const map50   = summary.mAP50   ?? summary.map50   ?? summary.map   ?? null;
  const prec    = summary.precision ?? null;
  const recall  = summary.recall  ?? null;
  const f1      = summary.f1_score ?? summary.f1 ?? null;
  const nImages = metrics.images_evaluated ?? data.images_evaluated ?? null;

  const grid = document.getElementById('coreStats');
  grid.innerHTML = `
    <div class="stat-card">
      <div class="label">mAP@0.5</div>
      <div class="value ${colorValue(map50)}">${map50 != null ? pct(map50) : '<span style="color:rgba(255,255,255,.3);font-size:1.2rem">N/A</span>'}</div>
      <div class="source-tag">mean Average Precision</div>
    </div>
    <div class="stat-card">
      <div class="label">Precision</div>
      <div class="value ${colorValue(prec)}">${prec != null ? pct(prec) : '<span style="color:rgba(255,255,255,.3);font-size:1.2rem">N/A</span>'}</div>
      <div class="source-tag">TP / (TP+FP)</div>
    </div>
    <div class="stat-card">
      <div class="label">Recall</div>
      <div class="value ${colorValue(recall)}">${recall != null ? pct(recall) : '<span style="color:rgba(255,255,255,.3);font-size:1.2rem">N/A</span>'}</div>
      <div class="source-tag">TP / (TP+FN)</div>
    </div>
    <div class="stat-card">
      <div class="label">F1 Score</div>
      <div class="value ${colorValue(f1)}">${f1 != null ? pct(f1) : '<span style="color:rgba(255,255,255,.3);font-size:1.2rem">N/A</span>'}</div>
      <div class="source-tag">harmonic mean P/R</div>
    </div>
    <div class="stat-card">
      <div class="label">Images Evaluated</div>
      <div class="value blue">${nImages != null ? nImages : 'N/A'}</div>
      <div class="source-tag">ground-truth test set</div>
    </div>
  `;
}

// ── Per-Class Table ──────────────────────────────────────────────────────────

function renderClassTable(data) {
  const metrics = data.metrics || data;
  const classes = metrics.per_class || metrics.classes || null;

  const tbody = document.getElementById('classTableBody');
  if (!classes || !Array.isArray(classes) || classes.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7">
      <div class="unavail-notice" style="margin:.5rem 0;">
        <span class="icon">⚠️</span>
        <span>Per-class data not available in evaluation report. Run <code>scripts/evaluate_ppe_model.py</code> to generate it.</span>
      </div>
    </td></tr>`;
    return;
  }

  tbody.innerHTML = classes.map(cls => {
    const ap = cls.ap ?? cls.mAP50 ?? null;
    const p  = cls.precision ?? null;
    const r  = cls.recall    ?? null;
    const f  = cls.f1        ?? (p != null && r != null ? 2*p*r/(p+r+1e-9) : null);
    const pctAp = ap != null ? ap * 100 : 0;
    return `
      <tr>
        <td><strong>${cls.name || cls.class_name || 'Unknown'}</strong></td>
        <td>${p  != null ? pct(p)  : '—'}</td>
        <td>${r  != null ? pct(r)  : '—'}</td>
        <td>${ap != null ? pct(ap) : '—'}</td>
        <td>${f  != null ? pct(f)  : '—'}</td>
        <td class="bar-cell">
          <div class="perf-bar-bg">
            <div class="perf-bar" style="width:${pctAp.toFixed(1)}%;background:${barColor(ap)};"></div>
          </div>
        </td>
        <td>${grade(ap)}</td>
      </tr>
    `;
  }).join('');
}

// ── Confusion Matrix ─────────────────────────────────────────────────────────

function renderConfusionMatrix(data) {
  const metrics = data.metrics || data;
  const cm = metrics.confusion_matrix ?? null;
  const labels = metrics.class_names ?? metrics.classes?.map(c => c.name) ?? null;

  const section = document.getElementById('cmSection');
  if (!cm || !Array.isArray(cm) || cm.length === 0) {
    section.innerHTML = `<div class="unavail-notice">
      <span class="icon">⚠️</span>
      <span>Confusion matrix not present in evaluation report. Re-run <code>scripts/evaluate_ppe_model.py --save-confusion-matrix</code>.</span>
    </div>`;
    return;
  }

  const n = cm.length;
  const cols = labels ? labels.length : n;
  const maxVal = Math.max(...cm.flat().filter(v => typeof v === 'number'));

  const colHeaders = labels || cm[0].map((_, i) => `C${i}`);

  let gridHtml = `<div class="cm-grid" style="--cols:${cols};">`;

  // corner
  gridHtml += `<div class="cm-cell header"></div>`;
  // col headers
  colHeaders.forEach(l => {
    gridHtml += `<div class="cm-cell header">${l}</div>`;
  });

  cm.forEach((row, ri) => {
    const rowLabel = labels ? labels[ri] : `C${ri}`;
    gridHtml += `<div class="cm-cell row-label">${rowLabel}</div>`;
    row.forEach((val, ci) => {
      const intensity = maxVal > 0 ? val / maxVal : 0;
      const isDiag = ri === ci;
      const bg = isDiag
        ? `rgba(34,197,94,${0.1 + intensity * 0.65})`
        : `rgba(239,68,68,${intensity * 0.5})`;
      gridHtml += `<div class="cm-cell" style="background:${bg};">${val}</div>`;
    });
  });

  gridHtml += `</div>`;
  gridHtml += `<p class="cm-legend">Rows = Actual, Columns = Predicted. Diagonal cells (green) are correct predictions.</p>`;
  section.innerHTML = gridHtml;
}

// ── Benchmark Table ──────────────────────────────────────────────────────────

function renderBenchmark(data) {
  const tbody = document.getElementById('benchTableBody');
  const results = data.results ?? data.benchmarks ?? data;

  if (!results || (Array.isArray(results) && results.length === 0)) {
    tbody.innerHTML = `<tr><td colspan="7">
      <div class="unavail-notice" style="margin:.5rem 0;">
        <span class="icon">⚠️</span>
        <span>No benchmark results available. Run <code>scripts/benchmark_model.py</code> to generate <code>reports/model_benchmark.json</code>.</span>
      </div>
    </td></tr>`;
    return;
  }

  const rows = Array.isArray(results) ? results : [results];
  tbody.innerHTML = rows.map(r => {
    const mode   = r.mode ?? r.input_type ?? r.type ?? 'N/A';
    const latency = r.avg_latency_ms ?? r.latency_ms ?? null;
    const fps    = r.throughput_fps ?? r.fps ?? null;
    const pre    = r.preprocess_ms  ?? null;
    const inf    = r.inference_ms   ?? null;
    const post   = r.postprocess_ms ?? null;
    const device = r.device ?? r.hardware ?? 'N/A';
    return `
      <tr>
        <td><strong>${mode}</strong></td>
        <td>${latency != null ? ms(latency) : '—'}</td>
        <td>${fps    != null ? fps.toFixed(1)+' FPS' : '—'}</td>
        <td>${pre    != null ? ms(pre)  : '—'}</td>
        <td>${inf    != null ? ms(inf)  : '—'}</td>
        <td>${post   != null ? ms(post) : '—'}</td>
        <td><span style="font-size:.75rem;color:rgba(255,255,255,.5);">${device}</span></td>
      </tr>
    `;
  }).join('');
}

// ── Diagnostics ──────────────────────────────────────────────────────────────

function renderDiagnostics(data) {
  const grid = document.getElementById('diagGrid');
  if (!data || Object.keys(data).length === 0) {
    grid.innerHTML = `<div class="diag-card">
      <div class="unavail-notice"><span class="icon">⚠️</span><span>Diagnostics unavailable.</span></div>
    </div>`;
    return;
  }

  const cards = [];

  // Model info
  if (data.model) {
    const m = data.model;
    cards.push(`
      <div class="diag-card">
        <h3>Model</h3>
        <ul class="kv-list">
          ${m.name       ? `<li><span class="k">Name</span><span class="v">${m.name}</span></li>` : ''}
          ${m.type       ? `<li><span class="k">Type</span><span class="v">${m.type}</span></li>` : ''}
          ${m.task       ? `<li><span class="k">Task</span><span class="v">${m.task}</span></li>` : ''}
          ${m.weights    ? `<li><span class="k">Weights</span><span class="v">${m.weights}</span></li>` : ''}
          ${m.input_size ? `<li><span class="k">Input Size</span><span class="v">${JSON.stringify(m.input_size)}</span></li>` : ''}
          ${m.classes    ? `<li><span class="k">Classes</span><span class="v">${m.classes}</span></li>` : ''}
        </ul>
      </div>
    `);
  }

  // Pipeline
  if (data.pipeline) {
    const p = data.pipeline;
    cards.push(`
      <div class="diag-card">
        <h3>Pipeline</h3>
        <ul class="kv-list">
          ${p.tracker_type       ? `<li><span class="k">Tracker</span><span class="v">${p.tracker_type}</span></li>` : ''}
          ${p.scene_graph_enabled != null ? `<li><span class="k">Scene Graph</span><span class="v">${p.scene_graph_enabled ? 'enabled' : 'disabled'}</span></li>` : ''}
          ${p.calibration_mode   ? `<li><span class="k">Calibration</span><span class="v">${p.calibration_mode}</span></li>` : ''}
          ${p.confidence_threshold != null ? `<li><span class="k">Conf Threshold</span><span class="v">${p.confidence_threshold}</span></li>` : ''}
          ${p.iou_threshold      != null ? `<li><span class="k">IoU Threshold</span><span class="v">${p.iou_threshold}</span></li>` : ''}
        </ul>
      </div>
    `);
  }

  // System
  if (data.system) {
    const s = data.system;
    cards.push(`
      <div class="diag-card">
        <h3>System</h3>
        <ul class="kv-list">
          ${s.device    ? `<li><span class="k">Device</span><span class="v">${s.device}</span></li>` : ''}
          ${s.cuda      ? `<li><span class="k">CUDA</span><span class="v">${s.cuda}</span></li>` : ''}
          ${s.torch     ? `<li><span class="k">PyTorch</span><span class="v">${s.torch}</span></li>` : ''}
          ${s.ultralytics ? `<li><span class="k">Ultralytics</span><span class="v">${s.ultralytics}</span></li>` : ''}
          ${s.python    ? `<li><span class="k">Python</span><span class="v">${s.python}</span></li>` : ''}
        </ul>
      </div>
    `);
  }

  // Raw fallback if no structured cards
  if (cards.length === 0) {
    const entries = Object.entries(data).filter(([, v]) => typeof v !== 'object').slice(0, 16);
    cards.push(`
      <div class="diag-card">
        <h3>Diagnostics</h3>
        <ul class="kv-list">
          ${entries.map(([k,v]) => `<li><span class="k">${k}</span><span class="v">${v}</span></li>`).join('')}
        </ul>
      </div>
    `);
  }

  grid.innerHTML = cards.join('');
}

// ── Main Load ────────────────────────────────────────────────────────────────

async function loadAllData() {
  const btn  = document.getElementById('refreshBtn');
  const info = document.getElementById('lastUpdated');
  btn.classList.add('spinning');
  info.textContent = 'Fetching…';

  const BASE = '';

  const [evalRes, benchRes, diagRes] = await Promise.allSettled([
    fetch(BASE + '/api/v1/model/evaluation').then(r => r.json()),
    fetch(BASE + '/api/v1/model/benchmark').then(r => r.json()),
    fetch(BASE + '/api/v1/model/diagnostics').then(r => r.json()),
  ]);

  if (evalRes.status === 'fulfilled') {
    try { renderCoreStats(evalRes.value); }    catch(e) { console.warn('coreStats', e); }
    try { renderClassTable(evalRes.value); }   catch(e) { console.warn('classTable', e); }
    try { renderConfusionMatrix(evalRes.value); } catch(e) { console.warn('cm', e); }
  } else {
    const msg = `<div class="unavail-notice"><span class="icon">❌</span><span>Could not fetch evaluation data: ${evalRes.reason}</span></div>`;
    document.getElementById('coreStats').innerHTML = msg;
    document.getElementById('classTableBody').innerHTML = `<tr><td colspan="7">${msg}</td></tr>`;
    document.getElementById('cmSection').innerHTML = msg;
  }

  if (benchRes.status === 'fulfilled') {
    try { renderBenchmark(benchRes.value); } catch(e) { console.warn('bench', e); }
  } else {
    document.getElementById('benchTableBody').innerHTML = `<tr><td colspan="7">
      <div class="unavail-notice" style="margin:.5rem 0;"><span class="icon">❌</span><span>Could not fetch benchmark data.</span></div>
    </td></tr>`;
  }

  if (diagRes.status === 'fulfilled') {
    try { renderDiagnostics(diagRes.value); } catch(e) { console.warn('diag', e); }
  } else {
    document.getElementById('diagGrid').innerHTML = `<div class="diag-card">
      <div class="unavail-notice"><span class="icon">❌</span><span>Could not fetch diagnostics.</span></div>
    </div>`;
  }

  btn.classList.remove('spinning');
  info.textContent = 'Updated: ' + new Date().toLocaleTimeString();
}

// Auto-load on page open
window.addEventListener('DOMContentLoaded', loadAllData);
</script>
</body>
</html>
"""

out_path = os.path.join(
    os.path.dirname(__file__),
    '..', 'frontend', 'eval-dashboard.html'
)
out_path = os.path.normpath(out_path)

with open(out_path, 'w', encoding='utf-8') as f:
    f.write(html)

print(f"Written: {out_path}  ({len(html)} bytes)")
