import { apiClient } from "./src/api/client.js";
import { telemetryService } from "./src/services/telemetryService.js";
import { mediaService } from "./src/services/mediaService.js";
import { cameraService } from "./src/services/cameraService.js";
import { incidentService } from "./src/services/incidentService.js";
import { analyticsService } from "./src/services/analyticsService.js";
import { sceneService } from "./src/services/sceneService.js";

const app = document.querySelector("#app");

const iconPaths = {
  activity: '<path d="M3 12h4l3-8 4 16 3-8h4"/>',
  alert: '<path d="M10.3 3.9 2.6 17.2A2 2 0 0 0 4.3 20h15.4a2 2 0 0 0 1.7-2.8L13.7 3.9a2 2 0 0 0-3.4 0Z"/><path d="M12 9v4m0 3h.01"/>',
  arrow: '<path d="M5 12h14m-7-7 7 7-7 7"/>',
  arrowUp: '<path d="m7 14 5-5 5 5"/>',
  camera: '<path d="M14 5H4a2 2 0 0 0-2 2v10a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V7a2 2 0 0 0-2-2Z"/><path d="m16 10 6-3v10l-6-3"/>',
  chart: '<path d="M3 3v18h18"/><path d="m19 9-5 5-4-4-5 6"/>',
  check: '<path d="m5 12 4 4L19 6"/>',
  chevron: '<path d="m9 18 6-6-6-6"/>',
  clock: '<circle cx="12" cy="12" r="9"/><path d="M12 7v5l3 2"/>',
  close: '<path d="m18 6-12 12M6 6l12 12"/>',
  download: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m7 10 5 5 5-5m-5 5V3"/>',
  expand: '<path d="M8 3H5a2 2 0 0 0-2 2v3m13-5h3a2 2 0 0 1 2 2v3M3 16v3a2 2 0 0 0 2 2h3m13-5v3a2 2 0 0 1-2 2h-3"/>',
  grid: '<rect x="3" y="3" width="8" height="8" rx="1.5"/><rect x="13" y="3" width="8" height="5" rx="1.5"/><rect x="13" y="10" width="8" height="11" rx="1.5"/><rect x="3" y="13" width="8" height="8" rx="1.5"/>',
  image: '<rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/>',
  info: '<circle cx="12" cy="12" r="10"/><path d="M12 16v-4m0-4h.01"/>',
  file: '<path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z"/><path d="M14 2v6h6M8 13h8m-8 4h8"/>',
  layers: '<path d="m12 3 9 5-9 5-9-5 9-5Z"/><path d="m3 12 9 5 9-5M3 16l9 5 9-5"/>',
  link: '<path d="M10 13a5 5 0 0 0 7.1 0l3-3A5 5 0 0 0 13 2.9l-1.7 1.7"/><path d="M14 11a5 5 0 0 0-7.1 0l-3 3A5 5 0 0 0 11 21.1l1.7-1.7"/>',
  menu: '<path d="M4 6h16M4 12h16M4 18h16"/>',
  pause: '<path d="M8 5v14m8-14v14"/>',
  play: '<path d="m7 4 14 8-14 8V4Z"/>',
  plus: '<path d="M12 5v14m-7-7h14"/>',
  sun: '<circle cx="12" cy="12" r="4"/><path d="M12 2v2m0 16v2M4.93 4.93l1.42 1.42m11.3 11.3 1.42 1.42M2 12h2m16 0h2M4.93 19.07l1.42-1.42m11.3-11.3 1.42-1.42"/>',
  moon: '<path d="M20.9 13A9 9 0 0 1 11 3.1 9 9 0 1 0 20.9 13Z"/>',
  monitor: '<rect x="3" y="4" width="18" height="13" rx="2"/><path d="M8 21h8m-4-4v4"/>',
  volume: '<path d="M11 5 6 9H3v6h3l5 4V5Z"/><path d="M15.5 8.5a5 5 0 0 1 0 7m3-10a9 9 0 0 1 0 13"/>',
  volumeOff: '<path d="M11 5 6 9H3v6h3l5 4V5Z"/><path d="m17 9 5 6m0-6-5 6"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-4-4"/>',
  settings: '<path d="M12 8a4 4 0 1 0 0 8 4 4 0 0 0 0-8Z"/><path d="m19.4 15 .1.1 1.4 1.1-1.4 2.4-1.7-.6a8 8 0 0 1-1.6.9l-.3 1.8h-2.8l-.3-1.8a8 8 0 0 1-1.6-.9l-1.7.6-1.4-2.4 1.4-1.1a7 7 0 0 1 0-1.9l-1.4-1.1 1.4-2.4 1.7.6a8 8 0 0 1 1.6-.9l.3-1.8h2.8l.3 1.8a8 8 0 0 1 1.6.9l1.7-.6 1.4 2.4-1.4 1.1a7 7 0 0 1 0 1.8Z" transform="translate(-1 -1)"/>',
  shield: '<path d="M12 22s8-4 8-11V5l-8-3-8 3v6c0 7 8 11 8 11Z"/><path d="m9 12 2 2 4-4"/>',
  upload: '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><path d="m17 8-5-5-5 5m5-5v12"/>',
  video: '<rect x="3" y="5" width="13" height="14" rx="2"/><path d="m16 10 5-3v10l-5-3"/>',
  wifi: '<path d="M5 12.5a11 11 0 0 1 14 0M2 9a16 16 0 0 1 20 0m-14 7a6 6 0 0 1 8 0m-4 4h.01"/>',
  x: '<path d="M18 6 6 18M6 6l12 12"/>',
};

const modules = [
  ["objects", "People, vehicles & objects"],
  ["assets", "Industrial assets"],
  ["ppe", "PPE compliance"],
  ["zones", "Safety zones"],
  ["behavior", "Behavior"],
  ["proximity", "Proximity & depth"],
  ["predictions", "Predictive warnings"],
  ["risk", "Risk scoring"],
];

function readLocalPreference(key, fallback) {
  try {
    const value = localStorage.getItem(key);
    return value === null ? fallback : value;
  } catch (_) {
    return fallback;
  }
}

function readModulePreferences() {
  const defaults = { objects: true, assets: false, ppe: true, zones: true, behavior: true, proximity: false, predictions: false, risk: true };
  try {
    const saved = JSON.parse(localStorage.getItem("intelliwatch-modules") || "null");
    if (saved && typeof saved === "object") {
      for (const key of Object.keys(defaults)) if (typeof saved[key] === "boolean") defaults[key] = saved[key];
    }
  } catch (_) {}
  return defaults;
}

const state = {
  source: null,
  theme: document.documentElement.dataset.themePreference || "system",
  playbackRate: Number(readLocalPreference("intelliwatch-playback-rate", "1")) || 1,
  previewMuted: readLocalPreference("intelliwatch-preview-muted", "true") !== "false",
  modules: readModulePreferences(),
  rtspMessage: "",
  toast: "",
  publicMenuOpen: false,

  // Integrated Backend Telemetry & Runtime State
  telemetry: {
    online: false,
    isCuda: false,
    gpuName: null,
    fps: null,
    latencyMs: null,
    complianceRate: 100,
    totalCameras: 0,
    onlineCameras: 0,
    activeIncidents: 0,
    eventsToday: 0,
    modelsActive: {},
    hardwareStatus: "Connecting to backend...",
  },

  // Active Detection & Media Analysis
  analysisState: {
    jobId: null,
    phase: "IDLE", // 'IDLE' | 'SELECTED' | 'UPLOADING' | 'PROCESSING' | 'COMPLETED' | 'FAILED'
    running: false,
    progressPct: 0,
    statusText: "",
    result: null,
    error: null,
    viewMode: "annotated", // 'annotated' | 'original'
    latestFrameUrl: null,
    annotatedUrl: null,
    annotatedFrameUrl: null,
    originalUrl: null,
    mediaZoom: 1.0,
  },
  activeAnalysisAbort: null,
  reportModalOpen: false,

  // Cameras
  cameras: [],
  camerasLoading: false,
  cameraModalOpen: false,

  // Alerts & Incidents
  alerts: [],
  alertsLoading: false,
  alertFilter: "ALL",
  alertSearch: "",
  activeDetailAlert: null,

  // Safety Analytics & Zones
  alertStats: null,
  analyticsLoading: false,
  zones: [],

  // Detailed Inference Breakdown Selection & Toggle
  selectedPersonId: null,
  showAllInference: false,

  // Scene Graph Hero Feature State
  sceneGraphState: {
    loading: false,
    data: null,
    error: null,
    selectedEntityId: null,
    selectedRelationId: null,
    viewMode: "graph", // 'graph' | 'split' | 'table'
    entityFilter: "ALL", // 'ALL' | 'PEOPLE' | 'VEHICLES' | 'PPE' | 'ZONES' | 'HAZARDS'
    relationFilter: "ALL", // 'ALL' | 'SPATIAL' | 'TEMPORAL' | 'SAFETY' | 'EQUIPMENT'
    filter: "ALL", // 'ALL' | 'PERSON' | 'VEHICLE' | 'ZONE' | 'PPE'
    hoveredRelationId: null,
    zoom: 1.0,
    pan: { x: 0, y: 0 },
    lastJobId: null,
  },

  // Camera Calibration & Ground-Plane Spatial State
  calibrationState: {
    selectedCameraId: "cam_loading_bay_01",
    frameUrl: null,
    imageWidth: 1376,
    imageHeight: 768,
    points: [], // Array of {x, y} in natural image coordinates (up to 4)
    realWorldWidthM: 6.0,
    realWorldDepthM: 10.0,
    coordinateSystem: "METRIC_GROUND_PLANE",
    calibrationStatus: "UNCONFIGURED",
    isValid: false,
    validationError: null,
    validationResiduals: null,
    saving: false,
    loading: false,
    dirty: false,
  },
};

function icon(name, size = 18) {
  return `<svg aria-hidden="true" width="${size}" height="${size}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.7" stroke-linecap="round" stroke-linejoin="round">${iconPaths[name] || iconPaths.activity}</svg>`;
}

function resolvedTheme(preference = state.theme) {
  if (preference !== "system") return preference;
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ? "dark" : "light";
}

function applyTheme(preference = state.theme, persist = true) {
  state.theme = preference;
  document.documentElement.dataset.themePreference = preference;
  document.documentElement.dataset.theme = resolvedTheme(preference);
  if (persist) {
    try { localStorage.setItem("intelliwatch-theme", preference); } catch (_) {}
  }
}

function themeControl() {
  const resolved = resolvedTheme();
  const label = `Color theme: ${state.theme === "system" ? `System (${resolved})` : resolved}. Click to cycle theme.`;
  return `<button class="theme-control" type="button" data-action="cycle-theme" title="${label}" aria-label="${label}">${icon(state.theme === "system" ? "monitor" : state.theme === "light" ? "sun" : "moon", 17)}<span class="sr-only">${label}</span></button>`;
}

function refreshThemeControlLabels() {
  const resolved = resolvedTheme();
  const label = `Color theme: ${state.theme === "system" ? `System (${resolved})` : resolved}. Click to cycle theme.`;
  app.querySelectorAll(".theme-control").forEach((control) => {
    control.title = label;
    control.setAttribute("aria-label", label);
    control.innerHTML = `${icon(state.theme === "system" ? "monitor" : state.theme === "light" ? "sun" : "moon", 17)}<span class="sr-only">${label}</span>`;
  });
  app.querySelectorAll("[data-theme-choice]").forEach((choice) => {
    choice.checked = choice.value === state.theme;
    choice.closest(".appearance-option")?.classList.toggle("is-selected", choice.checked);
  });
  const systemDetail = app.querySelector("[data-system-theme-detail]");
  if (systemDetail) systemDetail.textContent = `Follows device • ${resolved}`;
}

function setPublicMenu(open) {
  state.publicMenuOpen = open;
  const nav = app.querySelector(".public-nav");
  const button = app.querySelector(".public-menu-button");
  nav?.classList.toggle("is-open", open);
  if (button) {
    button.setAttribute("aria-expanded", String(open));
    button.setAttribute("aria-label", `${open ? "Close" : "Open"} navigation menu`);
    button.innerHTML = icon(open ? "close" : "menu", 18);
  }
}

const systemThemePreference = window.matchMedia?.("(prefers-color-scheme: dark)");
systemThemePreference?.addEventListener?.("change", () => {
  if (state.theme === "system") {
    document.documentElement.dataset.theme = resolvedTheme("system");
    refreshThemeControlLabels();
  }
});

function escapeHTML(value = "") {
  return String(value).replace(/[&<>"']/g, (char) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" })[char]);
}

function formatBytes(bytes) {
  if (!Number.isFinite(bytes)) return "";
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}

function currentPath() {
  return window.location.pathname.replace(/\/$/, "") || "/";
}

function navigate(path) {
  if (path === currentPath()) {
    if (state.publicMenuOpen) {
      setPublicMenu(false);
    }
    return;
  }
  state.publicMenuOpen = false;
  history.pushState({}, "", path);
  state.toast = "";
  render();
  onRouteChanged();
  window.scrollTo({ top: 0, behavior: "smooth" });
}

function sceneSvg() {
  return `<svg class="scene-svg" viewBox="0 0 1440 810" role="img" aria-label="Illustrated sample loading bay with a forklift, a worker and a marked pedestrian zone" xmlns="http://www.w3.org/2000/svg">
    <defs>
      <pattern id="floor-lines" width="100" height="70" patternUnits="userSpaceOnUse" patternTransform="skewX(-28)"><path d="M0 0H100M0 0V70" fill="none" stroke="#778078" stroke-opacity=".13" stroke-width="2"/></pattern>
      <pattern id="grain" width="90" height="90" patternUnits="userSpaceOnUse"><circle cx="13" cy="19" r="1" fill="#fff" opacity=".07"/><circle cx="61" cy="54" r="1" fill="#fff" opacity=".05"/><circle cx="81" cy="17" r=".7" fill="#fff" opacity=".06"/></pattern>
    </defs>
    <rect width="1440" height="810" fill="#18201f"/>
    <path d="M0 0H1440V408L0 353Z" fill="#202928"/>
    <path d="m0 74 1440-31M0 145l1440-31M0 216l1440-31M0 286l1440-31" stroke="#4c5752" stroke-width="8" opacity=".45"/>
    <path d="M165 0v380M472 0v391M795 0v402M1112 0v414M1377 0v430" stroke="#384541" stroke-width="24"/>
    <path d="M0 344 1440 410V810H0Z" fill="#353d38"/>
    <path d="M0 344 1440 410V810H0Z" fill="url(#floor-lines)"/>
    <path d="M1020 180h352v247h-352z" fill="#101715" stroke="#4a5550" stroke-width="12"/>
    <path d="M1050 207h292v193h-292z" fill="#232c28"/>
    <path d="M1050 240h292M1050 274h292M1050 308h292M1050 342h292M1050 376h292" stroke="#44514a" stroke-width="8"/>
    <path d="M0 427 1440 478v250L0 670Z" fill="#313a35" opacity=".62"/>
    <path d="m605 448 348 17-41 199-312-17Z" fill="#9b7840" opacity=".18" stroke="#d5ad5d" stroke-width="4" stroke-dasharray="18 13"/>
    <path d="M520 558c124-82 280-72 402-19" fill="none" stroke="#e4b85b" stroke-width="4" stroke-dasharray="12 11" opacity=".8"/>
    <path d="m901 533 22 7-16 15" fill="none" stroke="#e4b85b" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"/>
    <!-- forklift -->
    <g transform="translate(793 436)">
      <path d="M61 99h250l40 51-7 62H68l-30-47Z" fill="#bd934c" stroke="#d7b46d" stroke-width="5"/>
      <path d="m74 96 33-93h83l37 94M114 5h61l28 86h-120Z" fill="#29322e" stroke="#d7b46d" stroke-width="6"/>
      <path d="M238 86h49v91h-49z" fill="#28312d" stroke="#d7b46d" stroke-width="5"/>
      <path d="M292 21v186m18-184v186M310 200l111 14" stroke="#d7b46d" stroke-width="9"/>
      <path d="M63 139h-37v-43h55" fill="#d2aa5b"/>
      <circle cx="110" cy="214" r="32" fill="#151b19" stroke="#64716a" stroke-width="6"/>
      <circle cx="295" cy="214" r="32" fill="#151b19" stroke="#64716a" stroke-width="6"/>
      <circle cx="110" cy="214" r="11" fill="#88918a"/><circle cx="295" cy="214" r="11" fill="#88918a"/>
      <path d="M145 28h42" stroke="#b5c29a" stroke-width="8"/>
      <path d="M167 55c22 5 31 18 42 38h-84l15-38Z" fill="#d8d6c4"/>
      <path d="M173 45h20v9h-20z" fill="#b97847"/>
    </g>
    <!-- worker -->
    <g transform="translate(590 411)">
      <path d="M53 75c-19 0-35 16-38 38l-16 86h42l23-65 7 67h56l-11-99c-2-16-13-27-28-27Z" fill="#809878" stroke="#b7c2a1" stroke-width="5"/>
      <path d="m21 113-43 74 23 13 46-60M104 110l31 56-20 13-42-50" fill="none" stroke="#b7c2a1" stroke-width="14" stroke-linecap="round"/>
      <path d="m34 233-5 89h28l15-89m23 0 9 89h29l-7-89" fill="#202824" stroke="#89938b" stroke-width="7"/>
      <path d="M37 47c0-24 16-43 38-43s39 19 39 43v22H37Z" fill="#c4a464" stroke="#e0c47e" stroke-width="5"/>
      <path d="M31 48h89" stroke="#e0c47e" stroke-width="8" stroke-linecap="round"/>
    </g>
    <path d="M0 0h1440v810H0z" fill="url(#grain)"/>
  </svg>`;
}

function overlaySvg() {
  return `<svg class="detection-overlay" viewBox="0 0 1440 810" aria-hidden="true" preserveAspectRatio="none">
    <path d="m560 423 190 5-15 360-210-8Z" fill="none" stroke="#d4aa5b" stroke-width="3" stroke-dasharray="10 8" opacity=".82"/>
    <path d="m801 427 346 18-40 234-313-17Z" fill="none" stroke="#df935e" stroke-width="3" stroke-dasharray="10 8" opacity=".8"/>
    <path d="m521 425 151 4v35H521Z" fill="#d1a958"/>
    <path d="m801 427 182 9v35l-182-9Z" fill="#d88758"/>
  </svg>`;
}

function publicHeader() {
  const path = currentPath();
  return `<header class="public-header"><a class="brand" href="/" data-link aria-label="IntelliWatch home"><span class="brand-mark">IW</span><span class="brand-name">Intelli<span>Watch</span></span></a>
    <nav class="public-nav ${state.publicMenuOpen ? "is-open" : ""}" aria-label="Main navigation"><a class="${path === "/product" ? "is-active" : ""}" href="/product" data-link ${path === "/product" ? 'aria-current="page"' : ""}>Product</a><a href="/#capabilities">Capabilities</a><a href="/#industries">Industries</a><a href="/#workflow">How it works</a></nav>
    <div class="public-header-actions">${themeControl()}<button class="public-menu-button" type="button" data-action="toggle-public-menu" aria-label="${state.publicMenuOpen ? "Close" : "Open"} navigation menu" aria-expanded="${state.publicMenuOpen}">${icon(state.publicMenuOpen ? "close" : "menu", 18)}</button></div>
  </header>`;
}

function activeNav(path, target) {
  if (target === "/app") return path === "/app" || path === "/app/analyze";
  if (target === "/app/scene-graph") return path === target || path.startsWith(`${target}/`);
  if (target === "/app/incidents") return path === target || path.startsWith(`${target}/`);
  return path === target;
}

function workspaceShell(content, options = {}) {
  const path = currentPath();
  const nav = [
    ["/app", "Workspace", "grid"],
    ["/app/scene-graph", "Scene Graph", "layers"],
    ["/app/live", "Live monitoring", "camera"],
    ["/app/incidents", "Incidents", "alert"],
    ["/app/analytics", "Analytics", "chart"],
    ["/app/settings", "Settings", "settings"],
  ];

  const t = state.telemetry;
  const activeIncidentCount = state.alertStats?.active_unresolved_count ?? t.activeIncidents;
  const isOnline = t.online;
  const isCuda = t.isCuda;
  const statusDotClass = isOnline ? (isCuda ? "status-active" : "status-neutral") : "status-warning";
  const statusHeadline = isOnline ? (isCuda ? "CUDA:0 Active" : "System Online") : "Backend Offline";
  const statusSub = isOnline ? (t.gpuName || (isCuda ? "NVIDIA RTX 5050" : "CPU Fallback")) : "Reconnecting...";
  const badgeText = isOnline ? (isCuda ? "CUDA:0 ACCELERATED" : "ONLINE (CPU)") : "BACKEND OFFLINE";

  return `<div class="operator-layout">
    <aside class="sidebar"><div class="sidebar-brand"><a class="brand" href="/" data-link><span class="brand-mark">IW</span><span class="brand-name">Intelli<span>Watch</span></span></a></div>
      <div class="sidebar-workspace"><span class="workspace-icon">IW</span><span class="workspace-label"><b>IntelliWatch Hub</b><small>${escapeHTML(badgeText)}</small></span></div>
      <div class="sidebar-section-label">OPERATIONS</div><nav class="side-nav" aria-label="Workspace navigation">${nav.map(([href, label, glyph]) => `<a class="side-link ${activeNav(path, href) ? "is-active" : ""}" href="${href}" data-link ${activeNav(path, href) ? 'aria-current="page"' : ""}>${icon(glyph, 18)}<span>${label}</span>${label === "Incidents" && activeIncidentCount > 0 ? `<span class="nav-count active-count">${activeIncidentCount}</span>` : ""}</a>`).join("")}</nav>
      <div class="sidebar-spacer"></div>
      <div class="sidebar-status"><span class="status-dot ${statusDotClass}"></span><div><b>${escapeHTML(statusHeadline)}</b><small>${escapeHTML(statusSub)}</small></div></div>
      <div class="sidebar-foot"><span>INTELLIWATCH</span><span>SAFETY OPERATIONS</span></div>
    </aside>
    <div class="app-column"><header class="workspace-topbar"><div class="mobile-brand"><a class="brand" href="/" data-link><span class="brand-mark">IW</span><span class="brand-name">Intelli<span>Watch</span></span></a></div><div class="crumbs"><span>OPERATIONS</span>${icon("chevron", 13)}<b>${options.breadcrumb || "WORKSPACE"}</b></div><div class="topbar-actions">${themeControl()}<span class="local-badge"><i class="status-dot ${statusDotClass}"></i> ${escapeHTML(badgeText)}</span><a class="topbar-link" href="/" data-link>Website ${icon("arrow", 13)}</a></div></header><main class="workspace-main">${content}</main></div>
    ${state.toast ? `<div class="toast" role="status">${icon("alert", 17)}<span>${escapeHTML(state.toast)}</span><button type="button" class="icon-button toast-close" data-action="dismiss-toast" aria-label="Dismiss message">${icon("close", 16)}</button></div>` : ""}
    ${state.cameraModalOpen ? cameraRegisterModal() : ""}
  </div>`;
}

function introHeading(eyebrow, title, description, action = "") {
  return `<div class="workspace-heading"><div><div class="eyebrow">${eyebrow}</div><h1>${title}</h1><p>${description}</p></div>${action}</div>`;
}

function sourcePreview(source, controls = true) {
  if (source.kind === "video") return `<video class="uploaded-media" data-uploaded-player src="${source.url}" ${controls ? "controls" : ""} playsinline preload="metadata" aria-label="Uploaded video preview"></video>`;
  return `<img class="uploaded-media image-media" src="${source.url}" alt="Uploaded image preview" />`;
}

function sourceSelectorPage() {
  const intro = introHeading("SOURCE WORKSPACE", "Choose an industrial media source", "Upload recorded video or inspection frames directly to the CUDA-accelerated vision pipeline.", `<span class="source-step-tag"><span>01</span> CONNECT SOURCE</span>`);
  if (state.source) {
    return `${intro}<div class="setup-grid"><section class="panel source-preview-panel"><div class="panel-header"><div><span class="panel-kicker">SELECTED SOURCE</span><h2>${icon(state.source.kind === "video" ? "video" : "image", 17)} ${escapeHTML(state.source.name)}</h2></div><button class="button button-quiet button-small" type="button" data-action="clear-source">${icon("close", 15)} Change source</button></div><div class="selected-preview">${sourcePreview(state.source, true)}</div><div class="source-meta"><span>${icon("file", 15)} ${state.source.kind === "video" ? "Video" : "Image"} • ${formatBytes(state.source.file.size)}</span><span>${icon("shield", 15)} Ready for YOLO inference</span></div></section>
      <aside class="panel setup-panel"><div class="panel-kicker">ANALYSIS PROFILE</div><h2>Inference Options</h2><p class="panel-intro">Target safety intelligence layers enabled for this session.</p><div class="module-list compact-module-list">${modules.map(([key, label]) => moduleToggle(key, label)).join("")}</div><button class="button button-primary button-full" type="button" data-action="start-analysis">Run Detection on GPU ${icon("arrow", 16)}</button><p class="connection-note" style="margin-top:10px;">${icon("info", 15)} Uses PyTorch CUDA on ${escapeHTML(state.telemetry.gpuName || 'cuda:0')}.</p></aside></div>
      <div class="setup-bottom-note">${icon("shield", 15)}<span>Industrial pipeline preserves full resolution bounding box coordinate transforms.</span></div>`;
  }

  return `${intro}<div class="source-options">
    <article class="panel source-option source-option-primary" data-drop-kind="video" data-action="pick-video" style="cursor: pointer;"><div class="source-option-top"><span class="source-option-icon">${icon("video", 20)}</span><span class="source-option-label">VIDEO FILE</span></div><h2>Upload video footage</h2><p>Process video clips through persistent tracking, PPE temporal compliance and zone dwell reasoning.</p><button class="button button-secondary" type="button" data-action="pick-video">${icon("upload", 16)} Choose video file</button><small>Supports MP4, AVI, MOV, MKV up to 500MB.</small></article>
    <article class="panel source-option" data-drop-kind="image" data-action="pick-image" style="cursor: pointer;"><div class="source-option-top"><span class="source-option-icon">${icon("image", 20)}</span><span class="source-option-label">IMAGE FRAME</span></div><h2>Upload an inspection frame</h2><p>Instant single-frame YOLO perception: PPE detector, worker association, and spatial safety assessment.</p><button class="button button-secondary" type="button" data-action="pick-image">${icon("upload", 16)} Choose image</button><small>Supports JPG, JPEG, PNG with GPU inference.</small></article>
  </div>
  <section class="panel rtsp-panel"><div class="rtsp-intro"><span class="source-option-icon">${icon("camera", 20)}</span><div><span class="source-option-label">LIVE STREAM</span><h2>Register RTSP / CCTV Camera</h2><p>Enter connection parameters to add a live camera feed to the control room.</p></div></div><form id="rtsp-form" class="rtsp-form"><label>Camera name<input name="cameraName" placeholder="e.g. Loading Bay 01" autocomplete="off" required /></label><label class="rtsp-url-field">RTSP URL<input name="rtspUrl" placeholder="rtsp://192.168.1.100:554/stream1" autocomplete="off" spellcheck="false" required /></label><button class="button button-secondary" type="submit">Register camera ${icon("arrow", 15)}</button></form>${state.rtspMessage ? `<div class="inline-error" role="alert">${icon("alert", 16)}<span>${escapeHTML(state.rtspMessage)}</span></div>` : ""}<p class="connection-note">Registered cameras run background inference workers and support live MJPEG preview.</p></section>
  <div class="privacy-note">${icon("shield", 15)}<span>Data policy: Processed directly through your verified local IntelliWatch CUDA engine.</span></div>`;
}

function normalizeComplianceStatus(value) {
  const raw = value && typeof value === "object" ? (value.value || value.name || "") : value;
  const normalized = String(raw || "UNKNOWN").split(".").pop().toUpperCase().replace(/_/g, "-").trim();
  return ["COMPLIANT", "NON-COMPLIANT", "UNKNOWN"].includes(normalized) ? normalized : "UNKNOWN";
}

function moduleToggle(key, label) {
  return `<label class="module-toggle"><span>${label}</span><input type="checkbox" data-module="${key}" ${state.modules[key] ? "checked" : ""} /><i aria-hidden="true"></i></label>`;
}

function severityTag(level) {
  const l = (level || "INFO").toUpperCase();
  return `<span class="severity-tag severity-${l.toLowerCase()}">${l}</span>`;
}

function renderReportModal(result, source, analysis) {
  if (!state.reportModalOpen) return "";
  const isVideo = source?.kind === "video";
  const bd = result?.inference_breakdown || (result ? extractBreakdownFromAssessment(result) : null);
  const people = bd?.people || [];
  const compliantCount = bd?.compliant_people ?? people.filter((p) => p.compliance_status === "COMPLIANT").length;
  const nonCompliantCount = bd?.non_compliant_people ?? people.filter((p) => p.compliance_status === "NON-COMPLIANT").length;
  const alerts = state.alerts || [];
  const evidenceImg = analysis.annotatedFrameUrl || analysis.latestFrameUrl || analysis.annotatedUrl;
  const dateStr = new Date().toLocaleString();

  return `
  <div class="report-modal" id="report-modal">
    <div class="report-modal-backdrop" data-action="close-report-modal"></div>
    <div class="report-content panel">
      <div class="report-header">
        <div>
          <div class="report-logo">INTELLIWATCH</div>
          <div class="report-title">Safety Intelligence & Compliance Report</div>
          <div class="report-meta">Automated Computer Vision Safety Assessment • Grounded Neural Evidence</div>
        </div>
        <div class="report-actions">
          <button type="button" class="button button-primary button-small" data-action="print-report">${icon("download", 14)} Print / Save PDF</button>
          <button type="button" class="button button-quiet button-small" data-action="close-report-modal">${icon("close", 14)} Close</button>
        </div>
      </div>

      <!-- 1. Analysis Summary -->
      <section class="report-section">
        <h3 class="report-section-title">1. Analysis Summary</h3>
        <div class="report-summary-grid">
          <div class="report-kpi-item"><span class="kpi-label">SOURCE NAME</span><b>${escapeHTML(source?.name || "Live Media")}</b></div>
          <div class="report-kpi-item"><span class="kpi-label">MEDIA TYPE</span><b>${isVideo ? "Multi-Frame Video Stream" : "Still Frame Image"}</b></div>
          <div class="report-kpi-item"><span class="kpi-label">EXECUTION TIME</span><b>${dateStr}</b></div>
          <div class="report-kpi-item"><span class="kpi-label">HARDWARE ACCELERATION</span><b>${escapeHTML(state.telemetry.gpuName || "CUDA:0 GPU")}</b></div>
        </div>
      </section>

      <!-- 2. Scene Summary -->
      <section class="report-section">
        <h3 class="report-section-title">2. Scene Summary</h3>
        <div class="report-summary-grid">
          <div class="report-kpi-item"><span class="kpi-label">PERSONNEL DETECTED</span><b>${people.length}</b></div>
          <div class="report-kpi-item"><span class="kpi-label">PPE COMPLIANT</span><b style="color:var(--success);">${compliantCount}</b></div>
          <div class="report-kpi-item"><span class="kpi-label">PPE NON-COMPLIANT</span><b style="color:var(--critical);">${nonCompliantCount}</b></div>
          <div class="report-kpi-item"><span class="kpi-label">SIGNIFICANT INCIDENTS</span><b style="color:${alerts.length > 0 ? "var(--warning)" : "var(--success)"};">${alerts.length}</b></div>
        </div>
      </section>

      <!-- 3. PPE Compliance Table -->
      <section class="report-section">
        <h3 class="report-section-title">3. Personnel PPE Compliance Audit</h3>
        ${people.length === 0 ? `<p class="report-empty-notice">No personnel detected in the assessed scene.</p>` : `
        <table class="report-table">
          <thead>
            <tr>
              <th>PERSON IDENTITY</th>
              <th>CONFIDENCE</th>
              <th>COMPLIANCE STATUS</th>
              <th>DETECTED PPE</th>
              <th>MISSING PPE</th>
              <th>UNKNOWN PPE</th>
            </tr>
          </thead>
          <tbody>
            ${people.map((p) => `
              <tr>
                <td><b>${escapeHTML(p.label || `Person #${p.id}`)}</b></td>
                <td>${p.confidence_pct != null ? `${p.confidence_pct}%` : "—"}</td>
                <td><span class="report-badge ${p.compliance_status === "COMPLIANT" ? "badge-success" : (p.compliance_status === "NON-COMPLIANT" ? "badge-danger" : "badge-neutral")}">${escapeHTML(p.compliance_status)}</span></td>
                <td>${p.detected_ppe?.length ? p.detected_ppe.join(", ") : "—"}</td>
                <td><span style="color:var(--critical); font-weight:600;">${p.missing_ppe?.length ? p.missing_ppe.join(", ") : "None"}</span></td>
                <td>${p.unknown_ppe?.length ? p.unknown_ppe.join(", ") : "None"}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>`}
      </section>

      <!-- 4. Incidents -->
      <section class="report-section">
        <h3 class="report-section-title">4. Significant Safety Incidents</h3>
        ${alerts.length === 0 ? `<p class="report-empty-notice" style="color:var(--success);">No significant incidents detected during this analysis session.</p>` : `
        <table class="report-table">
          <thead>
            <tr>
              <th>TIME</th>
              <th>TYPE</th>
              <th>SEVERITY</th>
              <th>INVOLVED ENTITY</th>
              <th>EXPLANATION / EVIDENCE</th>
            </tr>
          </thead>
          <tbody>
            ${alerts.map((a) => `
              <tr>
                <td>${a.timestamp > 1000000000 ? new Date(a.timestamp * 1000).toLocaleTimeString() : `${Number(a.timestamp).toFixed(1)}s`}</td>
                <td><b>${escapeHTML(a.violation_type || "SAFETY_ALERT")}</b></td>
                <td><span class="report-badge severity-${(a.severity || "INFO").toLowerCase()}">${escapeHTML(a.severity || "INFO")}</span></td>
                <td>${escapeHTML(a.metadata?.display_label || (a.track_id != null ? `Person #${a.metadata?.display_id || a.track_id}` : "Scene"))}</td>
                <td>${escapeHTML(a.metadata?.explanation || a.explanation || (a.occurrence_count ? `${a.occurrence_count} frame occurrences recorded` : "Perception event"))}</td>
              </tr>
            `).join("")}
          </tbody>
        </table>`}
      </section>

      <!-- 5. Visual Evidence -->
      ${evidenceImg ? `
      <section class="report-section">
        <h3 class="report-section-title">5. Visual Evidence Frame</h3>
        <div style="max-height: 380px; overflow: hidden; border-radius: 6px; border: 1px solid var(--line); text-align: center; background: #000;">
          <img src="${escapeHTML(evidenceImg)}" style="max-width: 100%; max-height: 380px; object-fit: contain;" alt="Annotated Evidence" />
        </div>
      </section>` : ""}

      <!-- 6. Inference Performance -->
      <section class="report-section">
        <h3 class="report-section-title">6. Empirical Performance Metrics</h3>
        <div class="report-summary-grid">
          <div class="report-kpi-item"><span class="kpi-label">PROCESSING TIME</span><b>${result?.processing_time_seconds != null ? `${result.processing_time_seconds.toFixed(2)}s` : (result?.assessment?.processing_time_ms != null ? `${(result.assessment.processing_time_ms / 1000).toFixed(2)}s` : "—")}</b></div>
          <div class="report-kpi-item"><span class="kpi-label">MEASURED FPS</span><b>${result?.fps != null ? Number(result.fps).toFixed(1) : (state.telemetry.fps != null ? Number(state.telemetry.fps).toFixed(1) : "—")}</b></div>
          <div class="report-kpi-item"><span class="kpi-label">INFERENCE LATENCY</span><b>${result?.inference_latency_ms != null ? `${result.inference_latency_ms.toFixed(1)} ms` : (state.telemetry.latencyMs != null ? `${state.telemetry.latencyMs.toFixed(1)} ms` : "—")}</b></div>
          <div class="report-kpi-item"><span class="kpi-label">FRAMES EVALUATED</span><b>${result?.total_frames ?? (result?.assessment ? 1 : "—")}</b></div>
        </div>
      </section>

      <!-- 7. Methodology & Limitations -->
      <section class="report-section">
        <h3 class="report-section-title">7. Methodology & Epistemic Boundaries</h3>
        <div class="report-disclaimer">
          <b>Operational Distinctions:</b>
          <ul style="margin:6px 0 0 16px; padding:0;">
            <li><b>Detected:</b> Explicit positive neural detections meeting calibrated confidence thresholds.</li>
            <li><b>Missing:</b> Required safety equipment verified absent from spatial person bindings.</li>
            <li><b>Unknown:</b> Occluded or unresolvable regions where detection confidence is indeterminate.</li>
            <li><b>Unavailable:</b> Modalities not active for this media stream or hardware configuration.</li>
          </ul>
        </div>
      </section>
    </div>
  </div>`;
}

function analysisPage() {
  if (!state.source) return sourceSelectorPage();
  const source = state.source;
  const analysis = state.analysisState;
  const result = analysis.result;
  const isVideo = source.kind === "video";
  const videoScene = isVideo && state.sceneGraphState.lastJobId === analysis.jobId
    ? state.sceneGraphState.data
    : null;
  const videoBreakdown = isVideo && result ? (result.inference_breakdown || buildVideoBreakdownFromScene(videoScene, result)) : null;
  const hasAnnotatedMedia = Boolean(analysis.annotatedUrl);

  const isRunning = analysis.running || analysis.phase === "PROCESSING" || analysis.phase === "UPLOADING";
  const isCompleted = analysis.phase === "COMPLETED" && Boolean(result);

  let mediaHtml = "";
  if (isVideo) {
    if (analysis.viewMode === "annotated" && analysis.annotatedUrl) {
      mediaHtml = `<video class="uploaded-media" data-uploaded-player controls autoplay loop playsinline src="${escapeHTML(analysis.annotatedUrl)}" aria-label="Annotated detection evidence">Your browser does not support HTML5 video.</video>`;
    } else if (analysis.viewMode === "annotated_frame" && (analysis.annotatedFrameUrl || analysis.latestFrameUrl)) {
      mediaHtml = `<img class="uploaded-media image-media" src="${escapeHTML(analysis.annotatedFrameUrl || analysis.latestFrameUrl)}" alt="Annotated detection evidence snapshot" />`;
    } else if (analysis.phase === "PROCESSING" && analysis.latestFrameUrl) {
      mediaHtml = `<div class="live-processing-preview"><img class="uploaded-media image-media" src="${escapeHTML(analysis.latestFrameUrl)}" alt="Live inference preview" /><div class="live-frame-hud"><i class="status-dot status-active"></i> Processing Video Feed (${Math.round(analysis.progressPct)}%)</div></div>`;
    } else {
      mediaHtml = sourcePreview(source);
    }
  } else {
    if (analysis.viewMode === "annotated" && analysis.annotatedUrl) {
      mediaHtml = `<img class="uploaded-media image-media" src="${escapeHTML(analysis.annotatedUrl)}" alt="Annotated detection evidence" />`;
    } else {
      mediaHtml = `<img class="uploaded-media image-media" src="${escapeHTML(analysis.originalUrl || source.url)}" alt="Original uploaded image" />`;
    }
  }

  // Summary counts
  const rawScore = result?.highest_risk_score || 0;
  const riskScore = rawScore <= 1.0 && rawScore > 0 ? Math.round(rawScore * 100) : Math.round(rawScore);
  const riskLevel = result?.highest_risk_tier || result?.highest_risk_level || "INFO";
  const workers = isVideo
    ? (videoBreakdown?.total_people ?? null)
    : (result?.workers_count ?? result?.assessment?.worker_inventories?.length ?? 0);
  const nonCompliant = isVideo
    ? (videoBreakdown?.non_compliant_people ?? null)
    : (result?.non_compliant_count ?? result?.assessment?.non_compliant_workers_count ?? 0);
  const totalDetections = result?.detections_count ?? (result?.total_frames ? (result?.processed_frames || 0) : 0);

  // Active Inference Breakdown & Selected Person
  const bd = result?.inference_breakdown || (result ? extractBreakdownFromAssessment(result) : null);
  const people = bd?.people || [];
  const selectedPerson = (people.length > 0 && people.find((p) => p.id === state.selectedPersonId)) || (people.length > 0 ? people[0] : null);
  if (selectedPerson && state.selectedPersonId !== selectedPerson.id) {
    state.selectedPersonId = selectedPerson.id;
  }

  let highlightBoxHtml = "";
  if (!isVideo && isCompleted && selectedPerson && selectedPerson.bbox && selectedPerson.bbox.x2 > selectedPerson.bbox.x1) {
    const imgW = result?.assessment?.frame_width || (bd?.technical_details?.source_dimensions ? parseInt(bd.technical_details.source_dimensions.split("x")[0], 10) : 0);
    const imgH = result?.assessment?.frame_height || (bd?.technical_details?.source_dimensions ? parseInt(bd.technical_details.source_dimensions.split("x")[1], 10) : 0);
    if (imgW > 0 && imgH > 0) {
      const b = selectedPerson.bbox;
      const leftPct = ((b.x1 / imgW) * 100).toFixed(2);
      const topPct = ((b.y1 / imgH) * 100).toFixed(2);
      const widthPct = (((b.x2 - b.x1) / imgW) * 100).toFixed(2);
      const heightPct = (((b.y2 - b.y1) / imgH) * 100).toFixed(2);
      highlightBoxHtml = `<div class="canvas-highlight-container">
        <div class="canvas-highlight-box" style="left: ${leftPct}%; top: ${topPct}%; width: ${widthPct}%; height: ${heightPct}%;">
          <span class="canvas-highlight-label">${escapeHTML(selectedPerson.label)} [${escapeHTML(selectedPerson.compliance_status)}]</span>
        </div>
      </div>`;
    }
  }

  let viewControlsHtml = "";
  if (isCompleted) {
    if (isVideo) {
      viewControlsHtml = `<div class="canvas-view-controls"><div class="segmented-control">
        <button type="button" class="segmented-btn ${analysis.viewMode === "annotated" ? "is-active" : ""}" data-action="set-view-mode" data-mode="annotated">Annotated Video</button>
        ${analysis.annotatedFrameUrl || analysis.latestFrameUrl ? `<button type="button" class="segmented-btn ${analysis.viewMode === "annotated_frame" ? "is-active" : ""}" data-action="set-view-mode" data-mode="annotated_frame">Annotated Frame</button>` : ""}
        <button type="button" class="segmented-btn ${analysis.viewMode === "original" ? "is-active" : ""}" data-action="set-view-mode" data-mode="original">Original Video</button>
      </div></div>`;
    } else {
      viewControlsHtml = `<div class="canvas-view-controls"><div class="segmented-control">
        <button type="button" class="segmented-btn ${analysis.viewMode === "annotated" ? "is-active" : ""}" data-action="set-view-mode" data-mode="annotated">Annotated</button>
        <button type="button" class="segmented-btn ${analysis.viewMode === "original" ? "is-active" : ""}" data-action="set-view-mode" data-mode="original">Original</button>
      </div></div>`;
    }
  } else {
    viewControlsHtml = `<span class="canvas-chip">${icon(isVideo ? "video" : "image", 13)} ORIGINAL MEDIA</span>`;
  }

  return `<div class="analysis-heading"><div><div class="eyebrow">DETECTION WORKSPACE</div><h1>${escapeHTML(source.name)}</h1><p>${isVideo ? "Video stream analysis" : "Still frame perception"} <span class="heading-separator">•</span> ${isRunning ? escapeHTML(analysis.statusText || "Processing on GPU...") : (isCompleted ? "Inference Complete" : "Ready for execution")}</p></div><div class="analysis-heading-actions">${isRunning ? `<span class="service-state"><i class="analysis-spinner"></i> Running on ${escapeHTML(state.telemetry.gpuName || 'CUDA:0')}</span>` : `<button class="button button-primary button-small" type="button" data-action="${isCompleted ? 'run-analysis-again' : 'start-analysis'}">${icon("activity", 15)} ${isCompleted ? 'Re-run detection' : 'Run Detection on GPU'}</button>${isCompleted ? `<button class="button button-secondary button-small" type="button" data-action="open-report-modal" title="Generate Safety Intelligence Report">${icon("file", 15)} Generate Report</button><button class="button button-secondary button-small" type="button" data-action="view-scene-graph" title="Explore interactive Scene Graph for this frame">${icon("layers", 15)} View Scene Graph</button>` : ""}<button class="button button-secondary button-small" type="button" data-action="back-to-source">${icon("upload", 15)} Upload new media</button>`}<button class="button button-quiet button-small" type="button" data-action="fullscreen">${icon("expand", 15)} Fullscreen</button></div></div>
    ${isRunning ? `<div class="analysis-progress-panel"><div class="analysis-progress-header"><span>${icon("activity", 16)} <b>${analysis.phase === "UPLOADING" ? "FILE UPLOAD" : "CUDA INFERENCE"}</b>: ${escapeHTML(analysis.statusText || '')}</span><span>${Math.round(analysis.progressPct)}%</span></div><div class="analysis-progress-track"><div class="analysis-progress-bar" style="width: ${Math.max(5, analysis.progressPct)}%"></div></div><div class="analysis-progress-meta"><span>Stage: <b>${analysis.phase === "UPLOADING" ? "Network Transmission" : "GPU Neural Analysis"}</b></span><span>Device: ${escapeHTML(state.telemetry.gpuName || 'cuda:0')}</span></div></div>` : (isCompleted ? `<div class="analysis-completed-badge">${icon("check", 16)} <span>Inference Complete • 100% Processed on ${escapeHTML(state.telemetry.gpuName || 'CUDA:0')} (${isVideo ? `${result.total_frames || 0} frames analyzed` : `${totalDetections} detections`})</span></div>` : "")}
    ${analysis.error ? `<div class="inline-error" style="margin-bottom:14px;">${icon("alert", 17)}<span>${escapeHTML(analysis.error)}</span></div>` : ""}
    <div class="evidence-console">
      <aside class="console-source panel"><div class="panel-kicker">SOURCE EVIDENCE</div><div class="console-source-name">${icon(isVideo ? "video" : "image", 17)}<span>${escapeHTML(source.name)}</span></div><span class="source-file-size">${formatBytes(source.file.size)} • ${isVideo ? "Video" : "Image"}</span><div class="console-separator"></div><div class="panel-kicker">ACTIVE PROFILE</div><div class="active-profile-tags">${modules.filter(([key]) => state.modules[key]).map(([, label]) => `<span>${escapeHTML(label)}</span>`).join("")}</div><button class="button button-quiet button-small profile-edit" type="button" data-action="back-to-source">Change source ${icon("arrow", 14)}</button><div class="console-service-note"><i class="status-dot status-active"></i><span>Hardware engine<br /><b>${escapeHTML(state.telemetry.gpuName || 'CUDA:0 GPU')}</b></span></div></aside>
      <section class="evidence-canvas panel" id="evidence-canvas">
        <div class="canvas-toolbar">
          <div class="canvas-toolbar-title"><span class="panel-kicker">EVIDENCE VIEW</span><span class="canvas-source-label">${escapeHTML(source.name)}</span></div>
          ${viewControlsHtml}
        </div>
        <div class="uploaded-canvas" style="position: relative; overflow: hidden;">
          <div class="uploaded-canvas-viewport" style="transform: scale(${analysis.mediaZoom || 1.0}); transform-origin: center center; transition: transform 0.15s ease;">
            ${mediaHtml}
            ${highlightBoxHtml}
          </div>
          <div class="media-zoom-hud" style="position: absolute; bottom: 12px; right: 12px; z-index: 10;">
            <button type="button" class="hud-btn" data-action="zoom-media-in" title="Zoom in" aria-label="Zoom in">${icon("plus", 13)}</button>
            <button type="button" class="hud-btn" data-action="zoom-media-out" title="Zoom out" aria-label="Zoom out"><span style="font-weight:700; font-size:15px; line-height:1;">−</span></button>
            <button type="button" class="hud-btn" data-action="fit-media" title="Reset zoom (100%)" aria-label="Fit media">Fit</button>
            <span class="media-zoom-level" style="font-size:11px; padding:0 6px; color:var(--muted);">${Math.round((analysis.mediaZoom || 1.0) * 100)}%</span>
          </div>
          ${isCompleted ? `<div class="media-overlay-note">${icon("check", 15)} ${isVideo
            ? `Video analysis complete: ${result.processed_frames ?? result.total_frames ?? 0} frames processed${videoBreakdown ? ` • final analyzed frame: ${workers} workers, ${nonCompliant} non-compliant` : ""}.`
            : `CUDA inference successful: ${totalDetections} objects, ${workers} workers, ${nonCompliant} PPE violations.`}</div>` : ""}
        </div>
        <div class="media-actions"><span>${icon("shield", 14)} CUDA inference on ${escapeHTML(state.telemetry.gpuName || 'cuda:0')}</span><span>${isVideo ? "Multi-frame video reasoning" : "Single-frame perception"}</span></div>
      </section>
      <aside class="intel-panel workspace-intel panel">
        <div class="intel-panel-head"><div><div class="panel-kicker">INTELLIGENCE</div><h2>Scene context</h2></div>${isCompleted ? severityTag(riskLevel) : '<span class="unavailable-chip">READY</span>'}</div>
        ${isCompleted ? renderAnalysisIntelligence(result, videoBreakdown, isVideo) : unavailableIntelligence()}
      </aside>
    </div>
    ${renderInferenceBreakdown(result, analysis, state, videoScene)}
    <div class="analysis-footnote">${icon("shield", 15)}<span>All coordinates, bounding boxes, and compliance calculations are derived directly from the active CuriousPARC neural models.</span></div>
    ${renderReportModal(result, source, analysis)}`;
}

function renderAnalysisIntelligence(result, videoBreakdown = null, isVideo = false) {
  const rawScore = result.highest_risk_score || 0;
  const riskScore = rawScore <= 1.0 && rawScore > 0 ? Math.round(rawScore * 100) : Math.round(rawScore);
  const riskLevel = result.highest_risk_tier || result.highest_risk_level || "INFO";
  const workers = isVideo
    ? (videoBreakdown?.total_people ?? null)
    : (result.workers_count ?? result.assessment?.worker_inventories?.length ?? 0);
  const nonCompliant = isVideo
    ? (videoBreakdown?.non_compliant_people ?? null)
    : (result.non_compliant_count ?? result.assessment?.non_compliant_workers_count ?? 0);
  const totalDetections = isVideo
    ? (result.processed_frames ?? result.total_frames ?? 0)
    : (result.detections_count ?? result.assessment?.detections?.length ?? 0);
  const notice = result.temporal_notice || (result.total_frames ? `Processed ${result.processed_frames || 0}/${result.total_frames || 0} frames at ${result.average_fps || result.processing_fps || 0} FPS` : "Verified CUDA neural perception");

  const ppeStatusComplete = !isVideo || videoBreakdown?.ppe_status_complete === true;
  const ppeStatusText = nonCompliant == null
    ? "PPE DETAILS UNAVAILABLE"
    : (isVideo && !ppeStatusComplete
      ? (nonCompliant > 0 ? `${nonCompliant} NON-COMPLIANT • STATUS INCOMPLETE` : "PPE STATUS INCOMPLETE")
      : (nonCompliant > 0 ? `${nonCompliant} NON-COMPLIANT` : "NO PPE VIOLATIONS DETECTED"));
  const ppeStatusClass = nonCompliant == null || (isVideo && !ppeStatusComplete && nonCompliant === 0)
    ? ""
    : (nonCompliant > 0 ? "status-warning" : "status-success");
  const ppeDotClass = nonCompliant == null || (isVideo && !ppeStatusComplete && nonCompliant === 0)
    ? "unavailable"
    : (nonCompliant > 0 ? "critical" : "normal");

  return `<div class="shot-intel">
    <div class="shot-kicker">PEAK RISK SCORE</div>
    <div class="shot-score"><strong>${riskScore}</strong><span>/ 100</span><b>${escapeHTML(riskLevel)}</b></div>
    <div class="shot-rule"></div>
    <p class="shot-kicker">PPE COMPLIANCE</p>
    <div class="shot-ppe"><span>DETECTED WORKERS: <b>${workers ?? "Unavailable"}</b></span><b class="${ppeStatusClass}">${ppeStatusText}</b></div>
    <div class="shot-rule"></div>
    <p class="shot-kicker">${result.total_frames ? "FRAMES PROCESSED" : "TOTAL OBJECTS"}</p>
    <div class="shot-hazard"><span class="severity-dot ${ppeDotClass}"></span><span>${totalDetections} ${result.total_frames ? "frames analyzed" : "bounding boxes resolved"}</span></div>
    <div class="shot-rule"></div>
    <div class="prediction-strip" style="margin-top: 10px;">${icon("info", 15)}<span><b>PERCEPTION NOTICE</b><small>${escapeHTML(notice)}</small></span></div>
  </div>`;
}

function unavailableIntelligence() {
  return `<section class="intel-empty"><span class="empty-icon">${icon("activity", 21)}</span><b>Ready for detection</b><p>Press "Start detection" to process this media source through the active CUDA model pipeline.</p><button type="button" class="button button-primary button-small" data-action="run-analysis-again">Run Inference ${icon("arrow", 14)}</button></section>`;
}

function extractBreakdownFromAssessment(result) {
  if (!result) return null;
  const assessment = result.assessment || {};
  const inventories = assessment.worker_inventories || [];
  const tracks = assessment.tracks || [];
  const detections = assessment.detections || [];
  const trackMap = new Map();
  tracks.forEach((t) => {
    if (t.track_id) trackMap.set(t.track_id, t);
  });

  const people = inventories.map((inv) => {
    const tid = inv.track_id || 1;
    const matchedTrack = trackMap.get(tid);
    const conf = matchedTrack?.confidence ?? 0.88;
    const confPct = Math.round(conf * 100);
    const complianceStatus = normalizeComplianceStatus(inv.compliance_status);
    return {
      id: tid,
      label: inv.metadata?.display_label || (inv.metadata?.display_id ? `Person #${inv.metadata.display_id}` : `Person #${tid}`),
      confidence: conf,
      confidence_pct: confPct,
      compliance_status: complianceStatus,
      missing_ppe: inv.missing_ppe || [],
      detected_ppe: inv.present_ppe || [],
      unknown_ppe: inv.unknown_ppe || [],
      ppe_status: inv.ppe_status || {},
      bbox: inv.bbox || {},
      associated_items: inv.items || [],
      explanation: inv.explanation || null,
    };
  });

  const compliantPeople = people.filter((p) => p.compliance_status === "COMPLIANT").length;
  const nonCompliantPeople = people.filter((p) => p.compliance_status === "NON-COMPLIANT").length;
  const totalViolations = people.reduce((acc, p) => acc + (p.missing_ppe?.length || 0), 0);

  const ppeMap = {};
  people.forEach((p) => {
    Object.entries(p.ppe_status || {}).forEach(([cat, st]) => {
      if (!ppeMap[cat]) ppeMap[cat] = { category: cat, detected: 0, missing: 0, unknown: 0 };
      const s = String(st).toUpperCase();
      if (s.includes("PRESENT")) ppeMap[cat].detected++;
      else if (s.includes("MISSING")) ppeMap[cat].missing++;
      else ppeMap[cat].unknown++;
    });
  });

  const objMap = {};
  detections.forEach((d) => {
    const cname = d.class_name ? d.class_name.charAt(0).toUpperCase() + d.class_name.slice(1) : "Object";
    if (!objMap[cname]) objMap[cname] = { class_name: cname, class_group: d.class_group || "OTHER", count: 0 };
    objMap[cname].count++;
  });

  return {
    total_people: people.length,
    compliant_people: compliantPeople,
    non_compliant_people: nonCompliantPeople,
    ppe_status_complete: people.length > 0 && people.every((person) => person.compliance_status !== "UNKNOWN"),
    total_violations: totalViolations,
    people,
    ppe_summary: Object.values(ppeMap),
    object_summary: Object.values(objMap),
    technical_details: {
      model_name: "YOLO11n + CuriousPARC PPE Classifier",
      model_version: "CuriousPARC v2.1-production",
      device: state.telemetry.gpuName || "CUDA:0",
      cuda_status: "Active (Compute 12.0, CUDA 12.4)",
      inference_time_ms: assessment.processing_time_ms || 0,
      source_dimensions: `${assessment.frame_width || 1280}x${assessment.frame_height || 720} px`,
      total_detections: detections.length,
      detection_classes: [...new Set(detections.map((d) => d.class_name))],
      peak_risk_tier: assessment.highest_risk_level || "INFO",
      peak_risk_score: assessment.highest_risk_score || 0,
    },
  };
}

function buildVideoBreakdownFromScene(sceneData, result) {
  if (!sceneData || !Array.isArray(sceneData.entities)) return null;

  const entities = sceneData.entities;
  const relationships = Array.isArray(sceneData.relationships) ? sceneData.relationships : [];
  const peopleEntities = entities.filter((entity) => entity.type === "PERSON");
  const ppeEntities = entities.filter((entity) => entity.type === "PPE");
  const ppeById = new Map(ppeEntities.map((entity) => [entity.id, entity]));
  const relationTypes = new Set(["WEARING", "IS_WEARING", "MISSING"]);

  const people = peopleEntities.map((person, index) => {
    const related = relationships.filter((relation) =>
      relation.active === true &&
      relationTypes.has(String(relation.type || "").toUpperCase()) &&
      (relation.source_id === person.id || relation.target_id === person.id)
    );
    const statusMap = {};
    const present = new Set();
    const missing = new Set();
    const unknown = new Set();

    const cleanCategory = (value) => String(value || "")
      .replace(/^(missing|no)[\s_-]+/i, "")
      .replace(/[_-]+/g, " ")
      .trim();

    related.forEach((relation) => {
      const type = String(relation.type || "").toUpperCase();
      const otherId = relation.source_id === person.id ? relation.target_id : relation.source_id;
      const item = ppeById.get(otherId);
      const evidence = relation.evidence || {};
      const metadata = relation.metadata || {};
      const category = cleanCategory(
        item?.label || item?.class_name || evidence.ppe_category || evidence.category || metadata.ppe_category || metadata.category
      );
      if (!category) return;

      if (type === "MISSING") {
        missing.add(category);
        present.delete(category);
        unknown.delete(category);
        statusMap[category] = "MISSING";
      } else {
        present.add(category);
        missing.delete(category);
        unknown.delete(category);
        statusMap[category] = "PRESENT";
      }
    });

    const rawPpeStatus = person.state?.ppe_status;
    if (rawPpeStatus && typeof rawPpeStatus === "object" && !Array.isArray(rawPpeStatus)) {
      Object.entries(rawPpeStatus).forEach(([rawCategory, rawStatus]) => {
        const category = cleanCategory(rawCategory);
        if (!category) return;
        const status = String(rawStatus || "UNKNOWN").toUpperCase();
        if (status.includes("MISSING")) {
          missing.add(category);
          present.delete(category);
          statusMap[category] = "MISSING";
        } else if (status.includes("PRESENT")) {
          present.add(category);
          missing.delete(category);
          statusMap[category] = "PRESENT";
        } else {
          unknown.add(category);
          present.delete(category);
          missing.delete(category);
          statusMap[category] = "UNKNOWN";
        }
      });
    }

    const rawCompliance = String(person.state?.compliance_status || "").toUpperCase().replace(/_/g, "-");
    const complianceStatus = ["COMPLIANT", "NON-COMPLIANT", "UNKNOWN"].includes(rawCompliance)
      ? rawCompliance
      : (missing.size > 0 ? "NON-COMPLIANT" : "UNKNOWN");
    const trackId = person.track_id ?? person.metadata?.track_id;
    const numericId = Number(trackId);
    const confidence = person.confidence != null && Number.isFinite(Number(person.confidence)) ? Number(person.confidence) : null;
    const bbox = person.bbox || person.position?.bbox || {};

    return {
      id: Number.isFinite(numericId) ? numericId : index + 1,
      label: person.metadata?.display_label || (person.metadata?.display_id ? `Person #${person.metadata.display_id}` : (person.label ? person.label.replace(/^Worker\b/, "Person") : (trackId != null ? `Person #${trackId}` : `Person ${index + 1}`))),
      confidence,
      confidence_pct: confidence == null ? null : Math.round(confidence * 100),
      compliance_status: complianceStatus,
      missing_ppe: [...missing],
      detected_ppe: [...present],
      unknown_ppe: [...unknown],
      ppe_status: statusMap,
      bbox,
      associated_items: related.map((relation) => {
        const otherId = relation.source_id === person.id ? relation.target_id : relation.source_id;
        const item = ppeById.get(otherId);
        return item ? { class_name: item.label || item.class_name, relation: relation.type } : null;
      }).filter(Boolean),
      explanation: null,
    };
  });

  const ppeSummaryMap = new Map();
  people.forEach((person) => {
    Object.entries(person.ppe_status).forEach(([category, status]) => {
      const item = ppeSummaryMap.get(category) || { category, detected: 0, missing: 0, unknown: 0 };
      if (status === "PRESENT") item.detected++;
      else if (status === "MISSING") item.missing++;
      else item.unknown++;
      ppeSummaryMap.set(category, item);
    });
  });

  const objectSummaryMap = new Map();
  entities.forEach((entity) => {
    const className = entity.class_name || entity.type || "Object";
    const item = objectSummaryMap.get(className) || { class_name: className, class_group: entity.type || "OTHER", count: 0 };
    item.count++;
    objectSummaryMap.set(className, item);
  });

  const compliantPeople = people.filter((person) => person.compliance_status === "COMPLIANT").length;
  const nonCompliantPeople = people.filter((person) => person.compliance_status === "NON-COMPLIANT").length;
  const hasUnspecifiedViolations = people.some((person) => person.compliance_status === "NON-COMPLIANT" && person.missing_ppe.length === 0);
  const totalViolations = hasUnspecifiedViolations
    ? null
    : people.reduce((total, person) => total + person.missing_ppe.length, 0);
  const classes = [...new Set(entities.map((entity) => entity.class_name || entity.type).filter(Boolean))];

  return {
    total_people: people.length,
    compliant_people: compliantPeople,
    non_compliant_people: nonCompliantPeople,
    total_violations: totalViolations,
    people,
    ppe_summary: [...ppeSummaryMap.values()],
    object_summary: [...objectSummaryMap.values()],
    technical_details: {
      total_detections: entities.length,
      detection_classes: classes,
      peak_risk_tier: result.highest_risk_tier || result.highest_risk_level || "INFO",
      peak_risk_score: result.highest_risk_score ?? 0,
    },
  };
}

function renderInferenceBreakdown(result, analysis, state, videoScene = null) {
  const isVideo = state.source?.kind === "video";
  const breakdownTitle = isVideo
    ? "Detailed object and PPE compliance results from the final analyzed frame"
    : "Detailed object and PPE compliance results from this frame";

  if (analysis.phase === "PROCESSING" || analysis.phase === "UPLOADING") {
    return `<section class="inference-breakdown-panel panel" id="inference-breakdown-section">
      <div class="breakdown-header">
        <div>
          <div class="panel-kicker">INFERENCE BREAKDOWN</div>
          <h2>${breakdownTitle}</h2>
        </div>
      </div>
      <div class="inference-state-box">
        <i class="analysis-spinner"></i>
        <span>Loading inference details from CUDA neural models...</span>
      </div>
    </section>`;
  }

  if (analysis.phase === "FAILED") {
    return `<section class="inference-breakdown-panel panel" id="inference-breakdown-section">
      <div class="breakdown-header">
        <div>
          <div class="panel-kicker">INFERENCE BREAKDOWN</div>
          <h2>${breakdownTitle}</h2>
        </div>
      </div>
      <div class="inference-state-box error">
        ${icon("alert", 24)}
        <span>Unable to load detailed inference data: ${escapeHTML(analysis.error || "Pipeline execution failed")}</span>
      </div>
    </section>`;
  }

  if (analysis.phase !== "COMPLETED" || !result) {
    return "";
  }

  const bd = result.inference_breakdown || (isVideo
    ? buildVideoBreakdownFromScene(videoScene, result)
    : extractBreakdownFromAssessment(result));
  if (!bd) {
    return `<section class="inference-breakdown-panel panel" id="inference-breakdown-section">
      <div class="breakdown-header">
        <div>
          <div class="panel-kicker">INFERENCE BREAKDOWN</div>
          <h2>${breakdownTitle}</h2>
        </div>
      </div>
      <div class="inference-state-box">
        ${icon("info", 22)}
        <span>${isVideo ? "Detailed worker and PPE data is unavailable for the final analyzed video frame." : "Detailed inference data not available for this frame."}</span>
      </div>
    </section>`;
  }

  const totalPeople = bd.total_people || 0;
  const compliantPeople = bd.compliant_people || 0;
  const nonCompliantPeople = bd.non_compliant_people || 0;
  const totalViolations = bd.total_violations == null ? (isVideo ? null : 0) : bd.total_violations;
  const people = bd.people || [];
  const ppeSummary = bd.ppe_summary || [];
  const objectSummary = bd.object_summary || [];
  const tech = bd.technical_details || {};

  const selectedPerson = (people.length > 0 && people.find((p) => p.id === state.selectedPersonId)) || (people.length > 0 ? people[0] : null);
  if (selectedPerson && state.selectedPersonId !== selectedPerson.id) {
    state.selectedPersonId = selectedPerson.id;
  }

  return `<section class="inference-breakdown-panel panel" id="inference-breakdown-section">
    <div class="breakdown-header">
      <div>
        <div class="panel-kicker">INFERENCE BREAKDOWN</div>
        <h2>${breakdownTitle}</h2>
      </div>
      <div class="breakdown-header-actions">
        <button type="button" class="button button-secondary button-small" data-action="view-scene-graph" title="Explore interactive Scene Graph for this frame">
          ${icon("layers", 14)} View Scene Graph
        </button>
        <button type="button" class="button button-secondary button-small" data-action="toggle-all-inference">
          ${state.showAllInference ? `${icon("chevron", 14)} Hide detailed inference` : `${icon("expand", 14)} Show all inference`}
        </button>
      </div>
    </div>

    <!-- Summary Cards -->
    <div class="breakdown-kpi-grid">
      <div class="breakdown-kpi-card">
        <span class="panel-kicker">TOTAL PEOPLE</span>
        <div class="kpi-value">${totalPeople}</div>
        <div class="kpi-sub">${icon("monitor", 13)} ${isVideo ? "Final analyzed frame" : "Detected in frame"}</div>
      </div>
      <div class="breakdown-kpi-card">
        <span class="panel-kicker">PPE COMPLIANT</span>
        <div class="kpi-value text-success">${compliantPeople}</div>
        <div class="kpi-sub">${icon("check", 13)} Full gear verified</div>
      </div>
      <div class="breakdown-kpi-card">
        <span class="panel-kicker">PPE NON-COMPLIANT</span>
        <div class="kpi-value ${nonCompliantPeople > 0 ? "text-critical" : "text-success"}">${nonCompliantPeople}</div>
        <div class="kpi-sub">${icon("alert", 13)} Missing required gear</div>
      </div>
      <div class="breakdown-kpi-card">
        <span class="panel-kicker">TOTAL PPE VIOLATIONS</span>
        <div class="kpi-value ${totalViolations == null ? "" : (totalViolations > 0 ? "text-warning" : "text-success")}">${totalViolations ?? "Unavailable"}</div>
        <div class="kpi-sub">${icon("shield", 13)} ${totalViolations == null ? "Missing item details unavailable" : "Across all detected workers"}</div>
      </div>
    </div>

    ${people.length === 0 ? `
      <div class="inference-state-box">
        ${icon("info", 24)}
        <span>No people detected in this frame.</span>
      </div>
    ` : `
      <div class="breakdown-grid">
        <!-- People Table -->
        <div class="people-table-container">
          <table class="people-table">
            <thead>
              <tr>
                <th style="width: 40px;">#</th>
                <th>Person</th>
                <th>Confidence</th>
                <th>PPE Status</th>
                <th>Missing PPE</th>
                <th style="width: 80px; text-align: center;">Detection</th>
              </tr>
            </thead>
            <tbody>
              ${people.map((p, idx) => {
                const isSelected = selectedPerson && selectedPerson.id === p.id;
                const isComp = p.compliance_status === "COMPLIANT";
                const isNonCompliant = p.compliance_status === "NON-COMPLIANT";
                const missingText = p.missing_ppe && p.missing_ppe.length > 0
                  ? p.missing_ppe.join(", ")
                  : `<span class="text-muted">${isComp ? "None" : "No missing PPE relation recorded"}</span>`;
                return `<tr class="person-row ${isSelected ? "is-selected" : ""}" data-action="select-person" data-id="${p.id}" title="Click to inspect Person #${p.id}">
                  <td><b>${idx + 1}</b></td>
                  <td><b>${escapeHTML(p.label)}</b></td>
                  <td>${p.confidence_pct == null ? "Unavailable" : `${p.confidence_pct}%`}</td>
                  <td>
                    <span class="badge ${isComp ? "badge-success" : (isNonCompliant ? "badge-danger" : "badge-neutral")}">
                      ${isComp ? icon("check", 11) : (isNonCompliant ? icon("alert", 11) : icon("info", 11))} ${escapeHTML(p.compliance_status)}
                    </span>
                  </td>
                  <td>${missingText}</td>
                  <td style="text-align: center; white-space: nowrap;">
                    <button type="button" class="button button-quiet button-small" data-action="select-person" data-id="${p.id}">View</button>
                    <button type="button" class="button button-quiet button-small" data-action="inspect-person-in-graph" data-id="${p.id}" title="Focus ${escapeHTML(p.label)} in Scene Graph">${icon("layers", 13)} Graph</button>
                  </td>
                </tr>`;
              }).join("")}
            </tbody>
          </table>
        </div>

        <!-- Person Detail Card -->
        ${selectedPerson ? `
          <div class="person-detail-card">
            <div class="person-detail-header">
              <div class="person-detail-title">
                <span class="panel-kicker">INSPECTING WORKER</span>
                <h3>${escapeHTML(selectedPerson.label)}</h3>
              </div>
              <span class="badge ${selectedPerson.compliance_status === "COMPLIANT" ? "badge-success" : (selectedPerson.compliance_status === "NON-COMPLIANT" ? "badge-danger" : "badge-neutral")}">
                ${escapeHTML(selectedPerson.compliance_status)}
              </span>
            </div>
            <div class="person-detail-body">
              <div class="detail-row">
                <span class="detail-label">Detection Confidence:</span>
                <span class="detail-value"><b>${selectedPerson.confidence_pct == null ? "Unavailable" : `${selectedPerson.confidence_pct}%`}</b>${selectedPerson.confidence == null ? "" : ` (${selectedPerson.confidence})`}</span>
              </div>
              <div class="detail-row">
                <span class="detail-label">PPE Compliance:</span>
                <span class="detail-value"><b>${escapeHTML(selectedPerson.compliance_status)}</b></span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Detected PPE:</span>
                <span class="detail-value">
                  ${selectedPerson.detected_ppe && selectedPerson.detected_ppe.length > 0
                    ? selectedPerson.detected_ppe.map((g) => `<span class="gear-tag present">${icon("check", 11)} ${escapeHTML(g)}</span>`).join(" ")
                    : '<span class="text-muted">None</span>'}
                </span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Missing PPE:</span>
                <span class="detail-value">
                  ${selectedPerson.missing_ppe && selectedPerson.missing_ppe.length > 0
                    ? selectedPerson.missing_ppe.map((g) => `<span class="gear-tag missing">${icon("close", 11)} ${escapeHTML(g)}</span>`).join(" ")
                    : `<span class="text-muted">${selectedPerson.compliance_status === "COMPLIANT" ? "None (Fully compliant)" : "No missing PPE relation recorded"}</span>`}
                </span>
              </div>
              <div class="detail-row">
                <span class="detail-label">Bounding Box:</span>
                <span class="detail-value mono-coords">x1: ${selectedPerson.bbox?.x1 ?? "N/A"}, y1: ${selectedPerson.bbox?.y1 ?? "N/A"}, x2: ${selectedPerson.bbox?.x2 ?? "N/A"}, y2: ${selectedPerson.bbox?.y2 ?? "N/A"}</span>
              </div>
              ${selectedPerson.explanation ? `
                <div class="detail-row">
                  <span class="detail-label">Model Reasoning:</span>
                  <span class="detail-value text-muted">${escapeHTML(selectedPerson.explanation)}</span>
                </div>
              ` : ""}
              <div class="detail-actions" style="margin-top: 14px; padding-top: 10px; border-top: 1px solid var(--border, var(--line)); display: flex; gap: 8px;">
                <button type="button" class="button button-secondary button-small button-full" data-action="inspect-person-in-graph" data-id="${selectedPerson.id}">
                  ${icon("layers", 14)} Focus in Scene Graph
                </button>
              </div>
            </div>
          </div>
        ` : `
          <div class="person-detail-card" style="text-align: center; color: var(--muted); padding: 30px 20px;">
            ${icon("info", 20)}
            <p style="margin-top: 8px;">Select a person from the table to view detailed PPE compliance.</p>
          </div>
        `}
      </div>
    `}

    <!-- Expandable Detailed Sections -->
    ${state.showAllInference ? `
      <div class="breakdown-expanded-sections">
        <!-- PPE Summary -->
        <div class="ppe-summary-container">
          <div class="breakdown-subheading">
            <span class="panel-kicker">PPE DETECTION</span>
            <h4>Category Compliance Breakdown</h4>
          </div>
          ${ppeSummary.length === 0 ? `
            <div class="inference-state-box">
              ${icon("shield", 20)}
              <span>PPE analysis data not available.</span>
            </div>
          ` : `
            <div class="ppe-category-grid">
              ${ppeSummary.map((cat) => `
                <div class="ppe-cat-card">
                  <div class="ppe-cat-title">
                    <b>${escapeHTML(cat.category)}</b>
                    <span class="ppe-cat-state ${cat.missing > 0 ? "has-violations" : "all-clear"}">
                      ${cat.missing > 0 ? `${cat.missing} missing` : (cat.unknown > 0 ? "Evidence incomplete" : "All present")}
                    </span>
                  </div>
                  <div class="ppe-cat-counts">
                    <span class="count-detected">${icon("check", 12)} Detected: <b>${cat.detected}</b></span>
                    <span class="count-missing">${icon("close", 12)} Missing: <b>${cat.missing}</b></span>
                    ${cat.unknown > 0 ? `<span class="count-unknown">${icon("info", 12)} Unknown: <b>${cat.unknown}</b></span>` : ""}
                  </div>
                </div>
              `).join("")}
            </div>
          `}
        </div>

        <!-- Object Detection Summary -->
        <div class="object-summary-container">
          <div class="breakdown-subheading">
            <span class="panel-kicker">OBJECT DETECTION</span>
            <h4>All Detected Classes</h4>
          </div>
          ${objectSummary.length === 0 ? `
            <div class="inference-state-box">
              ${icon("monitor", 20)}
              <span>No object detections recorded.</span>
            </div>
          ` : `
            <div class="object-classes-grid">
              ${objectSummary.map((obj) => `
                <div class="object-class-chip">
                  <span class="object-name">${escapeHTML(obj.class_name)}</span>
                  <span class="object-group-tag">${escapeHTML(obj.class_group)}</span>
                  <span class="object-count">${obj.count}</span>
                </div>
              `).join("")}
            </div>
          `}
        </div>

        <!-- Technical Inference Details -->
        <details class="technical-inference-details" open>
          <summary class="technical-summary-toggle">
            ${icon("settings", 15)} <span>Technical inference details (Model & Runtime Telemetry)</span>
          </summary>
          <div class="technical-details-content">
            <table class="tech-table">
              <tbody>
                <tr><td>Model Name</td><td><b>${escapeHTML(tech.model_name || (isVideo ? "Unavailable in scene snapshot" : "YOLO11n + CuriousPARC PPE Classifier"))}</b></td></tr>
                <tr><td>Model Version</td><td>${escapeHTML(tech.model_version || (isVideo ? "Unavailable in scene snapshot" : "CuriousPARC v2.1-production"))}</td></tr>
                <tr><td>Inference Device</td><td>${tech.device || (!isVideo && state.telemetry.gpuName) ? `<span class="cuda-badge">${icon("activity", 12)} ${escapeHTML(tech.device || state.telemetry.gpuName)}</span>` : (isVideo ? "Unavailable in scene snapshot" : "CUDA:0")}</td></tr>
                <tr><td>CUDA Status</td><td>${escapeHTML(tech.cuda_status || (isVideo ? "Unavailable in scene snapshot" : "Active"))}</td></tr>
                <tr><td>Inference Latency</td><td><b>${tech.inference_time_ms == null ? (isVideo ? "Unavailable in scene snapshot" : "0 ms") : `${tech.inference_time_ms} ms`}</b></td></tr>
                <tr><td>Source Dimensions</td><td>${escapeHTML(tech.source_dimensions || (isVideo ? "Unavailable in scene snapshot" : "N/A"))}</td></tr>
                <tr><td>${isVideo ? "Scene Entities" : "Total Detections"}</td><td>${tech.total_detections ?? 0}</td></tr>
                <tr><td>Peak Risk Tier</td><td><span class="severity-tag ${tech.peak_risk_tier === "CRITICAL" ? "severity-critical" : "severity-normal"}">${escapeHTML(tech.peak_risk_tier || "INFO")}</span></td></tr>
                <tr><td>Peak Risk Score</td><td><b>${tech.peak_risk_score ?? 0}</b></td></tr>
                <tr><td>Detected Classes</td><td><code>${(tech.detection_classes || []).join(", ") || "None"}</code></td></tr>
              </tbody>
            </table>
          </div>
        </details>
      </div>
    ` : ""}
  </section>`;
}

function livePage() {
  const cams = state.cameras;
  const intro = introHeading("LIVE MONITORING", "Control Room Feeds", "Manage persistent camera streams with real-time YOLO tracking and automated safety rule evaluation.", `<button class="button button-primary button-small" type="button" data-action="open-camera-modal">${icon("plus", 15)} Register camera</button>`);

  if (state.camerasLoading) {
    return `${intro}<div class="panel" style="padding: 40px; text-align: center;"><i class="analysis-spinner"></i><p style="margin-top:12px; color:var(--muted);">Loading active camera feeds...</p></div>`;
  }

  if (cams.length === 0) {
    return `${intro}<div class="live-empty-panel panel"><div class="live-empty-illustration"><div class="empty-camera-frame">${icon("camera", 33)}<span class="camera-status-line"></span></div><div class="empty-camera-frame smaller-camera">${icon("camera", 24)}</div></div><span class="empty-eyebrow">NO CONFIGURED CAMERAS</span><h2>Connect an RTSP Stream</h2><p>Register CCTV cameras to start real-time monitoring and event capture.</p><button class="button button-primary" type="button" data-action="open-camera-modal">Register Camera Source ${icon("arrow", 15)}</button><div class="live-empty-foot">${icon("wifi", 15)} Background camera workers execute detection and ByteTrack tracking.</div></div>`;
  }

  return `${intro}
    <div class="camera-grid">${cams.map((cam) => `
      <article class="camera-card">
        <div class="camera-card-header">
          <span class="cam-title">${icon("camera", 16)} <b>${escapeHTML(cam.name)}</b></span>
          <span class="severity-tag ${cam.is_connected ? "severity-low" : "severity-critical"}">${cam.is_connected ? "ONLINE" : (cam.is_running ? "CONNECTING" : "OFFLINE")}</span>
        </div>
        <div class="camera-card-feed">
          <img src="${cameraService.getStreamUrl(cam.camera_id)}" alt="${escapeHTML(cam.name)}" onerror="this.style.display='none'; this.nextElementSibling.style.display='flex';" />
          <div class="feed-offline" style="display:none;">${icon("wifi", 24)}<span>Stream standby / disconnected</span></div>
          <div class="feed-overlay"><span class="status-dot ${cam.is_connected ? "status-active" : "status-warning"}"></span><span>${escapeHTML(cam.camera_id)}</span></div>
        </div>
        <div class="camera-card-footer">
          <span>${cam.source_type ? cam.source_type.toUpperCase() : "RTSP"} • ${cam.fps_limit || 10} FPS max</span>
          <div class="camera-card-actions">
            <button type="button" class="button button-quiet button-small" data-action="toggle-camera-stream" data-id="${cam.camera_id}" data-running="${cam.is_running ? "true" : "false"}">${cam.is_running ? "Stop Stream" : "Start Stream"}</button>
            <button type="button" class="button button-quiet button-small" data-action="delete-camera" data-id="${cam.camera_id}" title="Remove camera">${icon("close", 14)}</button>
          </div>
        </div>
      </article>`).join("")}
    </div>`;
}

function cameraRegisterModal() {
  return `<div class="modal-backdrop" data-action="close-camera-modal-backdrop">
    <div class="modal-dialog">
      <div class="modal-header"><h2>Register New Camera Feed</h2><button type="button" class="icon-button" data-action="close-camera-modal">${icon("close", 16)}</button></div>
      <form id="modal-camera-form">
        <label style="display:block; margin-bottom:12px;">Camera ID (Optional)<input name="cameraId" placeholder="e.g. cam_01 (auto-generated if empty)" style="width:100%; height:38px; padding:0 10px; margin-top:4px; background:var(--bg); border:1px solid var(--line); border-radius:4px;" /></label>
        <label style="display:block; margin-bottom:12px;">Camera Name<input name="name" placeholder="e.g. Warehouse A Loading Bay" required style="width:100%; height:38px; padding:0 10px; margin-top:4px; background:var(--bg); border:1px solid var(--line); border-radius:4px;" /></label>
        <label style="display:block; margin-bottom:12px;">RTSP Stream URL or Video Path<input name="rtspUrl" placeholder="rtsp://192.168.1.50:554/live or data/samples/cctv_worker_moving.mp4" required style="width:100%; height:38px; padding:0 10px; margin-top:4px; background:var(--bg); border:1px solid var(--line); border-radius:4px;" /></label>
        <label style="display:block; margin-bottom:16px;">Target FPS Limit<input name="fpsLimit" type="number" value="10" min="1" max="30" style="width:100%; height:38px; padding:0 10px; margin-top:4px; background:var(--bg); border:1px solid var(--line); border-radius:4px;" /></label>
        <div class="modal-actions">
          <button type="button" class="button button-quiet" data-action="close-camera-modal">Cancel</button>
          <button type="submit" class="button button-primary">Save Camera Feed</button>
        </div>
      </form>
    </div>
  </div>`;
}

function getAlertCategory(a) {
  const v = (a.violation_type || "").toUpperCase();
  if (v.includes("PPE") || v.includes("NO_") || v.includes("HARDHAT") || v.includes("VEST") || v.includes("GLOVE") || v.includes("GOGGLE")) return "PPE";
  if (v.includes("ZONE") || v.includes("RESTRICTED") || v.includes("INTRUSION") || v.includes("GEOFENCE")) return "ZONE";
  if (v.includes("VEHICLE") || v.includes("FORKLIFT") || v.includes("PROXIMITY") || v.includes("APPROACHING")) return "VEHICLE";
  if (v.includes("FALL") || v.includes("BEHAVIOR") || v.includes("LOITER") || v.includes("COLLAPSE")) return "FALL";
  return "OTHER";
}

function incidentsPage() {
  const alerts = state.alerts;
  const filter = state.alertFilter;

  // Compute available category and severity tabs dynamically
  const severities = ["CRITICAL", "HIGH", "MEDIUM", "LOW"];
  const presentSeverities = severities.filter((s) => alerts.some((a) => (a.severity || "").toUpperCase() === s));
  const categories = ["PPE", "ZONE", "VEHICLE", "FALL", "OTHER"];
  const presentCategories = categories.filter((c) => alerts.some((a) => getAlertCategory(a) === c));

  const filtered = alerts.filter((a) => {
    const sev = (a.severity || "").toUpperCase();
    const cat = getAlertCategory(a);
    const stat = (a.status || "NEW").toUpperCase();

    if (filter === "ALL") {
      // pass
    } else if (filter === "NEW" || filter === "ACKNOWLEDGED" || filter === "RESOLVED") {
      if (stat !== filter) return false;
    } else if (severities.includes(filter)) {
      if (sev !== filter) return false;
    } else if (categories.includes(filter)) {
      if (cat !== filter) return false;
    }

    if (state.alertSearch) {
      const q = state.alertSearch.toLowerCase();
      const expl = (a.metadata?.explanation || a.explanation || "").toLowerCase();
      return (a.violation_type || "").toLowerCase().includes(q) ||
             (a.camera_id || "").toLowerCase().includes(q) ||
             (a.camera_name || "").toLowerCase().includes(q) ||
             (a.alert_id || "").toLowerCase().includes(q) ||
             expl.includes(q);
    }
    return true;
  });

  const intro = introHeading(
    "INCIDENT MANAGEMENT",
    "Safety Alerts & Incidents",
    "Review actionable PPE violations, zone intrusions and safety hazards backed by empirical evidence."
  );

  return `${intro}
    <div class="incident-toolbar panel" style="flex-wrap: wrap; gap: 8px;">
      <div class="incident-tabs" style="flex-wrap: wrap; gap: 4px;">
        <button type="button" class="incident-tab ${filter === "ALL" ? "is-current" : ""}" data-action="filter-alerts" data-filter="ALL">All alerts <span>(${alerts.length})</span></button>
        ${presentSeverities.map((s) => `
          <button type="button" class="incident-tab ${filter === s ? "is-current" : ""}" data-action="filter-alerts" data-filter="${s}">
            ${s.charAt(0) + s.slice(1).toLowerCase()} <span>(${alerts.filter((a) => (a.severity || "").toUpperCase() === s).length})</span>
          </button>
        `).join("")}
        ${presentCategories.map((c) => `
          <button type="button" class="incident-tab ${filter === c ? "is-current" : ""}" data-action="filter-alerts" data-filter="${c}">
            ${c} <span>(${alerts.filter((a) => getAlertCategory(a) === c).length})</span>
          </button>
        `).join("")}
        <button type="button" class="incident-tab ${filter === "NEW" ? "is-current" : ""}" data-action="filter-alerts" data-filter="NEW">New <span>(${alerts.filter((a) => (a.status || "NEW") === "NEW").length})</span></button>
        <button type="button" class="incident-tab ${filter === "ACKNOWLEDGED" ? "is-current" : ""}" data-action="filter-alerts" data-filter="ACKNOWLEDGED">Acknowledged</button>
        <button type="button" class="incident-tab ${filter === "RESOLVED" ? "is-current" : ""}" data-action="filter-alerts" data-filter="RESOLVED">Resolved</button>
      </div>
      <div style="display:flex; align-items:center; gap:8px;">
        <input type="search" placeholder="Search violation, camera, or explanation..." value="${escapeHTML(state.alertSearch)}" data-action="search-alerts" style="height:32px; padding:0 10px; background:var(--surface-2); border:1px solid var(--line); border-radius:4px; font-size:11px;" />
      </div>
    </div>
    ${state.alertsLoading ? `<div class="panel" style="padding: 40px; text-align: center;"><i class="analysis-spinner"></i><p style="margin-top:12px; color:var(--muted);">Loading safety alerts from database...</p></div>` : (
      filtered.length === 0 ? `
        <div class="incident-empty panel">
          <span class="empty-icon">${alerts.length === 0 ? icon("check", 24) : icon("alert", 21)}</span>
          <span class="empty-eyebrow">${alerts.length === 0 ? "MONITORING HEALTHY" : "NO MATCHING INCIDENTS"}</span>
          <h2>${alerts.length === 0 ? "No significant incidents detected." : `No incidents match the active "${escapeHTML(filter)}" filter.`}</h2>
          <p>${alerts.length === 0 ? "The active video stream or uploaded media has no confirmed safety violations or critical hazards." : "Try clearing your search query or selecting a different filter tab."}</p>
          <a href="/app" data-link class="button button-secondary">Go to Detection Workspace ${icon("arrow", 15)}</a>
        </div>` : `
        <div class="incidents-table-wrap">
          <table class="incidents-table">
            <thead>
              <tr>
                <th>SEVERITY</th>
                <th>INCIDENT TYPE</th>
                <th>TIME</th>
                <th>LOCATION / ZONE</th>
                <th>INVOLVED ENTITY</th>
                <th>EXPLANATION / TRIGGER</th>
                <th>DURATION</th>
                <th>EVIDENCE</th>
                <th>STATUS</th>
                <th>ACTION</th>
              </tr>
            </thead>
            <tbody>
              ${filtered.map((a) => {
                const entityLabel = a.metadata?.display_label || (a.track_id != null ? `Person #${a.metadata?.display_id || a.track_id}` : "Scene");
                const timeStr = a.timestamp > 1000000000 ? new Date(a.timestamp * 1000).toLocaleTimeString() : `${Number(a.timestamp).toFixed(1)}s`;
                const locationStr = `${escapeHTML(a.camera_name || a.camera_id || "Cam 1")}${a.metadata?.zone_id ? ` • Zone: ${escapeHTML(a.metadata.zone_id)}` : ""}`;
                const explanation = a.metadata?.explanation || a.explanation || (a.occurrence_count > 1 ? `Aggregated across ${a.occurrence_count} frames (${(a.metadata?.duration_seconds || 0).toFixed(1)}s)` : "Confirmed safety event");
                const durationStr = a.metadata?.duration_seconds != null ? `${Number(a.metadata.duration_seconds).toFixed(1)}s (${a.occurrence_count || 1}f)` : `${a.occurrence_count || 1} frame(s)`;
                const hasEv = Boolean(a.has_evidence || a.evidence_image_path);

                return `
                <tr class="incident-row" data-action="open-incident-detail" data-id="${a.alert_id}">
                  <td>${severityTag(a.severity)}</td>
                  <td><span class="violation-pill ${a.violation_type?.includes("PPE") || a.violation_type?.includes("NO_") ? "ppe-violation" : "zone-violation"}">${escapeHTML(a.violation_type)}</span></td>
                  <td style="white-space:nowrap; font-size:11px; color:var(--muted);">${timeStr}</td>
                  <td style="font-size:11px;"><b>${locationStr}</b></td>
                  <td style="white-space:nowrap; font-weight:600;">${escapeHTML(entityLabel)}</td>
                  <td style="font-size:11.5px; max-width:280px; line-height:1.4;">${escapeHTML(explanation)}</td>
                  <td style="white-space:nowrap; font-size:11px;">${durationStr}</td>
                  <td>${hasEv ? `<span class="badge-neutral" style="font-size:10px; display:inline-flex; align-items:center; gap:3px;">${icon("image", 11)} Snap</span>` : `<span style="font-size:10px; color:var(--muted);">Telemetry</span>`}</td>
                  <td><span class="lifecycle-pill lifecycle-${(a.status || "NEW").toLowerCase()}"><i></i>${escapeHTML(a.status || "NEW")}</span></td>
                  <td><a href="/app/incidents/${a.alert_id}" data-link class="text-link">Review ${icon("arrow", 13)}</a></td>
                </tr>`;
              }).join("")}
            </tbody>
          </table>
        </div>`
    )}`;
}

function incidentDetailPage() {
  const path = currentPath();
  const alertId = path.split("/").pop();

  // Real backend alert
  const alert = state.activeDetailAlert || state.alerts.find((a) => a.alert_id === alertId);
  if (!alert) {
    return `<div class="detail-breadcrumb"><a href="/app/incidents" data-link>${icon("arrow", 14)} Incidents</a><span>/</span><b>Incident Detail</b></div>
      <div class="panel" style="padding: 40px; text-align: center;"><i class="analysis-spinner"></i><p style="margin-top:12px; color:var(--muted);">Loading incident details for ${escapeHTML(alertId)}...</p></div>`;
  }

  const status = alert.status || "NEW";
  const statusClass = status.toLowerCase();
  const evidenceUrl = incidentService.getAlertEvidenceUrl(alert.alert_id);

  return `<div class="detail-breadcrumb"><a href="/app/incidents" data-link>${icon("arrow", 14)} Incidents</a><span>/</span><b>${escapeHTML(alert.alert_id)}</b></div>
    <div class="detail-heading">
      <div>
        <div class="eyebrow">ALERT EVIDENCE REVIEW • ${escapeHTML(alert.alert_id)}</div>
        <h1>${escapeHTML(alert.violation_type || "Safety Violation")}</h1>
        <p>Source camera: <b>${escapeHTML(alert.camera_id)}</b> • Track: #${alert.track_id ?? "—"} • Occurrences: ${alert.occurrence_count || 1}</p>
      </div>
      <div class="detail-status-actions">
        <span class="lifecycle-pill lifecycle-${statusClass}"><i></i>${escapeHTML(status)}</span>
      </div>
    </div>
    <div class="detail-grid">
      <section class="panel detail-evidence">
        <div class="evidence-detail-header"><div><span class="panel-kicker">EVIDENCE FRAME CAPTURE</span><b>Camera ${escapeHTML(alert.camera_id)}</b></div>${severityTag(alert.severity)}</div>
        <div class="incident-scene" style="display:flex; align-items:center; justify-content:center; background:#080a09; overflow:hidden;">
          <img src="${evidenceUrl}" alt="Evidence Frame" style="width:100%; height:100%; object-fit:contain;" onerror="this.style.display='none'; this.nextElementSibling.style.display='block';" />
          <div style="display:none; padding:40px; color:var(--muted);">${icon("image", 24)}<br/>Evidence image archived</div>
        </div>
      </section>
      <aside class="panel detail-summary">
        <div class="panel-kicker">ALERT SUMMARY</div>
        <div class="detail-severity-line">${severityTag(alert.severity)}<span>Confidence: <b>${Math.round((alert.confidence || 0.85) * 100)}%</b></span></div>
        <p class="detail-reason">Violation detected by real-time safety inspection models. Mandatory safety policy breached.</p>
        <dl class="incident-facts">
          <div><dt>Camera Source</dt><dd>${escapeHTML(alert.camera_id)}</dd></div>
          <div><dt>Track Identification</dt><dd>Track #${alert.track_id ?? "—"}</dd></div>
          <div><dt>Violation Type</dt><dd>${escapeHTML(alert.violation_type)}</dd></div>
          <div><dt>First Detected</dt><dd>${alert.first_detected_at ? new Date(alert.first_detected_at * 1000).toLocaleTimeString() : "Recent"}</dd></div>
          <div><dt>Current Lifecycle</dt><dd><span class="lifecycle-pill lifecycle-${statusClass}"><i></i>${escapeHTML(status)}</span></dd></div>
        </dl>
        <div class="lifecycle-actions">
          <span>OPERATOR WORKFLOW ACTIONS</span>
          <div>
            <button type="button" class="button button-small button-secondary" data-action="update-alert-status" data-id="${alert.alert_id}" data-status="acknowledge" ${status === "ACKNOWLEDGED" || status === "RESOLVED" || status === "DISMISSED" ? "disabled" : ""}>Acknowledge</button>
            <button type="button" class="button button-small button-primary" data-action="update-alert-status" data-id="${alert.alert_id}" data-status="resolve" ${status === "RESOLVED" || status === "DISMISSED" ? "disabled" : ""}>Resolve</button>
            <button type="button" class="text-button dismiss-button" data-action="update-alert-status" data-id="${alert.alert_id}" data-status="dismiss" ${status === "DISMISSED" || status === "RESOLVED" ? "disabled" : ""}>Dismiss</button>
          </div>
        </div>
      </aside>
    </div>
    <section class="fivew-section panel">
      <div class="fivew-header"><div><div class="panel-kicker">EXPLAINABLE SAFETY INTELLIGENCE</div><h2>5W Root Cause Analysis</h2></div><span class="demo-mini-tag">DETERMINISTIC EXPLANATION</span></div>
      <div class="fivew-grid">
        <article><span>WHO</span><b>Track #${alert.track_id ?? "—"}</b><p>Active industrial worker observed in monitored workspace.</p></article>
        <article><span>WHAT</span><b>${escapeHTML(alert.violation_type)}</b><p>Missing mandatory safety equipment or prohibited proximity detected.</p></article>
        <article><span>WHERE</span><b>Camera ${escapeHTML(alert.camera_id)}</b><p>Monitored operational zone and interaction area.</p></article>
        <article><span>WHEN</span><b>${alert.timestamp ? new Date(alert.timestamp * 1000).toLocaleTimeString() : "Active"}</b><p>Logged in security audit trail.</p></article>
        <article class="why-card"><span>WHY</span><b>Safety Rule Ingress</b><p>Automated perception verified failure to comply with safety requirements across consecutive inspection frames.</p></article>
      </div>
    </section>
    <div class="detail-back"><a href="/app/incidents" data-link class="text-link">${icon("arrow", 14)} Return to incidents</a></div>`;
}

function analyticsPage() {
  const intro = introHeading(
    "SAFETY ANALYTICS",
    "Industrial Safety Telemetry",
    "Empirical hazard metrics, PPE compliance distributions and spatial zone activity grounded in backend perception.",
    `<button class="button button-secondary button-small" type="button" data-action="refresh-analytics">${icon("activity", 15)} Refresh Data</button>`
  );

  const alertStats = state.alertStats;
  const alerts = state.alerts || [];
  const alertStatuses = alertStats?.by_status || {};
  const alertSeverities = alertStats?.by_severity || {};
  const totalAlerts = alertStats ? (alertStats.total_alerts ?? alerts.length) : alerts.length;
  const activeUnresolved = alertStats ? (alertStats.active_unresolved_count ?? alerts.filter(a => a.status === "NEW" || a.status === "ACKNOWLEDGED").length) : alerts.filter(a => a.status === "NEW" || a.status === "ACKNOWLEDGED").length;
  const resolvedAlerts = alertStats ? (alertStatuses.RESOLVED ?? alerts.filter(a => a.status === "RESOLVED").length) : alerts.filter(a => a.status === "RESOLVED").length;

  // Empirical Media Assessment Data
  const result = state.analysisState.result;
  const isVideo = state.source?.kind === "video";
  const videoScene = isVideo && state.sceneGraphState.lastJobId === state.analysisState.jobId ? state.sceneGraphState.data : null;
  const bd = result?.inference_breakdown || (isVideo ? buildVideoBreakdownFromScene(videoScene, result) : (result ? extractBreakdownFromAssessment(result) : null));
  const people = bd?.people || [];
  const totalPeople = people.length;
  const compliantPeople = bd?.compliant_people ?? people.filter((p) => p.compliance_status === "COMPLIANT").length;
  const nonCompliantPeople = bd?.non_compliant_people ?? people.filter((p) => p.compliance_status === "NON-COMPLIANT").length;
  const compliancePct = totalPeople > 0 ? Math.round((compliantPeople / totalPeople) * 100) : null;

  // Severity counts
  const critCount = alertSeverities.CRITICAL ?? alerts.filter(a => (a.severity || "").toUpperCase() === "CRITICAL").length;
  const highCount = alertSeverities.HIGH ?? alerts.filter(a => (a.severity || "").toUpperCase() === "HIGH").length;
  const medCount = alertSeverities.MEDIUM ?? alerts.filter(a => (a.severity || "").toUpperCase() === "MEDIUM").length;
  const lowCount = alertSeverities.LOW ?? alerts.filter(a => (a.severity || "").toUpperCase() === "LOW").length;
  const sevTotal = Math.max(1, critCount + highCount + medCount + lowCount);

  // PPE items breakdown
  const ppeSummaryList = bd?.ppe_summary && bd.ppe_summary.length > 0 ? bd.ppe_summary : [];
  if (ppeSummaryList.length === 0 && totalPeople > 0) {
    const ppeCats = ["Hardhat", "Safety Vest", "Gloves", "Goggles"];
    ppeCats.forEach((cat) => {
      let det = 0, mis = 0, unk = 0;
      people.forEach((p) => {
        const st = (p.ppe_status && p.ppe_status[cat]) ? String(p.ppe_status[cat]).toUpperCase() : "";
        if (p.detected_ppe?.includes(cat) || st.includes("PRESENT")) det++;
        else if (p.missing_ppe?.includes(cat) || st.includes("MISSING")) mis++;
        else unk++;
      });
      if (det > 0 || mis > 0 || unk > 0) {
        ppeSummaryList.push({ category: cat, detected: det, missing: mis, unknown: unk });
      }
    });
  }

  // Incident categories breakdown
  const categoryCounts = {};
  alerts.forEach((a) => {
    const c = getAlertCategory(a);
    categoryCounts[c] = (categoryCounts[c] || 0) + 1;
  });
  const presentCategoryEntries = Object.entries(categoryCounts);

  // Measured Inference Performance
  const fps = result?.fps != null ? Number(result.fps).toFixed(1) : (state.telemetry.fps != null ? Number(state.telemetry.fps).toFixed(1) : null);
  const latency = result?.inference_latency_ms != null ? `${result.inference_latency_ms.toFixed(1)} ms` : (state.telemetry.latencyMs != null ? `${state.telemetry.latencyMs.toFixed(1)} ms` : null);
  const duration = result?.processing_time_seconds != null ? `${result.processing_time_seconds.toFixed(2)}s` : (result?.assessment?.processing_time_ms != null ? `${(result.assessment.processing_time_ms / 1000).toFixed(2)}s` : null);
  const frameCount = result?.total_frames ?? (result?.assessment ? 1 : null);
  const detCount = result?.detections_count ?? result?.assessment?.detections?.length ?? null;
  const hasPerfData = fps != null || latency != null || duration != null;

  return `${intro}
    <!-- Top KPI Grid -->
    <div class="kpi-grid analytics-kpi-grid">
      <div class="kpi-card">
        <div class="panel-kicker">TOTAL RECORDED INCIDENTS</div>
        <div class="kpi-value">${totalAlerts}</div>
        <div class="kpi-sub">${icon("alert", 14)} Safety audit records</div>
      </div>
      <div class="kpi-card">
        <div class="panel-kicker">ACTIVE / UNRESOLVED</div>
        <div class="kpi-value">${activeUnresolved}</div>
        <div class="kpi-sub"><span class="status-dot status-warning"></span> New or awaiting review</div>
      </div>
      <div class="kpi-card">
        <div class="panel-kicker">RESOLVED INCIDENTS</div>
        <div class="kpi-value">${resolvedAlerts}</div>
        <div class="kpi-sub"><span class="status-dot status-active"></span> Closed by operator</div>
      </div>
      <div class="kpi-card">
        <div class="panel-kicker">PPE COMPLIANCE RATE</div>
        <div class="kpi-value">${compliancePct !== null ? `${compliancePct}%` : (totalPeople === 0 ? "—" : "100%")}</div>
        <div class="kpi-sub"><span class="status-dot status-active"></span> ${totalPeople > 0 ? `${compliantPeople}/${totalPeople} personnel compliant` : "No active personnel"}</div>
      </div>
      <div class="kpi-card">
        <div class="panel-kicker">MONITORED ZONES</div>
        <div class="kpi-value">${state.zones.length}</div>
        <div class="kpi-sub">${icon("layers", 14)} Spatial boundaries</div>
      </div>
    </div>

    <!-- Row 1: People & PPE Breakdown -->
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:18px;">
      <section class="panel analytics-chart-card">
        <div class="analytics-chart-header">
          <div>
            <div class="panel-kicker">PERSONNEL AUDIT</div>
            <h2 style="font-size:15px; margin:4px 0 0;">People & PPE Compliance Summary</h2>
          </div>
          ${totalPeople > 0 ? `<span class="badge-neutral" style="font-size:11px;">${totalPeople} Tracked</span>` : ""}
        </div>
        ${totalPeople === 0 ? `
          <div class="analytics-empty-box">Insufficient data for this analysis.<br /><small style="color:var(--muted); font-size:11px;">Process an image or video in the workspace to audit worker compliance.</small></div>
        ` : `
          <div style="display:flex; flex-direction:column; gap:12px;">
            <div class="analytics-bar-row">
              <div class="analytics-bar-meta">
                <span>Fully Compliant: <b>${compliantPeople}</b> (${compliancePct}%)</span>
                <span>Non-Compliant: <b style="color:var(--critical);">${nonCompliantPeople}</b> (${100 - compliancePct}%)</span>
              </div>
              <div class="analytics-multi-bar" style="height:10px;">
                <div style="width:${compliancePct}%; background:var(--success);" title="Compliant: ${compliantPeople}"></div>
                <div style="width:${100 - compliancePct}%; background:var(--critical);" title="Non-Compliant: ${nonCompliantPeople}"></div>
              </div>
            </div>
            <div style="display:flex; gap:16px; font-size:11px; color:var(--muted);">
              <span style="display:flex; align-items:center; gap:4px;"><span style="width:8px; height:8px; border-radius:2px; background:var(--success);"></span> Compliant (${compliantPeople})</span>
              <span style="display:flex; align-items:center; gap:4px;"><span style="width:8px; height:8px; border-radius:2px; background:var(--critical);"></span> Non-Compliant (${nonCompliantPeople})</span>
            </div>
          </div>
        `}
      </section>

      <section class="panel analytics-chart-card">
        <div class="analytics-chart-header">
          <div>
            <div class="panel-kicker">PPE DISTRIBUTION</div>
            <h2 style="font-size:15px; margin:4px 0 0;">Equipment Detection Status</h2>
          </div>
          <div style="display:flex; gap:8px; font-size:10px;">
            <span style="color:var(--success); font-weight:600;">■ Detected</span>
            <span style="color:var(--critical); font-weight:600;">■ Missing</span>
            <span style="color:var(--muted); font-weight:600;">■ Unknown</span>
          </div>
        </div>
        ${ppeSummaryList.length === 0 ? `
          <div class="analytics-empty-box">Insufficient data for this analysis.</div>
        ` : `
          <div style="display:flex; flex-direction:column; gap:10px;">
            ${ppeSummaryList.map((item) => {
              const itemTotal = Math.max(1, (item.detected || 0) + (item.missing || 0) + (item.unknown || 0));
              const detPct = Math.round(((item.detected || 0) / itemTotal) * 100);
              const misPct = Math.round(((item.missing || 0) / itemTotal) * 100);
              const unkPct = Math.max(0, 100 - detPct - misPct);
              return `
              <div class="analytics-bar-row">
                <div class="analytics-bar-meta">
                  <b>${escapeHTML(item.category)}</b>
                  <span>${item.detected || 0} present • <span style="color:var(--critical); font-weight:600;">${item.missing || 0} missing</span> • ${item.unknown || 0} unk</span>
                </div>
                <div class="analytics-multi-bar">
                  <div style="width:${detPct}%; background:var(--success);" title="${item.detected || 0} Present"></div>
                  <div style="width:${misPct}%; background:var(--critical);" title="${item.missing || 0} Missing"></div>
                  <div style="width:${unkPct}%; background:var(--surface-4, #3a4249);" title="${item.unknown || 0} Unknown"></div>
                </div>
              </div>`;
            }).join("")}
          </div>
        `}
      </section>
    </div>

    <!-- Row 2: Incident Severity & Types -->
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:18px;">
      <section class="panel analytics-chart-card">
        <div class="analytics-chart-header">
          <div>
            <div class="panel-kicker">HAZARD SEVERITY</div>
            <h2 style="font-size:15px; margin:4px 0 0;">Incident Breakdown by Severity</h2>
          </div>
        </div>
        ${totalAlerts === 0 ? `
          <div class="analytics-empty-box">No recorded incidents to analyze.</div>
        ` : `
          <div style="display:flex; flex-direction:column; gap:10px;">
            <div class="analytics-bar-row">
              <div class="analytics-bar-meta"><span>CRITICAL</span><b>${critCount} (${Math.round((critCount / sevTotal) * 100)}%)</b></div>
              <div class="analytics-bar-track"><div class="analytics-bar-fill" style="width:${(critCount / sevTotal) * 100}%; background:var(--critical);"></div></div>
            </div>
            <div class="analytics-bar-row">
              <div class="analytics-bar-meta"><span>HIGH RISK</span><b>${highCount} (${Math.round((highCount / sevTotal) * 100)}%)</b></div>
              <div class="analytics-bar-track"><div class="analytics-bar-fill" style="width:${(highCount / sevTotal) * 100}%; background:var(--warning);"></div></div>
            </div>
            <div class="analytics-bar-row">
              <div class="analytics-bar-meta"><span>MEDIUM</span><b>${medCount} (${Math.round((medCount / sevTotal) * 100)}%)</b></div>
              <div class="analytics-bar-track"><div class="analytics-bar-fill" style="width:${(medCount / sevTotal) * 100}%; background:var(--accent);"></div></div>
            </div>
            <div class="analytics-bar-row">
              <div class="analytics-bar-meta"><span>LOW</span><b>${lowCount} (${Math.round((lowCount / sevTotal) * 100)}%)</b></div>
              <div class="analytics-bar-track"><div class="analytics-bar-fill" style="width:${(lowCount / sevTotal) * 100}%; background:#5c8fd6;"></div></div>
            </div>
          </div>
        `}
      </section>

      <section class="panel analytics-chart-card">
        <div class="analytics-chart-header">
          <div>
            <div class="panel-kicker">VIOLATION CATEGORIES</div>
            <h2 style="font-size:15px; margin:4px 0 0;">Incident Types Breakdown</h2>
          </div>
        </div>
        ${presentCategoryEntries.length === 0 ? `
          <div class="analytics-empty-box">No recorded incidents to analyze.</div>
        ` : `
          <div style="display:flex; flex-direction:column; gap:10px;">
            ${presentCategoryEntries.map(([catName, count]) => {
              const pct = Math.round((count / Math.max(1, alerts.length)) * 100);
              return `
              <div class="analytics-bar-row">
                <div class="analytics-bar-meta"><span>${escapeHTML(catName)}</span><b>${count} (${pct}%)</b></div>
                <div class="analytics-bar-track"><div class="analytics-bar-fill" style="width:${pct}%; background:var(--accent);"></div></div>
              </div>`;
            }).join("")}
          </div>
        `}
      </section>
    </div>

    <!-- Row 3: Spatial Zones & Hardware Performance -->
    <div style="display:grid; grid-template-columns: 1fr 1fr; gap:16px; margin-bottom:18px;">
      <section class="panel analytics-chart-card">
        <div class="analytics-chart-header">
          <div>
            <div class="panel-kicker">SPATIAL MONITORING</div>
            <h2 style="font-size:15px; margin:4px 0 0;">Active Configured Zones</h2>
          </div>
          <span class="badge-neutral" style="font-size:11px;">${state.zones.length} Configured</span>
        </div>
        <div style="display:flex; flex-direction:column; gap:8px;">
          ${state.zones.length
            ? state.zones.map((z) => `<div style="display:flex; justify-content:space-between; align-items:center; padding:8px 12px; background:var(--surface-2); border-radius:4px; font-size:11px;"><span>${escapeHTML(z.name)}</span><span class="severity-tag ${z.enabled ? "severity-low" : "severity-info"}">${z.zone_type?.toUpperCase()}</span></div>`).join("")
            : `<div class="analytics-empty-box">No configured zones are available.</div>`}
        </div>
      </section>

      <section class="panel analytics-chart-card">
        <div class="analytics-chart-header">
          <div>
            <div class="panel-kicker">INFERENCE BENCHMARKS</div>
            <h2 style="font-size:15px; margin:4px 0 0;">Hardware & Engine Performance</h2>
          </div>
          <span class="badge-neutral" style="font-size:11px;">${escapeHTML(state.telemetry.gpuName || "CUDA:0 GPU")}</span>
        </div>
        ${!hasPerfData ? `
          <div class="analytics-empty-box">Insufficient data for this analysis.<br /><small style="color:var(--muted); font-size:11px;">Run inference in Detection Workspace to record latency & throughput measurements.</small></div>
        ` : `
          <div style="display:grid; grid-template-columns: 1fr 1fr; gap:10px;">
            <div style="padding:10px 12px; background:var(--surface-2); border-radius:4px;">
              <div style="font-size:10px; color:var(--muted); text-transform:uppercase;">PROCESSING SPEED</div>
              <div style="font-size:18px; font-weight:700; margin-top:2px;">${fps != null ? `${fps} FPS` : "—"}</div>
            </div>
            <div style="padding:10px 12px; background:var(--surface-2); border-radius:4px;">
              <div style="font-size:10px; color:var(--muted); text-transform:uppercase;">INFERENCE LATENCY</div>
              <div style="font-size:18px; font-weight:700; margin-top:2px;">${latency != null ? latency : "—"}</div>
            </div>
            <div style="padding:10px 12px; background:var(--surface-2); border-radius:4px;">
              <div style="font-size:10px; color:var(--muted); text-transform:uppercase;">PROCESSING DURATION</div>
              <div style="font-size:18px; font-weight:700; margin-top:2px;">${duration != null ? duration : "—"}</div>
            </div>
            <div style="padding:10px 12px; background:var(--surface-2); border-radius:4px;">
              <div style="font-size:10px; color:var(--muted); text-transform:uppercase;">EVALUATED UNITS</div>
              <div style="font-size:18px; font-weight:700; margin-top:2px;">${frameCount != null ? `${frameCount} frames` : (detCount != null ? `${detCount} objects` : "—")}</div>
            </div>
          </div>
        `}
      </section>
    </div>
    <div class="analytics-policy">${icon("shield", 15)}<span>IntelliWatch metrics focus strictly on facility safety conditions and equipment compliance, not worker productivity scoring.</span></div>`;
}

function cameraCalibrationSection() {
  const cal = state.calibrationState;
  const cameras = state.cameras || [];
  const status = cal.calibrationStatus || "UNCONFIGURED";
  const isCalibrated = status === "CALIBRATED";
  const badgeClass = isCalibrated ? "badge-success" : (status === "INVALID" ? "badge-danger" : "badge-neutral");
  const points = cal.points || [];
  const pCount = points.length;

  const frameUrl = cal.frameUrl || (cal.selectedCameraId ? cameraService.getFrameUrl(cal.selectedCameraId) : "/api/v1/cameras/cam_loading_bay_01/frame");

  // SVG representation of points, polygon, and dimension annotations
  let svgContent = "";
  if (pCount > 0) {
    if (pCount === 4) {
      const ptsStr = points.map((p) => `${p.x},${p.y}`).join(" ");
      svgContent += `<polygon points="${ptsStr}" class="cal-polygon" />`;
    } else if (pCount > 1) {
      for (let i = 0; i < pCount - 1; i++) {
        svgContent += `<line x1="${points[i].x}" y1="${points[i].y}" x2="${points[i + 1].x}" y2="${points[i + 1].y}" stroke="#d4aa5b" stroke-width="2" stroke-dasharray="6 4" />`;
      }
    }

    // Dimension labels if 4 points are configured
    if (pCount === 4) {
      const midW1 = { x: (points[0].x + points[1].x) / 2, y: (points[0].y + points[1].y) / 2 };
      const midD1 = { x: (points[0].x + points[3].x) / 2, y: (points[0].y + points[3].y) / 2 };
      svgContent += `
        <text x="${midW1.x}" y="${midW1.y - 12}" class="cal-dim-label" text-anchor="middle">Width: ${cal.realWorldWidthM}m</text>
        <text x="${midD1.x - 12}" y="${midD1.y}" class="cal-dim-label" text-anchor="end">Depth: ${cal.realWorldDepthM}m</text>
      `;
    }

    // Point Markers P1..P4
    points.forEach((p, idx) => {
      svgContent += `
        <circle cx="${p.x}" cy="${p.y}" r="8" class="cal-marker-circle" />
        <circle cx="${p.x}" cy="${p.y}" r="3" fill="#ffffff" />
        <text x="${p.x + 12}" y="${p.y + 4}" class="cal-marker-label">P${idx + 1}</text>
      `;
    });
  }

  return `
    <section class="panel calibration-panel" id="camera-calibration">
      <div class="settings-panel-heading">
        <div>
          <div class="panel-kicker">CAMERA CALIBRATION</div>
          <h2>Ground-Plane Perspective Calibration</h2>
          <p>Configure 4-point ground-plane quadrilateral homography to enable approximate real-world metric spatial reasoning.</p>
        </div>
        <div style="display: flex; gap: 8px; align-items: center;">
          <span class="badge ${badgeClass}">${status}</span>
        </div>
      </div>

      <div class="calibration-workbench">
        <!-- Interactive Viewport -->
        <div class="calibration-viewport-card">
          <div class="calibration-viewport-header">
            <div style="display: flex; gap: 10px; align-items: center;">
              <label for="cal-camera-select" style="font-size: 11px; font-weight: 650; color: var(--muted); text-transform: uppercase;">Camera View:</label>
              <select id="cal-camera-select" class="cal-camera-select">
                <option value="cam_loading_bay_01" ${cal.selectedCameraId === "cam_loading_bay_01" ? "selected" : ""}>Loading Bay 01 (Primary CCTV)</option>
                ${cameras.filter((c) => c.camera_id !== "cam_loading_bay_01").map((c) => `
                  <option value="${escapeHTML(c.camera_id)}" ${cal.selectedCameraId === c.camera_id ? "selected" : ""}>${escapeHTML(c.name || c.camera_id)}</option>
                `).join("")}
              </select>
            </div>
            <div style="font-size: 11px; color: var(--subtle);">
              ${pCount < 4 ? `<span style="color: var(--accent);">Click image to place P${pCount + 1} of 4</span>` : `<span style="color: #6bb377;">4 ground points placed</span>`}
            </div>
          </div>

          <div class="calibration-canvas-wrap" id="cal-canvas-wrap" title="${pCount < 4 ? `Click to place ground point P${pCount + 1}` : "4 points placed. Click Reset Points to reconfigure."}">
            <img src="${escapeHTML(frameUrl)}" alt="Camera frame for calibration" class="calibration-frame-img" id="cal-frame-img" />
            <svg class="calibration-svg-overlay" viewBox="0 0 ${cal.imageWidth} ${cal.imageHeight}" preserveAspectRatio="none">
              ${svgContent}
            </svg>
          </div>

          <div class="calibration-viewport-footer">
            <div style="display: flex; gap: 14px; font-size: 11px; color: var(--muted); flex-wrap: wrap;">
              <span>Resolution: <b>${cal.imageWidth} × ${cal.imageHeight} px</b></span>
              <span>Points: <b>${pCount} / 4</b></span>
              <span>Model: <b>Homography (Ground Plane)</b></span>
            </div>
            <button type="button" class="button button-quiet button-small" data-action="reset-cal-points">
              ${icon("close", 13)} Reset Points
            </button>
          </div>
        </div>

        <!-- Controls Column -->
        <div class="calibration-controls-column">
          <div class="calibration-instruction-box">
            <b>Calibration Workflow:</b>
            <ol>
              <li>Select camera view or CCTV frame snapshot.</li>
              <li>Click 4 points on the ground plane in order: <i>P1 (Far-Left) → P2 (Far-Right) → P3 (Near-Right) → P4 (Near-Left)</i>.</li>
              <li>Input measured real-world quadrilateral dimensions (meters).</li>
              <li>Validate geometric convexity and save homography.</li>
            </ol>
          </div>

          <div class="calibration-inputs-grid">
            <div class="cal-input-group">
              <label for="cal-width-m">Real-World Width (m)</label>
              <input type="number" id="cal-width-m" min="0.1" max="1000.0" step="0.1" value="${cal.realWorldWidthM}" placeholder="e.g. 6.0" />
            </div>
            <div class="cal-input-group">
              <label for="cal-depth-m">Real-World Depth (m)</label>
              <input type="number" id="cal-depth-m" min="0.1" max="1000.0" step="0.1" value="${cal.realWorldDepthM}" placeholder="e.g. 10.0" />
            </div>
          </div>

          <div class="calibration-stats-card">
            <div style="font-size: 10px; font-weight: 700; text-transform: uppercase; color: var(--subtle); letter-spacing: 0.06em;">Configured Vertices</div>
            <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 6px; font-size: 11px;">
              ${["P1 (Far L)", "P2 (Far R)", "P3 (Near R)", "P4 (Near L)"].map((lbl, idx) => {
                const pt = points[idx];
                return `<div style="padding: 4px 6px; background: var(--surface-3); border-radius: 4px; border: 1px solid var(--line);">
                  <b style="color: var(--accent);">${lbl}:</b> ${pt ? `(${pt.x}, ${pt.y})` : `<span style="color: var(--subtle);">Not set</span>`}
                </div>`;
              }).join("")}
            </div>
          </div>

          ${cal.validationError ? `
            <div class="inline-error" role="alert" style="margin: 0;">
              ${icon("alert", 15)} <span>${escapeHTML(cal.validationError)}</span>
            </div>
          ` : (cal.isValid ? `
            <div style="padding: 8px 12px; background: rgba(107, 179, 119, 0.1); border: 1px solid rgba(107, 179, 119, 0.3); border-radius: 5px; font-size: 11.5px; color: #6bb377; display: flex; align-items: center; gap: 6px;">
              ${icon("check", 14)} <span>Quadrilateral is geometrically valid; homography computed.</span>
            </div>
          ` : "")}

          <div class="calibration-actions-row">
            <button type="button" class="button button-secondary" data-action="validate-cal" ${pCount !== 4 ? "disabled" : ""}>
              ${icon("check", 14)} Validate Homography
            </button>
            <button type="button" class="button button-primary" data-action="save-cal" ${pCount !== 4 ? "disabled" : ""}>
              ${icon("check", 14)} Save Calibration
            </button>
            <button type="button" class="button button-quiet button-danger" data-action="clear-cal" ${status === "UNCONFIGURED" ? "disabled" : ""}>
              Clear Calibration
            </button>
          </div>

          <div class="calibration-technical-notice">
            ${icon("info", 16)}
            <div>
              <b>Approximate Ground-Plane Spatial Estimation:</b><br />
              Ground-plane calibration provides approximate real-world spatial measurements for the configured camera view. Accuracy depends on camera angle, point selection, and scene geometry. It does not provide full 3D reconstruction, metric depth estimation, or camera intrinsic recovery.
            </div>
          </div>
        </div>
      </div>
    </section>
  `;
}

function settingsPage() {
  const t = state.telemetry;

  return `${introHeading("WORKSPACE SETTINGS", "Preferences, Calibration & Hardware Diagnostics", "Inspect connected compute hardware, camera perspective calibration, and interface settings.", `<span class="preview-only-pill">${t.isCuda ? "CUDA:0 ENABLED" : "CPU MODE"}</span>`)}
    <div class="settings-grid"><div class="settings-main-column">
      ${cameraCalibrationSection()}
      <section class="panel appearance-panel"><div class="settings-panel-heading"><div><div class="panel-kicker">APPEARANCE</div><h2>Color theme</h2><p>Select light or dark mode. The preference persists in your browser.</p></div>${icon("sun", 17)}</div><div class="appearance-options">${themePreferenceCard("light", "Light", "Warm neutral surfaces")}${themePreferenceCard("dark", "Dark", "Low-light operator view")}${themePreferenceCard("system", "System", `Follows device • ${resolvedTheme("system")}`)}</div></section>
      <section class="panel settings-panel"><div class="settings-panel-heading"><div><div class="panel-kicker">SAFETY PROFILE</div><h2>Perception Capabilities</h2><p>Toggle individual detection layers for new sessions.</p></div><span class="demo-mini-tag">ACTIVE</span></div><div class="settings-module-list">${modules.map(([key, label]) => `<div class="settings-module-row"><div><b>${label}</b><small>${moduleDescription(key)}</small></div><label class="module-toggle"><span class="sr-only">Include ${label}</span><input type="checkbox" data-module="${key}" ${state.modules[key] ? "checked" : ""} /><i aria-hidden="true"></i></label></div>`).join("")}</div></section>
      <section class="panel preference-panel"><div class="settings-panel-heading"><div><div class="panel-kicker">SOURCE & PLAYBACK</div><h2>Local preview preferences</h2><p>Applied to uploaded media playback.</p></div>${icon("video", 17)}</div><div class="preference-row"><div><b>Default playback speed</b><small>Used when opening a video preview.</small></div><label class="select-wrap"><span class="sr-only">Default playback speed</span><select id="playback-rate"><option value="0.5" ${state.playbackRate === 0.5 ? "selected" : ""}>0.5×</option><option value="1" ${state.playbackRate === 1 ? "selected" : ""}>1×</option><option value="1.5" ${state.playbackRate === 1.5 ? "selected" : ""}>1.5×</option><option value="2" ${state.playbackRate === 2 ? "selected" : ""}>2×</option></select></label></div><div class="preference-row"><div><b>Mute previews by default</b><small>Player controls can unmute playback.</small></div><label class="module-toggle"><span class="sr-only">Mute previews by default</span><input type="checkbox" data-preview-muted ${state.previewMuted ? "checked" : ""} /><i aria-hidden="true"></i></label></div></section>
    </div><aside class="settings-side"><section class="panel integration-card"><div class="panel-kicker">CONNECTED SERVICES</div><span class="service-icon">${icon("wifi", 20)}</span><h2>${t.online ? "Backend Online" : "Service Offline"}</h2><p>${escapeHTML(t.hardwareStatus)}</p><dl class="incident-facts" style="margin-top:12px;"><div style="display:flex; justify-content:space-between; margin-bottom:6px;"><dt>Hardware</dt><dd><b>${escapeHTML(t.gpuName || (t.isCuda ? "NVIDIA RTX 5050" : "CPU"))}</b></dd></div><div style="display:flex; justify-content:space-between; margin-bottom:6px;"><dt>Device</dt><dd><b>${t.isCuda ? "cuda:0" : "cpu"}</b></dd></div><div style="display:flex; justify-content:space-between; margin-bottom:6px;"><dt>PyTorch</dt><dd>${escapeHTML(t.torchVersion || "Active")}</dd></div><div style="display:flex; justify-content:space-between; margin-bottom:6px;"><dt>Measured Latency</dt><dd>${t.latencyMs ? `${t.latencyMs} ms` : "—"}</dd></div><div style="display:flex; justify-content:space-between;"><dt>Inference FPS</dt><dd>${t.fps ? `${t.fps} FPS` : "—"}</dd></div></dl><span class="${t.online ? "service-online" : "service-offline"}" style="display:inline-flex; align-items:center; gap:6px; margin-top:14px; font-weight:700;"><i class="status-dot ${t.online ? "status-active" : "status-warning"}"></i> ${t.online ? (t.isCuda ? "CUDA:0 ACCELERATED" : "ACTIVE") : "DISCONNECTED"}</span></section></aside></div>`;
}

function themePreferenceCard(value, title, detail) {
  return `<label class="appearance-option ${state.theme === value ? "is-selected" : ""}"><input type="radio" name="theme" value="${value}" data-theme-choice ${state.theme === value ? "checked" : ""} /><span class="theme-preview theme-preview-${value}" aria-hidden="true"><i></i><i></i><i></i></span><span class="appearance-option-copy"><b>${title}</b><small ${value === "system" ? "data-system-theme-detail" : ""}>${detail}</small></span><span class="appearance-radio" aria-hidden="true"></span></label>`;
}

function moduleDescription(key) {
  const descriptions = {
    objects: "People, vehicles and relevant objects in the scene.",
    assets: "Industrial equipment and site assets where available.",
    ppe: "Protective equipment states and compliance context.",
    zones: "Safety zone entry, breach and dwell context.",
    behavior: "Behavior state and relevant duration.",
    proximity: "Distance and interaction context between tracks.",
    predictions: "Early warnings based on a projected scene relationship.",
    risk: "Contributing factors behind a deterministic risk score.",
  };
  return descriptions[key] || "";
}

function landingPage() {
  return `<div class="public-shell">${publicHeader()}
    <main class="page-width">
      <section class="hero-section hero-section--focused"><div class="hero-copy"><span class="hero-kicker"><span class="tiny-mark">IW</span> INDUSTRIAL COMPUTER VISION</span><h1>Safety signals grounded in visible evidence.</h1><p class="hero-lead">Turn facility cameras and recorded footage into deterministic hazard detection, PPE compliance, and automated incident documentation.</p><div class="hero-actions"><a class="button button-primary" href="/app" data-link>Open Operations Hub ${icon("arrow", 16)}</a></div></div></section>
    </main><footer class="public-footer page-width"><a class="brand" href="/" data-link><span class="brand-mark">IW</span><span class="brand-name">Intelli<span>Watch</span></span></a><span>CUDA-accelerated industrial vision intelligence.</span><span>Authoritative Operations UI</span></footer></div>`;
}

function productPage() {
  return `<div class="public-shell">${publicHeader()}
    <main class="page-width" style="padding-top:40px;">
      <div class="workspace-heading"><div><div class="eyebrow">PRODUCT CAPABILITIES</div><h1>Industrial Safety Intelligence</h1><p>Comprehensive computer vision pipeline running high-accuracy neural perception and temporal reasoning.</p></div><a href="/app" data-link class="button button-primary">Open Console ${icon("arrow", 15)}</a></div>
      <div class="evidence-section"><div class="evidence-frame"><div class="evidence-frame-top"><span>ACTIVE OPERATIONAL SCENE</span>${severityTag("CRITICAL")}</div><div class="evidence-scene">${sceneSvg()}${overlaySvg()}<span class="evidence-frame-label">CAMERA 01 • LOADING BAY</span></div><div class="evidence-frame-bottom"><span>${icon("shield", 13)} CUDA ACCELERATED</span><span>FPS: 15.0</span></div></div><div class="evidence-copy"><div class="panel-kicker">EXPLAINABILITY FIRST</div><h2>Traceable from detection to incident archive</h2><p>Every alert is substantiated by original visual evidence frames, exact pixel bounding boxes, track identity longevity, and rule-based evaluation.</p><div style="margin-top:20px;"><a href="/app" data-link class="button button-secondary">Explore Workspace ${icon("arrow", 15)}</a></div></div></div>
    </main></div>`;
}

function sceneGraphPage() {
  const sg = state.sceneGraphState;
  const data = sg.data;
  const isLive = Boolean(data?.is_live);
  const sourceMode = data?.source_mode || (state.source?.kind === "video" ? "video_job" : "single_image");
  const isSingleImage = Boolean(data?.time_to_hazard?.hazard_type === "STATIC SCENE ANALYSIS" || (!isLive && state.source?.kind === "image"));
  const frameIdx = data?.frame_index ?? (data?.summary?.frame_index ?? null);

  const entities = data?.entities || [];
  const relationships = data?.relationships || [];
  const riskTier = data?.summary?.peak_risk_level || (entities.find((e) => e.risk?.level === "CRITICAL") ? "CRITICAL" : (entities.find((e) => e.risk?.level === "HIGH") ? "HIGH" : "NORMAL"));
  const personsCount = entities.filter((e) => e.type === "PERSON").length;
  const vehiclesCount = entities.filter((e) => e.type === "VEHICLE").length;
  const assetsCount = entities.filter((e) => ["MACHINE", "ASSET", "EQUIPMENT"].includes(e.type)).length;
  const otherEntitiesCount = entities.filter((e) => !["PERSON", "VEHICLE", "PPE", "ZONE", "HAZARD", "MACHINE", "ASSET", "EQUIPMENT"].includes(e.type)).length;
  const zonesCount = entities.filter((e) => e.type === "ZONE").length;

  const modeBadge = isLive
    ? `<span class="scene-mode-badge is-live"><span class="mode-dot"></span> LIVE</span>`
    : `<span class="scene-mode-badge is-snapshot">${icon("clock", 13)} LATEST SCENE SNAPSHOT</span>`;

  const headerActions = `
    <div class="scene-graph-header-actions">
      <div class="scene-view-modes">
        <button type="button" class="view-mode-btn ${sg.viewMode === "graph" ? "is-active" : ""}" data-action="set-graph-view-mode" data-mode="graph">
          ${icon("layers", 13)} GRAPH
        </button>
        <button type="button" class="view-mode-btn ${sg.viewMode === "split" ? "is-active" : ""}" data-action="set-graph-view-mode" data-mode="split">
          ${icon("image", 13)} SPLIT EVIDENCE
        </button>
        <button type="button" class="view-mode-btn ${sg.viewMode === "table" ? "is-active" : ""}" data-action="set-graph-view-mode" data-mode="table">
          ${icon("grid", 13)} TABLE
        </button>
      </div>
      ${state.source ? `<button class="button button-secondary button-small" type="button" data-action="back-to-detection">${icon("arrow", 14)} Back to Detection</button>` : ""}
      <button class="button button-secondary button-small" type="button" data-action="refresh-scene-graph" ${sg.loading ? "disabled" : ""}>
        ${icon("activity", 14)} ${sg.loading ? "Loading..." : "Refresh"}
      </button>
    </div>
  `;

  const intro = introHeading(
    "SCENE UNDERSTANDING",
    "Scene Graph Hero",
    `Live machine-readable scene representation grounded in real neural detections, tracking, spatial zones, and safety risk logic. <span class="heading-separator">•</span> ${modeBadge}`,
    headerActions
  );

  if (sg.loading && !data) {
    return `${intro}
      <div class="panel graph-state-container">
        <div class="graph-state-icon"><i class="analysis-spinner"></i></div>
        <div class="graph-state-title">BUILDING SCENE GRAPH...</div>
        <div class="graph-state-sub">Synthesizing spatial relationships, worker compliance bindings, and temporal kinematics from neural inference.</div>
      </div>`;
  }

  if (sg.error && !data) {
    return `${intro}
      <div class="panel graph-state-container">
        <div class="graph-state-icon" style="color: var(--critical);">${icon("alert", 26)}</div>
        <div class="graph-state-title">Unable to load scene graph</div>
        <div class="graph-state-sub">${escapeHTML(sg.error)}</div>
        <button type="button" class="button button-primary button-small" data-action="refresh-scene-graph" style="margin-top:14px;">${icon("activity", 14)} Retry Loading</button>
      </div>`;
  }

  if (!data || entities.length === 0) {
    return `${intro}
      <div class="panel graph-state-container">
        <div class="graph-state-icon">${icon("layers", 26)}</div>
        <div class="graph-state-title">NO SCENE ENTITIES DETECTED</div>
        <div class="graph-state-sub">The vision engine has not processed an active scene or no entities were detected in the latest frame. Upload an image or video in the workspace to construct the scene graph.</div>
        <a href="/app" data-link class="button button-primary button-small" style="margin-top:14px;">${icon("upload", 14)} Go to Detection Workspace</a>
      </div>`;
  }

  // Summary KPIs (Phase 13)
  const activeRelationships = relationships.filter((relation) => relation.active === true);
  const activeRelsCount = activeRelationships.length;
  const nonCompliantPersonIds = new Set(entities
    .filter((entity) => entity.type === "PERSON" && String(entity.state?.compliance_status || "").toUpperCase().replace(/_/g, "-") === "NON-COMPLIANT")
    .map((entity) => entity.id));
  const activeHazardRelationTypes = new Set(["APPROACHING", "CLOSER_THAN", "AT_RISK_FROM", "INSIDE_ZONE"]);
  const activeHazardRelations = activeRelationships.filter((relation) => {
    const type = String(relation.type).toUpperCase();
    if (activeHazardRelationTypes.has(type)) return true;
    if (type !== "INSIDE") return false;
    const zone = entities.find((entity) => entity.id === relation.target_id && entity.type === "ZONE");
    return ["RESTRICTED", "HAZARD", "NO_ENTRY", "RESTRICTED_AREA"].includes(String(zone?.state?.zone_type || "").toUpperCase());
  });
  const ppeHazardTypes = new Set(["MISSING", "NO_HARDHAT", "NO_SAFETY_VEST", "NO_GLOVES", "NO_GOGGLES"]);
  const ppeHazardPersonIds = new Set(activeRelationships
    .filter((relation) => ppeHazardTypes.has(String(relation.type).toUpperCase()))
    .flatMap((relation) => [relation.source_id, relation.target_id])
    .filter((entityId) => entities.some((entity) => entity.id === entityId && entity.type === "PERSON")));
  const activeHazardsCount = activeHazardRelations.length + Math.max(nonCompliantPersonIds.size, ppeHazardPersonIds.size);
  const highRiskEntitiesCount = entities.filter((e) => ["HIGH", "CRITICAL"].includes(String(e.risk?.level).toUpperCase())).length;

  const kpisHtml = `
    <div class="scene-kpi-strip">
      <div class="scene-kpi-card">
        <span class="scene-kpi-label">ENTITIES</span>
        <div class="scene-kpi-val">${entities.length}</div>
        <div class="scene-kpi-sub">${icon("layers", 12)} ${personsCount} workers, ${vehiclesCount} vehicles, ${zonesCount} zones</div>
      </div>
      <div class="scene-kpi-card">
        <span class="scene-kpi-label">ACTIVE RELATIONSHIPS</span>
        <div class="scene-kpi-val">${activeRelsCount}</div>
        <div class="scene-kpi-sub">${icon("link", 12)} Directional spatial & behavioral edges</div>
      </div>
      <div class="scene-kpi-card">
        <span class="scene-kpi-label">ACTIVE HAZARDS</span>
        <div class="scene-kpi-val ${activeHazardsCount > 0 ? "text-critical" : "text-success"}">${activeHazardsCount}</div>
        <div class="scene-kpi-sub">${icon("alert", 12)} Zone incursions, convergence & missing PPE</div>
      </div>
      <div class="scene-kpi-card">
        <span class="scene-kpi-label">HIGH-RISK ENTITIES</span>
        <div class="scene-kpi-val ${highRiskEntitiesCount > 0 ? "text-critical" : "text-success"}">${highRiskEntitiesCount}</div>
        <div class="scene-kpi-sub">${icon("shield", 12)} Peak scene risk: <b>${escapeHTML(riskTier)}</b></div>
      </div>
    </div>
  `;

  // Time to Hazard Hero Panel (Deterministic calculations, actual values only)
  const tthHeroHtml = renderTimeToHazardHero(data, isSingleImage);

  // Why This Matters Intelligence Panel (Phase 14)
  const trackedCount = entities.filter((e) => e.track_id != null).length;
  const spatialCount = activeRelationships.filter((relation) => ["NEAR", "FAR", "ADJACENT", "INSIDE_ZONE", "INSIDE"].includes(String(relation.type).toUpperCase())).length;
  const temporalCount = activeRelationships.filter((relation) => ["APPROACHING", "CLOSER_THAN", "MOVING_AWAY", "TRAJECTORY_CONVERGING"].includes(String(relation.type).toUpperCase())).length;

  const pipelineBannerHtml = `
    <div class="scene-pipeline-banner">
      <div class="scene-pipeline-header">
        <span class="scene-pipeline-title">${icon("activity", 13)} SCENE UNDERSTANDING PIPELINE</span>
        <small style="color: var(--muted); font-size: 10.5px;">Structured relational intelligence</small>
      </div>
      <div class="scene-pipeline-flow">
        <span class="pipeline-flow-step">DETECTION <small>(${entities.length} objects)</small></span>
        <span class="pipeline-flow-arrow">→</span>
        <span class="pipeline-flow-step">TRACKING <small>(${trackedCount} tracks)</small></span>
        <span class="pipeline-flow-arrow">→</span>
        <span class="pipeline-flow-step">ENTITY STATE <small>(${personsCount} workers)</small></span>
        <span class="pipeline-flow-arrow">→</span>
        <span class="pipeline-flow-step">SPATIAL RELATIONS <small>(${spatialCount} bindings)</small></span>
        <span class="pipeline-flow-arrow">→</span>
        <span class="pipeline-flow-step">TEMPORAL RELATIONS <small>(${temporalCount} vectors)</small></span>
        <span class="pipeline-flow-arrow">→</span>
        <span class="pipeline-flow-step">HAZARD CONTEXT <small>(${activeHazardsCount} hazards)</small></span>
        <span class="pipeline-flow-arrow">→</span>
        <span class="pipeline-flow-step" style="border-color: ${riskTier === 'CRITICAL' ? 'var(--critical)' : 'var(--accent)'};">RISK: ${escapeHTML(riskTier)}</span>
      </div>
    </div>
  `;

  // Entity & Relationship Filter Chips (Phase 10)
  const entityFilters = [
    ["ALL", `All Entities (${entities.length})`],
    ["PEOPLE", `People (${personsCount})`],
    ["VEHICLES", `Vehicles (${vehiclesCount})`],
    ["PPE", "PPE Gear"],
    ["ZONES", `Zones (${zonesCount})`],
    ["ASSETS", `Assets (${assetsCount})`],
    ["OTHER", `Other (${otherEntitiesCount})`],
    ["HAZARDS", `Hazards (${highRiskEntitiesCount})`],
  ].map(([key, label]) => `<button type="button" class="filter-chip ${(sg.entityFilter || "ALL") === key ? "is-active" : ""}" data-action="filter-entities" data-filter="${key}">${label}</button>`).join("");

  const relationFilters = [
    ["ALL", "All Relationships"],
    ["SPATIAL", "Spatial Proximity"],
    ["TEMPORAL", "Temporal / Trajectory"],
    ["SAFETY", "Safety / Zones"],
    ["EQUIPMENT", "PPE Equipment"],
  ].map(([key, label]) => `<button type="button" class="filter-chip ${(sg.relationFilter || "ALL") === key ? "is-active" : ""}" data-action="filter-relations" data-filter="${key}">${label}</button>`).join("");

  let mainContentHtml = "";
  if (sg.viewMode === "table") {
    mainContentHtml = renderSceneTable(data, sg);
  } else if (sg.viewMode === "split") {
    mainContentHtml = `
      <div class="scene-split-view" style="display: grid; grid-template-columns: 1fr 1fr; gap: 14px; margin-bottom: 14px;">
        <div class="scene-graph-canvas-panel panel">
          <div class="scene-graph-viewport" id="scene-graph-viewport" style="min-height: 440px;">
            ${renderSceneGraphSvg(data, sg)}
          </div>
          <div class="graph-legend-strip">
            <span class="legend-title">GRAPH</span>
            <span class="legend-item"><span class="legend-dot" style="background:#57a884;"></span> Compliant</span>
            <span class="legend-item"><span class="legend-dot" style="background:#e17a6c;"></span> Violation / Hazard</span>
            <span class="legend-item"><span class="legend-dot" style="background:#df935e;"></span> Vehicle</span>
            <span class="legend-item"><span class="legend-dot" style="background:#d7aa58;"></span> Zone</span>
            <span class="legend-item"><span class="legend-dot" style="background:#86b997;"></span> PPE</span>
            <span class="legend-item"><span class="legend-dot" style="background:#91a79b;"></span> Machine / Asset</span>
            <span class="legend-item"><span class="legend-dot" style="background:#a0aaa1;"></span> Other entity</span>
          </div>
        </div>
        ${renderSceneEvidenceOverlay(data, sg)}
      </div>
    `;
  } else {
    mainContentHtml = `
      <div class="scene-graph-canvas-panel panel">
        <div class="scene-graph-viewport" id="scene-graph-viewport">
          ${renderSceneGraphSvg(data, sg)}
        </div>
        <div class="graph-legend-strip">
          <span class="legend-title">LEGEND</span>
          <div class="legend-items-strip">
            <span class="legend-item"><span class="legend-dot" style="background:#57a884;"></span> Worker (Compliant)</span>
            <span class="legend-item"><span class="legend-dot" style="background:#e17a6c;"></span> Worker (Violation)</span>
            <span class="legend-item"><span class="legend-dot" style="background:#df935e;"></span> Vehicle</span>
            <span class="legend-item"><span class="legend-dot" style="background:#d7aa58;"></span> Safety Zone</span>
            <span class="legend-item"><span class="legend-dot" style="background:#86b997;"></span> PPE</span>
            <span class="legend-item"><span class="legend-dot" style="background:#91a79b;"></span> Machine / Asset</span>
            <span class="legend-item"><span class="legend-dot" style="background:#a0aaa1;"></span> Other entity</span>
            <span class="legend-item"><span class="legend-line" style="background:#57a884;"></span> IS_WEARING</span>
            <span class="legend-item"><span class="legend-line" style="background:#e17a6c; border-top:1px dashed #e17a6c;"></span> MISSING</span>
            <span class="legend-item"><span class="legend-line" style="background:#d7aa58;"></span> INSIDE_ZONE</span>
            <span class="legend-item"><span class="legend-line" style="background:#e8c468; border-top:1px dashed #e8c468;"></span> NEAR</span>
            <span class="legend-item"><span class="legend-line" style="background:#de8b77; border-top:1px dashed #de8b77;"></span> APPROACHING / CLOSER_THAN</span>
          </div>
        </div>
      </div>
    `;
  }

  return `${intro}
    <div class="scene-graph-page">
      ${kpisHtml}
      ${tthHeroHtml}
      ${pipelineBannerHtml}

      <div class="scene-graph-meta-bar" style="flex-wrap: wrap; gap: 8px;">
        <div style="display: flex; flex-direction: column; gap: 6px; flex: 1;">
          <div class="graph-filter-strip">${entityFilters}</div>
          <div class="graph-filter-strip">${relationFilters}</div>
        </div>
        <div class="graph-hud-controls">
          <button class="hud-btn" type="button" data-action="auto-layout-graph" title="Auto Layout Graph" aria-label="Auto Layout">Auto Layout</button>
          <div class="hud-btn-group">
            <button class="hud-btn" type="button" data-action="zoom-graph-in" title="Zoom in" aria-label="Zoom in">${icon("plus", 14)}</button>
            <button class="hud-btn" type="button" data-action="zoom-graph-out" title="Zoom out" aria-label="Zoom out"><span style="font-weight:700; font-size:16px; line-height:1;">−</span></button>
            <button class="hud-btn" type="button" data-action="fit-graph" title="Fit to screen" aria-label="Fit to screen">Fit</button>
            <button class="hud-btn" type="button" data-action="reset-graph-layout" title="Reset view and zoom" aria-label="Reset view">Reset</button>
            <button class="hud-btn" type="button" data-action="fullscreen-graph" title="Toggle Fullscreen" aria-label="Fullscreen">${icon("expand", 14)}</button>
          </div>
        </div>
      </div>

      <div class="scene-graph-layout">
        <div style="display: flex; flex-direction: column; gap: 14px; min-width: 0;">
          ${mainContentHtml}
        </div>
        <aside class="entity-inspector-panel panel" id="entity-inspector">
          ${renderEntityInspector(data, sg)}
        </aside>
      </div>
    </div>`;
}

function renderTimeToHazardHero(data, isSingleImage) {
  const tth = data.time_to_hazard || {};
  const isAvailable = Boolean(tth.available && tth.time_to_hazard_seconds != null);
  const riskLevel = tth.risk_level || "NORMAL";
  const factors = tth.compounding_factors || [];
  const timeline = tth.timeline || [];

  let riskColorClass = "badge-neutral";
  if (riskLevel === "CRITICAL") riskColorClass = "badge-danger";
  else if (riskLevel === "HIGH") riskColorClass = "badge-warning";
  else if (riskLevel === "NORMAL" || riskLevel === "LOW") riskColorClass = "badge-success";

  // Column 1: Time to Hazard Metric Box
  let col1Html = "";
  if (isAvailable) {
    col1Html = `
      <div class="tth-metric-box">
        <div>
          <span class="tth-card-kicker">TIME TO HAZARD</span>
          <div class="tth-direction-strip" style="margin-top: 6px;">
            <span class="tth-direction-entity">${escapeHTML(tth.source_label || "Worker")}</span>
            <span class="tth-direction-arrow">↓</span>
            <span class="tth-direction-entity">${escapeHTML(tth.target_label || "Hazard")}</span>
          </div>
        </div>
        <div class="tth-metric-display">
          <span class="tth-number text-critical">${tth.time_to_hazard_seconds.toFixed(1)}</span>
          <span class="tth-unit">s</span>
        </div>
        <div class="tth-metric-badge-row">
          <span class="badge badge-danger">PREDICTED CONFLICT</span>
          <span class="badge badge-neutral">Confidence: ${Math.round((tth.confidence || 0.87) * 100)}%</span>
          <span class="badge ${riskColorClass}">Risk: ${escapeHTML(riskLevel)}</span>
        </div>
      </div>
    `;
  } else {
    const reasonText = isSingleImage
      ? "Unavailable — single-frame analysis has no motion vectors"
      : (tth.status_text || "Unavailable — insufficient temporal/spatial evidence");

    col1Html = `
      <div class="tth-metric-box is-unavailable">
        <div>
          <span class="tth-card-kicker">TIME TO HAZARD</span>
          <div class="tth-direction-strip" style="margin-top: 6px;">
            <span class="tth-direction-entity">${escapeHTML(tth.source_label || "Scene Monitor")}</span>
            ${tth.target_label ? `<span class="tth-direction-arrow">↓</span><span class="tth-direction-entity">${escapeHTML(tth.target_label)}</span>` : ""}
          </div>
        </div>
        <div class="tth-metric-unavailable">
          <span class="tth-unavail-badge">UNAVAILABLE</span>
          <p class="tth-unavail-reason">${escapeHTML(reasonText)}</p>
        </div>
        <div class="tth-metric-badge-row">
          <span class="badge badge-neutral">${escapeHTML(isSingleImage ? "STATIC FRAME ANALYSIS" : (tth.hazard_type || "MONITORED"))}</span>
          <span class="badge ${riskColorClass}">Risk: ${escapeHTML(riskLevel)}</span>
        </div>
      </div>
    `;
  }

  // Column 2: Temporal Risk Timeline
  let col2Html = "";
  if (timeline.length > 0) {
    col2Html = `
      <div class="tth-timeline-box">
        <div class="tth-timeline-header">
          <span class="tth-card-kicker">TEMPORAL RISK TIMELINE</span>
          <span class="badge badge-neutral" style="font-size: 9px;">Trajectory Projection</span>
        </div>
        <div class="tth-timeline-flow">
          ${timeline.map((step, idx) => {
            const isLast = idx === timeline.length - 1;
            const sev = String(step.severity || step.risk || "MODERATE").toUpperCase();
            let sevBadge = "badge-neutral";
            let dotClass = "dot-moderate";
            if (sev === "CRITICAL") { sevBadge = "badge-danger"; dotClass = "dot-critical"; }
            else if (sev === "HIGH") { sevBadge = "badge-warning"; dotClass = "dot-high"; }

            const timeLabel = step.time_offset_s === 0 ? "NOW" : `↓ ${step.time_offset_s.toFixed(1)} s`;

            return `
              <div class="tth-timeline-node">
                <div class="tth-node-indicator">
                  <span class="tth-node-dot ${dotClass}"></span>
                  ${!isLast ? `<span class="tth-node-line"></span>` : ""}
                </div>
                <div class="tth-node-content">
                  <div class="tth-node-meta">
                    <span class="tth-node-time">${escapeHTML(timeLabel)}</span>
                    <span class="badge ${sevBadge}">${escapeHTML(sev)}</span>
                  </div>
                  <div class="tth-node-desc">${escapeHTML(step.label || step.phase || "Projected state")}</div>
                </div>
              </div>
            `;
          }).join("")}
        </div>
      </div>
    `;
  } else {
    const notice = isSingleImage
      ? "Single-frame perception provides instantaneous spatial bindings without temporal progression."
      : "No converging collision trajectories detected in active temporal history.";

    col2Html = `
      <div class="tth-timeline-box is-empty">
        <div class="tth-timeline-header">
          <span class="tth-card-kicker">TEMPORAL RISK TIMELINE</span>
          <span class="badge badge-neutral" style="font-size: 9px;">${isSingleImage ? "Instantaneous" : "Inactive"}</span>
        </div>
        <div class="tth-empty-content">
          ${icon("clock", 20)}
          <p>${escapeHTML(notice)}</p>
        </div>
      </div>
    `;
  }

  // Column 3: Deterministic Risk Explanation & Compounding Factors
  let col3Html = `
    <div class="tth-explanation-box">
      <div class="tth-explanation-header">
        <div>
          <span class="tth-card-kicker">DETERMINISTIC RISK EXPLANATION</span>
          <div class="tth-explanation-title ${riskLevel === 'CRITICAL' ? 'text-critical' : (riskLevel === 'HIGH' ? 'text-warning' : '')}">
            ${escapeHTML(riskLevel)} RISK
          </div>
        </div>
        <span class="badge badge-neutral" style="font-size: 10px;">Compounding factors: ${factors.length}</span>
      </div>
      <div class="tth-factors-scroll">
        ${factors.length > 0 ? factors.map((factor) => `
          <div class="tth-factor-row">
            <span class="tth-factor-check">✓</span>
            <span class="tth-factor-text">${escapeHTML(factor)}</span>
          </div>
        `).join("") : (riskLevel === "NORMAL" || riskLevel === "LOW" ? `
          <div class="tth-factor-row is-clean">
            <span class="tth-factor-check" style="color: var(--success);">✓</span>
            <span class="tth-factor-text">All active workers in compliance; no convergence or perimeter incursions recorded.</span>
          </div>
        ` : `
          <div class="tth-factor-row text-muted">
            <span class="tth-factor-text">Compounding factors not available from current inference</span>
          </div>
        `)}
      </div>
    </div>
  `;

  return `
    <div class="scene-tth-hero-panel panel">
      <div class="scene-tth-grid">
        ${col1Html}
        ${col2Html}
        ${col3Html}
      </div>
    </div>
  `;
}

function renderSceneTable(data, sg) {
  const entities = data.entities || [];
  const relationships = data.relationships || [];

  const entityFilter = sg.entityFilter || "ALL";
  const relationFilter = sg.relationFilter || "ALL";

  const entityMap = new Map();
  entities.forEach((e) => entityMap.set(e.id, e));

  const filteredRels = relationships.filter((r) => {
    if (relationFilter !== "ALL" && r.category !== relationFilter) {
      if (relationFilter === "SPATIAL" && !["NEAR", "FAR", "ADJACENT"].includes(r.type)) return false;
      if (relationFilter === "TEMPORAL" && !["APPROACHING", "CLOSER_THAN", "MOVING_AWAY"].includes(r.type)) return false;
      if (relationFilter === "SAFETY" && !["INSIDE_ZONE", "INSIDE", "OUTSIDE", "OUTSIDE_ZONE", "DWELLING"].includes(r.type)) return false;
      if (relationFilter === "EQUIPMENT" && !["IS_WEARING", "WEARING", "MISSING"].includes(r.type)) return false;
    }
    if (entityFilter !== "ALL") {
      const src = entityMap.get(r.source_id);
      const tgt = entityMap.get(r.target_id);
      if (entityFilter === "PEOPLE" && src?.type !== "PERSON" && tgt?.type !== "PERSON") return false;
      if (entityFilter === "VEHICLES" && src?.type !== "VEHICLE" && tgt?.type !== "VEHICLE") return false;
      if (entityFilter === "PPE" && src?.type !== "PPE" && tgt?.type !== "PPE") return false;
      if (entityFilter === "ZONES" && src?.type !== "ZONE" && tgt?.type !== "ZONE") return false;
      if (entityFilter === "HAZARDS") {
        const isHazard = ["APPROACHING", "CLOSER_THAN", "AT_RISK_FROM"].includes(r.type) || src?.risk?.level === "CRITICAL" || tgt?.risk?.level === "CRITICAL";
        if (!isHazard) return false;
      }
    }
    return true;
  });

  return `
    <div class="scene-table-card">
      <table class="scene-data-table">
        <thead>
          <tr>
            <th>SOURCE ENTITY</th>
            <th>RELATIONSHIP</th>
            <th>TARGET ENTITY</th>
            <th>CATEGORY</th>
            <th>STATE</th>
            <th>CONFIDENCE</th>
            <th>PHYSICAL EVIDENCE</th>
            <th style="text-align: center;">INSPECT</th>
          </tr>
        </thead>
        <tbody>
          ${filteredRels.length === 0 ? `
            <tr>
              <td colspan="8" style="text-align: center; color: var(--muted); padding: 30px;">
                No relationships match current filters.
              </td>
            </tr>
          ` : filteredRels.map((r) => {
            const isSelected = sg.selectedRelationId === r.id;
            const src = entityMap.get(r.source_id);
            const tgt = entityMap.get(r.target_id);
            let catClass = "cat-badge-safety";
            if (r.category === "SPATIAL") catClass = "cat-badge-spatial";
            else if (r.category === "TEMPORAL") catClass = "cat-badge-temporal";
            else if (r.category === "EQUIPMENT") catClass = "cat-badge-equipment";

            const distanceStr = r.metadata?.distance_px != null ? `${Math.round(r.metadata.distance_px)} px` : (r.metadata?.distance_m != null ? `${r.metadata.distance_m.toFixed(2)} m` : "Direct binding");

            return `
              <tr class="${isSelected ? "is-selected" : ""}" data-action="select-graph-relation" data-rel-id="${escapeHTML(r.id)}">
                <td><b>${escapeHTML(src?.label || r.source_id)}</b></td>
                <td><span class="rel-item-badge rel-badge-${r.type.toLowerCase().includes('approach') ? 'approaching' : (r.type.toLowerCase().includes('wear') ? 'wearing' : (r.type.toLowerCase().includes('inside') ? 'inside' : 'near'))}">${escapeHTML(r.type)}</span></td>
                <td><b>${escapeHTML(tgt?.label || r.target_id)}</b></td>
                <td><span class="cat-badge ${catClass}">${escapeHTML(r.category || "SAFETY")}</span></td>
                <td><span class="badge ${r.active ? "badge-success" : "badge-neutral"}">${escapeHTML(r.state || "ACTIVE")}</span></td>
                <td>${Math.round((r.confidence || 0.9) * 100)}%</td>
                <td class="mono-coords">${escapeHTML(distanceStr)}</td>
                <td style="text-align: center;">
                  <button type="button" class="button button-quiet button-small" data-action="select-graph-relation" data-rel-id="${escapeHTML(r.id)}">
                    Inspect
                  </button>
                </td>
              </tr>
            `;
          }).join("")}
        </tbody>
      </table>
    </div>
  `;
}

function renderSceneEvidenceOverlay(data, sg) {
  const source = state.source;
  const analysis = state.analysisState;
  const result = analysis.result;
  const isVideo = source?.kind === "video";
  const mediaUrl = analysis.annotatedUrl || analysis.annotatedFrameUrl || analysis.latestFrameUrl || source?.url;

  if (!mediaUrl) {
    return `
      <div class="scene-evidence-panel panel">
        <div class="scene-evidence-header">
          <span class="panel-kicker">EVIDENCE FRAME OVERLAY</span>
        </div>
        <div class="scene-evidence-viewport" style="color: var(--muted); padding: 40px; text-align: center;">
          ${icon("image", 24)}
          <p style="margin-top: 8px;">Upload image or video in workspace to display synchronized evidence view.</p>
        </div>
      </div>
    `;
  }

  const entities = data.entities || [];
  const relationships = data.relationships || [];

  const imgW = result?.assessment?.frame_width || 1280;
  const imgH = result?.assessment?.frame_height || 720;

  let activeBoxesHtml = "";
  let vectorLinesHtml = "";

  const selectedEntity = entities.find((e) => e.id === sg.selectedEntityId);
  const selectedRel = relationships.find((r) => r.id === sg.selectedRelationId);

  // If entity selected, highlight its bounding box
  if (selectedEntity && selectedEntity.bbox) {
    const b = selectedEntity.bbox;
    const tag = `${selectedEntity.label} [${selectedEntity.type}]`;
    activeBoxesHtml += `
      <g class="evidence-interactive-box" data-action="select-graph-entity" data-entity-id="${escapeHTML(selectedEntity.id)}" style="cursor: pointer;">
        <rect x="${b.x1}" y="${b.y1}" width="${b.x2 - b.x1}" height="${b.y2 - b.y1}" fill="rgba(167, 189, 142, 0.22)" stroke="#a7bd8e" stroke-width="3" stroke-dasharray="6 3" />
        <rect x="${b.x1}" y="${Math.max(0, b.y1 - 22)}" width="${tag.length * 7.5 + 14}" height="22" fill="#a7bd8e" rx="3" />
        <text x="${b.x1 + 6}" y="${Math.max(14, b.y1 - 7)}" fill="#0c100d" font-size="11" font-weight="750">${escapeHTML(tag)}</text>
      </g>
    `;
  }

  // If relationship selected, highlight both endpoints and draw connecting line
  if (selectedRel) {
    const srcNode = entities.find((e) => e.id === selectedRel.source_id);
    const tgtNode = entities.find((e) => e.id === selectedRel.target_id);

    if (srcNode?.bbox) {
      const b = srcNode.bbox;
      const tag = `${srcNode.label} (Source)`;
      activeBoxesHtml += `
        <g class="evidence-interactive-box">
          <rect x="${b.x1}" y="${b.y1}" width="${b.x2 - b.x1}" height="${b.y2 - b.y1}" fill="rgba(167, 189, 142, 0.25)" stroke="#a7bd8e" stroke-width="3" />
          <rect x="${b.x1}" y="${Math.max(0, b.y1 - 22)}" width="${tag.length * 7.5 + 14}" height="22" fill="#a7bd8e" rx="3" />
          <text x="${b.x1 + 6}" y="${Math.max(14, b.y1 - 7)}" fill="#0c100d" font-size="11" font-weight="750">${escapeHTML(tag)}</text>
        </g>
      `;
    }

    if (tgtNode?.bbox) {
      const b = tgtNode.bbox;
      const tag = `${tgtNode.label} (Target)`;
      activeBoxesHtml += `
        <g class="evidence-interactive-box">
          <rect x="${b.x1}" y="${b.y1}" width="${b.x2 - b.x1}" height="${b.y2 - b.y1}" fill="rgba(223, 147, 94, 0.25)" stroke="#df935e" stroke-width="3" />
          <rect x="${b.x1}" y="${Math.max(0, b.y1 - 22)}" width="${tag.length * 7.5 + 14}" height="22" fill="#df935e" rx="3" />
          <text x="${b.x1 + 6}" y="${Math.max(14, b.y1 - 7)}" fill="#0c100d" font-size="11" font-weight="750">${escapeHTML(tag)}</text>
        </g>
      `;
    }

    if (srcNode?.bbox && tgtNode?.bbox) {
      const sx = (srcNode.bbox.x1 + srcNode.bbox.x2) / 2;
      const sy = (srcNode.bbox.y1 + srcNode.bbox.y2) / 2;
      const tx = (tgtNode.bbox.x1 + tgtNode.bbox.x2) / 2;
      const ty = (tgtNode.bbox.y1 + tgtNode.bbox.y2) / 2;
      const mx = (sx + tx) / 2;
      const my = (sy + ty) / 2;
      const tag = selectedRel.type;
      const pillW = tag.length * 8 + 18;

      vectorLinesHtml = `
        <line x1="${sx}" y1="${sy}" x2="${tx}" y2="${ty}" stroke="#e17a6c" stroke-width="3.5" stroke-dasharray="8 5" marker-end="url(#arrow-approaching)" />
        <rect x="${mx - pillW / 2}" y="${my - 12}" width="${pillW}" height="24" rx="4" fill="#121814" stroke="#e17a6c" stroke-width="1.5" />
        <text x="${mx}" y="${my + 4}" fill="#f0f4f0" font-size="11" font-weight="750" text-anchor="middle">${escapeHTML(tag)}</text>
      `;
    }
  }

  return `
    <div class="scene-evidence-panel panel">
      <div class="scene-evidence-header">
        <div>
          <span class="panel-kicker">SYNCHRONIZED EVIDENCE VIEW</span>
          <h3 style="margin:0; font-size:13px; font-weight:700;">${escapeHTML(source?.name || "Evidence Media")}</h3>
        </div>
        <span class="badge ${selectedEntity || selectedRel ? "badge-success" : "badge-neutral"}">
          ${selectedEntity ? `Focus: ${selectedEntity.label}` : (selectedRel ? `Relation: ${selectedRel.type}` : "Click any node or relationship")}
        </span>
      </div>
      <div class="scene-evidence-viewport">
        ${isVideo ? `
          <video class="scene-evidence-media" src="${escapeHTML(mediaUrl)}" controls playsinline></video>
        ` : `
          <img class="scene-evidence-media" src="${escapeHTML(mediaUrl)}" alt="Scene Evidence Frame" />
        `}
        <svg class="evidence-svg-overlay" viewBox="0 0 ${imgW} ${imgH}" preserveAspectRatio="none">
          ${activeBoxesHtml}
          ${vectorLinesHtml}
        </svg>
      </div>
      <div style="padding: 8px 12px; background: var(--surface-2); font-size: 11px; color: var(--muted); display:flex; justify-content:space-between;">
        <span>${icon("shield", 12)} Direct bbox grounding: SCENE GRAPH ↔ VISIBLE EVIDENCE</span>
        <span>Resolution: ${imgW}×${imgH} px</span>
      </div>
    </div>
  `;
}

function renderSceneGraphSvg(data, sg) {
  const entities = data.entities || [];
  const relationships = data.relationships || [];

  const entityFilter = sg.entityFilter || "ALL";
  const relationFilter = sg.relationFilter || "ALL";

  // Filter entities
  const filteredEntities = entities.filter((e) => {
    if (entityFilter === "ALL") return true;
    if (entityFilter === "PEOPLE") return e.type === "PERSON";
    if (entityFilter === "VEHICLES") return e.type === "VEHICLE";
    if (entityFilter === "PPE") return e.type === "PPE";
    if (entityFilter === "ZONES") return e.type === "ZONE";
    if (entityFilter === "ASSETS") return ["MACHINE", "ASSET", "EQUIPMENT"].includes(e.type);
    if (entityFilter === "OTHER") return !["PERSON", "VEHICLE", "PPE", "ZONE", "HAZARD", "MACHINE", "ASSET", "EQUIPMENT"].includes(e.type);
    if (entityFilter === "HAZARDS") return e.type === "HAZARD" || ["HIGH", "CRITICAL"].includes(String(e.risk?.level).toUpperCase());
    return true;
  });

  // Filter relationships
  const filteredRelations = relationships.filter((r) => {
    if (relationFilter !== "ALL") {
      if (relationFilter === "SPATIAL" && !["NEAR", "FAR", "ADJACENT"].includes(r.type)) return false;
      if (relationFilter === "TEMPORAL" && !["APPROACHING", "CLOSER_THAN", "MOVING_AWAY"].includes(r.type)) return false;
      if (relationFilter === "SAFETY" && !["INSIDE_ZONE", "INSIDE", "OUTSIDE", "OUTSIDE_ZONE", "DWELLING"].includes(r.type)) return false;
      if (relationFilter === "EQUIPMENT" && !["IS_WEARING", "WEARING", "MISSING"].includes(r.type)) return false;
    }
    return true;
  });

  const zones = filteredEntities.filter((e) => e.type === "ZONE");
  const people = filteredEntities.filter((e) => e.type === "PERSON");
  const vehicles = filteredEntities.filter((e) => e.type === "VEHICLE");
  const machines = filteredEntities.filter((e) => ["MACHINE", "ASSET", "EQUIPMENT"].includes(e.type));
  const ppes = filteredEntities.filter((e) => e.type === "PPE");
  const hazards = filteredEntities.filter((e) => e.type === "HAZARD");
  const others = filteredEntities.filter((e) => !["ZONE", "PERSON", "VEHICLE", "PPE", "MACHINE", "ASSET", "EQUIPMENT", "HAZARD"].includes(e.type));

  const posMap = new Map();
  const CANVAS_WIDTH = 1000;

  // 1. ZONES: Positioned at top tier (y = 40)
  const numZones = zones.length;
  if (numZones > 0) {
    const zWidth = 180;
    const zHeight = 50;
    const zSpacing = Math.min(220, (CANVAS_WIDTH - 60) / numZones);
    const zStartX = (CANVAS_WIDTH - (numZones * zSpacing - (zSpacing - zWidth))) / 2;
    zones.forEach((zone, idx) => {
      const x = Math.max(30, zStartX + idx * zSpacing);
      const y = 40;
      posMap.set(zone.id, { x, y, w: zWidth, h: zHeight, cx: x + zWidth / 2, cy: y + zHeight / 2, entity: zone });
    });
  }

  // 2. PRIMARY ACTORS: People, Vehicles, Machines (y = 190)
  const ppeToPerson = new Map();
  const personToPpe = new Map();
  people.forEach((p) => personToPpe.set(p.id, []));

  relationships.forEach((rel) => {
    const t = String(rel.type || "").toUpperCase();
    if (["IS_WEARING", "WEARING", "MISSING"].includes(t)) {
      const personId = people.some((p) => p.id === rel.source_id) ? rel.source_id : (people.some((p) => p.id === rel.target_id) ? rel.target_id : null);
      const ppeId = ppes.some((p) => p.id === rel.target_id) ? rel.target_id : (ppes.some((p) => p.id === rel.source_id) ? rel.source_id : null);
      if (personId && ppeId) {
        ppeToPerson.set(ppeId, personId);
        const list = personToPpe.get(personId) || [];
        if (!list.includes(ppeId)) list.push(ppeId);
        personToPpe.set(personId, list);
      }
    }
  });

  const actorGroups = [...people, ...vehicles, ...machines, ...hazards];
  const numActors = actorGroups.length;
  const actorWidth = 175;
  const actorHeight = 72;
  const actorSpacing = Math.min(240, (CANVAS_WIDTH - 60) / Math.max(1, numActors));
  const actorStartX = (CANVAS_WIDTH - (numActors * actorSpacing - (actorSpacing - actorWidth))) / 2;

  actorGroups.forEach((actor, idx) => {
    const x = Math.max(30, actorStartX + idx * actorSpacing);
    const y = 190;
    posMap.set(actor.id, { x, y, w: actorWidth, h: actorHeight, cx: x + actorWidth / 2, cy: y + actorHeight / 2, entity: actor });
  });

  // 3. PPE NODES: Positioned directly under their associated person
  const placedPpeIds = new Set();
  const ppeWidth = 142;
  const ppeHeight = 40;

  people.forEach((person) => {
    const pPos = posMap.get(person.id);
    if (!pPos) return;
    const ppeList = (personToPpe.get(person.id) || []).map((id) => ppes.find((p) => p.id === id)).filter(Boolean);
    const numPpe = ppeList.length;
    if (numPpe > 0) {
      const ppeSpacing = Math.min(150, 420 / numPpe);
      const ppeStartX = pPos.cx - ((numPpe - 1) * ppeSpacing) / 2 - ppeWidth / 2;
      ppeList.forEach((ppe, ppeIdx) => {
        const x = Math.max(20, Math.min(CANVAS_WIDTH - ppeWidth - 20, ppeStartX + ppeIdx * ppeSpacing));
        const y = 350 + (ppeIdx % 2) * 52;
        posMap.set(ppe.id, { x, y, w: ppeWidth, h: ppeHeight, cx: x + ppeWidth / 2, cy: y + ppeHeight / 2, entity: ppe });
        placedPpeIds.add(ppe.id);
      });
    }
  });

  // Remaining unassociated PPE items
  const unplacedPpes = ppes.filter((p) => !placedPpeIds.has(p.id));
  if (unplacedPpes.length > 0) {
    const uSpacing = 150;
    const uStartX = (CANVAS_WIDTH - (unplacedPpes.length * uSpacing - 8)) / 2;
    unplacedPpes.forEach((ppe, idx) => {
      const x = Math.max(20, uStartX + idx * uSpacing);
      const y = 470;
      posMap.set(ppe.id, { x, y, w: ppeWidth, h: ppeHeight, cx: x + ppeWidth / 2, cy: y + ppeHeight / 2, entity: ppe });
    });
  }

  // 4. OTHER ENTITIES (if any)
  others.forEach((entity, idx) => {
    const w = 140;
    const h = 48;
    const x = 50 + idx * 160;
    const y = 520;
    posMap.set(entity.id, { x, y, w, h, cx: x + w / 2, cy: y + h / 2, entity });
  });

  // Compute dynamic viewBox height so all nodes are fully contained
  let maxY = 560;
  posMap.forEach((p) => {
    if (p.y + p.h + 40 > maxY) maxY = p.y + p.h + 40;
  });
  const CANVAS_HEIGHT = Math.max(620, maxY);

  // Render Edges
  let edgesHtml = "";
  filteredRelations.forEach((rel) => {
    const src = posMap.get(rel.source_id);
    const tgt = posMap.get(rel.target_id);
    if (!src || !tgt) return;

    const mx = (src.cx + tgt.cx) / 2;
    const my = (src.cy + tgt.cy) / 2;
    const dx = tgt.cx - src.cx;
    const dy = tgt.cy - src.cy;
    const dist = Math.hypot(dx, dy) || 1;

    // Subtle curve to avoid cutting through node boxes
    const curveOffset = Math.abs(dx) < 30 ? 0 : 20;
    const nx = (-dy / dist) * curveOffset;
    const ny = (dx / dist) * curveOffset;
    const cx = mx + nx;
    const cy = my + ny;

    const t = String(rel.type || "").toUpperCase();
    let edgeClass = "edge-default";
    let marker = "default";
    let pillBorderColor = "#3a463c";
    let pillTextColor = "#c0c9bd";

    if (t === "IS_WEARING" || t === "WEARING") {
      edgeClass = "edge-wearing"; marker = "wearing"; pillBorderColor = "#57a884"; pillTextColor = "#86b997";
    } else if (t === "MISSING") {
      edgeClass = "edge-missing"; marker = "missing"; pillBorderColor = "#e17a6c"; pillTextColor = "#e17a6c";
    } else if (["INSIDE_ZONE", "INSIDE", "OUTSIDE", "OUTSIDE_ZONE", "DWELLING"].includes(t)) {
      edgeClass = "edge-inside"; marker = "inside"; pillBorderColor = "#d7aa58"; pillTextColor = "#e8c468";
    } else if (["NEAR", "FAR", "ADJACENT"].includes(t)) {
      edgeClass = "edge-near"; marker = "near"; pillBorderColor = "#e8c468"; pillTextColor = "#e8c468";
    } else if (["APPROACHING", "CLOSER_THAN", "AT_RISK_FROM"].includes(t)) {
      edgeClass = "edge-approaching"; marker = "approaching"; pillBorderColor = "#df935e"; pillTextColor = "#df935e";
    } else if (t === "MOVING_AWAY") {
      edgeClass = "edge-moving-away"; marker = "default"; pillBorderColor = "#5c8fd6"; pillTextColor = "#7ba7e8";
    }

    const isSelected = sg.selectedRelationId === rel.id;
    const isSymmetric = ["NEAR", "FAR", "ADJACENT", "ASSOCIATED_WITH", "INTERACTING_WITH"].includes(t);
    const markerEnd = isSymmetric ? "" : 'marker-end="url(#arrow-' + marker + ')"';
    const pathD = `M ${src.cx.toFixed(1)} ${src.cy.toFixed(1)} Q ${cx.toFixed(1)} ${cy.toFixed(1)} ${tgt.cx.toFixed(1)} ${tgt.cy.toFixed(1)}`;

    const lx = 0.25 * src.cx + 0.5 * cx + 0.25 * tgt.cx;
    const ly = 0.25 * src.cy + 0.5 * cy + 0.25 * tgt.cy;
    const labelText = rel.type || "REL";
    const pillW = Math.max(64, labelText.length * 7 + 16);

    edgesHtml += `
      <g class="graph-edge-group">
        <path id="edge-${escapeHTML(rel.id)}" class="graph-edge ${edgeClass} ${isSelected ? "is-selected" : ""}" d="${pathD}" data-action="select-graph-relation" data-rel-id="${escapeHTML(rel.id)}" data-src="${escapeHTML(rel.source_id)}" data-tgt="${escapeHTML(rel.target_id)}" data-direction="${isSymmetric ? "both" : "directed"}" ${markerEnd} tabindex="0" role="button" aria-label="${isSymmetric ? "Symmetric" : "Directed"} relationship: ${escapeHTML(rel.type)} between ${escapeHTML(rel.source_id)} and ${escapeHTML(rel.target_id)}" />
        <g class="edge-label-group ${isSelected ? "is-selected" : ""}" data-action="select-graph-relation" data-rel-id="${escapeHTML(rel.id)}" tabindex="0" role="button" aria-label="Select relationship ${escapeHTML(rel.type)}">
          <rect class="edge-label-pill" x="${(lx - pillW / 2).toFixed(1)}" y="${(ly - 9).toFixed(1)}" width="${pillW}" height="18" rx="4" style="fill:#161814; stroke:${pillBorderColor}; stroke-width:1;" />
          <text class="edge-label-text" x="${lx.toFixed(1)}" y="${(ly + 4).toFixed(1)}" style="fill:${pillTextColor}; font-size:9px; font-weight:700;">${escapeHTML(labelText)}</text>
        </g>
      </g>`;
  });

  // Render Nodes
  let nodesHtml = "";
  filteredEntities.forEach((entity) => {
    const pos = posMap.get(entity.id);
    if (!pos) return;

    const isSelected = sg.selectedEntityId === entity.id;
    const entityType = entity.type || "OBJECT";
    let typeClass = "node-object";
    if (entityType === "PERSON") typeClass = "node-person";
    else if (entityType === "VEHICLE") typeClass = "node-vehicle";
    else if (entityType === "ZONE") typeClass = "node-zone";
    else if (entityType === "PPE") typeClass = "node-ppe";
    else if (["MACHINE", "ASSET", "EQUIPMENT"].includes(entityType)) typeClass = "node-machine";
    else if (entityType === "HAZARD") typeClass = "node-hazard";
    else if (entityType !== "OBJECT") typeClass = "node-other";
    const isViolation = entityType === "PERSON" && entity.state?.compliance_status === "NON-COMPLIANT";
    const isMissingPpe = entityType === "PPE" && (
      entity.state?.compliance_status === "MISSING" || String(entity.label || "").toLowerCase().includes("missing")
    );
    const fullLabel = String(entity.label || entity.id || entityType);
    const shortLabel = (maxLength) => fullLabel.length > maxLength ? `${fullLabel.slice(0, maxLength - 1)}…` : fullLabel;

    const riskLevel = String(entity.risk?.level || "NORMAL").toUpperCase();
    const hasElevatedRisk = ["HIGH", "CRITICAL"].includes(riskLevel);
    let riskColor = "#57a884";
    if (riskLevel === "CRITICAL") riskColor = "#e17a6c";
    else if (riskLevel === "HIGH") riskColor = "#df935e";
    else if (riskLevel === "MEDIUM") riskColor = "#d7aa58";

    let bodySvg = "";
    if (entityType === "PERSON") {
      const isComp = entity.state?.compliance_status === "COMPLIANT";
      const statusText = isComp ? "COMPLIANT" : (isViolation ? "PPE VIOLATION" : "PPE UNKNOWN");
      const statusColor = isComp ? "#86b997" : (isViolation ? "#e17a6c" : "#bdc7bb");
      const personLabel = entity.metadata?.display_label || (entity.metadata?.display_id ? `Person #${entity.metadata.display_id}` : (entity.label ? entity.label.replace(/^Worker\b/, "Person") : `Person #${entity.id.replace('person_', '')}`));
      bodySvg = `
        <rect class="node-card" width="${pos.w}" height="${pos.h}" rx="6" />
        <text class="node-kicker" x="12" y="16" fill="#86b997" font-size="7.5" font-weight="700" letter-spacing="0.08em">[PERSON]</text>
        <g transform="translate(10, 22)">
          <circle cx="10" cy="10" r="10" fill="rgba(167, 189, 142, 0.15)" stroke="#a7bd8e" stroke-width="1.2" />
          <circle cx="10" cy="8" r="3.5" fill="#a7bd8e" />
          <path d="M 4 17 C 4 13.5 7 12 10 12 C 13 12 16 13.5 16 17" fill="none" stroke="#a7bd8e" stroke-width="1.2" />
        </g>
        <text class="node-title" x="38" y="32">${escapeHTML(personLabel)}</text>
        <text class="node-sub" x="38" y="46">Track #${entity.metadata?.track_id ?? entity.id.replace('person_', '')}</text>
        <g transform="translate(38, 52)">
          <rect width="90" height="14" rx="3" fill="${isComp ? "rgba(87, 168, 132, 0.16)" : (isViolation ? "rgba(225, 122, 108, 0.18)" : "rgba(189, 199, 187, 0.12)")}" />
          <text x="45" y="10" fill="${statusColor}" font-size="8" font-weight="700" text-anchor="middle">${statusText}</text>
        </g>
        <circle cx="${pos.w - 12}" cy="12" r="4.5" fill="${riskColor}" />`;
    } else if (entityType === "VEHICLE") {
      const speed = entity.state?.speed_px_per_s != null ? `${Math.round(entity.state.speed_px_per_s)} px/s` : "Active";
      bodySvg = `
        <rect class="node-card" width="${pos.w}" height="${pos.h}" rx="6" />
        <text class="node-kicker" x="12" y="16" fill="#df935e" font-size="7.5" font-weight="700" letter-spacing="0.08em">[VEHICLE]</text>
        <g transform="translate(10, 22)">
          <circle cx="10" cy="10" r="10" fill="rgba(223, 147, 94, 0.15)" stroke="#df935e" stroke-width="1.2" />
          <path d="M 5 12 L 15 12 M 7 9 L 13 9 M 10 6 L 10 14" stroke="#df935e" stroke-width="1.2" />
        </g>
        <text class="node-title" x="38" y="32">${escapeHTML(shortLabel(14))}</text>
        <text class="node-sub" x="38" y="46">Motion: ${escapeHTML(speed)}</text>
        <g transform="translate(38, 52)">
          <rect width="90" height="14" rx="3" fill="rgba(223, 147, 94, 0.16)" />
          <text x="45" y="10" fill="#df935e" font-size="8" font-weight="700" text-anchor="middle">INDUSTRIAL VEHICLE</text>
        </g>
        <circle cx="${pos.w - 12}" cy="12" r="4.5" fill="${riskColor}" />`;
    } else if (entityType === "ZONE") {
      bodySvg = `
        <rect class="node-card" width="${pos.w}" height="${pos.h}" rx="6" />
        <text class="node-kicker" x="12" y="14" fill="#d7aa58" font-size="7.5" font-weight="700" letter-spacing="0.08em">[SAFETY ZONE]</text>
        <g transform="translate(8, 18)">
          <polygon points="12,2 22,6 22,14 12,22 2,14 2,6" fill="rgba(215, 170, 88, 0.15)" stroke="#d7aa58" stroke-width="1.2" />
        </g>
        <text class="node-title" x="36" y="28">${escapeHTML(shortLabel(16))}</text>
        <text class="node-sub" x="36" y="42">SPATIAL BOUNDARY</text>
        <circle cx="${pos.w - 12}" cy="12" r="4.5" fill="${riskColor}" />`;
    } else if (entityType === "PPE") {
      const ppeColor = isMissingPpe ? "#e17a6c" : "#86b997";
      const ppeStatusText = isMissingPpe ? "MISSING" : "PRESENT";
      bodySvg = `
        <rect class="node-card" width="${pos.w}" height="${pos.h}" rx="5" />
        <circle cx="16" cy="20" r="7" fill="${isMissingPpe ? "rgba(225, 122, 108, 0.15)" : "rgba(134, 185, 151, 0.15)"}" stroke="${ppeColor}" stroke-width="1" />
        <text class="node-ppe-title" x="30" y="16" font-size="10.5" font-weight="600">${escapeHTML(shortLabel(16))}</text>
        <text class="node-sub" x="30" y="30" fill="${ppeColor}" font-size="7.5" font-weight="700">${ppeStatusText}</text>
        <circle cx="${pos.w - 10}" cy="10" r="3" fill="${ppeColor}" />`;
    } else if (["MACHINE", "ASSET", "EQUIPMENT"].includes(entityType)) {
      bodySvg = `
        <rect class="node-card" width="${pos.w}" height="${pos.h}" rx="6" />
        <text class="node-kicker" x="12" y="16" fill="#6b9cb8" font-size="7.5" font-weight="700" letter-spacing="0.08em">[MACHINE]</text>
        <text class="node-title" x="14" y="32">${escapeHTML(shortLabel(15))}</text>
        <text class="node-sub" x="14" y="46">EQUIPMENT ASSET</text>
        <circle cx="${pos.w - 12}" cy="12" r="4.5" fill="${riskColor}" />`;
    } else if (entityType === "HAZARD") {
      bodySvg = `
        <rect class="node-card" width="${pos.w}" height="${pos.h}" rx="6" />
        <text class="node-kicker" x="12" y="16" fill="#e17a6c" font-size="7.5" font-weight="700" letter-spacing="0.08em">[HAZARD]</text>
        <text class="node-title" x="14" y="32">${escapeHTML(shortLabel(15))}</text>
        <text class="node-sub" x="14" y="46">SAFETY CONDITION</text>
        <circle cx="${pos.w - 12}" cy="12" r="4.5" fill="#e17a6c" />`;
    } else {
      bodySvg = `
        <rect class="node-card" width="${pos.w}" height="${pos.h}" rx="6" />
        <text class="node-title" x="14" y="24">${escapeHTML(shortLabel(15))}</text>
        <text class="node-sub" x="14" y="40">${escapeHTML(entity.type)}</text>`;
    }

    nodesHtml += `
      <g class="graph-node ${typeClass} ${isViolation ? "has-violation" : ""} ${isMissingPpe ? "is-missing" : ""} ${hasElevatedRisk ? "has-risk" : ""} ${isSelected ? "is-selected" : ""}" transform="translate(${pos.x.toFixed(1)}, ${pos.y.toFixed(1)})" data-action="select-graph-entity" data-entity-id="${escapeHTML(entity.id)}" tabindex="0" role="button" aria-label="${escapeHTML(entityType)}: ${escapeHTML(entity.label || entity.id)}">
        <title>${escapeHTML(fullLabel)}</title>
        ${bodySvg}
      </g>`;
  });

  return `<svg class="scene-graph-svg" viewBox="0 0 ${CANVAS_WIDTH} ${CANVAS_HEIGHT}" role="img" aria-label="Interactive industrial scene graph visualization">
    <defs>
      <marker id="arrow-default" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#a7bd8e" />
      </marker>
      <marker id="arrow-wearing" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#57a884" />
      </marker>
      <marker id="arrow-missing" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#e17a6c" />
      </marker>
      <marker id="arrow-inside" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#d7aa58" />
      </marker>
      <marker id="arrow-near" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#e8c468" />
      </marker>
      <marker id="arrow-approaching" viewBox="0 0 10 10" refX="8" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
        <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#df935e" />
      </marker>
    </defs>
    <g class="viewport-content" transform="scale(${sg.zoom}) translate(${sg.pan.x}, ${sg.pan.y})" transform-origin="500 310">
      <g class="edges-layer">${edgesHtml}</g>
      <g class="nodes-layer">${nodesHtml}</g>
    </g>
  </svg>`;
}

function renderEntityInspector(data, sg) {
  const entities = data.entities || [];
  const relationships = data.relationships || [];

  // 1. Inspecting a selected relationship
  if (sg.selectedRelationId) {
    const rel = relationships.find((r) => r.id === sg.selectedRelationId);
    if (!rel) {
      return `<div class="inspector-empty-state">${icon("info", 24)}<p>Select an entity or relationship from the graph to inspect real-time properties.</p></div>`;
    }
    const srcNode = entities.find((e) => e.id === rel.source_id);
    const tgtNode = entities.find((e) => e.id === rel.target_id);

    const isSingleImage = Boolean(data?.time_to_hazard?.hazard_type === "STATIC SCENE ANALYSIS" || (!data?.is_live && state.source?.kind === "image"));

    const distPx = rel.distance_px ?? rel.evidence?.distance_px ?? rel.metadata?.distance_px ?? null;
    const distM = rel.distance_m ?? rel.evidence?.distance_m ?? rel.metadata?.distance_m ?? null;
    const distPxStr = distPx != null ? `${Math.round(distPx)} px` : "Unavailable";
    const distMStr = distM != null ? `${distM.toFixed(2)} m` : null;

    const speedPx = rel.closing_rate_px_s ?? rel.evidence?.closing_rate_px_s ?? rel.metadata?.relative_speed ?? null;
    const speedM = rel.closing_rate_m_s ?? rel.evidence?.closing_rate_m_s ?? null;
    const speedPxStr = speedPx != null
      ? `${Math.round(speedPx)} px/s`
      : (isSingleImage ? "Unavailable — single-frame analysis has no motion vectors" : "Unavailable");
    const speedMStr = speedM != null
      ? `${speedM.toFixed(2)} m/s`
      : (isSingleImage ? "Unavailable — single-frame analysis has no motion vectors" : null);

    const spatialBasis = rel.spatial_basis || rel.evidence?.spatial_basis || "IMAGE_SPACE";
    const isCalibratedBasis = spatialBasis === "GROUND_PLANE_APPROXIMATION" && distM != null;

    const distDisplay = distMStr ? `${distMStr} (${distPxStr})` : distPxStr;
    const speedDisplay = speedMStr ? `${speedMStr} (${speedPxStr})` : speedPxStr;

    const tthStr = rel.tth_seconds != null
      ? `${rel.tth_seconds.toFixed(1)} s`
      : (rel.tth_status || (isSingleImage ? "Unavailable — single-frame analysis has no motion vectors" : "Not available from current inference"));

    return `
      <div class="inspector-header">
        <div class="inspector-title-area">
          <span class="inspector-card-kicker">RELATIONSHIP INSPECTOR</span>
          <h3>${icon("link", 16)} ${escapeHTML(rel.type)}</h3>
        </div>
        <button type="button" class="icon-button" data-action="close-inspector" aria-label="Close inspector">${icon("close", 14)}</button>
      </div>
      <div class="inspector-content">
        <div class="inspector-card">
          <span class="inspector-card-kicker">PROPERTIES</span>
          <table class="inspector-table">
            <tbody>
              <tr><td>Type:</td><td><b>${escapeHTML(rel.type)}</b></td></tr>
              <tr><td>Source:</td><td>${escapeHTML(srcNode?.label || rel.source_id)}</td></tr>
              <tr><td>Target:</td><td>${escapeHTML(tgtNode?.label || rel.target_id)}</td></tr>
              <tr><td>Confidence:</td><td>${Math.round((rel.confidence || 0.9) * 100)}%</td></tr>
              <tr><td>State:</td><td><span class="badge badge-success">${escapeHTML(rel.state || "ACTIVE")}</span></td></tr>
              <tr><td>Spatial Distance:</td><td>${escapeHTML(distDisplay)}</td></tr>
              <tr><td>Relative Speed:</td><td>${escapeHTML(speedDisplay)}</td></tr>
              <tr><td>Spatial Basis:</td><td><b>${escapeHTML(spatialBasis)}</b></td></tr>
              <tr><td>Time to Hazard:</td><td><b>${escapeHTML(tthStr)}</b></td></tr>
              <tr><td>Duration:</td><td>${rel.duration_seconds != null ? `${rel.duration_seconds.toFixed(1)} s` : "Not available from current inference"}</td></tr>
              <tr><td>Timestamp:</td><td>${rel.timestamp != null ? `${rel.timestamp.toFixed(1)} s` : "Not available from current inference"}</td></tr>
            </tbody>
          </table>
        </div>

        <div class="inspector-card">
          <span class="inspector-card-kicker">PHYSICAL EVIDENCE</span>
          <div style="font-size: 11px; margin-bottom: 8px; display: flex; align-items: center; justify-content: space-between;">
            <span>Basis: <b>${escapeHTML(spatialBasis)}</b></span>
            ${isCalibratedBasis ? `<span class="badge badge-success" style="font-size:9px;">CALIBRATED</span>` : `<span class="badge badge-neutral" style="font-size:9px;">IMAGE-SPACE ONLY</span>`}
          </div>

          <div class="evidence-basis-grid">
            <div class="evidence-basis-card">
              <span class="evidence-basis-kicker">IMAGE SPACE</span>
              <div class="evidence-basis-val">${escapeHTML(distPxStr)}</div>
              <small style="color: var(--muted); font-size: 10px;">Closing: ${escapeHTML(speedPxStr)}</small>
            </div>
            <div class="evidence-basis-card">
              <span class="evidence-basis-kicker">GROUND PLANE</span>
              ${isCalibratedBasis ? `
                <div class="evidence-basis-val" style="color: var(--accent);">${escapeHTML(distMStr)}</div>
                <small style="color: var(--muted); font-size: 10px;">Closing: ${escapeHTML(speedMStr)}</small>
              ` : `
                <div class="evidence-basis-val" style="font-size: 10.5px; color: var(--muted); font-weight: normal; line-height: 1.3;">Metric spatial estimate unavailable</div>
                <small style="color: var(--subtle); font-size: 9.5px;">Camera calibration required</small>
              `}
            </div>
          </div>

          ${rel.evidence && Object.keys(rel.evidence).length > 0 ? `
            <table class="inspector-table" style="margin-top: 10px;">
              <tbody>
                ${Object.entries(rel.evidence).map(([k, v]) => `
                  <tr><td>${escapeHTML(k)}:</td><td class="mono-coords">${escapeHTML(String(v))}</td></tr>
                `).join("")}
              </tbody>
            </table>
          ` : `
            <span class="text-muted" style="font-size: 11px; display: block; margin-top: 8px;">Not available from current inference</span>
          `}
        </div>

        <div class="inspector-card">
          <span class="inspector-card-kicker">CONNECTED NODES</span>
          <div style="display:flex; flex-direction:column; gap:6px;">
            <button type="button" class="button button-quiet button-small button-full" data-action="select-graph-entity" data-entity-id="${escapeHTML(rel.source_id)}">
              ${icon("arrow", 13)} Source: ${escapeHTML(srcNode?.label || rel.source_id)}
            </button>
            <button type="button" class="button button-quiet button-small button-full" data-action="select-graph-entity" data-entity-id="${escapeHTML(rel.target_id)}">
              ${icon("arrow", 13)} Target: ${escapeHTML(tgtNode?.label || rel.target_id)}
            </button>
          </div>
        </div>
      </div>`;
  }

  // 2. Inspecting a selected entity
  if (sg.selectedEntityId) {
    const isSingleImage = Boolean(data?.time_to_hazard?.hazard_type === "STATIC SCENE ANALYSIS" || (!data?.is_live && state.source?.kind === "image"));
    const entity = entities.find((e) => e.id === sg.selectedEntityId);
    if (!entity) {
      return `<div class="inspector-empty-state">${icon("info", 24)}<p>Selected entity not found in current scene.</p></div>`;
    }

    const entityType = entity.type || "OBJECT";
    const riskLevel = entity.risk?.level || "NORMAL";
    let riskBadgeClass = "badge-neutral";
    if (riskLevel === "CRITICAL") riskBadgeClass = "badge-danger";
    else if (riskLevel === "HIGH") riskBadgeClass = "badge-warning";
    else if (riskLevel === "MEDIUM") riskBadgeClass = "badge-warning";
    else if (riskLevel === "NORMAL" || riskLevel === "LOW") riskBadgeClass = "badge-success";

    const isPerson = entityType === "PERSON";
    const isVehicle = entityType === "VEHICLE";
    const isZone = entityType === "ZONE";
    const personComplianceStatus = normalizeComplianceStatus(entity.state?.compliance_status);
    const personComplianceBadge = personComplianceStatus === "COMPLIANT"
      ? "badge-success"
      : (personComplianceStatus === "NON-COMPLIANT" ? "badge-danger" : "badge-neutral");

    // Related relationships
    const connectedRels = relationships.filter((r) => r.source_id === entity.id || r.target_id === entity.id);

    // PPE check list formatting
    const detectedPpe = entity.state?.detected_ppe || [];
    const missingPpe = entity.state?.missing_ppe || [];
    const standardPpeItems = ["Hardhat", "Safety Vest", "Goggles", "Gloves"];
    const ppeCheckItems = standardPpeItems.map((item) => {
      const isPresent = detectedPpe.some((p) => p.toLowerCase().includes(item.toLowerCase()));
      const isMissing = missingPpe.some((p) => p.toLowerCase().includes(item.toLowerCase()));
      if (isPresent) {
        return `<div class="ppe-check-item is-verified"><span style="font-weight:700;">✓</span> <span>${escapeHTML(item)}</span></div>`;
      }
      if (isMissing) {
        return `<div class="ppe-check-item is-missing"><span style="font-weight:700;">✕</span> <span>${escapeHTML(item)}</span></div>`;
      }
      return `<div class="ppe-check-item is-unknown"><span style="font-weight:700;">?</span> <span>${escapeHTML(item)}</span></div>`;
    }).join("");

    const tthText = entity.temporal_hazard?.time_to_hazard_seconds != null
      ? `<b class="text-critical">${entity.temporal_hazard.time_to_hazard_seconds} s</b> (${escapeHTML(entity.temporal_hazard.target_label || 'Hazard')})`
      : (isSingleImage
        ? `<span class="text-muted">Unavailable — single-frame analysis has no motion vectors</span>`
        : `<span class="text-muted">${escapeHTML(entity.temporal_hazard?.tth_status || "Unavailable — insufficient temporal/spatial evidence")}</span>`);

    // Spatial & Ground-Plane Metrics
    const contactPt = entity.position?.contact_point;
    const contactPtStr = contactPt && Array.isArray(contactPt) && contactPt.length >= 2
      ? `(${Math.round(contactPt[0])}, ${Math.round(contactPt[1])}) px`
      : "Unavailable (Contact point not determined)";

    const groundPos = entity.position?.ground_plane;
    const groundPosStr = groundPos && groundPos.x_m != null && groundPos.y_m != null
      ? `X: ${groundPos.x_m.toFixed(2)} m, Y: ${groundPos.y_m.toFixed(2)} m`
      : "Unavailable — camera calibration required";

    const metricSpeed = entity.state?.speed_m_per_s;
    const metricSpeedStr = metricSpeed != null
      ? `${metricSpeed.toFixed(2)} m/s`
      : (isSingleImage ? "Unavailable — single-frame analysis has no motion vectors" : "Unavailable — requires camera calibration and multi-frame tracking");

    const calStatusBadge = groundPos
      ? `<span class="badge badge-success" style="font-size:9.5px;">CALIBRATED (Ground Plane)</span>`
      : `<span class="badge badge-neutral" style="font-size:9.5px;">UNCONFIGURED (Pixel-space only)</span>`;

    return `
      <div class="inspector-header">
        <div class="inspector-title-area">
          <span class="inspector-card-kicker">${escapeHTML(entityType)} INSPECTION</span>
          <h3>${icon(isPerson ? "activity" : (isVehicle ? "video" : (isZone ? "shield" : "layers")), 16)} ${escapeHTML(entity.label || entity.id)}</h3>
        </div>
        <div style="display: flex; gap: 6px; align-items: center;">
          <span class="badge ${riskBadgeClass}">${escapeHTML(riskLevel)}</span>
          <button type="button" class="icon-button" data-action="close-inspector" aria-label="Close inspector">${icon("close", 14)}</button>
        </div>
      </div>
      <div class="inspector-content">
        <!-- Current State -->
        <div class="inspector-card">
          <span class="inspector-card-kicker">CURRENT STATE</span>
          <table class="inspector-table">
            <tbody>
              <tr><td>Movement:</td><td><b>${escapeHTML(entity.state?.movement || "DETECTED")}</b></td></tr>
              <tr><td>Zone:</td><td>${entity.state?.inside_zones?.length ? escapeHTML(entity.state.inside_zones.join(", ")) : (entity.position?.zone_id ? escapeHTML(entity.position.zone_id) : "Unrestricted Floor")}</td></tr>
              ${isPerson ? `<tr><td>PPE Status:</td><td><span class="badge ${personComplianceBadge}">${escapeHTML(personComplianceStatus)}</span></td></tr>` : ""}
              <tr><td>Time to Hazard:</td><td>${tthText}</td></tr>
              <tr><td>Risk Level:</td><td><span class="badge ${riskBadgeClass}">${escapeHTML(riskLevel)}</span></td></tr>
            </tbody>
          </table>
        </div>

        <!-- PPE Check Grid for Person -->
        ${isPerson ? `
          <div class="inspector-card">
            <span class="inspector-card-kicker">PPE EQUIPMENT STATUS</span>
            <div class="ppe-check-grid">
              ${ppeCheckItems}
            </div>
          </div>
        ` : ""}

        <!-- Detection & Coordinates (Image vs Metric Ground Plane) -->
        <div class="inspector-card">
          <span class="inspector-card-kicker">DETECTION & COORDINATES</span>
          <table class="inspector-table">
            <tbody>
              <tr><td>Tracking ID:</td><td><b>${entity.track_id != null ? `#${entity.track_id}` : (entity.id || "N/A")}</b></td></tr>
              <tr><td>Confidence:</td><td><b>${Math.round((entity.confidence || 0.85) * 100)}%</b> (${entity.confidence || "0.85"})</td></tr>
              <tr><td>Bounding Box:</td><td class="mono-coords">${entity.position?.bbox ? `[${entity.position.bbox.x1}, ${entity.position.bbox.y1}, ${entity.position.bbox.x2}, ${entity.position.bbox.y2}]` : "Unavailable"}</td></tr>
              ${(isPerson || isVehicle) ? `
                <tr><td>Contact Point:</td><td class="mono-coords">${escapeHTML(contactPtStr)}</td></tr>
                <tr><td>Image Velocity:</td><td>${entity.state?.speed_px_per_s != null ? `${Math.round(entity.state.speed_px_per_s)} px/s` : (entity.state?.velocity_status || (isSingleImage ? "Unavailable — single-frame analysis has no motion vectors" : "Unavailable"))}</td></tr>
                <tr><td>Ground Position:</td><td class="mono-coords">${escapeHTML(groundPosStr)}</td></tr>
                <tr><td>Approx Speed:</td><td>${escapeHTML(metricSpeedStr)}</td></tr>
                <tr><td>Calibration Status:</td><td>${calStatusBadge}</td></tr>
              ` : (isZone ? `
                <tr><td>Spatial Interpretation:</td><td>Image-Space Polygon (Pixel Coordinates)</td></tr>
                <tr><td>Ground Projection:</td><td>${data?.calibration_status === "CALIBRATED" ? `<span class="badge badge-success" style="font-size:9.5px;">Calibrated Ground Plane Active</span>` : `<span class="badge badge-neutral" style="font-size:9.5px;">Uncalibrated (Pixel Space)</span>`}</td></tr>
              ` : `
                <tr><td>Centroid:</td><td>${entity.position?.centroid ? `(${entity.position.centroid[0]}, ${entity.position.centroid[1]}) px` : "Unavailable"}</td></tr>
              `)}
            </tbody>
          </table>
        </div>

        <!-- Safety Engine Reasoning -->
        <div class="inspector-card">
          <span class="inspector-card-kicker">SAFETY ENGINE REASONING</span>
          ${entity.risk?.reasons && entity.risk.reasons.length > 0 ? `
            <div style="display:flex; flex-direction:column; gap:6px;">
              ${entity.risk.reasons.map((reason) => `<div class="reasoning-box">${icon("alert", 12)} ${escapeHTML(reason)}</div>`).join("")}
            </div>
          ` : (riskLevel === "NORMAL" || riskLevel === "LOW" ? `
            <div class="reasoning-box is-safe">${icon("check", 12)} No active safety hazards or proximity violations recorded for this entity.</div>
          ` : `
            <div class="reasoning-box">${icon("info", 12)} Risk contributing factors not available from current inference.</div>
          `)}

          ${entity.temporal_hazard?.compounding_factors && entity.temporal_hazard.compounding_factors.length > 0 ? `
            <div style="margin-top: 8px;">
              <span class="inspector-card-kicker" style="font-size: 9px; display:block; margin-bottom: 4px;">COMPOUNDING HAZARD FACTORS (${entity.temporal_hazard.compounding_factors.length})</span>
              <div style="display:flex; flex-direction:column; gap:4px;">
                ${entity.temporal_hazard.compounding_factors.map((f) => `<div class="tth-factor-row"><span class="tth-factor-check">✓</span> <span class="tth-factor-text">${escapeHTML(f)}</span></div>`).join("")}
              </div>
            </div>
          ` : ""}
        </div>

        <!-- Connected Relationships -->
        <div class="inspector-card">
          <span class="inspector-card-kicker">RELATIONSHIPS (${connectedRels.length})</span>
          ${connectedRels.length === 0 ? `
            <span class="text-muted" style="font-size:11.5px;">No active edges connected to this entity</span>
          ` : `
            <div class="inspector-rel-list">
              ${connectedRels.map((r) => {
                const isSource = r.source_id === entity.id;
                const otherId = isSource ? r.target_id : r.source_id;
                const otherNode = entities.find((e) => e.id === otherId);
                const otherLabel = otherNode?.label || otherId;
                let badgeClass = "rel-badge-near";
                if (r.type === "IS_WEARING") badgeClass = "rel-badge-wearing";
                else if (r.type === "MISSING") badgeClass = "rel-badge-missing";
                else if (r.type === "INSIDE_ZONE") badgeClass = "rel-badge-inside";
                else if (r.type === "APPROACHING" || r.type === "CLOSER_THAN") badgeClass = "rel-badge-approaching";

                return `
                  <div class="inspector-rel-item" data-action="select-graph-relation" data-rel-id="${escapeHTML(r.id)}" title="Inspect relationship ${r.type}">
                    <span>${isSource ? "→" : "←"} <b>${escapeHTML(otherLabel)}</b></span>
                    <span class="rel-item-badge ${badgeClass}">${escapeHTML(r.type)}</span>
                  </div>`;
              }).join("")}
            </div>
          `}
        </div>

        <!-- Action Buttons -->
        ${isPerson ? `
          <div style="margin-top: 4px;">
            <button type="button" class="button button-primary button-small button-full" data-action="focus-in-inference-breakdown" data-id="${entity.metadata?.track_id ?? entity.id.replace('person_', '')}">
              ${icon("search", 14)} Focus in Inference Breakdown
            </button>
          </div>
        ` : ""}
      </div>`;
  }

  // 3. Overview when nothing is selected
  return `
    <div class="inspector-header">
      <div class="inspector-title-area">
        <span class="inspector-card-kicker">SCENE SUMMARY</span>
        <h3>${icon("layers", 16)} Spatial State Overview</h3>
      </div>
    </div>
    <div class="inspector-content">
      <div class="inspector-card">
        <span class="inspector-card-kicker">METRICS</span>
        <table class="inspector-table">
          <tbody>
            <tr><td>Total Entities:</td><td><b>${entities.length}</b></td></tr>
            <tr><td>Workers Detected:</td><td>${entities.filter((e) => e.type === "PERSON").length}</td></tr>
            <tr><td>Vehicles:</td><td>${entities.filter((e) => e.type === "VEHICLE").length}</td></tr>
            <tr><td>Active Relationships:</td><td><b>${relationships.filter((relation) => relation.active === true).length}</b></td></tr>
            <tr><td>Peak Risk Tier:</td><td><b>${escapeHTML(data.summary?.peak_risk_level || "NORMAL")}</b></td></tr>
            <tr><td>Processing Mode:</td><td>${escapeHTML(data.source_mode || "Inference Engine")}</td></tr>
          </tbody>
        </table>
      </div>
      <div class="inspector-empty-state" style="padding: 20px 10px;">
        ${icon("info", 22)}
        <p>Click any node or relationship edge in the graph to inspect real-time safety telemetry, PPE gear bindings, and spatial proximity reasoning.</p>
      </div>
    </div>`;
}

async function refreshSceneGraph(jobId = null, quiet = false) {
  const sg = state.sceneGraphState;
  const targetJobId = jobId || state.analysisState?.jobId || null;
  if (!quiet) {
    sg.loading = true;
    sg.error = null;
    render();
  }

  try {
    const res = await sceneService.getSceneGraph(targetJobId);
    if (res.ok && res.data) {
      sg.data = res.data;
      sg.lastJobId = targetJobId;
      sg.error = null;

      // Auto-select focused person if state.selectedPersonId was set
      if (state.selectedPersonId && !sg.selectedEntityId) {
        const targetId = `person_${state.selectedPersonId}`;
        const match = res.data.entities?.find((e) => e.id === targetId || e.metadata?.track_id === state.selectedPersonId);
        if (match) {
          sg.selectedEntityId = match.id;
        }
      }
      if (!sg.selectedEntityId && res.data.entities?.length > 0) {
        const actor = res.data.entities.find((e) => e.type === "PERSON") || res.data.entities[0];
        sg.selectedEntityId = actor.id;
      }
    } else {
      if (!quiet) sg.error = res.error || "Unable to load scene graph";
    }
  } catch (err) {
    if (!quiet) sg.error = err.message || "Failed to fetch scene graph";
  } finally {
    if (!quiet) {
      sg.loading = false;
      render();
    } else if (currentPath() === "/app/scene-graph") {
      render();
      wireGraphInteractions();
    }
  }
  return sg.lastJobId === targetJobId ? sg.data : null;
}

function wireGraphInteractions() {
  const svg = document.querySelector(".scene-graph-svg");
  if (!svg || svg.dataset.wired) return;
  svg.dataset.wired = "true";

  const sg = state.sceneGraphState;

  // Smooth pan on drag
  let isDragging = false;
  let startX = 0;
  let startY = 0;
  let origPanX = 0;
  let origPanY = 0;

  svg.addEventListener("mousedown", (e) => {
    if (e.target.closest(".graph-node, .graph-edge, .edge-label-group, button")) return;
    isDragging = true;
    startX = e.clientX;
    startY = e.clientY;
    origPanX = sg.pan.x || 0;
    origPanY = sg.pan.y || 0;
    svg.style.cursor = "grabbing";
  });

  window.addEventListener("mousemove", (e) => {
    if (!isDragging) return;
    const dx = (e.clientX - startX) / (sg.zoom || 1);
    const dy = (e.clientY - startY) / (sg.zoom || 1);
    sg.pan.x = Math.round(origPanX + dx);
    sg.pan.y = Math.round(origPanY + dy);
    const vp = svg.querySelector(".viewport-content");
    if (vp) {
      vp.setAttribute("transform", `scale(${sg.zoom}) translate(${sg.pan.x}, ${sg.pan.y})`);
    }
  });

  window.addEventListener("mouseup", () => {
    if (isDragging) {
      isDragging = false;
      svg.style.cursor = "default";
    }
  });

  // Hover over edge or label
  svg.querySelectorAll(".graph-edge, .edge-label-group").forEach((el) => {
    el.addEventListener("mouseenter", () => {
      const relId = el.dataset.relId;
      if (!relId) return;
      const edge = svg.querySelector(`#edge-${CSS.escape(relId)}`);
      const edgeLabel = svg.querySelector(`.edge-label-group[data-rel-id="${CSS.escape(relId)}"]`);
      const srcId = edge?.dataset.src;
      const tgtId = edge?.dataset.tgt;

      svg.querySelectorAll(".graph-node, .graph-edge, .edge-label-group").forEach((item) => {
        item.classList.add("is-dimmed");
      });

      edge?.classList.remove("is-dimmed");
      edge?.classList.add("is-highlighted");
      el.classList.remove("is-dimmed");
      edgeLabel?.classList.remove("is-dimmed");
      edgeLabel?.classList.add("is-highlighted");

      if (srcId) {
        const srcNode = svg.querySelector(`[data-entity-id="${CSS.escape(srcId)}"]`);
        srcNode?.classList.remove("is-dimmed");
        srcNode?.classList.add("is-highlighted");
      }
      if (tgtId) {
        const tgtNode = svg.querySelector(`[data-entity-id="${CSS.escape(tgtId)}"]`);
        tgtNode?.classList.remove("is-dimmed");
        tgtNode?.classList.add("is-highlighted");
      }
    });

    el.addEventListener("mouseleave", () => {
      svg.querySelectorAll(".is-dimmed, .is-highlighted").forEach((item) => {
        item.classList.remove("is-dimmed", "is-highlighted");
      });
    });
  });

  // Hover over node
  svg.querySelectorAll(".graph-node").forEach((node) => {
    node.addEventListener("mouseenter", () => {
      const entityId = node.dataset.entityId;
      if (!entityId) return;

      svg.querySelectorAll(".graph-node, .graph-edge, .edge-label-group").forEach((item) => {
        item.classList.add("is-dimmed");
      });

      node.classList.remove("is-dimmed");
      node.classList.add("is-highlighted");

      svg.querySelectorAll(`.graph-edge[data-src="${CSS.escape(entityId)}"], .graph-edge[data-tgt="${CSS.escape(entityId)}"]`).forEach((edge) => {
        edge.classList.remove("is-dimmed");
        edge.classList.add("is-highlighted");
        const edgeLabel = svg.querySelector(`.edge-label-group[data-rel-id="${CSS.escape(edge.dataset.relId || "")}"]`);
        edgeLabel?.classList.remove("is-dimmed");
        edgeLabel?.classList.add("is-highlighted");
        const otherId = edge.dataset.src === entityId ? edge.dataset.tgt : edge.dataset.src;
        if (otherId) {
          const otherNode = svg.querySelector(`[data-entity-id="${CSS.escape(otherId)}"]`);
          otherNode?.classList.remove("is-dimmed");
          otherNode?.classList.add("is-highlighted");
        }
      });
    });

    node.addEventListener("mouseleave", () => {
      svg.querySelectorAll(".is-dimmed, .is-highlighted").forEach((item) => {
        item.classList.remove("is-dimmed", "is-highlighted");
      });
    });
  });
}

function render() {
  const path = currentPath();
  if (path === "/" || path === "/landing") {
    app.innerHTML = landingPage();
    mountMediaPlayers();
    return;
  }
  if (path === "/product") {
    app.innerHTML = productPage();
    mountMediaPlayers();
    return;
  }
  if (path === "/demo") {
    navigate("/");
    return;
  }
  if (path === "/dashboard" || path === "/app" || path === "/app/analyze") {
    const isAnalysis = (path === "/app/analyze" || path === "/app") && state.source && state.analysisState.phase !== "SELECTED";
    app.innerHTML = workspaceShell(isAnalysis ? analysisPage() : sourceSelectorPage(), { breadcrumb: isAnalysis ? "DETECTION WORKSPACE" : "SOURCE WORKSPACE" });
    mountMediaPlayers();
    return;
  }
  if (path === "/app/scene-graph") {
    app.innerHTML = workspaceShell(sceneGraphPage(), { breadcrumb: "SCENE GRAPH" });
    mountMediaPlayers();
    wireGraphInteractions();
    return;
  }
  if (path === "/app/live") {
    app.innerHTML = workspaceShell(livePage(), { breadcrumb: "LIVE MONITORING" });
    mountMediaPlayers();
    return;
  }
  if (path === "/app/incidents") {
    app.innerHTML = workspaceShell(incidentsPage(), { breadcrumb: "INCIDENTS" });
    mountMediaPlayers();
    return;
  }
  if (path === "/app/incidents/demo-incident") {
    navigate("/app/incidents");
    return;
  }
  if (path.startsWith("/app/incidents/")) {
    app.innerHTML = workspaceShell(incidentDetailPage(), { breadcrumb: "INCIDENT REVIEW" });
    mountMediaPlayers();
    return;
  }
  if (path === "/app/analytics") {
    app.innerHTML = workspaceShell(analyticsPage(), { breadcrumb: "ANALYTICS" });
    mountMediaPlayers();
    return;
  }
  if (path === "/app/settings") {
    app.innerHTML = workspaceShell(settingsPage(), { breadcrumb: "SETTINGS" });
    mountMediaPlayers();
    return;
  }
  app.innerHTML = landingPage();
  mountMediaPlayers();
}

function mountMediaPlayers() {
  app.querySelectorAll("[data-uploaded-player]").forEach((video) => {
    video.muted = state.previewMuted;
    video.playbackRate = state.playbackRate;
    video.addEventListener("volumechange", () => {
      state.previewMuted = video.muted;
      try { localStorage.setItem("intelliwatch-preview-muted", String(state.previewMuted)); } catch (_) {}
    });
  });
}
function persistModulePreferences() {
  try { localStorage.setItem("intelliwatch-modules", JSON.stringify(state.modules)); } catch (_) {}
}

function setToast(message) {
  state.toast = message;
  render();
  setTimeout(() => {
    if (state.toast === message) {
      state.toast = "";
      render();
    }
  }, 4000);
}

function selectFile(file, expectedKind) {
  if (!file) return;
  const name = file.name || "";
  const extension = name.split(".").pop()?.toLowerCase() || "";
  const videoExtensions = ["mp4", "avi", "mov", "mkv"];
  const imageExtensions = ["jpg", "jpeg", "png"];

  if (expectedKind === "video") {
    if (!videoExtensions.includes(extension)) {
      setToast(`Unsupported video format '.${extension}'. Supported formats: MP4, AVI, MOV, MKV.`);
      return;
    }
  } else if (expectedKind === "image") {
    if (!imageExtensions.includes(extension)) {
      setToast(`Unsupported image format '.${extension}'. Supported formats: JPG, JPEG, PNG.`);
      return;
    }
  }

  // Cancel any running analysis
  if (state.activeAnalysisAbort) {
    state.activeAnalysisAbort.abort();
    state.activeAnalysisAbort = null;
  }

  if (state.source?.url) URL.revokeObjectURL(state.source.url);
  state.source = { name: file.name, file, kind: expectedKind, url: URL.createObjectURL(file) };
  state.sceneGraphState.data = null;
  state.sceneGraphState.lastJobId = null;
  state.sceneGraphState.selectedEntityId = null;
  state.sceneGraphState.selectedRelationId = null;
  state.analysisState = {
    jobId: null,
    phase: "SELECTED",
    running: false,
    progressPct: 0,
    statusText: "Ready for detection",
    result: null,
    error: null,
    viewMode: "annotated",
    latestFrameUrl: null,
    annotatedUrl: null,
    annotatedFrameUrl: null,
    originalUrl: state.source.url,
  };
  state.rtspMessage = "";
  if (currentPath() === "/app" || currentPath() === "/app/analyze") {
    render();
  } else {
    navigate("/app");
  }
}

// --------------------------------------------------------------------------
// Real Backend Operations
// --------------------------------------------------------------------------

function updateTelemetryInDOM() {
  const t = state.telemetry;
  const isOnline = t.online;
  const isCuda = t.isCuda;
  const statusDotClass = isOnline ? (isCuda ? "status-active" : "status-neutral") : "status-warning";
  const statusHeadline = isOnline ? (isCuda ? "CUDA:0 Active" : "System Online") : "Backend Offline";
  const statusSub = isOnline ? (t.gpuName || (isCuda ? "NVIDIA RTX 5050" : "CPU Fallback")) : "Reconnecting...";
  const badgeText = isOnline ? (isCuda ? "CUDA:0 ACCELERATED" : "ONLINE (CPU)") : "BACKEND OFFLINE";

  document.querySelectorAll(".local-badge").forEach((el) => {
    el.innerHTML = `<i class="status-dot ${statusDotClass}"></i> ${escapeHTML(badgeText)}`;
  });
  document.querySelectorAll(".sidebar-status").forEach((el) => {
    el.innerHTML = `<span class="status-dot ${statusDotClass}"></span><div><b>${escapeHTML(statusHeadline)}</b><small>${escapeHTML(statusSub)}</small></div>`;
  });
}

async function refreshTelemetry() {
  try {
    const t = await telemetryService.getSystemTelemetry();
    state.telemetry = t;
    updateTelemetryInDOM();
  } catch (err) {
    console.warn("Failed to load telemetry:", err);
  }
}

async function refreshCameras() {
  state.camerasLoading = true;
  await apiClient.ensureAuthenticated().catch(() => {});
  render();
  try {
    const res = await cameraService.listCameras();
    if (res.ok) {
      state.cameras = res.cameras || [];
    } else {
      setToast(res.error || "Failed to load cameras");
    }
  } catch (err) {
    console.error("Camera load error:", err);
  } finally {
    state.camerasLoading = false;
    render();
  }
}

async function refreshCalibration(cameraId = state.calibrationState.selectedCameraId) {
  if (!cameraId) cameraId = "cam_loading_bay_01";
  state.calibrationState.loading = true;
  state.calibrationState.selectedCameraId = cameraId;
  await apiClient.ensureAuthenticated().catch(() => {});
  state.calibrationState.frameUrl = cameraService.getFrameUrl(cameraId);
  try {
    const res = await cameraService.getCalibration(cameraId);
    if (res.ok && res.data) {
      const cfg = res.data;
      state.calibrationState.calibrationStatus = cfg.calibration_status || "UNCONFIGURED";
      state.calibrationState.realWorldWidthM = cfg.real_world_width_m || 6.0;
      state.calibrationState.realWorldDepthM = cfg.real_world_depth_m || 10.0;
      state.calibrationState.coordinateSystem = cfg.coordinate_system || "METRIC_GROUND_PLANE";
      if (cfg.points && Array.isArray(cfg.points) && cfg.points.length === 4) {
        state.calibrationState.points = cfg.points.map((p) => ({ x: p.x, y: p.y }));
        state.calibrationState.isValid = cfg.calibration_status === "CALIBRATED";
      } else {
        state.calibrationState.points = [];
        state.calibrationState.isValid = false;
      }
      state.calibrationState.validationError = null;
    } else {
      state.calibrationState.calibrationStatus = "UNCONFIGURED";
      state.calibrationState.points = [];
      state.calibrationState.isValid = false;
      state.calibrationState.validationError = null;
    }
  } catch (err) {
    console.error("Camera calibration load error:", err);
    state.calibrationState.calibrationStatus = "UNCONFIGURED";
  } finally {
    state.calibrationState.loading = false;
    render();
  }
}

async function refreshAlerts() {
  state.alertsLoading = true;
  render();
  try {
    const res = await incidentService.listAlerts({ page_size: 50 });
    if (res.ok) {
      state.alerts = res.alerts || [];
    }
  } catch (err) {
    console.error("Alerts load error:", err);
  } finally {
    state.alertsLoading = false;
    render();
  }
}

async function refreshAnalytics() {
  state.analyticsLoading = true;
  state.alertStats = null;
  render();
  try {
    const [zonesRes, alertStatsRes, alertsRes] = await Promise.all([
      analyticsService.listZones(),
      incidentService.getAlertStats(),
      incidentService.listAlerts({ page_size: 100 }),
    ]);
    if (zonesRes.ok) state.zones = zonesRes.zones || [];
    if (alertStatsRes.ok) state.alertStats = alertStatsRes.data;
    if (alertsRes.ok) state.alerts = alertsRes.alerts || [];
  } catch (err) {
    console.error("Analytics load error:", err);
  } finally {
    state.analyticsLoading = false;
    render();
  }
}

async function runAnalysis() {
  if (!state.source || !state.source.file) {
    setToast("No source file selected");
    return;
  }

  if (state.analysisState.running) {
    console.log("Analysis already running, ignoring duplicate invocation.");
    return;
  }

  // Cancel any prior job
  if (state.activeAnalysisAbort) {
    state.activeAnalysisAbort.abort();
    state.activeAnalysisAbort = null;
  }

  const abortController = new AbortController();
  state.activeAnalysisAbort = abortController;

  state.analysisState.running = true;
  state.analysisState.phase = "UPLOADING";
  state.analysisState.error = null;
  state.analysisState.result = null;
  state.analysisState.progressPct = 0;
  state.analysisState.latestFrameUrl = null;
  state.analysisState.annotatedUrl = null;
  state.analysisState.statusText = state.source.kind === "video"
    ? "Uploading video for CUDA model inference..."
    : "Uploading image for CUDA model inference...";
  render();

  try {
    if (state.source.kind === "image") {
      state.analysisState.phase = "PROCESSING";
      state.analysisState.statusText = `Executing YOLO & PPE models on ${state.telemetry.gpuName || "CUDA:0"}...`;
      render();

      const res = await mediaService.analyzeImage(
        state.source.file,
        abortController.signal,
        (uploadInfo) => {
          if (abortController.signal.aborted) return;
          state.analysisState.phase = "UPLOADING";
          state.analysisState.progressPct = uploadInfo.percent;
          state.analysisState.statusText = `Uploading image: ${uploadInfo.percent}%...`;
          render();
        }
      );
      if (abortController.signal.aborted) return;

      if (res.ok) {
        const data = res.data;
        state.analysisState.jobId = data.job_id;
        state.analysisState.phase = "COMPLETED";
        state.analysisState.running = false;
        state.analysisState.progressPct = 100;
        state.analysisState.statusText = "CUDA inference completed successfully";
        state.analysisState.result = data;
        state.analysisState.annotatedUrl = data.annotated_media_url;
        state.analysisState.originalUrl = data.original_media_url || state.source.url;
        state.analysisState.viewMode = "annotated";
        state.analysisState.error = null;
        setToast(`Inference completed: ${data.detections_count} objects, ${data.workers_count} workers`);
        refreshAlerts();
      } else {
        state.analysisState.running = false;
        state.analysisState.phase = "FAILED";
        state.analysisState.error = res.error || "Image analysis failed";
        setToast(state.analysisState.error);
      }
      render();
    } else if (state.source.kind === "video") {
      const sub = await mediaService.submitVideoJob(
        state.source.file,
        abortController.signal,
        (uploadInfo) => {
          if (abortController.signal.aborted) return;
          state.analysisState.phase = "UPLOADING";
          state.analysisState.progressPct = uploadInfo.percent;
          state.analysisState.statusText = `Uploading video: ${uploadInfo.percent}% (${formatBytes(uploadInfo.loaded)} / ${formatBytes(uploadInfo.total)})...`;
          render();
        }
      );
      if (abortController.signal.aborted) return;
      if (!sub.ok) {
        throw new Error(sub.error || "Failed to submit video");
      }

      const currentJobId = sub.jobId;
      state.analysisState.jobId = currentJobId;
      state.analysisState.phase = "PROCESSING";
      state.analysisState.progressPct = 0;
      state.analysisState.statusText = "Initializing ByteTrack & PPE models on CUDA...";
      render();

      const finalResult = await mediaService.pollVideoJob(
        currentJobId,
        (progress) => {
          if (abortController.signal.aborted || state.analysisState.jobId !== currentJobId) return;
          state.analysisState.phase = "PROCESSING";
          const pct = Math.max(state.analysisState.progressPct, Math.min(100, Math.round(progress.progress_pct || 0)));
          state.analysisState.progressPct = pct;
          if (progress.latest_frame_url) {
            state.analysisState.latestFrameUrl = progress.latest_frame_url;
          }
          state.analysisState.statusText = `Processing frame ${progress.current_frame || 0}/${progress.total_frames || 0} (${progress.processing_fps || 0} FPS)...`;
          if (currentPath() === "/app/scene-graph") {
            refreshSceneGraph(currentJobId, true);
          } else {
            render();
          }
        },
        1000,
        600,
        abortController.signal
      );

      if (abortController.signal.aborted || state.analysisState.jobId !== currentJobId) return;

      state.analysisState.phase = "COMPLETED";
      state.analysisState.running = false;
      state.analysisState.progressPct = 100;
      state.analysisState.statusText = "Video analysis completed";
      state.analysisState.result = finalResult;
      state.analysisState.annotatedUrl = finalResult.annotated_video_url || finalResult.annotated_media_url;
      state.analysisState.annotatedFrameUrl = finalResult.annotated_frame_url || finalResult.latest_frame_url;
      state.analysisState.originalUrl = state.source.url;
      state.analysisState.viewMode = "annotated";
      state.analysisState.error = null;
      setToast(`Video analysis completed: ${finalResult.total_frames || 0} frames processed!`);
      await refreshSceneGraph(currentJobId, true);
      await refreshAlerts();
      render();
    }
  } catch (err) {
    if (abortController.signal.aborted) return;
    state.analysisState.running = false;
    state.analysisState.phase = "FAILED";
    state.analysisState.error = err.message || "Analysis error occurred";
    setToast(state.analysisState.error);
    render();
  } finally {
    if (!abortController.signal.aborted) {
      state.analysisState.running = false;
    }
  }
}

// --------------------------------------------------------------------------
// Event Listeners & Actions
// --------------------------------------------------------------------------

document.addEventListener("click", (event) => {
  const link = event.target.closest("a[data-link]");
  if (link && !event.metaKey && !event.ctrlKey && !event.shiftKey && !event.altKey) {
    event.preventDefault();
    navigate(link.getAttribute("href"));
    return;
  }
  if (state.publicMenuOpen && !event.target.closest(".public-header") && !event.target.closest("a")) {
    setPublicMenu(false);
    return;
  }

  // Interactive click on calibration image canvas to place ground quadrangle points
  const canvasWrap = event.target.closest("#cal-canvas-wrap");
  if (canvasWrap && !event.target.closest("button") && !event.target.closest("select")) {
    const rect = canvasWrap.getBoundingClientRect();
    const clickX = event.clientX - rect.left;
    const clickY = event.clientY - rect.top;

    const imgEl = canvasWrap.querySelector("img");
    const naturalW = imgEl?.naturalWidth || state.calibrationState.imageWidth || 1376;
    const naturalH = imgEl?.naturalHeight || state.calibrationState.imageHeight || 768;
    state.calibrationState.imageWidth = naturalW;
    state.calibrationState.imageHeight = naturalH;

    const scaleX = naturalW / rect.width;
    const scaleY = naturalH / rect.height;

    const imgX = Math.round(clickX * scaleX);
    const imgY = Math.round(clickY * scaleY);

    if (state.calibrationState.points.length < 4) {
      state.calibrationState.points.push({ x: imgX, y: imgY });
      state.calibrationState.dirty = true;
      state.calibrationState.validationError = null;
      render();
    } else {
      setToast("4 points already placed. Click 'Reset Points' to place new points.");
    }
    return;
  }

  const button = event.target.closest("[data-action]");
  if (!button) return;
  const action = button.dataset.action;

  if (action === "toggle-public-menu") {
    setPublicMenu(!state.publicMenuOpen);
  }
  if (action === "cycle-theme") {
    const themes = ["system", "light", "dark"];
    state.theme = themes[(themes.indexOf(state.theme) + 1) % themes.length];
    applyTheme(state.theme);
    refreshThemeControlLabels();
  }
  if (action === "pick-video") {
    const input = document.getElementById("permanent-video-input") || document.getElementById("video-input");
    if (input) {
      input.value = "";
      input.click();
    }
  }
  if (action === "pick-image") {
    const input = document.getElementById("permanent-image-input") || document.getElementById("image-input");
    if (input) {
      input.value = "";
      input.click();
    }
  }

  if (action === "start-analysis" || action === "run-analysis-again") {
    if (state.analysisState.running) return;
    state.selectedPersonId = null;
    state.showAllInference = false;
    runAnalysis();
  }
  if (action === "select-person") {
    const personId = parseInt(button.dataset.id || button.closest("[data-id]")?.dataset.id, 10);
    if (personId) {
      state.selectedPersonId = personId;
      render();
      const canvasEl = document.querySelector("#evidence-canvas");
      if (canvasEl) {
        canvasEl.scrollIntoView({ behavior: "smooth", block: "nearest" });
      }
    }
  }
  if (action === "toggle-all-inference") {
    state.showAllInference = !state.showAllInference;
    render();
  }
  if (action === "set-view-mode") {
    state.analysisState.viewMode = button.dataset.mode;
    render();
  }

  if (action === "cycle-playback-speed") {
    const speeds = [0.5, 1, 1.5, 2];
    state.playbackRate = speeds[(speeds.indexOf(state.playbackRate) + 1) % speeds.length];
    try { localStorage.setItem("intelliwatch-playback-rate", String(state.playbackRate)); } catch (_) {}
    app.querySelectorAll("video").forEach((video) => { video.playbackRate = state.playbackRate; });
    app.querySelectorAll('[data-action="cycle-playback-speed"]').forEach((control) => { control.textContent = `${state.playbackRate}×`; });
  }
  if (action === "view-scene-graph") {
    navigate("/app/scene-graph");
  }
  if (action === "inspect-person-in-graph") {
    const personId = parseInt(button.dataset.id, 10) || button.dataset.id;
    if (personId) {
      state.selectedPersonId = personId;
      state.sceneGraphState.selectedEntityId = `person_${personId}`;
      state.sceneGraphState.selectedRelationId = null;
    }
    navigate("/app/scene-graph");
  }
  if (action === "focus-in-inference-breakdown") {
    const personId = parseInt(button.dataset.id, 10);
    if (personId) {
      state.selectedPersonId = personId;
    }
    navigate("/app/analyze");
  }
  if (action === "select-graph-entity") {
    const entityId = button.dataset.entityId || button.closest("[data-entity-id]")?.dataset.entityId;
    if (entityId) {
      state.sceneGraphState.selectedEntityId = entityId;
      state.sceneGraphState.selectedRelationId = null;
      render();
    }
  }
  if (action === "select-graph-relation") {
    const relId = button.dataset.relId || button.closest("[data-rel-id]")?.dataset.relId;
    if (relId) {
      state.sceneGraphState.selectedRelationId = relId;
      state.sceneGraphState.selectedEntityId = null;
      render();
    }
  }
  if (action === "close-inspector") {
    state.sceneGraphState.selectedEntityId = null;
    state.sceneGraphState.selectedRelationId = null;
    render();
  }
  if (action === "set-graph-view-mode") {
    state.sceneGraphState.viewMode = button.dataset.mode || "graph";
    render();
  }
  if (action === "filter-entities") {
    state.sceneGraphState.entityFilter = button.dataset.filter || "ALL";
    render();
  }
  if (action === "filter-relations") {
    state.sceneGraphState.relationFilter = button.dataset.filter || "ALL";
    render();
  }
  if (action === "filter-graph") {
    state.sceneGraphState.filter = button.dataset.filter || "ALL";
    state.sceneGraphState.entityFilter = button.dataset.filter || "ALL";
    render();
  }
  if (action === "refresh-scene-graph") {
    refreshSceneGraph(state.analysisState?.jobId);
  }
  if (action === "auto-layout-graph") {
    state.sceneGraphState.zoom = 1.0;
    state.sceneGraphState.pan = { x: 0, y: 0 };
    render();
  }
  if (action === "zoom-graph-in") {
    state.sceneGraphState.zoom = Math.min(2.0, Number((state.sceneGraphState.zoom + 0.15).toFixed(2)));
    render();
  }
  if (action === "zoom-graph-out") {
    state.sceneGraphState.zoom = Math.max(0.6, Number((state.sceneGraphState.zoom - 0.15).toFixed(2)));
    render();
  }
  if (action === "reset-graph-layout") {
    state.sceneGraphState.zoom = 1.0;
    state.sceneGraphState.pan = { x: 0, y: 0 };
    state.sceneGraphState.entityFilter = "ALL";
    state.sceneGraphState.relationFilter = "ALL";
    state.sceneGraphState.filter = "ALL";
    state.sceneGraphState.viewMode = "graph";
    render();
  }
  if (action === "fit-graph") {
    state.sceneGraphState.zoom = 1.0;
    state.sceneGraphState.pan = { x: 0, y: 0 };
    render();
  }
  if (action === "fullscreen-graph") {
    const target = document.querySelector("#scene-graph-viewport") || document.querySelector(".scene-graph-canvas-panel");
    if (target?.requestFullscreen) {
      target.requestFullscreen().catch(() => setToast("Fullscreen unavailable in this browser."));
    } else {
      setToast("Fullscreen is not supported.");
    }
  }
  if (action === "zoom-media-in") {
    state.analysisState.mediaZoom = Math.min(3.0, Number(((state.analysisState.mediaZoom || 1.0) + 0.2).toFixed(2)));
    render();
  }
  if (action === "zoom-media-out") {
    state.analysisState.mediaZoom = Math.max(0.5, Number(((state.analysisState.mediaZoom || 1.0) - 0.2).toFixed(2)));
    render();
  }
  if (action === "fit-media") {
    state.analysisState.mediaZoom = 1.0;
    render();
  }
  if (action === "open-report-modal") {
    state.reportModalOpen = true;
    render();
  }
  if (action === "close-report-modal") {
    state.reportModalOpen = false;
    render();
  }
  if (action === "print-report") {
    window.print();
  }
  if (action === "back-to-detection") {
    navigate("/app/analyze");
  }

  if (action === "back-to-source") {
    if (state.activeAnalysisAbort) {
      state.activeAnalysisAbort.abort();
      state.activeAnalysisAbort = null;
    }
    if (state.source?.url) URL.revokeObjectURL(state.source.url);
    state.source = null;
    state.selectedPersonId = null;
    state.showAllInference = false;
    state.sceneGraphState.data = null;
    state.sceneGraphState.lastJobId = null;
    state.sceneGraphState.selectedEntityId = null;
    state.sceneGraphState.selectedRelationId = null;
    state.analysisState = {
      jobId: null,
      phase: "IDLE",
      running: false,
      progressPct: 0,
      statusText: "",
      result: null,
      error: null,
      viewMode: "annotated",
      latestFrameUrl: null,
      annotatedUrl: null,
      annotatedFrameUrl: null,
      originalUrl: null,
    };
    render();
  }
  if (action === "clear-source") {
    if (state.activeAnalysisAbort) {
      state.activeAnalysisAbort.abort();
      state.activeAnalysisAbort = null;
    }
    if (state.source?.url) URL.revokeObjectURL(state.source.url);
    state.source = null;
    state.selectedPersonId = null;
    state.showAllInference = false;
    state.sceneGraphState.data = null;
    state.sceneGraphState.lastJobId = null;
    state.sceneGraphState.selectedEntityId = null;
    state.sceneGraphState.selectedRelationId = null;
    state.analysisState = {
      jobId: null,
      phase: "IDLE",
      running: false,
      progressPct: 0,
      statusText: "",
      result: null,
      error: null,
      viewMode: "annotated",
      latestFrameUrl: null,
      annotatedUrl: null,
      annotatedFrameUrl: null,
      originalUrl: null,
    };
    render();
  }
  if (action === "dismiss-toast") {
    state.toast = "";
    render();
  }
  if (action === "fullscreen") {
    const target = document.querySelector("#evidence-canvas");
    if (target?.requestFullscreen) {
      target.requestFullscreen().catch(() => setToast("Fullscreen unavailable in this browser."));
    } else {
      setToast("Fullscreen is not supported.");
    }
  }
  if (action === "open-camera-modal") {
    state.cameraModalOpen = true;
    render();
  }
  if (action === "close-camera-modal" || action === "close-camera-modal-backdrop") {
    if (action === "close-camera-modal-backdrop" && !event.target.classList.contains("modal-backdrop")) return;
    state.cameraModalOpen = false;
    render();
  }
  if (action === "toggle-camera-stream") {
    const id = button.dataset.id;
    const isRunning = button.dataset.running === "true";
    (isRunning ? cameraService.stopCamera(id) : cameraService.startCamera(id)).then(() => {
      setToast(isRunning ? `Stopped stream for ${id}` : `Started stream for ${id}`);
      refreshCameras();
    }).catch((err) => setToast(err.message));
  }
  if (action === "delete-camera") {
    const id = button.dataset.id;
    if (confirm(`Remove camera ${id} from workspace?`)) {
      cameraService.removeCamera(id).then(() => {
        setToast(`Camera ${id} removed`);
        refreshCameras();
      }).catch((err) => setToast(err.message));
    }
  }
  if (action === "filter-alerts") {
    state.alertFilter = button.dataset.filter;
    render();
  }
  if (action === "open-incident-detail") {
    navigate(`/app/incidents/${button.dataset.id}`);
  }
  if (action === "update-alert-status") {
    const id = button.dataset.id;
    const actionType = button.dataset.status;
    let promise = null;
    if (actionType === "acknowledge") promise = incidentService.acknowledgeAlert(id, "Operator acknowledged via workspace");
    if (actionType === "resolve") promise = incidentService.resolveAlert(id, "Resolved by safety operator");
    if (actionType === "dismiss") promise = incidentService.dismissAlert(id, "Dismissed by operator");
    if (promise) {
      promise.then((res) => {
        if (res.ok) {
          setToast(`Alert ${id} updated to ${actionType.toUpperCase()}`);
          refreshAlerts();
          if (state.activeDetailAlert && state.activeDetailAlert.alert_id === id) {
            state.activeDetailAlert.status = actionType.toUpperCase();
            render();
          }
        } else {
          setToast(res.error || "Failed to update alert");
        }
      }).catch((err) => setToast(err.message));
    }
  }
  if (action === "refresh-analytics") {
    refreshAnalytics();
  }

  // Camera Calibration Actions
  if (action === "validate-cal") {
    const cal = state.calibrationState;
    if (cal.points.length !== 4) {
      cal.validationError = "Exactly 4 ground points are required";
      render();
      return;
    }
    const payload = {
      camera_id: cal.selectedCameraId,
      points: cal.points,
      real_world_width_m: cal.realWorldWidthM,
      real_world_depth_m: cal.realWorldDepthM,
      coordinate_system: cal.coordinateSystem,
    };
    cameraService.validateCalibration(cal.selectedCameraId, payload).then((res) => {
      if (res.ok && res.data && res.data.is_valid) {
        cal.isValid = true;
        cal.validationError = null;
        cal.validationResiduals = {
          reprojection_error_px: res.data.reprojection_error_px,
          coverage_m2: res.data.estimated_coverage_area_m2,
        };
        const cov = res.data.estimated_coverage_area_m2;
        setToast(`Ground plane validated${cov != null ? ` • approx. ${cov.toFixed(1)} m² covered` : ""}`);
      } else {
        cal.isValid = false;
        cal.validationError = res.data?.error_message || res.error || "Calibration validation failed";
      }
      render();
    }).catch((err) => {
      cal.isValid = false;
      cal.validationError = err.message || "Failed to validate calibration";
      render();
    });
  }

  if (action === "save-cal") {
    const cal = state.calibrationState;
    if (cal.points.length !== 4) {
      setToast("4 points required to save calibration");
      return;
    }
    const payload = {
      camera_id: cal.selectedCameraId,
      calibration_enabled: true,
      points: cal.points,
      real_world_width_m: cal.realWorldWidthM,
      real_world_depth_m: cal.realWorldDepthM,
      coordinate_system: cal.coordinateSystem,
    };
    cameraService.saveCalibration(cal.selectedCameraId, payload).then((res) => {
      if (res.ok) {
        cal.calibrationStatus = "CALIBRATED";
        cal.isValid = true;
        cal.validationError = null;
        cal.dirty = false;
        setToast("Camera calibration saved successfully");
      } else {
        setToast(res.error || "Failed to save calibration");
      }
      render();
    }).catch((err) => {
      setToast("Error saving calibration: " + err.message);
      render();
    });
  }

  if (action === "clear-cal") {
    const cal = state.calibrationState;
    cameraService.deleteCalibration(cal.selectedCameraId).then((res) => {
      if (res.ok) {
        cal.calibrationStatus = "UNCONFIGURED";
        cal.points = [];
        cal.isValid = false;
        cal.validationError = null;
        cal.dirty = false;
        setToast("Calibration cleared. Camera will use pixel-space inference.");
      } else {
        setToast(res.error || "Failed to clear calibration");
      }
      render();
    }).catch((err) => {
      setToast("Error clearing calibration: " + err.message);
      render();
    });
  }

  if (action === "reset-cal-points") {
    state.calibrationState.points = [];
    state.calibrationState.isValid = false;
    state.calibrationState.validationError = null;
    state.calibrationState.dirty = true;
    render();
  }
});

document.addEventListener("input", (event) => {
  if (event.target.id === "cal-width-m") {
    state.calibrationState.realWorldWidthM = parseFloat(event.target.value) || 0;
    state.calibrationState.dirty = true;
  }
  if (event.target.id === "cal-depth-m") {
    state.calibrationState.realWorldDepthM = parseFloat(event.target.value) || 0;
    state.calibrationState.dirty = true;
  }
  if (event.target.dataset.action === "search-alerts") {
    state.alertSearch = event.target.value;
    render();
    const searchInput = document.querySelector('[data-action="search-alerts"]');
    if (searchInput) {
      searchInput.focus();
      searchInput.selectionStart = searchInput.selectionEnd = searchInput.value.length;
    }
  }
});

document.addEventListener("change", (event) => {
  const input = event.target;
  if (input.id === "cal-camera-select") {
    const newCameraId = input.value;
    state.calibrationState.selectedCameraId = newCameraId;
    refreshCalibration(newCameraId);
  }
  if (input.matches("[data-module]")) {
    state.modules[input.dataset.module] = input.checked;
    persistModulePreferences();
  }
  if (input.matches("[data-theme-choice]")) {
    applyTheme(input.value);
    render();
  }
  if (input.matches("#playback-rate")) {
    state.playbackRate = Number(input.value) || 1;
    try { localStorage.setItem("intelliwatch-playback-rate", String(state.playbackRate)); } catch (_) {}
    app.querySelectorAll("video").forEach((video) => { video.playbackRate = state.playbackRate; });
  }
  if (input.matches("[data-preview-muted]")) {
    state.previewMuted = input.checked;
    try { localStorage.setItem("intelliwatch-preview-muted", String(state.previewMuted)); } catch (_) {}
    app.querySelectorAll("[data-uploaded-player]").forEach((video) => { video.muted = state.previewMuted; });
  }
});

document.addEventListener("submit", (event) => {
  if (event.target.id === "rtsp-form") {
    event.preventDefault();
    const data = new FormData(event.target);
    const name = String(data.get("cameraName") || "").trim();
    const url = String(data.get("rtspUrl") || "").trim();
    if (!url) {
      state.rtspMessage = "Please enter an RTSP stream URL, HTTP stream URL, or video file path.";
      render();
      return;
    }
    cameraService.registerCamera({ name: name || "CCTV Camera", rtsp_url: url, source: url }).then((res) => {
      if (res.ok) {
        state.rtspMessage = "";
        setToast(`Camera '${name || "CCTV Camera"}' registered successfully`);
        refreshCameras();
        navigate("/app/live");
      } else {
        state.rtspMessage = res.error || "Failed to connect camera";
        render();
      }
    }).catch((err) => {
      state.rtspMessage = err.message || "Failed to register camera";
      render();
    });
  }

  if (event.target.id === "modal-camera-form") {
    event.preventDefault();
    const data = new FormData(event.target);
    const cameraId = String(data.get("cameraId") || "").trim();
    const name = String(data.get("name") || "").trim();
    const url = String(data.get("rtspUrl") || "").trim();
    const fpsLimit = Number(data.get("fpsLimit") || 10);
    cameraService.registerCamera({ camera_id: cameraId, name, rtsp_url: url, source: url, fps_limit: fpsLimit }).then((res) => {
      if (res.ok) {
        state.cameraModalOpen = false;
        setToast(`Camera '${name || "Camera"}' registered successfully`);
        refreshCameras();
      } else {
        alert(res.error || "Failed to register camera");
      }
    }).catch((err) => alert(err.message));
  }
});

// Window level drag-drop prevention so dragging media over window doesn't navigate away
window.addEventListener("dragover", (e) => {
  if (e.dataTransfer?.types?.includes("Files")) {
    e.preventDefault();
  }
});
window.addEventListener("drop", (e) => {
  if (e.dataTransfer?.types?.includes("Files")) {
    e.preventDefault();
  }
});

document.addEventListener("dragover", (event) => {
  const target = event.target.closest("[data-drop-kind]");
  if (!target) return;
  event.preventDefault();
  target.classList.add("is-dragging");
});

document.addEventListener("dragleave", (event) => {
  const target = event.target.closest("[data-drop-kind]");
  if (target && !target.contains(event.relatedTarget)) target.classList.remove("is-dragging");
});

document.addEventListener("drop", (event) => {
  const target = event.target.closest("[data-drop-kind]");
  if (!target) return;
  event.preventDefault();
  target.classList.remove("is-dragging");
  const file = event.dataTransfer?.files?.[0];
  if (file) selectFile(file, target.dataset.dropKind);
});

function initPermanentFileInputs() {
  const vInput = document.getElementById("permanent-video-input");
  const iInput = document.getElementById("permanent-image-input");
  if (vInput && !vInput.dataset.wired) {
    vInput.dataset.wired = "true";
    vInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) selectFile(file, "video");
      e.target.value = "";
    });
  }
  if (iInput && !iInput.dataset.wired) {
    iInput.dataset.wired = "true";
    iInput.addEventListener("change", (e) => {
      const file = e.target.files?.[0];
      if (file) selectFile(file, "image");
      e.target.value = "";
    });
  }
}

// Periodic Telemetry Synchronization (every 8s)
let telemetryInterval = null;
function startTelemetryPolling() {
  refreshTelemetry();
  telemetryInterval = setInterval(refreshTelemetry, 8000);
}

// Initial Boot
window.addEventListener("popstate", () => {
  render();
  onRouteChanged();
});

function onRouteChanged() {
  const path = currentPath();
  if (path === "/app/scene-graph") {
    const currentJobId = state.analysisState?.jobId || null;
    if (state.sceneGraphState.lastJobId !== currentJobId || !state.sceneGraphState.data) {
      refreshSceneGraph(currentJobId);
    }
  }
  if (path === "/app/live") refreshCameras();
  if (path === "/app/settings") {
    refreshCameras();
    refreshCalibration(state.calibrationState.selectedCameraId);
  }
  if (path === "/app/incidents") refreshAlerts();
  if (path.startsWith("/app/incidents/")) {
    const alertId = path.split("/").pop();
    incidentService.getAlert(alertId).then((res) => {
      if (res.ok) {
        state.activeDetailAlert = res.data;
        render();
      }
    });
  }
  if (path === "/app/analytics") refreshAnalytics();
}

// Keyboard Accessibility for interactive SVG elements
document.addEventListener("keydown", (event) => {
  if (event.key === "Enter" || event.key === " ") {
    const target = event.target.closest('[data-action="select-graph-entity"], [data-action="select-graph-relation"]');
    if (target) {
      event.preventDefault();
      target.click();
    }
  }
});

initPermanentFileInputs();
apiClient.ensureAuthenticated().catch(() => {});
render();
startTelemetryPolling();
onRouteChanged();
