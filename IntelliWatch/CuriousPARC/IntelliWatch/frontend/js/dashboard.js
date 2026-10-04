/**
 * IntelliWatch Operator Dashboard Logic
 * Connects to FastAPI backend endpoints via polling to provide real-time
 * industrial safety telemetry, scene understanding graphs, PPE compliance,
 * risk reasoning, early-warning alerts, and evidence inspection.
 */

// Configuration & Runtime State
const CONFIG = {
  apiBase: "",
  pollIntervalMs: 1000,
  maxCanvasWidth: 1280,
  maxCanvasHeight: 720,
};

let state = {
  isPolling: true,
  pollTimer: null,
  currentAssessment: null,
  incidents: [],
  systemStatus: null,
  activeTab: "scene",
  selectedIncident: null,

  // Step 23: Authentication & RBAC State
  currentUser: null,
  authToken: localStorage.getItem("intelliwatch_token") || "",
  users: [],
  auditLogs: [],
  auditFilters: {
    event_type: "",
    outcome: "",
    actor: "",
    page: 1,
    pageSize: 20,
  },

  // Media Upload & Analysis State
  uploadMode: "image", // "image" or "video"
  selectedFile: null,
  activeJobId: null,
  jobPollTimer: null,
  isSingleFrame: false,
  currentMediaSource: null, // null (empty) | "upload" | "camera"
  uploadZones: [],

  // Step 16: Interactive Safety Zones State
  zones: [],
  isDrawingZone: false,
  currentZonePoints: [],
  cursorPos: null,
  editingZoneId: null,

  // Step 17: Explainability, Evidence & Analytics State
  incidentFilters: {
    risk_level: "",
    event_type: "",
    track_id: "",
    zone: "",
  },
  analyticsSummary: null,
  activeExplanation: null,

  // Step 22: Alerting & Lifecycle State
  alerts: [],
  alertStats: null,
  selectedAlert: null,
  alertFilters: {
    status: "",
    severity: "",
    search: "",
    page: 1,
    pageSize: 20,
  },
};

/**
 * Step 23: Centralized authenticated fetch wrapper.
 * Injects Authorization: Bearer <token> and handles 401 Unauthorized by prompting login.
 */
async function authFetch(url, options = {}) {
  const opts = { ...options };
  opts.headers = { ...(opts.headers || {}) };

  const token = state.authToken || localStorage.getItem("intelliwatch_token");
  if (token) {
    opts.headers["Authorization"] = `Bearer ${token}`;
  }

  const res = await fetch(url, opts);
  if (res.status === 401) {
    // Session expired or invalid
    state.authToken = "";
    localStorage.removeItem("intelliwatch_token");
    showLoginModal("Session expired or authentication required. Please log in.");
  }
  return res;
}

// DOM Elements
const elements = {
  cpuBadge: document.getElementById("cpuBadge"),
  fpsBadge: document.getElementById("fpsBadge"),
  latencyBadge: document.getElementById("latencyBadge"),
  tracksBadge: document.getElementById("tracksBadge"),
  incidentsBadge: document.getElementById("incidentsBadge"),
  alertsBadge: document.getElementById("alertsBadge"),
  statusDot: document.getElementById("statusDot"),
  pollToggleBtn: document.getElementById("pollToggleBtn"),
  refreshBtn: document.getElementById("refreshBtn"),
  canvas: document.getElementById("sceneCanvas"),
  sceneRelationshipsList: document.getElementById("sceneRelationshipsList"),
  entitySummaryList: document.getElementById("entitySummaryList"),
  safetyEventContainer: document.getElementById("safetyEventContainer"),
  riskContainer: document.getElementById("riskContainer"),
  warningContainer: document.getElementById("warningContainer"),
  ppeContainer: document.getElementById("ppeContainer"),
  behaviorContainer: document.getElementById("behaviorContainer"),
  depthContainer: document.getElementById("depthContainer"),
  temporalEntityProfiles: document.getElementById("temporalEntityProfiles"),
  temporalTimeline: document.getElementById("temporalTimeline"),
  incidentTableBody: document.getElementById("incidentTableBody"),
  evidenceModal: document.getElementById("evidenceModal"),
  closeModalBtn: document.getElementById("closeModalBtn"),
  modalTitle: document.getElementById("modalTitle"),
  modalEvidenceImage: document.getElementById("modalEvidenceImage"),
  noEvidenceNotice: document.getElementById("noEvidenceNotice"),
  modalIncidentDetails: document.getElementById("modalIncidentDetails"),

  // Upload & Analysis Controls
  uploadSection: document.getElementById("uploadSection"),
  modeImageBtn: document.getElementById("modeImageBtn"),
  modeVideoBtn: document.getElementById("modeVideoBtn"),
  modeStreamBtn: document.getElementById("modeStreamBtn"),
  uploadBody: document.getElementById("uploadBody"),
  streamManagerBody: document.getElementById("streamManagerBody"),
  camIdInput: document.getElementById("camIdInput"),
  camNameInput: document.getElementById("camNameInput"),
  camSourceInput: document.getElementById("camSourceInput"),
  camSamplingInput: document.getElementById("camSamplingInput"),
  camMaxFpsInput: document.getElementById("camMaxFpsInput"),
  addCameraBtn: document.getElementById("addCameraBtn"),
  cameraCountBadge: document.getElementById("cameraCountBadge"),
  analysisCameraSelect: document.getElementById("analysisCameraSelect"),
  analysisCameraCalibrationNotice: document.getElementById("analysisCameraCalibrationNotice"),
  calibrationCameraSelect: document.getElementById("calibrationCameraSelect"),
  calibrationCameraStatus: document.getElementById("calibrationCameraStatus"),
  calibrationStatusBadge: document.getElementById("calibrationStatusBadge"),
  calibrationSelectedCameraLabel: document.getElementById("calibrationSelectedCameraLabel"),
  calibrationRefreshFrameBtn: document.getElementById("calibrationRefreshFrameBtn"),
  calibrationImageStage: document.getElementById("calibrationImageStage"),
  calibrationImageEmpty: document.getElementById("calibrationImageEmpty"),
  calibrationImageEmptyTitle: document.getElementById("calibrationImageEmptyTitle"),
  calibrationImageEmptyHint: document.getElementById("calibrationImageEmptyHint"),
  calibrationImageLayer: document.getElementById("calibrationImageLayer"),
  calibrationFrameImage: document.getElementById("calibrationFrameImage"),
  calibrationPointOverlay: document.getElementById("calibrationPointOverlay"),
  calibrationPointPolygon: document.getElementById("calibrationPointPolygon"),
  calibrationPointLine: document.getElementById("calibrationPointLine"),
  calibrationPointMarkers: document.getElementById("calibrationPointMarkers"),
  calibrationFrameDimensions: document.getElementById("calibrationFrameDimensions"),
  calibrationPointCount: document.getElementById("calibrationPointCount"),
  calibrationPointList: document.getElementById("calibrationPointList"),
  calibrationWidthInput: document.getElementById("calibrationWidthInput"),
  calibrationDepthInput: document.getElementById("calibrationDepthInput"),
  calibrationSaveSummary: document.getElementById("calibrationSaveSummary"),
  calibrationResetPointsBtn: document.getElementById("calibrationResetPointsBtn"),
  calibrationValidateBtn: document.getElementById("calibrationValidateBtn"),
  calibrationSaveBtn: document.getElementById("calibrationSaveBtn"),
  calibrationClearBtn: document.getElementById("calibrationClearBtn"),
  calibrationFeedback: document.getElementById("calibrationFeedback"),
  calibrationValidationResult: document.getElementById("calibrationValidationResult"),
  calibrationSceneMetricContext: document.getElementById("calibrationSceneMetricContext"),
  refreshCamerasBtn: document.getElementById("refreshCamerasBtn"),
  camerasGrid: document.getElementById("camerasGrid"),
  dropZone: document.getElementById("dropZone"),
  dropZoneIcon: document.getElementById("dropZoneIcon"),
  dropZoneText: document.getElementById("dropZoneText"),
  dropZoneHint: document.getElementById("dropZoneHint"),
  mediaFileInput: document.getElementById("mediaFileInput"),
  selectedFileCard: document.getElementById("selectedFileCard"),
  selectedFileIcon: document.getElementById("selectedFileIcon"),
  selectedFileName: document.getElementById("selectedFileName"),
  selectedFileType: document.getElementById("selectedFileType"),
  selectedFileSize: document.getElementById("selectedFileSize"),
  clearFileBtn: document.getElementById("clearFileBtn"),
  analyzeBtn: document.getElementById("analyzeBtn"),
  newAnalysisBtn: document.getElementById("newAnalysisBtn"),

  // Processing & Telemetry Card
  processingCard: document.getElementById("processingCard"),
  analysisStatusPill: document.getElementById("analysisStatusPill"),
  processingTitle: document.getElementById("processingTitle"),
  processingPct: document.getElementById("processingPct"),
  analysisProgressBar: document.getElementById("analysisProgressBar"),
  telFile: document.getElementById("telFile"),
  telFrames: document.getElementById("telFrames"),
  telFps: document.getElementById("telFps"),
  telRisk: document.getElementById("telRisk"),
  telTracks: document.getElementById("telTracks"),
  telIncidents: document.getElementById("telIncidents"),
  processingNotice: document.getElementById("processingNotice"),

  // Viewport Media Elements
  viewportStateBadge: document.getElementById("viewportStateBadge"),
  viewportContainer: document.getElementById("viewportContainer"),
  viewportImage: document.getElementById("viewportImage"),
  viewportVideo: document.getElementById("viewportVideo"),
  viewportLiveStream: document.getElementById("viewportLiveStream"),
  viewportCameraSelect: document.getElementById("viewportCameraSelect"),
  viewportCompletedSummary: document.getElementById("viewportCompletedSummary"),
  sumTotalFrames: document.getElementById("sumTotalFrames"),
  sumProcessedFrames: document.getElementById("sumProcessedFrames"),
  sumAvgFps: document.getElementById("sumAvgFps"),
  sumAvgLatency: document.getElementById("sumAvgLatency"),
  sumIncidents: document.getElementById("sumIncidents"),
  sumRisk: document.getElementById("sumRisk"),
  viewIncidentsBtn: document.getElementById("viewIncidentsBtn"),

  // Viewport & Detection Debug Telemetry
  dbgSourceDim: document.getElementById("dbgSourceDim"),
  dbgViewportDim: document.getElementById("dbgViewportDim"),
  dbgDisplayedDim: document.getElementById("dbgDisplayedDim"),
  dbgScaleFactor: document.getElementById("dbgScaleFactor"),
  dbgLetterboxOffset: document.getElementById("dbgLetterboxOffset"),
  dbgRawPersons: document.getElementById("dbgRawPersons"),
  dbgFilteredPersons: document.getElementById("dbgFilteredPersons"),
  dbgTrackCount: document.getElementById("dbgTrackCount"),
  dbgRenderedBoxes: document.getElementById("dbgRenderedBoxes"),

  // Step 16: Interactive Safety Zones Controls
  zoneOverlayCanvas: document.getElementById("zoneOverlayCanvas"),
  tabZonesBtn: document.getElementById("tabZonesBtn"),
  startDrawZoneBtn: document.getElementById("startDrawZoneBtn"),
  zoneDrawForm: document.getElementById("zoneDrawForm"),
  zoneFormTitle: document.getElementById("zoneFormTitle"),
  zonePointCountBadge: document.getElementById("zonePointCountBadge"),
  zoneNameInput: document.getElementById("zoneNameInput"),
  zoneTypeSelect: document.getElementById("zoneTypeSelect"),
  zoneDwellInput: document.getElementById("zoneDwellInput"),
  saveZoneBtn: document.getElementById("saveZoneBtn"),
  clearPointsBtn: document.getElementById("clearPointsBtn"),
  cancelDrawBtn: document.getElementById("cancelDrawBtn"),
  configuredZoneCount: document.getElementById("configuredZoneCount"),
  configuredZonesList: document.getElementById("configuredZonesList"),

  // Step 17: Analytics & Filter Controls
  tabAnalyticsBtn: document.getElementById("tabAnalyticsBtn"),
  refreshAnalyticsBtn: document.getElementById("refreshAnalyticsBtn"),
  kpiTotalIncidents: document.getElementById("kpiTotalIncidents"),
  kpiCriticalIncidents: document.getElementById("kpiCriticalIncidents"),
  kpiZoneIntrusions: document.getElementById("kpiZoneIntrusions"),
  kpiPpeViolations: document.getElementById("kpiPpeViolations"),
  kpiAvgDuration: document.getElementById("kpiAvgDuration"),
  kpiActiveIncidents: document.getElementById("kpiActiveIncidents"),
  kpiProximityEvents: document.getElementById("kpiProximityEvents"),
  analyticsEventTypeDist: document.getElementById("analyticsEventTypeDist"),
  analyticsActiveZonesList: document.getElementById("analyticsActiveZonesList"),

  filterRiskTier: document.getElementById("filterRiskTier"),
  filterEventType: document.getElementById("filterEventType"),
  filterTrackId: document.getElementById("filterTrackId"),
  filterZone: document.getElementById("filterZone"),
  applyFiltersBtn: document.getElementById("applyFiltersBtn"),
  resetFiltersBtn: document.getElementById("resetFiltersBtn"),

  downloadPackageBtn: document.getElementById("downloadPackageBtn"),
  modalEvidenceBadge: document.getElementById("modalEvidenceBadge"),
  modalRiskTrendCanvas: document.getElementById("modalRiskTrendCanvas"),
  riskTrendStatus: document.getElementById("riskTrendStatus"),
  riskTrendNotice: document.getElementById("riskTrendNotice"),
  fwWho: document.getElementById("fwWho"),
  fwWhat: document.getElementById("fwWhat"),
  fwWhere: document.getElementById("fwWhere"),
  fwWhen: document.getElementById("fwWhen"),
  fwWhyList: document.getElementById("fwWhyList"),
  riskBreakdownBody: document.getElementById("riskBreakdownBody"),
  riskBreakdownFinal: document.getElementById("riskBreakdownFinal"),
  modalIncidentTimeline: document.getElementById("modalIncidentTimeline"),
  modalSceneAndWarnings: document.getElementById("modalSceneAndWarnings"),

  // Step 22: Alert Center & Detail Modal Controls
  tabAlertsBtn: document.getElementById("tabAlertsBtn"),
  alertTabBadge: document.getElementById("alertTabBadge"),
  kpiAlertsTotal: document.getElementById("kpiAlertsTotal"),
  kpiAlertsNew: document.getElementById("kpiAlertsNew"),
  kpiAlertsAck: document.getElementById("kpiAlertsAck"),
  kpiAlertsCritical: document.getElementById("kpiAlertsCritical"),
  alertFilterStatus: document.getElementById("alertFilterStatus"),
  alertFilterSeverity: document.getElementById("alertFilterSeverity"),
  alertFilterSearch: document.getElementById("alertFilterSearch"),
  refreshAlertsBtn: document.getElementById("refreshAlertsBtn"),
  alertsFeedList: document.getElementById("alertsFeedList"),
  alertPrevPageBtn: document.getElementById("alertPrevPageBtn"),
  alertNextPageBtn: document.getElementById("alertNextPageBtn"),
  alertPageInfo: document.getElementById("alertPageInfo"),

  // Alert Detail Modal
  alertDetailModal: document.getElementById("alertDetailModal"),
  alertModalTitle: document.getElementById("alertModalTitle"),
  alertModalSeverity: document.getElementById("alertModalSeverity"),
  alertModalStatus: document.getElementById("alertModalStatus"),
  closeAlertModalBtn: document.getElementById("closeAlertModalBtn"),
  alertEvidenceImage: document.getElementById("alertEvidenceImage"),
  alertNoEvidenceNotice: document.getElementById("alertNoEvidenceNotice"),
  alertEvidenceBadge: document.getElementById("alertEvidenceBadge"),
  alertDetId: document.getElementById("alertDetId"),
  alertDetViolation: document.getElementById("alertDetViolation"),
  alertDetCamera: document.getElementById("alertDetCamera"),
  alertDetTrack: document.getElementById("alertDetTrack"),
  alertDetConf: document.getElementById("alertDetConf"),
  alertDetOccurrences: document.getElementById("alertDetOccurrences"),
  alertDetFirst: document.getElementById("alertDetFirst"),
  alertDetLast: document.getElementById("alertDetLast"),
  alertHistoryList: document.getElementById("alertHistoryList"),
  alertOperatorName: document.getElementById("alertOperatorName"),
  alertActionNotes: document.getElementById("alertActionNotes"),
  btnModalAck: document.getElementById("btnModalAck"),
  btnModalResolve: document.getElementById("btnModalResolve"),
  btnModalDismiss: document.getElementById("btnModalDismiss"),
  alertActionFeedback: document.getElementById("alertActionFeedback"),

  // Step 23: Auth & Admin DOM Elements
  userPill: document.getElementById("userPill"),
  headerUsername: document.getElementById("headerUsername"),
  headerRoleBadge: document.getElementById("headerRoleBadge"),
  logoutBtn: document.getElementById("logoutBtn"),
  changePasswordBtn: document.getElementById("changePasswordBtn"),
  tabUsersBtn: document.getElementById("tabUsersBtn"),
  tabAuditBtn: document.getElementById("tabAuditBtn"),

  // Login Modal
  loginModal: document.getElementById("loginModal"),
  loginForm: document.getElementById("loginForm"),
  loginUsername: document.getElementById("loginUsername"),
  loginPassword: document.getElementById("loginPassword"),
  loginErrorMsg: document.getElementById("loginErrorMsg"),
  loginSubmitBtn: document.getElementById("loginSubmitBtn"),

  // Change Password Modal
  changePasswordModal: document.getElementById("changePasswordModal"),
  changePasswordForm: document.getElementById("changePasswordForm"),
  currentPasswordInput: document.getElementById("currentPasswordInput"),
  newPasswordInput: document.getElementById("newPasswordInput"),
  confirmPasswordInput: document.getElementById("confirmPasswordInput"),
  closeChangePasswordBtn: document.getElementById("closeChangePasswordBtn"),
  changePasswordMsg: document.getElementById("changePasswordMsg"),
  savePasswordBtn: document.getElementById("savePasswordBtn"),

  // User Management
  userTableBody: document.getElementById("userTableBody"),
  createUserModalBtn: document.getElementById("createUserModalBtn"),
  createUserModal: document.getElementById("createUserModal"),
  createUserForm: document.getElementById("createUserForm"),
  closeCreateUserBtn: document.getElementById("closeCreateUserBtn"),
  createUsernameInput: document.getElementById("createUsernameInput"),
  createUserRoleSelect: document.getElementById("createUserRoleSelect"),
  createPasswordInput: document.getElementById("createPasswordInput"),
  createPermittedCamerasInput: document.getElementById("createPermittedCamerasInput"),
  createUserMsg: document.getElementById("createUserMsg"),
  saveNewUserBtn: document.getElementById("saveNewUserBtn"),

  // Audit Logs
  refreshAuditBtn: document.getElementById("refreshAuditBtn"),
  auditFilterEvent: document.getElementById("auditFilterEvent"),
  auditFilterOutcome: document.getElementById("auditFilterOutcome"),
  auditFilterActor: document.getElementById("auditFilterActor"),
  applyAuditFiltersBtn: document.getElementById("applyAuditFiltersBtn"),
  auditLogList: document.getElementById("auditLogList"),
  auditPrevPageBtn: document.getElementById("auditPrevPageBtn"),
  auditNextPageBtn: document.getElementById("auditNextPageBtn"),
  auditPageInfo: document.getElementById("auditPageInfo"),
};

// -------------------------------------------------------------
// INITIALIZATION
// -------------------------------------------------------------
document.addEventListener("DOMContentLoaded", async () => {
  setupTabs();
  setupEventListeners();
  setupAuthControls();
  setupUploadControls();
  setupCalibrationWorkbench();
  initCanvas();
  initZonesModule();
  setupViewportObserver();
  fitViewportMedia();

  // Step 23: Initialize session and fetch user profile
  await initAuthSession();

  await loadCameras();
  fetchData();
  startPolling();
});


function setupTabs() {
  const tabs = document.querySelectorAll(".tab-btn");
  tabs.forEach((tab) => {
    tab.addEventListener("click", () => {
      tabs.forEach((t) => t.classList.remove("active"));
      document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

      tab.classList.add("active");
      const targetId = tab.dataset.tab;
      const targetContent = document.getElementById(targetId);
      if (targetContent) targetContent.classList.add("active");
      state.activeTab = targetId;

      if (targetId === "tab-analytics") {
        fetchAnalytics();
      } else if (targetId === "tab-diagnostics") {
        fetchModelDiagnostics();
      } else if (targetId === "tab-calibration") {
        refreshCalibrationSceneContext();
      }
    });
  });
}

function setupEventListeners() {
  if (elements.pollToggleBtn) {
    elements.pollToggleBtn.addEventListener("click", () => {
      state.isPolling = !state.isPolling;
      elements.pollToggleBtn.textContent = state.isPolling ? "Pause Polling" : "Resume Polling";
      elements.pollToggleBtn.classList.toggle("btn-primary", !state.isPolling);
      if (state.isPolling) {
        startPolling();
      } else {
        stopPolling();
      }
    });
  }

  if (elements.refreshBtn) {
    elements.refreshBtn.addEventListener("click", () => {
      fetchData();
    });
  }

  if (elements.closeModalBtn) {
    elements.closeModalBtn.addEventListener("click", closeModal);
  }

  if (elements.evidenceModal) {
    elements.evidenceModal.addEventListener("click", (e) => {
      if (e.target === elements.evidenceModal) closeModal();
    });
  }

  // Step 17: Incident Filters
  if (elements.applyFiltersBtn) {
    elements.applyFiltersBtn.addEventListener("click", () => {
      state.incidentFilters.risk_level = elements.filterRiskTier ? elements.filterRiskTier.value : "";
      state.incidentFilters.event_type = elements.filterEventType ? elements.filterEventType.value : "";
      state.incidentFilters.track_id = elements.filterTrackId ? elements.filterTrackId.value.trim() : "";
      state.incidentFilters.zone = elements.filterZone ? elements.filterZone.value.trim() : "";
      fetchFilteredIncidents();
    });
  }

  if (elements.resetFiltersBtn) {
    elements.resetFiltersBtn.addEventListener("click", () => {
      if (elements.filterRiskTier) elements.filterRiskTier.value = "";
      if (elements.filterEventType) elements.filterEventType.value = "";
      if (elements.filterTrackId) elements.filterTrackId.value = "";
      if (elements.filterZone) elements.filterZone.value = "";
      state.incidentFilters = { risk_level: "", event_type: "", track_id: "", zone: "" };
      fetchFilteredIncidents();
    });
  }

  // Step 17: Analytics & Evidence Package
  if (elements.refreshAnalyticsBtn) {
    elements.refreshAnalyticsBtn.addEventListener("click", () => {
      fetchAnalytics();
    });
  }

  if (elements.downloadPackageBtn) {
    elements.downloadPackageBtn.addEventListener("click", () => {
      downloadEvidencePackage();
    });
  }

  // Step 22: Alert Feed Controls & Detail Modal
  if (elements.refreshAlertsBtn) {
    elements.refreshAlertsBtn.addEventListener("click", () => {
      fetchAlerts();
      fetchAlertStats();
    });
  }

  if (elements.alertFilterStatus) {
    elements.alertFilterStatus.addEventListener("change", (e) => {
      state.alertFilters.status = e.target.value;
      state.alertFilters.page = 1;
      fetchAlerts();
    });
  }

  if (elements.alertFilterSeverity) {
    elements.alertFilterSeverity.addEventListener("change", (e) => {
      state.alertFilters.severity = e.target.value;
      state.alertFilters.page = 1;
      fetchAlerts();
    });
  }

  if (elements.alertFilterSearch) {
    let searchDebounce = null;
    elements.alertFilterSearch.addEventListener("input", (e) => {
      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(() => {
        state.alertFilters.search = e.target.value.trim();
        state.alertFilters.page = 1;
        fetchAlerts();
      }, 300);
    });
  }

  if (elements.alertPrevPageBtn) {
    elements.alertPrevPageBtn.addEventListener("click", () => {
      if (state.alertFilters.page > 1) {
        state.alertFilters.page--;
        fetchAlerts();
      }
    });
  }

  if (elements.alertNextPageBtn) {
    elements.alertNextPageBtn.addEventListener("click", () => {
      state.alertFilters.page++;
      fetchAlerts();
    });
  }

  if (elements.closeAlertModalBtn) {
    elements.closeAlertModalBtn.addEventListener("click", closeAlertDetailModal);
  }

  if (elements.alertDetailModal) {
    elements.alertDetailModal.addEventListener("click", (e) => {
      if (e.target === elements.alertDetailModal) closeAlertDetailModal();
    });
  }

  if (elements.btnModalAck) {
    elements.btnModalAck.addEventListener("click", () => performAlertAction("acknowledge"));
  }
  if (elements.btnModalResolve) {
    elements.btnModalResolve.addEventListener("click", () => performAlertAction("resolve"));
  }
  if (elements.btnModalDismiss) {
    elements.btnModalDismiss.addEventListener("click", () => performAlertAction("dismiss"));
  }
}

// -------------------------------------------------------------
// UPLOAD & ANALYSIS WORKFLOW
// -------------------------------------------------------------
function setupUploadControls() {
  if (!elements.uploadSection) return;

  // Mode toggling
  if (elements.modeImageBtn && elements.modeVideoBtn) {
    elements.modeImageBtn.addEventListener("click", () => setUploadMode("image"));
    elements.modeVideoBtn.addEventListener("click", () => setUploadMode("video"));
  }
  if (elements.modeStreamBtn) {
    elements.modeStreamBtn.addEventListener("click", () => setUploadMode("stream"));
  }
  if (elements.addCameraBtn) {
    elements.addCameraBtn.addEventListener("click", () => registerCameraFromForm());
  }
  if (elements.refreshCamerasBtn) {
    elements.refreshCamerasBtn.addEventListener("click", () => loadCameras());
  }
  if (elements.viewportCameraSelect) {
    elements.viewportCameraSelect.addEventListener("change", (e) => {
      selectCameraForViewing(e.target.value);
    });
  }
  if (elements.analysisCameraSelect) {
    elements.analysisCameraSelect.addEventListener("change", updateUploadCameraStatus);
  }

  // File Picker Click
  if (elements.dropZone && elements.mediaFileInput) {
    elements.dropZone.addEventListener("click", () => {
      elements.mediaFileInput.click();
    });

    // Drag and Drop events
    elements.dropZone.addEventListener("dragover", (e) => {
      e.preventDefault();
      elements.dropZone.classList.add("dragover");
    });

    elements.dropZone.addEventListener("dragleave", (e) => {
      e.preventDefault();
      elements.dropZone.classList.remove("dragover");
    });

    elements.dropZone.addEventListener("drop", (e) => {
      e.preventDefault();
      elements.dropZone.classList.remove("dragover");
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length > 0) {
        handleFileSelected(e.dataTransfer.files[0]);
      }
    });

    elements.mediaFileInput.addEventListener("change", (e) => {
      if (e.target.files && e.target.files.length > 0) {
        handleFileSelected(e.target.files[0]);
      }
    });
  }

  // Clear File Selection
  if (elements.clearFileBtn) {
    elements.clearFileBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      clearSelectedFile();
    });
  }

  // Analyze Button
  if (elements.analyzeBtn) {
    elements.analyzeBtn.addEventListener("click", () => {
      executeAnalysis();
    });
  }

  // New Analysis Button
  if (elements.newAnalysisBtn) {
    elements.newAnalysisBtn.addEventListener("click", () => {
      resetAnalysisState();
    });
  }

  // View Incidents Button
  if (elements.viewIncidentsBtn) {
    elements.viewIncidentsBtn.addEventListener("click", () => {
      const incTable = document.querySelector(".incident-panel");
      if (incTable) incTable.scrollIntoView({ behavior: "smooth" });
    });
  }
}

function getFileExtension(filename) {
  return (filename.substring(filename.lastIndexOf(".")) || "").toLowerCase();
}

function formatBytes(bytes) {
  if (bytes < 1024) return bytes + " B";
  if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + " KB";
  return (bytes / (1024 * 1024)).toFixed(1) + " MB";
}

function setUploadMode(mode) {
  state.uploadMode = mode;
  if (elements.modeImageBtn && elements.modeVideoBtn) {
    elements.modeImageBtn.classList.toggle("active", mode === "image");
    elements.modeVideoBtn.classList.toggle("active", mode === "video");
  }
  if (elements.modeStreamBtn) {
    elements.modeStreamBtn.classList.toggle("active", mode === "stream");
  }

  if (mode === "stream") {
    if (elements.uploadBody) elements.uploadBody.style.display = "none";
    if (elements.streamManagerBody) elements.streamManagerBody.style.display = "block";
    loadCameras();
    return;
  } else {
    if (elements.uploadBody) elements.uploadBody.style.display = "flex";
    if (elements.streamManagerBody) elements.streamManagerBody.style.display = "none";
  }

  if (elements.mediaFileInput) {
    elements.mediaFileInput.accept = mode === "image"
      ? ".jpg,.jpeg,.png"
      : ".mp4,.avi,.mov,.mkv";
  }

  if (elements.dropZoneHint) {
    elements.dropZoneHint.textContent = mode === "image"
      ? "Supported: JPG, JPEG, PNG (Max 25MB)"
      : "Supported: MP4, AVI, MOV, MKV (Max 100MB)";
  }

  // If already selected file doesn't match new mode, clear it
  if (state.selectedFile) {
    const ext = getFileExtension(state.selectedFile.name);
    const validImg = [".jpg", ".jpeg", ".png"].includes(ext);
    const validVid = [".mp4", ".avi", ".mov", ".mkv"].includes(ext);
    if ((mode === "image" && !validImg) || (mode === "video" && !validVid)) {
      clearSelectedFile();
    }
  }
}

function handleFileSelected(file) {
  const ext = getFileExtension(file.name);
  const isImageExt = [".jpg", ".jpeg", ".png"].includes(ext);
  const isVideoExt = [".mp4", ".avi", ".mov", ".mkv"].includes(ext);

  if (!isImageExt && !isVideoExt) {
    alert("Unsupported file type. Please upload JPG, PNG, MP4, AVI, MOV, or MKV.");
    return;
  }

  // Automatically sync mode with dropped media type
  if (isImageExt && state.uploadMode !== "image") {
    setUploadMode("image");
  } else if (isVideoExt && state.uploadMode !== "video") {
    setUploadMode("video");
  }

  state.selectedFile = file;

  // Update selected file card
  if (elements.selectedFileName) elements.selectedFileName.textContent = file.name;
  if (elements.selectedFileType) elements.selectedFileType.textContent = isImageExt ? "IMAGE" : "VIDEO";
  if (elements.selectedFileSize) elements.selectedFileSize.textContent = formatBytes(file.size);
  if (elements.selectedFileIcon) elements.selectedFileIcon.textContent = isImageExt ? "🖼️" : "🎬";

  if (elements.dropZone) elements.dropZone.style.display = "none";
  if (elements.selectedFileCard) elements.selectedFileCard.style.display = "flex";
  if (elements.analyzeBtn) elements.analyzeBtn.disabled = false;
  if (elements.viewportStateBadge) elements.viewportStateBadge.textContent = "[FILE SELECTED - READY]";
}

function clearSelectedFile() {
  state.selectedFile = null;
  if (elements.mediaFileInput) elements.mediaFileInput.value = "";
  if (elements.dropZone) elements.dropZone.style.display = "flex";
  if (elements.selectedFileCard) elements.selectedFileCard.style.display = "none";
  if (elements.analyzeBtn) elements.analyzeBtn.disabled = true;
  if (elements.viewportStateBadge) elements.viewportStateBadge.textContent = "[IDLE]";
}

async function executeAnalysis() {
  if (!state.selectedFile) return;

  const file = state.selectedFile;
  const isImage = state.uploadMode === "image";

  // Transition UI to PROCESSING state
  elements.analyzeBtn.disabled = true;
  elements.analyzeBtn.textContent = "Analyzing...";
  elements.processingCard.style.display = "block";
  elements.analysisStatusPill.className = "status-pill status-processing";
  elements.analysisStatusPill.textContent = "PROCESSING";
  elements.processingTitle.textContent = isImage
    ? "IntelliWatch is analyzing image..."
    : "IntelliWatch is analyzing video stream...";
  elements.processingPct.textContent = "0%";
  elements.analysisProgressBar.style.width = "0%";
  elements.processingNotice.style.display = "none";
  elements.viewportCompletedSummary.style.display = "none";
  elements.viewportStateBadge.textContent = "[PROCESSING]";

  elements.telFile.textContent = file.name;
  elements.telFrames.textContent = isImage ? "0 / 1" : "0 / --";
  elements.telFps.textContent = "-- FPS";
  elements.telRisk.textContent = "--";
  elements.telTracks.textContent = "--";
  elements.telIncidents.textContent = "--";

  const formData = new FormData();
  formData.append("file", file);
  const sourceCameraId = elements.analysisCameraSelect?.value;
  if (sourceCameraId) formData.append("camera_id", sourceCameraId);

  if (isImage) {
    let result;
    try {
      const res = await authFetch(`${CONFIG.apiBase}/api/v1/analyze/image`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errJson.detail || "Image analysis failed");
      }

      result = await res.json();
    } catch (uploadErr) {
      elements.analysisStatusPill.className = "status-pill status-failed";
      elements.analysisStatusPill.textContent = "FAILED";
      elements.processingTitle.textContent = "Analysis Failed";
      elements.processingNotice.className = "processing-notice error";
      elements.processingNotice.textContent = uploadErr.message;
      elements.processingNotice.style.display = "block";
      elements.analyzeBtn.disabled = false;
      elements.analyzeBtn.textContent = "▶ ANALYZE MEDIA";
      elements.newAnalysisBtn.style.display = "inline-flex";
      return;
    }

    // Backend analysis was successful!
    state.currentMediaSource = "upload";
    state.isSingleFrame = true;
    state.currentAssessment = result.assessment;

    // Update Viewport with annotated image
    elements.canvas.style.display = "none";
    elements.viewportVideo.style.display = "none";
    if (elements.viewportLiveStream) elements.viewportLiveStream.style.display = "none";
    elements.viewportImage.src = result.annotated_media_url + "?t=" + Date.now();
    elements.viewportImage.style.display = "block";
    if (result.diagnostics_info) {
      updateViewportDebugOutput(result.diagnostics_info);
    }
    fitViewportMedia();

    const detCount = result.detections_count || (result.assessment && result.assessment.detections ? result.assessment.detections.length : 0);
    const workerCount = result.workers_count || 0;
    if (detCount === 0 && workerCount === 0) {
      elements.viewportStateBadge.textContent = "[SINGLE-FRAME COMPLETE: 0 DETECTIONS]";
    } else {
      elements.viewportStateBadge.textContent = "[SINGLE-FRAME COMPLETE]";
    }

    // Update Processing Card to COMPLETED
    elements.analysisStatusPill.className = "status-pill status-completed";
    elements.analysisStatusPill.textContent = "COMPLETED";
    elements.processingTitle.textContent = "Image Analysis Complete";
    elements.processingPct.textContent = "100%";
    elements.analysisProgressBar.style.width = "100%";
    if (elements.processingNotice) elements.processingNotice.style.display = "none";

    const fpsVal = result.assessment && result.assessment.processing_time_ms > 0
      ? (1000.0 / result.assessment.processing_time_ms).toFixed(1)
      : "N/A";
    elements.telFrames.textContent = "1 / 1";
    elements.telFps.textContent = `${fpsVal} FPS`;
    elements.telRisk.textContent = result.highest_risk_level;
    elements.telTracks.textContent = result.workers_count;
    elements.telIncidents.textContent = (result.new_incident_ids || []).length;

    // Show New Analysis button
    elements.analyzeBtn.style.display = "none";
    elements.newAnalysisBtn.style.display = "inline-flex";

    // Render tab data with protective guard so rendering issues cannot falsify analysis success
    try {
      renderAssessment(result.assessment);
    } catch (renderErr) {
      console.error("[INTELLIWATCH] Error rendering assessment tabs:", renderErr);
    }

    // Re-fetch incidents table so any triggered incident appears
    try {
      fetchData();
    } catch (fetchErr) {
      console.warn("[INTELLIWATCH] Error fetching background data:", fetchErr);
    }
  } else {
    // Video Analysis (Asynchronous)
    try {
      const res = await authFetch(`${CONFIG.apiBase}/api/v1/analyze/video`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errJson = await res.json().catch(() => ({ detail: res.statusText }));
        throw new Error(errJson.detail || "Video upload failed");
      }

      const submitData = await res.json();
      state.activeJobId = submitData.job_id;
      state.isSingleFrame = false;

      // Start polling for video progress
      startJobPolling(submitData.job_id);

    } catch (err) {
      elements.analysisStatusPill.className = "status-pill status-failed";
      elements.analysisStatusPill.textContent = "FAILED";
      elements.processingTitle.textContent = "Analysis Failed";
      elements.processingNotice.className = "processing-notice error";
      elements.processingNotice.textContent = err.message;
      elements.processingNotice.style.display = "block";
      elements.analyzeBtn.disabled = false;
      elements.analyzeBtn.textContent = "▶ ANALYZE MEDIA";
      elements.newAnalysisBtn.style.display = "inline-flex";
    }
  }
}

function startJobPolling(jobId) {
  stopJobPolling();
  state.jobPollTimer = setInterval(() => pollJobStatus(jobId), 1000);
}

function stopJobPolling() {
  if (state.jobPollTimer) {
    clearInterval(state.jobPollTimer);
    state.jobPollTimer = null;
  }
}

async function pollJobStatus(jobId) {
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/analyze/status/${jobId}`);
    if (!res.ok) return;
    const st = await res.json();

    // Update Telemetry
    const pct = Math.min(100, Math.max(0, st.progress_pct || 0)).toFixed(0);
    elements.processingPct.textContent = `${pct}%`;
    elements.analysisProgressBar.style.width = `${pct}%`;

    elements.telFrames.textContent = `${st.current_frame} / ${st.total_frames || '?'}`;
    elements.telFps.textContent = `${(st.processing_fps || 0).toFixed(1)} FPS`;
    elements.telRisk.textContent = st.current_risk_level || "INFO";
    elements.telTracks.textContent = st.active_tracks || 0;
    elements.telIncidents.textContent = st.incidents_count || 0;

    // Live frame preview update
    if (st.latest_frame_url) {
      elements.canvas.style.display = "none";
      elements.viewportVideo.style.display = "none";
      elements.viewportImage.src = st.latest_frame_url + "?t=" + Date.now();
      elements.viewportImage.style.display = "block";
    }

    // Refresh current assessment and incidents
    fetchData();

    if (st.status === "COMPLETED") {
      stopJobPolling();
      handleVideoJobCompleted(jobId);
    } else if (st.status === "FAILED") {
      stopJobPolling();
      elements.analysisStatusPill.className = "status-pill status-failed";
      elements.analysisStatusPill.textContent = "FAILED";
      elements.processingTitle.textContent = "Analysis Failed";
      elements.processingNotice.className = "processing-notice error";
      elements.processingNotice.textContent = st.error_message || "Video processing encountered an error.";
      elements.processingNotice.style.display = "block";
      elements.analyzeBtn.style.display = "none";
      elements.newAnalysisBtn.style.display = "inline-flex";
    }
  } catch (err) {
    console.warn("Job polling error:", err);
  }
}

async function handleVideoJobCompleted(jobId) {
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/analyze/result/${jobId}`);
    if (!res.ok) throw new Error("Could not retrieve video analysis results");
    const result = await res.json();

    elements.analysisStatusPill.className = "status-pill status-completed";
    elements.analysisStatusPill.textContent = "COMPLETED";
    elements.processingTitle.textContent = "Video Stream Analysis Complete";
    elements.processingPct.textContent = "100%";
    elements.analysisProgressBar.style.width = "100%";

    // Display annotated video in viewport
    state.currentMediaSource = "upload";
    state.isSingleFrame = false;
    if (result.annotated_video_url) {
      elements.viewportImage.style.display = "none";
      if (elements.viewportLiveStream) elements.viewportLiveStream.style.display = "none";
      elements.canvas.style.display = "none";
      elements.viewportVideo.src = result.annotated_video_url;
      elements.viewportVideo.style.display = "block";
      elements.viewportVideo.play().catch(() => {});
    }

    // Populate summary card
    elements.sumTotalFrames.textContent = result.total_frames;
    elements.sumProcessedFrames.textContent = result.processed_frames;
    elements.sumAvgFps.textContent = (result.average_fps || 0).toFixed(1) + " FPS";
    elements.sumAvgLatency.textContent = (result.average_latency_ms || 0).toFixed(1) + " ms";
    elements.sumIncidents.textContent = result.total_incidents;
    elements.sumRisk.textContent = result.highest_risk_tier;
    elements.viewportCompletedSummary.style.display = "flex";

    elements.viewportStateBadge.textContent = "[ANALYSIS COMPLETE]";
    elements.analyzeBtn.style.display = "none";
    elements.newAnalysisBtn.style.display = "inline-flex";

    fetchData();
  } catch (err) {
    console.error("Error retrieving completed video results:", err);
  }
}

function resetAnalysisState() {
  stopJobPolling();
  state.activeJobId = null;
  state.isSingleFrame = false;
  state.currentMediaSource = null;
  clearSelectedFile();

  // Reset viewport
  if (elements.viewportImage) elements.viewportImage.style.display = "none";
  if (elements.viewportLiveStream) elements.viewportLiveStream.style.display = "none";
  if (elements.viewportVideo) {
    elements.viewportVideo.style.display = "none";
    elements.viewportVideo.pause();
    elements.viewportVideo.removeAttribute("src");
    elements.viewportVideo.load();
  }
  if (elements.viewportCompletedSummary) elements.viewportCompletedSummary.style.display = "none";
  if (elements.canvas) elements.canvas.style.display = "block";
  initCanvas();
  fitViewportMedia();
  renderZonesOverlay();

  // Reset processing card
  if (elements.processingCard) elements.processingCard.style.display = "none";
  if (elements.processingNotice) elements.processingNotice.style.display = "none";
  if (elements.analysisProgressBar) elements.analysisProgressBar.style.width = "0%";
  if (elements.analyzeBtn) {
    elements.analyzeBtn.style.display = "inline-flex";
    elements.analyzeBtn.disabled = true;
    elements.analyzeBtn.textContent = "▶ ANALYZE MEDIA";
  }
  if (elements.newAnalysisBtn) elements.newAnalysisBtn.style.display = "none";
  if (elements.viewportStateBadge) elements.viewportStateBadge.textContent = "[NO ACTIVE SCENE - IDLE]";
}


function initCanvas() {
  if (!elements.canvas) return;
  const ctx = elements.canvas.getContext("2d");
  const cw = elements.canvas.width;
  const ch = elements.canvas.height;

  ctx.fillStyle = "#0a0e17";
  ctx.fillRect(0, 0, cw, ch);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.03)";
  ctx.lineWidth = 1;
  for (let x = 0; x < cw; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, ch);
    ctx.stroke();
  }
  for (let y = 0; y < ch; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(cw, y);
    ctx.stroke();
  }

  ctx.fillStyle = "#64748b";
  ctx.font = "bold 15px -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif";
  ctx.textAlign = "center";
  ctx.fillText("NO ANALYZED SCENE LOADED", cw / 2, ch / 2 - 12);

  ctx.fillStyle = "#475569";
  ctx.font = "12px monospace";
  ctx.fillText("Upload an image / video file or select a live camera feed to begin perception analysis", cw / 2, ch / 2 + 16);

  if (elements.viewportStateBadge && !state.currentMediaSource) {
    elements.viewportStateBadge.textContent = "[NO ACTIVE SCENE - IDLE]";
  }
}

// -------------------------------------------------------------
// POLLING & DATA FETCHING
// -------------------------------------------------------------
function startPolling() {
  stopPolling();
  state.pollTimer = setInterval(fetchData, CONFIG.pollIntervalMs);
}

function stopPolling() {
  if (state.pollTimer) {
    clearInterval(state.pollTimer);
    state.pollTimer = null;
  }
}

function buildIncidentsUrl() {
  let url = `${CONFIG.apiBase}/api/v1/incidents?limit=50`;
  if (state.incidentFilters.risk_level) {
    url += `&risk_level=${encodeURIComponent(state.incidentFilters.risk_level)}`;
  }
  if (state.incidentFilters.event_type) {
    url += `&event_type=${encodeURIComponent(state.incidentFilters.event_type)}`;
  }
  if (state.incidentFilters.track_id) {
    url += `&track_id=${encodeURIComponent(state.incidentFilters.track_id)}`;
  }
  if (state.incidentFilters.zone) {
    url += `&zone=${encodeURIComponent(state.incidentFilters.zone)}`;
  }
  return url;
}

async function fetchFilteredIncidents() {
  try {
    const url = buildIncidentsUrl();
    const res = await authFetch(url);
    if (res.ok) {
      state.incidents = await res.json();
      renderIncidents(state.incidents);
    }
  } catch (err) {
    console.warn("Could not fetch filtered incidents:", err);
  }
}

async function fetchData() {
  try {
    const incidentsUrl = buildIncidentsUrl();
    const alertsUrl = buildAlertsUrl();

    const [assessmentRes, incidentsRes, statusRes, temporalRes, zonesRes, alertsRes, alertStatsRes, sysMetricsRes] = await Promise.all([
      authFetch(`${CONFIG.apiBase}/api/v1/assessment/current`).catch(() => null),
      authFetch(incidentsUrl).catch(() => null),
      authFetch(`${CONFIG.apiBase}/api/v1/status`).catch(() => null),
      authFetch(`${CONFIG.apiBase}/api/v1/temporal/timeline?n=30`).catch(() => null),
      authFetch(`${CONFIG.apiBase}/api/v1/zones`).catch(() => null),
      authFetch(alertsUrl).catch(() => null),
      authFetch(`${CONFIG.apiBase}/api/v1/alerts/statistics`).catch(() => null),
      authFetch(`${CONFIG.apiBase}/api/v1/system/metrics`).catch(() => null),
    ]);

    if (sysMetricsRes && sysMetricsRes.ok) {
      state.systemMetrics = await sysMetricsRes.json();
      const statComp = document.getElementById("statComplianceRate");
      if (statComp && state.systemMetrics.compliance_rate !== null && state.systemMetrics.compliance_rate !== undefined) {
        statComp.textContent = `${state.systemMetrics.compliance_rate.toFixed(1)}%`;
      }
      const statMode = document.getElementById("statInferenceMode");
      if (statMode && state.systemMetrics.execution_device) {
        statMode.textContent = state.systemMetrics.execution_device.toUpperCase() === "CPU" ? "CPU FALLBACK" : `${state.systemMetrics.execution_device.toUpperCase()} ACCEL`;
      }
      const statFpsLat = document.getElementById("statFpsLatency");
      if (statFpsLat && state.systemMetrics.fps && state.systemMetrics.avg_latency_ms) {
        statFpsLat.textContent = `${state.systemMetrics.fps.toFixed(1)} FPS · ${state.systemMetrics.avg_latency_ms.toFixed(0)} ms`;
      }
    }

    if (assessmentRes && assessmentRes.ok) {
      const assessmentData = await assessmentRes.json();
      if (!state.isSingleFrame) {
        state.currentAssessment = assessmentData;
        renderAssessment(state.currentAssessment);
      }
      elements.statusDot.className = "status-dot";
    }

    if (incidentsRes && incidentsRes.ok) {
      state.incidents = await incidentsRes.json();
      renderIncidents(state.incidents);
    }

    if (statusRes && statusRes.ok) {
      state.systemStatus = await statusRes.json();
      renderStatus(state.systemStatus);
    }

    if (temporalRes && temporalRes.ok) {
      const timeline = await temporalRes.json();
      renderTemporalTimeline(timeline);
    }

    if (zonesRes && zonesRes.ok) {
      state.zones = await zonesRes.json();
      renderConfiguredZonesList();
      renderZonesOverlay();
    }

    if (alertsRes && alertsRes.ok) {
      const alertData = await alertsRes.json();
      state.alerts = alertData.items || [];
      renderAlertsFeed(alertData);
    }

    if (alertStatsRes && alertStatsRes.ok) {
      state.alertStats = await alertStatsRes.json();
      renderAlertStats(state.alertStats);
    }
  } catch (err) {
    console.warn("Polling warning:", err);
    elements.statusDot.className = "status-dot danger";
  }
}

// -------------------------------------------------------------
// STEP 22: ALERTING & LIFECYCLE CONTROLLERS
// -------------------------------------------------------------
function buildAlertsUrl() {
  let url = `${CONFIG.apiBase}/api/v1/alerts?page=${state.alertFilters.page}&page_size=${state.alertFilters.pageSize}`;
  if (state.alertFilters.status) {
    url += `&status=${encodeURIComponent(state.alertFilters.status)}`;
  }
  if (state.alertFilters.severity) {
    url += `&severity=${encodeURIComponent(state.alertFilters.severity)}`;
  }
  if (state.alertFilters.search) {
    url += `&search=${encodeURIComponent(state.alertFilters.search)}`;
  }
  return url;
}

async function fetchAlerts() {
  try {
    const res = await authFetch(buildAlertsUrl());
    if (res.ok) {
      const data = await res.json();
      state.alerts = data.items || [];
      renderAlertsFeed(data);
    }
  } catch (err) {
    console.warn("Could not fetch alerts:", err);
  }
}

async function fetchAlertStats() {
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/alerts/statistics`);
    if (res.ok) {
      state.alertStats = await res.json();
      renderAlertStats(state.alertStats);
    }
  } catch (err) {
    console.warn("Could not fetch alert stats:", err);
  }
}

function renderAlertStats(stats) {
  if (!stats) return;
  if (elements.kpiAlertsTotal) elements.kpiAlertsTotal.textContent = stats.total_alerts || 0;
  const newCount = (stats.by_status && stats.by_status.NEW) || 0;
  const ackCount = (stats.by_status && stats.by_status.ACKNOWLEDGED) || 0;
  const critCount = (stats.by_severity && stats.by_severity.CRITICAL) || 0;

  if (elements.kpiAlertsNew) elements.kpiAlertsNew.textContent = newCount;
  if (elements.kpiAlertsAck) elements.kpiAlertsAck.textContent = ackCount;
  if (elements.kpiAlertsCritical) elements.kpiAlertsCritical.textContent = critCount;

  // Header badge update
  if (elements.alertsBadge) {
    elements.alertsBadge.textContent = `Alerts: ${stats.total_alerts} (${newCount} New)`;
    elements.alertsBadge.style.color = newCount > 0 ? "#ef4444" : "var(--text-secondary)";
  }

  // Overview KPI stat cards
  const statAlertsEl = document.getElementById("statTotalAlerts");
  if (statAlertsEl) {
    statAlertsEl.textContent = stats.total_alerts || 0;
  }
  const statUnresolvedEl = document.getElementById("statUnresolvedAlerts");
  if (statUnresolvedEl) {
    const unres = stats.active_unresolved_count !== undefined ? stats.active_unresolved_count : (newCount + ackCount);
    statUnresolvedEl.textContent = `${unres} UNRESOLVED INCIDENTS`;
  }
  const statComplianceEl = document.getElementById("statComplianceRate");
  if (statComplianceEl) {
    if (state.systemMetrics && state.systemMetrics.compliance_rate !== null && state.systemMetrics.compliance_rate !== undefined) {
      statComplianceEl.textContent = `${state.systemMetrics.compliance_rate.toFixed(1)}%`;
    } else if (stats.total_alerts === 0) {
      statComplianceEl.textContent = "100.0%";
    }
  }

  // Tab notification badge
  if (elements.alertTabBadge) {
    if (newCount > 0) {
      elements.alertTabBadge.style.display = "inline-block";
      elements.alertTabBadge.textContent = newCount;
    } else {
      elements.alertTabBadge.style.display = "none";
    }
  }
}

function renderAlertsFeed(data) {
  if (!elements.alertsFeedList) return;
  const items = (data && data.items) || [];
  const total = (data && data.total) || 0;
  const page = (data && data.page) || 1;
  const totalPages = (data && data.total_pages) || 1;

  if (elements.alertPageInfo) {
    elements.alertPageInfo.textContent = `Page ${page} of ${totalPages} (${total} total)`;
  }
  if (elements.alertPrevPageBtn) {
    elements.alertPrevPageBtn.disabled = page <= 1;
  }
  if (elements.alertNextPageBtn) {
    elements.alertNextPageBtn.disabled = page >= totalPages;
  }

  if (items.length === 0) {
    elements.alertsFeedList.innerHTML = `
      <div style="color: var(--text-muted); font-size: 11px; padding: 24px; text-align: center;">
        No safety alerts matching current criteria.
      </div>
    `;
    return;
  }

  elements.alertsFeedList.innerHTML = items.map((alert) => {
    const dt = new Date(alert.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const isNew = alert.status === "NEW";
    const isAck = alert.status === "ACKNOWLEDGED";
    const isResolved = alert.status === "RESOLVED";
    const isDismissed = alert.status === "DISMISSED";

    return `
      <div class="alert-item-card" style="background: rgba(15, 23, 42, 0.65); border: 1px solid var(--border-color); border-left: 3px solid ${getSeverityColor(alert.severity)}; border-radius: 4px; padding: 8px 10px;">
        <div style="display: flex; justify-content: space-between; align-items: flex-start; margin-bottom: 4px;">
          <div style="display: flex; align-items: center; gap: 6px; flex-wrap: wrap;">
            <span class="alert-pill sev-${alert.severity}">${alert.severity}</span>
            <strong style="color: #fff; font-size: 11px;">${escapeHtml(alert.violation_type)}</strong>
            <span class="alert-status-pill status-${alert.status}">${alert.status}</span>
            ${alert.occurrence_count > 1 ? `<span style="font-size: 9px; background: rgba(255,255,255,0.1); padding: 1px 4px; border-radius: 2px; color: #94a3b8;">x${alert.occurrence_count}</span>` : ""}
          </div>
          <span style="font-size: 10px; font-family: var(--font-mono); color: var(--text-muted);">${dt}</span>
        </div>

        <div style="display: flex; justify-content: space-between; align-items: center; margin-top: 4px; font-size: 10px; color: var(--text-muted);">
          <div style="display: flex; gap: 8px;">
            <span>Cam: <strong style="color: #cbd5e1;">${escapeHtml(alert.camera_id)}</strong></span>
            <span>Worker: <strong style="color: #cbd5e1;">${alert.track_id !== null ? '#' + alert.track_id : 'N/A'}</strong></span>
            <span>Conf: <strong style="color: #cbd5e1;">${(alert.confidence * 100).toFixed(0)}%</strong></span>
          </div>
          <div class="alert-row-actions">
            ${isNew ? `<button class="btn-ack" onclick="quickAcknowledgeAlert('${alert.alert_id}', event)">✓ Ack</button>` : ""}
            ${!isResolved && !isDismissed ? `<button class="btn-resolve" onclick="quickResolveAlert('${alert.alert_id}', event)">Resolve</button>` : ""}
            <button class="btn-dismiss" onclick="openAlertDetailModal('${alert.alert_id}')">Details ↗</button>
          </div>
        </div>
      </div>
    `;
  }).join("");
}

function getSeverityColor(sev) {
  if (sev === "CRITICAL") return "#ef4444";
  if (sev === "HIGH") return "#f97316";
  if (sev === "MEDIUM") return "#fbbf24";
  return "#64748b";
}

async function quickAcknowledgeAlert(alertId, e) {
  if (e) e.stopPropagation();
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/alerts/${encodeURIComponent(alertId)}/acknowledge`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user: (state.currentUser && state.currentUser.username) || "operator", notes: "Quick acknowledged from feed" }),
    });
    if (res.ok) {
      fetchAlerts();
      fetchAlertStats();
    }
  } catch (err) {
    console.error("Failed to acknowledge alert:", err);
  }
}

async function quickResolveAlert(alertId, e) {
  if (e) e.stopPropagation();
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/alerts/${encodeURIComponent(alertId)}/resolve`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ user: (state.currentUser && state.currentUser.username) || "operator", notes: "Quick resolved from feed" }),
    });
    if (res.ok) {
      fetchAlerts();
      fetchAlertStats();
    }
  } catch (err) {
    console.error("Failed to resolve alert:", err);
  }
}

async function openAlertDetailModal(alertId) {
  if (!elements.alertDetailModal) return;
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/alerts/${encodeURIComponent(alertId)}`);
    if (!res.ok) return;
    const alert = await res.json();
    state.selectedAlert = alert;

    if (elements.alertModalTitle) elements.alertModalTitle.textContent = `ALERT: ${alert.alert_id}`;
    if (elements.alertModalSeverity) {
      elements.alertModalSeverity.className = `alert-pill sev-${alert.severity}`;
      elements.alertModalSeverity.textContent = alert.severity;
    }
    if (elements.alertModalStatus) {
      elements.alertModalStatus.className = `alert-status-pill status-${alert.status}`;
      elements.alertModalStatus.textContent = alert.status;
    }

    if (elements.alertDetId) elements.alertDetId.textContent = alert.alert_id;
    if (elements.alertDetViolation) elements.alertDetViolation.textContent = alert.violation_type;
    if (elements.alertDetCamera) elements.alertDetCamera.textContent = `${alert.camera_name} (${alert.camera_id})`;
    if (elements.alertDetTrack) elements.alertDetTrack.textContent = alert.track_id !== null ? `#${alert.track_id}` : "Unassigned / Scene";
    if (elements.alertDetConf) elements.alertDetConf.textContent = `${(alert.confidence * 100).toFixed(1)}%`;
    if (elements.alertDetOccurrences) elements.alertDetOccurrences.textContent = `${alert.occurrence_count} frames/events`;
    if (elements.alertDetFirst) elements.alertDetFirst.textContent = new Date(alert.first_detected_at * 1000).toLocaleString();
    if (elements.alertDetLast) elements.alertDetLast.textContent = new Date(alert.last_detected_at * 1000).toLocaleString();

    // Evidence image
    if (alert.has_evidence && alert.evidence_image_path) {
      if (elements.alertEvidenceImage) {
        const tokenParam = state.authToken ? `&token=${encodeURIComponent(state.authToken)}` : "";
        elements.alertEvidenceImage.src = `${CONFIG.apiBase}/api/v1/alerts/${encodeURIComponent(alert.alert_id)}/evidence?t=` + Date.now() + tokenParam;
        elements.alertEvidenceImage.style.display = "block";
      }
      if (elements.alertNoEvidenceNotice) elements.alertNoEvidenceNotice.style.display = "none";
      if (elements.alertEvidenceBadge) elements.alertEvidenceBadge.style.display = "block";
    } else {
      if (elements.alertEvidenceImage) elements.alertEvidenceImage.style.display = "none";
      if (elements.alertNoEvidenceNotice) elements.alertNoEvidenceNotice.style.display = "block";
      if (elements.alertEvidenceBadge) elements.alertEvidenceBadge.style.display = "none";
    }

    // Transition History
    if (elements.alertHistoryList) {
      const history = alert.transition_history || [];
      if (history.length === 0) {
        elements.alertHistoryList.innerHTML = `<div style="color: var(--text-muted); font-size: 10px; padding: 8px;">No transition logs.</div>`;
      } else {
        elements.alertHistoryList.innerHTML = history.map((h) => {
          const ts = new Date(h.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
          return `
            <div class="alert-history-item">
              <div style="display: flex; justify-content: space-between; font-weight: 600; color: #fff;">
                <span>${h.from_status ? h.from_status + ' ➔ ' : ''}${h.to_status}</span>
                <span style="font-family: var(--font-mono); color: var(--text-muted);">${ts}</span>
              </div>
              <div style="color: var(--text-secondary); margin-top: 2px;">By: <strong>${escapeHtml(h.user)}</strong></div>
              ${h.notes ? `<div style="color: var(--text-muted); font-style: italic; margin-top: 1px;">"${escapeHtml(h.notes)}"</div>` : ''}
            </div>
          `;
        }).join("");
      }
    }

    // Enable/disable buttons based on current status
    const isResolved = alert.status === "RESOLVED";
    const isDismissed = alert.status === "DISMISSED";
    if (elements.btnModalAck) elements.btnModalAck.disabled = alert.status !== "NEW";
    if (elements.btnModalResolve) elements.btnModalResolve.disabled = isResolved || isDismissed;
    if (elements.btnModalDismiss) elements.btnModalDismiss.disabled = isDismissed;
    if (elements.alertActionNotes) elements.alertActionNotes.value = "";
    if (elements.alertActionFeedback) elements.alertActionFeedback.style.display = "none";

    elements.alertDetailModal.style.display = "flex";
  } catch (err) {
    console.error("Could not open alert modal:", err);
  }
}

function closeAlertDetailModal() {
  if (elements.alertDetailModal) {
    elements.alertDetailModal.style.display = "none";
  }
  state.selectedAlert = null;
}

async function performAlertAction(action) {
  if (!state.selectedAlert) return;
  const alertId = state.selectedAlert.alert_id;
  const user = (elements.alertOperatorName && elements.alertOperatorName.value.trim()) || (state.currentUser && state.currentUser.username) || "operator";
  const notes = (elements.alertActionNotes && elements.alertActionNotes.value.trim()) || "";

  if (action === "dismiss" && !notes) {
    if (elements.alertActionFeedback) {
      elements.alertActionFeedback.textContent = "Please provide a dismissal reason in the notes box.";
      elements.alertActionFeedback.style.color = "#ef4444";
      elements.alertActionFeedback.style.display = "block";
    }
    return;
  }

  let endpoint = `${CONFIG.apiBase}/api/v1/alerts/${encodeURIComponent(alertId)}/${action}`;
  let payload = { user: user };
  if (action === "dismiss") {
    payload.reason = notes;
  } else {
    payload.notes = notes;
  }

  try {
    const res = await authFetch(endpoint, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (res.ok) {
      await openAlertDetailModal(alertId);
      fetchAlerts();
      fetchAlertStats();
      if (elements.alertActionFeedback) {
        elements.alertActionFeedback.textContent = `Alert successfully marked as ${action.toUpperCase()}!`;
        elements.alertActionFeedback.style.color = "#10b981";
        elements.alertActionFeedback.style.display = "block";
      }
    } else {
      const err = await res.json();
      if (elements.alertActionFeedback) {
        elements.alertActionFeedback.textContent = err.detail || "Action failed.";
        elements.alertActionFeedback.style.color = "#ef4444";
        elements.alertActionFeedback.style.display = "block";
      }
    }
  } catch (err) {
    console.error(`Failed to execute ${action} on alert:`, err);
  }
}

window.quickAcknowledgeAlert = quickAcknowledgeAlert;
window.quickResolveAlert = quickResolveAlert;
window.openAlertDetailModal = openAlertDetailModal;
window.closeAlertDetailModal = closeAlertDetailModal;


// -------------------------------------------------------------
// RENDERING FUNCTIONS
// -------------------------------------------------------------
function renderStatus(status) {
  if (!status) return;
  if (elements.cpuBadge) elements.cpuBadge.textContent = status.is_cpu_mode ? "CPU Mode" : "GPU (CUDA)";
  if (elements.fpsBadge) elements.fpsBadge.textContent = `${status.processing_fps.toFixed(1)} FPS`;
  if (elements.latencyBadge) elements.latencyBadge.textContent = `${status.avg_latency_ms.toFixed(1)} ms`;
  if (elements.tracksBadge) elements.tracksBadge.textContent = `Tracks: ${status.active_tracks_count}`;
  if (elements.incidentsBadge) elements.incidentsBadge.textContent = `Incidents: ${status.total_recorded_incidents}`;
}

function renderAssessment(data) {
  if (!data) return;

  renderCanvas(data);
  renderSceneUnderstanding(data);
  renderRiskAndEvents(data);
  renderPPECompliance(data);
  renderBehaviorAndDepth(data);
  // Step 15: render temporal intelligence from inline assessment data
  if (typeof renderTemporalIntelligence === "function") {
    renderTemporalIntelligence(data.entity_profiles || [], data.temporal_events || []);
  }
  // Step 16: sync safety zones overlay with frame resolution
  renderZonesOverlay();
}

function renderCanvas(data) {
  const sw = data.frame_width || 1280;
  const sh = data.frame_height || 720;
  if (elements.canvas.width !== sw || elements.canvas.height !== sh) {
    elements.canvas.width = sw;
    elements.canvas.height = sh;
  }
  const ctx = elements.canvas.getContext("2d");
  const cw = elements.canvas.width;
  const ch = elements.canvas.height;
  const sx = 1.0;
  const sy = 1.0;

  // Clear dark background with faint grid
  ctx.fillStyle = "#0a0e17";
  ctx.fillRect(0, 0, cw, ch);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.04)";
  ctx.lineWidth = 1;
  for (let x = 0; x < cw; x += 40) {
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, ch);
    ctx.stroke();
  }
  for (let y = 0; y < ch; y += 40) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(cw, y);
    ctx.stroke();
  }

  // 1. Draw Zones
  if (data.zone_memberships && data.zone_memberships.length > 0) {
    const zoneMap = {};
    data.zone_memberships.forEach((zm) => {
      zoneMap[zm.zone_id] = zm;
    });
  }

  // 2. Draw Tracked Objects & Trajectories
  const tracks = data.tracks || [];
  tracks.forEach((trk) => {
    const bbox = trk.bbox;
    if (!bbox) return;

    const bx = bbox.x1 * sx;
    const by = bbox.y1 * sy;
    const bw = (bbox.x2 - bbox.x1) * sx;
    const bh = (bbox.y2 - bbox.y1) * sy;

    // Track color (deterministic based on class category)
    const cgroup = (trk.class_group || "").toUpperCase();
    const cname = (trk.class_name || "").toLowerCase();
    let boxColor = "#94a3b8"; // default slate
    if (cgroup === "PERSON" || cname === "person" || cname === "worker") {
      boxColor = "#3b82f6"; // blue
    } else if (cgroup === "VEHICLE" || ["forklift", "industrial vehicle", "forklift truck", "truck", "car", "vehicle", "agv", "van"].includes(cname)) {
      boxColor = "#f59e0b"; // amber
    } else if (cgroup === "MACHINE" || ["machine", "machinery", "robotic arm", "conveyor", "crane", "press"].includes(cname)) {
      boxColor = "#8b5cf6"; // purple
    } else if (cgroup === "INFRASTRUCTURE" || ["pallet", "safety barrier", "electrical cabinet", "guardrail"].includes(cname)) {
      boxColor = "#06b6d4"; // cyan
    } else if (cgroup === "PPE") {
      boxColor = "#10b981"; // emerald
    }

    // Trajectory trail
    if (trk.trajectory && trk.trajectory.length > 1) {
      ctx.beginPath();
      ctx.strokeStyle = boxColor;
      ctx.lineWidth = 1.5;
      trk.trajectory.forEach((pt, i) => {
        const px = pt.cx * sx;
        const py = pt.cy * sy;
        if (i === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    }

    // Bounding Box
    ctx.strokeStyle = boxColor;
    ctx.lineWidth = 2;
    ctx.strokeRect(bx, by, bw, bh);

    // Label badge
    const label = `#${trk.track_id} ${trk.class_name || "obj"} ${(trk.confidence ? trk.confidence.toFixed(2) : "")}`;
    ctx.font = "10px monospace";
    const textWidth = ctx.measureText(label).width;

    ctx.fillStyle = boxColor;
    ctx.fillRect(bx, Math.max(0, by - 16), textWidth + 8, 16);
    ctx.fillStyle = "#ffffff";
    ctx.fillText(label, bx + 4, Math.max(12, by - 4));
  });

  // 3. Draw Scene Graph Relationships (Connecting Lines)
  if (data.scene && data.scene.relationships) {
    const nodeMap = {};
    (data.scene.nodes || []).forEach((n) => {
      nodeMap[n.node_id] = n;
    });

    data.scene.relationships.forEach((rel) => {
      const src = nodeMap[rel.source_node_id];
      const tgt = nodeMap[rel.target_node_id];
      if (src && tgt && src.centroid && tgt.centroid) {
        const x1 = src.centroid[0] * sx;
        const y1 = src.centroid[1] * sy;
        const x2 = tgt.centroid[0] * sx;
        const y2 = tgt.centroid[1] * sy;

        const relType = rel.relation_type;
        const isApproaching = relType === "APPROACHING";
        ctx.strokeStyle = isApproaching ? "#ef4444" : "#f59e0b";
        ctx.lineWidth = isApproaching ? 2 : 1;
        ctx.setLineDash([4, 4]);

        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.lineTo(x2, y2);
        ctx.stroke();
        ctx.setLineDash([]);

        // Midpoint badge
        const mx = (x1 + x2) / 2;
        const my = (y1 + y2) / 2;
        ctx.fillStyle = "rgba(10, 15, 25, 0.85)";
        ctx.fillRect(mx - 30, my - 8, 60, 16);
        ctx.strokeStyle = isApproaching ? "#ef4444" : "#f59e0b";
        ctx.strokeRect(mx - 30, my - 8, 60, 16);
        ctx.fillStyle = "#ffffff";
        ctx.font = "8px monospace";
        ctx.textAlign = "center";
        ctx.fillText(relType, mx, my + 3);
        ctx.textAlign = "start";
      }
    });
  }

  // 4. In-viewport Telemetry Banner
  ctx.fillStyle = "rgba(10, 15, 25, 0.85)";
  ctx.fillRect(10, 10, 260, 48);
  ctx.strokeStyle = "rgba(59, 130, 246, 0.4)";
  ctx.strokeRect(10, 10, 260, 48);

  ctx.fillStyle = "#94a3b8";
  ctx.font = "10px monospace";
  ctx.fillText(`FRAME: ${data.frame_id || 0} | TIME: ${(data.timestamp || 0).toFixed(2)}s`, 18, 26);
  ctx.fillText(`ACTIVE TRACKS: ${data.total_active_tracks || 0} | LATENCY: ${(data.processing_time_ms || 0).toFixed(1)}ms`, 18, 44);
}

function renderSceneUnderstanding(data) {
  if (!elements.sceneRelationshipsList) return;

  const scene = data.scene;
  if (!scene || !scene.relationships || scene.relationships.length === 0) {
    elements.sceneRelationshipsList.innerHTML = `
      <div style="color: var(--text-muted); font-size: 11px; padding: 12px; text-align: center;">
        No active scene relationships detected in current frame.
      </div>`;
  } else {
    elements.sceneRelationshipsList.innerHTML = scene.relationships
      .map((r) => {
        const srcLabel = r.source_node_id.replace("_", " #");
        const tgtLabel = r.target_node_id.replace("_", " #");
        const relType = r.relation_type;
        return `
          <div class="relationship-card">
            <span class="rel-node">${srcLabel}</span>
            <span class="rel-arrow">➔</span>
            <span class="rel-predicate">${relType}</span>
            <span class="rel-arrow">➔</span>
            <span class="rel-node">${tgtLabel}</span>
          </div>`;
      })
      .join("");
  }

  if (elements.entitySummaryList) {
    const s = (scene && scene.summary) ? scene.summary : {};
    const detections = data.detections || [];
    const sourceBreakdown = {};
    detections.forEach(d => {
      const src = d.source_model || "yolo11n";
      sourceBreakdown[src] = (sourceBreakdown[src] || 0) + 1;
    });

    elements.entitySummaryList.innerHTML = `
      <div style="display: grid; grid-template-columns: repeat(4, 1fr); gap: 6px; font-size: 11px;">
        <div style="background: rgba(0,0,0,0.3); padding: 6px; border-radius: 4px; text-align: center;">
          <div style="color: var(--text-muted); font-size: 8px;">WORKERS</div>
          <div style="font-weight: 700; color: #3b82f6;">${s.workers_count || 0}</div>
        </div>
        <div style="background: rgba(0,0,0,0.3); padding: 6px; border-radius: 4px; text-align: center;">
          <div style="color: var(--text-muted); font-size: 8px;">VEHICLES</div>
          <div style="font-weight: 700; color: #f59e0b;">${s.vehicles_count || 0}</div>
        </div>
        <div style="background: rgba(0,0,0,0.3); padding: 6px; border-radius: 4px; text-align: center;">
          <div style="color: var(--text-muted); font-size: 8px;">MACHINERY</div>
          <div style="font-weight: 700; color: #8b5cf6;">${s.machines_count || 0}</div>
        </div>
        <div style="background: rgba(0,0,0,0.3); padding: 6px; border-radius: 4px; text-align: center;">
          <div style="color: var(--text-muted); font-size: 8px;">DETECTIONS</div>
          <div style="font-weight: 700; color: #06b6d4;">${detections.length}</div>
        </div>
      </div>
      <div style="display: flex; gap: 6px; margin-top: 6px; font-size: 9px; flex-wrap: wrap;">
        <span style="background: rgba(59,130,246,0.15); border: 1px solid rgba(59,130,246,0.4); padding: 2px 6px; border-radius: 3px; color: #93c5fd;">
          YOLO11n: ${sourceBreakdown["yolo11n"] || 0}
        </span>
        <span style="background: rgba(245,158,11,0.15); border: 1px solid rgba(245,158,11,0.4); padding: 2px 6px; border-radius: 3px; color: #fcd34d;">
          YOLO-World: ${sourceBreakdown["yolo_world"] || 0}
        </span>
        <span style="background: rgba(16,185,129,0.15); border: 1px solid rgba(16,185,129,0.4); padding: 2px 6px; border-radius: 3px; color: #6ee7b7;">
          SafetyVision: ${sourceBreakdown["safetyvision_ppe"] || 0}
        </span>
      </div>`;
  }
}

function renderActiveSafetyEvent(data) {
  if (!elements.safetyEventContainer) return;

  const events = data.active_events || [];
  const compoundEvent = events.find(
    (e) => (e.metadata && e.metadata.audit_5w) || e.risk_level === "CRITICAL" || e.risk_level === "HIGH"
  );

  if (!compoundEvent && events.length === 0) {
    elements.safetyEventContainer.innerHTML = "";
    return;
  }

  const ev = compoundEvent || events[0];
  const riskTier = ev.risk_level || "MEDIUM";
  const meta = ev.metadata || {};
  const audit = meta.audit_5w || null;

  // Spatial Proximity badge
  let spatialState = meta.spatial_state || "UNKNOWN";
  if (spatialState === "UNKNOWN" && data.scene && data.scene.relationships) {
    const rel = data.scene.relationships.find((r) => r.proximity_state);
    if (rel) spatialState = rel.proximity_state;
  }

  // Zone info
  let zoneName = meta.zone_id || "Facility Floor";
  if (data.zone_memberships && data.zone_memberships.length > 0) {
    const activeZone = data.zone_memberships.find((z) => z.is_inside);
    if (activeZone) zoneName = activeZone.zone_id || zoneName;
  }

  // Involved relationship
  let relDisplay = "Personnel & Asset Interaction";
  if (data.scene && data.scene.relationships && data.scene.relationships.length > 0) {
    const firstRel = data.scene.relationships[0];
    relDisplay = `${firstRel.source_node_id.replace("_", " #")} ➔ [${firstRel.relation_type}] ➔ ${firstRel.target_node_id.replace("_", " #")}`;
  } else if (ev.involved_track_ids && ev.involved_track_ids.length >= 2) {
    relDisplay = `Worker #${ev.involved_track_ids[0]} ➔ [INTERACTION] ➔ Entity #${ev.involved_track_ids[1]}`;
  }

  // Contributing factors checklist
  const factors = ev.risk_factors || [];
  const factorTypes = factors.map((f) => (f.factor_type || "").toUpperCase());
  const hasProximity =
    factorTypes.some((t) => t.includes("PROXIMITY")) ||
    spatialState === "VERY_NEAR" ||
    spatialState === "NEAR";
  const hasApproaching = factorTypes.some((t) => t.includes("APPROACHING"));
  const hasZone =
    factorTypes.some((t) => t.includes("ZONE")) ||
    (data.zone_memberships && data.zone_memberships.some((z) => z.is_inside));
  const hasPpe =
    factorTypes.some((t) => t.includes("PPE")) ||
    (data.ppe_assessments && data.ppe_assessments.some((p) => !p.compliant));

  const whatText = audit ? audit.what : ev.explanation || ev.event_type;
  const whoText = audit ? audit.who : `Track #${(ev.involved_track_ids || []).join(", #")}`;
  const whereText = audit ? audit.where : zoneName;
  const whyText = audit ? audit.why : ev.deterministic_explanation || ev.explanation;

  elements.safetyEventContainer.innerHTML = `
    <div class="safety-event-card ${riskTier}">
      <div class="safety-event-header">
        <div class="safety-event-title">
          <span>⚡ INDUSTRIAL SAFETY EVENT</span>
          <span class="risk-badge badge-${riskTier}">${riskTier}</span>
        </div>
        <span class="spatial-badge spatial-badge-${spatialState}">${spatialState.replace(/_/g, " ")}</span>
      </div>

      <div class="safety-rel-flow">
        <span style="color: var(--text-muted); font-size: 10px;">RELATION:</span>
        <strong style="color: #93c5fd;">${relDisplay}</strong>
      </div>

      <div class="safety-factors-checklist">
        <div class="checklist-item ${hasProximity ? "active" : ""}">
          <span>${hasProximity ? "☑" : "☐"}</span> Proximity: ${spatialState.replace(/_/g, " ")}
        </div>
        <div class="checklist-item ${hasApproaching ? "active" : ""}">
          <span>${hasApproaching ? "☑" : "☐"}</span> Approaching Kinematics
        </div>
        <div class="checklist-item ${hasZone ? "active" : ""}">
          <span>${hasZone ? "☑" : "☐"}</span> Restricted Area (${zoneName})
        </div>
        <div class="checklist-item ${hasPpe ? "active" : ""}">
          <span>${hasPpe ? "☑" : "☐"}</span> PPE Non-Compliance
        </div>
      </div>

      <div class="audit-5w-grid">
        <div class="audit-field">
          <span class="audit-label">WHAT</span>
          <span class="audit-val">${whatText}</span>
        </div>
        <div class="audit-field">
          <span class="audit-label">WHO</span>
          <span class="audit-val">${whoText}</span>
        </div>
        <div class="audit-field">
          <span class="audit-label">WHERE</span>
          <span class="audit-val">${whereText}</span>
        </div>
        <div class="audit-field">
          <span class="audit-label">WHY</span>
          <span class="audit-val">${whyText}</span>
        </div>
      </div>
    </div>
  `;
}

function renderRiskAndEvents(data) {
  renderActiveSafetyEvent(data);
  if (!elements.riskContainer) return;

  const events = data.active_events || [];
  const riskTier = data.highest_risk_level || "INFO";
  const riskScore = data.highest_risk_score || 0.0;

  let riskHtml = `
    <div class="risk-card ${riskTier}">
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <span class="risk-badge badge-${riskTier}">${riskTier} RISK TIER</span>
        <span style="font-family: var(--font-mono); font-weight: 700; font-size: 12px;">SCORE: ${riskScore.toFixed(1)}</span>
      </div>
  `;

  if (events.length === 0) {
    riskHtml += `
      <div style="color: var(--text-muted); font-size: 11px; padding: 6px 0;">
        Normal operational state. No active safety events confirmed.
      </div>
    `;
  } else {
    riskHtml += `<div style="display: flex; flex-direction: column; gap: 8px; margin-top: 4px;">`;
    events.forEach((ev) => {
      const factorsHtml = (ev.risk_factors || [])
        .map((f) => `<div class="factor-item">${f.explanation || f.factor_type}</div>`)
        .join("");

      riskHtml += `
        <div style="border-top: 1px solid var(--border-color); padding-top: 6px;">
          <div style="font-weight: 700; color: var(--text-primary); font-size: 11px;">
            ${ev.event_type.replace(/_/g, " ")}
          </div>
          <div class="risk-explanation">"${ev.explanation}"</div>
          ${factorsHtml ? `<div class="factors-list" style="margin-top: 4px;">${factorsHtml}</div>` : ""}
        </div>
      `;
    });
    riskHtml += `</div>`;
  }
  riskHtml += `</div>`;
  elements.riskContainer.innerHTML = riskHtml;

  // Early-Warning Indicators
  if (elements.warningContainer) {
    if (state.isSingleFrame) {
      elements.warningContainer.innerHTML = `
        <div class="single-frame-notice">
          <strong>Single-Frame Analysis:</strong> Trajectory forecasting and predictive safety escalation require multi-frame temporal video observations.
        </div>`;
    } else {
      const warnings = data.active_early_warnings || [];
      if (warnings.length === 0) {
        elements.warningContainer.innerHTML = `
          <div style="color: var(--text-muted); font-size: 11px; padding: 8px; text-align: center;">
            No predictive early warnings currently active.
          </div>`;
      } else {
        elements.warningContainer.innerHTML = warnings
          .map((w) => `
            <div style="background: rgba(249, 115, 22, 0.08); border: 1px solid rgba(249, 115, 22, 0.4); border-left: 3px solid #f97316; padding: 6px 10px; border-radius: 4px;">
              <div style="display: flex; justify-content: space-between; font-size: 10px; font-weight: 700; color: #f97316;">
                <span>⚠ ${w.indicator_type.replace(/_/g, " ")}</span>
                <span>[${w.severity}]</span>
              </div>
              <div style="font-size: 10px; color: var(--text-secondary); margin-top: 2px;">
                ${w.explanation}
              </div>
            </div>
          `)
          .join("");
      }
    }
  }
}

function renderPPECompliance(data) {
  if (!elements.ppeContainer) return;

  const inventories = data.worker_inventories || [];
  if (inventories.length === 0) {
    elements.ppeContainer.innerHTML = `
      <div style="color: var(--text-muted); font-size: 11px; padding: 12px; text-align: center;">
        No active workers detected in scene.
      </div>`;
    return;
  }

  elements.ppeContainer.innerHTML = inventories
    .map((inv) => {
      const statusClass = inv.compliance_status || "UNKNOWN";
      const ppeMap = inv.ppe_status || {};
      const items = ["Hardhat", "Safety Vest", "Gloves", "Goggles"];

      const slotsHtml = items
        .map((item) => {
          const state = ppeMap[item] || "UNKNOWN";
          return `
            <div class="ppe-slot ${state}">
              <span style="font-weight: 600;">${item}</span>
              <span>${state}</span>
            </div>`;
        })
        .join("");

      return `
        <div class="worker-ppe-card">
          <div class="worker-header">
            <span>Worker #${inv.track_id}</span>
            <span class="risk-badge badge-${statusClass === "COMPLIANT" ? "LOW" : statusClass === "NON_COMPLIANT" ? "CRITICAL" : "INFO"}">
              ${statusClass}
            </span>
          </div>
          <div class="ppe-grid">${slotsHtml}</div>
          <div style="font-size: 10px; color: var(--text-muted); font-style: italic;">
            ${inv.explanation || ""}
          </div>
        </div>`;
    })
    .join("");
}

function renderBehaviorAndDepth(data) {
  if (elements.behaviorContainer) {
    if (state.isSingleFrame) {
      elements.behaviorContainer.innerHTML = `
        <div class="single-frame-notice">
          <strong>Single-Frame Analysis:</strong> Velocity, trajectories, and temporal behavior are not available for single-frame images. Movement is not fabricated.
        </div>`;
    } else {
      const behaviors = data.behavior_states || [];
      if (behaviors.length === 0) {
        elements.behaviorContainer.innerHTML = `
          <div style="color: var(--text-muted); font-size: 11px; padding: 8px; text-align: center;">
            No behavior observations recorded.
          </div>`;
      } else {
        elements.behaviorContainer.innerHTML = behaviors
          .map((b) => `
            <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); padding: 6px 10px; border-radius: 4px; display: flex; justify-content: space-between; font-size: 11px;">
              <span>Track #${b.track_id}</span>
              <span style="font-family: var(--font-mono); font-weight: 600; color: ${b.primary_behavior === "RAPID_MOVEMENT" ? "#ef4444" : "#60a5fa"};">
                ${b.primary_behavior}
              </span>
              <span style="color: var(--text-muted); font-size: 10px;">
                ${(b.image_speed_px_per_s || 0).toFixed(0)} px/s
              </span>
            </div>
          `)
          .join("");
      }
    }
  }

  if (elements.depthContainer) {
    elements.depthContainer.innerHTML = `
      <div style="background: rgba(0,0,0,0.3); border: 1px solid var(--border-color); border-radius: 4px; padding: 8px; font-size: 11px;">
        <div style="color: #94a3b8; font-weight: 600; margin-bottom: 4px;">Perception: Monocular Relative Depth</div>
        <div style="font-size: 10px; color: var(--text-muted); line-height: 1.4;">
          Depth values operate on inverse relative disparity. Higher values represent closer proximity to camera.
          <span style="color: #fbbf24;">(Non-metric: calibrated physical meters not claimed)</span>.
        </div>
      </div>`;
  }
}

function renderTemporalIntelligence(profiles, events) {
  const container = elements.temporalEntityProfiles || document.getElementById("temporalEntityProfiles");
  const timelineEl = elements.temporalTimeline || document.getElementById("temporalTimeline");

  if (!container) return;

  if (state.isSingleFrame) {
    container.innerHTML = `
      <div class="single-frame-notice">
        <strong>Single-Frame Analysis:</strong> Entity temporal profiles and state transition timelines require multi-frame temporal video observations.
      </div>`;
    if (timelineEl) {
      timelineEl.innerHTML = `
        <div style="color: var(--text-muted); font-size: 11px; padding: 8px; text-align: center;">
          No temporal transitions recorded for single-frame analysis.
        </div>`;
    }
    return;
  }

  // Multi-frame video profiles
  const profileList = Array.isArray(profiles) ? profiles : [];
  if (profileList.length === 0) {
    container.innerHTML = `
      <div style="color: var(--text-muted); font-size: 11px; padding: 8px; text-align: center;">
        Awaiting temporal intelligence data...
      </div>`;
  } else {
    container.innerHTML = profileList.map((p) => {
      const beh = (p.current_behavior || "UNKNOWN").toUpperCase();
      let behClass = "beh-unknown";
      if (beh.includes("MOVING")) behClass = "beh-moving";
      else if (beh.includes("STATIONARY")) behClass = "beh-stationary";
      else if (beh.includes("RAPID")) behClass = "beh-rapid";
      else if (beh.includes("FALL")) behClass = "beh-fall";
      else if (beh.includes("PROLONGED")) behClass = "beh-prolonged";

      const prox = (p.current_spatial_state || "UNKNOWN").toUpperCase();
      let proxClass = "prox-unknown";
      if (prox === "VERY_NEAR") proxClass = "prox-very-near";
      else if (prox === "NEAR") proxClass = "prox-near";
      else if (prox === "MODERATE") proxClass = "prox-moderate";
      else if (prox === "FAR") proxClass = "prox-far";

      const durStr = p.behavior_duration_s != null ? `${p.behavior_duration_s.toFixed(1)}s` : "--";
      const spdStr = p.image_speed_px_per_s != null ? `${p.image_speed_px_per_s.toFixed(0)} px/s` : "--";

      return `
        <div class="temporal-entity-card">
          <div class="temporal-entity-header">
            <span class="temporal-entity-name">Track #${p.track_id} (${p.class_name || "entity"})</span>
            <span class="temporal-behavior-badge ${behClass}">${beh}</span>
          </div>
          <div class="temporal-info-row">
            <div class="temporal-info-item">
              <span class="temporal-info-label">Duration</span>
              <span class="temporal-info-val">${durStr}</span>
            </div>
            <div class="temporal-info-item">
              <span class="temporal-info-label">Speed</span>
              <span class="temporal-info-val highlight">${spdStr}</span>
            </div>
            <div class="temporal-info-item">
              <span class="temporal-info-label">Proximity</span>
              <span class="temporal-proximity-badge ${proxClass}">${prox}</span>
            </div>
          </div>
        </div>`;
    }).join("");
  }

  // Render events in timeline if available
  if (timelineEl && Array.isArray(events) && events.length > 0) {
    renderTemporalTimeline(events);
  }
}

function renderTemporalTimeline(timeline) {
  const timelineEl = elements.temporalTimeline || document.getElementById("temporalTimeline");
  if (!timelineEl) return;

  const entries = Array.isArray(timeline) ? timeline : [];
  if (entries.length === 0) {
    timelineEl.innerHTML = `
      <div style="color: var(--text-muted); font-size: 11px; padding: 8px; text-align: center;">
        No transitions recorded yet.
      </div>`;
    return;
  }

  timelineEl.innerHTML = entries.map((e) => {
    const cat = (e.category || "").toLowerCase();
    let dotClass = "cat-detected";
    if (cat.includes("behavior")) dotClass = "cat-behavior";
    else if (cat.includes("spatial")) dotClass = "cat-spatial";
    else if (cat.includes("escalation") || cat.includes("risk-up")) dotClass = "cat-risk-up";
    else if (cat.includes("decay") || cat.includes("risk-down")) dotClass = "cat-risk-down";
    else if (cat.includes("warning")) dotClass = "cat-warning";
    else if (cat.includes("incident")) dotClass = "cat-incident";

    const ts = e.timestamp != null ? `${e.timestamp.toFixed(1)}s` : "--";
    const label = e.label || e.description || e.category || "Transition";

    return `
      <div class="temporal-entity-timeline-entry">
        <span class="tel-ts">${ts}</span>
        <span class="tel-dot ${dotClass}"></span>
        <span class="tel-label">${label}</span>
      </div>`;
  }).join("");
}

function renderIncidents(incidents) {
  if (!elements.incidentTableBody) return;

  if (!incidents || incidents.length === 0) {
    elements.incidentTableBody.innerHTML = `
      <tr>
        <td colspan="6" style="text-align: center; color: var(--text-muted); padding: 16px;">
          No recorded safety incidents.
        </td>
      </tr>`;
    return;
  }

  elements.incidentTableBody.innerHTML = incidents
    .map((inc) => {
      const riskClass = inc.risk_level || "INFO";
      const tracksStr = (inc.involved_track_ids || []).map((t) => `#${t}`).join(", ") || "N/A";
      const timeStr = `${inc.timestamp.toFixed(2)}s (Frame ${inc.frame_id})`;

      return `
        <tr>
          <td style="font-family: var(--font-mono); font-size: 10px;">${timeStr}</td>
          <td><span class="risk-badge badge-${riskClass}">${riskClass}</span></td>
          <td style="font-weight: 600;">${inc.event_type.replace(/_/g, " ")}</td>
          <td>${tracksStr}</td>
          <td style="max-width: 280px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;" title="${inc.explanation}">
            ${inc.explanation}
          </td>
          <td>
            <button class="btn" style="padding: 2px 8px; font-size: 10px;" onclick="openIncidentModal('${inc.incident_id}')">
              Evidence
            </button>
          </td>
        </tr>`;
    })
    .join("");
}

// -------------------------------------------------------------
// STEP 17: EXPLAINABILITY, EVIDENCE & ANALYTICS
// -------------------------------------------------------------
function escapeHtml(str) {
  if (str == null) return "";
  return String(str)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#039;");
}

window.openIncidentModal = async function (incidentId) {
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/incidents/${incidentId}/explanation`);
    if (!res.ok) throw new Error("Could not retrieve incident explainability package");
    const expl = await res.json();
    state.activeExplanation = expl;
    state.selectedIncident = expl.incident;

    const inc = expl.incident;
    if (elements.modalTitle) {
      elements.modalTitle.textContent = `INCIDENT AUDIT: ${inc.incident_id} [${inc.risk_level} - ${inc.event_type.replace(/_/g, " ")}]`;
    }

    // Evidence Snapshot display
    if (expl.has_snapshot) {
      elements.modalEvidenceImage.src = `${CONFIG.apiBase}/api/v1/incidents/${incidentId}/evidence?t=${Date.now()}`;
      elements.modalEvidenceImage.style.display = "block";
      elements.noEvidenceNotice.style.display = "none";
      if (elements.modalEvidenceBadge) {
        elements.modalEvidenceBadge.innerHTML = '<span class="badge" style="background:#065f46;color:#34d399;font-weight:600;">Snapshot Available</span>';
      }
    } else {
      elements.modalEvidenceImage.style.display = "none";
      elements.noEvidenceNotice.style.display = "block";
      elements.noEvidenceNotice.textContent = "Metadata-Only Record: No image snapshot was captured for this event.";
      if (elements.modalEvidenceBadge) {
        elements.modalEvidenceBadge.innerHTML = '<span class="badge" style="background:#78350f;color:#fcd34d;font-weight:600;">Metadata Only</span>';
      }
    }

    // Five-W Structured Causation
    if (elements.fwWho) elements.fwWho.textContent = expl.five_w.who || "Unassigned Monitored Actor";
    if (elements.fwWhat) elements.fwWhat.textContent = expl.five_w.what || "Safety Event";
    if (elements.fwWhere) elements.fwWhere.textContent = expl.five_w.where || "Operational Monitored Zone";
    if (elements.fwWhen) elements.fwWhen.textContent = expl.five_w.when || "Recorded Timestamp";
    if (elements.fwWhyList) {
      const whyItems = expl.five_w.why || [];
      elements.fwWhyList.innerHTML = whyItems.length > 0
        ? whyItems.map((w) => `<li>${escapeHtml(w)}</li>`).join("")
        : "<li>No causal factors logged.</li>";
    }

    // Risk Score Breakdown Table
    if (elements.riskBreakdownBody) {
      let rows = "";
      (expl.risk_breakdown.base_contributions || []).forEach((item) => {
        rows += `
          <tr>
            <td style="font-weight:600; color:#cbd5e1;">${escapeHtml(item.factor_type.replace(/_/g, " "))}</td>
            <td style="font-family:var(--font-mono); color:#38bdf8;">+${item.score_contribution.toFixed(1)}</td>
            <td style="color:var(--text-secondary); font-size:10px;">${escapeHtml(item.explanation)}</td>
          </tr>`;
      });
      (expl.risk_breakdown.escalation_penalties || []).forEach((item) => {
        rows += `
          <tr style="background: rgba(239, 68, 68, 0.08);">
            <td style="font-weight:600; color:#f87171;">${escapeHtml(item.factor_type.replace(/_/g, " "))}</td>
            <td style="font-family:var(--font-mono); color:#f87171;">+${item.score_contribution.toFixed(1)}</td>
            <td style="color:#fca5a5; font-size:10px;">${escapeHtml(item.explanation)}</td>
          </tr>`;
      });
      if (!rows) {
        rows = `<tr><td colspan="3" style="text-align:center; color:var(--text-muted); padding:8px;">No risk factor breakdown available.</td></tr>`;
      }
      elements.riskBreakdownBody.innerHTML = rows;
    }

    if (elements.riskBreakdownFinal) {
      elements.riskBreakdownFinal.innerHTML = `
        <span>Final Calculated Risk: <strong style="color:#f87171;">${expl.risk_breakdown.final_calculated_score.toFixed(1)} / 100</strong></span>
        <span style="color:var(--text-muted); font-size:10px;">Classification Tier: <strong style="color:#38bdf8;">${inc.risk_level}</strong></span>
      `;
    }

    // Incident Timeline
    if (elements.modalIncidentTimeline) {
      const timelineEntries = expl.timeline || [];
      if (timelineEntries.length === 0) {
        elements.modalIncidentTimeline.innerHTML = '<div style="color:var(--text-muted); font-size:11px; padding:8px;">No temporal progression logged around this event window.</div>';
      } else {
        elements.modalIncidentTimeline.innerHTML = timelineEntries
          .map((entry) => {
            const phaseBadgeColor = entry.phase === "BEFORE" ? "#38bdf8" : entry.phase === "EVENT" ? "#ef4444" : "#a855f7";
            return `
              <div class="modal-timeline-item" style="border-left: 3px solid ${phaseBadgeColor}; margin-bottom: 8px; padding-left: 8px; background: rgba(255,255,255,0.02); border-radius: 2px;">
                <div style="display:flex; justify-content:space-between; align-items:center; font-size:10px; font-family:var(--font-mono); margin-bottom: 2px;">
                  <span style="font-weight:700; color:${phaseBadgeColor};">${entry.phase} [${entry.timestamp.toFixed(2)}s / Frame ${entry.frame_id}]</span>
                  <span style="color:var(--text-muted);">Risk: ${entry.risk_score.toFixed(1)}</span>
                </div>
                <div style="font-size:11px; color:#e2e8f0;">${escapeHtml(entry.summary)}</div>
              </div>`;
          })
          .join("");
      }
    }

    // Risk Trend Trajectory Sparkline
    renderRiskTrendChart(expl.risk_trend, inc.timestamp);

    // Scene Graph Relationships & Preceding Early Warnings
    if (elements.modalSceneAndWarnings) {
      const rels = inc.scene_relationships || [];
      const warnings = inc.early_warnings || [];
      let html = "";
      if (rels.length > 0) {
        html += '<div style="margin-bottom:8px;"><strong style="color:#94a3b8; font-size:10px; text-transform:uppercase;">Scene Spatial Relationships:</strong>';
        html += rels
          .map(
            (r) =>
              `<div style="font-size:11px; color:#cbd5e1; margin-top:2px;">• Track #${r.subject_track_id} ${escapeHtml(r.predicate)} Track #${r.object_track_id} (${escapeHtml(r.spatial_relation || "")}, distance: ${(r.distance_px || 0).toFixed(0)}px)</div>`
          )
          .join("");
        html += "</div>";
      }
      if (warnings.length > 0) {
        html += '<div><strong style="color:#f59e0b; font-size:10px; text-transform:uppercase;">Preceding Trajectory Warnings:</strong>';
        html += warnings
          .map(
            (w) =>
              `<div style="font-size:11px; color:#fde68a; margin-top:2px;">⚠ [${escapeHtml(w.indicator_type || "")}] ${escapeHtml(w.explanation || "")} (Horizon: ${(w.predicted_time_horizon_s || 0).toFixed(1)}s)</div>`
          )
          .join("");
        html += "</div>";
      }
      if (!html) {
        html = '<div style="color:var(--text-muted); font-size:11px;">No scene spatial relations or early-warning indicators recorded.</div>';
      }
      elements.modalSceneAndWarnings.innerHTML = html;
    }

    elements.evidenceModal.classList.add("open");
  } catch (err) {
    alert(`Could not load incident explanation: ${err.message}`);
  }
};

function closeModal() {
  if (elements.evidenceModal) {
    elements.evidenceModal.classList.remove("open");
  }
}

async function downloadEvidencePackage() {
  if (!state.selectedIncident) {
    alert("No incident selected for evidence package download.");
    return;
  }
  const incidentId = state.selectedIncident.incident_id;
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/incidents/${incidentId}/package`);
    if (!res.ok) throw new Error("Failed to export evidence package");
    const data = await res.json();
    const blob = new Blob([JSON.stringify(data, null, 2)], { type: "application/json" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = `incident_${incidentId}_package.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  } catch (err) {
    alert(`Could not download evidence package: ${err.message}`);
  }
}

function renderRiskTrendChart(points, eventTimestamp) {
  const canvas = elements.modalRiskTrendCanvas;
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;

  ctx.clearRect(0, 0, w, h);

  if (!points || points.length < 2) {
    if (elements.riskTrendStatus) elements.riskTrendStatus.textContent = "Insufficient Data";
    if (elements.riskTrendNotice) {
      elements.riskTrendNotice.textContent = "Insufficient temporal data for trend trajectory (< 2 data points).";
    }
    ctx.fillStyle = "rgba(148, 163, 184, 0.4)";
    ctx.font = "11px Inter, sans-serif";
    ctx.textAlign = "center";
    ctx.fillText("Insufficient temporal points (< 2) for trajectory curve", w / 2, h / 2);
    return;
  }

  const tVals = points.map((p) => p.timestamp);
  const sVals = points.map((p) => p.risk_score);
  const minT = Math.min(...tVals);
  const maxT = Math.max(...tVals);
  const maxS = Math.max(100, ...sVals);
  const minS = 0;

  const tSpan = maxT - minT || 1;
  const sSpan = maxS - minS || 1;

  // Background and Grid
  ctx.fillStyle = "#0c121e";
  ctx.fillRect(0, 0, w, h);

  ctx.strokeStyle = "rgba(255, 255, 255, 0.05)";
  ctx.lineWidth = 1;
  for (let y = 0; y <= h; y += h / 4) {
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }

  // Margins
  const padX = 24;
  const padY = 16;
  const chartW = w - padX * 2;
  const chartH = h - padY * 2;

  function toX(t) {
    return padX + ((t - minT) / tSpan) * chartW;
  }
  function toY(s) {
    return padY + chartH - ((s - minS) / sSpan) * chartH;
  }

  // Gradient area under curve
  const grad = ctx.createLinearGradient(0, padY, 0, padY + chartH);
  grad.addColorStop(0, "rgba(239, 68, 68, 0.3)");
  grad.addColorStop(1, "rgba(239, 68, 68, 0.0)");

  ctx.beginPath();
  ctx.moveTo(toX(points[0].timestamp), padY + chartH);
  points.forEach((p) => {
    ctx.lineTo(toX(p.timestamp), toY(p.risk_score));
  });
  ctx.lineTo(toX(points[points.length - 1].timestamp), padY + chartH);
  ctx.closePath();
  ctx.fillStyle = grad;
  ctx.fill();

  // Trajectory line
  ctx.beginPath();
  ctx.strokeStyle = "#f87171";
  ctx.lineWidth = 2;
  points.forEach((p, i) => {
    const x = toX(p.timestamp);
    const y = toY(p.risk_score);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Nodes
  points.forEach((p) => {
    const x = toX(p.timestamp);
    const y = toY(p.risk_score);
    const isEvent = Math.abs(p.timestamp - eventTimestamp) < 0.15;

    ctx.beginPath();
    ctx.arc(x, y, isEvent ? 5 : 3, 0, Math.PI * 2);
    ctx.fillStyle = isEvent ? "#ef4444" : "#38bdf8";
    ctx.fill();
    ctx.strokeStyle = "#ffffff";
    ctx.lineWidth = 1;
    ctx.stroke();
  });

  // Trend direction
  const firstScore = points[0].risk_score;
  const lastScore = points[points.length - 1].risk_score;
  const diff = lastScore - firstScore;
  let statusText = "Stable Trajectory";
  if (diff > 4) statusText = `Escalating (+${diff.toFixed(1)})`;
  else if (diff < -4) statusText = `De-escalating (${diff.toFixed(1)})`;

  if (elements.riskTrendStatus) elements.riskTrendStatus.textContent = statusText;
  if (elements.riskTrendNotice) {
    elements.riskTrendNotice.textContent = `Monitored across ${points.length} frames (${minT.toFixed(2)}s to ${maxT.toFixed(2)}s). Red node indicates event frame.`;
  }
}

async function fetchAnalytics() {
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/analytics/summary`);
    if (!res.ok) throw new Error("Failed to fetch analytics");
    const data = await res.json();
    state.analyticsSummary = data;
    renderAnalytics(data);
  } catch (err) {
    console.warn("Could not fetch analytics:", err);
  }
}

function renderAnalytics(data) {
  if (!data) return;
  if (elements.kpiTotalIncidents) elements.kpiTotalIncidents.textContent = data.total_incidents;
  if (elements.kpiCriticalIncidents) elements.kpiCriticalIncidents.textContent = data.critical_incidents;
  if (elements.kpiZoneIntrusions) elements.kpiZoneIntrusions.textContent = data.zone_intrusions;
  if (elements.kpiPpeViolations) elements.kpiPpeViolations.textContent = data.ppe_violations;
  if (elements.kpiAvgDuration) elements.kpiAvgDuration.textContent = `${data.avg_incident_duration_s.toFixed(1)}s`;
  if (elements.kpiActiveIncidents) elements.kpiActiveIncidents.textContent = data.active_incidents;
  if (elements.kpiProximityEvents) elements.kpiProximityEvents.textContent = data.proximity_events;

  // Event Type Distribution
  if (elements.analyticsEventTypeDist) {
    const dist = data.event_type_distribution || {};
    const keys = Object.keys(dist);
    const total = data.total_incidents || 1;
    if (keys.length === 0) {
      elements.analyticsEventTypeDist.innerHTML = '<div style="color:var(--text-muted); font-size:12px; padding:12px;">No safety events recorded in distribution.</div>';
    } else {
      elements.analyticsEventTypeDist.innerHTML = keys
        .map((k) => {
          const count = dist[k];
          const pct = Math.min(100, Math.round((count / total) * 100));
          return `
            <div class="analytics-bar-row">
              <span class="analytics-bar-label">${escapeHtml(k.replace(/_/g, " "))}</span>
              <div class="analytics-bar-track">
                <div class="analytics-bar-fill" style="width: ${pct}%;"></div>
              </div>
              <span class="analytics-bar-val">${count} (${pct}%)</span>
            </div>`;
        })
        .join("");
    }
  }

  // Active Safety Zones
  if (elements.analyticsActiveZonesList) {
    const zones = data.active_zones || [];
    if (zones.length === 0) {
      elements.analyticsActiveZonesList.innerHTML = '<div style="color:var(--text-muted); font-size:12px; padding:12px;">No active safety zones configured or recorded.</div>';
    } else {
      elements.analyticsActiveZonesList.innerHTML = zones
        .map(
          (z) => `
          <div style="background: rgba(255,255,255,0.03); border: 1px solid var(--border-color); border-radius: 4px; padding: 10px 14px; margin-bottom: 8px; display: flex; justify-content: space-between; align-items: center;">
            <div>
              <div style="font-weight: 600; color: #f1f5f9; font-size: 13px;">${escapeHtml(z.name || z.zone_id)}</div>
              <div style="font-size: 11px; color: var(--text-muted); margin-top: 2px;">Type: <span style="text-transform: capitalize;">${escapeHtml(z.zone_type || "restricted")}</span> | Max Dwell: ${z.max_dwell_seconds != null ? z.max_dwell_seconds + "s" : "N/A"}</div>
            </div>
            <div style="text-align: right;">
              <div style="font-family: var(--font-mono); font-weight: 700; color: ${z.incident_count > 0 ? "#ef4444" : "#10b981"}; font-size: 14px;">
                ${z.incident_count} Incidents
              </div>
              <div style="font-size: 10px; color: var(--text-muted);">Risk: ${escapeHtml(z.risk_tier || "LOW")}</div>
            </div>
          </div>
        `
        )
        .join("");
    }
  }
}

// -------------------------------------------------------------
// STEP 16: INTERACTIVE SAFETY ZONES & SCENE CONFIGURATION
// -------------------------------------------------------------
function initZonesModule() {
  if (elements.startDrawZoneBtn) {
    elements.startDrawZoneBtn.addEventListener("click", () => {
      startDrawingZone();
    });
  }

  if (elements.cancelDrawBtn) {
    elements.cancelDrawBtn.addEventListener("click", () => {
      cancelDrawingZone();
    });
  }

  if (elements.clearPointsBtn) {
    elements.clearPointsBtn.addEventListener("click", () => {
      clearDrawingPoints();
    });
  }

  if (elements.saveZoneBtn) {
    elements.saveZoneBtn.addEventListener("click", () => {
      saveZone();
    });
  }

  if (elements.zoneOverlayCanvas) {
    // Vertex placement on click
    elements.zoneOverlayCanvas.addEventListener("click", (e) => {
      if (!state.isDrawingZone) return;
      const [origX, origY] = displayToOriginal(e.clientX, e.clientY);
      state.currentZonePoints.push([origX, origY]);
      updatePointBadge();
      renderZonesOverlay();
    });

    // Real-time rubberband guideline on mousemove
    elements.zoneOverlayCanvas.addEventListener("mousemove", (e) => {
      if (!state.isDrawingZone || state.currentZonePoints.length === 0) return;
      state.cursorPos = [e.clientX, e.clientY];
      renderZonesOverlay();
    });

    elements.zoneOverlayCanvas.addEventListener("mouseleave", () => {
      if (state.cursorPos) {
        state.cursorPos = null;
        renderZonesOverlay();
      }
    });
  }

  window.addEventListener("resize", () => {
    renderZonesOverlay();
  });

  fetchZones();
}

function getZoneColor(zoneType) {
  switch ((zoneType || "").toLowerCase()) {
    case "restricted":
      return [239, 68, 68];
    case "hazard":
      return [245, 158, 11];
    case "machine":
      return [168, 85, 247];
    case "safety":
      return [16, 185, 129];
    case "monitored":
      return [56, 189, 248];
    case "safe":
      return [34, 197, 94];
    default:
      return [59, 130, 246];
  }
}

function getActiveMediaDimensions() {
  let targetEl = elements.canvas;
  let origW = elements.canvas ? elements.canvas.width || 1280 : 1280;
  let origH = elements.canvas ? elements.canvas.height || 720 : 720;

  if (elements.viewportVideo && elements.viewportVideo.style.display !== "none" && elements.viewportVideo.videoWidth > 0) {
    targetEl = elements.viewportVideo;
    origW = elements.viewportVideo.videoWidth;
    origH = elements.viewportVideo.videoHeight;
  } else if (elements.viewportImage && elements.viewportImage.style.display !== "none") {
    targetEl = elements.viewportImage;
    if (elements.viewportImage.naturalWidth > 0) {
      origW = elements.viewportImage.naturalWidth;
      origH = elements.viewportImage.naturalHeight;
    } else if (state.currentAssessment && state.currentAssessment.frame_width) {
      origW = state.currentAssessment.frame_width;
      origH = state.currentAssessment.frame_height;
    }
  } else if (elements.viewportLiveStream && elements.viewportLiveStream.style.display !== "none" && elements.viewportLiveStream.naturalWidth > 0) {
    targetEl = elements.viewportLiveStream;
    origW = elements.viewportLiveStream.naturalWidth;
    origH = elements.viewportLiveStream.naturalHeight;
  } else if (state.currentAssessment && state.currentAssessment.frame_width && state.currentAssessment.frame_height) {
    origW = state.currentAssessment.frame_width;
    origH = state.currentAssessment.frame_height;
  }

  return { targetEl, origW, origH };
}

function getViewportTransform() {
  const { targetEl, origW, origH } = getActiveMediaDimensions();
  const container = elements.viewportContainer;
  const cW = container ? container.clientWidth : origW;
  const cH = container ? container.clientHeight : origH;

  const scale = Math.min(cW / Math.max(1, origW), cH / Math.max(1, origH));
  const dispW = Math.max(1, Math.round(origW * scale));
  const dispH = Math.max(1, Math.round(origH * scale));
  const offX = Math.round((cW - dispW) / 2);
  const offY = Math.round((cH - dispH) / 2);

  return { origW, origH, cW, cH, dispW, dispH, offX, offY, scale, targetEl };
}

function fitViewportMedia() {
  const container = elements.viewportContainer;
  if (!container) return;

  const { targetEl, origW, origH, cW, cH, dispW, dispH, offX, offY, scale } = getViewportTransform();
  if (!targetEl || origW <= 0 || origH <= 0 || cW <= 0 || cH <= 0) return;

  // Position and dimension active media element
  targetEl.style.position = "absolute";
  targetEl.style.left = `${offX}px`;
  targetEl.style.top = `${offY}px`;
  targetEl.style.width = `${dispW}px`;
  targetEl.style.height = `${dispH}px`;
  targetEl.style.maxWidth = "none";
  targetEl.style.maxHeight = "none";
  targetEl.style.objectFit = "contain";

  // Position and dimension zoneOverlayCanvas exactly to match displayed media
  if (elements.zoneOverlayCanvas) {
    const canvas = elements.zoneOverlayCanvas;
    canvas.style.position = "absolute";
    canvas.style.left = `${offX}px`;
    canvas.style.top = `${offY}px`;
    canvas.style.width = `${dispW}px`;
    canvas.style.height = `${dispH}px`;
    if (canvas.width !== dispW || canvas.height !== dispH) {
      canvas.width = dispW;
      canvas.height = dispH;
    }
  }

  // Update telemetry and debug info
  updateViewportDebugOutput({
    source_width: origW,
    source_height: origH,
    viewport_width: cW,
    viewport_height: cH,
    displayed_width: dispW,
    displayed_height: dispH,
    scale_factor: Number(scale.toFixed(4)),
    letterbox_offset_x: offX,
    letterbox_offset_y: offY,
  });

  renderZonesOverlay();
}

function updateViewportDebugOutput(info = {}) {
  const currentDiag = state.lastDiagnostics || {};
  const merged = { ...currentDiag, ...info };
  state.lastDiagnostics = merged;

  if (elements.dbgSourceDim && merged.source_width && merged.source_height) {
    elements.dbgSourceDim.textContent = `${merged.source_width} × ${merged.source_height}`;
  }
  if (elements.dbgViewportDim && merged.viewport_width && merged.viewport_height) {
    elements.dbgViewportDim.textContent = `${merged.viewport_width} × ${merged.viewport_height}`;
  }
  if (elements.dbgDisplayedDim && merged.displayed_width && merged.displayed_height) {
    elements.dbgDisplayedDim.textContent = `${merged.displayed_width} × ${merged.displayed_height}`;
  }
  if (elements.dbgScaleFactor && merged.scale_factor != null) {
    elements.dbgScaleFactor.textContent = `${merged.scale_factor}`;
  }
  if (elements.dbgLetterboxOffset && merged.letterbox_offset_x != null && merged.letterbox_offset_y != null) {
    elements.dbgLetterboxOffset.textContent = `X: ${merged.letterbox_offset_x}px, Y: ${merged.letterbox_offset_y}px`;
  }
  if (elements.dbgRawPersons && merged.raw_person_count != null) {
    elements.dbgRawPersons.textContent = `${merged.raw_person_count}`;
  }
  if (elements.dbgFilteredPersons && merged.filtered_person_count != null) {
    elements.dbgFilteredPersons.textContent = `${merged.filtered_person_count}`;
  }
  if (elements.dbgTrackCount && merged.track_count != null) {
    elements.dbgTrackCount.textContent = `${merged.track_count}`;
  }
  if (elements.dbgRenderedBoxes && merged.rendered_boxes_count != null) {
    elements.dbgRenderedBoxes.textContent = `${merged.rendered_boxes_count}`;
  }

  console.log("[INTELLIWATCH VIEWPORT DIAGNOSTICS]", {
    source: `${merged.source_width}x${merged.source_height}`,
    viewport: `${merged.viewport_width}x${merged.viewport_height}`,
    displayed: `${merged.displayed_width}x${merged.displayed_height}`,
    scale: merged.scale_factor,
    letterbox: `(${merged.letterbox_offset_x}px, ${merged.letterbox_offset_y}px)`,
    raw_persons: merged.raw_person_count,
    filtered_persons: merged.filtered_person_count,
    tracks: merged.track_count,
    rendered_boxes: merged.rendered_boxes_count,
  });
}

function displayToOriginal(clientX, clientY) {
  const { origW, origH } = getActiveMediaDimensions();
  if (!elements.zoneOverlayCanvas) return [0, 0];
  const rect = elements.zoneOverlayCanvas.getBoundingClientRect();
  const clickX = clientX - rect.left;
  const clickY = clientY - rect.top;
  const scaleX = rect.width > 0 ? origW / rect.width : 1.0;
  const scaleY = rect.height > 0 ? origH / rect.height : 1.0;

  let origX = Math.max(0, Math.min(origW, Math.round(clickX * scaleX)));
  let origY = Math.max(0, Math.min(origH, Math.round(clickY * scaleY)));

  return [origX, origY];
}

function originalToDisplay(origX, origY) {
  const { origW, origH, dispW, dispH } = getViewportTransform();
  const scaleX = origW > 0 ? dispW / origW : 1.0;
  const scaleY = origH > 0 ? dispH / origH : 1.0;
  return [origX * scaleX, origY * scaleY];
}

function setupViewportObserver() {
  window.addEventListener("resize", () => {
    fitViewportMedia();
  });

  if (window.ResizeObserver && elements.viewportContainer) {
    const ro = new ResizeObserver(() => {
      fitViewportMedia();
    });
    ro.observe(elements.viewportContainer);
  }

  if (elements.viewportImage) {
    elements.viewportImage.addEventListener("load", () => {
      fitViewportMedia();
    });
  }

  if (elements.viewportVideo) {
    elements.viewportVideo.addEventListener("loadedmetadata", () => {
      fitViewportMedia();
    });
  }
}

async function fetchZones() {
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/zones`);
    if (!res.ok) return;
    state.zones = await res.json();
    renderConfiguredZonesList();
    renderZonesOverlay();
  } catch (err) {
    console.warn("Could not fetch zones:", err);
  }
}

function renderConfiguredZonesList() {
  if (!elements.configuredZonesList) return;
  if (elements.configuredZoneCount) {
    elements.configuredZoneCount.textContent = (state.zones || []).length;
  }

  if (!state.zones || state.zones.length === 0) {
    elements.configuredZonesList.innerHTML = `
      <div style="color: var(--text-muted); font-size: 11px; padding: 12px; text-align: center;">
        No safety zones configured yet. Click "+ Draw Zone" to configure a zone.
      </div>`;
    return;
  }

  elements.configuredZonesList.innerHTML = state.zones
    .map((z) => {
      const zid = z.zone_id || z.id;
      const zType = (z.zone_type || "restricted").toLowerCase();
      const dwellVal = z.max_dwell_seconds != null ? z.max_dwell_seconds : z.max_dwell_time;
      const dwellText = dwellVal != null ? `Max Dwell: ${dwellVal}s` : "No Dwell Limit";
      const ptsCount = (z.polygon || []).length;

      return `
        <div class="zone-card">
          <div class="zone-card-header">
            <div class="zone-card-title">
              <span>${z.name}</span>
            </div>
            <span class="zone-badge zone-badge-${zType}">${zType}</span>
          </div>
          <div class="zone-card-meta">
            <span>ID: <code>${zid}</code></span>
            <span>•</span>
            <span>${ptsCount} Vertices</span>
            <span>•</span>
            <span>${dwellText}</span>
          </div>
          <div class="zone-card-actions" style="margin-top: 4px; justify-content: flex-end;">
            <button class="btn btn-sm" style="padding: 2px 8px; font-size: 10px;" onclick="window.editZone('${zid}')">
              ✎ Edit
            </button>
            <button class="btn btn-sm btn-danger" style="padding: 2px 8px; font-size: 10px;" onclick="window.deleteZone('${zid}')">
              ✕ Delete
            </button>
          </div>
        </div>
      `;
    })
    .join("");
}

function renderZonesOverlay() {
  if (!elements.zoneOverlayCanvas || !elements.viewportContainer) return;
  const canvas = elements.zoneOverlayCanvas;
  const container = elements.viewportContainer;

  const w = container.clientWidth;
  const h = container.clientHeight;
  if (w <= 0 || h <= 0) return;

  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }

  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, w, h);

  // 1. If currently in interactive drawing mode, ALWAYS allow drawing and show zones
  if (state.isDrawingZone) {
    drawZonesList(ctx, state.zones || []);
    drawDrawingInProgress(ctx, container);
    return;
  }

  // 2. Empty state: Before media analysis, do NOT draw default placeholder polygons!
  if (!state.currentMediaSource) {
    return;
  }

  // 3. Uploaded media (image or video): Do NOT render arbitrary camera default zones!
  // Configured safety zones only apply to scenes where explicitly configured
  if (state.currentMediaSource === "upload") {
    if (state.uploadZones && state.uploadZones.length > 0) {
      drawZonesList(ctx, state.uploadZones);
    }
    return;
  }

  // 4. Configured camera stream (e.g. cam_01): Render explicitly configured zones for that camera
  if (state.currentMediaSource === "camera") {
    const camId = currentViewingCameraId || "cam_01";
    const camZones = (state.zones || []).filter((z) => !z.camera_id || z.camera_id === camId);
    drawZonesList(ctx, camZones);
  }
}

function drawZonesList(ctx, zonesToRender) {
  (zonesToRender || []).forEach((z) => {
    if (!z.polygon || z.polygon.length < 3) return;
    const colorRgb = z.color && z.color.length === 3 ? z.color : getZoneColor(z.zone_type);
    const [r, g, b] = colorRgb;

    ctx.save();
    ctx.beginPath();
    let sumX = 0;
    let sumY = 0;

    z.polygon.forEach((pt, i) => {
      const [dx, dy] = originalToDisplay(pt[0], pt[1]);
      sumX += dx;
      sumY += dy;
      if (i === 0) ctx.moveTo(dx, dy);
      else ctx.lineTo(dx, dy);
    });

    ctx.closePath();
    ctx.fillStyle = `rgba(${r}, ${g}, ${b}, 0.22)`;
    ctx.fill();

    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.85)`;
    ctx.lineWidth = 2;
    ctx.stroke();

    // Vertices
    z.polygon.forEach((pt) => {
      const [dx, dy] = originalToDisplay(pt[0], pt[1]);
      ctx.beginPath();
      ctx.arc(dx, dy, 3, 0, 2 * Math.PI);
      ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1;
      ctx.stroke();
    });

    // Centroid Label Badge
    const centroidX = sumX / z.polygon.length;
    const centroidY = sumY / z.polygon.length;
    const labelText = `${z.name} [${(z.zone_type || "").toUpperCase()}]`;

    ctx.font = "bold 9px monospace";
    const textMetrics = ctx.measureText(labelText);
    const pad = 4;
    const badgeW = textMetrics.width + pad * 2;
    const badgeH = 14;

    ctx.fillStyle = "rgba(10, 15, 25, 0.85)";
    ctx.strokeStyle = `rgba(${r}, ${g}, ${b}, 0.8)`;
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.roundRect(centroidX - badgeW / 2, centroidY - badgeH / 2, badgeW, badgeH, 3);
    ctx.fill();
    ctx.stroke();

    ctx.fillStyle = `rgb(${r}, ${g}, ${b})`;
    ctx.textAlign = "center";
    ctx.textBaseline = "middle";
    ctx.fillText(labelText, centroidX, centroidY);
    ctx.restore();
  });
}

function drawDrawingInProgress(ctx, container) {
  if (state.isDrawingZone && state.currentZonePoints.length > 0) {
    ctx.save();
    ctx.setLineDash([4, 4]);
    ctx.strokeStyle = "#38bdf8";
    ctx.lineWidth = 2;

    ctx.beginPath();
    state.currentZonePoints.forEach((pt, idx) => {
      const [dx, dy] = originalToDisplay(pt[0], pt[1]);
      if (idx === 0) ctx.moveTo(dx, dy);
      else ctx.lineTo(dx, dy);
    });

    // Rubberband line to cursor
    if (state.cursorPos) {
      const cRect = container.getBoundingClientRect();
      const curX = state.cursorPos[0] - cRect.left;
      const curY = state.cursorPos[1] - cRect.top;
      ctx.lineTo(curX, curY);

      if (state.currentZonePoints.length >= 2) {
        // Guide back to first point
        const [firstDx, firstDy] = originalToDisplay(state.currentZonePoints[0][0], state.currentZonePoints[0][1]);
        ctx.strokeStyle = "rgba(56, 189, 248, 0.4)";
        ctx.stroke();
        ctx.beginPath();
        ctx.moveTo(curX, curY);
        ctx.lineTo(firstDx, firstDy);
      }
    }
    ctx.stroke();
    ctx.setLineDash([]);

    // Draw numbered vertices
    state.currentZonePoints.forEach((pt, idx) => {
      const [dx, dy] = originalToDisplay(pt[0], pt[1]);
      ctx.beginPath();
      ctx.arc(dx, dy, 6, 0, 2 * Math.PI);
      ctx.fillStyle = "#38bdf8";
      ctx.fill();
      ctx.strokeStyle = "#ffffff";
      ctx.lineWidth = 1.5;
      ctx.stroke();

      ctx.font = "bold 8px monospace";
      ctx.fillStyle = "#0f172a";
      ctx.textAlign = "center";
      ctx.textBaseline = "middle";
      ctx.fillText(`${idx + 1}`, dx, dy);
    });

    ctx.restore();
  }
}

function updatePointBadge() {
  if (!elements.zonePointCountBadge) return;
  const count = state.currentZonePoints.length;
  elements.zonePointCountBadge.textContent = `${count} points ${count < 3 ? "(min 3)" : "✓"}`;
  elements.zonePointCountBadge.style.color = count >= 3 ? "#34d399" : "#fbbf24";
}

function startDrawingZone() {
  state.isDrawingZone = true;
  state.currentZonePoints = [];
  state.cursorPos = null;
  state.editingZoneId = null;

  if (elements.zoneFormTitle) elements.zoneFormTitle.textContent = "NEW SAFETY ZONE";
  if (elements.zoneNameInput) elements.zoneNameInput.value = "";
  if (elements.zoneDwellInput) elements.zoneDwellInput.value = "";
  if (elements.zoneTypeSelect) elements.zoneTypeSelect.value = "restricted";
  if (elements.zoneDrawForm) elements.zoneDrawForm.style.display = "block";
  if (elements.zoneOverlayCanvas) elements.zoneOverlayCanvas.classList.add("drawing-active");

  updatePointBadge();
  renderZonesOverlay();
}

function cancelDrawingZone() {
  state.isDrawingZone = false;
  state.currentZonePoints = [];
  state.cursorPos = null;
  state.editingZoneId = null;

  if (elements.zoneDrawForm) elements.zoneDrawForm.style.display = "none";
  if (elements.zoneOverlayCanvas) elements.zoneOverlayCanvas.classList.remove("drawing-active");

  renderZonesOverlay();
}

function clearDrawingPoints() {
  state.currentZonePoints = [];
  state.cursorPos = null;
  updatePointBadge();
  renderZonesOverlay();
}

async function saveZone() {
  const name = elements.zoneNameInput ? elements.zoneNameInput.value.trim() : "";
  const zoneType = elements.zoneTypeSelect ? elements.zoneTypeSelect.value : "restricted";
  const dwellVal = elements.zoneDwellInput && elements.zoneDwellInput.value
    ? parseFloat(elements.zoneDwellInput.value)
    : null;

  if (!name) {
    alert("Please provide a name for this safety zone.");
    if (elements.zoneNameInput) elements.zoneNameInput.focus();
    return;
  }

  if (state.currentZonePoints.length < 3) {
    alert("Safety zones require at least 3 polygon vertices. Click on the scene image/video to place vertices.");
    return;
  }

  const { origW, origH } = getActiveMediaDimensions();
  const payload = {
    name: name,
    zone_type: zoneType,
    polygon: state.currentZonePoints,
    max_dwell_seconds: dwellVal,
    enabled: true,
    source_resolution: [origW, origH],
  };

  try {
    let res;
    if (state.editingZoneId) {
      res = await authFetch(`${CONFIG.apiBase}/api/v1/zones/${state.editingZoneId}`, {
        method: "PUT",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    } else {
      res = await authFetch(`${CONFIG.apiBase}/api/v1/zones`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
    }

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Failed to save safety zone");
    }

    cancelDrawingZone();
    await fetchZones();
  } catch (err) {
    alert(`Could not save zone: ${err.message}`);
  }
}

window.editZone = function (zoneId) {
  const zone = (state.zones || []).find((z) => (z.zone_id || z.id) === zoneId);
  if (!zone) return;

  const zid = zone.zone_id || zone.id;
  state.editingZoneId = zid;
  state.isDrawingZone = true;
  state.currentZonePoints = JSON.parse(JSON.stringify(zone.polygon || []));
  state.cursorPos = null;

  const dwellVal = zone.max_dwell_seconds != null ? zone.max_dwell_seconds : zone.max_dwell_time;
  if (elements.zoneFormTitle) elements.zoneFormTitle.textContent = `EDIT ZONE: ${zone.name}`;
  if (elements.zoneNameInput) elements.zoneNameInput.value = zone.name || "";
  if (elements.zoneTypeSelect) elements.zoneTypeSelect.value = (zone.zone_type || "restricted").toLowerCase();
  if (elements.zoneDwellInput) elements.zoneDwellInput.value = dwellVal != null ? dwellVal : "";
  if (elements.zoneDrawForm) elements.zoneDrawForm.style.display = "block";
  if (elements.zoneOverlayCanvas) elements.zoneOverlayCanvas.classList.add("drawing-active");

  // Activate Safety Zones tab
  const tabBtn = document.getElementById("tabZonesBtn");
  if (tabBtn) tabBtn.click();

  updatePointBadge();
  renderZonesOverlay();
};

window.deleteZone = async function (zoneId) {
  const confirmed = confirm(`Are you sure you want to delete safety zone '${zoneId}'?`);
  if (!confirmed) return;

  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/zones/${zoneId}`, {
      method: "DELETE",
    });

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || "Failed to delete safety zone");
    }

    if (state.editingZoneId === zoneId) {
      cancelDrawingZone();
    }

    await fetchZones();
  } catch (err) {
    alert(`Could not delete zone: ${err.message}`);
  }
};

// -------------------------------------------------------------
// STEP 21: RTSP & MULTI-CAMERA STREAM INGESTION HANDLERS
// -------------------------------------------------------------
let activeStreamPollingInterval = null;
let currentViewingCameraId = null;

// Stage 6B: explicit, camera-scoped state for the calibration workbench.
const calibrationWorkbenchState = {
  cameras: [],
  selectedCameraId: "",
  calibration: null,
  points: [],
  frameUrl: "",
  frameWidth: 0,
  frameHeight: 0,
  frameError: "",
  cameraListError: "",
  configRequestError: "",
  busy: "",
  validation: null,
  validationState: "idle",
  validationError: "",
  feedback: "",
  feedbackTone: "",
  isDirty: false,
  selectionRequestId: 0,
  frameRequestId: 0,
  draggingPointIndex: null,
  movedPointDuringDrag: false,
  suppressStageClick: false,
};

function setupCalibrationWorkbench() {
  if (elements.calibrationCameraSelect) {
    elements.calibrationCameraSelect.addEventListener("change", (event) => {
      selectCalibrationCamera(event.target.value);
    });
  }
  if (elements.calibrationRefreshFrameBtn) {
    elements.calibrationRefreshFrameBtn.addEventListener("click", () => {
      if (calibrationWorkbenchState.selectedCameraId) {
        loadCalibrationFrame(calibrationWorkbenchState.selectedCameraId, calibrationWorkbenchState.selectionRequestId);
      }
    });
  }
  if (elements.calibrationResetPointsBtn) {
    elements.calibrationResetPointsBtn.addEventListener("click", resetCalibrationPoints);
  }
  if (elements.calibrationValidateBtn) {
    elements.calibrationValidateBtn.addEventListener("click", validateCalibrationDraft);
  }
  if (elements.calibrationSaveBtn) {
    elements.calibrationSaveBtn.addEventListener("click", saveCalibrationDraft);
  }
  if (elements.calibrationClearBtn) {
    elements.calibrationClearBtn.addEventListener("click", clearCameraCalibration);
  }
  [elements.calibrationWidthInput, elements.calibrationDepthInput].forEach((input) => {
    if (!input) return;
    input.addEventListener("input", () => {
      markCalibrationDraftChanged();
      renderCalibrationWorkbench();
    });
  });

  if (elements.calibrationImageStage) {
    elements.calibrationImageStage.addEventListener("pointerdown", beginCalibrationPointDrag);
    elements.calibrationImageStage.addEventListener("pointermove", moveCalibrationPoint);
    elements.calibrationImageStage.addEventListener("pointerup", finishCalibrationPointDrag);
    elements.calibrationImageStage.addEventListener("pointercancel", finishCalibrationPointDrag);
    elements.calibrationImageStage.addEventListener("click", addCalibrationPointFromImage);
  }

  if (typeof ResizeObserver !== "undefined" && elements.calibrationImageStage) {
    const resizeObserver = new ResizeObserver(fitCalibrationImageToStage);
    resizeObserver.observe(elements.calibrationImageStage);
  } else {
    window.addEventListener("resize", fitCalibrationImageToStage);
  }
  renderCalibrationWorkbench();
}

function markCalibrationDraftChanged() {
  calibrationWorkbenchState.isDirty = true;
  calibrationWorkbenchState.validation = null;
  calibrationWorkbenchState.validationState = "idle";
  calibrationWorkbenchState.validationError = "";
}

function setCalibrationFeedback(message, tone = "") {
  calibrationWorkbenchState.feedback = message || "";
  calibrationWorkbenchState.feedbackTone = tone;
}

function syncCalibrationCameraList(cameras) {
  calibrationWorkbenchState.cameraListError = "";
  calibrationWorkbenchState.cameras = Array.isArray(cameras) ? cameras : [];
  const select = elements.calibrationCameraSelect;
  if (!select) return;

  select.replaceChildren();
  if (calibrationWorkbenchState.cameras.length === 0) {
    const option = document.createElement("option");
    option.value = "";
    option.textContent = "No registered cameras";
    select.appendChild(option);
    select.disabled = true;
    if (elements.calibrationImageEmptyTitle) elements.calibrationImageEmptyTitle.textContent = "No registered cameras";
    if (elements.calibrationImageEmptyHint) elements.calibrationImageEmptyHint.textContent = "Register a camera in Live Cameras before calibrating it.";
    if (calibrationWorkbenchState.selectedCameraId) {
      calibrationWorkbenchState.selectionRequestId += 1;
      releaseCalibrationFrame();
    }
    calibrationWorkbenchState.selectedCameraId = "";
    calibrationWorkbenchState.calibration = null;
    calibrationWorkbenchState.points = [];
    calibrationWorkbenchState.frameError = "";
    calibrationWorkbenchState.configRequestError = "";
    calibrationWorkbenchState.validation = null;
    calibrationWorkbenchState.validationState = "idle";
    calibrationWorkbenchState.validationError = "";
    calibrationWorkbenchState.busy = "";
    calibrationWorkbenchState.isDirty = false;
    if (elements.calibrationWidthInput) elements.calibrationWidthInput.value = "";
    if (elements.calibrationDepthInput) elements.calibrationDepthInput.value = "";
    setCalibrationFeedback("No registered cameras are available. Register a camera before calibrating it.");
    renderCalibrationWorkbench();
    return;
  }

  calibrationWorkbenchState.cameras.forEach((camera) => {
    const option = document.createElement("option");
    option.value = camera.camera_id;
    option.textContent = `${camera.camera_id} — ${camera.name || camera.camera_id}`;
    select.appendChild(option);
  });
  select.disabled = false;

  const stillRegistered = calibrationWorkbenchState.cameras.some(
    (camera) => camera.camera_id === calibrationWorkbenchState.selectedCameraId,
  );
  const nextCameraId = stillRegistered
    ? calibrationWorkbenchState.selectedCameraId
    : calibrationWorkbenchState.cameras[0].camera_id;
  select.value = nextCameraId;
  if (nextCameraId !== calibrationWorkbenchState.selectedCameraId) {
    selectCalibrationCamera(nextCameraId);
  } else {
    renderCalibrationWorkbench();
  }
}

function markCalibrationCameraListFailure(message) {
  calibrationWorkbenchState.cameraListError = message || "Registered cameras could not be loaded.";
  if (calibrationWorkbenchState.cameras.length === 0) {
    const select = elements.calibrationCameraSelect;
    if (select) {
      select.replaceChildren();
      const option = document.createElement("option");
      option.value = "";
      option.textContent = "Camera list unavailable";
      select.appendChild(option);
      select.disabled = true;
    }
  }
  if (!calibrationWorkbenchState.selectedCameraId) {
    setCalibrationFeedback(calibrationWorkbenchState.cameraListError, "error");
    if (elements.calibrationImageEmptyTitle) elements.calibrationImageEmptyTitle.textContent = "Camera inventory unavailable";
    if (elements.calibrationImageEmptyHint) elements.calibrationImageEmptyHint.textContent = calibrationWorkbenchState.cameraListError;
  } else {
    setCalibrationFeedback(`Camera inventory refresh failed: ${calibrationWorkbenchState.cameraListError}`, "error");
  }
  renderCalibrationWorkbench();
}

function currentCalibrationCamera() {
  return calibrationWorkbenchState.cameras.find(
    (camera) => camera.camera_id === calibrationWorkbenchState.selectedCameraId,
  ) || null;
}

function releaseCalibrationFrame() {
  if (calibrationWorkbenchState.frameUrl) {
    URL.revokeObjectURL(calibrationWorkbenchState.frameUrl);
    calibrationWorkbenchState.frameUrl = "";
  }
  if (elements.calibrationFrameImage) elements.calibrationFrameImage.removeAttribute("src");
  if (elements.calibrationImageLayer) elements.calibrationImageLayer.hidden = true;
  if (elements.calibrationImageEmpty) elements.calibrationImageEmpty.hidden = false;
  calibrationWorkbenchState.frameWidth = 0;
  calibrationWorkbenchState.frameHeight = 0;
}

async function selectCalibrationCamera(cameraId) {
  const camera = calibrationWorkbenchState.cameras.find((item) => item.camera_id === cameraId);
  if (!camera) return;

  calibrationWorkbenchState.selectionRequestId += 1;
  const requestId = calibrationWorkbenchState.selectionRequestId;
  calibrationWorkbenchState.selectedCameraId = cameraId;
  calibrationWorkbenchState.calibration = null;
  calibrationWorkbenchState.points = [];
  calibrationWorkbenchState.frameError = "";
  calibrationWorkbenchState.configRequestError = "";
  calibrationWorkbenchState.validation = null;
  calibrationWorkbenchState.validationState = "idle";
  calibrationWorkbenchState.validationError = "";
  calibrationWorkbenchState.isDirty = false;
  calibrationWorkbenchState.busy = "loading";
  if (elements.calibrationWidthInput) elements.calibrationWidthInput.value = "";
  if (elements.calibrationDepthInput) elements.calibrationDepthInput.value = "";
  setCalibrationFeedback(`Loading frame and calibration for ${camera.camera_id}...`, "pending");
  releaseCalibrationFrame();
  renderCalibrationWorkbench();

  await Promise.all([
    loadCalibrationConfiguration(cameraId, requestId),
    loadCalibrationFrame(cameraId, requestId),
  ]);
  if (requestId !== calibrationWorkbenchState.selectionRequestId) return;

  calibrationWorkbenchState.busy = "";
  if (calibrationWorkbenchState.configRequestError) {
    setCalibrationFeedback(`Calibration status could not be retrieved: ${calibrationWorkbenchState.configRequestError}`, "error");
  } else if (calibrationWorkbenchState.frameError) {
    setCalibrationFeedback(`Camera frame could not be loaded: ${calibrationWorkbenchState.frameError}`, "error");
  } else if (calibrationWorkbenchState.calibration?.calibration_status === "UNCONFIGURED") {
    setCalibrationFeedback(`No calibration is configured for ${camera.camera_id}. Select the four ground-plane corners to create one.`);
  } else {
    setCalibrationFeedback(`Loaded ${camera.camera_id} calibration. Drag existing markers to adjust the saved points before validating or saving.`);
  }
  warnIfCalibrationPointsOutsideFrame();
  renderCalibrationWorkbench();
  refreshCalibrationSceneContext();
}

async function loadCalibrationConfiguration(cameraId, requestId) {
  calibrationWorkbenchState.configRequestError = "";
  try {
    const response = await authFetch(`${CONFIG.apiBase}/api/v1/cameras/${encodeURIComponent(cameraId)}/calibration`);
    if (!response.ok) throw new Error(await calibrationResponseError(response));
    const calibration = await response.json();
    if (requestId !== calibrationWorkbenchState.selectionRequestId) return;

    calibrationWorkbenchState.calibration = calibration;
    const loadedPoints = Array.isArray(calibration.points) ? calibration.points : [];
    calibrationWorkbenchState.points = loadedPoints.length === 4 && loadedPoints.every(
      (point) => Number.isFinite(Number(point.x)) && Number.isFinite(Number(point.y)),
    )
      ? loadedPoints.map((point) => ({ x: Number(point.x), y: Number(point.y) }))
      : [];
    calibrationWorkbenchState.isDirty = false;
    if (elements.calibrationWidthInput) {
      elements.calibrationWidthInput.value = calibration.calibration_status === "UNCONFIGURED"
        ? ""
        : String(calibration.real_world_width_m ?? "");
    }
    if (elements.calibrationDepthInput) {
      elements.calibrationDepthInput.value = calibration.calibration_status === "UNCONFIGURED"
        ? ""
        : String(calibration.real_world_depth_m ?? "");
    }
    renderCalibrationWorkbench();
  } catch (error) {
    if (requestId !== calibrationWorkbenchState.selectionRequestId) return;
    calibrationWorkbenchState.configRequestError = error.message || "Calibration request failed.";
    renderCalibrationWorkbench();
  }
}

async function loadCalibrationFrame(cameraId, requestId) {
  const frameRequestId = ++calibrationWorkbenchState.frameRequestId;
  calibrationWorkbenchState.frameError = "";
  releaseCalibrationFrame();
  if (elements.calibrationImageEmptyTitle) elements.calibrationImageEmptyTitle.textContent = "Loading camera frame...";
  if (elements.calibrationImageEmptyHint) elements.calibrationImageEmptyHint.textContent = "Fetching the selected camera snapshot.";
  renderCalibrationWorkbench();

  let frameUrl = "";
  try {
    const response = await authFetch(
      `${CONFIG.apiBase}/api/v1/cameras/${encodeURIComponent(cameraId)}/snapshot?t=${Date.now()}`,
    );
    if (!response.ok) throw new Error(await calibrationResponseError(response));
    const blob = await response.blob();
    frameUrl = URL.createObjectURL(blob);
    if (requestId !== calibrationWorkbenchState.selectionRequestId || frameRequestId !== calibrationWorkbenchState.frameRequestId) {
      URL.revokeObjectURL(frameUrl);
      return;
    }

    const image = elements.calibrationFrameImage;
    await new Promise((resolve, reject) => {
      image.onload = resolve;
      image.onerror = () => reject(new Error("The camera snapshot could not be decoded as an image."));
      image.src = frameUrl;
    });
    if (requestId !== calibrationWorkbenchState.selectionRequestId || frameRequestId !== calibrationWorkbenchState.frameRequestId) {
      URL.revokeObjectURL(frameUrl);
      return;
    }

    calibrationWorkbenchState.frameUrl = frameUrl;
    calibrationWorkbenchState.frameWidth = image.naturalWidth;
    calibrationWorkbenchState.frameHeight = image.naturalHeight;
    if (elements.calibrationImageLayer) elements.calibrationImageLayer.hidden = false;
    if (elements.calibrationImageEmpty) elements.calibrationImageEmpty.hidden = true;
    fitCalibrationImageToStage();
    renderCalibrationWorkbench();
  } catch (error) {
    if (frameUrl) URL.revokeObjectURL(frameUrl);
    if (requestId !== calibrationWorkbenchState.selectionRequestId || frameRequestId !== calibrationWorkbenchState.frameRequestId) return;
    calibrationWorkbenchState.frameError = error.message || "Camera snapshot request failed.";
    if (elements.calibrationImageEmptyTitle) elements.calibrationImageEmptyTitle.textContent = "Camera frame unavailable";
    if (elements.calibrationImageEmptyHint) elements.calibrationImageEmptyHint.textContent = calibrationWorkbenchState.frameError;
    renderCalibrationWorkbench();
  }
}

function fitCalibrationImageToStage() {
  const stage = elements.calibrationImageStage;
  const layer = elements.calibrationImageLayer;
  const width = calibrationWorkbenchState.frameWidth;
  const height = calibrationWorkbenchState.frameHeight;
  if (!stage || !layer || !width || !height || layer.hidden) return;

  const availableWidth = stage.clientWidth;
  const availableHeight = stage.clientHeight;
  if (!availableWidth || !availableHeight) return;
  const scale = Math.min(availableWidth / width, availableHeight / height);
  const displayWidth = Math.max(1, width * scale);
  const displayHeight = Math.max(1, height * scale);
  layer.style.width = `${displayWidth}px`;
  layer.style.height = `${displayHeight}px`;
  if (elements.calibrationPointOverlay) {
    elements.calibrationPointOverlay.setAttribute("viewBox", `0 0 ${width} ${height}`);
  }
  renderCalibrationPointOverlay();
}

function calibrationSourcePointFromClient(clientX, clientY, clampToImage = false) {
  const layer = elements.calibrationImageLayer;
  if (!layer || layer.hidden || !calibrationWorkbenchState.frameWidth || !calibrationWorkbenchState.frameHeight) return null;
  const rect = layer.getBoundingClientRect();
  if (!rect.width || !rect.height) return null;
  const relativeX = (clientX - rect.left) / rect.width;
  const relativeY = (clientY - rect.top) / rect.height;
  if (!clampToImage && (relativeX < 0 || relativeX > 1 || relativeY < 0 || relativeY > 1)) return null;
  const normalizedX = Math.max(0, Math.min(1, relativeX));
  const normalizedY = Math.max(0, Math.min(1, relativeY));
  return {
    x: normalizedX * calibrationWorkbenchState.frameWidth,
    y: normalizedY * calibrationWorkbenchState.frameHeight,
  };
}

function beginCalibrationPointDrag(event) {
  const marker = event.target.closest?.("[data-point-index]");
  if (!marker || calibrationWorkbenchState.busy || !calibrationWorkbenchState.frameUrl) return;
  const pointIndex = Number(marker.dataset.pointIndex);
  if (!Number.isInteger(pointIndex) || pointIndex < 0 || pointIndex >= calibrationWorkbenchState.points.length) return;
  calibrationWorkbenchState.draggingPointIndex = pointIndex;
  calibrationWorkbenchState.movedPointDuringDrag = false;
  calibrationWorkbenchState.suppressStageClick = false;
  try {
    elements.calibrationImageStage.setPointerCapture(event.pointerId);
  } catch (_) {
    // Pointer capture is a convenience; dragging still works without it.
  }
  event.preventDefault();
}

function moveCalibrationPoint(event) {
  const pointIndex = calibrationWorkbenchState.draggingPointIndex;
  if (pointIndex == null) return;
  const point = calibrationSourcePointFromClient(event.clientX, event.clientY, true);
  if (!point) return;
  calibrationWorkbenchState.points[pointIndex] = point;
  calibrationWorkbenchState.movedPointDuringDrag = true;
  markCalibrationDraftChanged();
  setCalibrationFeedback(`P${pointIndex + 1} moved to ${formatCalibrationCoordinate(point.x)}, ${formatCalibrationCoordinate(point.y)} source pixels.`);
  renderCalibrationWorkbench();
}

function finishCalibrationPointDrag(event) {
  if (calibrationWorkbenchState.draggingPointIndex == null) return;
  calibrationWorkbenchState.draggingPointIndex = null;
  if (calibrationWorkbenchState.movedPointDuringDrag) {
    calibrationWorkbenchState.suppressStageClick = true;
    window.setTimeout(() => { calibrationWorkbenchState.suppressStageClick = false; }, 0);
  }
  calibrationWorkbenchState.movedPointDuringDrag = false;
  try {
    if (elements.calibrationImageStage.hasPointerCapture(event.pointerId)) {
      elements.calibrationImageStage.releasePointerCapture(event.pointerId);
    }
  } catch (_) {
    // The pointer may already have been released by the browser.
  }
}

function addCalibrationPointFromImage(event) {
  if (calibrationWorkbenchState.suppressStageClick) {
    calibrationWorkbenchState.suppressStageClick = false;
    return;
  }
  if (!calibrationWorkbenchState.selectedCameraId || !calibrationWorkbenchState.frameUrl || calibrationWorkbenchState.busy) return;
  if (event.target.closest?.("[data-point-index]")) return;
  if (calibrationWorkbenchState.points.length >= 4) {
    setCalibrationFeedback("Four points are already selected. Drag a numbered point to reposition it, or reset the points first.");
    renderCalibrationWorkbench();
    return;
  }
  const point = calibrationSourcePointFromClient(event.clientX, event.clientY);
  if (!point) return;
  calibrationWorkbenchState.points.push(point);
  markCalibrationDraftChanged();
  const pointNumber = calibrationWorkbenchState.points.length;
  setCalibrationFeedback(
    pointNumber === 4
      ? "Four points selected. Confirm the rectangle dimensions, then validate the proposed calibration."
      : `P${pointNumber} set to ${formatCalibrationCoordinate(point.x)}, ${formatCalibrationCoordinate(point.y)} source pixels. Select P${pointNumber + 1}.`,
  );
  renderCalibrationWorkbench();
}

function resetCalibrationPoints() {
  if (calibrationWorkbenchState.busy) return;
  calibrationWorkbenchState.points = [];
  markCalibrationDraftChanged();
  setCalibrationFeedback("Points reset. Click the image to select P1, P2, P3, and P4 again.");
  renderCalibrationWorkbench();
}

function formatCalibrationCoordinate(value) {
  return Number.isFinite(value) ? value.toFixed(1) : "--";
}

function renderCalibrationPointOverlay() {
  const points = calibrationWorkbenchState.points;
  if (elements.calibrationPointLine) {
    elements.calibrationPointLine.setAttribute("points", points.map((point) => `${point.x},${point.y}`).join(" "));
  }
  if (elements.calibrationPointPolygon) {
    elements.calibrationPointPolygon.setAttribute(
      "points",
      points.length === 4 ? points.map((point) => `${point.x},${point.y}`).join(" ") : "",
    );
  }
  if (!elements.calibrationPointMarkers) return;
  const markerRadius = Math.max(8, Math.min(22, calibrationWorkbenchState.frameWidth * 0.012));
  elements.calibrationPointMarkers.replaceChildren();
  points.forEach((point, index) => {
    const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
    group.setAttribute("class", "calibration-point-marker");
    group.setAttribute("data-point-index", String(index));
    group.setAttribute("role", "button");
    group.setAttribute("aria-label", `Point ${index + 1}, drag to reposition`);
    group.setAttribute("tabindex", "0");

    const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
    circle.setAttribute("cx", String(point.x));
    circle.setAttribute("cy", String(point.y));
    circle.setAttribute("r", String(markerRadius));
    group.appendChild(circle);

    const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
    label.setAttribute("x", String(point.x));
    label.setAttribute("y", String(point.y));
    label.textContent = String(index + 1);
    group.appendChild(label);
    elements.calibrationPointMarkers.appendChild(group);
  });
}

function calibrationDimensionValues() {
  const width = Number(elements.calibrationWidthInput?.value);
  const depth = Number(elements.calibrationDepthInput?.value);
  if (!Number.isFinite(width) || width <= 0.05 || width > 500) {
    return { error: "Known width must be greater than 0.05 m and no more than 500 m." };
  }
  if (!Number.isFinite(depth) || depth <= 0.05 || depth > 500) {
    return { error: "Known depth must be greater than 0.05 m and no more than 500 m." };
  }
  return { width, depth };
}

function getCalibrationDraftPayload() {
  if (!currentCalibrationCamera()) return { error: "Select a registered camera first." };
  if (!calibrationWorkbenchState.frameUrl || !calibrationWorkbenchState.frameWidth || !calibrationWorkbenchState.frameHeight) {
    return { error: "Load a camera frame before defining source-image points." };
  }
  if (calibrationWorkbenchState.points.length !== 4) {
    return { error: "Select exactly four points on the image before saving or validating." };
  }
  const inBounds = calibrationWorkbenchState.points.every((point) =>
    Number.isFinite(point.x) && Number.isFinite(point.y) &&
    point.x >= 0 && point.x <= calibrationWorkbenchState.frameWidth &&
    point.y >= 0 && point.y <= calibrationWorkbenchState.frameHeight,
  );
  if (!inBounds) return { error: "One or more saved points fall outside this frame. Confirm the camera image resolution before submitting." };
  const dimensions = calibrationDimensionValues();
  if (dimensions.error) return dimensions;
  return {
    payload: {
      points: calibrationWorkbenchState.points.map((point) => ({ x: point.x, y: point.y })),
      real_world_width_m: dimensions.width,
      real_world_depth_m: dimensions.depth,
    },
  };
}

function renderCalibrationWorkbench() {
  const currentCamera = currentCalibrationCamera();
  const stateChip = elements.calibrationStatusBadge;
  const calibration = calibrationWorkbenchState.calibration;
  if (stateChip) {
    let label = "UNCONFIGURED";
    let status = "UNCONFIGURED";
    if (!calibrationWorkbenchState.selectedCameraId && calibrationWorkbenchState.cameras.length === 0 && calibrationWorkbenchState.cameraListError) {
      label = "REQUEST FAILED";
      status = "REQUEST FAILED";
    } else if (!calibrationWorkbenchState.selectedCameraId && calibrationWorkbenchState.cameras.length === 0) {
      label = "NO CAMERA";
      status = "UNCONFIGURED";
    } else if (calibrationWorkbenchState.busy === "loading") {
      label = "LOADING...";
      status = "LOADING";
    } else if (calibrationWorkbenchState.busy === "validate") {
      label = "VALIDATING...";
      status = "CHECKING";
    } else if (calibrationWorkbenchState.busy === "save") {
      label = "SAVING...";
      status = "CHECKING";
    } else if (calibrationWorkbenchState.busy === "clear") {
      label = "CLEARING...";
      status = "CHECKING";
    } else if (calibrationWorkbenchState.configRequestError) {
      label = "REQUEST FAILED";
      status = "REQUEST FAILED";
    } else if (calibration) {
      status = calibration.calibration_status || "UNCONFIGURED";
      label = status;
      if (status === "CALIBRATED") {
        label = calibration.calibration_enabled === false ? "CALIBRATED / DISABLED" : "CALIBRATED / ENABLED";
      }
    }
    stateChip.textContent = label;
    stateChip.dataset.state = status;
  }

  if (elements.calibrationCameraStatus) {
    elements.calibrationCameraStatus.textContent = currentCamera?.status || (calibrationWorkbenchState.cameraListError ? "LIST ERROR" : "NO CAMERA");
    elements.calibrationCameraStatus.dataset.state = currentCamera?.status || "UNKNOWN";
  }
  if (elements.calibrationSelectedCameraLabel) {
    elements.calibrationSelectedCameraLabel.textContent = currentCamera
      ? `${currentCamera.name || currentCamera.camera_id} / ${currentCamera.camera_id}`
      : "Select a registered camera";
  }
  if (elements.calibrationFrameDimensions) {
    elements.calibrationFrameDimensions.textContent = calibrationWorkbenchState.frameWidth && calibrationWorkbenchState.frameHeight
      ? `SOURCE IMAGE: ${calibrationWorkbenchState.frameWidth} × ${calibrationWorkbenchState.frameHeight} PX`
      : "SOURCE IMAGE: -- × -- PX";
  }

  renderCalibrationPointOverlay();
  renderCalibrationPointList();
  renderCalibrationFeedback();
  renderCalibrationValidationResult();

  const pointsCount = calibrationWorkbenchState.points.length;
  if (elements.calibrationPointCount) elements.calibrationPointCount.textContent = `${pointsCount} / 4 POINTS`;
  const busy = Boolean(calibrationWorkbenchState.busy);
  const cameraSelected = Boolean(currentCamera);
  const payloadState = getCalibrationDraftPayload();
  const draftReady = !payloadState.error;
  if (elements.calibrationCameraSelect) elements.calibrationCameraSelect.disabled = busy || calibrationWorkbenchState.cameras.length === 0;
  if (elements.calibrationRefreshFrameBtn) elements.calibrationRefreshFrameBtn.disabled = !cameraSelected || busy;
  if (elements.calibrationWidthInput) elements.calibrationWidthInput.disabled = !cameraSelected || !calibrationWorkbenchState.frameUrl || busy;
  if (elements.calibrationDepthInput) elements.calibrationDepthInput.disabled = !cameraSelected || !calibrationWorkbenchState.frameUrl || busy;
  if (elements.calibrationResetPointsBtn) elements.calibrationResetPointsBtn.disabled = !pointsCount || busy;
  if (elements.calibrationValidateBtn) elements.calibrationValidateBtn.disabled = !draftReady || busy;
  if (elements.calibrationSaveBtn) elements.calibrationSaveBtn.disabled = !draftReady || busy;
  if (elements.calibrationClearBtn) {
    elements.calibrationClearBtn.disabled = !cameraSelected || !calibration || calibration.calibration_status === "UNCONFIGURED" || busy;
  }

  if (elements.calibrationSaveSummary) {
    if (!cameraSelected) {
      elements.calibrationSaveSummary.textContent = calibrationWorkbenchState.cameraListError
        ? "Camera inventory is unavailable. Refresh the camera list to continue."
        : "Register a camera to begin calibration.";
    } else if (payloadState.error) {
      elements.calibrationSaveSummary.textContent = `${currentCamera.camera_id} · ${pointsCount}/4 points · ${payloadState.error}`;
    } else {
      const dimensionText = `${payloadState.payload.real_world_width_m} m × ${payloadState.payload.real_world_depth_m} m`;
      elements.calibrationSaveSummary.textContent = `${currentCamera.camera_id} · 4/4 source-pixel points · ${dimensionText}${calibrationWorkbenchState.isDirty ? " · Draft changes not saved" : ""}`;
    }
  }
}

function renderCalibrationPointList() {
  const list = elements.calibrationPointList;
  if (!list) return;
  list.replaceChildren();
  for (let index = 0; index < 4; index += 1) {
    const row = document.createElement("li");
    const name = document.createElement("b");
    name.textContent = `P${index + 1}`;
    const value = document.createElement("span");
    const point = calibrationWorkbenchState.points[index];
    value.textContent = point
      ? `${formatCalibrationCoordinate(point.x)}, ${formatCalibrationCoordinate(point.y)} px`
      : "Awaiting point";
    row.append(name, value);
    list.appendChild(row);
  }
}

function renderCalibrationFeedback() {
  const feedback = elements.calibrationFeedback;
  if (!feedback) return;
  feedback.textContent = calibrationWorkbenchState.feedback || "";
  feedback.dataset.tone = calibrationWorkbenchState.feedbackTone || "";
}

function renderCalibrationValidationResult() {
  const result = elements.calibrationValidationResult;
  if (!result) return;
  result.replaceChildren();
  result.dataset.tone = "";

  if (calibrationWorkbenchState.busy === "validate") {
    result.textContent = "CHECKING · Sending the proposed points to backend calibration validation...";
    result.dataset.tone = "pending";
    return;
  }
  if (calibrationWorkbenchState.validationState === "error") {
    result.textContent = `REQUEST FAILED · ${calibrationWorkbenchState.validationError}`;
    result.dataset.tone = "error";
    return;
  }
  if (!calibrationWorkbenchState.validation) return;

  const validation = calibrationWorkbenchState.validation;
  const heading = document.createElement("strong");
  heading.textContent = validation.is_valid
    ? `VALID · Backend status ${validation.status}.`
    : `INVALID · Backend status ${validation.status}.`;
  result.appendChild(heading);
  result.dataset.tone = validation.is_valid ? "success" : "error";

  if (!validation.is_valid) {
    const detail = document.createElement("div");
    detail.textContent = validation.error_message || "The backend rejected the proposed geometry.";
    result.appendChild(detail);
    return;
  }

  if (Number.isFinite(validation.estimated_coverage_area_m2)) {
    const coverage = document.createElement("div");
    coverage.textContent = `Estimated mapped area: ${validation.estimated_coverage_area_m2.toFixed(2)} m².`;
    result.appendChild(coverage);
  }
  if (Number.isFinite(validation.reprojection_error_px)) {
    const residual = document.createElement("div");
    residual.textContent = `Fitting-point reprojection residual: ${validation.reprojection_error_px.toFixed(3)} px. This is not an independent ground-truth accuracy measure.`;
    result.appendChild(residual);
  }
  if (Array.isArray(validation.sample_projections) && validation.sample_projections.length) {
    const projections = document.createElement("ul");
    validation.sample_projections.forEach((sample) => {
      const row = document.createElement("li");
      const groundPoint = sample.ground_point_m;
      row.textContent = Array.isArray(groundPoint) && groundPoint.length >= 2
        ? `${sample.vertex}: (${Number(groundPoint[0]).toFixed(2)}, ${Number(groundPoint[1]).toFixed(2)}) m`
        : `${sample.vertex}: projection unavailable`;
      projections.appendChild(row);
    });
    result.appendChild(projections);
  }
}

function warnIfCalibrationPointsOutsideFrame() {
  const { points, frameWidth, frameHeight } = calibrationWorkbenchState;
  if (points.length === 4 && frameWidth && frameHeight && points.some(
    (point) => point.x < 0 || point.x > frameWidth || point.y < 0 || point.y > frameHeight,
  )) {
    setCalibrationFeedback("Saved point coordinates extend beyond this frame. Check the camera resolution before changing or saving this calibration.", "error");
  }
}

async function calibrationResponseError(response) {
  try {
    const body = await response.json();
    const detail = body?.detail || body?.message;
    if (typeof detail === "string") return detail;
    if (detail) return JSON.stringify(detail);
  } catch (_) {
    // Some endpoints may return plain text for unexpected failures.
  }
  return response.statusText || `HTTP ${response.status}`;
}

async function validateCalibrationDraft() {
  if (calibrationWorkbenchState.busy) return;
  const draft = getCalibrationDraftPayload();
  if (draft.error) {
    setCalibrationFeedback(draft.error, "error");
    renderCalibrationWorkbench();
    return;
  }

  const cameraId = calibrationWorkbenchState.selectedCameraId;
  calibrationWorkbenchState.busy = "validate";
  calibrationWorkbenchState.validation = null;
  calibrationWorkbenchState.validationState = "pending";
  calibrationWorkbenchState.validationError = "";
  setCalibrationFeedback(`Validating the proposed geometry for ${cameraId}...`, "pending");
  renderCalibrationWorkbench();
  try {
    const response = await authFetch(`${CONFIG.apiBase}/api/v1/cameras/${encodeURIComponent(cameraId)}/calibration/validate`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(draft.payload),
    });
    if (!response.ok) throw new Error(await calibrationResponseError(response));
    const validation = await response.json();
    if (cameraId !== calibrationWorkbenchState.selectedCameraId) return;
    calibrationWorkbenchState.validation = validation;
    calibrationWorkbenchState.validationState = validation.is_valid ? "valid" : "invalid";
    calibrationWorkbenchState.validationError = "";
    setCalibrationFeedback(
      validation.is_valid
        ? "Backend validation passed for this proposal. It is not saved until you choose Save calibration."
        : `Backend validation rejected this proposal: ${validation.error_message || validation.status}.`,
      validation.is_valid ? "success" : "error",
    );
  } catch (error) {
    if (cameraId !== calibrationWorkbenchState.selectedCameraId) return;
    calibrationWorkbenchState.validationState = "error";
    calibrationWorkbenchState.validationError = error.message || "Validation request failed.";
    setCalibrationFeedback("Validation request failed. The saved calibration state has not changed.", "error");
  } finally {
    if (cameraId === calibrationWorkbenchState.selectedCameraId) {
      calibrationWorkbenchState.busy = "";
      renderCalibrationWorkbench();
    }
  }
}

async function saveCalibrationDraft() {
  if (calibrationWorkbenchState.busy) return;
  const draft = getCalibrationDraftPayload();
  if (draft.error) {
    setCalibrationFeedback(draft.error, "error");
    renderCalibrationWorkbench();
    return;
  }

  const cameraId = calibrationWorkbenchState.selectedCameraId;
  const requestId = calibrationWorkbenchState.selectionRequestId;
  calibrationWorkbenchState.busy = "save";
  setCalibrationFeedback(`Saving four points for ${cameraId}...`, "pending");
  renderCalibrationWorkbench();
  try {
    const response = await authFetch(`${CONFIG.apiBase}/api/v1/cameras/${encodeURIComponent(cameraId)}/calibration`, {
      method: "PUT",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ camera_id: cameraId, calibration_enabled: true, ...draft.payload }),
    });
    if (!response.ok) throw new Error(await calibrationResponseError(response));
    const saved = await response.json();
    if (requestId !== calibrationWorkbenchState.selectionRequestId) return;
    calibrationWorkbenchState.calibration = saved;
    calibrationWorkbenchState.validation = null;
    calibrationWorkbenchState.validationState = "idle";
    calibrationWorkbenchState.isDirty = false;
    if (saved.calibration_status !== "CALIBRATED" || saved.calibration_enabled === false) {
      setCalibrationFeedback(`Save completed. Backend reports ${saved.calibration_status}${saved.calibration_enabled === false ? " / DISABLED" : ""}; metric calibration is not active.`, "error");
    } else {
      setCalibrationFeedback(`Saved for ${cameraId}. Backend status: ${saved.calibration_status}, enabled: ${saved.calibration_enabled}.`, "success");
    }
    await loadCalibrationConfiguration(cameraId, requestId);
    if (requestId === calibrationWorkbenchState.selectionRequestId && calibrationWorkbenchState.configRequestError) {
      setCalibrationFeedback("Save completed, but refreshing the authoritative calibration state failed. Refresh the camera selection to retry.", "error");
    }
    refreshCalibrationSceneContext();
  } catch (error) {
    if (requestId === calibrationWorkbenchState.selectionRequestId) {
      setCalibrationFeedback(`Calibration was not saved: ${error.message || "Request failed."}`, "error");
    }
  } finally {
    if (requestId === calibrationWorkbenchState.selectionRequestId) {
      calibrationWorkbenchState.busy = "";
      renderCalibrationWorkbench();
    }
  }
}

async function clearCameraCalibration() {
  if (calibrationWorkbenchState.busy || !calibrationWorkbenchState.selectedCameraId) return;
  const cameraId = calibrationWorkbenchState.selectedCameraId;
  const camera = currentCalibrationCamera();
  if (!window.confirm(`Remove the saved calibration for ${camera?.name || cameraId} (${cameraId})? Scenes from this camera will use IMAGE_SPACE until a calibration is saved again.`)) return;

  const requestId = calibrationWorkbenchState.selectionRequestId;
  calibrationWorkbenchState.busy = "clear";
  setCalibrationFeedback(`Removing calibration for ${cameraId}...`, "pending");
  renderCalibrationWorkbench();
  try {
    const response = await authFetch(`${CONFIG.apiBase}/api/v1/cameras/${encodeURIComponent(cameraId)}/calibration`, { method: "DELETE" });
    if (!response.ok) throw new Error(await calibrationResponseError(response));
    const result = await response.json();
    if (requestId !== calibrationWorkbenchState.selectionRequestId) return;
    calibrationWorkbenchState.points = [];
    calibrationWorkbenchState.validation = null;
    calibrationWorkbenchState.validationState = "idle";
    calibrationWorkbenchState.isDirty = false;
    await loadCalibrationConfiguration(cameraId, requestId);
    if (requestId !== calibrationWorkbenchState.selectionRequestId) return;
    if (calibrationWorkbenchState.configRequestError) {
      setCalibrationFeedback("Delete completed, but the refreshed calibration status could not be retrieved.", "error");
    } else {
      setCalibrationFeedback(
        result.deleted
          ? "Calibration removed. This camera has no active metric calibration; matching scenes remain in IMAGE_SPACE."
          : "No saved calibration existed for this camera. The backend reports it as UNCONFIGURED.",
        "success",
      );
    }
    refreshCalibrationSceneContext();
  } catch (error) {
    if (requestId === calibrationWorkbenchState.selectionRequestId) {
      setCalibrationFeedback(`Calibration could not be cleared: ${error.message || "Request failed."}`, "error");
    }
  } finally {
    if (requestId === calibrationWorkbenchState.selectionRequestId) {
      calibrationWorkbenchState.busy = "";
      renderCalibrationWorkbench();
    }
  }
}

async function refreshCalibrationSceneContext() {
  const display = elements.calibrationSceneMetricContext;
  if (!display) return;
  display.textContent = "Reading the current scene's stored source camera...";
  try {
    // Intentionally omit camera_id: backend source_camera_id remains authoritative.
    const response = await authFetch(`${CONFIG.apiBase}/api/v1/scene/graph`);
    if (!response.ok) throw new Error(await calibrationResponseError(response));
    const scene = await response.json();
    const sourceCameraId = scene.source_camera_id || scene.camera_id || "unassigned";
    const spatialBasis = scene.spatial_basis || "IMAGE_SPACE";
    const calibrationStatus = scene.calibration_status || "UNCONFIGURED";
    const selectedCameraId = calibrationWorkbenchState.selectedCameraId;
    const cameraMatch = selectedCameraId
      ? (selectedCameraId === sourceCameraId ? "MATCHES SELECTED CAMERA" : "DIFFERENT FROM SELECTED CAMERA")
      : "NO WORKBENCH CAMERA SELECTED";
    const entities = Array.isArray(scene.entities) ? scene.entities : [];
    const relationships = Array.isArray(scene.relationships) ? scene.relationships : [];
    const hasGroundPosition = entities.some((entity) => entity.position?.ground_plane != null);
    const hasPixelPosition = entities.some((entity) => entity.position?.contact_point != null);
    const hasGroundDistance = relationships.some((relationship) => relationship.distance_m != null);
    const hasPixelDistance = relationships.some((relationship) => relationship.distance_px != null);
    const hasGroundSpeed = entities.some((entity) => entity.state?.speed_m_per_s != null);
    const hasPixelSpeed = entities.some((entity) => entity.state?.speed_px_per_s != null);
    const positionBasis = hasGroundPosition ? "GROUND_PLANE_APPROXIMATION" : (hasPixelPosition ? "IMAGE_SPACE" : "unavailable");
    const distanceBasis = hasGroundDistance ? "GROUND_PLANE_APPROXIMATION" : (hasPixelDistance ? "IMAGE_SPACE" : "unavailable");
    const speedBasis = hasGroundSpeed
      ? "GROUND_PLANE_APPROXIMATION (m/s)"
      : (hasPixelSpeed ? "IMAGE_SPACE (px/s); m/s unavailable" : "unavailable");
    const tth = scene.time_to_hazard || {};
    const tthBasis = tth.available === true && tth.time_to_hazard_seconds != null
      ? "IMAGE_SPACE estimate"
      : "unavailable";
    display.textContent = `Source ${sourceCameraId} · calibration ${calibrationStatus} · ${cameraMatch} · position ${positionBasis} · distance ${distanceBasis} · speed ${speedBasis} · TTH ${tthBasis}`;
  } catch (error) {
    display.textContent = `Scene context unavailable · ${error.message || "request failed"}`;
  }
}

function escapeHtml(str) {
  if (str == null) return "";
  return String(str)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

async function loadCameras() {
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/cameras`);
    if (!res.ok) {
      markCalibrationCameraListFailure(await calibrationResponseError(res));
      return;
    }
    const data = await res.json();
    let calibrations = [];
    let calibrationStatusAvailable = false;
    try {
      const calibrationRes = await authFetch(`${CONFIG.apiBase}/api/v1/calibrations`);
      if (calibrationRes.ok) {
        calibrations = await calibrationRes.json();
        calibrationStatusAvailable = true;
      }
    } catch (err) {
      console.warn("Failed to load camera calibration IDs:", err);
    }
    renderCamerasList(data, calibrations, calibrationStatusAvailable);
  } catch (err) {
    console.warn("Failed to load cameras list:", err);
    markCalibrationCameraListFailure(err.message || "Registered cameras could not be loaded.");
  }
}

function renderCamerasList(data, calibrations = [], calibrationStatusAvailable = false) {
  if (elements.cameraCountBadge) {
    elements.cameraCountBadge.textContent =
      `Registered Cameras: ${data.total_cameras} Active (${data.connected_count} Connected)`;
  }

  const statCamEl = document.getElementById("statActiveCameras");
  if (statCamEl) {
    const conn = String(data.connected_count || 0).padStart(2, '0');
    const tot = String(data.total_cameras || 0).padStart(2, '0');
    statCamEl.textContent = `${conn} / ${tot}`;
  }

  if (elements.viewportCameraSelect) {
    const currentVal = elements.viewportCameraSelect.value;
    elements.viewportCameraSelect.innerHTML = "";
    if (!data.cameras || data.cameras.length === 0) {
      const opt = document.createElement("option");
      opt.value = "cam_01";
      opt.textContent = "cam_01 (Default)";
      elements.viewportCameraSelect.appendChild(opt);
    } else {
      data.cameras.forEach((cam) => {
        const opt = document.createElement("option");
        opt.value = cam.camera_id;
        opt.textContent = `${cam.camera_id} (${cam.name})`;
        if (cam.camera_id === currentVal) opt.selected = true;
        elements.viewportCameraSelect.appendChild(opt);
      });
    }
  }

  if (elements.analysisCameraSelect) {
    const currentVal = elements.analysisCameraSelect.value;
    const sourceOptions = new Map();
    const calibrationsByCameraId = new Map(
      (calibrations || []).filter((calibration) => calibration.camera_id).map((calibration) => [calibration.camera_id, calibration]),
    );
    (data.cameras || []).forEach((cam) => {
      sourceOptions.set(cam.camera_id, cam.name || cam.camera_id);
    });
    (calibrations || []).forEach((calibration) => {
      if (calibration.camera_id && !sourceOptions.has(calibration.camera_id)) {
        sourceOptions.set(calibration.camera_id, "Configured calibration");
      }
    });

    elements.analysisCameraSelect.innerHTML = "";
    const unassigned = document.createElement("option");
    unassigned.value = "";
    unassigned.textContent = "Unknown / unassigned";
    elements.analysisCameraSelect.appendChild(unassigned);
    sourceOptions.forEach((name, cameraId) => {
      const option = document.createElement("option");
      option.value = cameraId;
      option.textContent = `${cameraId} (${name})`;
      const calibration = calibrationsByCameraId.get(cameraId);
      option.dataset.calibrationConfigured = String(Boolean(calibration));
      option.dataset.calibrationStatus = calibration?.calibration_status || "";
      option.dataset.calibrationEnabled = calibration ? String(calibration.calibration_enabled !== false) : "";
      elements.analysisCameraSelect.appendChild(option);
    });
    elements.analysisCameraSelect.value = sourceOptions.has(currentVal) ? currentVal : "";
    elements.analysisCameraSelect.dataset.calibrationStatusAvailable = String(calibrationStatusAvailable);
    updateUploadCameraStatus();
  }

  syncCalibrationCameraList(data.cameras || []);

  if (!elements.camerasGrid) return;
  elements.camerasGrid.innerHTML = "";

  if (!data.cameras || data.cameras.length === 0) {
    elements.camerasGrid.innerHTML = `
      <div style="grid-column: 1/-1; padding: 16px; text-align: center; color: var(--text-muted); font-size: 12px; background: rgba(0,0,0,0.2); border-radius: 6px;">
        No active CCTV streams registered yet. Use the form above to add an RTSP feed or local CCTV video.
      </div>
    `;
    return;
  }

  data.cameras.forEach((cam) => {
    const card = document.createElement("div");
    card.style.cssText = "background: rgba(17, 23, 36, 0.9); border: 1px solid var(--border-color); border-radius: 6px; padding: 10px; display: flex; flex-direction: column; gap: 8px;";

    let statusColor = "var(--text-muted)";
    if (cam.status === "CONNECTED") statusColor = "var(--color-low)";
    else if (cam.status === "RECONNECTING" || cam.status === "CONNECTING") statusColor = "var(--color-medium)";
    else if (cam.status === "ERROR") statusColor = "var(--color-critical)";

    card.innerHTML = `
      <div style="display: flex; justify-content: space-between; align-items: center;">
        <div>
          <strong style="color: var(--text-primary); font-size: 12px;">${escapeHtml(cam.camera_id)}</strong>
          <span style="color: var(--text-secondary); font-size: 11px; margin-left: 4px;">${escapeHtml(cam.name)}</span>
        </div>
        <span style="font-size: 10px; font-weight: 700; color: ${statusColor}; background: rgba(0,0,0,0.4); padding: 2px 6px; border-radius: 4px; border: 1px solid ${statusColor};">
          ${cam.status}
        </span>
      </div>

      <div style="font-size: 10px; color: var(--text-muted); font-family: var(--font-mono); display: flex; align-items: center; justify-content: space-between; gap: 4px;" title="${escapeHtml(cam.source_sanitized)}">
        <span style="overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 60%;">SRC: ${escapeHtml(cam.source_sanitized)}</span>
        ${!String(cam.source_sanitized || "").toLowerCase().startsWith("rtsp://")
          ? `<span style="background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.4); font-size: 8px; padding: 1px 4px; border-radius: 3px; font-weight: 700; white-space: nowrap;">VIDEO FILE / DEMO SOURCE</span>`
          : `<span style="background: rgba(52, 211, 153, 0.15); color: #34d399; border: 1px solid rgba(52, 211, 153, 0.4); font-size: 8px; padding: 1px 4px; border-radius: 3px; font-weight: 700; white-space: nowrap;">RTSP CAMERA</span>`
        }
      </div>

      <div style="display: grid; grid-template-columns: repeat(3, 1fr); gap: 4px; background: rgba(0,0,0,0.3); padding: 6px; border-radius: 4px; font-size: 10px; text-align: center;">
        <div><div style="color: var(--text-muted); font-size: 9px;">INGEST</div><strong>${cam.ingest_fps.toFixed(1)} FPS</strong></div>
        <div><div style="color: var(--text-muted); font-size: 9px;">AI PROC</div><strong style="color: var(--color-info);">${cam.processing_fps.toFixed(1)} FPS</strong></div>
        <div><div style="color: var(--text-muted); font-size: 9px;">LATENCY</div><strong>${cam.avg_latency_ms.toFixed(0)} ms</strong></div>
        <div><div style="color: var(--text-muted); font-size: 9px;">DROPPED</div><strong style="${cam.dropped_frames_count > 0 ? 'color: var(--color-medium);' : ''}">${cam.dropped_frames_count}</strong></div>
        <div><div style="color: var(--text-muted); font-size: 9px;">TRACKS</div><strong>${cam.active_tracks_count}</strong></div>
        <div><div style="color: var(--text-muted); font-size: 9px;">INCIDENTS</div><strong style="${cam.active_incidents_count > 0 ? 'color: var(--color-critical);' : ''}">${cam.active_incidents_count}</strong></div>
      </div>

      <div style="display: flex; gap: 6px; margin-top: 2px;">
        ${cam.is_active
          ? `<button class="btn btn-sm btn-danger" style="flex: 1; padding: 4px 6px; font-size: 11px;" onclick="stopCameraStream('${escapeHtml(cam.camera_id)}')">⏹ Stop</button>`
          : `<button class="btn btn-sm btn-primary" style="flex: 1; padding: 4px 6px; font-size: 11px;" onclick="startCameraStream('${escapeHtml(cam.camera_id)}')">▶ Start</button>`
        }
        <button class="btn btn-sm" style="flex: 1; padding: 4px 6px; font-size: 11px;" onclick="selectCameraForViewing('${escapeHtml(cam.camera_id)}')">📺 View Live</button>
        <button class="btn btn-sm" style="padding: 4px 6px; font-size: 11px; color: var(--color-critical);" onclick="removeCameraStream('${escapeHtml(cam.camera_id)}')">✕</button>
      </div>
    `;
    elements.camerasGrid.appendChild(card);
  });
}

function updateUploadCameraStatus() {
  if (!elements.analysisCameraSelect || !elements.analysisCameraCalibrationNotice) return;

  const selected = elements.analysisCameraSelect.selectedOptions[0];
  const cameraId = elements.analysisCameraSelect.value;
  if (!cameraId) {
    elements.analysisCameraCalibrationNotice.textContent =
      "Unknown source. Calibration unavailable; metric values will remain in image space.";
    return;
  }

  if (elements.analysisCameraSelect.dataset.calibrationStatusAvailable !== "true") {
    elements.analysisCameraCalibrationNotice.textContent =
      "Calibration status unavailable. The source camera identity will be preserved.";
    return;
  }

  const status = selected?.dataset.calibrationStatus;
  const enabled = selected?.dataset.calibrationEnabled;
  if (selected?.dataset.calibrationConfigured !== "true") {
    elements.analysisCameraCalibrationNotice.textContent =
      `No calibration configured for ${cameraId}. Analysis will remain in image space.`;
  } else if (enabled === "false" || status !== "CALIBRATED") {
    elements.analysisCameraCalibrationNotice.textContent =
      `Calibration status: ${status || "unavailable"} for ${cameraId}; metric values are unavailable.`;
  } else {
    elements.analysisCameraCalibrationNotice.textContent =
      `Configured calibration status: ${status} for ${cameraId}. Calibration is camera-scoped.`;
  }
}

async function registerCameraFromForm() {
  const camId = (elements.camIdInput?.value || "").trim();
  const name = (elements.camNameInput?.value || "").trim();
  const source = (elements.camSourceInput?.value || "").trim();
  const sampling = parseInt(elements.camSamplingInput?.value || "1", 10);
  const maxFps = parseFloat(elements.camMaxFpsInput?.value || "10.0");

  if (!camId || !source) {
    alert("Please provide both Camera ID and Stream Source.");
    return;
  }

  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/cameras`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        camera_id: camId,
        name: name || camId,
        source: source,
        sampling_interval: sampling,
        max_processing_fps: maxFps,
        auto_start: true,
      }),
    });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      alert("Failed to register camera: " + (err.detail || res.statusText));
      return;
    }
    selectCameraForViewing(camId);
    await loadCameras();
  } catch (err) {
    alert("Error registering camera: " + err.message);
  }
}

window.startCameraStream = async function (camId) {
  try {
    await authFetch(`${CONFIG.apiBase}/api/v1/cameras/${encodeURIComponent(camId)}/start`, { method: "POST" });
    await loadCameras();
  } catch (err) {
    console.error("Failed to start camera:", err);
  }
};

window.stopCameraStream = async function (camId) {
  try {
    await authFetch(`${CONFIG.apiBase}/api/v1/cameras/${encodeURIComponent(camId)}/stop`, { method: "POST" });
    await loadCameras();
  } catch (err) {
    console.error("Failed to stop camera:", err);
  }
};

window.removeCameraStream = async function (camId) {
  if (!confirm(`Are you sure you want to remove camera '${camId}'?`)) return;
  try {
    await authFetch(`${CONFIG.apiBase}/api/v1/cameras/${encodeURIComponent(camId)}`, { method: "DELETE" });
    if (currentViewingCameraId === camId) {
      currentViewingCameraId = null;
      if (elements.viewportLiveStream) elements.viewportLiveStream.style.display = "none";
      if (elements.canvas) elements.canvas.style.display = "block";
    }
    await loadCameras();
  } catch (err) {
    console.error("Failed to remove camera:", err);
  }
};

window.selectCameraForViewing = function (camId) {
  state.currentMediaSource = "camera";
  state.isSingleFrame = false;
  currentViewingCameraId = camId;
  if (elements.viewportCameraSelect) {
    elements.viewportCameraSelect.value = camId;
  }
  if (elements.viewportStateBadge) {
    elements.viewportStateBadge.textContent = `[LIVE STREAM: ${camId}]`;
  }

  // Hide canvas, image, video; show live stream img
  if (elements.canvas) elements.canvas.style.display = "none";
  if (elements.viewportImage) elements.viewportImage.style.display = "none";
  if (elements.viewportVideo) elements.viewportVideo.style.display = "none";
  if (elements.viewportCompletedSummary) elements.viewportCompletedSummary.style.display = "none";

  const tokenParam = state.authToken ? `&token=${encodeURIComponent(state.authToken)}` : "";
  if (elements.viewportLiveStream) {
    elements.viewportLiveStream.style.display = "block";
    elements.viewportLiveStream.src = `${CONFIG.apiBase}/api/v1/cameras/${encodeURIComponent(camId)}/snapshot?t=` + Date.now() + tokenParam;
  }

  renderZonesOverlay();

  // Start polling snapshot & assessment for this camera
  if (activeStreamPollingInterval) clearInterval(activeStreamPollingInterval);
  activeStreamPollingInterval = setInterval(() => {
    if (currentViewingCameraId !== camId) {
      clearInterval(activeStreamPollingInterval);
      return;
    }
    if (elements.viewportLiveStream && elements.viewportLiveStream.style.display !== "none") {
      elements.viewportLiveStream.src = `${CONFIG.apiBase}/api/v1/cameras/${encodeURIComponent(camId)}/snapshot?t=` + Date.now() + tokenParam;
    }
    authFetch(`${CONFIG.apiBase}/api/v1/cameras/${encodeURIComponent(camId)}`)
      .then((r) => r.json())
      .then((d) => {
        if (d && d.camera) {
          if (elements.fpsBadge) elements.fpsBadge.textContent = `${d.camera.processing_fps.toFixed(1)} FPS`;
          if (elements.latencyBadge) elements.latencyBadge.textContent = `${d.camera.avg_latency_ms.toFixed(0)} ms`;
          if (elements.tracksBadge) elements.tracksBadge.textContent = `Tracks: ${d.camera.active_tracks_count}`;
          if (elements.incidentsBadge) elements.incidentsBadge.textContent = `Incidents: ${d.camera.active_incidents_count}`;
        }
        if (d && d.latest_assessment) {
          renderSceneAssessment(d.latest_assessment);
        }
      })
      .catch(() => {});
  }, 600);
};

// =========================================================================
// STEP 23: AUTHENTICATION, RBAC & ADMINISTRATION CONTROLLERS
// =========================================================================

function setupAuthControls() {
  if (elements.loginForm) {
    elements.loginForm.addEventListener("submit", handleLoginSubmit);
  }
  if (elements.loginSubmitBtn) {
    elements.loginSubmitBtn.addEventListener("click", handleLoginSubmit);
  }
  if (elements.logoutBtn) {
    elements.logoutBtn.addEventListener("click", handleLogout);
  }
  if (elements.changePasswordBtn) {
    elements.changePasswordBtn.addEventListener("click", () => {
      if (elements.changePasswordModal) elements.changePasswordModal.style.display = "flex";
      if (elements.changePasswordMsg) elements.changePasswordMsg.style.display = "none";
      if (elements.changePasswordForm) elements.changePasswordForm.reset();
    });
  }
  if (elements.closeChangePasswordBtn) {
    elements.closeChangePasswordBtn.addEventListener("click", () => {
      if (elements.changePasswordModal) elements.changePasswordModal.style.display = "none";
    });
  }
  if (elements.savePasswordBtn) {
    elements.savePasswordBtn.addEventListener("click", handleChangePasswordSubmit);
  }
  if (elements.createUserModalBtn) {
    elements.createUserModalBtn.addEventListener("click", () => {
      if (elements.createUserModal) elements.createUserModal.style.display = "flex";
      if (elements.createUserMsg) elements.createUserMsg.style.display = "none";
      if (elements.createUserForm) elements.createUserForm.reset();
    });
  }
  if (elements.closeCreateUserBtn) {
    elements.closeCreateUserBtn.addEventListener("click", () => {
      if (elements.createUserModal) elements.createUserModal.style.display = "none";
    });
  }
  if (elements.saveNewUserBtn) {
    elements.saveNewUserBtn.addEventListener("click", handleCreateUserSubmit);
  }
  if (elements.refreshAuditBtn) {
    elements.refreshAuditBtn.addEventListener("click", () => {
      fetchAuditLogs();
    });
  }
  if (elements.applyAuditFiltersBtn) {
    elements.applyAuditFiltersBtn.addEventListener("click", () => {
      state.auditFilters.event_type = elements.auditFilterEvent?.value || "";
      state.auditFilters.outcome = elements.auditFilterOutcome?.value || "";
      state.auditFilters.actor = elements.auditFilterActor?.value.trim() || "";
      state.auditFilters.page = 1;
      fetchAuditLogs();
    });
  }
  if (elements.auditPrevPageBtn) {
    elements.auditPrevPageBtn.addEventListener("click", () => {
      if (state.auditFilters.page > 1) {
        state.auditFilters.page--;
        fetchAuditLogs();
      }
    });
  }
  if (elements.auditNextPageBtn) {
    elements.auditNextPageBtn.addEventListener("click", () => {
      state.auditFilters.page++;
      fetchAuditLogs();
    });
  }
}

async function initAuthSession() {
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/auth/me`);
    if (res.ok) {
      state.currentUser = await res.json();
      applyUserPermissions(state.currentUser);
    } else {
      showLoginModal();
    }
  } catch (err) {
    console.warn("Auth initialization failed:", err);
  }
}

function showLoginModal(msg) {
  if (elements.loginModal) {
    elements.loginModal.style.display = "flex";
  }
  if (msg && elements.loginErrorMsg) {
    elements.loginErrorMsg.textContent = msg;
    elements.loginErrorMsg.style.display = "block";
  }
}

function hideLoginModal() {
  if (elements.loginModal) {
    elements.loginModal.style.display = "none";
  }
  if (elements.loginErrorMsg) {
    elements.loginErrorMsg.style.display = "none";
  }
}

async function handleLoginSubmit(e) {
  if (e) e.preventDefault();
  const username = elements.loginUsername?.value.trim() || "";
  const password = elements.loginPassword?.value || "";

  if (!username || !password) {
    if (elements.loginErrorMsg) {
      elements.loginErrorMsg.textContent = "Please enter both username and password.";
      elements.loginErrorMsg.style.display = "block";
    }
    return;
  }

  try {
    const res = await fetch(`${CONFIG.apiBase}/api/v1/auth/login`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ username, password }),
    });

    const data = await res.json();
    if (!res.ok) {
      if (elements.loginErrorMsg) {
        elements.loginErrorMsg.textContent = data.detail || "Authentication failed.";
        elements.loginErrorMsg.style.display = "block";
      }
      return;
    }

    const token = data.access_token || data.token;
    state.authToken = token;
    state.currentUser = data.user;
    localStorage.setItem("intelliwatch_token", token);

    hideLoginModal();
    applyUserPermissions(data.user);

    fetchData();
  } catch (err) {
    if (elements.loginErrorMsg) {
      elements.loginErrorMsg.textContent = `Login error: ${err.message}`;
      elements.loginErrorMsg.style.display = "block";
    }
  }
}

async function handleLogout() {
  try {
    await authFetch(`${CONFIG.apiBase}/api/v1/auth/logout`, { method: "POST" });
  } catch (err) {
    console.warn("Logout request failed:", err);
  } finally {
    state.authToken = "";
    state.currentUser = null;
    localStorage.removeItem("intelliwatch_token");
    showLoginModal("You have been successfully logged out.");
  }
}

function applyUserPermissions(user) {
  if (!user) return;

  if (elements.headerUsername) elements.headerUsername.textContent = user.username;
  if (elements.headerRoleBadge) {
    elements.headerRoleBadge.textContent = user.role;
    elements.headerRoleBadge.className = `badge badge-${user.role.toLowerCase()}`;
  }

  const isAdmin = user.role === "ADMIN";
  const isSafetyManager = user.role === "SAFETY_MANAGER" || isAdmin;
  const isOperator = user.role === "OPERATOR" || isSafetyManager;

  // Admin tabs visibility
  if (elements.tabUsersBtn) elements.tabUsersBtn.style.display = isAdmin ? "inline-block" : "none";
  if (elements.tabAuditBtn) elements.tabAuditBtn.style.display = isAdmin ? "inline-block" : "none";

  // Operator lifecycle actions
  if (elements.btnModalDismiss) {
    elements.btnModalDismiss.style.display = isSafetyManager ? "inline-block" : "none";
  }

  if (isAdmin) {
    fetchUsers();
    fetchAuditLogs();
  }
}

async function handleChangePasswordSubmit() {
  const currentPassword = elements.currentPasswordInput?.value || "";
  const newPassword = elements.newPasswordInput?.value || "";
  const confirmPassword = elements.confirmPasswordInput?.value || "";

  if (!currentPassword || !newPassword) {
    alert("Please fill out all password fields.");
    return;
  }
  if (newPassword !== confirmPassword) {
    alert("New passwords do not match.");
    return;
  }
  if (newPassword.length < 8) {
    alert("New password must be at least 8 characters long.");
    return;
  }

  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/auth/change-password`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        current_password: currentPassword,
        new_password: newPassword,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      if (elements.changePasswordMsg) {
        elements.changePasswordMsg.textContent = data.detail || "Password change failed.";
        elements.changePasswordMsg.style.color = "#ef4444";
        elements.changePasswordMsg.style.display = "block";
      }
      return;
    }

    if (elements.changePasswordMsg) {
      elements.changePasswordMsg.textContent = "Password updated successfully!";
      elements.changePasswordMsg.style.color = "#10b981";
      elements.changePasswordMsg.style.display = "block";
    }

    setTimeout(() => {
      if (elements.changePasswordModal) elements.changePasswordModal.style.display = "none";
    }, 1200);
  } catch (err) {
    alert(`Error updating password: ${err.message}`);
  }
}

async function fetchUsers() {
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/users`);
    if (!res.ok) return;
    const data = await res.json();
    state.users = data.users || [];
    renderUsersTable(state.users);
  } catch (err) {
    console.warn("Could not fetch user list:", err);
  }
}

function renderUsersTable(users) {
  if (!elements.userTableBody) return;
  if (!users || users.length === 0) {
    elements.userTableBody.innerHTML = `<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 12px;">No user accounts found.</td></tr>`;
    return;
  }

  elements.userTableBody.innerHTML = users.map((u) => {
    const cams = u.permitted_cameras ? u.permitted_cameras.join(", ") : "All Cameras";
    const isActive = u.status === "ACTIVE";
    return `
      <tr>
        <td style="font-weight: 600; color: #fff;">${escapeHtml(u.username)}</td>
        <td><span class="badge badge-${u.role.toLowerCase()}">${u.role}</span></td>
        <td><span class="badge ${isActive ? 'badge-active' : 'badge-disabled'}">${u.status}</span></td>
        <td style="color: var(--text-muted); font-size: 9px;">${escapeHtml(cams)}</td>
        <td>
          <button class="btn btn-sm ${isActive ? 'btn-danger' : 'btn-primary'}" style="padding: 2px 6px; font-size: 9px;" onclick="toggleUserStatus('${u.user_id}', '${isActive ? 'DISABLED' : 'ACTIVE'}')">
            ${isActive ? 'Disable' : 'Enable'}
          </button>
        </td>
      </tr>
    `;
  }).join("");
}

window.toggleUserStatus = async function (userId, newStatus) {
  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/users/${encodeURIComponent(userId)}`, {
      method: "PATCH",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ status: newStatus }),
    });

    if (!res.ok) {
      const err = await res.json();
      alert(`Action failed: ${err.detail || "Unable to update status"}`);
      return;
    }
    fetchUsers();
    fetchAuditLogs();
  } catch (err) {
    alert(`Failed to update user: ${err.message}`);
  }
};

async function handleCreateUserSubmit() {
  const username = elements.createUsernameInput?.value.trim() || "";
  const role = elements.createUserRoleSelect?.value || "OPERATOR";
  const password = elements.createPasswordInput?.value || "";
  const camText = elements.createPermittedCamerasInput?.value.trim() || "";
  const permittedCameras = camText ? camText.split(",").map((c) => c.trim()).filter(Boolean) : null;

  if (!username || !password) {
    alert("Please provide both username and password.");
    return;
  }
  if (password.length < 8) {
    alert("Password must be at least 8 characters.");
    return;
  }

  try {
    const res = await authFetch(`${CONFIG.apiBase}/api/v1/users`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        username,
        role,
        password,
        permitted_cameras: permittedCameras,
      }),
    });

    const data = await res.json();
    if (!res.ok) {
      if (elements.createUserMsg) {
        elements.createUserMsg.textContent = data.detail || "Failed to create user.";
        elements.createUserMsg.style.color = "#ef4444";
        elements.createUserMsg.style.display = "block";
      }
      return;
    }

    if (elements.createUserMsg) {
      elements.createUserMsg.textContent = "User created successfully!";
      elements.createUserMsg.style.color = "#10b981";
      elements.createUserMsg.style.display = "block";
    }

    fetchUsers();
    fetchAuditLogs();
    setTimeout(() => {
      if (elements.createUserModal) elements.createUserModal.style.display = "none";
    }, 1000);
  } catch (err) {
    alert(`Could not create user: ${err.message}`);
  }
}

async function fetchAuditLogs() {
  try {
    let url = `${CONFIG.apiBase}/api/v1/audit-logs?page=${state.auditFilters.page}&page_size=${state.auditFilters.pageSize}`;
    if (state.auditFilters.event_type) url += `&event_type=${encodeURIComponent(state.auditFilters.event_type)}`;
    if (state.auditFilters.outcome) url += `&outcome=${encodeURIComponent(state.auditFilters.outcome)}`;
    if (state.auditFilters.actor) url += `&actor=${encodeURIComponent(state.auditFilters.actor)}`;

    const res = await authFetch(url);
    if (!res.ok) return;
    const data = await res.json();
    state.auditLogs = data.items || [];
    renderAuditLogs(data);
  } catch (err) {
    console.warn("Could not fetch audit logs:", err);
  }
}

function renderAuditLogs(data) {
  if (!elements.auditLogList) return;
  const items = data.items || [];

  if (elements.auditPageInfo) {
    elements.auditPageInfo.textContent = `Page ${data.page} of ${data.total_pages || 1} (${data.total} total)`;
  }
  if (elements.auditPrevPageBtn) elements.auditPrevPageBtn.disabled = data.page <= 1;
  if (elements.auditNextPageBtn) elements.auditNextPageBtn.disabled = data.page >= data.total_pages;

  if (items.length === 0) {
    elements.auditLogList.innerHTML = `<div style="text-align: center; color: var(--text-muted); font-size: 11px; padding: 16px;">No audit records found.</div>`;
    return;
  }

  elements.auditLogList.innerHTML = items.map((entry) => {
    const ts = new Date(entry.timestamp * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
    const dt = new Date(entry.timestamp * 1000).toLocaleDateString();
    const outcomeCls = `outcome-${entry.action_outcome.toLowerCase()}`;
    const actor = entry.actor_username || "anonymous";

    return `
      <div class="audit-card">
        <div class="audit-card-header">
          <div>
            <strong style="color: #93c5fd;">${escapeHtml(entry.event_type)}</strong>
            <span style="color: var(--text-muted); margin-left: 6px;">by <strong>${escapeHtml(actor)}</strong></span>
          </div>
          <span class="outcome-pill ${outcomeCls}">${entry.action_outcome}</span>
        </div>
        <div class="audit-card-body">
          <span>${entry.resource_type ? `${entry.resource_type}:${entry.resource_id || ''}` : (entry.ip_address || '')}</span>
          <span style="font-family: var(--font-mono);">${dt} ${ts}</span>
        </div>
      </div>
    `;
  }).join("");
}

// -------------------------------------------------------------
// STEP 12 & 13: MODEL ACCURACY & OPERATOR DIAGNOSTICS HANDLERS
// -------------------------------------------------------------
async function fetchModelDiagnostics() {
  try {
    const [evalRes, benchRes, diagRes] = await Promise.all([
      authFetch(`${CONFIG.apiBase}/api/v1/model/evaluation`).catch(() => null),
      authFetch(`${CONFIG.apiBase}/api/v1/model/benchmark`).catch(() => null),
      authFetch(`${CONFIG.apiBase}/api/v1/model/diagnostics`).catch(() => null),
    ]);

    if (diagRes && diagRes.ok) {
      const diagData = await diagRes.json();
      const devEl = document.getElementById("diagDevice");
      if (devEl) devEl.textContent = `${(diagData.resolved_device || "cpu").toUpperCase()} EXECUTION`;
      const gpuEl = document.getElementById("diagGpu");
      if (gpuEl && diagData.device_diagnostics) gpuEl.textContent = diagData.device_diagnostics.device_name || "N/A";
      const noticeEl = document.getElementById("diagGpuNotice");
      if (noticeEl) noticeEl.textContent = diagData.hardware_status || "";
    }

    if (evalRes && evalRes.ok) {
      const ev = await evalRes.json();
      const latEl = document.getElementById("diagLatency");
      if (latEl && ev.operational) latEl.textContent = `${ev.operational.average_latency_ms.toFixed(1)} ms`;
      const fpsEl = document.getElementById("diagFps");
      if (fpsEl && ev.operational) fpsEl.textContent = `${ev.operational.fps.toFixed(1)} FPS`;

      const tbody = document.getElementById("diagMetricsTable");
      if (tbody && ev.metrics) {
        tbody.innerHTML = `
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 4px 2px; font-weight: 600; color: #f8fafc;">PERSON (Worker)</td>
            <td style="padding: 4px 2px; color: #34d399;">${(ev.metrics.person.precision * 100).toFixed(1)}%</td>
            <td style="padding: 4px 2px; color: #34d399;">${(ev.metrics.person.recall * 100).toFixed(1)}%</td>
            <td style="padding: 4px 2px;">${ev.metrics.person.f1.toFixed(3)}</td>
            <td style="padding: 4px 2px; font-weight: 700; color: #38bdf8;">${ev.metrics.person.map50.toFixed(3)}</td>
          </tr>
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 4px 2px; font-weight: 600; color: #f8fafc;">HELMET (Hardhat)</td>
            <td style="padding: 4px 2px; color: #34d399;">${(ev.metrics.hardhat.precision * 100).toFixed(1)}%</td>
            <td style="padding: 4px 2px; color: #34d399;">${(ev.metrics.hardhat.recall * 100).toFixed(1)}%</td>
            <td style="padding: 4px 2px;">${ev.metrics.hardhat.f1.toFixed(3)}</td>
            <td style="padding: 4px 2px; font-weight: 700; color: #38bdf8;">${ev.metrics.hardhat.map50.toFixed(3)}</td>
          </tr>
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 4px 2px; font-weight: 600; color: #f8fafc;">SAFETY VEST</td>
            <td style="padding: 4px 2px; color: #34d399;">${(ev.metrics.safety_vest.precision * 100).toFixed(1)}%</td>
            <td style="padding: 4px 2px; color: #34d399;">${(ev.metrics.safety_vest.recall * 100).toFixed(1)}%</td>
            <td style="padding: 4px 2px;">${ev.metrics.safety_vest.f1.toFixed(3)}</td>
            <td style="padding: 4px 2px; font-weight: 700; color: #38bdf8;">${ev.metrics.safety_vest.map50.toFixed(3)}</td>
          </tr>
          <tr>
            <td colspan="5" style="padding: 6px 2px; font-size: 9px; color: var(--text-muted);">
              Spatial PPE Association Accuracy: <strong style="color: #34d399;">${(ev.ppe_association.association_accuracy * 100).toFixed(1)}%</strong> (${ev.ppe_association.correct_associations} matched pairings)
            </td>
          </tr>
        `;
      }
    }

    if (benchRes && benchRes.ok) {
      const bench = await benchRes.json();
      const benchTbody = document.getElementById("diagBenchmarkTable");
      if (benchTbody && bench.candidates) {
        benchTbody.innerHTML = Object.values(bench.candidates).map((c) => `
          <tr style="border-bottom: 1px solid rgba(255,255,255,0.05);">
            <td style="padding: 3px 2px; color: #f8fafc; font-weight: 600;">${escapeHtml(c.model_name)}</td>
            <td style="padding: 3px 2px;">${c.person_recall != null ? (c.person_recall * 100).toFixed(0) + '%' : 'N/A'}</td>
            <td style="padding: 3px 2px;">${c.helmet_recall != null ? (c.helmet_recall * 100).toFixed(0) + '%' : 'N/A'}</td>
            <td style="padding: 3px 2px;">${c.vest_recall != null ? (c.vest_recall * 100).toFixed(0) + '%' : 'N/A'}</td>
            <td style="padding: 3px 2px;">${c.latency_ms != null ? c.latency_ms.toFixed(0) + ' ms' : 'N/A'}</td>
            <td style="padding: 3px 2px; font-size: 8px; color: ${c.status === 'Evaluated' ? '#34d399' : '#f59e0b'};">${escapeHtml(c.status)}</td>
          </tr>
        `).join("");
      }
    }
  } catch (err) {
    console.warn("Failed to load model diagnostics:", err);
  }
}

document.addEventListener("DOMContentLoaded", () => {
  const refreshDiagBtn = document.getElementById("refreshDiagnosticsBtn");
  if (refreshDiagBtn) {
    refreshDiagBtn.addEventListener("click", () => {
      fetchModelDiagnostics();
    });
  }
});



