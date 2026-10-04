/**
 * IntelliWatch API Endpoint Definitions
 * Maps frontend services directly to existing CuriousPARC backend routes.
 */

export const ENDPOINTS = {
  // Health & System Telemetry
  HEALTH: "/api/v1/health",
  METRICS: "/api/v1/system/metrics",
  STATUS: "/api/v1/status",
  DIAGNOSTICS: "/api/v1/model/diagnostics",
  BENCHMARK: "/api/v1/model/benchmark",
  EVALUATION: "/api/v1/model/evaluation",
  CONFIG: "/api/v1/config",

  // Media Analysis
  ANALYZE_IMAGE: "/api/v1/analyze/image",
  ANALYZE_VIDEO: "/api/v1/analyze/video",
  ANALYZE_STATUS: (jobId) => `/api/v1/analyze/status/${encodeURIComponent(jobId)}`,
  ANALYZE_RESULT: (jobId) => `/api/v1/analyze/result/${encodeURIComponent(jobId)}`,
  ANALYZE_MEDIA: (mediaId) => `/api/v1/media/${encodeURIComponent(mediaId)}`,

  // Scene Graph & Situational Understanding
  SCENE_CURRENT: "/api/v1/scene/current",
  SCENE_GRAPH: "/api/v1/scene/graph",
  SCENE_JOB: (jobId) => `/api/v1/scene/current?job_id=${encodeURIComponent(jobId)}`,
  SCENE_ID: (sceneId) => `/api/v1/scene/${encodeURIComponent(sceneId)}`,

  // Cameras & Video Streams
  CAMERAS: "/api/v1/cameras",
  CAMERA_DETAIL: (id) => `/api/v1/cameras/${encodeURIComponent(id)}`,
  CAMERA_START: (id) => `/api/v1/cameras/${encodeURIComponent(id)}/start`,
  CAMERA_STOP: (id) => `/api/v1/cameras/${encodeURIComponent(id)}/stop`,
  CAMERA_SNAPSHOT: (id) => `/api/v1/cameras/${encodeURIComponent(id)}/snapshot`,
  CAMERA_FRAME: (id) => `/api/v1/cameras/${encodeURIComponent(id)}/frame`,
  CAMERA_STREAM: (id) => `/api/v1/cameras/${encodeURIComponent(id)}/stream`,
  CAMERA_CALIBRATION: (id) => `/api/v1/cameras/${encodeURIComponent(id)}/calibration`,
  CAMERA_CALIBRATION_VALIDATE: (id) => `/api/v1/cameras/${encodeURIComponent(id)}/calibration/validate`,
  CALIBRATIONS: "/api/v1/calibrations",

  // Alerts & Incident Management
  ALERTS: "/api/v1/alerts",
  ALERT_STATS: "/api/v1/alerts/statistics",
  ALERT_DETAIL: (id) => `/api/v1/alerts/${encodeURIComponent(id)}`,
  ALERT_ACKNOWLEDGE: (id) => `/api/v1/alerts/${encodeURIComponent(id)}/acknowledge`,
  ALERT_RESOLVE: (id) => `/api/v1/alerts/${encodeURIComponent(id)}/resolve`,
  ALERT_DISMISS: (id) => `/api/v1/alerts/${encodeURIComponent(id)}/dismiss`,
  ALERT_EVIDENCE: (id) => `/api/v1/alerts/${encodeURIComponent(id)}/evidence`,

  // Incident Lifecycle & Explanations
  INCIDENTS: "/api/v1/incidents",
  INCIDENT_DETAIL: (id) => `/api/v1/incidents/${encodeURIComponent(id)}`,
  INCIDENT_EXPLANATION: (id) => `/api/v1/incidents/${encodeURIComponent(id)}/explanation`,
  INCIDENT_TIMELINE: (id) => `/api/v1/incidents/${encodeURIComponent(id)}/timeline`,
  INCIDENT_PACKAGE: (id) => `/api/v1/incidents/${encodeURIComponent(id)}/package`,

  // Safety Analytics & Spatial Zones
  ANALYTICS_SUMMARY: "/api/v1/analytics/summary",
  ZONES: "/api/v1/zones",
  ZONE_DETAIL: (id) => `/api/v1/zones/${encodeURIComponent(id)}`,

  // Authentication & Session
  AUTH_LOGIN: "/api/v1/auth/login",
  AUTH_LOGOUT: "/api/v1/auth/logout",
  AUTH_ME: "/api/v1/auth/me",
};
