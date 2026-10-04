/**
 * Telemetry and Diagnostics Service
 * Queries runtime health, CUDA GPU hardware acceleration status,
 * inference FPS, frame latency, and active model versions.
 */

import { apiClient } from "../api/client.js";
import { ENDPOINTS } from "../api/endpoints.js";

export const telemetryService = {
  /**
   * Fetches comprehensive system telemetry combining metrics, diagnostics, and status.
   */
  async getSystemTelemetry() {
    const [healthRes, metricsRes, diagRes, statusRes] = await Promise.all([
      apiClient.get(ENDPOINTS.HEALTH),
      apiClient.get(ENDPOINTS.METRICS),
      apiClient.get(ENDPOINTS.DIAGNOSTICS),
      apiClient.get(ENDPOINTS.STATUS),
    ]);

    const isOnline = healthRes.ok && healthRes.data?.status === "ok";

    if (!isOnline) {
      return {
        online: false,
        error: healthRes.error || "Backend service unreachable",
        device: "Unknown",
        isCuda: false,
        gpuName: null,
        fps: null,
        latencyMs: null,
        complianceRate: null,
        totalCameras: 0,
        onlineCameras: 0,
        activeIncidents: 0,
        eventsToday: 0,
        modelsActive: {},
        hardwareStatus: "Offline",
      };
    }

    const metrics = metricsRes.ok ? metricsRes.data : {};
    const diagnostics = diagRes.ok ? diagRes.data : {};
    const status = statusRes.ok ? statusRes.data : {};

    const deviceDiag = diagnostics.device_diagnostics || {};
    const isCuda = (deviceDiag.cuda_available === true && deviceDiag.optimal_device === "cuda") ||
                   (status.device === "cuda" && status.is_cpu_mode === false) ||
                   (metrics.execution_device === "cuda");

    const gpuName = deviceDiag.device_name || (isCuda ? "NVIDIA CUDA GPU" : null);

    return {
      online: true,
      error: null,
      device: isCuda ? "CUDA (GPU)" : "CPU",
      isCuda,
      gpuName,
      fps: metrics.fps !== undefined ? metrics.fps : status.processing_fps,
      latencyMs: metrics.avg_latency_ms !== undefined ? metrics.avg_latency_ms : status.avg_latency_ms,
      complianceRate: metrics.compliance_rate !== undefined ? metrics.compliance_rate : 100.0,
      totalCameras: metrics.cameras_total || 0,
      onlineCameras: metrics.cameras_online || 0,
      activeIncidents: metrics.active_incidents || 0,
      eventsToday: metrics.events_today || 0,
      modelsActive: diagnostics.models_active || {},
      hardwareStatus: diagnostics.hardware_status || (isCuda ? `Active on ${gpuName}` : "CPU mode"),
      pipelineState: status.pipeline_state || "idle",
      torchVersion: deviceDiag.torch_version,
      torchCudaVersion: deviceDiag.torch_cuda_version,
    };
  },

  async getHealth() {
    return apiClient.get(ENDPOINTS.HEALTH);
  },

  async getDiagnostics() {
    return apiClient.get(ENDPOINTS.DIAGNOSTICS);
  },
};
