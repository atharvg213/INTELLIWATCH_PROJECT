/**
 * Camera and Video Stream Service
 * Interacts with CuriousPARC CameraManager to list cameras, register RTSP feeds,
 * start/stop live inference workers, and obtain stream/snapshot URLs.
 */

import { apiClient } from "../api/client.js";
import { ENDPOINTS } from "../api/endpoints.js";

export const cameraService = {
  async listCameras() {
    const res = await apiClient.get(ENDPOINTS.CAMERAS);
    if (!res.ok) {
      return { ok: false, error: res.error, cameras: [], total: 0, connected: 0 };
    }
    const rawCameras = (res.data && res.data.cameras) || [];
    const cameras = rawCameras.map((cam) => {
      const isConnected = cam.status === "CONNECTED" || cam.is_active === true;
      const isRunning = cam.is_active === true || cam.status === "CONNECTED";
      return {
        ...cam,
        is_connected: isConnected,
        is_running: isRunning,
        fps_limit: cam.max_processing_fps || cam.stream_fps || 10,
        source_type: cam.source_sanitized && cam.source_sanitized.startsWith("http") ? "HTTP" : "RTSP",
      };
    });
    return {
      ok: true,
      cameras,
      total: (res.data && res.data.total_cameras) || cameras.length,
      connected: (res.data && res.data.connected_count) || cameras.filter((c) => c.is_connected).length,
    };
  },

  async registerCamera({ camera_id, name, rtsp_url, source, source_type = "rtsp", fps_limit = 10, max_processing_fps = 10 } = {}) {
    const cid = camera_id || `cam_${Date.now().toString(36)}`;
    const src = source || rtsp_url || "";
    const fps = Number(max_processing_fps || fps_limit || 10);
    const res = await apiClient.post(ENDPOINTS.CAMERAS, {
      camera_id: cid,
      name: name || `Camera ${cid}`,
      source: src,
      sampling_interval: 1,
      max_processing_fps: fps,
      auto_start: true,
      loop_file: true,
    });
    return res;
  },

  async startCamera(cameraId) {
    return apiClient.post(ENDPOINTS.CAMERA_START(cameraId));
  },

  async stopCamera(cameraId) {
    return apiClient.post(ENDPOINTS.CAMERA_STOP(cameraId));
  },

  async removeCamera(cameraId) {
    return apiClient.delete(ENDPOINTS.CAMERA_DETAIL(cameraId));
  },

  getSnapshotUrl(cameraId) {
    const base = apiClient.getBaseUrl();
    const token = apiClient.getToken();
    const tokenParam = token ? `?token=${encodeURIComponent(token)}` : "";
    return `${base}${ENDPOINTS.CAMERA_SNAPSHOT(cameraId)}${tokenParam}`;
  },

  getStreamUrl(cameraId) {
    const base = apiClient.getBaseUrl();
    const token = apiClient.getToken();
    const tokenParam = token ? `?token=${encodeURIComponent(token)}` : "";
    return `${base}${ENDPOINTS.CAMERA_STREAM(cameraId)}${tokenParam}`;
  },

  getFrameUrl(cameraId) {
    const base = apiClient.getBaseUrl();
    const token = apiClient.getToken();
    const tokenParam = token ? `?token=${encodeURIComponent(token)}` : "";
    return `${base}${ENDPOINTS.CAMERA_FRAME(cameraId)}${tokenParam}`;
  },

  async getCalibration(cameraId) {
    return apiClient.get(ENDPOINTS.CAMERA_CALIBRATION(cameraId));
  },

  async saveCalibration(cameraId, data) {
    return apiClient.put(ENDPOINTS.CAMERA_CALIBRATION(cameraId), data);
  },

  async deleteCalibration(cameraId) {
    return apiClient.delete(ENDPOINTS.CAMERA_CALIBRATION(cameraId));
  },

  async validateCalibration(cameraId, data) {
    return apiClient.post(ENDPOINTS.CAMERA_CALIBRATION_VALIDATE(cameraId), data);
  },

  async listCalibrations() {
    return apiClient.get(ENDPOINTS.CALIBRATIONS);
  },
};
