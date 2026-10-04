/**
 * Media Analysis Service
 * Coordinates image and video upload, asynchronous job submission,
 * polling progress telemetry, and resolving annotated evidence URLs.
 */

import { apiClient } from "../api/client.js";
import { ENDPOINTS } from "../api/endpoints.js";

function resolveMediaUrl(url, apiBase) {
  if (!url) return null;
  if (url.startsWith("http://") || url.startsWith("https://")) return url;
  return `${apiBase}${url.startsWith("/") ? "" : "/"}${url}`;
}

export const mediaService = {
  /**
   * Uploads and runs synchronous single-frame intelligence pipeline on an image.
   * Returns complete detection, tracking, PPE, risk, and annotated media data.
   */
  async analyzeImage(file, signal = null, onUploadProgress = null) {
    const res = await apiClient.upload(
      ENDPOINTS.ANALYZE_IMAGE,
      file,
      "file",
      {},
      signal ? { signal } : {},
      onUploadProgress
    );
    if (!res.ok) {
      return {
        ok: false,
        error: res.error || "Failed to analyze image",
        data: null,
      };
    }

    const data = res.data;
    const apiBase = apiClient.getBaseUrl();
    const annotatedUrl = resolveMediaUrl(data.annotated_media_url, apiBase);
    const originalUrl = resolveMediaUrl(data.original_media_url, apiBase);

    return {
      ok: true,
      data: {
        ...data,
        annotated_media_url: annotatedUrl,
        original_media_url: originalUrl,
        detections_count: data.detections_count ?? (data.assessment?.detections?.length || 0),
        workers_count: data.workers_count ?? (data.assessment?.worker_inventories?.length || 0),
        non_compliant_count: data.non_compliant_count ?? (data.assessment?.non_compliant_workers_count || 0),
        highest_risk_level: data.highest_risk_level || "INFO",
        highest_risk_score: data.highest_risk_score || 0,
      },
      error: null,
    };
  },

  /**
   * Submits a video file for asynchronous batch intelligence analysis.
   */
  async submitVideoJob(file, signal = null, onUploadProgress = null) {
    const res = await apiClient.upload(
      ENDPOINTS.ANALYZE_VIDEO,
      file,
      "file",
      {},
      signal ? { signal } : {},
      onUploadProgress
    );
    if (!res.ok) {
      return {
        ok: false,
        error: res.error || "Failed to queue video analysis",
        jobId: null,
      };
    }
    return {
      ok: true,
      jobId: res.data.job_id,
      status: res.data.status,
      filename: res.data.filename,
      message: res.data.message,
    };
  },

  /**
   * Fetches real-time status of an ongoing video analysis job.
   */
  async getVideoJobStatus(jobId, signal = null) {
    const res = await apiClient.get(
      ENDPOINTS.ANALYZE_STATUS(jobId),
      {},
      signal ? { signal } : {}
    );
    if (!res.ok) {
      return {
        ok: false,
        error: res.error || "Failed to retrieve job status",
        data: null,
      };
    }

    const data = res.data;
    const apiBase = apiClient.getBaseUrl();
    const annotatedUrl = resolveMediaUrl(data.annotated_media_url, apiBase);
    const frameUrl = resolveMediaUrl(data.latest_frame_url, apiBase);

    return {
      ok: true,
      data: {
        ...data,
        annotated_media_url: annotatedUrl,
        latest_frame_url: frameUrl,
      },
    };
  },

  /**
   * Fetches final comprehensive result metrics upon video analysis completion.
   */
  async getVideoResult(jobId, signal = null) {
    const res = await apiClient.get(
      ENDPOINTS.ANALYZE_RESULT(jobId),
      {},
      signal ? { signal } : {}
    );
    if (!res.ok) {
      return {
        ok: false,
        error: res.error || "Failed to retrieve final video result",
        data: null,
      };
    }

    const data = res.data;
    const apiBase = apiClient.getBaseUrl();
    const videoUrl = resolveMediaUrl(data.annotated_video_url || data.annotated_media_url, apiBase);
    const frameUrl = resolveMediaUrl(data.annotated_frame_url || data.latest_frame_url, apiBase);

    return {
      ok: true,
      data: {
        ...data,
        annotated_video_url: videoUrl,
        annotated_media_url: videoUrl,
        annotated_frame_url: frameUrl,
        highest_risk_level: data.highest_risk_tier || data.highest_risk_level || "INFO",
        highest_risk_score: data.highest_risk_score || 0,
        total_incidents: data.total_incidents ?? (data.new_incident_ids ? data.new_incident_ids.length : 0),
      },
    };
  },

  /**
   * Polls a video job until completion, failure, or cancellation.
   * On completion, fetches final authoritative result payload.
   */
  async pollVideoJob(jobId, onProgress = null, intervalMs = 1000, maxAttempts = 600, signal = null) {
    let attempts = 0;
    while (attempts < maxAttempts) {
      if (signal && signal.aborted) {
        throw new Error("Video analysis polling cancelled");
      }
      attempts++;

      const res = await this.getVideoJobStatus(jobId, signal);
      if (!res.ok) {
        if (res.aborted) {
          throw new Error("Video analysis polling cancelled");
        }
        throw new Error(res.error || "Error checking video job status");
      }

      const statusData = res.data;
      if (onProgress && typeof onProgress === "function") {
        onProgress(statusData);
      }

      if (statusData.status === "COMPLETED") {
        // Fetch full final result payload from /api/v1/analyze/result/{jobId}
        try {
          const resultRes = await this.getVideoResult(jobId, signal);
          if (resultRes.ok && resultRes.data) {
            return {
              ...statusData,
              ...resultRes.data,
              status: "COMPLETED",
              progress_pct: 100.0,
            };
          }
        } catch (_) {}
        return {
          ...statusData,
          status: "COMPLETED",
          progress_pct: 100.0,
        };
      }

      if (statusData.status === "FAILED") {
        throw new Error(statusData.error_message || "Video analysis job failed");
      }

      await new Promise((resolve) => setTimeout(resolve, intervalMs));
    }
    throw new Error("Video analysis timed out while processing");
  },
};
