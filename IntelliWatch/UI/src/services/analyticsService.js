/**
 * Safety Analytics and Spatial Zones Service
 * Queries aggregate safety telemetry, event distributions,
 * and spatial zone configurations.
 */

import { apiClient } from "../api/client.js";
import { ENDPOINTS } from "../api/endpoints.js";

export const analyticsService = {
  async getSummary() {
    const res = await apiClient.get(ENDPOINTS.ANALYTICS_SUMMARY);
    if (!res.ok) return { ok: false, data: null, error: res.error };
    return { ok: true, data: res.data };
  },

  async listZones(enabledOnly = false) {
    const res = await apiClient.get(ENDPOINTS.ZONES, { enabled_only: enabledOnly });
    if (!res.ok) return { ok: false, zones: [], error: res.error };
    return { ok: true, zones: Array.isArray(res.data) ? res.data : [] };
  },
};
