/**
 * Incident and Alert Management Service
 * Manages safety alert listings, lifecycle status changes (acknowledge, resolve, dismiss),
 * incident explanation packages, and evidence URLs.
 */

import { apiClient } from "../api/client.js";
import { ENDPOINTS } from "../api/endpoints.js";

export const incidentService = {
  async listAlerts(params = {}) {
    const res = await apiClient.get(ENDPOINTS.ALERTS, params);
    if (!res.ok) {
      return { ok: false, alerts: [], total: 0, error: res.error };
    }
    const data = res.data || {};
    return {
      ok: true,
      alerts: data.items || data.alerts || [],
      total: data.total || 0,
      page: data.page || 1,
      totalPages: data.total_pages || 1,
    };
  },

  async getAlertStats() {
    const res = await apiClient.get(ENDPOINTS.ALERT_STATS);
    if (!res.ok) return { ok: false, data: null, error: res.error };
    return { ok: true, data: res.data };
  },

  async getAlert(alertId) {
    const res = await apiClient.get(ENDPOINTS.ALERT_DETAIL(alertId));
    return res;
  },

  async acknowledgeAlert(alertId, notes = "") {
    return apiClient.post(ENDPOINTS.ALERT_ACKNOWLEDGE(alertId), { notes });
  },

  async resolveAlert(alertId, notes = "") {
    return apiClient.post(ENDPOINTS.ALERT_RESOLVE(alertId), { resolution_notes: notes });
  },

  async dismissAlert(alertId, reason = "") {
    return apiClient.post(ENDPOINTS.ALERT_DISMISS(alertId), { dismiss_reason: reason });
  },

  getAlertEvidenceUrl(alertId) {
    const base = apiClient.getBaseUrl();
    const token = apiClient.getToken();
    const tokenParam = token ? `?token=${encodeURIComponent(token)}` : "";
    return `${base}${ENDPOINTS.ALERT_EVIDENCE(alertId)}${tokenParam}`;
  },

  async listIncidents(params = {}) {
    const res = await apiClient.get(ENDPOINTS.INCIDENTS, params);
    if (!res.ok) return { ok: false, incidents: [], error: res.error };
    return { ok: true, incidents: Array.isArray(res.data) ? res.data : [] };
  },

  async getIncident(incidentId) {
    return apiClient.get(ENDPOINTS.INCIDENT_DETAIL(incidentId));
  },

  async getIncidentExplanation(incidentId) {
    return apiClient.get(ENDPOINTS.INCIDENT_EXPLANATION(incidentId));
  },
};
