/**
 * IntelliWatch API Client
 * Centralized HTTP request handler with authentication, base URL management,
 * and structured error handling.
 */

import { ENDPOINTS } from "./endpoints.js";

const TOKEN_KEY = "intelliwatch_access_token";
const USER_KEY = "intelliwatch_user";

class ApiClient {
  constructor() {
    this._apiBase = this._detectApiBase();
    this._token = typeof localStorage !== "undefined" ? (localStorage.getItem(TOKEN_KEY) || null) : null;
    this._user = null;
    if (typeof localStorage !== "undefined") {
      try {
        this._user = JSON.parse(localStorage.getItem(USER_KEY) || "null");
      } catch (_) {}
    }
    this._authPromise = null;
  }

  _detectApiBase() {
    // If explicitly set in localStorage, use that
    if (typeof localStorage !== "undefined") {
      const custom = localStorage.getItem("intelliwatch_api_base");
      if (custom) return custom.replace(/\/+$/, "");
    }

    // If global override exists
    if (typeof window !== "undefined" && window.__INTELLIWATCH_API_BASE__) {
      return window.__INTELLIWATCH_API_BASE__.replace(/\/+$/, "");
    }

    // Default to origin
    if (typeof window !== "undefined" && window.location && window.location.origin) {
      // If served on standard port or dev server, use current origin
      return window.location.origin;
    }
    return "http://127.0.0.1:8000";
  }

  getBaseUrl() {
    return this._apiBase;
  }

  setBaseUrl(url) {
    this._apiBase = (url || "").replace(/\/+$/, "");
    if (typeof localStorage !== "undefined") {
      if (url) {
        localStorage.setItem("intelliwatch_api_base", this._apiBase);
      } else {
        localStorage.removeItem("intelliwatch_api_base");
      }
    }
  }

  getToken() {
    return this._token;
  }

  setToken(token, user = null) {
    this._token = token;
    if (typeof localStorage !== "undefined") {
      if (token) {
        localStorage.setItem(TOKEN_KEY, token);
      } else {
        localStorage.removeItem(TOKEN_KEY);
      }
      if (user) {
        this._user = user;
        localStorage.setItem(USER_KEY, JSON.stringify(user));
      } else if (!token) {
        this._user = null;
        localStorage.removeItem(USER_KEY);
      }
    } else {
      this._user = user;
    }
  }

  getUser() {
    return this._user;
  }

  /**
   * Transparently authenticates using default operator credentials if unauthenticated.
   * Enables seamless access to authenticated camera and alert APIs.
   */
  async ensureAuthenticated() {
    if (this._token) return this._token;
    if (this._authPromise) return this._authPromise;

    this._authPromise = (async () => {
      try {
        const res = await fetch(`${this._apiBase}${ENDPOINTS.AUTH_LOGIN}`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            username: "admin",
            password: "IntelliWatch2026!",
          }),
        });
        if (res.ok) {
          const data = await res.json();
          this.setToken(data.access_token, data.user);
          return data.access_token;
        }
      } catch (err) {
        console.warn("Auto-authentication attempt failed:", err);
      } finally {
        this._authPromise = null;
      }
      return null;
    })();

    return this._authPromise;
  }

  async request(path, options = {}) {
    const url = path.startsWith("http://") || path.startsWith("https://")
      ? path
      : `${this._apiBase}${path.startsWith("/") ? path : `/${path}`}`;

    const headers = new Headers(options.headers || {});

    // Add authorization header if token exists
    if (this._token && !headers.has("Authorization")) {
      headers.set("Authorization", `Bearer ${this._token}`);
    }

    const fetchOptions = {
      ...options,
      headers,
    };

    try {
      let res = await fetch(url, fetchOptions);

      // If 401 Unauthorized, attempt auto-login once and retry
      if (res.status === 401) {
        const newToken = await this.ensureAuthenticated();
        if (newToken) {
          headers.set("Authorization", `Bearer ${newToken}`);
          res = await fetch(url, { ...fetchOptions, headers });
        }
      }

      const contentType = res.headers.get("content-type") || "";
      let data = null;
      if (contentType.includes("application/json")) {
        data = await res.json();
      } else if (contentType.includes("text/")) {
        data = await res.text();
      } else {
        data = await res.blob();
      }

      if (!res.ok) {
        const errorDetail = (data && data.detail) || res.statusText || `HTTP ${res.status}`;
        return {
          ok: false,
          status: res.status,
          error: errorDetail,
          data: null,
        };
      }

      return {
        ok: true,
        status: res.status,
        data,
        error: null,
      };
    } catch (err) {
      if (err.name === "AbortError") {
        return {
          ok: false,
          status: 0,
          error: "Request cancelled",
          aborted: true,
          data: null,
        };
      }
      console.error(`API request failed [${url}]:`, err);
      return {
        ok: false,
        status: 0,
        error: err.message || "Network request failed. Backend may be offline.",
        data: null,
      };
    }
  }

  async get(path, params = {}, options = {}) {
    const query = new URLSearchParams();
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null) query.append(k, String(v));
    }
    const qStr = query.toString();
    const fullPath = qStr ? `${path}${path.includes("?") ? "&" : "?"}${qStr}` : path;
    return this.request(fullPath, { method: "GET", ...options });
  }

  async post(path, body = null, options = {}) {
    const headers = options.headers || {};
    let reqBody = null;
    if (body !== null) {
      headers["Content-Type"] = "application/json";
      reqBody = JSON.stringify(body);
    }
    return this.request(path, { method: "POST", ...options, headers, body: reqBody });
  }

  async patch(path, body = null, options = {}) {
    const headers = options.headers || {};
    let reqBody = null;
    if (body !== null) {
      headers["Content-Type"] = "application/json";
      reqBody = JSON.stringify(body);
    }
    return this.request(path, { method: "PATCH", ...options, headers, body: reqBody });
  }

  async delete(path, options = {}) {
    return this.request(path, { method: "DELETE", ...options });
  }

  async upload(path, file, fieldName = "file", extraFields = {}, options = {}, onProgress = null) {
    const url = path.startsWith("http://") || path.startsWith("https://")
      ? path
      : `${this._apiBase}${path.startsWith("/") ? path : `/${path}`}`;

    const formData = new FormData();
    const fileName = file.name || (file.type && file.type.includes("video") ? "upload.mp4" : "upload.jpg");
    formData.append(fieldName, file, fileName);
    for (const [k, v] of Object.entries(extraFields)) {
      formData.append(k, String(v));
    }

    if (onProgress && typeof onProgress === "function" && typeof XMLHttpRequest !== "undefined") {
      return new Promise((resolve) => {
        const xhr = new XMLHttpRequest();
        xhr.open("POST", url, true);

        if (this._token) {
          xhr.setRequestHeader("Authorization", `Bearer ${this._token}`);
        }

        if (options.signal) {
          options.signal.addEventListener("abort", () => {
            xhr.abort();
            resolve({ ok: false, status: 0, error: "Upload cancelled", data: null, aborted: true });
          });
        }

        xhr.upload.onprogress = (e) => {
          if (e.lengthComputable && e.total > 0) {
            const pct = Math.round((e.loaded / e.total) * 100);
            onProgress({ loaded: e.loaded, total: e.total, percent: pct });
          }
        };

        xhr.onload = () => {
          let data = null;
          const contentType = xhr.getResponseHeader("content-type") || "";
          try {
            if (contentType.includes("application/json")) {
              data = JSON.parse(xhr.responseText);
            } else {
              data = xhr.responseText;
            }
          } catch (_) {}

          if (xhr.status >= 200 && xhr.status < 300) {
            resolve({ ok: true, status: xhr.status, data, error: null });
          } else {
            const errorDetail = (data && data.detail) || xhr.statusText || `HTTP ${xhr.status}`;
            resolve({ ok: false, status: xhr.status, error: errorDetail, data: null });
          }
        };

        xhr.onerror = () => {
          resolve({ ok: false, status: 0, error: "Network error during upload. Backend may be offline.", data: null });
        };

        xhr.ontimeout = () => {
          resolve({ ok: false, status: 0, error: "Upload timed out.", data: null });
        };

        xhr.send(formData);
      });
    }

    return this.request(path, {
      method: "POST",
      ...options,
      body: formData,
    });
  }
}

export const apiClient = new ApiClient();
