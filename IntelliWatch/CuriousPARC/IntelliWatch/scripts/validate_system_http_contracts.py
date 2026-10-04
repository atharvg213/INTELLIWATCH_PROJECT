"""
Comprehensive HTTP / DOM / API Validation Script
Verifies:
1. SPA routes and CSS/JS asset delivery
2. CSS calibration styling tokens
3. JS app bundle integrity
4. Live Camera calibration API (Validate, Save, Get, Delete)
5. Live Camera frame delivery (CCTV snapshot fallback)
6. Scene graph calibration integration over live HTTP
"""

import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8000"

TOKEN = None

def _login():
    global TOKEN
    body = json.dumps({"username": "admin", "password": "IntelliWatch2026!"}).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}/api/v1/auth/login", data=body, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        TOKEN = json.loads(resp.read().decode("utf-8"))["access_token"]

def _hdrs(extra=None):
    h = dict(extra or {})
    if TOKEN:
        h["Authorization"] = f"Bearer {TOKEN}"
    return h

def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}", headers=_hdrs())
    with urllib.request.urlopen(req) as resp:
        return resp.status, resp.headers, resp.read()

def post_json(path, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}{path}", data=body, headers=_hdrs({"Content-Type": "application/json"}), method="POST")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def put_json(path, data):
    body = json.dumps(data).encode("utf-8")
    req = urllib.request.Request(f"{BASE_URL}{path}", data=body, headers=_hdrs({"Content-Type": "application/json"}), method="PUT")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def delete_req(path):
    req = urllib.request.Request(f"{BASE_URL}{path}", headers=_hdrs(), method="DELETE")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read().decode("utf-8"))

def run_checks():
    print("=== LIVE SYSTEM HTTP / DOM / API VALIDATION ===")
    _login()
    print("[PASS] 0. Authenticated as admin (same flow as UI auto-login).")

    # 1. HTML index
    status, headers, body = get("/app/settings")
    assert status == 200
    html = body.decode("utf-8")
    assert 'id="app"' in html
    assert 'src="/app.js"' in html
    assert 'href="/styles.css"' in html
    print("[PASS] 1. SPA entrypoint /app/settings served correctly.")

    # 2. CSS Bundle
    status, headers, body = get("/styles.css")
    assert status == 200
    css = body.decode("utf-8")
    assert ".calibration-panel" in css
    assert ".calibration-workbench" in css
    assert ".cal-polygon" in css
    assert ".evidence-basis-grid" in css
    assert ".evidence-basis-card" in css
    print("[PASS] 2. Stylesheet styles.css contains all Camera Calibration and Dual-Evidence CSS tokens.")

    # 3. JS App Bundle
    status, headers, body = get("/app.js")
    assert status == 200
    js = body.decode("utf-8")
    assert "cameraCalibrationSection" in js
    assert "refreshCalibration" in js
    assert "validate-cal" in js
    assert "save-cal" in js
    assert "clear-cal" in js
    assert "reset-cal-points" in js
    assert "GROUND PLANE" in js
    assert "IMAGE SPACE" in js
    print("[PASS] 3. Application bundle app.js contains all Calibration UI handlers, state management, and inspectors.")

    # 4. Camera frame delivery
    status, headers, body = get("/api/v1/cameras/cam_loading_bay_01/frame")
    assert status == 200
    ct = headers.get("Content-Type", "")
    assert "image/jpeg" in ct
    assert len(body) > 10000
    print(f"[PASS] 4. Camera frame endpoint served {len(body)} bytes JPEG image for cam_loading_bay_01.")

    # 5. Validation API (Valid)
    valid_payload = {
        "camera_id": "cam_loading_bay_01",
        "points": [
            {"x": 100.0, "y": 150.0},
            {"x": 600.0, "y": 150.0},
            {"x": 750.0, "y": 550.0},
            {"x": 50.0, "y": 550.0}
        ],
        "real_world_width_m": 8.0,
        "real_world_depth_m": 12.0,
        "coordinate_system": "METRIC_GROUND_PLANE"
    }
    status, resp = post_json("/api/v1/cameras/cam_loading_bay_01/calibration/validate", valid_payload)
    assert status == 200
    assert resp["is_valid"] is True
    assert resp["reprojection_error_px"] is not None
    print(f"[PASS] 5. Live calibration validation: is_valid=True, reprojection_error_px={resp['reprojection_error_px']}, coverage_m2={resp.get('estimated_coverage_area_m2')}")

    # 6. Validation API (Degenerate Collinear)
    collinear_payload = {
        "camera_id": "cam_loading_bay_01",
        "points": [
            {"x": 100.0, "y": 100.0},
            {"x": 200.0, "y": 200.0},
            {"x": 300.0, "y": 300.0},
            {"x": 400.0, "y": 500.0}
        ],
        "real_world_width_m": 8.0,
        "real_world_depth_m": 12.0
    }
    status, resp = post_json("/api/v1/cameras/cam_loading_bay_01/calibration/validate", collinear_payload)
    assert status == 200
    assert resp["is_valid"] is False
    assert "collinear" in resp["error_message"].lower() or "degenerate" in resp["error_message"].lower()
    print(f"[PASS] 6. Live validation caught collinear degenerate points: '{resp['error_message']}'")

    # 7. Save Calibration (PUT)
    save_payload = {
        "camera_id": "cam_loading_bay_01",
        "calibration_enabled": True,
        "points": valid_payload["points"],
        "real_world_width_m": 8.0,
        "real_world_depth_m": 12.0,
        "coordinate_system": "METRIC_GROUND_PLANE"
    }
    status, resp = put_json("/api/v1/cameras/cam_loading_bay_01/calibration", save_payload)
    assert status == 200
    assert resp["calibration_status"] == "CALIBRATED"
    print("[PASS] 7. PUT /api/v1/cameras/cam_loading_bay_01/calibration saved successfully.")

    # 8. Retrieve Calibration (GET)
    status, headers, body = get("/api/v1/cameras/cam_loading_bay_01/calibration")
    assert status == 200
    retrieved = json.loads(body.decode("utf-8"))
    assert retrieved["calibration_status"] == "CALIBRATED"
    assert len(retrieved["points"]) == 4
    assert retrieved["real_world_width_m"] == 8.0
    print("[PASS] 8. GET /api/v1/cameras/cam_loading_bay_01/calibration returned saved configuration.")

    # 9. Query Scene Graph with calibrated camera
    status, headers, body = get("/api/v1/scene/graph?camera_id=cam_loading_bay_01")
    assert status == 200
    sg = json.loads(body.decode("utf-8"))
    assert sg["calibration_status"] == "CALIBRATED"
    assert sg["spatial_basis"] == "GROUND_PLANE_APPROXIMATION"
    print("[PASS] 9. GET /api/v1/scene/graph?camera_id=cam_loading_bay_01 enriched with GROUND_PLANE_APPROXIMATION.")

    # 10. Query Scene Graph with uncalibrated camera (fallback)
    status, headers, body = get("/api/v1/scene/graph?camera_id=cam_unknown_999")
    assert status == 200
    sg_uncal = json.loads(body.decode("utf-8"))
    assert sg_uncal["calibration_status"] == "UNCONFIGURED"
    assert sg_uncal["spatial_basis"] == "IMAGE_SPACE"
    print("[PASS] 10. GET /api/v1/scene/graph?camera_id=cam_unknown_999 verified pure IMAGE_SPACE fallback.")

    # 11. Delete Calibration (DELETE)
    status, resp = delete_req("/api/v1/cameras/cam_loading_bay_01/calibration")
    assert status == 200
    assert resp["deleted"] is True
    st2, _, b2 = get("/api/v1/cameras/cam_loading_bay_01/calibration")
    assert json.loads(b2.decode("utf-8"))["calibration_status"] == "UNCONFIGURED"
    print("[PASS] 11. DELETE /api/v1/cameras/cam_loading_bay_01/calibration successfully cleared calibration.")

    print("\n>>> ALL 11 LIVE HTTP / DOM / API CONTRACT CHECKS PASSED SUCCESSFULLY. <<<")

if __name__ == "__main__":
    run_checks()
