"""API contract coverage used by the camera calibration workbench."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api import routes
from backend.main import app
from backend.schemas.camera import CameraListResponse, CameraStatus, CameraTelemetrySchema
from backend.services.calibration_service import CalibrationService


@pytest.fixture
def client():
    return TestClient(app)


def _jpeg_bytes():
    ok, encoded = cv2.imencode(".jpg", np.zeros((24, 40, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


def test_workbench_reads_registered_camera_and_current_frame(client, monkeypatch):
    camera = CameraTelemetrySchema(
        camera_id="cam_workbench",
        name="North loading bay",
        source_sanitized="rtsp://camera/stream",
        status=CameraStatus.CONNECTED,
        is_active=True,
    )
    jpeg = _jpeg_bytes()
    camera_manager = SimpleNamespace(
        list_cameras=lambda: CameraListResponse(
            total_cameras=1,
            connected_count=1,
            cameras=[camera],
        ),
        get_camera=lambda camera_id: SimpleNamespace(get_latest_jpeg=lambda: jpeg)
        if camera_id == camera.camera_id
        else None,
    )
    monkeypatch.setattr(routes, "get_camera_manager", lambda: camera_manager)

    camera_response = client.get("/api/v1/cameras")
    frame_response = client.get(f"/api/v1/cameras/{camera.camera_id}/snapshot")

    assert camera_response.status_code == 200
    assert camera_response.json()["cameras"][0]["camera_id"] == "cam_workbench"
    assert camera_response.json()["cameras"][0]["status"] == "CONNECTED"
    assert frame_response.status_code == 200
    assert frame_response.headers["content-type"].startswith("image/jpeg")
    assert frame_response.content == jpeg


def test_workbench_calibration_validate_save_get_and_delete_contract(client, monkeypatch, tmp_path):
    camera_id = "cam_workbench_contract"
    calibration_service = CalibrationService(tmp_path / "calibrations.json")
    monkeypatch.setattr(routes, "get_calibration_service", lambda: calibration_service)
    points = [
        {"x": 100, "y": 80},
        {"x": 420, "y": 80},
        {"x": 420, "y": 300},
        {"x": 100, "y": 300},
    ]

    initial = client.get(f"/api/v1/cameras/{camera_id}/calibration")
    invalid = client.post(
        f"/api/v1/cameras/{camera_id}/calibration/validate",
        json={
            "points": [
                {"x": 10, "y": 10},
                {"x": 20, "y": 20},
                {"x": 30, "y": 30},
                {"x": 40, "y": 40},
            ],
            "real_world_width_m": 8.0,
            "real_world_depth_m": 12.0,
        },
    )
    valid = client.post(
        f"/api/v1/cameras/{camera_id}/calibration/validate",
        json={
            "points": points,
            "real_world_width_m": 8.0,
            "real_world_depth_m": 12.0,
        },
    )
    saved = client.put(
        f"/api/v1/cameras/{camera_id}/calibration",
        json={
            "camera_id": camera_id,
            "calibration_enabled": True,
            "points": points,
            "real_world_width_m": 8.0,
            "real_world_depth_m": 12.0,
        },
    )
    loaded = client.get(f"/api/v1/cameras/{camera_id}/calibration")
    deleted = client.delete(f"/api/v1/cameras/{camera_id}/calibration")
    cleared = client.get(f"/api/v1/cameras/{camera_id}/calibration")

    assert initial.status_code == 200
    assert initial.json()["calibration_status"] == "UNCONFIGURED"
    assert invalid.status_code == 200
    assert invalid.json()["is_valid"] is False
    assert invalid.json()["status"] in {"INVALID", "DEGENERATE"}
    assert invalid.json()["error_message"]
    assert valid.status_code == 200
    assert valid.json()["is_valid"] is True
    assert valid.json()["status"] == "CALIBRATED"
    assert len(valid.json()["sample_projections"]) == 4
    assert saved.status_code == 200
    assert saved.json()["calibration_status"] == "CALIBRATED"
    assert saved.json()["calibration_enabled"] is True
    assert saved.json()["points"] == points
    assert saved.json()["real_world_width_m"] == 8.0
    assert saved.json()["real_world_depth_m"] == 12.0
    assert loaded.status_code == 200
    assert loaded.json()["camera_id"] == camera_id
    assert loaded.json()["calibration_status"] == "CALIBRATED"
    assert deleted.status_code == 200
    assert deleted.json()["deleted"] is True
    assert cleared.status_code == 200
    assert cleared.json()["calibration_status"] == "UNCONFIGURED"


def test_workbench_payload_with_unknown_camera_keeps_stage6a_rejection(client, monkeypatch):
    monkeypatch.setattr(
        routes,
        "get_camera_manager",
        lambda: SimpleNamespace(
            list_cameras=lambda: CameraListResponse(total_cameras=0, connected_count=0, cameras=[]),
            get_camera=lambda _: None,
        ),
    )
    monkeypatch.setattr(routes, "get_calibration_service", lambda: SimpleNamespace(get_calibration=lambda _: None))

    response = client.post(
        "/api/v1/analyze/image",
        data={"camera_id": "camera_not_registered"},
        files={"file": ("sample.png", _jpeg_bytes(), "image/jpeg")},
    )

    assert response.status_code == 400
    assert "Unknown camera_id" in response.json()["detail"]
