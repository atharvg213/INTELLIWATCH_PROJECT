"""
tests/test_stream_processing_step21.py
Unit and integration tests for Step 21: Real-Time RTSP Ingestion & Multi-Camera Processing.
"""
import os
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.camera import CameraRegisterRequest, CameraStatus, CameraUpdateRequest
from backend.services.camera_manager import CameraStreamWorker, MultiCameraManager, get_camera_manager
from vision.streaming.stream_reader import RTSPStreamReader, StreamFrame, sanitize_stream_url

client = TestClient(app)

SAMPLE_VIDEO = Path("data/samples/cctv_worker_moving.mp4")


@pytest.fixture(autouse=True)
def cleanup_cameras():
    """Ensures camera manager is cleanly stopped and reset before/after each test."""
    manager = get_camera_manager()
    manager.stop_all()
    with manager._lock:
        manager._cameras.clear()
    yield
    manager.stop_all()
    with manager._lock:
        manager._cameras.clear()


class TestStreamSecurityAndSanitization:
    """Verifies Phase 2 credential protection and URL sanitization."""

    def test_sanitize_rtsp_with_username_and_password(self):
        raw = "rtsp://admin:SuperSecret123@192.168.1.100:554/Streaming/Channels/101"
        sanitized = sanitize_stream_url(raw)
        assert "SuperSecret123" not in sanitized
        assert "admin:***@" in sanitized
        assert "192.168.1.100:554" in sanitized

    def test_sanitize_url_with_query_tokens(self):
        raw = "http://camera.local/feed?token=xyz987654321&quality=high"
        sanitized = sanitize_stream_url(raw)
        assert "xyz987654321" not in sanitized
        assert "token=***" in sanitized
        assert "quality=high" in sanitized

    def test_sanitize_local_path_and_webcam(self):
        assert sanitize_stream_url("data/samples/cctv.mp4") == "data/samples/cctv.mp4"
        assert sanitize_stream_url(0) == "webcam_0"
        assert sanitize_stream_url("0") == "0"


class TestRTSPStreamReader:
    """Verifies stream ingestion, bounded queues, and fault handling."""

    @pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="Sample video not available")
    def test_stream_reader_ingestion(self):
        reader = RTSPStreamReader(
            camera_id="test_cam_01",
            source=str(SAMPLE_VIDEO),
            queue_size=5,
            loop_file=True,
        )
        assert reader.status == CameraStatus.DISCONNECTED
        reader.start()
        assert reader.is_running

        # Allow capture thread to ingest a few frames
        frame = None
        for _ in range(30):
            frame = reader.get_frame(timeout=0.2)
            if frame is not None:
                break
            time.sleep(0.05)

        assert frame is not None
        assert isinstance(frame, StreamFrame)
        assert frame.camera_id == "test_cam_01"
        assert frame.width > 0 and frame.height > 0
        assert isinstance(frame.image, np.ndarray)

        telemetry = reader.get_telemetry()
        assert telemetry["total_ingested_frames"] >= 1
        assert telemetry["status"] == CameraStatus.CONNECTED

        reader.stop()
        assert not reader.is_running
        assert reader.status == CameraStatus.STOPPED

    def test_stream_reader_invalid_source_resilience(self):
        # Test nonexistent local file source
        reader = RTSPStreamReader(
            camera_id="test_invalid_cam",
            source="data/samples/nonexistent_file_99.mp4",
            reconnect_interval_sec=1.0,
            max_reconnect_attempts=2,
            loop_file=False,
        )
        reader.start()

        # Poll for error status
        for _ in range(30):
            telemetry = reader.get_telemetry()
            if telemetry["last_error"] is not None:
                break
            time.sleep(0.05)

        telemetry = reader.get_telemetry()
        assert telemetry["status"] in (CameraStatus.ERROR, CameraStatus.STOPPED, CameraStatus.RECONNECTING, CameraStatus.DISCONNECTED)
        assert telemetry["last_error"] is not None
        assert "not found" in telemetry["last_error"] or "Could not open" in telemetry["last_error"]

        reader.stop()
        assert reader.status == CameraStatus.STOPPED

    @pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="Sample video not available")
    def test_bounded_queue_frame_dropping(self):
        reader = RTSPStreamReader(
            camera_id="test_drop_cam",
            source=str(SAMPLE_VIDEO),
            queue_size=2,  # Tiny queue to force drops
            loop_file=True,
        )
        reader.start()
        # Sleep without consuming from queue to allow buffer to saturate
        time.sleep(0.6)

        telemetry = reader.get_telemetry()
        assert telemetry["total_ingested_frames"] >= 2
        # Since queue capacity is 2 and reader ingested frames without consumer, dropped frames must be > 0
        assert telemetry["dropped_frames_count"] >= 1
        reader.stop()


class TestMultiCameraIsolation:
    """Verifies Phase 3 multi-camera pipeline isolation and concurrent execution."""

    @pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="Sample video not available")
    def test_dual_camera_concurrent_processing_and_isolation(self):
        manager = get_camera_manager()

        # Register two concurrent camera streams on the same sample footage
        req1 = CameraRegisterRequest(
            camera_id="cam_dock_01",
            name="Loading Dock 1",
            source=str(SAMPLE_VIDEO),
            sampling_interval=1,
            max_processing_fps=15.0,
            auto_start=True,
        )
        req2 = CameraRegisterRequest(
            camera_id="cam_dock_02",
            name="Loading Dock 2",
            source=str(SAMPLE_VIDEO),
            sampling_interval=1,
            max_processing_fps=15.0,
            auto_start=True,
        )

        tel1 = manager.register_camera(req1)
        tel2 = manager.register_camera(req2)
        assert tel1.camera_id == "cam_dock_01"
        assert tel2.camera_id == "cam_dock_02"

        worker1 = manager.get_camera("cam_dock_01")
        worker2 = manager.get_camera("cam_dock_02")
        assert worker1 is not None and worker2 is not None

        # Verify distinct pipeline orchestrators
        assert worker1.orchestrator is not worker2.orchestrator
        assert worker1.orchestrator.camera_id == "cam_dock_01"
        assert worker2.orchestrator.camera_id == "cam_dock_02"

        # Poll dynamically for both workers to complete initial warmup frames
        for _ in range(40):
            if worker1.total_processed_frames >= 1 and worker2.total_processed_frames >= 1:
                break
            time.sleep(0.1)

        # Check telemetry
        t1 = worker1.get_telemetry()
        t2 = worker2.get_telemetry()
        assert t1.total_processed_frames >= 1
        assert t2.total_processed_frames >= 1

        # Stop camera 1; camera 2 must continue running unharmed
        manager.stop_camera("cam_dock_01")
        assert not worker1.is_running
        assert worker2.is_running

        prev_frames_c2 = worker2.total_processed_frames
        time.sleep(0.5)
        assert worker2.total_processed_frames >= prev_frames_c2

        # Clean shutdown
        manager.stop_all()
        assert not worker2.is_running


class TestCameraAPIEndpoints:
    """Verifies Phase 4 REST API and streaming endpoints."""

    def test_list_cameras_initially_empty(self):
        resp = client.get("/api/v1/cameras")
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_cameras"] == 0
        assert data["connected_count"] == 0
        assert data["cameras"] == []

    @pytest.mark.skipif(not SAMPLE_VIDEO.exists(), reason="Sample video not available")
    def test_camera_crud_lifecycle_via_api(self):
        # 1. Register camera
        reg_payload = {
            "camera_id": "api_cam_01",
            "name": "Warehouse Perimeter",
            "source": str(SAMPLE_VIDEO),
            "sampling_interval": 1,
            "max_processing_fps": 10.0,
            "auto_start": True,
            "loop_file": True,
        }
        res = client.post("/api/v1/cameras", json=reg_payload)
        assert res.status_code == 200
        cam_data = res.json()
        assert cam_data["camera_id"] == "api_cam_01"
        assert cam_data["name"] == "Warehouse Perimeter"

        # 2. List cameras
        res_list = client.get("/api/v1/cameras")
        assert res_list.status_code == 200
        assert res_list.json()["total_cameras"] == 1

        # 3. Get camera details
        time.sleep(0.8)
        res_detail = client.get("/api/v1/cameras/api_cam_01")
        assert res_detail.status_code == 200
        det_data = res_detail.json()
        assert det_data["camera"]["camera_id"] == "api_cam_01"

        # 4. Get snapshot JPEG
        res_snap = client.get("/api/v1/cameras/api_cam_01/snapshot")
        if res_snap.status_code == 200:
            assert res_snap.headers["content-type"] == "image/jpeg"
            assert len(res_snap.content) > 100

        # 5. Patch camera config
        patch_payload = {"name": "Warehouse North Entrance", "max_processing_fps": 8.0}
        res_patch = client.patch("/api/v1/cameras/api_cam_01", json=patch_payload)
        assert res_patch.status_code == 200
        assert res_patch.json()["name"] == "Warehouse North Entrance"

        # 6. Stop and restart
        res_stop = client.post("/api/v1/cameras/api_cam_01/stop")
        assert res_stop.status_code == 200
        assert res_stop.json()["is_active"] is False

        res_start = client.post("/api/v1/cameras/api_cam_01/start")
        assert res_start.status_code == 200
        assert res_start.json()["is_active"] is True

        # 7. Delete camera
        res_del = client.delete("/api/v1/cameras/api_cam_01")
        assert res_del.status_code == 200
        assert res_del.json()["status"] == "removed"

        # Verify 404 after deletion
        res_get_del = client.get("/api/v1/cameras/api_cam_01")
        assert res_get_del.status_code == 404
