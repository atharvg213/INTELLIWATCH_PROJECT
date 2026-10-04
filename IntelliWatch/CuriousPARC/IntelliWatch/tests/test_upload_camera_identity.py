"""Regression coverage for source-camera identity on uploaded analysis jobs."""
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api import routes
from backend.main import app
from backend.schemas.analysis import ImageAnalysisResponse, JobStatus, MediaType, VideoJobSubmitResponse
from backend.schemas.assessment import FrameAssessment
from backend.schemas.detection import BoundingBox
from backend.schemas.scene_graph import (
    FrameScene,
    RelationLifecycle,
    SceneNode,
    SceneNodeType,
    SceneRelation,
    SceneRelationType,
)
from backend.services.job_manager import JobManager
import backend.services.job_manager as job_manager_module
import backend.services.pipeline_manager as pipeline_manager_module
import backend.services.calibration_service as calibration_service_module


@pytest.fixture
def client():
    return TestClient(app)


def _png_bytes():
    ok, encoded = cv2.imencode(".png", np.zeros((32, 48, 3), dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class _CameraRegistry:
    def __init__(self, camera_ids):
        self.camera_ids = set(camera_ids)

    def get_camera(self, camera_id):
        return SimpleNamespace(camera_id=camera_id) if camera_id in self.camera_ids else None


class _CalibrationRegistry:
    def __init__(self, camera_ids=()):
        self.camera_ids = set(camera_ids)
        self.lookup_ids = []

    def get_calibration(self, camera_id):
        self.lookup_ids.append(camera_id)
        if camera_id in self.camera_ids:
            return SimpleNamespace(camera_id=camera_id, enabled=True, calibration_status="CALIBRATED")
        return None

    def get_calibrator(self, camera_id):
        self.lookup_ids.append(camera_id)
        if camera_id == "cam_A":
            return _FakeCalibrator()
        return None


class _FakeCalibrator:
    camera_id = "cam_A"
    enabled = True
    is_valid = True

    def image_to_ground(self, x, y):
        return (x / 10.0, y / 10.0)

    def compute_ground_distance(self, point_a, point_b):
        return 4.25

    def compute_ground_speed(self, x, y, speed_px_per_s, velocity_vector=None):
        return 1.75


def _install_upload_registries(monkeypatch, camera_ids=("cam_A", "cam_B"), calibration_ids=()):
    monkeypatch.setattr(routes, "get_camera_manager", lambda: _CameraRegistry(camera_ids))
    monkeypatch.setattr(routes, "get_calibration_service", lambda: _CalibrationRegistry(calibration_ids))


def _image_response(camera_id):
    pipeline_camera_id = camera_id or "unassigned"
    scene = FrameScene(
        timestamp=0.0,
        camera_id=pipeline_camera_id,
        source_camera_id=camera_id,
    )
    return ImageAnalysisResponse(
        job_id="job_img_camera_identity",
        camera_id=pipeline_camera_id,
        source_camera_id=camera_id,
        filename="upload.png",
        annotated_media_url="/api/v1/media/result.jpg",
        assessment=FrameAssessment(camera_id=pipeline_camera_id, scene=scene),
    )


def test_image_upload_passes_selected_camera_id_to_job_manager(client, monkeypatch):
    _install_upload_registries(monkeypatch)
    received = {}

    class FakeJobManager:
        def process_image(self, **kwargs):
            received.update(kwargs)
            return _image_response(kwargs.get("camera_id"))

    monkeypatch.setattr(routes, "get_job_manager", lambda: FakeJobManager())
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("worker.png", _png_bytes(), "image/png")},
        data={"camera_id": "cam_A"},
    )

    assert response.status_code == 200
    assert received["camera_id"] == "cam_A"
    assert response.json()["camera_id"] == "cam_A"
    assert response.json()["source_camera_id"] == "cam_A"


def test_video_upload_passes_selected_camera_id_to_job_manager(client, monkeypatch):
    _install_upload_registries(monkeypatch)
    received = {}

    class FakeJobManager:
        def start_video_job(self, **kwargs):
            received.update(kwargs)
            return VideoJobSubmitResponse(
                job_id="job_vid_camera_identity",
                camera_id=kwargs.get("camera_id") or "unassigned",
                source_camera_id=kwargs.get("camera_id"),
                filename=kwargs["original_filename"],
            )

    monkeypatch.setattr(routes, "get_job_manager", lambda: FakeJobManager())
    response = client.post(
        "/api/v1/analyze/video",
        files={"file": ("worker.mp4", b"video-bytes", "video/mp4")},
        data={"camera_id": "cam_B"},
    )

    assert response.status_code == 202
    assert received["camera_id"] == "cam_B"
    assert response.json()["camera_id"] == "cam_B"
    assert response.json()["source_camera_id"] == "cam_B"


def test_upload_rejects_camera_id_unknown_to_registry_and_configuration(client, monkeypatch):
    _install_upload_registries(monkeypatch, camera_ids=(), calibration_ids=())
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("worker.png", _png_bytes(), "image/png")},
        data={"camera_id": "cam_missing"},
    )
    assert response.status_code == 400
    assert "Unknown camera_id" in response.json()["detail"]


def test_upload_accepts_camera_id_from_calibration_configuration(client, monkeypatch):
    _install_upload_registries(monkeypatch, camera_ids=(), calibration_ids=("cam_configured",))

    class FakeJobManager:
        def process_image(self, **kwargs):
            return _image_response(kwargs.get("camera_id"))

    monkeypatch.setattr(routes, "get_job_manager", lambda: FakeJobManager())
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("worker.png", _png_bytes(), "image/png")},
        data={"camera_id": "cam_configured"},
    )

    assert response.status_code == 200
    assert response.json()["source_camera_id"] == "cam_configured"


class _FakeOrchestrator:
    def __init__(self, camera_id):
        self.camera_id = camera_id
        self.reset_count = 0
        self.received_camera_id = None

    def reset(self):
        self.reset_count += 1

    def process_frame(self, frame, **kwargs):
        self.received_camera_id = self.camera_id
        scene = FrameScene(
            frame_id=kwargs.get("frame_id", 1),
            timestamp=kwargs.get("timestamp", 0.0),
            camera_id=self.camera_id,
            source_camera_id=None if self.camera_id == "unassigned" else self.camera_id,
        )
        assessment = FrameAssessment(camera_id=self.camera_id, scene=scene)
        return assessment, frame.copy()

    def get_performance_stats(self):
        return {"processing_fps": 1.0, "avg_latency_ms": 1000.0}


class _FakePipelineManager:
    def __init__(self):
        self.orchestrators = {}
        self.received_camera_ids = []

    def get_orchestrator(self, camera_id=None):
        resolved = camera_id or "unassigned"
        self.received_camera_ids.append(resolved)
        return self.orchestrators.setdefault(resolved, _FakeOrchestrator(resolved))


def test_image_job_propagates_camera_to_orchestrator_and_job_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(job_manager_module, "INPUT_UPLOADS_DIR", tmp_path / "input")
    monkeypatch.setattr(job_manager_module, "OUTPUT_UPLOADS_DIR", tmp_path / "output")
    pipeline_manager = _FakePipelineManager()
    monkeypatch.setattr(job_manager_module, "get_pipeline_manager", lambda: pipeline_manager)

    manager = JobManager()
    result = manager.process_image(_png_bytes(), "worker.png", camera_id="cam_B")
    job = manager.get_job(result.job_id)

    assert pipeline_manager.received_camera_ids == ["cam_B"]
    assert result.source_camera_id == "cam_B"
    assert result.assessment.camera_id == "cam_B"
    assert result.assessment.scene.source_camera_id == "cam_B"
    assert job["camera_id"] == "cam_B"
    assert job["source_camera_id"] == "cam_B"


def test_video_job_stores_selected_camera_id_in_job_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(job_manager_module, "INPUT_UPLOADS_DIR", tmp_path / "input")
    monkeypatch.setattr(job_manager_module, "OUTPUT_UPLOADS_DIR", tmp_path / "output")

    class FakeCapture:
        def __init__(self, _path):
            pass

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 10.0,
                cv2.CAP_PROP_FRAME_COUNT: 1.0,
                cv2.CAP_PROP_FRAME_WIDTH: 32.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 24.0,
            }.get(prop, 0.0)

        def read(self):
            return True, np.zeros((24, 32, 3), dtype=np.uint8)

        def release(self):
            pass

    class DormantThread:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

        def start(self):
            pass

    monkeypatch.setattr(job_manager_module.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(job_manager_module.threading, "Thread", DormantThread)
    manager = JobManager()

    response = manager.start_video_job(b"video-bytes", "worker.mp4", camera_id="cam_B")
    job = manager.get_job(response.job_id)

    assert response.camera_id == "cam_B"
    assert response.source_camera_id == "cam_B"
    assert job["camera_id"] == "cam_B"
    assert job["source_camera_id"] == "cam_B"


def test_pipeline_manager_keeps_camera_orchestrators_separate(monkeypatch):
    created = []

    def fake_orchestrator(**kwargs):
        orchestrator = _FakeOrchestrator(kwargs["camera_id"])
        created.append(orchestrator)
        return orchestrator

    monkeypatch.setattr(pipeline_manager_module, "EndToEndPipelineOrchestrator", fake_orchestrator)
    manager = pipeline_manager_module.PipelineManager()

    camera_a = manager.get_orchestrator("cam_A")
    camera_b = manager.get_orchestrator("cam_B")
    unassigned = manager.get_orchestrator()

    assert camera_a.camera_id == "cam_A"
    assert camera_b.camera_id == "cam_B"
    assert camera_a is not camera_b
    assert manager.get_orchestrator("cam_A") is camera_a
    assert unassigned.camera_id == "unassigned"
    assert len(created) == 3


def test_video_worker_passes_job_camera_id_to_pipeline(monkeypatch, tmp_path):
    frame = np.zeros((24, 32, 3), dtype=np.uint8)

    class FakeCapture:
        def __init__(self, _path):
            self.frames = [frame]

        def isOpened(self):
            return True

        def get(self, prop):
            return {
                cv2.CAP_PROP_FPS: 10.0,
                cv2.CAP_PROP_FRAME_COUNT: 1.0,
                cv2.CAP_PROP_FRAME_WIDTH: 32.0,
                cv2.CAP_PROP_FRAME_HEIGHT: 24.0,
            }.get(prop, 0.0)

        def read(self):
            return (True, self.frames.pop()) if self.frames else (False, None)

        def release(self):
            pass

    class FakeWriter:
        def isOpened(self):
            return True

        def write(self, _frame):
            pass

        def release(self):
            pass

    monkeypatch.setattr(job_manager_module.cv2, "VideoCapture", FakeCapture)
    monkeypatch.setattr(job_manager_module.cv2, "VideoWriter_fourcc", lambda *_args: 0)
    monkeypatch.setattr(job_manager_module.cv2, "VideoWriter", lambda *_args: FakeWriter())
    monkeypatch.setattr(job_manager_module.cv2, "imwrite", lambda *_args: True)

    pipeline_manager = _FakePipelineManager()
    monkeypatch.setattr(job_manager_module, "get_pipeline_manager", lambda: pipeline_manager)
    manager = JobManager()
    job_id = "job_vid_pipeline_camera"
    manager._jobs[job_id] = {
        "job_id": job_id,
        "camera_id": "cam_B",
        "source_camera_id": "cam_B",
        "media_type": MediaType.VIDEO,
        "status": JobStatus.QUEUED,
        "filename": "worker.mp4",
        "input_path": tmp_path / "input.mp4",
        "output_video_path": tmp_path / "out.mp4",
        "latest_frame_path": tmp_path / "latest.jpg",
        "annotated_frame_path": tmp_path / "annotated.jpg",
        "total_frames": 1,
        "processed_frames": 0,
        "current_frame": 0,
        "progress_pct": 0.0,
        "processing_fps": 0.0,
        "avg_latency_ms": 0.0,
        "current_risk_level": "INFO",
        "highest_risk_tier": "INFO",
        "highest_risk_score": 0.0,
        "active_tracks": 0,
        "incidents_count": 0,
        "new_incident_ids": [],
        "error_message": None,
        "fps": 10.0,
        "width": 32,
        "height": 24,
    }

    manager._run_video_worker(job_id)

    assert pipeline_manager.received_camera_ids == ["cam_B"]
    assert manager.get_job(job_id)["assessment"].camera_id == "cam_B"
    assert manager.get_job(job_id)["assessment"].scene.source_camera_id == "cam_B"


def _scene_for_camera(camera_id):
    person = SceneNode(
        node_id="person_1",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        bbox=BoundingBox(x1=10, y1=10, x2=30, y2=70),
        centroid=(20, 40),
        contact_point=(20, 70),
        attributes={"speed_px_per_s": 30.0},
    )
    vehicle = SceneNode(
        node_id="forklift_1",
        node_type=SceneNodeType.VEHICLE,
        class_name="forklift",
        bbox=BoundingBox(x1=60, y1=10, x2=90, y2=70),
        centroid=(75, 40),
        contact_point=(75, 70),
    )
    relation = SceneRelation(
        relation_id="approach_1",
        source_node_id="forklift_1",
        target_node_id="person_1",
        relation_type=SceneRelationType.APPROACHING,
        lifecycle=RelationLifecycle.ACTIVE,
        evidence={"distance_px": 55.0, "closing_rate_px_s": 20.0},
    )
    return FrameScene(
        timestamp=1.0,
        camera_id=camera_id,
        source_camera_id=camera_id if camera_id != "unassigned" else None,
        nodes=[person, vehicle],
        relationships=[relation],
        is_video=True,
    )


def test_job_scene_uses_its_source_identity_and_never_borrows_camera_a_calibration(client, monkeypatch):
    calibration_service = _CalibrationRegistry(camera_ids=("cam_A",))
    monkeypatch.setattr(calibration_service_module, "get_calibration_service", lambda: calibration_service)
    manager = JobManager()
    job_id = "job_vid_cam_b_scene"
    scene = _scene_for_camera("cam_B")
    manager._jobs[job_id] = {
        "job_id": job_id,
        "camera_id": "cam_B",
        "source_camera_id": "cam_B",
        "media_type": MediaType.VIDEO,
        "assessment": FrameAssessment(camera_id="cam_B", scene=scene),
    }
    monkeypatch.setattr(job_manager_module, "get_job_manager", lambda: manager)

    response = client.get(f"/api/v1/scene/graph?job_id={job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["camera_id"] == "cam_B"
    assert data["source_camera_id"] == "cam_B"
    assert data["calibration_status"] == "UNCONFIGURED"
    assert data["spatial_basis"] == "IMAGE_SPACE"
    assert calibration_service.lookup_ids == ["cam_B", "cam_B"]
    assert all(entity["position"].get("x_m") is None for entity in data["entities"])
    person = next(entity for entity in data["entities"] if entity["id"] == "person_1")
    assert person["state"]["speed_m_per_s"] is None
    assert data["relationships"][0]["distance_m"] is None
    assert data["relationships"][0]["closing_rate_m_s"] is None

    mismatched = client.get(f"/api/v1/scene/graph?job_id={job_id}&camera_id=cam_A")
    assert mismatched.status_code == 200
    assert mismatched.json()["source_camera_id"] == "cam_B"
    assert mismatched.json()["relationships"][0]["distance_m"] is None


def test_unassigned_job_scene_stays_in_image_space(client, monkeypatch):
    calibration_service = _CalibrationRegistry(camera_ids=("cam_A",))
    monkeypatch.setattr(calibration_service_module, "get_calibration_service", lambda: calibration_service)
    manager = JobManager()
    job_id = "job_img_unassigned_scene"
    manager._jobs[job_id] = {
        "job_id": job_id,
        "camera_id": "unassigned",
        "source_camera_id": None,
        "media_type": MediaType.IMAGE,
        "assessment": FrameAssessment(camera_id="unassigned", scene=_scene_for_camera("unassigned")),
    }
    monkeypatch.setattr(job_manager_module, "get_job_manager", lambda: manager)

    response = client.get(f"/api/v1/scene/graph?job_id={job_id}")
    assert response.status_code == 200
    data = response.json()
    assert data["camera_id"] == "unassigned"
    assert data["source_camera_id"] is None
    assert data["calibration_status"] == "UNAVAILABLE"
    assert data["spatial_basis"] == "IMAGE_SPACE"
    assert calibration_service.lookup_ids == []
    assert data["relationships"][0]["distance_m"] is None
    assert data["entities"][0]["position"].get("x_m") is None


def test_missing_camera_id_preserves_image_upload_with_explicit_unassigned_metadata(client, monkeypatch):
    _install_upload_registries(monkeypatch)

    class FakeJobManager:
        def process_image(self, **kwargs):
            return _image_response(kwargs.get("camera_id"))

    monkeypatch.setattr(routes, "get_job_manager", lambda: FakeJobManager())
    response = client.post(
        "/api/v1/analyze/image",
        files={"file": ("worker.png", _png_bytes(), "image/png")},
    )

    assert response.status_code == 200
    assert response.json()["camera_id"] == "unassigned"
    assert response.json()["source_camera_id"] is None

