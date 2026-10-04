import os
import pytest
import numpy as np
import cv2
from pathlib import Path

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from backend.schemas.tracking import FrameTracks, TrackedObject, TrackState
from configs.settings import get_settings
from intelligence.behavior.behavior_engine import BehaviorEngine
from intelligence.events.ppe_compliance import PPEComplianceEngine
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from vision.detection.ppe_association import PPEAssociationEngine
from vision.detection.ppe_detector import PPEDetector
from vision.detection.yolo_detector import YOLODetector
from vision.tracking.bytetrack_tracker import ByteTrackTracker
from vision.utils.device import (
    get_device_diagnostics,
    is_cuda_executable,
    probe_cuda_kernel,
    resolve_device,
)


@pytest.fixture
def sample_image_path():
    path = Path("data/samples/industrial_cctv.jpg")
    if not path.exists():
        pytest.skip("Sample image data/samples/industrial_cctv.jpg not found")
    return str(path)


@pytest.fixture
def sample_video_path():
    path = Path("data/samples/cctv_worker_moving.mp4")
    if not path.exists():
        pytest.skip("Sample video data/samples/cctv_worker_moving.mp4 not found")
    return str(path)


class TestGPUResilienceAndDiagnostics:
    """Verifies Phase 1 GPU compatibility, hardware probing, and CPU fallback."""

    def test_device_diagnostics_structure(self):
        diag = get_device_diagnostics()
        assert "cuda_available" in diag
        assert "cuda_kernel_executable" in diag
        assert "optimal_device" in diag
        if diag.get("cuda_kernel_executable"):
            assert "cuda" in diag["optimal_device"]
        else:
            assert diag["optimal_device"] == "cpu"

    def test_probe_cuda_kernel_returns_bool_and_message(self):
        ok, msg = probe_cuda_kernel()
        assert isinstance(ok, bool)
        if not ok:
            assert isinstance(msg, str)
            assert "CUDA" in msg or "kernel" in msg or "unavailable" in msg
        else:
            assert msg is None or isinstance(msg, str)

    def test_resolve_device_safety(self):
        diag = get_device_diagnostics()
        resolved = resolve_device("cuda")
        if diag.get("cuda_kernel_executable"):
            assert "cuda" in resolved
        else:
            assert resolved == "cpu"
        # Explicit cpu requested returns cpu
        assert resolve_device("cpu") == "cpu"


class TestRealWorldPerceptionEvaluation:
    """Verifies Phase 2 real-world perception on authentic CCTV assets."""

    def test_yolo_detection_on_real_image(self, sample_image_path):
        img = cv2.imread(sample_image_path)
        assert img is not None and img.shape[0] > 0
        detector = YOLODetector(device="cpu")
        detections = detector.detect(img)
        assert len(detections.detections) > 0
        person_dets = [d for d in detections.detections if d.class_name.lower() in ("person", "worker")]
        assert len(person_dets) >= 1
        assert person_dets[0].confidence >= 0.70

    def test_ppe_detection_and_negative_class_recognition(self, sample_image_path):
        img = cv2.imread(sample_image_path)
        ppe_detector = PPEDetector(device="cpu")
        detections = ppe_detector.detect(img)
        assert isinstance(detections, FrameDetections)
        # Verify model outputs classes
        class_names = [d.class_name for d in detections.detections]
        assert len(class_names) > 0

    def test_harness_and_fall_class_normalization(self):
        engine = PPEAssociationEngine()
        norm_harness = engine.normalize_class("no_harness")
        assert norm_harness == ("Harness", True)

        norm_pos_harness = engine.normalize_class("harness")
        assert norm_pos_harness == ("Harness", False)

        norm_fall = engine.normalize_class("fall-detected")
        assert norm_fall == ("Fall-Detected", False)


class TestModelAssistedFallDetectionAndBehavior:
    """Verifies Phase 3 improvements: fall detection routing and behavior integration."""

    def test_behavior_engine_external_fall_trigger(self):
        engine = BehaviorEngine()
        bbox = BoundingBox(x1=100, y1=100, x2=200, y2=300)
        track = TrackedObject(
            track_id=42,
            class_id=0,
            class_name="worker",
            bbox=bbox,
            confidence=0.9,
            state=TrackState.ACTIVE,
        )
        frame_tracks = FrameTracks(
            frame_id=1,
            timestamp=0.033,
            active_tracks=[track],
            total_track_count=1,
        )

        # Process with external_fall_tracks containing track 42
        states, events = engine.process(frame_tracks, external_fall_tracks={42})
        assert len(states) == 1
        assert states[0].flag_possible_fall is True
        # An event for possible fall should be generated
        fall_events = [e for e in events if "fall" in e.event_type.value.lower()]
        assert len(fall_events) >= 1
        assert fall_events[0].track_id == 42

    def test_orchestrator_fall_detection_integration(self):
        orchestrator = EndToEndPipelineOrchestrator(
            enable_depth_model=False,
            enable_industrial_model=False,
            enable_ppe_model=False,  # Use manual injection
            device="cpu",
        )
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        person_bbox = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
        manual_dets = [
            DetectionResult(
                class_id=0,
                class_name="person",
                confidence=0.95,
                bbox=person_bbox,
            )
        ]
        # Inject Fall-Detected PPE detection
        fall_bbox = BoundingBox(x1=110.0, y1=120.0, x2=190.0, y2=280.0)
        manual_ppe = [
            DetectionResult(
                class_id=0,
                class_name="Fall-Detected",
                confidence=0.85,
                bbox=fall_bbox,
            )
        ]

        assessment, annotated = orchestrator.process_frame(
            frame=dummy_frame,
            frame_id=1,
            timestamp=0.033,
            manual_detections=manual_dets,
            manual_ppe_detections=manual_ppe,
        )

        assert assessment.frame_id == 1
        # Worker should be tracked
        assert len(assessment.tracks) >= 1
        # Behavior assessment should record the fall
        worker_state = assessment.behavior_states[0]
        assert worker_state.flag_possible_fall is True


class TestEndToEndVideoProcessing:
    """Verifies Phase 4 video execution and stability."""

    def test_process_moving_worker_video(self, sample_video_path):
        cap = cv2.VideoCapture(sample_video_path)
        assert cap.isOpened(), "Could not open sample video"

        orchestrator = EndToEndPipelineOrchestrator(
            enable_depth_model=False,
            enable_industrial_model=False,
            enable_ppe_model=True,
            device="cpu",
        )

        frames_processed = 0
        total_tracks_seen = set()

        while frames_processed < 15:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            assessment, annotated = orchestrator.process_frame(
                frame=frame,
                frame_id=frames_processed,
                timestamp=frames_processed / 30.0,
            )
            frames_processed += 1
            for t in assessment.tracks:
                total_tracks_seen.add(t.track_id)
            assert annotated.shape == frame.shape

        cap.release()
        assert frames_processed == 15
        assert len(total_tracks_seen) >= 1
