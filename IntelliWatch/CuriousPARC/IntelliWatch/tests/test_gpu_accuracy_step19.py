"""
tests/test_gpu_accuracy_step19.py
==================================
Unit and Integration tests for Step 19:
GPU-Accelerated Accuracy Improvement, Device Probing, and Precision Enhancements.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from backend.schemas.tracking import TrackState
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from vision.detection.ppe_detector import PPEDetector
from vision.detection.yolo_detector import YOLODetector
from vision.tracking.bytetrack_tracker import ByteTrackTracker
from vision.utils.device import (
    get_device_diagnostics,
    is_cuda_executable,
    probe_cuda_kernel,
    resolve_device,
)


# ==============================================================================
# 1. Device Probing & Acceleration Utilities
# ==============================================================================

def test_device_diagnostics_fields():
    """Verify get_device_diagnostics returns all expected keys and telemetry."""
    diag = get_device_diagnostics()
    assert "torch_version" in diag
    assert "torch_cuda_version" in diag
    assert "cuda_available" in diag
    assert "device_count" in diag
    assert "device_name" in diag
    assert "compute_capability" in diag
    assert "cuda_kernel_executable" in diag
    assert "optimal_device" in diag
    assert diag["optimal_device"] in ("cuda", "cpu")


def test_resolve_device_explicit_cpu():
    """Verify 'cpu' or 'none' always resolves to 'cpu'."""
    assert resolve_device("cpu") == "cpu"
    assert resolve_device("none") == "cpu"
    assert resolve_device("CPU") == "cpu"


def test_resolve_device_auto_and_cuda_safety():
    """
    Verify 'auto' or 'cuda' never crashes and returns an executable device
    ('cuda' if compatible, 'cpu' if unsupported kernels like sm_120 on cu126).
    """
    dev_auto = resolve_device("auto")
    dev_none = resolve_device(None)
    assert dev_auto in ("cuda", "cpu")
    assert dev_none == dev_auto

    dev_cuda = resolve_device("cuda")
    assert dev_cuda in ("cuda", "cpu")
    if not is_cuda_executable():
        assert dev_cuda == "cpu"
        assert dev_auto == "cpu"


def test_probe_cuda_kernel_mock_failure():
    """Verify probe_cuda_kernel handles exceptions gracefully."""
    with patch("torch.cuda.is_available", return_value=True):
        with patch("torch.zeros", side_effect=RuntimeError("Mock CUDA kernel failure")):
            compatible, err = probe_cuda_kernel()
            assert compatible is False
            assert "Mock CUDA kernel failure" in err


# ==============================================================================
# 2. Detector Device Initialization & Fallback
# ==============================================================================

def test_yolo_detector_device_resolution():
    """Verify YOLODetector initializes with safe resolved device."""
    det = YOLODetector(model_path="weights/yolo11n.pt", device="auto")
    assert det.device in ("cuda", "cpu")
    if not is_cuda_executable():
        assert det.device == "cpu"


def test_ppe_detector_device_resolution():
    """Verify PPEDetector initializes with safe resolved device."""
    ppe = PPEDetector(model_path="weights/ppe_yolov8n.pt", device="auto")
    assert ppe.device in ("cuda", "cpu")
    if not is_cuda_executable():
        assert ppe.device == "cpu"


def test_yolo_detector_runtime_cuda_fallback():
    """Verify YOLODetector falls back to CPU if inference on GPU raises error."""
    det = YOLODetector.__new__(YOLODetector)
    det.confidence_threshold = 0.25
    det.iou_threshold = 0.45
    det.device = "cuda"
    det.imgsz = 640
    det.map_to_original = False
    det._filter_classes = None
    det._class_names = {0: "person"}

    mock_model = MagicMock()
    # First call (on cuda) raises AcceleratorError, second call (on cpu) succeeds
    mock_box = MagicMock()
    mock_box.conf = [0.9]
    mock_box.cls = [0]
    mock_box.xyxy = np.array([[10.0, 10.0, 50.0, 100.0]])
    mock_result = MagicMock()
    mock_result.boxes = [mock_box]

    call_count = 0
    def side_effect(*args, **kwargs):
        nonlocal call_count
        call_count += 1
        if kwargs.get("device") == "cuda":
            raise RuntimeError("CUDA error: no kernel image is available")
        return [mock_result]

    mock_model.side_effect = side_effect
    det._model = mock_model

    dummy_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    res = det.detect(dummy_frame)
    assert len(res.detections) == 1
    assert det.device == "cpu"
    assert call_count == 2


# ==============================================================================
# 3. ByteTrack Tracker Class Consistency & Voting
# ==============================================================================

def test_bytetrack_class_voting_accuracy():
    """
    Verify tracker updates class from transient misclassification to dominant class
    as consistent evidence arrives.
    """
    tracker = ByteTrackTracker(track_high_thresh=0.20, track_low_thresh=0.05)

    # Frame 1: Worker detected as 'bench' (cid 15) with low conf 0.55
    box = BoundingBox(x1=100.0, y1=100.0, x2=160.0, y2=250.0)
    det_f1 = FrameDetections(
        frame_id=1, timestamp=0.033, frame_width=640, frame_height=480,
        detections=[DetectionResult(class_id=15, class_name="bench", confidence=0.55, bbox=box)]
    )
    tracks_f1 = tracker.update(det_f1)
    assert len(tracks_f1.active_tracks) == 1
    tid = tracks_f1.active_tracks[0].track_id

    # Frame 2: Same object detected as 'person' (cid 0) with high conf 0.92
    box_f2 = BoundingBox(x1=102.0, y1=100.0, x2=162.0, y2=250.0)
    det_f2 = FrameDetections(
        frame_id=2, timestamp=0.066, frame_width=640, frame_height=480,
        detections=[DetectionResult(class_id=0, class_name="person", confidence=0.92, bbox=box_f2)]
    )
    tracks_f2 = tracker.update(det_f2)
    assert len(tracks_f2.active_tracks) == 1
    # Evidence for 'person' (0.92) exceeds 'bench' (0.55), class updates accurately!
    assert tracks_f2.active_tracks[0].class_name == "person"
    assert tracks_f2.active_tracks[0].class_id == 0


# ==============================================================================
# 4. Orchestrator Integration & Worker Class Support
# ==============================================================================

def test_orchestrator_initializes_with_auto_device():
    """Verify EndToEndPipelineOrchestrator initializes safely with device='auto'."""
    orch = EndToEndPipelineOrchestrator(
        device="auto",
        enable_ppe_model=False,
        enable_industrial_model=False,
        enable_depth_model=False,
    )
    assert orch.device_str in ("cuda", "cpu")
    if not is_cuda_executable():
        assert orch.device_str == "cpu"


def test_orchestrator_supports_worker_class_for_ppe():
    """Verify tracks with class_name='worker' are evaluated for PPE compliance."""
    orch = EndToEndPipelineOrchestrator(
        device="cpu",
        enable_ppe_model=True,
        enable_industrial_model=False,
        enable_depth_model=False,
    )

    dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    worker_box = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=400.0)
    hardhat_box = BoundingBox(x1=120.0, y1=90.0, x2=180.0, y2=140.0)

    manual_dets = [
        DetectionResult(class_id=0, class_name="worker", confidence=0.90, bbox=worker_box)
    ]
    manual_ppe = [
        DetectionResult(class_id=1, class_name="Hardhat", confidence=0.88, bbox=hardhat_box)
    ]

    assessment, ann = orch.process_frame(
        frame=dummy_frame,
        frame_id=1,
        timestamp=0.033,
        manual_detections=manual_dets,
        manual_ppe_detections=manual_ppe,
    )

    assert len(assessment.tracks) == 1
    assert assessment.tracks[0].class_name in ("worker", "person")
    # PPE inventory was generated for the worker
    assert len(assessment.worker_inventories) == 1
    inv = assessment.worker_inventories[0]
    assert len(inv.items) == 1
    assert inv.items[0].class_name == "Hardhat"
