"""
Unit tests for Step 5A PPE Detection Model Integration.
Verifies:
  1. PPEDetector initialization and configuration loading
  2. Empty / invalid input error handling
  3. Output schema conformity (FrameDetections, DetectionResult, BoundingBox)
  4. Coordinate preservation in original image coordinates
  5. Confidence threshold filtering
  6. Explicit CPU device configuration
  7. Detector instance reuse across multiple frames (no per-frame reloading)
  8. Class filtering behavior
"""
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from configs.settings import get_settings
from vision.detection.base import BaseDetector
from vision.detection.ppe_detector import PPEInferenceError, PPEModelLoadError, PPEDetector
from vision.preprocessing.frame import FrameData


# 1. Model Initialization
def test_ppe_detector_initialization():
    settings = get_settings()
    detector = PPEDetector(
        model_path="weights/ppe_yolov8n.pt",
        confidence_threshold=0.30,
        device="cpu",
    )
    assert isinstance(detector, BaseDetector)
    assert detector.is_ready
    assert detector.device == "cpu"
    assert detector.confidence_threshold == 0.30
    assert len(detector.class_names) >= 5
    # Verify core safety gear classes are mapped
    class_values = [c.lower() for c in detector.class_names.values()]
    assert any("hardhat" in c or "helmet" in c for c in class_values)
    assert any("vest" in c for c in class_values)


# 2. Empty / Invalid Input Handling
def test_ppe_detector_invalid_input():
    detector = PPEDetector(device="cpu")

    # None input
    with pytest.raises(ValueError, match="cannot be None"):
        detector.detect(None)

    # Empty array
    with pytest.raises(ValueError, match="non-empty"):
        detector.detect(np.empty((0, 0, 3), dtype=np.uint8))


# 3. Output Schema Validation
def test_ppe_detector_output_schema():
    detector = PPEDetector(device="cpu", confidence_threshold=0.10)
    # Synthetic frame with dimensions
    canvas = np.full((360, 640, 3), 100, dtype=np.uint8)
    frame_detections = detector.detect(canvas, frame_id=42, timestamp=1.4)

    assert isinstance(frame_detections, FrameDetections)
    assert frame_detections.frame_id == 42
    assert frame_detections.timestamp == 1.4
    assert frame_detections.frame_width == 640
    assert frame_detections.frame_height == 360

    for det in frame_detections.detections:
        assert isinstance(det, DetectionResult)
        assert isinstance(det.class_id, int)
        assert isinstance(det.class_name, str)
        assert 0.0 <= det.confidence <= 1.0
        assert isinstance(det.bbox, BoundingBox)
        assert det.bbox.x1 <= det.bbox.x2
        assert det.bbox.y1 <= det.bbox.y2


# 4. Coordinate Transformation & Preservation
def test_ppe_detector_coordinate_preservation():
    detector = PPEDetector(device="cpu", map_to_original=True)

    # Create FrameData with letterboxing scaling metadata (orig: 1280x720 -> letterbox: 640x640)
    letterboxed = np.zeros((640, 640, 3), dtype=np.uint8)
    frame_data = FrameData(
        image=letterboxed,
        frame_index=1,
        timestamp=0.033,
        width=640,
        height=640,
        original_shape=(720, 1280),
        scale_factor=0.5,
        pad_offset=(140, 0),  # top pad=140, left pad=0
    )

    result = detector.detect(frame_data)
    assert isinstance(result, FrameDetections)
    assert result.frame_width == 1280
    assert result.frame_height == 720

    for d in result.detections:
        # Clamped within original dimensions
        assert 0.0 <= d.bbox.x1 <= 1280.0
        assert 0.0 <= d.bbox.y1 <= 720.0
        assert 0.0 <= d.bbox.x2 <= 1280.0
        assert 0.0 <= d.bbox.y2 <= 720.0


# 5. Confidence Threshold Filtering
def test_ppe_detector_confidence_filtering():
    high_conf_detector = PPEDetector(confidence_threshold=0.99, device="cpu")
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)
    res = high_conf_detector.detect(canvas)
    # At 0.99 threshold on black frame, no noise detection should pass
    assert len(res.detections) == 0


# 6. Device Configuration (CPU Verification)
def test_ppe_detector_device_config():
    detector = PPEDetector(device="cpu")
    assert detector.device == "cpu"
    # Verify underlying PyTorch model weights are on CPU
    param = next(detector._model.model.parameters())
    assert param.device.type == "cpu"


# 7. Detector Reuse (Single Initialization)
def test_ppe_detector_model_reuse():
    detector = PPEDetector(device="cpu")
    initial_model_obj = detector._model

    canvas1 = np.zeros((300, 300, 3), dtype=np.uint8)
    canvas2 = np.ones((300, 300, 3), dtype=np.uint8) * 128

    detector.detect(canvas1, frame_id=1)
    detector.detect(canvas2, frame_id=2)

    # Model reference must stay identical (not recreated per frame)
    assert detector._model is initial_model_obj


# 8. Class Filter Configuration
def test_ppe_detector_class_filtering():
    detector = PPEDetector(device="cpu")
    detector.set_class_filter(["Hardhat", "Safety Vest"])
    assert detector._filter_classes == {"Hardhat", "Safety Vest"}

    detector.set_class_filter(None)
    assert detector._filter_classes is None
