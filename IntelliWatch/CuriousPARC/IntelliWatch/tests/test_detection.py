"""
Comprehensive Unit and Integration Tests for Step 3: Object Detection.
Tests YOLODetector initialization, interface compliance, confidence filtering,
class filtering, empty detection handling, schema conversion, visualizer,
and end-to-end inference on local test assets.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock
import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from vision.detection.base import BaseDetector
from vision.detection.visualizer import DetectionVisualizer
from vision.detection.yolo_detector import InferenceError, ModelLoadError, YOLODetector
from vision.preprocessing.frame import FrameData


class DummyBox:
    """Mock for Ultralytics box representation."""
    def __init__(self, xyxy, conf, cls):
        self.xyxy = np.array([xyxy])
        self.conf = np.array([conf])
        self.cls = np.array([cls])


class DummyResult:
    """Mock for Ultralytics inference result."""
    def __init__(self, boxes):
        self.boxes = boxes


def test_detector_interface_compliance():
    """Verify YOLODetector strictly implements BaseDetector ABC."""
    assert issubclass(YOLODetector, BaseDetector)
    detector = YOLODetector.__new__(YOLODetector)
    assert hasattr(detector, "load_model")
    assert hasattr(detector, "detect")
    assert hasattr(detector, "class_names")


def test_detector_initialization_defaults():
    """Verify YOLODetector initializes with proper configuration defaults."""
    # Initialize without immediate loading to test attributes
    det = YOLODetector(
        model_path="weights/yolo11n.pt",
        confidence_threshold=0.30,
        iou_threshold=0.40,
        device="cpu",
    )
    assert det.confidence_threshold == 0.30
    assert det.iou_threshold == 0.40
    assert det.device == "cpu"
    assert det.is_loaded is True
    assert 0 in det.class_names
    assert det.class_names[0] == "person"


def test_invalid_frame_inputs():
    """Verify detector raises appropriate ValueError on None or empty arrays."""
    det = YOLODetector(model_path="weights/yolo11n.pt")

    with pytest.raises(ValueError, match="Input frame cannot be None"):
        det.detect(None)

    empty_arr = np.array([])
    with pytest.raises(ValueError, match="Frame image must be a non-empty numpy.ndarray"):
        det.detect(empty_arr)


def test_mock_inference_and_schema_conversion():
    """
    Test end-to-end schema conversion, bounding box extraction,
    and confidence thresholding using an isolated mock YOLO model.
    Guarantees offline deterministic validation.
    """
    det = YOLODetector.__new__(YOLODetector)
    det.confidence_threshold = 0.50
    det.iou_threshold = 0.45
    det.device = "cpu"
    det.imgsz = 640
    det.map_to_original = True
    det._filter_classes = None
    det._class_names = {0: "person", 2: "car", 7: "truck"}

    # Mock YOLO model response with 3 candidate boxes:
    # Box 1: person with 0.85 conf (pass)
    # Box 2: car with 0.30 conf (fail threshold < 0.50)
    # Box 3: truck with 0.60 conf (pass)
    mock_boxes = [
        DummyBox(xyxy=[100.0, 50.0, 200.0, 300.0], conf=0.85, cls=0),
        DummyBox(xyxy=[300.0, 100.0, 500.0, 400.0], conf=0.30, cls=2),
        DummyBox(xyxy=[50.0, 200.0, 250.0, 450.0], conf=0.60, cls=7),
    ]
    mock_model = MagicMock()
    mock_model.return_value = [DummyResult(boxes=mock_boxes)]
    det._model = mock_model

    # Run detection on raw frame
    dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
    res = det.detect(dummy_img, frame_id=42, timestamp=1.4)

    assert isinstance(res, FrameDetections)
    assert res.frame_id == 42
    assert res.timestamp == 1.4
    assert res.frame_width == 640
    assert res.frame_height == 480
    assert len(res.detections) == 2

    d1, d2 = res.detections
    assert d1.class_name == "person"
    assert d1.confidence == 0.85
    assert d1.bbox.x1 == 100.0

    assert d2.class_name == "truck"
    assert d2.confidence == 0.60
    assert d2.bbox.x1 == 50.0


def test_class_filtering_behavior():
    """Verify that class filtering accurately rejects non-matching classes."""
    det = YOLODetector.__new__(YOLODetector)
    det.confidence_threshold = 0.25
    det.iou_threshold = 0.45
    det.device = "cpu"
    det.imgsz = 640
    det.map_to_original = True
    det._filter_classes = {"person"}  # Only allow 'person'
    det._class_names = {0: "person", 2: "car", 7: "truck"}

    mock_boxes = [
        DummyBox(xyxy=[10.0, 10.0, 50.0, 50.0], conf=0.90, cls=0),  # person -> keep
        DummyBox(xyxy=[60.0, 60.0, 120.0, 120.0], conf=0.95, cls=7),  # truck -> reject
    ]
    mock_model = MagicMock()
    mock_model.return_value = [DummyResult(boxes=mock_boxes)]
    det._model = mock_model

    raw_frame = np.zeros((200, 200, 3), dtype=np.uint8)
    res = det.detect(raw_frame)

    assert len(res.detections) == 1
    assert res.detections[0].class_name == "person"

    # Reset filter and verify all allowed
    det.set_class_filter(None)
    res_all = det.detect(raw_frame)
    assert len(res_all.detections) == 2


def test_detection_on_preprocessed_framedata():
    """Verify coordinate transformation is invoked when passing FrameData."""
    det = YOLODetector.__new__(YOLODetector)
    det.confidence_threshold = 0.25
    det.iou_threshold = 0.45
    det.device = "cpu"
    det.imgsz = 640
    det.map_to_original = True
    det._filter_classes = None
    det._class_names = {0: "person"}

    # Preprocessed frame was scaled by 0.5 with top padding of 50px
    # Target box in 640x640: [100, 150, 200, 350]
    # In original: x1 = 100 / 0.5 = 200; y1 = (150 - 50) / 0.5 = 200
    mock_boxes = [DummyBox(xyxy=[100.0, 150.0, 200.0, 350.0], conf=0.80, cls=0)]
    mock_model = MagicMock()
    mock_model.return_value = [DummyResult(boxes=mock_boxes)]
    det._model = mock_model

    frame_data = FrameData(
        frame_index=10,
        timestamp=0.333,
        image=np.zeros((640, 640, 3), dtype=np.uint8),
        width=640,
        height=640,
        is_preprocessed=True,
        original_shape=(1000, 1000),
        scale_factor=0.5,
        pad_offset=(50, 0),
    )

    res = det.detect(frame_data)
    assert res.frame_id == 10
    assert res.frame_width == 1000
    assert res.frame_height == 1000
    assert len(res.detections) == 1

    box = res.detections[0].bbox
    assert box.x1 == pytest.approx(200.0, abs=0.1)
    assert box.y1 == pytest.approx(200.0, abs=0.1)
    assert box.x2 == pytest.approx(400.0, abs=0.1)
    assert box.y2 == pytest.approx(600.0, abs=0.1)


def test_empty_detections_handling():
    """Verify that an image with 0 detections produces a valid empty FrameDetections."""
    det = YOLODetector.__new__(YOLODetector)
    det.confidence_threshold = 0.25
    det.iou_threshold = 0.45
    det.device = "cpu"
    det.imgsz = 640
    det.map_to_original = True
    det._filter_classes = None
    det._class_names = {0: "person"}

    mock_model = MagicMock()
    mock_model.return_value = [DummyResult(boxes=[])]
    det._model = mock_model

    res = det.detect(np.zeros((100, 100, 3), dtype=np.uint8))
    assert isinstance(res, FrameDetections)
    assert len(res.detections) == 0


def test_visualizer_utility(tmp_path):
    """Verify DetectionVisualizer renders bounding boxes and persists cleanly."""
    visualizer = DetectionVisualizer(box_thickness=2, font_scale=0.5)
    canvas = np.full((400, 600, 3), 50, dtype=np.uint8)

    det = DetectionResult(
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=BoundingBox(x1=50.0, y1=50.0, x2=200.0, y2=350.0),
    )
    frame_dets = FrameDetections(
        frame_id=5,
        timestamp=0.166,
        detections=[det],
        frame_width=600,
        frame_height=400,
    )

    annotated = visualizer.draw_detections(canvas, frame_dets)
    assert annotated.shape == canvas.shape
    # Canvas should have been copied, not modified in place
    assert not np.array_equal(annotated, canvas)

    out_file = tmp_path / "test_annotated.jpg"
    saved_path = visualizer.save_annotated_frame(annotated, out_file)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0


@pytest.mark.skipif(
    not (Path("weights/yolo11n.pt").exists() or Path("yolo11n.pt").exists()),
    reason="YOLO weights not found locally",
)
def test_real_model_inference_on_sample_image():
    """
    Integration test running real YOLO11 inference on the local sample image.
    Verifies that real person detection succeeds on data/samples/sample_test.jpg.
    """
    img_path = Path("data/samples/sample_test.jpg")
    if not img_path.exists():
        pytest.skip("Sample test image not present.")

    detector = YOLODetector(
        model_path="weights/yolo11n.pt",
        confidence_threshold=0.25,
        classes=["person"],
    )
    img = cv2.imread(str(img_path))
    assert img is not None

    result = detector.detect(img)
    assert isinstance(result, FrameDetections)
    # The worker in the sample image should be detected
    assert len(result.detections) >= 1
    person_det = result.detections[0]
    assert person_det.class_name == "person"
    assert person_det.confidence >= 0.50
