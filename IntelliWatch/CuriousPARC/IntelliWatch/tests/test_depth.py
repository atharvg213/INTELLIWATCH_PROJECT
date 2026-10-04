"""
Unit tests for Monocular Depth Estimation, DepthResult Schemas,
DepthAnythingEstimator, Object Depth Sampling, and Depth Visualizer.
"""
import sys
from pathlib import Path
import pytest
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from backend.schemas.depth import (
    DepthResult,
    DepthStatistics,
    DepthType,
    ObjectDepth,
)
from backend.schemas.tracking import FrameTracks, TrackPoint, TrackedObject
from vision.depth.depth_estimator import DepthAnythingEstimator
from vision.depth.depth_pipeline import DepthPipeline
from vision.depth.visualizer import DepthVisualizer
from vision.preprocessing.frame import FrameData


class MockDepthOutput:
    def __init__(self, depth_tensor: torch.Tensor):
        self.predicted_depth = depth_tensor


class MockProcessor:
    def __call__(self, images, return_tensors="pt"):
        return {"pixel_values": torch.zeros((1, 3, 256, 256))}


class MockDepthModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.param = torch.nn.Parameter(torch.zeros(1))

    def forward(self, **kwargs):
        # Create a deterministic synthetic depth gradient (1, 256, 256)
        y = torch.linspace(10.0, 50.0, 256).unsqueeze(1).repeat(1, 256)
        return MockDepthOutput(y.unsqueeze(0))


@pytest.fixture
def mock_depth_estimator():
    processor = MockProcessor()
    model = MockDepthModel()
    return DepthAnythingEstimator(
        model_name="mock-depth-model",
        device="cpu",
        processor=processor,
        model=model,
    )


# ---------------------------------------------------------------------------
# Test 1: Depth Schema Validation
# ---------------------------------------------------------------------------

def test_depth_schemas_validation():
    stats = DepthStatistics(
        min_depth=0.1,
        max_depth=0.9,
        mean_depth=0.5,
        median_depth=0.52,
        percentile_25=0.3,
        percentile_75=0.7,
        depth_unit="relative",
    )
    assert stats.median_depth == 0.52
    assert stats.depth_unit == "relative"

    bbox = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0)
    obj_depth = ObjectDepth(
        track_id=17,
        class_name="person",
        bbox=bbox,
        depth_stats=stats,
        contact_point=(150.0, 300.0),
        contact_depth=0.75,
        is_metric=False,
    )
    assert obj_depth.track_id == 17
    assert obj_depth.is_metric is False
    assert obj_depth.contact_depth == 0.75

    result = DepthResult(
        frame_id=1,
        timestamp=0.033,
        width=640,
        height=480,
        depth_type=DepthType.RELATIVE,
        is_metric=False,
        min_depth=0.0,
        max_depth=1.0,
        mean_depth=0.48,
        median_depth=0.50,
        processing_time_ms=12.5,
        object_depths=[obj_depth],
    )
    assert result.width == 640
    assert result.height == 480
    assert result.is_metric is False
    assert len(result.object_depths) == 1


# ---------------------------------------------------------------------------
# Test 2: Estimator Initialization & Device Configuration
# ---------------------------------------------------------------------------

def test_estimator_initialization(mock_depth_estimator):
    assert mock_depth_estimator.device_str == "cpu"
    assert mock_depth_estimator.device == torch.device("cpu")
    assert mock_depth_estimator.model is not None
    assert mock_depth_estimator.processor is not None


# ---------------------------------------------------------------------------
# Test 3: Invalid Frame Handling
# ---------------------------------------------------------------------------

def test_invalid_frame_handling(mock_depth_estimator):
    with pytest.raises(ValueError):
        mock_depth_estimator.estimate_depth(None)

    with pytest.raises(ValueError):
        mock_depth_estimator.estimate_depth(np.zeros((0, 0, 3), dtype=np.uint8))

    with pytest.raises(TypeError):
        mock_depth_estimator.estimate_depth("invalid_string_frame")


# ---------------------------------------------------------------------------
# Test 4: Output Shape & Native Dimension Preservation
# ---------------------------------------------------------------------------

def test_output_shape_matches_native_resolution(mock_depth_estimator):
    # Test non-square native resolution (e.g. 720p 1280x720)
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    depth_map, result = mock_depth_estimator.estimate_depth(frame)

    assert depth_map.shape == (720, 1280)
    assert depth_map.dtype == np.float32
    assert result.width == 1280
    assert result.height == 720
    assert result.depth_type == DepthType.RELATIVE
    assert result.is_metric is False


# ---------------------------------------------------------------------------
# Test 5: FrameData Container Input
# ---------------------------------------------------------------------------

def test_estimate_depth_with_framedata(mock_depth_estimator):
    img = np.ones((480, 640, 3), dtype=np.uint8) * 128
    fdata = FrameData(
        frame_index=42,
        timestamp=1.40,
        image=img,
        width=640,
        height=480,
    )
    depth_map, result = mock_depth_estimator.estimate_depth(fdata)

    assert depth_map.shape == (480, 640)
    assert result.frame_id == 42
    assert result.timestamp == 1.40
    assert result.processing_time_ms is not None
    assert result.processing_time_ms > 0


# ---------------------------------------------------------------------------
# Test 6: Depth Normalization to [0.0, 1.0]
# ---------------------------------------------------------------------------

def test_depth_normalization(mock_depth_estimator):
    frame = np.zeros((300, 400, 3), dtype=np.uint8)

    # With normalization
    depth_norm, res_norm = mock_depth_estimator.estimate_depth(frame, normalize=True)
    assert np.min(depth_norm) == pytest.approx(0.0, abs=1e-4)
    assert np.max(depth_norm) == pytest.approx(1.0, abs=1e-4)
    assert res_norm.min_depth == pytest.approx(0.0, abs=1e-4)
    assert res_norm.max_depth == pytest.approx(1.0, abs=1e-4)

    # Without normalization
    depth_raw, res_raw = mock_depth_estimator.estimate_depth(frame, normalize=False)
    assert np.min(depth_raw) >= 10.0
    assert np.max(depth_raw) <= 50.0


# ---------------------------------------------------------------------------
# Test 7: Object Bounding-Box Depth Sampling
# ---------------------------------------------------------------------------

def test_sample_object_depth_statistics(mock_depth_estimator):
    # Create a synthetic 100x100 depth map where values equal y coordinate / 100.0
    h, w = 100, 100
    y_coords, _ = np.mgrid[0:h, 0:w]
    depth_map = (y_coords / 100.0).astype(np.float32)

    bbox = BoundingBox(x1=20.0, y1=20.0, x2=80.0, y2=60.0)
    obj_depth = mock_depth_estimator.sample_object_depth(
        depth_map=depth_map,
        bbox=bbox,
        class_name="worker",
        track_id=1,
    )

    assert obj_depth.track_id == 1
    assert obj_depth.class_name == "worker"
    assert obj_depth.depth_stats.min_depth == pytest.approx(0.20, abs=0.02)
    assert obj_depth.depth_stats.max_depth == pytest.approx(0.59, abs=0.02)
    assert obj_depth.depth_stats.median_depth == pytest.approx(0.40, abs=0.02)
    assert obj_depth.is_metric is False


# ---------------------------------------------------------------------------
# Test 8: Contact-Point Depth Sampling (Person Feet vs Vehicle Centroid)
# ---------------------------------------------------------------------------

def test_contact_point_depth_sampling(mock_depth_estimator):
    h, w = 100, 100
    y_coords, _ = np.mgrid[0:h, 0:w]
    depth_map = (y_coords / 100.0).astype(np.float32)

    bbox = BoundingBox(x1=20.0, y1=20.0, x2=40.0, y2=80.0)

    # For person: contact point is at feet (y ~ 80.0)
    person_depth = mock_depth_estimator.sample_object_depth(
        depth_map=depth_map,
        bbox=bbox,
        class_name="person",
        track_id=1,
    )
    assert person_depth.contact_point == (30.0, 80.0)
    assert person_depth.contact_depth == pytest.approx(0.78, abs=0.05)

    # For forklift: contact point is centroid (y ~ 50.0)
    forklift_depth = mock_depth_estimator.sample_object_depth(
        depth_map=depth_map,
        bbox=bbox,
        class_name="forklift",
        track_id=2,
    )
    assert forklift_depth.contact_point == (30.0, 50.0)
    assert forklift_depth.contact_depth == pytest.approx(0.50, abs=0.05)


# ---------------------------------------------------------------------------
# Test 9: Model Lifecycle & Reuse
# ---------------------------------------------------------------------------

def test_model_lifecycle_reuse(mock_depth_estimator):
    # Verify that multiple calls succeed without reinitializing or leaking memory
    frame1 = np.zeros((100, 100, 3), dtype=np.uint8)
    frame2 = np.zeros((100, 100, 3), dtype=np.uint8)

    d1, r1 = mock_depth_estimator.estimate_depth(frame1)
    d2, r2 = mock_depth_estimator.estimate_depth(frame2)

    assert d1.shape == (100, 100)
    assert d2.shape == (100, 100)
    assert r1.depth_type == DepthType.RELATIVE
    assert r2.depth_type == DepthType.RELATIVE


# ---------------------------------------------------------------------------
# Test 10: DepthVisualizer Colorization & Overlays
# ---------------------------------------------------------------------------

def test_depth_visualizer():
    vis = DepthVisualizer()
    depth_map = np.linspace(0.0, 1.0, 10000).reshape((100, 100)).astype(np.float32)
    rgb_img = np.zeros((100, 100, 3), dtype=np.uint8)

    # Colorization
    color_map = vis.colorize_depth(depth_map, colormap="inferno")
    assert color_map.shape == (100, 100, 3)
    assert color_map.dtype == np.uint8

    # Blended Overlay
    blended = vis.overlay_depth_on_image(rgb_img, depth_map, alpha=0.5)
    assert blended.shape == (100, 100, 3)

    # Object Depths Annotation
    bbox = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=80.0)
    stats = DepthStatistics(
        min_depth=0.1, max_depth=0.8, mean_depth=0.45, median_depth=0.5,
        percentile_25=0.25, percentile_75=0.65, depth_unit="relative"
    )
    obj = ObjectDepth(
        track_id=1,
        class_name="worker",
        bbox=bbox,
        depth_stats=stats,
        contact_point=(30.0, 80.0),
        contact_depth=0.8,
        is_metric=False,
    )
    annotated = vis.draw_object_depths(rgb_img, [obj])
    assert annotated.shape == rgb_img.shape
    assert np.any(annotated > 0)

    # Side-by-side composite
    side_by_side = vis.create_side_by_side(rgb_img, depth_map, [obj], frame_id=1, timestamp=0.033)
    # 2 panes wide + top banner
    assert side_by_side.shape[1] == 200
    assert side_by_side.shape[0] == 140


# ---------------------------------------------------------------------------
# Test 11: DepthPipeline with Detections and Tracks
# ---------------------------------------------------------------------------

def test_depth_pipeline_integration(mock_depth_estimator):
    pipeline = DepthPipeline(estimator=mock_depth_estimator)
    img = np.zeros((100, 100, 3), dtype=np.uint8)

    bbox = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=80.0)
    det = DetectionResult(class_id=0, class_name="person", confidence=0.9, bbox=bbox)
    detections = FrameDetections(frame_id=1, timestamp=0.0, detections=[det])

    _, result_det = pipeline.process_frame(img, detections=detections)
    assert len(result_det.object_depths) == 1
    assert result_det.object_depths[0].class_name == "person"
    assert result_det.object_depths[0].contact_point is not None

    track = TrackedObject(
        track_id=17,
        class_id=0,
        class_name="worker",
        confidence=0.95,
        bbox=bbox,
        trajectory=[TrackPoint(frame_id=1, x=30.0, y=45.0, timestamp=0.0)],
    )
    tracks = FrameTracks(frame_id=1, timestamp=0.0, active_tracks=[track])

    _, result_tracks = pipeline.process_frame(img, tracks=tracks)
    assert len(result_tracks.object_depths) == 1
    assert result_tracks.object_depths[0].track_id == 17
