"""
Unit tests for bounding box coordinate transformations, letterbox unpadding,
rescaling, and boundary clamping. Completely offline and deterministic.
"""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from vision.detection.coordinates import map_detections_to_original, unpad_and_rescale_bbox


def test_unpad_and_rescale_bbox_basic():
    """
    Test coordinate restoration for an image letterboxed from 1920x1080 to 640x640:
    - Scale factor = 640 / 1920 = 0.333333333
    - New unpadded = 640 x 360
    - Padding top = (640 - 360) / 2 = 140, pad_left = 0
    """
    orig_shape = (1080, 1920)
    scale_factor = 640.0 / 1920.0
    pad_offset = (140, 0)

    # In preprocessed 640x640 space:
    # A box corresponding to original [300, 150, 600, 450] is at:
    # x1 = 300 * (1/3) + 0 = 100.0
    # y1 = 150 * (1/3) + 140 = 190.0
    # x2 = 600 * (1/3) + 0 = 200.0
    # y2 = 450 * (1/3) + 140 = 290.0
    preproc_box = BoundingBox(x1=100.0, y1=190.0, x2=200.0, y2=290.0)

    restored = unpad_and_rescale_bbox(
        bbox=preproc_box,
        scale_factor=scale_factor,
        pad_offset=pad_offset,
        original_shape=orig_shape,
    )

    assert restored.x1 == pytest.approx(300.0, abs=0.5)
    assert restored.y1 == pytest.approx(150.0, abs=0.5)
    assert restored.x2 == pytest.approx(600.0, abs=0.5)
    assert restored.y2 == pytest.approx(450.0, abs=0.5)


def test_unpad_and_rescale_bbox_clamping():
    """
    Verify coordinates that fall outside the image boundaries (or inside letterbox padding)
    are strictly clamped to [0, orig_w] and [0, orig_h].
    """
    orig_shape = (720, 1280)
    scale_factor = 0.5
    pad_offset = (50, 20)

    # Box partially into top and left padding
    out_of_bounds_box = BoundingBox(x1=10.0, y1=20.0, x2=700.0, y2=450.0)

    restored = unpad_and_rescale_bbox(
        bbox=out_of_bounds_box,
        scale_factor=scale_factor,
        pad_offset=pad_offset,
        original_shape=orig_shape,
    )

    # x1_unpad = 10 - 20 = -10 -> clamped to 0.0
    assert restored.x1 == 0.0
    # y1_unpad = 20 - 50 = -30 -> clamped to 0.0
    assert restored.y1 == 0.0
    # x2_unpad = 700 - 20 = 680 / 0.5 = 1360 -> clamped to orig_w 1280.0
    assert restored.x2 == 1280.0
    # y2_unpad = 450 - 50 = 400 / 0.5 = 800 -> clamped to orig_h 720.0
    assert restored.y2 == 720.0


def test_unpad_and_rescale_bbox_no_scaling():
    """Verify passthrough when scale_factor is None or 1.0 without padding."""
    box = BoundingBox(x1=50.0, y1=60.0, x2=150.0, y2=200.0)
    restored = unpad_and_rescale_bbox(
        bbox=box,
        scale_factor=1.0,
        pad_offset=(0, 0),
        original_shape=(1080, 1920),
    )
    assert restored.x1 == 50.0
    assert restored.y1 == 60.0
    assert restored.x2 == 150.0
    assert restored.y2 == 200.0


def test_map_detections_to_original():
    """Verify batch mapping over a FrameDetections schema."""
    det1 = DetectionResult(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=300.0),
    )
    frame_det = FrameDetections(
        frame_id=1,
        timestamp=0.033,
        detections=[det1],
        frame_width=640,
        frame_height=640,
    )

    mapped = map_detections_to_original(
        frame_detections=frame_det,
        scale_factor=0.5,
        pad_offset=(0, 0),
        original_shape=(1280, 1280),
    )

    assert len(mapped.detections) == 1
    assert mapped.detections[0].bbox.x1 == 200.0
    assert mapped.detections[0].bbox.y1 == 200.0
    assert mapped.detections[0].bbox.x2 == 400.0
    assert mapped.detections[0].bbox.y2 == 600.0
    assert mapped.frame_width == 1280
    assert mapped.frame_height == 1280
