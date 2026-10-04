import logging
from typing import Optional, Tuple, Union
from backend.schemas.detection import BoundingBox, FrameDetections

logger = logging.getLogger("intelliwatch.coordinates")


def compute_box_iou(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """Computes Intersection-over-Union (IoU) between two bounding boxes."""
    x_left = max(box_a.x1, box_b.x1)
    y_top = max(box_a.y1, box_b.y1)
    x_right = min(box_a.x2, box_b.x2)
    y_bottom = min(box_a.y2, box_b.y2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
    area_b = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)
    union_area = area_a + area_b - intersection_area

    if union_area <= 0.0:
        return 0.0

    return float(intersection_area / union_area)


def compute_box_iomin(box_a: BoundingBox, box_b: BoundingBox) -> float:
    """
    Computes Intersection-over-Minimum (IoMin) / Containment ratio between two bounding boxes.
    Measures the fraction of the smaller box that is enclosed within the larger box:
        IoMin(A, B) = area(A ∩ B) / min(area(A), area(B))
    Returns 1.0 if one box is completely contained inside the other.
    """
    x_left = max(box_a.x1, box_b.x1)
    y_top = max(box_a.y1, box_b.y1)
    x_right = min(box_a.x2, box_b.x2)
    y_bottom = min(box_a.y2, box_b.y2)

    if x_right <= x_left or y_bottom <= y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    area_a = max(0.0, box_a.x2 - box_a.x1) * max(0.0, box_a.y2 - box_a.y1)
    area_b = max(0.0, box_b.x2 - box_b.x1) * max(0.0, box_b.y2 - box_b.y1)
    min_area = min(area_a, area_b)

    if min_area <= 0.0:
        return 0.0

    return float(intersection_area / min_area)


def unpad_and_rescale_bbox(
    bbox: Union[BoundingBox, Tuple[float, float, float, float]],
    scale_factor: Optional[float],
    pad_offset: Optional[Tuple[int, int]],
    original_shape: Optional[Tuple[int, int]],
) -> BoundingBox:
    """
    Transforms a bounding box from preprocessed letterbox coordinates back
    to the original video/CCTV frame coordinates.

    Letterbox Transformation Inversion:
      1. Undo padding: subtract pad_left from X coordinates, subtract pad_top from Y coordinates.
      2. Undo scaling: divide by scale_factor.
      3. Boundary clamping: clamp coordinates to [0, orig_width] and [0, orig_height].

    Args:
        bbox: BoundingBox object or tuple of (x1, y1, x2, y2).
        scale_factor: Scale factor used during resizing (new_dim / old_dim).
        pad_offset: Tuple of (pad_top, pad_left) in pixels.
        original_shape: Tuple of (orig_height, orig_width) of the source frame.

    Returns:
        BoundingBox in original video pixel coordinates.
    """
    if isinstance(bbox, BoundingBox):
        x1, y1, x2, y2 = bbox.x1, bbox.y1, bbox.x2, bbox.y2
    else:
        x1, y1, x2, y2 = float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])

    # If no preprocessing scaling or padding occurred, return original with basic clamping if shape given
    if scale_factor is None or scale_factor <= 0.0:
        scale_factor = 1.0

    pad_top, pad_left = (0, 0)
    if pad_offset is not None and len(pad_offset) == 2:
        pad_top, pad_left = pad_offset

    # Step 1: Undo padding
    x1_unpad = x1 - pad_left
    y1_unpad = y1 - pad_top
    x2_unpad = x2 - pad_left
    y2_unpad = y2 - pad_top

    # Step 2: Undo scaling
    orig_x1 = x1_unpad / scale_factor
    orig_y1 = y1_unpad / scale_factor
    orig_x2 = x2_unpad / scale_factor
    orig_y2 = y2_unpad / scale_factor

    # Step 3: Clamp to original image boundaries if known
    if original_shape is not None and len(original_shape) == 2:
        orig_h, orig_w = original_shape
        orig_x1 = max(0.0, min(float(orig_w), orig_x1))
        orig_y1 = max(0.0, min(float(orig_h), orig_y1))
        orig_x2 = max(0.0, min(float(orig_w), orig_x2))
        orig_y2 = max(0.0, min(float(orig_h), orig_y2))
    else:
        orig_x1 = max(0.0, orig_x1)
        orig_y1 = max(0.0, orig_y1)
        orig_x2 = max(0.0, orig_x2)
        orig_y2 = max(0.0, orig_y2)

    # Ensure box dimensions are non-negative
    orig_x2 = max(orig_x1, orig_x2)
    orig_y2 = max(orig_y1, orig_y2)

    return BoundingBox(
        x1=round(orig_x1, 2),
        y1=round(orig_y1, 2),
        x2=round(orig_x2, 2),
        y2=round(orig_y2, 2),
    )


def map_detections_to_original(
    frame_detections: FrameDetections,
    scale_factor: Optional[float],
    pad_offset: Optional[Tuple[int, int]],
    original_shape: Optional[Tuple[int, int]],
) -> FrameDetections:
    """
    Transforms all bounding boxes in a FrameDetections object from preprocessed
    coordinates to original video coordinates.
    """
    if scale_factor is None and pad_offset is None:
        return frame_detections

    orig_h, orig_w = (None, None)
    if original_shape is not None and len(original_shape) == 2:
        orig_h, orig_w = original_shape

    mapped_detections = []
    for det in frame_detections.detections:
        mapped_box = unpad_and_rescale_bbox(
            bbox=det.bbox,
            scale_factor=scale_factor,
            pad_offset=pad_offset,
            original_shape=original_shape,
        )
        mapped_detections.append(
            det.model_copy(update={"bbox": mapped_box})
        )

    return FrameDetections(
        frame_id=frame_detections.frame_id,
        timestamp=frame_detections.timestamp,
        detections=mapped_detections,
        frame_width=orig_w or frame_detections.frame_width,
        frame_height=orig_h or frame_detections.frame_height,
    )
