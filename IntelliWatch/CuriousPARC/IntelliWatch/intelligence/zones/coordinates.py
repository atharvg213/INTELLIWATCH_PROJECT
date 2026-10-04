"""
intelligence/zones/coordinates.py
Step 16 - Coordinate System Transformations for Interactive Safety Zones.

Converts between displayed (rendered) viewport coordinates and original source frame coordinates,
accounting for:
- Aspect ratio preservation and scaling
- Letterboxing and pillarboxing (padding offsets)
- Container and viewport offsets
- Clamping and boundary checks
- Cross-resolution adaptation
"""
import math
from typing import Any, Dict, List, Optional, Tuple, Union


def display_to_original(
    display_x: float,
    display_y: float,
    display_width: float,
    display_height: float,
    original_width: float,
    original_height: float,
    object_fit: str = "contain",
    viewport_offset_x: float = 0.0,
    viewport_offset_y: float = 0.0,
    clamp: bool = True,
) -> Tuple[float, float]:
    """
    Transforms a display/viewport click coordinate back into the original image/frame coordinate space.

    Args:
        display_x: Click X coordinate relative to viewport container.
        display_y: Click Y coordinate relative to viewport container.
        display_width: Rendered width of the viewport element in pixels.
        display_height: Rendered height of the viewport element in pixels.
        original_width: Width of the underlying source frame/video in pixels.
        original_height: Height of the underlying source frame/video in pixels.
        object_fit: Scaling mode: 'contain' (default letterboxed) or 'fill' (stretched).
        viewport_offset_x: Horizontal offset of element inside container.
        viewport_offset_y: Vertical offset of element inside container.
        clamp: Whether to clamp resulting coordinates to [0, original_dimension].

    Returns:
        (orig_x, orig_y) in original frame pixels.
    """
    if display_width <= 0 or display_height <= 0:
        raise ValueError(f"Display dimensions must be positive, got ({display_width}, {display_height}).")
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"Original dimensions must be positive, got ({original_width}, {original_height}).")

    # Remove viewport offsets
    rel_x = display_x - viewport_offset_x
    rel_y = display_y - viewport_offset_y

    if object_fit == "contain":
        scale = min(display_width / original_width, display_height / original_height)
        rendered_w = original_width * scale
        rendered_h = original_height * scale
        pad_x = (display_width - rendered_w) / 2.0
        pad_y = (display_height - rendered_h) / 2.0

        orig_x = (rel_x - pad_x) / scale
        orig_y = (rel_y - pad_y) / scale
    elif object_fit == "fill":
        orig_x = rel_x * (original_width / display_width)
        orig_y = rel_y * (original_height / display_height)
    else:
        raise ValueError(f"Unsupported object_fit mode '{object_fit}'. Use 'contain' or 'fill'.")

    if clamp:
        orig_x = max(0.0, min(float(original_width), orig_x))
        orig_y = max(0.0, min(float(original_height), orig_y))

    return round(orig_x, 2), round(orig_y, 2)


def original_to_display(
    original_x: float,
    original_y: float,
    display_width: float,
    display_height: float,
    original_width: float,
    original_height: float,
    object_fit: str = "contain",
    viewport_offset_x: float = 0.0,
    viewport_offset_y: float = 0.0,
) -> Tuple[float, float]:
    """
    Transforms an original image/frame coordinate into the display viewport coordinate space
    for rendering overlays on top of the displayed image.

    Returns:
        (disp_x, disp_y) in viewport display pixels.
    """
    if display_width <= 0 or display_height <= 0:
        raise ValueError(f"Display dimensions must be positive, got ({display_width}, {display_height}).")
    if original_width <= 0 or original_height <= 0:
        raise ValueError(f"Original dimensions must be positive, got ({original_width}, {original_height}).")

    if object_fit == "contain":
        scale = min(display_width / original_width, display_height / original_height)
        rendered_w = original_width * scale
        rendered_h = original_height * scale
        pad_x = (display_width - rendered_w) / 2.0
        pad_y = (display_height - rendered_h) / 2.0

        disp_x = original_x * scale + pad_x + viewport_offset_x
        disp_y = original_y * scale + pad_y + viewport_offset_y
    elif object_fit == "fill":
        disp_x = original_x * (display_width / original_width) + viewport_offset_x
        disp_y = original_y * (display_height / original_height) + viewport_offset_y
    else:
        raise ValueError(f"Unsupported object_fit mode '{object_fit}'.")

    return round(disp_x, 2), round(disp_y, 2)


def transform_polygon_display_to_original(
    display_polygon: List[Union[List[float], Tuple[float, float], Dict[str, float]]],
    display_width: float,
    display_height: float,
    original_width: float,
    original_height: float,
    object_fit: str = "contain",
    viewport_offset_x: float = 0.0,
    viewport_offset_y: float = 0.0,
    clamp: bool = True,
) -> List[List[float]]:
    """
    Transforms an entire polygon from display coordinates to original frame coordinates.
    """
    transformed = []
    for pt in display_polygon:
        if isinstance(pt, dict):
            dx, dy = float(pt["x"]), float(pt["y"])
        else:
            dx, dy = float(pt[0]), float(pt[1])

        ox, oy = display_to_original(
            display_x=dx,
            display_y=dy,
            display_width=display_width,
            display_height=display_height,
            original_width=original_width,
            original_height=original_height,
            object_fit=object_fit,
            viewport_offset_x=viewport_offset_x,
            viewport_offset_y=viewport_offset_y,
            clamp=clamp,
        )
        transformed.append([ox, oy])
    return transformed


def transform_polygon_original_to_display(
    original_polygon: List[Union[List[float], Tuple[float, float]]],
    display_width: float,
    display_height: float,
    original_width: float,
    original_height: float,
    object_fit: str = "contain",
    viewport_offset_x: float = 0.0,
    viewport_offset_y: float = 0.0,
) -> List[Tuple[float, float]]:
    """
    Transforms an entire polygon from original frame coordinates to display viewport coordinates.
    """
    transformed = []
    for pt in original_polygon:
        ox, oy = float(pt[0]), float(pt[1])
        dx, dy = original_to_display(
            original_x=ox,
            original_y=oy,
            display_width=display_width,
            display_height=display_height,
            original_width=original_width,
            original_height=original_height,
            object_fit=object_fit,
            viewport_offset_x=viewport_offset_x,
            viewport_offset_y=viewport_offset_y,
        )
        transformed.append((dx, dy))
    return transformed


def scale_polygon_between_resolutions(
    polygon: List[List[float]],
    from_resolution: Tuple[int, int],
    to_resolution: Tuple[int, int],
) -> List[List[float]]:
    """
    Scales polygon vertices when switching media resolutions.
    """
    from_w, from_h = from_resolution
    to_w, to_h = to_resolution
    if from_w <= 0 or from_h <= 0 or to_w <= 0 or to_h <= 0:
        raise ValueError("Resolutions must contain positive width and height.")
    if (from_w, from_h) == (to_w, to_h):
        return [list(pt) for pt in polygon]

    scale_x = float(to_w) / float(from_w)
    scale_y = float(to_h) / float(from_h)
    return [[round(pt[0] * scale_x, 2), round(pt[1] * scale_y, 2)] for pt in polygon]


def is_polygon_resolution_compatible(
    polygon: List[List[float]],
    resolution: Tuple[int, int],
    tolerance: float = 1.05,
) -> bool:
    """
    Checks if polygon coordinates are roughly within bounds of a specified resolution.
    Returns False if points extend beyond tolerance limit.
    """
    w, h = resolution
    max_w = w * tolerance
    max_h = h * tolerance
    for pt in polygon:
        x, y = pt[0], pt[1]
        if x < -5.0 or x > max_w or y < -5.0 or y > max_h:
            return False
    return True
