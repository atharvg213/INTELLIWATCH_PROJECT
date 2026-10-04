"""
intelligence/behavior/motion.py
================================
Image-space motion analysis utilities for Step 8 Behavior Analysis.

All quantities here are image-space measurements in pixels or
pixels/second. They do NOT represent real-world physical velocity,
acceleration, or distance without separate metric calibration.

Public API
----------
- compute_heading_change(h1, h2) -> float
- compute_speed_average(observations, n) -> Optional[float]
- compute_recent_accel(observations) -> Optional[float]
- check_direction_change(observations, threshold_deg) -> bool
- detect_fall_heuristic(obs, prev_obs, ar_threshold, disp_threshold) -> bool
"""

import math
import logging
from typing import Deque, List, Optional, Tuple, TYPE_CHECKING

if TYPE_CHECKING:
    from intelligence.behavior.temporal_state import MotionObservation

logger = logging.getLogger("intelliwatch.behavior.motion")


def compute_heading_change(h1: float, h2: float) -> float:
    """
    Returns the minimum angular difference between two headings (0–360 deg).
    Result is always in [0, 180].
    """
    diff = abs(h2 - h1) % 360.0
    return min(diff, 360.0 - diff)


def compute_speed_average(
    observations: "Deque[MotionObservation]",
    n: int = 3,
) -> Optional[float]:
    """
    Returns the mean image-space speed (px/s) over the last n observations.
    Returns None if there are no observations.
    """
    recent = list(observations)[-n:] if len(observations) > 0 else []
    if not recent:
        return None
    return sum(o.speed_px_per_s for o in recent) / len(recent)


def compute_recent_accel(
    observations: "Deque[MotionObservation]",
) -> Optional[float]:
    """
    Returns the image-space acceleration magnitude (px/s^2) from the most
    recent observation. Returns None if no observations.
    """
    if not observations:
        return None
    return observations[-1].accel_px_per_s2


def check_direction_change(
    observations: "Deque[MotionObservation]",
    threshold_deg: float = 60.0,
    lookback: int = 4,
) -> bool:
    """
    Returns True if a significant direction change occurred within the last
    `lookback` observations.

    Compares headings of consecutive pairs of observations that have
    meaningful displacement (> 1 pixel) to suppress noise-induced direction
    changes from near-stationary tracks.
    """
    valid = [o for o in list(observations)[-lookback:] if o.heading_degrees is not None and o.displacement > 1.0]
    if len(valid) < 2:
        return False
    for i in range(1, len(valid)):
        change = compute_heading_change(valid[i - 1].heading_degrees, valid[i].heading_degrees)
        if change >= threshold_deg:
            return True
    return False


def detect_fall_heuristic(
    obs: "MotionObservation",
    prev_obs: Optional["MotionObservation"],
    ar_change_threshold: float = 0.5,
    disp_threshold: float = 30.0,
) -> bool:
    """
    Applies a conservative bounding-box geometry heuristic to detect
    a fall-like event.

    Heuristic: a possible fall is flagged when BOTH conditions hold:
      1. The bounding box aspect ratio (height/width) drops abruptly, AND
      2. The centroid displacement in the current observation is large.

    This is NOT medically or semantically reliable fall detection.
    It is a preliminary geometric signal labeled POSSIBLE_FALL.
    A future pose-based step is required for reliable fall detection.

    Returns True if both conditions are satisfied, False otherwise.
    """
    if prev_obs is None or obs.aspect_ratio is None or prev_obs.aspect_ratio is None:
        return False

    ar_drop = prev_obs.aspect_ratio - obs.aspect_ratio
    if ar_drop >= ar_change_threshold and obs.displacement >= disp_threshold:
        logger.debug(
            "Fall heuristic triggered: ar_drop=%.3f (threshold=%.3f), disp=%.1fpx (threshold=%.1fpx)",
            ar_drop, ar_change_threshold, obs.displacement, disp_threshold,
        )
        return True
    return False


def compute_depth_trend(
    depth_history: List[Optional[float]],
    window: int = 4,
    threshold: float = 0.05,
) -> Optional[str]:
    """
    Computes a qualitative depth trend from a recent list of relative depth values.

    Returns 'approaching', 'receding', or 'stable'.
    Returns None if insufficient valid data.

    NOTE: 'approaching' / 'receding' refer to relative depth CHANGE only.
    They do NOT imply metric distance or collision prediction.
    """
    valid = [d for d in depth_history[-window:] if d is not None]
    if len(valid) < 2:
        return None
    # Relative depth from Depth-Anything: higher value = closer to camera (inverted disparity convention)
    delta = valid[-1] - valid[0]
    if abs(delta) < threshold:
        return "stable"
    return "approaching" if delta > 0 else "receding"
