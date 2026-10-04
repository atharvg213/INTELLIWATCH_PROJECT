"""
intelligence/scene_graph/relations.py
Deterministic spatial, kinematic, and categorical relationship evaluation.
Supports image-space distance, relative depth gating, and separation trend analysis.
"""
import math
from typing import List, Optional, Tuple
from backend.schemas.scene_graph import SceneNodeType


def classify_node_type(class_name: str, class_group: Optional[str] = None) -> SceneNodeType:
    """
    Maps an object or detector class name to a canonical SceneNodeType.
    """
    if class_group:
        cg = str(class_group).upper().strip()
        if cg == "PERSON":
            return SceneNodeType.PERSON
        if cg == "VEHICLE":
            return SceneNodeType.VEHICLE
        if cg == "MACHINE":
            return SceneNodeType.MACHINE
        if cg == "PPE":
            return SceneNodeType.PPE
        if cg == "ZONE":
            return SceneNodeType.ZONE

    cl = (class_name or "").lower().strip()
    if cl in ("person", "worker", "pedestrian", "human", "operator"):
        return SceneNodeType.PERSON
    if cl in (
        "forklift", "forklift truck", "industrial vehicle", "truck",
        "car", "vehicle", "bus", "van", "agv", "automated guided vehicle", "tractor",
    ):
        return SceneNodeType.VEHICLE
    if cl in (
        "machine", "machinery", "industrial machine", "conveyor",
        "conveyor belt", "crane", "robot", "robotic arm", "press",
        "lathe", "generator", "pump",
    ):
        return SceneNodeType.MACHINE
    if cl in (
        "hardhat", "helmet", "safety vest", "vest", "gloves",
        "goggles", "mask", "boots", "no-hardhat", "no-safety vest",
        "no-helmet", "no-vest", "no-gloves", "no-goggles",
    ):
        return SceneNodeType.PPE
    if "zone" in cl or cl in ("restricted", "hazard", "monitored"):
        return SceneNodeType.ZONE
    return SceneNodeType.OBJECT


def compute_euclidean_distance(pt1: Tuple[float, float], pt2: Tuple[float, float]) -> float:
    """
    Computes 2D Euclidean distance in pixels between two coordinate points.
    """
    dx = pt1[0] - pt2[0]
    dy = pt1[1] - pt2[1]
    return math.hypot(dx, dy)


def is_depth_consistent(
    depth1: Optional[float],
    depth2: Optional[float],
    max_depth_diff: float = 0.20,
) -> bool:
    """
    Verifies whether two entities share compatible relative depth.
    If depth is unavailable for either entity, returns True (no false negative veto).
    If relative depth difference exceeds threshold, entities cannot be physically proximate.
    """
    if depth1 is None or depth2 is None:
        return True
    return abs(depth1 - depth2) <= max_depth_diff


def classify_proximity_state(
    dist_px: Optional[float],
    very_near_thresh: float = 75.0,
    near_thresh: float = 150.0,
    moderate_thresh: float = 300.0,
    far_thresh: float = 400.0,
) -> str:
    """
    Classifies image-space separation distance into discrete qualitative tiers:
      - VERY_NEAR: dist <= very_near_thresh
      - NEAR: dist <= near_thresh
      - MODERATE: dist <= moderate_thresh
      - FAR: dist >= far_thresh
      - UNKNOWN: when distance is None, negative, or invalid

    Adheres to technical honesty: represents relative uncalibrated spatial separation.
    """
    if dist_px is None or math.isnan(dist_px) or dist_px < 0:
        return "UNKNOWN"
    if dist_px <= very_near_thresh:
        return "VERY_NEAR"
    if dist_px <= near_thresh:
        return "NEAR"
    if dist_px <= moderate_thresh:
        return "MODERATE"
    return "FAR"


def compute_separation_velocity(
    distance_history: List[Tuple[float, float]],
) -> Optional[float]:
    """
    Computes net separation rate (px/s) over distance history.
    Negative = closing distance (approaching), Positive = opening distance (moving away).
    """
    if len(distance_history) < 2:
        return None
    t_start, d_start = distance_history[0]
    t_end, d_end = distance_history[-1]
    dt = t_end - t_start
    if dt <= 0.001:
        return None
    return float((d_end - d_start) / dt)


def compute_separation_trend(
    distance_history: List[Tuple[float, float]],
    min_samples: int = 3,
    min_rate_px_per_s: float = 15.0,
) -> Optional[str]:
    """
    Evaluates temporal change in spatial separation between two entities.

    Args:
        distance_history: Chronological list of (timestamp, distance_px).
        min_samples: Minimum observations required before determining a trend.
        min_rate_px_per_s: Minimum rate of separation change (px/s) to classify as
                           APPROACHING or MOVING_AWAY rather than STABLE.

    Returns:
        'APPROACHING' if separation is decreasing significantly,
        'MOVING_AWAY' if separation is increasing significantly,
        'STABLE' if separation change is below rate threshold,
        or None if insufficient history exists.
    """
    if len(distance_history) < min_samples:
        return None

    # Use first and last samples in the window to compute net separation velocity
    t_start, d_start = distance_history[0]
    t_end, d_end = distance_history[-1]
    dt = t_end - t_start

    if dt <= 0.001:
        return None

    rate_px_s = (d_end - d_start) / dt

    # Check intermediate consistency: distance should be predominantly monotonic
    if rate_px_s <= -min_rate_px_per_s:
        # Check that end distance is strictly smaller than start
        if d_end < d_start:
            return "APPROACHING"
    elif rate_px_s >= min_rate_px_per_s:
        if d_end > d_start:
            return "MOVING_AWAY"

    return "STABLE"

