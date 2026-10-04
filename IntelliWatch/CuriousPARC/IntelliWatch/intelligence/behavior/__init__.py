"""
intelligence/behavior
=====================
Step 8 — Behavior and Temporal Analysis package.

Public API:
    BehaviorEngine      — main frame-by-frame processor
    TrackStateRegistry  — per-track temporal state store
    TrackTemporalState  — individual track state
    MotionObservation   — per-frame motion snapshot

All motion values are image-space (pixels, px/s) — NOT real-world physical units.
"""

from intelligence.behavior.behavior_engine import BehaviorEngine
from intelligence.behavior.temporal_state import (
    MotionObservation,
    TrackStateRegistry,
    TrackTemporalState,
)
from intelligence.behavior.motion import (
    compute_heading_change,
    compute_speed_average,
    compute_recent_accel,
    check_direction_change,
    detect_fall_heuristic,
    compute_depth_trend,
)

__all__ = [
    "BehaviorEngine",
    "TrackStateRegistry",
    "TrackTemporalState",
    "MotionObservation",
    "compute_heading_change",
    "compute_speed_average",
    "compute_recent_accel",
    "check_direction_change",
    "detect_fall_heuristic",
    "compute_depth_trend",
]
