"""
intelligence/behavior/temporal_state.py
========================================
Per-track temporal state management for Step 8 Behavior Analysis.

This module maintains a bounded sliding window of centroid observations
for each active track. It provides the raw motion snapshots consumed by
the motion estimator and behavior classifiers.

Design constraints
------------------
- No external model inference: purely algorithmic over existing TrackPoint data.
- Memory bounded: history older than BEHAVIOR_HISTORY_SECONDS is pruned each update.
- Handles track gaps and reappearances without crashing.
- CPU-only; no GPU dependency.
"""

import collections
import logging
import math
from dataclasses import dataclass, field
from typing import Deque, Dict, List, Optional, Tuple

logger = logging.getLogger("intelliwatch.behavior.temporal_state")


@dataclass
class MotionObservation:
    """
    A single motion observation derived from two consecutive centroid positions.
    All values are image-space (pixels or pixels/second).
    """
    timestamp: float           # Wall-clock / stream timestamp of the newer observation
    frame_id: int              # Frame index of the newer observation
    cx: float                  # Centroid x (pixels)
    cy: float                  # Centroid y (pixels)
    bbox_width: float = 0.0    # Bounding box width (pixels) at this observation
    bbox_height: float = 0.0   # Bounding box height (pixels) at this observation

    # Image-space displacement since previous observation
    dx: float = 0.0            # pixels
    dy: float = 0.0            # pixels
    displacement: float = 0.0  # Euclidean displacement (pixels)

    # Image-space velocity (displacement / elapsed time)
    vx: float = 0.0            # pixels/second
    vy: float = 0.0            # pixels/second
    speed_px_per_s: float = 0.0   # magnitude (pixels/second)

    # Image-space acceleration (delta velocity / elapsed time)
    ax: float = 0.0            # pixels/second^2
    ay: float = 0.0            # pixels/second^2
    accel_px_per_s2: float = 0.0  # magnitude (pixels/second^2)

    # Heading angle in degrees (0=right/East, 90=down/South, 180=left/West, 270=up/North)
    heading_degrees: Optional[float] = None

    # Elapsed time since previous observation (seconds)
    dt: float = 0.0

    # Aspect ratio (height / width) at this observation for fall heuristic
    aspect_ratio: Optional[float] = None


@dataclass
class TrackTemporalState:
    """
    Sliding-window temporal state for one track.

    Stores the bounded observation history and derived continuous counters
    needed for behavior classification.
    """
    track_id: int
    history_seconds: float = 5.0  # Maximum age of retained observations (seconds)

    # Bounded deque of centroid positions with timestamps
    # Each element: (timestamp, frame_id, cx, cy, bbox_width, bbox_height)
    _centroid_history: Deque[Tuple[float, int, float, float, float, float]] = field(
        default_factory=lambda: collections.deque()
    )

    # Derived motion observation history (one per consecutive pair)
    motion_history: Deque[MotionObservation] = field(
        default_factory=lambda: collections.deque()
    )

    # Continuous counters
    stationary_duration_s: float = 0.0   # Seconds continuously below stationary threshold
    last_seen_timestamp: Optional[float] = None
    first_seen_timestamp: Optional[float] = None

    # Previous velocity components (needed for acceleration)
    _prev_vx: float = 0.0
    _prev_vy: float = 0.0
    _prev_speed: float = 0.0

    def __post_init__(self):
        self._centroid_history = collections.deque()
        self.motion_history = collections.deque()

    def add_observation(
        self,
        timestamp: float,
        frame_id: int,
        cx: float,
        cy: float,
        bbox_x1: float = 0.0,
        bbox_y1: float = 0.0,
        bbox_x2: float = 0.0,
        bbox_y2: float = 0.0,
    ) -> Optional[MotionObservation]:
        """
        Record a new centroid observation and compute motion quantities.

        Returns a MotionObservation if there was a previous point to
        compute motion against, else None (first observation).
        """
        bw = max(0.0, bbox_x2 - bbox_x1)
        bh = max(0.0, bbox_y2 - bbox_y1)
        aspect = (bh / bw) if bw > 1e-6 else None

        if self.first_seen_timestamp is None:
            self.first_seen_timestamp = timestamp
        self.last_seen_timestamp = timestamp

        # Compute motion if we have a prior point
        obs: Optional[MotionObservation] = None
        if len(self._centroid_history) > 0:
            prev_ts, prev_fid, prev_cx, prev_cy, prev_bw, prev_bh = self._centroid_history[-1]

            dt = timestamp - prev_ts
            if dt < 1e-9:
                # Zero or negative time delta: use fallback (avoid division by zero)
                dt_safe = 1e-6
                logger.debug(
                    "track %d: zero/negative dt (%.6f) between frames %d->%d; "
                    "clamping to 1e-6 for velocity computation",
                    self.track_id, dt, prev_fid, frame_id,
                )
            else:
                dt_safe = dt

            dx = cx - prev_cx
            dy = cy - prev_cy
            displacement = math.hypot(dx, dy)

            vx = dx / dt_safe
            vy = dy / dt_safe
            speed = math.hypot(vx, vy)

            ax = (vx - self._prev_vx) / dt_safe
            ay = (vy - self._prev_vy) / dt_safe
            accel = math.hypot(ax, ay)

            heading: Optional[float] = None
            if displacement > 1e-3:
                # atan2: angle from positive x-axis; flip y because image y increases downward
                raw_angle = math.degrees(math.atan2(dy, dx))
                heading = raw_angle % 360.0

            obs = MotionObservation(
                timestamp=timestamp,
                frame_id=frame_id,
                cx=cx,
                cy=cy,
                bbox_width=bw,
                bbox_height=bh,
                dx=dx,
                dy=dy,
                displacement=displacement,
                vx=vx,
                vy=vy,
                speed_px_per_s=speed,
                ax=ax,
                ay=ay,
                accel_px_per_s2=accel,
                heading_degrees=heading,
                dt=dt,
                aspect_ratio=aspect,
            )

            self.motion_history.append(obs)
            self._prev_vx = vx
            self._prev_vy = vy
            self._prev_speed = speed
        else:
            # First observation: only store centroid, no motion yet
            self._prev_vx = 0.0
            self._prev_vy = 0.0
            self._prev_speed = 0.0

        # Store raw centroid
        self._centroid_history.append((timestamp, frame_id, cx, cy, bw, bh))

        # Prune history beyond the configured time window
        self._prune(timestamp)

        return obs

    def _prune(self, current_timestamp: float) -> None:
        """Remove observations older than history_seconds from both deques."""
        cutoff = current_timestamp - self.history_seconds
        while self._centroid_history and self._centroid_history[0][0] < cutoff:
            self._centroid_history.popleft()
        while self.motion_history and self.motion_history[0].timestamp < cutoff:
            self.motion_history.popleft()

    def latest_observation(self) -> Optional[MotionObservation]:
        """Returns the most recent MotionObservation, or None if fewer than 2 points."""
        if not self.motion_history:
            return None
        return self.motion_history[-1]

    def observation_count(self) -> int:
        """Number of raw centroid points currently in the window."""
        return len(self._centroid_history)

    def motion_count(self) -> int:
        """Number of motion observations (centroid pairs) in the window."""
        return len(self.motion_history)

    @property
    def centroid_history(self) -> List[Tuple[float, int, float, float, float, float]]:
        """Read-only snapshot of centroid history."""
        return list(self._centroid_history)


class TrackStateRegistry:
    """
    Central registry that owns a TrackTemporalState for every active track.

    Handles creation, lookup, and garbage collection of stale tracks.
    """

    def __init__(self, history_seconds: float = 5.0):
        self.history_seconds = history_seconds
        self._states: Dict[int, TrackTemporalState] = {}

    def get_or_create(self, track_id: int) -> TrackTemporalState:
        """Returns existing state for track_id, creating a new one if absent."""
        if track_id not in self._states:
            self._states[track_id] = TrackTemporalState(
                track_id=track_id, history_seconds=self.history_seconds
            )
            logger.debug("TrackStateRegistry: created new state for track %d", track_id)
        return self._states[track_id]

    def get(self, track_id: int) -> Optional[TrackTemporalState]:
        """Returns state for track_id, or None if not registered."""
        return self._states.get(track_id)

    def remove(self, track_id: int) -> None:
        """Explicitly removes a track (e.g. on REMOVED lifecycle state)."""
        self._states.pop(track_id, None)
        logger.debug("TrackStateRegistry: removed state for track %d", track_id)

    def purge_stale(self, current_timestamp: float, max_age_seconds: float = 30.0) -> List[int]:
        """
        Removes tracks not seen for more than max_age_seconds.
        Returns list of purged track IDs.
        """
        purged = []
        for tid, state in list(self._states.items()):
            if (
                state.last_seen_timestamp is not None
                and (current_timestamp - state.last_seen_timestamp) > max_age_seconds
            ):
                del self._states[tid]
                purged.append(tid)
        if purged:
            logger.debug("TrackStateRegistry: purged stale tracks: %s", purged)
        return purged

    @property
    def active_track_ids(self) -> List[int]:
        return list(self._states.keys())

    def __len__(self) -> int:
        return len(self._states)
