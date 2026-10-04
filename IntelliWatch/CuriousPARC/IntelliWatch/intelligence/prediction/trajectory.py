"""
intelligence/prediction/trajectory.py
Step 11 - Short-Horizon Trajectory Projection and Hazard-Direction Analysis.
Uses existing image-space trajectory history from ByteTrack (TrackPoint sequence).
Strictly adheres to technical honesty: image-space velocity is NOT real-world speed (m/s);
projected trajectory is a short-horizon kinematic extrapolation, NOT guaranteed future prediction.
"""
from typing import Any, Dict, List, Optional, Tuple
import math
import logging
import cv2
import numpy as np

from backend.schemas.tracking import TrackedObject, TrackPoint
from backend.schemas.zones import RestrictedZone
from backend.schemas.prediction import ProjectedTrajectory
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.prediction.trajectory")


class TrajectoryAnalyzer:
    """
    Analyzes historical track positions to compute short-horizon image-space projections
    and evaluates spatial convergence toward restricted zones, vehicles, and machinery.
    """

    def __init__(self, settings_override: Optional[Dict[str, Any]] = None):
        self._settings = get_settings()
        self._overrides = settings_override or {}

    def _get_setting(self, key: str, default: Any) -> Any:
        return self._overrides.get(key, getattr(self._settings, key, default))

    def project_trajectory(
        self,
        track: TrackedObject,
        horizon_frames: Optional[int] = None,
        fps: float = 30.0,
    ) -> Optional[ProjectedTrajectory]:
        """
        Calculates short-horizon projected position for a single tracked entity using
        its recent image-space centroid points.
        Returns None if trajectory history is insufficient.
        """
        min_history = int(self._get_setting("PREDICTION_MIN_TRACK_HISTORY", 3))
        history_len = int(self._get_setting("PREDICTION_HISTORY_FRAMES", 5))
        horiz_frames = (
            horizon_frames
            if horizon_frames is not None
            else int(self._get_setting("PREDICTION_HORIZON_FRAMES", 15))
        )

        history = track.trajectory
        if not history or len(history) < min_history:
            return None

        recent_points = history[-history_len:]
        if len(recent_points) < 2:
            return None

        first_pt = recent_points[0]
        last_pt = recent_points[-1]

        dt = last_pt.timestamp - first_pt.timestamp
        if dt <= 1e-4:
            # Fall back to frame difference assuming uniform fps
            df = max(1, last_pt.frame_id - first_pt.frame_id)
            dt = df / max(1.0, fps)

        dx = last_pt.cx - first_pt.cx
        dy = last_pt.cy - first_pt.cy
        disp = math.hypot(dx, dy)

        horizon_sec = horiz_frames / max(1.0, fps)

        # Estimate average image-space velocity
        vx = dx / dt
        vy = dy / dt
        speed_px_per_s = disp / dt

        cur_x = float(last_pt.cx)
        cur_y = float(last_pt.cy)

        # If displacement is tiny, object is stationary
        if disp < 2.0:
            proj_x = cur_x
            proj_y = cur_y
            heading = None
        else:
            proj_x = cur_x + vx * horizon_sec
            proj_y = cur_y + vy * horizon_sec
            # Calculate heading in degrees (0 = right, 90 = down in image coords)
            heading = (math.degrees(math.atan2(dy, dx)) + 360.0) % 360.0

        return ProjectedTrajectory(
            track_id=track.track_id,
            current_position=(round(cur_x, 2), round(cur_y, 2)),
            projected_position=(round(proj_x, 2), round(proj_y, 2)),
            horizon_frames=horiz_frames,
            horizon_seconds=round(horizon_sec, 3),
            estimated_speed_px_per_s=round(speed_px_per_s, 2),
            heading_degrees=round(heading, 1) if heading is not None else None,
            history_points_used=len(recent_points),
            timestamp=last_pt.timestamp,
        )

    def check_trajectory_toward_zone(
        self,
        projection: ProjectedTrajectory,
        zone: RestrictedZone,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determines if a projected trajectory heads toward or enters a restricted zone polygon.
        Returns:
            Tuple of (is_toward_zone, geometric_evidence)
        """
        if len(zone.polygon) < 3:
            return False, {}

        contour = np.array(zone.polygon, dtype=np.float32)

        cur_pt = (float(projection.current_position[0]), float(projection.current_position[1]))
        proj_pt = (float(projection.projected_position[0]), float(projection.projected_position[1]))

        # cv2.pointPolygonTest: positive = inside, zero = on edge, negative = outside distance
        dist_cur = cv2.pointPolygonTest(contour, cur_pt, measureDist=True)
        dist_proj = cv2.pointPolygonTest(contour, proj_pt, measureDist=True)

        # If current point is already strictly inside the zone, entry has already occurred
        if dist_cur >= 0:
            return False, {"status": "already_inside"}

        zone_margin = float(self._get_setting("PREDICTION_ZONE_MARGIN", 30.0))

        # Check if projected point is either inside the zone or significantly closer than current point
        will_enter = dist_proj >= 0.0
        closing_distance = dist_proj - dist_cur  # Note: negative dists become less negative as you get closer

        is_heading_toward = will_enter or (closing_distance > zone_margin)

        closing_speed_px_per_s = None
        time_to_hazard_s = None
        if closing_distance > 0 and projection.horizon_seconds > 0:
            closing_speed_px_per_s = round(closing_distance / projection.horizon_seconds, 1)
            if closing_speed_px_per_s > 2.0:
                time_to_hazard_s = round(abs(dist_cur) / closing_speed_px_per_s, 2)

        evidence = {
            "track_id": projection.track_id,
            "zone_id": zone.zone_id,
            "zone_name": zone.name,
            "current_distance_px": round(abs(dist_cur), 1),
            "projected_distance_px": round(abs(dist_proj), 1) if dist_proj < 0 else 0.0,
            "will_intersect_boundary": will_enter,
            "closing_distance_px": round(closing_distance, 1),
            "closing_speed_px_per_s": closing_speed_px_per_s,
            "time_to_hazard_seconds": time_to_hazard_s,
            "horizon_frames": projection.horizon_frames,
            "horizon_seconds": projection.horizon_seconds,
        }

        return is_heading_toward, evidence

    def check_trajectory_toward_point(
        self,
        projection: ProjectedTrajectory,
        target_point: Tuple[float, float],
        min_approach_delta_px: float = 20.0,
        max_target_distance_px: float = 350.0,
    ) -> Tuple[bool, Dict[str, Any]]:
        """
        Determines if a projected trajectory is converging toward a target coordinate point
        (such as a vehicle or machine location).
        """
        cur_pt = projection.current_position
        proj_pt = projection.projected_position

        d_cur = math.hypot(cur_pt[0] - target_point[0], cur_pt[1] - target_point[1])
        d_proj = math.hypot(proj_pt[0] - target_point[0], proj_pt[1] - target_point[1])

        if d_cur > max_target_distance_px:
            return False, {}

        approach_delta = d_cur - d_proj
        is_converging = approach_delta >= min_approach_delta_px

        closing_speed_px_per_s = None
        time_to_hazard_s = None
        if approach_delta > 0 and projection.horizon_seconds > 0:
            closing_speed_px_per_s = round(approach_delta / projection.horizon_seconds, 1)
            if closing_speed_px_per_s > 2.0:
                dist_to_hazard = max(0.0, d_cur - 60.0)  # 60px contact threshold
                time_to_hazard_s = round(dist_to_hazard / closing_speed_px_per_s, 2)

        evidence = {
            "current_distance_px": round(d_cur, 1),
            "projected_distance_px": round(d_proj, 1),
            "approach_delta_px": round(approach_delta, 1),
            "closing_speed_px_per_s": closing_speed_px_per_s,
            "time_to_hazard_seconds": time_to_hazard_s,
            "horizon_frames": projection.horizon_frames,
            "horizon_seconds": projection.horizon_seconds,
        }

        return is_converging, evidence
