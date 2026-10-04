"""
backend/services/calibration_service.py
Camera Ground-Plane Perspective Calibration and Metric Spatial Estimation Service for IntelliWatch.

Responsibilities:
  - Manages planar homography calculations mapping image coordinates (u, v) to ground coordinates (X_m, Y_m).
  - Enforces strict geometric validation (non-collinear, convex, non-singular quadrilateral).
  - Computes reprojection error and metric distance/speed transformations.
  - Persists camera calibrations to configs/calibrations.json with thread-safe file I/O.
  - Guarantees non-breaking pixel-space fallback when calibration is absent or invalid.

Documentation Note:
All transformed metric values represent "Approximate ground-plane spatial estimation".
They do NOT provide full 3D metric reconstruction or intrinsic camera parameter recovery.
"""
import json
import logging
import math
import os
from pathlib import Path
import threading
import time
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np

from backend.schemas.calibration import (
    CalibrationSaveRequest,
    CalibrationStatus,
    CalibrationValidateRequest,
    CalibrationValidateResponse,
    CameraCalibrationConfig,
    Point2D,
)
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.calibration_service")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def check_quadrilateral_validity(points: List[Point2D]) -> Tuple[bool, str]:
    """
    Validates that exactly 4 2D points form a non-degenerate, non-self-intersecting convex quadrilateral.
    """
    if len(points) != 4:
        return False, f"Exactly 4 ground plane corners required, received {len(points)}."

    coords = [(p.x, p.y) for p in points]

    # Check minimum separation between vertices
    for i in range(4):
        for j in range(i + 1, 4):
            dx = coords[i][0] - coords[j][0]
            dy = coords[i][1] - coords[j][1]
            dist = math.hypot(dx, dy)
            if dist < 8.0:
                return False, f"Calibration points P{i+1} and P{j+1} are too close ({round(dist, 1)}px apart). Minimum separation is 8px."

    # Compute shoelace area of the quadrilateral
    shoelace_area = 0.0
    for i in range(4):
        j = (i + 1) % 4
        shoelace_area += coords[i][0] * coords[j][1] - coords[j][0] * coords[i][1]
    shoelace_area = 0.5 * shoelace_area

    if abs(shoelace_area) < 100.0:
        return False, f"Quadrilateral pixel area is degenerate ({round(abs(shoelace_area), 1)}px²). Ensure points span a visible ground region."

    # Check convexity & consistent winding order via 2D cross products
    cross_products = []
    for i in range(4):
        p_prev = coords[i]
        p_curr = coords[(i + 1) % 4]
        p_next = coords[(i + 2) % 4]
        v1 = (p_curr[0] - p_prev[0], p_curr[1] - p_prev[1])
        v2 = (p_next[0] - p_curr[0], p_next[1] - p_curr[1])
        cp = v1[0] * v2[1] - v1[1] * v2[0]
        cross_products.append(cp)

    signs = [math.copysign(1.0, cp) for cp in cross_products if abs(cp) > 1e-5]
    if len(signs) < 4 or any(s != signs[0] for s in signs):
        return False, "Quadrilateral must be strictly convex without self-intersecting or collinear edges."

    # Check collinearity of any 3 points
    for i in range(4):
        p1 = coords[i]
        p2 = coords[(i + 1) % 4]
        p3 = coords[(i + 2) % 4]
        tri_area = abs(0.5 * ((p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])))
        if tri_area < 25.0:
            return False, f"Points P{i+1}, P{((i+1)%4)+1}, P{((i+2)%4)+1} are nearly collinear (triangle area={round(tri_area, 1)}px²)."

    return True, "Valid convex quadrilateral."


class GroundPlaneCalibrator:
    """
    Mathematical engine performing planar homography transformations between image space and ground plane.
    """

    def __init__(self, config: CameraCalibrationConfig):
        self.camera_id = config.camera_id
        self.config = config
        self.enabled = config.calibration_enabled
        self.width_m = config.real_world_width_m
        self.depth_m = config.real_world_depth_m
        self.points = config.points

        self.H: Optional[np.ndarray] = None
        self.H_inv: Optional[np.ndarray] = None
        self.is_valid = False
        self.error_message: Optional[str] = None
        self.reprojection_error_px = 0.0

        if self.enabled and len(self.points) == 4:
            self._compile_homography()

    def _compile_homography(self) -> None:
        valid, msg = check_quadrilateral_validity(self.points)
        if not valid:
            self.is_valid = False
            self.error_message = msg
            logger.warning(f"Calibration for camera [{self.camera_id}] invalid: {msg}")
            return

        # Canonical ground plane coordinates in meters:
        # P1 -> (0, 0)
        # P2 -> (W, 0)
        # P3 -> (W, D)
        # P4 -> (0, D)
        src_pts = np.float32([[p.x, p.y] for p in self.points])
        dst_pts = np.float32([
            [0.0, 0.0],
            [self.width_m, 0.0],
            [self.width_m, self.depth_m],
            [0.0, self.depth_m],
        ])

        try:
            H = cv2.getPerspectiveTransform(src_pts, dst_pts)
            det = np.linalg.det(H)
            if abs(det) < 1e-12 or np.isnan(det) or np.isinf(det):
                self.is_valid = False
                self.error_message = "Homography matrix is numerically singular or degenerate."
                return

            H_inv = cv2.getPerspectiveTransform(dst_pts, src_pts)

            # Compute reprojection residual error in pixels
            reproj_errs = []
            for i in range(4):
                ground_pt = dst_pts[i]
                img_reproj = self._transform_point(ground_pt[0], ground_pt[1], H_inv)
                if img_reproj is not None:
                    orig = src_pts[i]
                    err = math.hypot(img_reproj[0] - orig[0], img_reproj[1] - orig[1])
                    reproj_errs.append(err)

            self.H = H
            self.H_inv = H_inv
            self.reprojection_error_px = round(float(np.mean(reproj_errs)), 3) if reproj_errs else 0.0
            self.is_valid = True
            self.error_message = None

            # Update config with computed matrices
            self.config.homography_matrix = H.tolist()
            self.config.inverse_homography_matrix = H_inv.tolist()
            self.config.reprojection_error_px = self.reprojection_error_px
            self.config.estimated_coverage_area_m2 = round(self.width_m * self.depth_m, 2)
            self.config.calibration_status = CalibrationStatus.CALIBRATED
        except Exception as e:
            self.is_valid = False
            self.error_message = f"Failed to compute perspective transformation: {e}"
            logger.error(f"Error computing homography for camera [{self.camera_id}]: {e}")

    @staticmethod
    def _transform_point(x: float, y: float, matrix: np.ndarray) -> Optional[Tuple[float, float]]:
        vec = np.array([x, y, 1.0], dtype=np.float64)
        res = matrix.dot(vec)
        if not np.isfinite(res).all():
            return None
        w = res[2]
        if abs(w) < 1e-8:
            return None
        out_x = float(res[0] / w)
        out_y = float(res[1] / w)
        if not math.isfinite(out_x) or not math.isfinite(out_y):
            return None
        return out_x, out_y

    def image_to_ground(self, u: float, v: float) -> Optional[Tuple[float, float]]:
        """
        Transforms image pixel coordinate (u, v) into approximate ground plane metric coordinate (X_m, Y_m).
        Returns None if calibration is inactive or point is behind vanishing horizon.
        """
        if not self.is_valid or self.H is None:
            return None
        polygon = np.asarray([[p.x, p.y] for p in self.points], dtype=np.float32)
        if cv2.pointPolygonTest(polygon, (float(u), float(v)), False) < 0:
            return None
        pt = self._transform_point(u, v, self.H)
        if pt is None:
            return None
        return round(pt[0], 3), round(pt[1], 3)

    def ground_to_image(self, x_m: float, y_m: float) -> Optional[Tuple[float, float]]:
        """
        Transforms ground plane coordinate (X_m, Y_m) back to image pixel coordinate (u, v).
        """
        if not self.is_valid or self.H_inv is None:
            return None
        pt = self._transform_point(x_m, y_m, self.H_inv)
        if pt is None:
            return None
        return round(pt[0], 1), round(pt[1], 1)

    def compute_ground_distance(
        self,
        pt1_image: Tuple[float, float],
        pt2_image: Tuple[float, float],
    ) -> Optional[float]:
        """
        Computes Euclidean ground distance in meters between two image points.
        Returns None if calibration is inactive or points cannot be projected.
        """
        g1 = self.image_to_ground(pt1_image[0], pt1_image[1])
        g2 = self.image_to_ground(pt2_image[0], pt2_image[1])
        if g1 is None or g2 is None:
            return None
        return round(math.hypot(g2[0] - g1[0], g2[1] - g1[1]), 3)

    def compute_ground_speed(
        self,
        u: float,
        v: float,
        speed_px_per_s: Optional[float],
        velocity_px_s: Optional[Tuple[float, float]] = None,
    ) -> Optional[float]:
        """
        Computes approximate ground plane speed in m/s for an entity at image contact point (u, v).
        Returns None if calibration is inactive or speed cannot be determined.
        """
        if not self.is_valid or self.H is None or speed_px_per_s is None:
            return None
        try:
            speed_px_per_s = float(speed_px_per_s)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(speed_px_per_s):
            return None
        if speed_px_per_s < 0.0:
            return None
        if speed_px_per_s == 0.0:
            return 0.0

        # A scalar pixel speed has no direction under a perspective transform.
        # Only convert when tracking provides a usable image-space velocity vector.
        if velocity_px_s is None:
            return None
        try:
            vx, vy = float(velocity_px_s[0]), float(velocity_px_s[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not math.isfinite(vx) or not math.isfinite(vy):
            return None
        vector_magnitude = math.hypot(vx, vy)
        if vector_magnitude <= 1e-8:
            return None

        # Tracker vectors carry direction; scale their unit vector by the measured
        # pixel/second magnitude so the one-second displacement has the right units.
        vx_px_s = vx / vector_magnitude * speed_px_per_s
        vy_px_s = vy / vector_magnitude * speed_px_per_s
        p0 = self.image_to_ground(u, v)
        p1 = self.image_to_ground(u + vx_px_s, v + vy_px_s)
        if p0 is not None and p1 is not None:
            speed_m_s = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
            return round(speed_m_s, 2) if math.isfinite(speed_m_s) else None

        return None

    def compute_ground_closing_rate(
        self,
        point_a_image: Tuple[float, float],
        velocity_a_px_s: Tuple[float, float],
        point_b_image: Tuple[float, float],
        velocity_b_px_s: Tuple[float, float],
    ) -> Optional[float]:
        """Estimate pair closing rate from directed pixel motion over one second."""
        if not self.is_valid or self.H is None:
            return None
        try:
            ax, ay = float(point_a_image[0]), float(point_a_image[1])
            bx, by = float(point_b_image[0]), float(point_b_image[1])
            avx, avy = float(velocity_a_px_s[0]), float(velocity_a_px_s[1])
            bvx, bvy = float(velocity_b_px_s[0]), float(velocity_b_px_s[1])
        except (TypeError, ValueError, IndexError):
            return None
        if not all(math.isfinite(value) for value in (ax, ay, bx, by, avx, avy, bvx, bvy)):
            return None

        distance_now = self.compute_ground_distance((ax, ay), (bx, by))
        distance_after_one_second = self.compute_ground_distance(
            (ax + avx, ay + avy),
            (bx + bvx, by + bvy),
        )
        if distance_now is None or distance_after_one_second is None:
            return None
        closing_rate = distance_now - distance_after_one_second
        return round(closing_rate, 2) if math.isfinite(closing_rate) else None


class CalibrationService:
    """
    Central repository for camera ground-plane perspective calibrations.
    Coordinates persistence to configs/calibrations.json and real-time calibrator caching.
    """

    def __init__(self, config_path: Optional[Union[str, Path]] = None):
        self._lock = threading.RLock()
        settings = get_settings()
        if config_path is None:
            self.config_path = PROJECT_ROOT / getattr(settings, "CALIBRATION_CONFIG_PATH", "configs/calibrations.json")
        else:
            self.config_path = Path(config_path)

        self._calibrations: Dict[str, CameraCalibrationConfig] = {}
        self._calibrators: Dict[str, GroundPlaneCalibrator] = {}
        self._load()

    def _load(self) -> None:
        with self._lock:
            if not self.config_path.exists():
                logger.info(f"Calibration config file {self.config_path} not found. Starting with empty store.")
                return

            try:
                with open(self.config_path, "r", encoding="utf-8") as f:
                    data = json.load(f)

                if isinstance(data, list):
                    for item in data:
                        cfg = CameraCalibrationConfig.model_validate(item)
                        self._calibrations[cfg.camera_id] = cfg
                        self._calibrators[cfg.camera_id] = GroundPlaneCalibrator(cfg)
                elif isinstance(data, dict):
                    for cid, item in data.items():
                        if "camera_id" not in item:
                            item["camera_id"] = cid
                        cfg = CameraCalibrationConfig.model_validate(item)
                        self._calibrations[cid] = cfg
                        self._calibrators[cid] = GroundPlaneCalibrator(cfg)

                logger.info(f"Loaded {len(self._calibrations)} camera calibrations from {self.config_path}")
            except Exception as e:
                logger.error(f"Failed to load camera calibrations from {self.config_path}: {e}")

    def _persist(self) -> None:
        with self._lock:
            try:
                self.config_path.parent.mkdir(parents=True, exist_ok=True)
                serialized = [cfg.model_dump() for cfg in self._calibrations.values()]
                tmp_path = self.config_path.with_suffix(".tmp")
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(serialized, f, indent=2)
                os.replace(tmp_path, self.config_path)
            except Exception as e:
                logger.error(f"Failed to persist camera calibrations to {self.config_path}: {e}")

    def get_calibration(self, camera_id: str) -> Optional[CameraCalibrationConfig]:
        """Returns the calibration configuration for a specific camera if registered."""
        with self._lock:
            return self._calibrations.get(camera_id)

    def get_calibrator(self, camera_id: str) -> Optional[GroundPlaneCalibrator]:
        """Returns the compiled GroundPlaneCalibrator for a specific camera."""
        with self._lock:
            cal = self._calibrators.get(camera_id)
            if cal and cal.is_valid and cal.enabled:
                return cal
            return None

    def list_calibrations(self) -> List[CameraCalibrationConfig]:
        """Returns all registered camera calibration configurations."""
        with self._lock:
            return list(self._calibrations.values())

    def save_calibration(self, camera_id: str, req: CalibrationSaveRequest) -> CameraCalibrationConfig:
        """
        Creates or updates a camera calibration, compiles the homography, and persists to disk.
        """
        with self._lock:
            valid, msg = check_quadrilateral_validity(req.points)
            if not valid:
                raise ValueError(msg)

            config = CameraCalibrationConfig(
                camera_id=camera_id,
                calibration_enabled=req.calibration_enabled,
                points=req.points,
                real_world_width_m=req.real_world_width_m,
                real_world_depth_m=req.real_world_depth_m,
                updated_at=time.time(),
            )
            calibrator = GroundPlaneCalibrator(config)
            if not calibrator.is_valid:
                raise ValueError(calibrator.error_message or "Calibration calculation resulted in degenerate geometry.")

            self._calibrations[camera_id] = calibrator.config
            self._calibrators[camera_id] = calibrator
            self._persist()

            logger.info(f"Saved calibration for camera [{camera_id}] (W={req.real_world_width_m}m, D={req.real_world_depth_m}m, Err={calibrator.reprojection_error_px}px)")
            return calibrator.config

    def delete_calibration(self, camera_id: str) -> bool:
        """Removes the calibration configuration for a camera and resets to pixel space."""
        with self._lock:
            if camera_id in self._calibrations:
                del self._calibrations[camera_id]
                self._calibrators.pop(camera_id, None)
                self._persist()
                logger.info(f"Deleted calibration for camera [{camera_id}]. Reverted to pixel space.")
                return True
            return False

    def validate_calibration(self, req: CalibrationValidateRequest) -> CalibrationValidateResponse:
        """
        Validates a proposed quadrilateral calibration without saving it.
        """
        valid, msg = check_quadrilateral_validity(req.points)
        if not valid:
            return CalibrationValidateResponse(
                is_valid=False,
                status=CalibrationStatus.INVALID,
                error_message=msg,
            )

        test_cfg = CameraCalibrationConfig(
            camera_id="preview",
            calibration_enabled=True,
            points=req.points,
            real_world_width_m=req.real_world_width_m,
            real_world_depth_m=req.real_world_depth_m,
        )
        calibrator = GroundPlaneCalibrator(test_cfg)

        if not calibrator.is_valid:
            return CalibrationValidateResponse(
                is_valid=False,
                status=CalibrationStatus.DEGENERATE,
                error_message=calibrator.error_message or "Homography calculation failed.",
            )

        # Generate sample vertex projections
        projections = []
        for i, p in enumerate(req.points):
            g = calibrator.image_to_ground(p.x, p.y)
            projections.append({
                "vertex": f"P{i+1}",
                "image_point": [p.x, p.y],
                "ground_point_m": list(g) if g else None,
            })

        return CalibrationValidateResponse(
            is_valid=True,
            status=CalibrationStatus.CALIBRATED,
            homography_matrix=calibrator.config.homography_matrix,
            reprojection_error_px=calibrator.reprojection_error_px,
            estimated_coverage_area_m2=calibrator.config.estimated_coverage_area_m2,
            sample_projections=projections,
        )


# Global singleton instance
_calibration_service_instance: Optional[CalibrationService] = None
_instance_lock = threading.Lock()


def get_calibration_service() -> CalibrationService:
    """Returns the process-wide CalibrationService singleton."""
    global _calibration_service_instance
    with _instance_lock:
        if _calibration_service_instance is None:
            _calibration_service_instance = CalibrationService()
        return _calibration_service_instance
