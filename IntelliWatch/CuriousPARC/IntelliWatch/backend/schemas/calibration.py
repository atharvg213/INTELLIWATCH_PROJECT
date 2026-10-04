"""
backend/schemas/calibration.py
Schemas for Camera Ground-Plane Perspective Calibration and Approximate Real-World Spatial Reasoning.

Documentation note:
All metric coordinates, distances, and velocities produced under this model represent
"Approximate ground-plane spatial estimation" derived from planar homography.
This does NOT constitute full 3D reconstruction, metric depth estimation, or camera intrinsic recovery.
"""
from enum import Enum
import time
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator


class CalibrationStatus(str, Enum):
    """Operational lifecycle state of a camera calibration."""
    CALIBRATED = "CALIBRATED"
    UNCONFIGURED = "UNCONFIGURED"
    INVALID = "INVALID"
    DEGENERATE = "DEGENERATE"


class Point2D(BaseModel):
    """2D point in image pixel coordinates."""
    x: float = Field(..., description="Horizontal pixel coordinate (u)")
    y: float = Field(..., description="Vertical pixel coordinate (v)")


class GroundCoordinate(BaseModel):
    """2D point in approximate ground-plane metric coordinates (meters)."""
    x_m: float = Field(..., description="Ground-plane X coordinate in meters (along calibrated width)")
    y_m: float = Field(..., description="Ground-plane Y coordinate in meters (along calibrated depth)")


class CameraCalibrationConfig(BaseModel):
    """
    Complete configuration representing approximate ground-plane perspective calibration
    for a single camera feed.
    """
    camera_id: str = Field(..., description="Identifier of the camera this calibration applies to")
    calibration_enabled: bool = Field(default=True, description="Whether metric spatial estimation is active")
    points: List[Point2D] = Field(default_factory=list, description="4 ground-plane quadrilateral vertices in image space [P1, P2, P3, P4]")
    real_world_width_m: float = Field(default=5.0, gt=0.05, le=500.0, description="Real-world width between P1-P2 and P4-P3 in meters")
    real_world_depth_m: float = Field(default=5.0, gt=0.05, le=500.0, description="Real-world depth between P1-P4 and P2-P3 in meters")
    coordinate_system: str = Field(default="GROUND_PLANE_CARTESIAN_METERS", description="Spatial reference frame")
    calibration_status: CalibrationStatus = Field(default=CalibrationStatus.UNCONFIGURED, description="Current calibration health status")
    homography_matrix: Optional[List[List[float]]] = Field(default=None, description="3x3 homography matrix mapping image (u,v) to ground (X_m, Y_m)")
    inverse_homography_matrix: Optional[List[List[float]]] = Field(default=None, description="3x3 inverse matrix mapping ground (X_m, Y_m) to image (u,v)")
    reprojection_error_px: Optional[float] = Field(default=None, description="Average reprojection error across calibration vertices in pixels")
    estimated_coverage_area_m2: Optional[float] = Field(default=None, description="Area of the ground plane quadrilateral in square meters")
    notes: str = Field(
        default="Approximate ground-plane spatial estimation",
        description="Technical scope disclosure",
    )
    created_at: float = Field(default_factory=time.time, description="Creation timestamp")
    updated_at: float = Field(default_factory=time.time, description="Last update timestamp")

    @model_validator(mode="before")
    @classmethod
    def populate_points_format(cls, data: Any) -> Any:
        if isinstance(data, dict):
            pts = data.get("points")
            if isinstance(pts, list) and pts:
                norm_pts = []
                for p in pts:
                    if isinstance(p, dict):
                        norm_pts.append(Point2D(x=float(p["x"]), y=float(p["y"])))
                    elif isinstance(p, (list, tuple)) and len(p) >= 2:
                        norm_pts.append(Point2D(x=float(p[0]), y=float(p[1])))
                    elif isinstance(p, Point2D):
                        norm_pts.append(p)
                data["points"] = norm_pts
        return data


class CalibrationSaveRequest(BaseModel):
    """Payload to create or update calibration for a specific camera."""
    camera_id: Optional[str] = Field(default=None, description="Camera identifier (defaults to path parameter if omitted)")
    calibration_enabled: bool = Field(default=True, description="Enable ground-plane perspective calculations")
    points: List[Point2D] = Field(..., min_length=4, max_length=4, description="Exact 4 ordered ground-plane corners [P1, P2, P3, P4]")
    real_world_width_m: float = Field(..., gt=0.05, le=500.0, description="Known physical width in meters")
    real_world_depth_m: float = Field(..., gt=0.05, le=500.0, description="Known physical depth in meters")


class CalibrationValidateRequest(BaseModel):
    """Payload to validate a proposed 4-point calibration without saving."""
    points: List[Point2D] = Field(..., description="Proposed 4 ground-plane corners in image space")
    real_world_width_m: float = Field(..., gt=0.05, le=500.0, description="Known physical width in meters")
    real_world_depth_m: float = Field(..., gt=0.05, le=500.0, description="Known physical depth in meters")


class CalibrationValidateResponse(BaseModel):
    """Result of calibration validation inspection."""
    is_valid: bool = Field(..., description="True if 4 points form a non-degenerate convex quadrilateral")
    status: CalibrationStatus = Field(..., description="Assessed status")
    error_message: Optional[str] = Field(default=None, description="Explanation if invalid")
    homography_matrix: Optional[List[List[float]]] = Field(default=None, description="Calculated 3x3 homography matrix")
    reprojection_error_px: Optional[float] = Field(default=None, description="Reprojection residual error in pixels")
    estimated_coverage_area_m2: Optional[float] = Field(default=None, description="Ground area covered by the quadrilateral in m^2")
    sample_projections: Optional[List[Dict[str, Any]]] = Field(default=None, description="Sample vertex image-to-ground projections")
    notes: str = Field(default="Approximate ground-plane spatial estimation")
