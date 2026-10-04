from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from backend.schemas.detection import BoundingBox


class DepthType(str, Enum):
    RELATIVE = "relative"
    METRIC = "metric"


class DepthStatistics(BaseModel):
    """
    Statistical summary of depth values within a 2D bounding box or subregion.
    """
    min_depth: float = Field(..., description="Minimum depth value in region")
    max_depth: float = Field(..., description="Maximum depth value in region")
    mean_depth: float = Field(..., description="Mean depth value in region")
    median_depth: float = Field(..., description="Median (50th percentile) depth in region")
    percentile_25: float = Field(..., description="25th percentile depth value")
    percentile_75: float = Field(..., description="75th percentile depth value")
    depth_unit: str = Field(
        default="relative",
        description="Depth unit/interpretation ('relative' normalized/disparity or 'meters')"
    )


class ObjectDepth(BaseModel):
    """
    Spatial depth estimation associated with an individual detected or tracked object.
    """
    track_id: Optional[int] = Field(default=None, description="Persistent tracking ID if available")
    class_name: Optional[str] = Field(default=None, description="Detected object class label")
    bbox: BoundingBox = Field(..., description="Object bounding box in original CCTV resolution")
    depth_stats: DepthStatistics = Field(..., description="Depth distribution statistics across object mask/bbox")
    contact_point: Optional[Tuple[float, float]] = Field(
        default=None,
        description="(x, y) ground contact coordinate tested (e.g. feet for persons)"
    )
    contact_depth: Optional[float] = Field(
        default=None,
        description="Estimated relative depth specifically at or around the ground contact point"
    )
    is_metric: bool = Field(
        default=False,
        description="Explicit flag: False for relative/uncalibrated depth, True only if metric meters"
    )


class DepthResult(BaseModel):
    """
    Frame-level monocular depth estimation metadata and summary statistics.
    Dense depth maps (NumPy arrays) are managed alongside this schema.
    """
    frame_id: Optional[int] = Field(default=None, description="Frame sequence counter")
    timestamp: float = Field(..., description="Video stream timestamp in seconds")
    width: int = Field(..., description="Native frame width corresponding to depth map")
    height: int = Field(..., description="Native frame height corresponding to depth map")
    depth_type: DepthType = Field(
        default=DepthType.RELATIVE,
        description="Measurement modality: relative (uncalibrated) or metric"
    )
    is_metric: bool = Field(
        default=False,
        description="Explicit indicator confirming whether depth values represent true physical meters"
    )
    min_depth: float = Field(..., description="Global minimum depth value in the frame")
    max_depth: float = Field(..., description="Global maximum depth value in the frame")
    mean_depth: float = Field(..., description="Global mean depth value in the frame")
    median_depth: float = Field(..., description="Global median depth value in the frame")
    processing_time_ms: Optional[float] = Field(
        default=None,
        description="Execution duration of depth inference in milliseconds"
    )
    depth_map_path: Optional[str] = Field(
        default=None,
        description="Optional file path to saved colorized depth map or serialized array"
    )
    object_depths: List[ObjectDepth] = Field(
        default_factory=list,
        description="Sampled depth measurements for tracked or detected objects in this frame"
    )
