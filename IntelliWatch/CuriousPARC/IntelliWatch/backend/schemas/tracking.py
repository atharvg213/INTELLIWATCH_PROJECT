from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, model_validator
from enum import Enum
from backend.schemas.detection import BoundingBox, ClassGroup, infer_class_group


class TrackState(str, Enum):
    """Lifecycle state of an individual tracked object."""
    NEW = "NEW"           # Created this frame, not yet confirmed
    ACTIVE = "ACTIVE"     # Actively matched across frames
    LOST = "LOST"         # Temporarily unmatched; Kalman filter predicts position
    REMOVED = "REMOVED"   # Exceeded max_lost_frames; track is dead


class TrackPoint(BaseModel):
    """Single historical centroid point stored in a track's trajectory."""
    frame_id: int = Field(..., description="Video frame index of this observation")
    timestamp: float = Field(default=0.0, description="Video timestamp in seconds")
    cx: float = Field(default=0.0, description="Centroid x-coordinate in original video pixels")
    cy: float = Field(default=0.0, description="Centroid y-coordinate in original video pixels")

    @model_validator(mode="before")
    @classmethod
    def _handle_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if "x" in data and "cx" not in data:
                data["cx"] = data["x"]
            if "y" in data and "cy" not in data:
                data["cy"] = data["y"]
        return data

    @property
    def x(self) -> float:
        return self.cx

    @property
    def y(self) -> float:
        return self.cy


class TrackedObject(BaseModel):
    """
    Persistent tracked entity across sequential video frames.

    Coordinate system: All bounding box and centroid coordinates are in the
    **original CCTV/video pixel space**, NOT the 640x640 letterbox space.
    Step 3 coordinate inversion is applied upstream before detections reach the tracker.
    """
    track_id: int = Field(..., description="Unique persistent ID assigned by ByteTrack")
    class_id: int = Field(..., description="COCO numeric class ID from YOLO")
    class_name: str = Field(..., description="Human-readable class label (e.g. 'person')")
    class_group: ClassGroup = Field(default=ClassGroup.OTHER, description="High-level category hierarchy")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence at last observation")
    bbox: BoundingBox = Field(..., description="Current bounding box in original video pixel coordinates")
    state: TrackState = Field(default=TrackState.ACTIVE, description="Current track lifecycle state")
    frame_index: int = Field(default=0, description="Current frame index")
    timestamp: float = Field(default=0.0, description="Current frame timestamp in seconds")

    # Centroid in original pixel coordinates
    centroid_x: Optional[float] = Field(default=None, description="Bounding box centroid x in original video pixels")
    centroid_y: Optional[float] = Field(default=None, description="Bounding box centroid y in original video pixels")

    # Trajectory history - limited to TRACK_HISTORY_LENGTH most recent points
    trajectory: List[TrackPoint] = Field(
        default_factory=list,
        description="Historical centroid positions for motion analysis (max configurable length)"
    )

    # Image-space motion estimates (NOT physical speed, NOT meters/second)
    velocity_x: Optional[float] = Field(
        None,
        description="Image-space horizontal displacement (pixels/frame) - NOT real-world speed"
    )
    velocity_y: Optional[float] = Field(
        None,
        description="Image-space vertical displacement (pixels/frame) - NOT real-world speed"
    )
    speed_pixels_per_second: Optional[float] = Field(
        None,
        description="Magnitude of image-space velocity in pixels/second - NOT real-world speed"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supplementary track metadata including display_id and user-facing labels"
    )

    @model_validator(mode="after")
    def _compute_centroid(self) -> "TrackedObject":
        if self.centroid_x is None:
            self.centroid_x = round((self.bbox.x1 + self.bbox.x2) / 2.0, 2)
        if self.centroid_y is None:
            self.centroid_y = round((self.bbox.y1 + self.bbox.y2) / 2.0, 2)
        if self.class_group == ClassGroup.OTHER:
            inferred = infer_class_group(self.class_name)
            if inferred != ClassGroup.OTHER:
                self.class_group = inferred
        return self

    @property
    def centroid(self) -> Tuple[float, float]:
        """Centroid coordinates tuple (cx, cy) in original video pixels."""
        return (self.centroid_x or 0.0, self.centroid_y or 0.0)


class FrameTracks(BaseModel):
    """All active/visible tracks in a given video frame."""
    frame_id: int = Field(..., description="Sequential video frame index")
    timestamp: float = Field(..., description="Video timestamp in seconds")
    active_tracks: List[TrackedObject] = Field(
        default_factory=list,
        description="All tracks currently visible in this frame (state=ACTIVE or NEW)"
    )
    total_track_count: int = Field(
        default=0,
        description="Cumulative unique tracks created since tracker initialization"
    )

