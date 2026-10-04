from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, model_validator


class ClassGroup(str, Enum):
    """High-level semantic categorization for industrial objects."""
    PERSON = "PERSON"
    PPE = "PPE"
    VEHICLE = "VEHICLE"
    MACHINE = "MACHINE"
    INFRASTRUCTURE = "INFRASTRUCTURE"
    OTHER = "OTHER"


def infer_class_group(class_name: str) -> ClassGroup:
    """Infers the high-level ClassGroup from a granular class name."""
    c = (class_name or "").lower().strip()
    if c in ("person", "worker", "pedestrian", "operator", "human"):
        return ClassGroup.PERSON
    if c in (
        "hardhat", "helmet", "safety vest", "vest", "gloves",
        "goggles", "mask", "boots", "no-hardhat", "no-safety vest",
        "no-helmet", "no-vest", "no-gloves", "no-goggles",
    ):
        return ClassGroup.PPE
    if c in (
        "forklift", "forklift truck", "industrial vehicle", "truck",
        "car", "vehicle", "bus", "van", "agv", "automated guided vehicle",
        "tractor",
    ):
        return ClassGroup.VEHICLE
    if c in (
        "machine", "machinery", "industrial machine", "conveyor",
        "conveyor belt", "crane", "robot", "robotic arm", "press",
        "lathe", "generator", "pump",
    ):
        return ClassGroup.MACHINE
    if c in (
        "pallet", "safety barrier", "barrier", "guardrail",
        "electrical cabinet", "cabinet", "shelf", "rack",
        "traffic cone", "cone", "bollard",
    ):
        return ClassGroup.INFRASTRUCTURE
    return ClassGroup.OTHER


class BoundingBox(BaseModel):
    """Coordinates of an object bounding box in pixels."""
    x1: float = Field(..., description="Top-left X coordinate")
    y1: float = Field(..., description="Top-left Y coordinate")
    x2: float = Field(..., description="Bottom-right X coordinate")
    y2: float = Field(..., description="Bottom-right Y coordinate")

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def center(self) -> tuple[float, float]:
        return (self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0


class DetectionResult(BaseModel):
    """Individual detected object in a single video frame."""
    detection_id: Optional[str] = Field(None, description="Unique detection identifier")
    class_id: int = Field(..., description="Numeric class ID")
    class_name: str = Field(..., description="Human-readable class name (e.g., worker, forklift, helmet)")
    class_group: ClassGroup = Field(default=ClassGroup.OTHER, description="High-level category hierarchy")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: BoundingBox = Field(..., description="Object bounding box")
    source_model: Optional[str] = Field(None, description="Source model that produced this detection")
    frame_index: Optional[int] = Field(None, description="Frame index of observation")
    timestamp: Optional[float] = Field(None, description="Timestamp in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary detector metadata")

    @model_validator(mode="before")
    @classmethod
    def _populate_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("detection_id"):
                data["detection_id"] = f"det_{uuid.uuid4().hex[:8]}"
            if "class_group" not in data or data.get("class_group") in (ClassGroup.OTHER, "OTHER", None):
                cname = data.get("class_name", "")
                inferred = infer_class_group(cname)
                if inferred != ClassGroup.OTHER:
                    data["class_group"] = inferred
        return data


class FrameDetections(BaseModel):
    """Collection of all detections belonging to a specific video frame."""
    frame_id: int = Field(..., ge=0, description="Sequential frame number")
    timestamp: float = Field(..., description="Video or capture timestamp in seconds")
    detections: list[DetectionResult] = Field(default_factory=list, description="List of detected objects")
    frame_width: Optional[int] = Field(None, description="Source frame width")
    frame_height: Optional[int] = Field(None, description="Source frame height")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic and processing metadata")

