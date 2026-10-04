from enum import Enum
from typing import Any, Optional
from pydantic import BaseModel, Field


class SeverityLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class EventType(str, Enum):
    PPE_VIOLATION = "ppe_violation"
    ZONE_INTRUSION = "zone_intrusion"
    ZONE_ENTRY = "zone_entry"
    ZONE_EXIT = "zone_exit"
    ZONE_DWELL_EXCEEDED = "zone_dwell_exceeded"
    FORKLIFT_PROXIMITY = "forklift_proximity"
    FALL_DETECTED = "fall_detected"
    FIRE_SMOKE = "fire_smoke"
    LOITERING = "loitering"
    MACHINE_ANOMALY = "machine_anomaly"
    UNSAFE_BEHAVIOR = "unsafe_behavior"


class IndustrialEvent(BaseModel):
    """Structured safety or operational incident detected in the scene."""
    event_id: str = Field(..., description="UUID or unique event identifier")
    event_type: EventType = Field(..., description="Category of the industrial event")
    timestamp: float = Field(..., description="Epoch or stream timestamp of occurrence")
    camera_id: str = Field("cam_01", description="Identifier of the reporting camera feed")
    tracked_object_ids: list[int] = Field(
        default_factory=list,
        description="List of persistent track IDs involved in the event"
    )
    severity: SeverityLevel = Field(..., description="Risk or severity classification")
    evidence_reference: Optional[str] = Field(
        None,
        description="File path or URL to keyframe snapshot/evidence crop"
    )
    explanation: str = Field(
        ...,
        description="Human-readable explanation of why this alert was triggered"
    )
    metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary supplementary context (e.g. zone ID, distance in meters)"
    )
