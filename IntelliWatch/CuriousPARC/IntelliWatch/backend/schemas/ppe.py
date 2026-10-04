from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.schemas.detection import BoundingBox, DetectionResult


class PPEItemState(str, Enum):
    """Observation status of a specific PPE item on a tracked worker."""
    PRESENT = "PRESENT"       # Confirmed detected and spatially associated
    MISSING = "MISSING"       # Confirmed absent despite adequate visual observability
    UNKNOWN = "UNKNOWN"       # Insufficient visual evidence (e.g. occlusion, truncation, distance)


class ComplianceStatus(str, Enum):
    """Overall safety gear compliance status of a tracked worker."""
    COMPLIANT = "COMPLIANT"         # All mandatory PPE items are verified PRESENT
    NON_COMPLIANT = "NON_COMPLIANT" # One or more mandatory PPE items are verified MISSING
    UNKNOWN = "UNKNOWN"             # Insufficient evidence to certify compliance


class AssociatedPPEItem(BaseModel):
    """A specific PPE detection spatially mapped to a tracked worker."""
    class_id: int = Field(..., description="Detector class ID")
    class_name: str = Field(..., description="Canonical or raw PPE class name (e.g. 'Hardhat')")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    bbox: BoundingBox = Field(..., description="Bounding box in original video pixel space")
    association_score: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Spatial match quality score between PPE item and worker"
    )
    is_negative: bool = Field(
        default=False,
        description="True if this is an explicit negative violation detection (e.g. 'NO-Hardhat')"
    )
    low_confidence_candidate: bool = Field(
        default=False,
        description="True when the detection passed the association-aware candidate floor below the configured detector threshold"
    )
    body_region: Optional[str] = Field(
        default=None,
        description="Target anatomical region (e.g. 'head', 'torso', 'hands', 'face')"
    )


class WorkerPPEInventory(BaseModel):
    """Structured PPE inventory and compliance evaluation for an individual tracked worker."""
    track_id: int = Field(..., description="ByteTrack persistent identifier of the worker")
    timestamp: float = Field(default=0.0, description="Video timestamp in seconds")
    bbox: BoundingBox = Field(..., description="Worker bounding box in original video pixels")
    items: List[AssociatedPPEItem] = Field(
        default_factory=list,
        description="All individual PPE items spatially associated with this worker"
    )
    ppe_status: Dict[str, PPEItemState] = Field(
        default_factory=dict,
        description="Per-category PPE state mapping (e.g. {'Hardhat': PRESENT, 'Gloves': MISSING})"
    )
    compliance_status: ComplianceStatus = Field(
        default=ComplianceStatus.UNKNOWN,
        description="Overall compliance verdict for this worker"
    )
    missing_ppe: List[str] = Field(
        default_factory=list,
        description="List of required PPE items evaluated as MISSING"
    )
    present_ppe: List[str] = Field(
        default_factory=list,
        description="List of required PPE items evaluated as PRESENT"
    )
    unknown_ppe: List[str] = Field(
        default_factory=list,
        description="List of required PPE items evaluated as UNKNOWN (insufficient evidence)"
    )
    explanation: Optional[str] = Field(
        default=None,
        description="Explainable breakdown of compliance decision"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Supplementary worker telemetry and stable display mapping"
    )


class FramePPEAssociation(BaseModel):
    """Complete collection of worker PPE inventories and unassociated PPE items for a video frame."""
    frame_id: int = Field(..., ge=0, description="Video frame index")
    timestamp: float = Field(default=0.0, description="Video timestamp in seconds")
    worker_inventories: List[WorkerPPEInventory] = Field(
        default_factory=list,
        description="PPE inventories for each tracked person in the frame"
    )
    unassociated_ppe: List[DetectionResult] = Field(
        default_factory=list,
        description="PPE detections that could not be reliably associated with any tracked person"
    )
