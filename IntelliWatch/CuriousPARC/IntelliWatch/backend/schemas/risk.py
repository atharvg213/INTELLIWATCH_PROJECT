"""
backend/schemas/risk.py
Step 10 - Risk & Event Reasoning schemas.
Defines transparent, explainable safety risk factors, controlled event taxonomies,
risk classifications, and lifecycle states.
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class RiskLevel(str, Enum):
    """Controlled risk severity tiers."""
    INFO = "INFO"
    LOW = "LOW"
    MEDIUM = "MEDIUM"
    HIGH = "HIGH"
    CRITICAL = "CRITICAL"


class RiskFactorType(str, Enum):
    """Categorical types of verifiable safety risk factors."""
    PPE_NON_COMPLIANCE = "PPE_NON_COMPLIANCE"
    RESTRICTED_ZONE_INTRUSION = "RESTRICTED_ZONE_INTRUSION"
    RESTRICTED_ZONE_DWELL = "RESTRICTED_ZONE_DWELL"
    RAPID_MOVEMENT = "RAPID_MOVEMENT"
    SUDDEN_MOVEMENT = "SUDDEN_MOVEMENT"
    PROLONGED_STATIONARY = "PROLONGED_STATIONARY"
    FALL_LIKE_BEHAVIOR = "FALL_LIKE_BEHAVIOR"
    PERSON_VEHICLE_PROXIMITY = "PERSON_VEHICLE_PROXIMITY"
    APPROACHING_VEHICLE = "APPROACHING_VEHICLE"
    WORKER_NEAR_MACHINE = "WORKER_NEAR_MACHINE"
    MULTIPLE_RISK_FACTORS = "MULTIPLE_RISK_FACTORS"
    OTHER_SCENE_ANOMALY = "OTHER_SCENE_ANOMALY"


class RiskEventType(str, Enum):
    """Controlled taxonomy of safety and operational incident events."""
    PPE_VIOLATION = "PPE_VIOLATION"
    RESTRICTED_ZONE_INTRUSION = "RESTRICTED_ZONE_INTRUSION"
    RESTRICTED_ZONE_DWELL = "RESTRICTED_ZONE_DWELL"
    RAPID_MOVEMENT_EVENT = "RAPID_MOVEMENT_EVENT"
    FALL_LIKE_EVENT = "FALL_LIKE_EVENT"
    PERSON_VEHICLE_PROXIMITY = "PERSON_VEHICLE_PROXIMITY"
    APPROACHING_VEHICLE = "APPROACHING_VEHICLE"
    WORKER_MACHINE_RISK = "WORKER_MACHINE_RISK"
    COMPOUND_SAFETY_EVENT = "COMPOUND_SAFETY_EVENT"


class EventLifecycleState(str, Enum):
    """Temporal lifecycle states of a detected safety event."""
    CANDIDATE = "CANDIDATE"   # Condition observed, awaiting temporal confirmation frames
    CONFIRMED = "CONFIRMED"   # Temporally confirmed in current frame
    ACTIVE = "ACTIVE"         # Confirmed and persisting across frames
    ENDED = "ENDED"           # Ceased or actor departed
    COOLDOWN = "COOLDOWN"     # Temporarily suppressed post-end to prevent rapid re-triggering


class RiskFactor(BaseModel):
    """
    Individual verifiable risk factor contributing to overall scene or entity risk.
    """
    factor_id: str = Field(..., description="Unique deterministic identifier for this factor observation")
    factor_type: RiskFactorType = Field(..., description="Categorical classification of the risk factor")
    severity_contribution: float = Field(..., description="Configured numeric risk score contribution")
    involved_entity_ids: List[str] = Field(default_factory=list, description="IDs of entities involved (e.g. 'person_17')")
    source_evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Factual, verifiable evidence without model confidence conflation",
    )
    timestamp: float = Field(..., description="Video stream timestamp in seconds")
    explanation: str = Field(..., description="Deterministic, explainable justification for this factor")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary context")


class RiskEvent(BaseModel):
    """
    Explainable safety or operational event produced by deterministic risk reasoning.
    """
    event_id: str = Field(..., description="Unique event instance identifier (UUID or deterministic key)")
    event_type: RiskEventType = Field(..., description="Controlled event type from safety taxonomy")
    lifecycle_state: EventLifecycleState = Field(
        default=EventLifecycleState.CANDIDATE,
        description="Current temporal lifecycle phase",
    )
    timestamp: float = Field(..., description="Timestamp of the current observation")
    start_timestamp: float = Field(..., description="Timestamp when event was first observed")
    end_timestamp: Optional[float] = Field(default=None, description="Timestamp when event ended (if ended)")
    involved_entities: List[str] = Field(default_factory=list, description="IDs of all participating scene entities")
    risk_level: RiskLevel = Field(..., description="Classified risk tier: INFO, LOW, MEDIUM, HIGH, CRITICAL")
    risk_score: float = Field(..., description="Transparent aggregated numeric risk score")
    risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Underlying risk factors supporting this event without double-counting",
    )
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Structured verifiable geometric and temporal evidence",
    )
    explanation: str = Field(..., description="Deterministic explainable description of the event")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional context")

    @property
    def severity_tier(self) -> RiskLevel:
        """Alias for risk_level."""
        return self.risk_level

    @property
    def deterministic_explanation(self) -> str:
        """Alias for explanation."""
        return self.explanation


class RiskSummary(BaseModel):
    """
    Aggregated summary of risk levels across all active confirmed events in the scene.
    """
    info: int = Field(default=0, description="Count of active events at INFO level")
    low: int = Field(default=0, description="Count of active events at LOW risk level")
    medium: int = Field(default=0, description="Count of active events at MEDIUM risk level")
    high: int = Field(default=0, description="Count of active events at HIGH risk level")
    critical: int = Field(default=0, description="Count of active events at CRITICAL risk level")
    total_active_events: int = Field(default=0, description="Total active confirmed safety events")
    max_risk_level: RiskLevel = Field(default=RiskLevel.INFO, description="Highest active risk level in the scene")
    max_risk_score: float = Field(default=0.0, description="Highest risk score among active events")


class FrameRiskAssessment(BaseModel):
    """
    Complete structured risk assessment for a single video frame.
    """
    frame_id: Optional[int] = Field(default=None, description="Sequential frame index")
    timestamp: float = Field(default=0.0, description="Video timestamp in seconds")
    camera_id: str = Field(default="cam_01", description="Camera feed identifier")
    active_events: List[RiskEvent] = Field(
        default_factory=list,
        description="All confirmed and active events occurring in this frame",
    )
    recent_events: List[RiskEvent] = Field(
        default_factory=list,
        description="Events updated in this frame (newly confirmed or ended)",
    )
    risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="All atomic risk factors evaluated in this frame",
    )
    risk_summary: RiskSummary = Field(
        default_factory=RiskSummary,
        description="Aggregated risk tier counts and maximum severity",
    )
