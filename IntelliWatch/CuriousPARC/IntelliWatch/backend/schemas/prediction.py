"""
backend/schemas/prediction.py
Step 11 - Predictive & Advanced Anomaly Intelligence schemas.
Defines early-warning indicator taxonomies, projected trajectory models,
temporal anomaly indicators, and predictive assessment schemas.
Strict technical honesty: projected trajectories are short-horizon geometric projections,
NOT guaranteed future paths; risk escalation is a trend observation, NOT accident prediction.
"""
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field


class EarlyWarningIndicatorType(str, Enum):
    """Controlled taxonomy of early-warning safety indicators."""
    TRAJECTORY_TOWARD_RESTRICTED_ZONE = "TRAJECTORY_TOWARD_RESTRICTED_ZONE"
    TRAJECTORY_TOWARD_VEHICLE         = "TRAJECTORY_TOWARD_VEHICLE"
    TRAJECTORY_TOWARD_MACHINE         = "TRAJECTORY_TOWARD_MACHINE"
    RISK_ESCALATING                   = "RISK_ESCALATING"
    PERSISTENT_UNSAFE_PATTERN         = "PERSISTENT_UNSAFE_PATTERN"
    REPEATED_PPE_VIOLATION            = "REPEATED_PPE_VIOLATION"
    REPEATED_ZONE_VIOLATION           = "REPEATED_ZONE_VIOLATION"
    PERSISTENT_VEHICLE_PROXIMITY      = "PERSISTENT_VEHICLE_PROXIMITY"
    PERSISTENT_MACHINE_PROXIMITY      = "PERSISTENT_MACHINE_PROXIMITY"
    TEMPORAL_BEHAVIOR_ANOMALY         = "TEMPORAL_BEHAVIOR_ANOMALY"
    REPEATED_COMPOUND_RISK            = "REPEATED_COMPOUND_RISK"


class IndicatorLifecycleState(str, Enum):
    """Temporal confirmation lifecycle for early-warning indicators."""
    CANDIDATE = "CANDIDATE"   # Observed, awaiting persistence threshold
    CONFIRMED = "CONFIRMED"   # Newly confirmed in current observation
    ACTIVE    = "ACTIVE"      # Confirmed and sustained across multiple frames
    ENDED     = "ENDED"       # Condition ceased or actor departed


class IndicatorSeverity(str, Enum):
    """Urgency / severity level of the early-warning indicator."""
    LOW      = "LOW"
    MEDIUM   = "MEDIUM"
    HIGH     = "HIGH"
    CRITICAL = "CRITICAL"


class ProjectedTrajectory(BaseModel):
    """
    Short-horizon projected position for a tracked entity based on recent image-space kinematics.
    NOTE: All values operate in image-space pixels. They do NOT represent real-world metric distance.
    """
    track_id: int = Field(..., description="Track ID from ByteTrack")
    current_position: Tuple[float, float] = Field(..., description="Current centroid or contact point (x, y)")
    projected_position: Tuple[float, float] = Field(..., description="Projected position (x, y) at horizon")
    horizon_frames: int = Field(default=15, description="Number of frames ahead for projection")
    horizon_seconds: float = Field(default=0.5, description="Estimated time horizon in seconds")
    estimated_speed_px_per_s: float = Field(default=0.0, description="Image-space speed magnitude (px/s)")
    heading_degrees: Optional[float] = Field(default=None, description="Image-space bearing angle (0-360 deg)")
    history_points_used: int = Field(default=0, description="Number of historical trajectory points used")
    timestamp: float = Field(default=0.0, description="Timestamp of the projection")


class EarlyWarningIndicator(BaseModel):
    """
    Explainable early-warning safety indicator produced by predictive temporal reasoning.
    """
    indicator_id: str = Field(..., description="Unique deterministic or UUID identifier")
    indicator_type: EarlyWarningIndicatorType = Field(..., description="Controlled indicator category")
    lifecycle_state: IndicatorLifecycleState = Field(
        default=IndicatorLifecycleState.CANDIDATE,
        description="Temporal lifecycle status (CANDIDATE, CONFIRMED, ACTIVE, ENDED)",
    )
    timestamp: float = Field(..., description="Timestamp of the observation")
    first_observed_timestamp: float = Field(..., description="Timestamp when indicator was first observed")
    end_timestamp: Optional[float] = Field(default=None, description="Timestamp when condition ceased (if ended)")
    track_ids: List[int] = Field(default_factory=list, description="Involved ByteTrack IDs")
    entity_ids: List[str] = Field(default_factory=list, description="Involved node IDs (e.g. 'person_17')")
    target_zone_id: Optional[str] = Field(default=None, description="Target restricted zone ID if relevant")
    severity: IndicatorSeverity = Field(default=IndicatorSeverity.MEDIUM, description="Urgency tier")
    horizon_frames: Optional[int] = Field(default=None, description="Projection horizon in frames if applicable")
    projected_trajectory: Optional[ProjectedTrajectory] = Field(default=None, description="Associated projection")
    trend: Optional[str] = Field(default=None, description="Qualitative trend label: escalating, persistent, etc.")
    evidence: Dict[str, Any] = Field(default_factory=dict, description="Verifiable geometric and temporal evidence")
    supporting_events: List[str] = Field(default_factory=list, description="IDs of supporting confirmed RiskEvents")
    supporting_risk_factors: List[str] = Field(default_factory=list, description="Types of contributing RiskFactors")
    explanation: str = Field(..., description="Deterministic, explainable justification")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary context")


class FramePredictionAssessment(BaseModel):
    """
    Complete structured predictive & anomaly intelligence assessment for a single video frame.
    """
    frame_id: Optional[int] = Field(default=None, description="Frame index")
    timestamp: float = Field(default=0.0, description="Video timestamp in seconds")
    camera_id: str = Field(default="cam_01", description="Camera feed identifier")
    projected_trajectories: List[ProjectedTrajectory] = Field(
        default_factory=list,
        description="Short-horizon kinematic projections for active tracks",
    )
    active_indicators: List[EarlyWarningIndicator] = Field(
        default_factory=list,
        description="Confirmed and active early-warning indicators",
    )
    recent_indicators: List[EarlyWarningIndicator] = Field(
        default_factory=list,
        description="Indicators newly confirmed or ended in this frame",
    )
    total_active_indicators: int = Field(default=0, description="Count of active confirmed indicators")
    max_indicator_severity: IndicatorSeverity = Field(
        default=IndicatorSeverity.LOW,
        description="Highest severity among active indicators",
    )
