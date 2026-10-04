"""
backend/schemas/explanation.py
Step 17 - Explainability, Evidence & Safety Analytics Schemas.

Defines structured, deterministic schemas for:
- Five-W Incident Explanations (WHO, WHAT, WHERE, WHEN, WHY)
- Risk Score Breakdown (base factor weights, multi-factor escalation, final score)
- Chronological Incident Timelines
- Risk Trend Visualizations
- Complete Incident Evidence Packages
- Safety Analytics Summaries
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.schemas.risk import EventLifecycleState, RiskEventType, RiskLevel


class FiveWExplanation(BaseModel):
    """
    Deterministic Five-W (Who, What, Where, When, Why) structured explanation
    derived entirely from perception, tracking, compliance, behavior, and risk state.
    Zero generative AI or LLMs used.
    """
    who: str = Field(..., description="Involved entities, ByteTrack IDs, PPE compliance, and behavior state")
    what: str = Field(..., description="Categorical and human-readable description of the detected incident")
    where: str = Field(..., description="Configured zone, area, or camera image-space location")
    when: str = Field(..., description="Timestamp in seconds and frame index of occurrence")
    why: List[str] = Field(
        default_factory=list,
        description="Factual, deterministic bullet points explaining why this event was classified as a risk",
    )


class RiskBreakdownItem(BaseModel):
    """Itemized contribution of a single atomic risk factor."""
    factor_type: str = Field(..., description="Risk factor taxonomy type")
    contribution: float = Field(..., description="Configured numeric weight added to base score")
    explanation: str = Field(..., description="Deterministic factual explanation of factor")


class RiskScoreBreakdown(BaseModel):
    """
    Transparent mathematical risk score breakdown.
    Exposes the existing risk engine calculation without hidden weights.
    """
    base_score: float = Field(..., description="Sum of atomic factor severity contributions")
    factor_contributions: List[RiskBreakdownItem] = Field(
        default_factory=list,
        description="List of individual factor additions",
    )
    escalation: float = Field(
        default=0.0,
        description="Compounding escalation added for concurrent multi-factor hazards",
    )
    final_score: float = Field(..., description="Base score + escalation")
    risk_tier: RiskLevel = Field(..., description="Assigned risk severity tier (INFO, LOW, MEDIUM, HIGH, CRITICAL)")
    thresholds: Dict[str, float] = Field(
        default_factory=dict,
        description="Configured score threshold boundaries for each risk tier",
    )


class IncidentTimelineEntry(BaseModel):
    """A chronological event point relative to an incident's lifecycle."""
    timestamp: float = Field(..., description="Timestamp in seconds")
    frame_id: int = Field(..., description="Frame sequence number")
    category: str = Field(..., description="Category: DETECTION, BEHAVIOR, ZONE, SPATIAL, RISK, WARNING, INCIDENT")
    label: str = Field(..., description="Short headline label")
    description: str = Field(..., description="Descriptive sentence")
    risk_level: Optional[str] = Field(default=None, description="Associated risk tier if applicable")


class RiskTrendPoint(BaseModel):
    """Historical risk point for risk-over-time trend visualization."""
    timestamp: float = Field(..., description="Observation timestamp in seconds")
    frame_id: int = Field(..., description="Frame index")
    risk_score: float = Field(..., description="Numeric risk score at this point")
    risk_level: str = Field(..., description="Risk level string at this point")


class IncidentExplanationResponse(BaseModel):
    """
    Complete explainability payload for a selected safety incident.
    Answers all 8 operator questions: What, Who, Where, When, Why, Evidence,
    Risk evolution, and Before/After sequence.
    """
    incident_id: str = Field(..., description="Unique incident ID")
    event_type: RiskEventType = Field(..., description="Categorical incident event type")
    risk_level: RiskLevel = Field(..., description="Risk severity tier")
    risk_score: float = Field(..., description="Numeric risk score")
    lifecycle_state: EventLifecycleState = Field(..., description="Lifecycle status: CANDIDATE, CONFIRMED, ACTIVE, ENDED, COOLDOWN")
    timestamp: float = Field(..., description="Primary incident timestamp in seconds")
    end_timestamp: Optional[float] = Field(default=None, description="Timestamp when incident ended")
    duration_s: Optional[float] = Field(default=None, description="Duration of active incident in seconds")
    frame_id: int = Field(..., description="Frame sequence number")
    camera_id: str = Field(default="cam_01", description="Camera sensor identifier")

    # The 5W Explanation
    five_w: FiveWExplanation = Field(..., description="Structured 5W explanation")

    # Risk Calculation Breakdown
    risk_breakdown: RiskScoreBreakdown = Field(..., description="Mathematical score breakdown")

    # Chronological Incident Progression
    timeline: List[IncidentTimelineEntry] = Field(
        default_factory=list,
        description="Chronological sequence of transitions surrounding the incident",
    )

    # Risk Trend Visualization Data
    risk_trend: List[RiskTrendPoint] = Field(
        default_factory=list,
        description="Time-series risk score trajectory",
    )

    # Scene & Predictive Context
    scene_relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant active scene-graph relationships",
    )
    early_warnings: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Associated early-warning predictive indicators",
    )

    # Evidence details
    has_snapshot: bool = Field(default=False, description="Whether visual snapshot image exists")
    snapshot_url: Optional[str] = Field(default=None, description="Relative URL to retrieve visual JPEG evidence")
    evidence_notice: str = Field(default="", description="Honest visual evidence notice")


class AnalyticsSummary(BaseModel):
    """
    Descriptive safety analytics summary metrics across recorded operational history.
    Strictly descriptive statistics; zero employee evaluation or profiling claims.
    """
    total_incidents: int = Field(default=0, description="Total confirmed safety incidents recorded")
    critical_incidents: int = Field(default=0, description="Count of CRITICAL severity incidents")
    high_risk_incidents: int = Field(default=0, description="Count of HIGH severity incidents")
    medium_risk_incidents: int = Field(default=0, description="Count of MEDIUM severity incidents")
    low_risk_incidents: int = Field(default=0, description="Count of LOW severity incidents")
    info_risk_incidents: int = Field(default=0, description="Count of INFO severity incidents")
    active_incidents: int = Field(default=0, description="Incidents currently active or unended")

    # Event Type Counts
    zone_intrusions: int = Field(default=0, description="Restricted zone intrusion incident count")
    zone_dwell_exceeded: int = Field(default=0, description="Restricted zone dwell timeout count")
    ppe_violations: int = Field(default=0, description="PPE non-compliance incident count")
    rapid_movement_events: int = Field(default=0, description="Rapid / sudden movement incident count")
    vehicle_proximity_events: int = Field(default=0, description="Person-vehicle proximity incident count")
    fall_like_events: int = Field(default=0, description="Fall-like heuristic event count")
    compound_events: int = Field(default=0, description="Multi-hazard compound event count")

    # Time Metrics
    average_incident_duration_s: Optional[float] = Field(default=None, description="Mean duration of completed incidents in seconds")

    # Distributions
    event_type_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Frequency count by categorical event type",
    )
    risk_tier_distribution: Dict[str, int] = Field(
        default_factory=dict,
        description="Frequency count by risk severity tier",
    )
    most_active_zones: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Frequency ranking of restricted zones with incident occurrences",
    )
