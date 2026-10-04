"""
backend/schemas/assessment.py
Step 12 - Comprehensive End-to-End Frame Assessment Schema.
Integrates all perception, tracking, compliance, spatial, behavior,
scene graph, risk, and early-warning intelligence outputs for a single frame.
"""
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.schemas.detection import DetectionResult
from backend.schemas.tracking import TrackedObject
from backend.schemas.ppe import WorkerPPEInventory
from backend.schemas.zones import ZoneMembership
from backend.schemas.depth import DepthStatistics, ObjectDepth
from backend.schemas.behavior import BehaviorState
from backend.schemas.scene_graph import FrameScene, SituationalSummary
from backend.schemas.risk import FrameRiskAssessment, RiskEvent, RiskLevel
from backend.schemas.prediction import FramePredictionAssessment, EarlyWarningIndicator
from backend.schemas.temporal import TemporalLogEntry, TemporalEntityProfile


class FrameAssessment(BaseModel):
    """
    Comprehensive structured scene assessment for a single video frame.
    Connects all intelligence layers into a single explainable record.
    """
    frame_id: int = Field(default=0, description="Sequential video frame index")
    timestamp: float = Field(default=0.0, description="Video timestamp in seconds")
    camera_id: str = Field(default="cam_01", description="Camera feed identifier")
    frame_width: int = Field(default=1280, description="Native frame width in pixels")
    frame_height: int = Field(default=720, description="Native frame height in pixels")
    processing_time_ms: float = Field(default=0.0, description="End-to-end processing latency in ms")

    # Perception & Tracking
    detections: List[DetectionResult] = Field(default_factory=list, description="Raw object detections")
    ppe_detections: List[DetectionResult] = Field(default_factory=list, description="Raw PPE object detections")
    tracks: List[TrackedObject] = Field(default_factory=list, description="Active persistent tracks")
    total_active_tracks: int = Field(default=0, description="Count of currently active tracks")

    # PPE & Compliance
    worker_inventories: List[WorkerPPEInventory] = Field(
        default_factory=list,
        description="Per-worker PPE inventory and compliance status",
    )
    non_compliant_workers_count: int = Field(default=0, description="Count of verified non-compliant workers")

    # Spatial & Zones
    zone_memberships: List[ZoneMembership] = Field(
        default_factory=list,
        description="Active spatial zone occupancy states",
    )
    occupied_zones_count: int = Field(default=0, description="Count of zones currently occupied")

    # Relative Depth Perception
    depth_statistics: Optional[DepthStatistics] = Field(
        default=None,
        description="Frame-level relative depth percentiles (disparity/inverse distance)",
    )
    object_depths: List[ObjectDepth] = Field(
        default_factory=list,
        description="Per-object relative depth samples and contact depths",
    )

    # Behavior & Temporal Analysis
    behavior_states: List[BehaviorState] = Field(
        default_factory=list,
        description="Current behavior classification for active entities",
    )

    # Scene Graph & Situational Awareness
    scene: Optional[FrameScene] = Field(
        default=None,
        description="Current scene graph nodes and directional relationships",
    )
    situational_summary: Optional[SituationalSummary] = Field(
        default=None,
        description="Aggregated count of entities, zones, and spatial relations",
    )

    # Risk & Event Reasoning
    risk_assessment: Optional[FrameRiskAssessment] = Field(
        default=None,
        description="Current frame risk assessment with active events and factors",
    )
    active_events: List[RiskEvent] = Field(
        default_factory=list,
        description="Active confirmed safety and operational events",
    )
    highest_risk_level: RiskLevel = Field(
        default=RiskLevel.INFO,
        description="Highest active risk tier (INFO, LOW, MEDIUM, HIGH, CRITICAL)",
    )
    highest_risk_score: float = Field(default=0.0, description="Peak active risk score")

    # Predictive & Early-Warning Intelligence
    prediction_assessment: Optional[FramePredictionAssessment] = Field(
        default=None,
        description="Projected trajectories and early-warning safety indicators",
    )
    active_early_warnings: List[EarlyWarningIndicator] = Field(
        default_factory=list,
        description="Active confirmed early-warning safety indicators",
    )

    # Incident alerts triggered in this frame
    new_incident_ids: List[str] = Field(
        default_factory=list,
        description="IDs of newly confirmed incident evidence records",
    )

    # Step 15 — Temporal Intelligence
    temporal_events: List[TemporalLogEntry] = Field(
        default_factory=list,
        description="Meaningful state-change events recorded in this frame (not every frame)",
    )
    entity_profiles: List[TemporalEntityProfile] = Field(
        default_factory=list,
        description="Per-entity temporal profiles summarizing behavior, spatial state, and risk trend",
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Arbitrary pipeline, detector, and diagnostic metadata",
    )
