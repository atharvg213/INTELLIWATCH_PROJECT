"""
backend/schemas/evidence.py
Step 12 - Incident Evidence & System Status Schemas.
Defines transparent, verifiable incident evidence records,
historical queries, and operational telemetry schemas.
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

from backend.schemas.risk import EventLifecycleState, RiskEventType, RiskFactor, RiskLevel
from backend.schemas.prediction import EarlyWarningIndicator


class IncidentRecord(BaseModel):
    """
    Structured evidence record for a confirmed industrial safety incident.
    Stores factual metadata, structured risk factors, deterministic explanation,
    and optional local snapshot filepath.
    """
    incident_id: str = Field(..., description="Unique deterministic or UUID identifier")
    timestamp: float = Field(..., description="Video stream timestamp in seconds")
    frame_id: int = Field(..., description="Frame sequence index when incident was recorded")
    camera_id: str = Field(default="cam_01", description="Camera / sensor feed identifier")
    event_type: RiskEventType = Field(..., description="Categorical incident event type")
    risk_level: RiskLevel = Field(..., description="Assigned risk tier (INFO, LOW, MEDIUM, HIGH, CRITICAL)")
    risk_score: float = Field(..., description="Computed numeric risk severity score")
    lifecycle_state: EventLifecycleState = Field(
        default=EventLifecycleState.CONFIRMED,
        description="Temporal lifecycle state (CANDIDATE, CONFIRMED, ACTIVE, ENDED, COOLDOWN)",
    )

    # Participating Entities
    involved_track_ids: List[int] = Field(default_factory=list, description="ByteTrack IDs involved")
    involved_entity_ids: List[str] = Field(default_factory=list, description="Scene node IDs (e.g. 'person_17')")
    location_or_zone: Optional[str] = Field(default=None, description="Associated restricted zone or area name")

    # Explainability & Evidence
    risk_factors: List[RiskFactor] = Field(
        default_factory=list,
        description="Atomic factual risk factors contributing to this incident",
    )
    explanation: str = Field(..., description="Deterministic explainable justification of the incident")
    scene_relationships: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Relevant active scene graph relationships at time of incident",
    )
    early_warnings: List[EarlyWarningIndicator] = Field(
        default_factory=list,
        description="Associated early-warning indicators active during or prior to incident",
    )

    # Snapshot storage
    has_snapshot: bool = Field(default=False, description="Whether an image snapshot file is stored")
    snapshot_path: Optional[str] = Field(default=None, description="Local path to annotated evidence snapshot image")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary contextual telemetry")


class SystemStatusResponse(BaseModel):
    """
    Operational system status and telemetry response model.
    Technically honest: explicitly states CPU processing mode,
    measured processing latency, and active perception models.
    """
    status: str = Field(default="healthy", description="Operational status of backend")
    pipeline_state: str = Field(default="idle", description="Pipeline state: idle, running, paused, error")
    device: str = Field(default="cpu", description="Configured inference device (cpu, cuda)")
    is_cpu_mode: bool = Field(default=True, description="Whether currently running in CPU mode")
    current_source: Optional[str] = Field(default=None, description="Active video stream source")
    current_frame_id: int = Field(default=0, description="Most recently processed frame index")
    processing_fps: float = Field(default=0.0, description="Measured average frame processing rate")
    avg_latency_ms: float = Field(default=0.0, description="Average per-frame processing latency in ms")
    total_frames_processed: int = Field(default=0, description="Total frames processed since start")
    active_tracks_count: int = Field(default=0, description="Current number of active tracked entities")
    active_incidents_count: int = Field(default=0, description="Current number of active safety incidents")
    total_recorded_incidents: int = Field(default=0, description="Total confirmed incident evidence records stored")
    models_loaded: Dict[str, bool] = Field(
        default_factory=dict,
        description="Readiness status of individual perception models",
    )
