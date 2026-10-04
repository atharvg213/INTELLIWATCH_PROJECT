"""
backend/schemas/temporal.py
Step 15 — Temporal Event Log Schema.

Records meaningful state/relationship transitions over time —
not every frame, only significant changes that advance understanding.

Technical honesty:
- All motion values are image-space (pixels or pixels/second).
- No metric distance claimed from relative depth.
- Timestamps are video stream timestamps in seconds.
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class TemporalEventCategory(str, Enum):
    """Category of a temporal event log entry."""
    ENTITY_DETECTED     = "ENTITY_DETECTED"      # New track confirmed
    ENTITY_LOST         = "ENTITY_LOST"          # Track disappeared
    BEHAVIOR_TRANSITION = "BEHAVIOR_TRANSITION"  # Primary behavior changed
    SPATIAL_TRANSITION  = "SPATIAL_TRANSITION"   # Proximity tier changed (FAR→NEAR etc.)
    RELATIONSHIP_START  = "RELATIONSHIP_START"   # New spatial relation confirmed
    RELATIONSHIP_END    = "RELATIONSHIP_END"     # Spatial relation ended
    RISK_ESCALATION     = "RISK_ESCALATION"      # Risk level went up
    RISK_DECAY          = "RISK_DECAY"           # Risk level went down
    EARLY_WARNING       = "EARLY_WARNING"        # New early-warning indicator
    INCIDENT_CONFIRMED  = "INCIDENT_CONFIRMED"   # Incident evidence recorded


class TemporalLogEntry(BaseModel):
    """
    A single meaningful state-change event recorded in the temporal log.
    Only significant transitions are recorded — not every frame.
    """
    seq:         int   = Field(..., description="Monotonic sequence number across the session")
    timestamp:   float = Field(..., description="Video stream timestamp (seconds)")
    frame_id:    int   = Field(..., description="Frame index when this event occurred")
    category:    TemporalEventCategory = Field(..., description="Category of the transition")
    entity_id:   str   = Field(..., description="Primary entity (e.g. 'person_1', 'forklift_2')")
    target_id:   Optional[str] = Field(default=None, description="Secondary entity if relational")
    label:       str   = Field(..., description="Short human-readable label (e.g. 'MOVING → RAPID_MOVEMENT')")
    description: str   = Field(..., description="Longer deterministic explanation")
    prev_state:  Optional[str] = Field(default=None, description="Previous state value")
    new_state:   str           = Field(...,            description="New state value")
    metadata:    Dict[str, Any] = Field(default_factory=dict)


class TemporalEntityProfile(BaseModel):
    """
    Summary temporal profile for a single tracked entity.
    Built incrementally from TemporalLogEntry records.
    """
    entity_id:           str   = Field(..., description="Node ID (e.g. 'person_1')")
    track_id:            int   = Field(..., description="ByteTrack persistent ID")
    class_name:          str   = Field(..., description="Detector class name")
    first_seen_ts:       float = Field(default=0.0, description="Timestamp first observed")
    last_seen_ts:        float = Field(default=0.0, description="Timestamp most recently observed")
    current_behavior:    str   = Field(default="UNKNOWN", description="Confirmed primary behavior")
    behavior_duration_s: float = Field(default=0.0, description="Seconds in current behavior")
    current_spatial_state: Optional[str] = Field(default=None, description="Most recent proximity tier")
    spatial_target_id:   Optional[str]   = Field(default=None, description="Entity this proximity relates to")
    image_speed_px_per_s: Optional[float] = Field(default=None)
    heading_degrees:      Optional[float]  = Field(default=None)
    depth_trend:          Optional[str]    = Field(default=None)
    risk_trend_label:     Optional[str]    = Field(default=None,
                              description="'LOW → MEDIUM → HIGH', etc. Max 5 most recent levels")
    active_early_warnings: List[str]       = Field(default_factory=list)
    timeline:              List[TemporalLogEntry] = Field(
        default_factory=list,
        description="Ordered log of significant transitions for this entity (most recent 20)",
    )
