"""
backend/schemas/alert.py
Step 22 — Production-Oriented Safety Alerting, Evidence Capture & Alert Lifecycle Schemas.

Defines schemas for:
- Alert Lifecycle State Machine (NEW, ACKNOWLEDGED, RESOLVED, DISMISSED)
- Configurable Severity Levels (CRITICAL, HIGH, MEDIUM, LOW)
- Complete Traceable Alert Records with Visual Evidence references
- State Transition Audit History
- Request/Response contracts for Alert management, filtering, pagination, and statistics
"""
from enum import Enum
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class AlertStatus(str, Enum):
    """Lifecycle state machine for safety alerts."""
    NEW = "NEW"
    ACKNOWLEDGED = "ACKNOWLEDGED"
    RESOLVED = "RESOLVED"
    DISMISSED = "DISMISSED"


class AlertSeverity(str, Enum):
    """Configurable alert severity levels."""
    CRITICAL = "CRITICAL"
    HIGH = "HIGH"
    MEDIUM = "MEDIUM"
    LOW = "LOW"


class AlertHistoryEntry(BaseModel):
    """Audit log entry capturing state transitions and operator attribution."""
    from_status: Optional[AlertStatus] = Field(default=None, description="Previous status before transition")
    to_status: AlertStatus = Field(..., description="Target status after transition")
    timestamp: float = Field(..., description="Unix epoch timestamp in seconds of state change")
    user: str = Field(default="operator", description="Operator ID or username who initiated action")
    notes: Optional[str] = Field(default=None, description="Operational notes or dismissal justification")


class AlertRecord(BaseModel):
    """
    Actionable, traceable safety alert ticket generated from perception events.
    Includes camera, worker, severity, confidence, lifecycle status, visual evidence,
    and audit transition history.
    """
    alert_id: str = Field(..., description="Unique deterministic or UUID identifier (e.g. 'alert_abc123')")
    camera_id: str = Field(default="cam_01", description="Identifier of the originating camera feed")
    camera_name: str = Field(default="Main Camera", description="Human-readable camera / location name")
    track_id: Optional[int] = Field(default=None, description="Worker tracking ID if tracked, None if untracked/scene-level")
    violation_type: str = Field(..., description="Standardized violation type (e.g. NO_HARDHAT, FALL_DETECTED)")
    severity: AlertSeverity = Field(default=AlertSeverity.HIGH, description="Assigned severity level")
    confidence: float = Field(default=0.85, ge=0.0, le=1.0, description="Confidence score of underlying perception detection")
    status: AlertStatus = Field(default=AlertStatus.NEW, description="Current lifecycle state")

    # Timestamps & Temporal Correlation
    timestamp: float = Field(..., description="Unix epoch timestamp (seconds) when alert was first triggered")
    first_detected_at: float = Field(..., description="Timestamp of initial event detection")
    last_detected_at: float = Field(..., description="Timestamp of most recent event detection within correlation window")
    occurrence_count: int = Field(default=1, ge=1, description="Number of consecutive or correlated detections aggregated")

    # Lifecycle Actor Attribution
    acknowledged_at: Optional[float] = Field(default=None, description="Timestamp when operator acknowledged alert")
    acknowledged_by: Optional[str] = Field(default=None, description="Username/ID of operator who acknowledged")
    resolved_at: Optional[float] = Field(default=None, description="Timestamp when alert was resolved")
    resolved_by: Optional[str] = Field(default=None, description="Username/ID of operator who marked resolved")
    resolution_notes: Optional[str] = Field(default=None, description="Resolution comments or mitigation actions taken")
    dismissed_at: Optional[float] = Field(default=None, description="Timestamp when alert was dismissed")
    dismissed_by: Optional[str] = Field(default=None, description="Username/ID of operator who dismissed")
    dismiss_reason: Optional[str] = Field(default=None, description="Justification for dismissal (e.g. false positive)")

    # Visual Evidence
    has_evidence: bool = Field(default=False, description="Whether visual evidence image snapshot was persisted")
    evidence_image_path: Optional[str] = Field(default=None, description="Local or relative disk path to evidence snapshot")

    # Metadata & Explainability
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Supplementary perception metadata (box, zone, etc.)")
    transition_history: List[AlertHistoryEntry] = Field(default_factory=list, description="Chronological audit history")


class AlertAcknowledgeRequest(BaseModel):
    """Payload to acknowledge an alert."""
    user: str = Field(default="operator", description="Operator identifier")
    notes: Optional[str] = Field(default=None, description="Optional acknowledgment note")


class AlertResolveRequest(BaseModel):
    """Payload to resolve an alert."""
    user: str = Field(default="operator", description="Operator identifier")
    notes: Optional[str] = Field(default=None, description="Required or optional resolution notes")


class AlertDismissRequest(BaseModel):
    """Payload to dismiss an alert."""
    user: str = Field(default="operator", description="Operator identifier")
    reason: str = Field(..., description="Mandatory reason for dismissing alert (e.g. 'False positive reflection')")


class AlertListResponse(BaseModel):
    """Paginated response for alert queries."""
    items: List[AlertRecord] = Field(default_factory=list, description="Page of alert records")
    total: int = Field(default=0, description="Total matching alerts count")
    page: int = Field(default=1, description="Current page number (1-indexed)")
    page_size: int = Field(default=20, description="Number of items per page")
    total_pages: int = Field(default=1, description="Total number of pages")


class AlertStatsResponse(BaseModel):
    """Aggregated operational statistics of the safety alerting system."""
    total_alerts: int = Field(default=0, description="Total lifetime recorded alerts")
    by_status: Dict[str, int] = Field(default_factory=dict, description="Count of alerts by lifecycle status")
    by_severity: Dict[str, int] = Field(default_factory=dict, description="Count of alerts by severity tier")
    by_violation: Dict[str, int] = Field(default_factory=dict, description="Count of alerts by violation type")
    active_unresolved_count: int = Field(default=0, description="Total active alerts in NEW or ACKNOWLEDGED state")
