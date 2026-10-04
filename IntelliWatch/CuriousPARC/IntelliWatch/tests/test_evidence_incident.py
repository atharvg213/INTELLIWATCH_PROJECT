"""
tests/test_evidence_incident.py
Step 12 Test Suite - Incident Evidence Management System.
Validates:
- IncidentRecord schema validation
- Local storage persistence in data/output/incidents/
- Snapshot image writing and metadata-only record handling
- Temporal lifecycle updates (CONFIRMED -> ACTIVE -> ENDED)
- Incident retrieval, filtering, and isolation
"""
from pathlib import Path
import numpy as np
import pytest

from backend.schemas.evidence import IncidentRecord
from backend.schemas.risk import (
    EventLifecycleState,
    RiskEventType,
    RiskFactor,
    RiskFactorType,
    RiskLevel,
)
from backend.services.incident_store import IncidentStore


@pytest.fixture
def temp_incident_store(tmp_path):
    """Returns an IncidentStore pointing to a temporary test directory."""
    return IncidentStore(storage_dir=tmp_path / "incidents")


@pytest.fixture
def sample_factor():
    """Generates a verifiable factual risk factor."""
    return RiskFactor(
        factor_id="factor_ppe_01",
        factor_type=RiskFactorType.PPE_NON_COMPLIANCE,
        severity_contribution=40.0,
        involved_entity_ids=["person_17"],
        timestamp=1.5,
        explanation="Worker #17 observed without mandatory Hardhat.",
    )


@pytest.fixture
def sample_incident(sample_factor):
    """Generates a structured IncidentRecord."""
    return IncidentRecord(
        incident_id="inc_test_001",
        timestamp=1.5,
        frame_id=45,
        camera_id="cam_01",
        event_type=RiskEventType.PPE_VIOLATION,
        risk_level=RiskLevel.HIGH,
        risk_score=75.0,
        lifecycle_state=EventLifecycleState.CONFIRMED,
        involved_track_ids=[17],
        involved_entity_ids=["person_17"],
        location_or_zone="restricted_zone_a",
        risk_factors=[sample_factor],
        explanation="Worker #17 entered Restricted Zone A while PPE compliance was incomplete.",
    )


def test_incident_record_schema(sample_incident):
    """Verifies IncidentRecord fields and serialization."""
    assert sample_incident.incident_id == "inc_test_001"
    assert sample_incident.risk_level == RiskLevel.HIGH
    assert sample_incident.risk_score == 75.0
    assert sample_incident.involved_track_ids == [17]
    assert len(sample_incident.risk_factors) == 1
    assert "Worker #17" in sample_incident.explanation


def test_incident_store_record_with_snapshot(temp_incident_store, sample_incident):
    """Verifies that an incident with an image frame saves a JPG and JSON file to disk."""
    frame = np.full((360, 640, 3), 100, dtype=np.uint8)
    saved = temp_incident_store.record_incident(sample_incident, frame_image=frame)

    assert saved.has_snapshot is True
    assert saved.snapshot_path is not None
    assert Path(saved.snapshot_path).exists()
    assert Path(saved.snapshot_path).name == "inc_test_001.jpg"

    # JSON metadata must also exist
    json_path = temp_incident_store.storage_dir / "inc_test_001.json"
    assert json_path.exists()

    # Retrieval from memory
    retrieved = temp_incident_store.get_incident("inc_test_001")
    assert retrieved is not None
    assert retrieved.incident_id == "inc_test_001"


def test_incident_store_metadata_only_record(temp_incident_store, sample_incident):
    """Verifies clean handling when no image snapshot is available."""
    saved = temp_incident_store.record_incident(sample_incident, frame_image=None)

    assert saved.has_snapshot is False
    assert saved.snapshot_path is None
    # JSON metadata must still be written
    assert (temp_incident_store.storage_dir / "inc_test_001.json").exists()


def test_incident_store_filtering(temp_incident_store, sample_factor):
    """Verifies incident listing with status and risk level filtering."""
    inc1 = IncidentRecord(
        incident_id="inc_001",
        timestamp=1.0,
        frame_id=30,
        camera_id="cam_01",
        event_type=RiskEventType.PPE_VIOLATION,
        risk_level=RiskLevel.LOW,
        risk_score=20.0,
        lifecycle_state=EventLifecycleState.ENDED,
        explanation="Worker missing gloves.",
    )
    inc2 = IncidentRecord(
        incident_id="inc_002",
        timestamp=2.0,
        frame_id=60,
        camera_id="cam_01",
        event_type=RiskEventType.RESTRICTED_ZONE_INTRUSION,
        risk_level=RiskLevel.HIGH,
        risk_score=80.0,
        lifecycle_state=EventLifecycleState.CONFIRMED,
        explanation="Unauthorized zone entry.",
    )
    temp_incident_store.record_incident(inc1)
    temp_incident_store.record_incident(inc2)

    # All incidents (reverse chronological: inc_002, inc_001)
    all_inc = temp_incident_store.list_incidents()
    assert len(all_inc) == 2
    assert all_inc[0].incident_id == "inc_002"

    # Filter by risk level
    high_inc = temp_incident_store.list_incidents(risk_level="HIGH")
    assert len(high_inc) == 1
    assert high_inc[0].incident_id == "inc_002"

    # Filter by status
    ended_inc = temp_incident_store.list_incidents(status="ENDED")
    assert len(ended_inc) == 1
    assert ended_inc[0].incident_id == "inc_001"


def test_incident_store_lifecycle_update(temp_incident_store, sample_incident):
    """Verifies updating the lifecycle state of a recorded incident."""
    temp_incident_store.record_incident(sample_incident)
    assert sample_incident.lifecycle_state == EventLifecycleState.CONFIRMED

    updated = temp_incident_store.update_incident_lifecycle(
        "inc_test_001",
        new_lifecycle=EventLifecycleState.ENDED,
        end_ts=5.5,
    )
    assert updated is not None
    assert updated.lifecycle_state == EventLifecycleState.ENDED
    assert updated.metadata.get("end_timestamp") == 5.5

    # Verify active list no longer includes it
    active = temp_incident_store.get_active_incidents()
    assert len(active) == 0


def test_incident_store_reset(temp_incident_store, sample_incident):
    """Verifies memory reset clears in-memory incidents."""
    temp_incident_store.record_incident(sample_incident)
    assert temp_incident_store.total_count() == 1

    temp_incident_store.reset()
    assert temp_incident_store.total_count() == 0
    assert temp_incident_store.get_incident("inc_test_001") is None
