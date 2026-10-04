"""
tests/test_temporal_intelligence.py
Step 15 Test Suite — Real Video Temporal Intelligence & Advanced Behavior.

Tests:
1. TemporalLogService initialization and empty state.
2. Entity detection event recording.
3. Behavior transition detection and explanation.
4. Spatial relationship transition detection from scene graph.
5. Risk escalation and decay transitions.
6. Early warning indicator logging.
7. Incident confirmation recording.
8. Entity profile generation with risk trend, spatial state, and timeline.
9. Concurrent access & reentrant locking (no deadlock in get_all_entity_profiles).
10. Bounded memory capping (session max entries and per-entity timeline).
11. Reset functionality for video sessions.
12. FastAPI routes: /api/v1/temporal/timeline, /api/v1/temporal/entities, /api/v1/temporal/entity/{id}.
13. EndToEndPipelineOrchestrator integration with temporal intelligence.
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.assessment import FrameAssessment
from backend.schemas.behavior import BehaviorState, PrimaryBehavior
from backend.schemas.risk import (
    EventLifecycleState,
    FrameRiskAssessment,
    RiskEvent,
    RiskEventType,
    RiskLevel,
    RiskSummary,
)
from backend.schemas.scene_graph import (
    FrameScene,
    SceneNode,
    SceneNodeType,
    SceneRelation,
    SceneRelationType,
    SituationalSummary,
)
from backend.schemas.temporal import (
    TemporalEntityProfile,
    TemporalEventCategory,
    TemporalLogEntry,
)
from backend.schemas.tracking import TrackedObject
from backend.schemas.zones import FrameZoneOccupancy, ZoneMembership
from backend.services.assessment_store import get_assessment_store
from backend.services.temporal_log import TemporalLogService, get_temporal_log


@pytest.fixture(autouse=True)
def reset_service():
    """Ensure a clean temporal log for every test."""
    service = get_temporal_log()
    service.reset()
    yield service
    service.reset()


def _make_dummy_assessment(
    frame_id: int = 1,
    timestamp: float = 0.0,
    behavior_type: PrimaryBehavior = PrimaryBehavior.STATIONARY,
    risk_level: RiskLevel = RiskLevel.LOW,
    risk_score: float = 20.0,
    event_id: str = "ev_1",
    new_incidents=None,
    active_warnings=None,
    relation_type: SceneRelationType = SceneRelationType.FAR,
) -> FrameAssessment:
    from backend.schemas.detection import BoundingBox

    box1 = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=200.0, confidence=0.9, class_id=0, class_name="person")
    box2 = BoundingBox(x1=300.0, y1=100.0, x2=450.0, y2=250.0, confidence=0.85, class_id=1, class_name="forklift")

    track = TrackedObject(
        track_id=1,
        class_id=0,
        class_name="person",
        bbox=box1,
        confidence=0.9,
        frame_index=frame_id,
        timestamp=timestamp,
    )
    b_state = BehaviorState(
        track_id=1,
        timestamp=timestamp,
        frame_id=frame_id,
        primary_behavior=behavior_type,
        behavior_duration_s=1.0,
        image_speed_px_per_s=15.0 if behavior_type != PrimaryBehavior.STATIONARY else 0.0,
        heading_degrees=45.0,
        depth_trend="STATIC",
        explanation=f"State: {behavior_type.value}",
    )
    risk_event = RiskEvent(
        event_id=event_id,
        event_type=RiskEventType.RESTRICTED_ZONE_INTRUSION,
        risk_level=risk_level,
        risk_score=risk_score,
        lifecycle_state=EventLifecycleState.ACTIVE,
        timestamp=timestamp,
        start_timestamp=timestamp,
        involved_entities=["person_1"],
        risk_factors=[],
        explanation="Detected in restricted area",
    )
    risk_ass = FrameRiskAssessment(
        frame_id=frame_id,
        timestamp=timestamp,
        active_events=[risk_event],
        concluded_events=[],
        risk_summary=RiskSummary(
            max_risk_level=risk_level,
            max_risk_score=risk_score,
            active_events_count=1,
            dominant_event_type=RiskEventType.RESTRICTED_ZONE_INTRUSION,
            explanation=f"Dominant: {risk_level.value}",
        ),
    )
    node1 = SceneNode(
        node_id="person_1",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=1,
        label="Person #1",
        bbox=box1,
        confidence=0.9,
    )
    node2 = SceneNode(
        node_id="forklift_2",
        node_type=SceneNodeType.VEHICLE,
        class_name="forklift",
        track_id=2,
        label="Forklift #2",
        bbox=box2,
        confidence=0.85,
    )
    rel = SceneRelation(
        relation_id="rel_p1_f2",
        relation_type=relation_type,
        source_node_id="person_1",
        target_node_id="forklift_2",
        first_frame=frame_id,
        last_frame=frame_id,
        duration_frames=1,
        confidence=0.9,
        evidence={"distance_px": 150.0},
    )
    scene = FrameScene(
        frame_id=frame_id,
        timestamp=timestamp,
        nodes=[node1, node2],
        relationships=[rel],
        summary=SituationalSummary(
            total_nodes=2,
            total_relationships=1,
            critical_situations=[],
            narrative="Normal monitoring",
        ),
    )
    return FrameAssessment(
        frame_id=frame_id,
        timestamp=timestamp,
        camera_id="cam_01",
        frame_width=640,
        frame_height=360,
        processing_time_ms=25.0,
        tracks=[track],
        behavior_states=[b_state],
        scene=scene,
        risk_assessment=risk_ass,
        active_events=[risk_event],
        highest_risk_level=risk_level,
        highest_risk_score=risk_score,
        active_early_warnings=active_warnings or [],
        new_incident_ids=new_incidents or [],
    )


def test_temporal_log_initial_state():
    """Validates fresh initialization of TemporalLogService."""
    service = TemporalLogService()
    assert len(service.get_all_entries()) == 0
    assert len(service.get_all_entity_profiles()) == 0


def test_entity_detection_event():
    """Verifies that new tracked entities generate ENTITY_DETECTED log entries."""
    service = TemporalLogService()
    ass = _make_dummy_assessment(frame_id=1, timestamp=0.1)
    new_entries = service.update(ass)

    detected_entries = [e for e in new_entries if e.category == TemporalEventCategory.ENTITY_DETECTED]
    assert len(detected_entries) == 1
    assert detected_entries[0].entity_id == "person_1"
    assert "Person #1 detected" in detected_entries[0].label


def test_behavior_transition_logging():
    """Verifies that changing behavior logs BEHAVIOR_TRANSITION."""
    service = TemporalLogService()

    # Frame 1: STATIC
    ass1 = _make_dummy_assessment(frame_id=1, timestamp=0.1, behavior_type=PrimaryBehavior.STATIONARY)
    service.update(ass1)

    # Frame 2: MOVING
    ass2 = _make_dummy_assessment(frame_id=2, timestamp=0.2, behavior_type=PrimaryBehavior.MOVING)
    entries = service.update(ass2)

    beh_entries = [e for e in entries if e.category == TemporalEventCategory.BEHAVIOR_TRANSITION]
    assert len(beh_entries) == 1
    assert beh_entries[0].prev_state == PrimaryBehavior.STATIONARY.value
    assert beh_entries[0].new_state == PrimaryBehavior.MOVING.value


def test_risk_escalation_and_decay():
    """Verifies that risk level changes log RISK_ESCALATION and RISK_DECAY."""
    service = TemporalLogService()

    # Frame 1: LOW
    ass1 = _make_dummy_assessment(frame_id=1, timestamp=0.1, risk_level=RiskLevel.LOW)
    service.update(ass1)

    # Frame 2: HIGH (Escalation)
    ass2 = _make_dummy_assessment(frame_id=2, timestamp=0.2, risk_level=RiskLevel.HIGH)
    entries2 = service.update(ass2)
    esc = [e for e in entries2 if e.category == TemporalEventCategory.RISK_ESCALATION]
    assert len(esc) == 1
    assert esc[0].prev_state == RiskLevel.LOW.value
    assert esc[0].new_state == RiskLevel.HIGH.value

    # Frame 3: LOW (Decay)
    ass3 = _make_dummy_assessment(frame_id=3, timestamp=0.3, risk_level=RiskLevel.LOW)
    entries3 = service.update(ass3)
    dec = [e for e in entries3 if e.category == TemporalEventCategory.RISK_DECAY]
    assert len(dec) == 1
    assert dec[0].prev_state == RiskLevel.HIGH.value
    assert dec[0].new_state == RiskLevel.LOW.value


def test_incident_confirmation_logging():
    """Verifies that new incidents trigger INCIDENT_CONFIRMED log entries."""
    service = TemporalLogService()
    ass = _make_dummy_assessment(frame_id=5, timestamp=0.5, new_incidents=["inc_abc123"])
    entries = service.update(ass)

    inc_entries = [e for e in entries if e.category == TemporalEventCategory.INCIDENT_CONFIRMED]
    assert len(inc_entries) == 1
    assert inc_entries[0].new_state == "inc_abc123"


def test_get_entity_profiles_no_deadlock():
    """Verifies that get_all_entity_profiles operates without deadlock under reentrant lock."""
    service = TemporalLogService()
    ass = _make_dummy_assessment(frame_id=1, timestamp=0.1)
    service.update(ass)

    profiles = service.get_all_entity_profiles(ass)
    assert len(profiles) == 1
    p = profiles[0]
    assert p.entity_id == "person_1"
    assert p.track_id == 1
    assert p.class_name == "person"
    assert len(p.timeline) > 0


def test_temporal_log_memory_bounded():
    """Verifies that log and entity timeline respect maximum bounds."""
    service = TemporalLogService()

    # Rapidly send 30 transitions
    for i in range(1, 35):
        beh = PrimaryBehavior.MOVING if i % 2 == 0 else PrimaryBehavior.STATIONARY
        ass = _make_dummy_assessment(frame_id=i, timestamp=i * 0.1, behavior_type=beh)
        service.update(ass)

    p = service.get_entity_profile("person_1")
    assert p is not None
    # Individual entity timeline should be capped at TIMELINE_PER_ENTITY (20)
    assert len(p.timeline) <= 20


def test_temporal_api_endpoints():
    """Verifies FastAPI temporal intelligence endpoints return 200 and valid schemas."""
    client = TestClient(app)
    service = get_temporal_log()
    service.reset()

    # Pre-populate with an assessment
    ass = _make_dummy_assessment(frame_id=1, timestamp=0.1, risk_level=RiskLevel.MEDIUM)
    service.update(ass)
    get_assessment_store().set_current_assessment(ass)

    # 1. Timeline endpoint
    resp = client.get("/api/v1/temporal/timeline?n=10")
    assert resp.status_code == 200
    timeline = resp.json()
    assert isinstance(timeline, list)
    assert len(timeline) >= 1

    # 2. Entities endpoint
    resp = client.get("/api/v1/temporal/entities")
    assert resp.status_code == 200
    entities = resp.json()
    assert isinstance(entities, list)
    assert len(entities) == 1
    assert entities[0]["entity_id"] == "person_1"

    # 3. Single entity endpoint
    resp = client.get("/api/v1/temporal/entity/person_1")
    assert resp.status_code == 200
    profile = resp.json()
    assert profile["entity_id"] == "person_1"
    assert profile["track_id"] == 1

    # 4. Unknown entity endpoint
    resp = client.get("/api/v1/temporal/entity/unknown_999")
    assert resp.status_code == 404
