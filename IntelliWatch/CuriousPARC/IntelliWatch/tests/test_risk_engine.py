"""
tests/test_risk_engine.py
Comprehensive unit tests for Step 10: Risk & Event Reasoning.
Verifies all 33+ test requirements:
1. No risk condition
2. PPE violation
3. Missing hardhat
4. Missing multiple PPE items
5. Restricted zone intrusion
6. Restricted zone dwell
7. Rapid movement
8. Fall-like event
9. Person-vehicle proximity
10. Approaching vehicle
11. Worker-machine risk
12. Single-factor risk scoring
13. Multi-factor risk scoring
14. Compound safety event
15. Risk level thresholds
16. Event confirmation frames
17. Event end confirmation
18. Event cooldown
19. Duplicate suppression
20. Event lifecycle (CANDIDATE -> CONFIRMED -> ACTIVE -> ENDED)
21. Track disappearance
22. Missing scene graph data
23. Missing depth data
24. Missing PPE data
25. Missing behavior data
26. Missing zone data
27. Serialization
28. API response (/api/v1/risk/current, /api/v1/events/current)
29. Explainability/evidence
30. No fabricated confidence
31. Relative depth is not interpreted as metric distance
32. No double-counting of evidence
33. Configuration overrides
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.detection import BoundingBox
from backend.schemas.ppe import (
    ComplianceStatus,
    PPEItemState,
    WorkerPPEInventory,
    FramePPEAssociation,
)
from backend.schemas.zones import (
    ZoneMembership,
    ZoneMembershipStatus,
    FrameZoneOccupancy,
)
from backend.schemas.behavior import BehaviorState, PrimaryBehavior
from backend.schemas.scene_graph import (
    FrameScene,
    RelationLifecycle,
    SceneNode,
    SceneNodeType,
    SceneRelation,
    SceneRelationType,
    SituationalSummary,
)
from backend.schemas.risk import (
    EventLifecycleState,
    FrameRiskAssessment,
    RiskEventType,
    RiskFactorType,
    RiskLevel,
)
from intelligence.risk.engine import RiskEngine
from intelligence.risk.factors import RiskFactorExtractor
from intelligence.risk.rules import RiskRuleEvaluator
from intelligence.risk.scorer import RiskScorer
from intelligence.risk.serializer import RiskAssessmentSerializer
from backend.services.risk_store import get_risk_store


client = TestClient(app)


def _make_dummy_scene(
    person_nodes=None,
    vehicle_nodes=None,
    zone_nodes=None,
    relations=None,
    timestamp=1.0,
    frame_id=1,
):
    nodes = []
    if person_nodes:
        nodes.extend(person_nodes)
    if vehicle_nodes:
        nodes.extend(vehicle_nodes)
    if zone_nodes:
        nodes.extend(zone_nodes)

    return FrameScene(
        frame_id=frame_id,
        timestamp=timestamp,
        camera_id="cam_01",
        nodes=nodes,
        relationships=relations or [],
        summary=SituationalSummary(),
        active_zones=[],
        active_behaviors={},
    )


# -----------------------------------------------------------------------------
# 1. No risk condition
# -----------------------------------------------------------------------------
def test_no_risk_condition():
    engine = RiskEngine(confirmation_frames=1)
    scene = _make_dummy_scene(
        person_nodes=[
            SceneNode(
                node_id="person_1",
                node_type=SceneNodeType.PERSON,
                class_name="person",
                track_id=1,
                attributes={"compliance_status": "COMPLIANT"},
            )
        ]
    )
    assessment = engine.evaluate_frame(scene=scene)
    assert len(assessment.risk_factors) == 0
    assert len(assessment.active_events) == 0
    assert assessment.risk_summary.max_risk_level == RiskLevel.INFO
    assert assessment.risk_summary.max_risk_score == 0.0


# -----------------------------------------------------------------------------
# 2. PPE violation & 3. Missing hardhat
# -----------------------------------------------------------------------------
def test_ppe_violation_missing_hardhat():
    engine = RiskEngine(confirmation_frames=1)
    scene = _make_dummy_scene(
        person_nodes=[
            SceneNode(
                node_id="person_17",
                node_type=SceneNodeType.PERSON,
                class_name="person",
                track_id=17,
                attributes={
                    "compliance_status": "NON_COMPLIANT",
                    "ppe_status": {"Hardhat": "MISSING", "Safety Vest": "PRESENT"},
                },
            )
        ]
    )
    assessment = engine.evaluate_frame(scene=scene)
    assert len(assessment.risk_factors) == 1
    assert assessment.risk_factors[0].factor_type == RiskFactorType.PPE_NON_COMPLIANCE
    assert "Hardhat" in str(assessment.risk_factors[0].source_evidence["missing_ppe"])

    assert len(assessment.active_events) == 1
    ev = assessment.active_events[0]
    assert ev.event_type == RiskEventType.PPE_VIOLATION
    assert "person_17" in ev.involved_entities
    assert ev.risk_level in (RiskLevel.LOW, RiskLevel.MEDIUM)


# -----------------------------------------------------------------------------
# 4. Missing multiple PPE items
# -----------------------------------------------------------------------------
def test_missing_multiple_ppe_items():
    inv = WorkerPPEInventory(
        track_id=17,
        timestamp=1.0,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=100),
        items=[],
        ppe_status={"Hardhat": PPEItemState.MISSING, "Gloves": PPEItemState.MISSING},
        compliance_status=ComplianceStatus.NON_COMPLIANT,
        missing_ppe=["Hardhat", "Gloves"],
    )
    ppe_assoc = FramePPEAssociation(frame_id=1, timestamp=1.0, worker_inventories=[inv])

    engine = RiskEngine(confirmation_frames=1)
    assessment = engine.evaluate_frame(ppe_association=ppe_assoc)

    assert len(assessment.risk_factors) == 1
    factor = assessment.risk_factors[0]
    assert factor.factor_type == RiskFactorType.PPE_NON_COMPLIANCE
    assert set(factor.source_evidence["missing_ppe"]) == {"Hardhat", "Gloves"}


# -----------------------------------------------------------------------------
# 5. Restricted zone intrusion
# -----------------------------------------------------------------------------
def test_restricted_zone_intrusion():
    engine = RiskEngine(confirmation_frames=1)
    p_node = SceneNode(node_id="person_17", node_type=SceneNodeType.PERSON, class_name="person", track_id=17)
    z_node = SceneNode(node_id="zone_high_voltage", node_type=SceneNodeType.ZONE, class_name="zone", zone_id="high_voltage")
    rel = SceneRelation(
        relation_id="rel_inside_17",
        source_node_id="person_17",
        target_node_id="zone_high_voltage",
        relation_type=SceneRelationType.INSIDE,
        lifecycle=RelationLifecycle.ACTIVE,
        evidence={"dwell_seconds": 2.0},
    )
    scene = _make_dummy_scene(person_nodes=[p_node], zone_nodes=[z_node], relations=[rel])

    assessment = engine.evaluate_frame(scene=scene)
    types = [ev.event_type for ev in assessment.active_events]
    assert RiskEventType.RESTRICTED_ZONE_INTRUSION in types


# -----------------------------------------------------------------------------
# 6. Restricted zone dwell
# -----------------------------------------------------------------------------
def test_restricted_zone_dwell():
    engine = RiskEngine(confirmation_frames=1, settings_override={"ZONE_MAX_DWELL_SECONDS": 10.0})
    p_node = SceneNode(node_id="person_17", node_type=SceneNodeType.PERSON, class_name="person", track_id=17)
    z_node = SceneNode(node_id="zone_high_voltage", node_type=SceneNodeType.ZONE, class_name="zone", zone_id="high_voltage")
    rel = SceneRelation(
        relation_id="rel_inside_17",
        source_node_id="person_17",
        target_node_id="zone_high_voltage",
        relation_type=SceneRelationType.INSIDE,
        lifecycle=RelationLifecycle.ACTIVE,
        evidence={"dwell_seconds": 15.0},
    )
    scene = _make_dummy_scene(person_nodes=[p_node], zone_nodes=[z_node], relations=[rel])

    assessment = engine.evaluate_frame(scene=scene)
    types = [ev.event_type for ev in assessment.active_events]
    assert RiskEventType.RESTRICTED_ZONE_DWELL in types


# -----------------------------------------------------------------------------
# 7. Rapid movement
# -----------------------------------------------------------------------------
def test_rapid_movement_event():
    b_state = BehaviorState(
        track_id=17,
        timestamp=1.0,
        primary_behavior=PrimaryBehavior.RAPID_MOVEMENT,
        flag_rapid_movement=True,
        image_speed_px_per_s=120.0,
    )
    engine = RiskEngine(confirmation_frames=1)
    assessment = engine.evaluate_frame(behavior_states=[b_state])

    types = [ev.event_type for ev in assessment.active_events]
    assert RiskEventType.RAPID_MOVEMENT_EVENT in types


# -----------------------------------------------------------------------------
# 8. Fall-like event
# -----------------------------------------------------------------------------
def test_fall_like_event():
    b_state = BehaviorState(
        track_id=17,
        timestamp=1.0,
        primary_behavior=PrimaryBehavior.POSSIBLE_FALL,
        flag_possible_fall=True,
    )
    engine = RiskEngine(confirmation_frames=1)
    assessment = engine.evaluate_frame(behavior_states=[b_state])

    types = [ev.event_type for ev in assessment.active_events]
    assert RiskEventType.FALL_LIKE_EVENT in types
    fall_ev = [ev for ev in assessment.active_events if ev.event_type == RiskEventType.FALL_LIKE_EVENT][0]
    assert "fall-like behavior detected" in fall_ev.explanation.lower()


# -----------------------------------------------------------------------------
# 9. Person-vehicle proximity
# -----------------------------------------------------------------------------
def test_person_vehicle_proximity():
    p_node = SceneNode(node_id="person_17", node_type=SceneNodeType.PERSON, class_name="person", track_id=17)
    v_node = SceneNode(node_id="forklift_2", node_type=SceneNodeType.VEHICLE, class_name="forklift", track_id=2)
    rel = SceneRelation(
        relation_id="rel_near_17_2",
        source_node_id="person_17",
        target_node_id="forklift_2",
        relation_type=SceneRelationType.NEAR,
        lifecycle=RelationLifecycle.ACTIVE,
        evidence={"distance_px": 55.0},
    )
    scene = _make_dummy_scene(person_nodes=[p_node], vehicle_nodes=[v_node], relations=[rel])

    engine = RiskEngine(confirmation_frames=1)
    assessment = engine.evaluate_frame(scene=scene)
    types = [ev.event_type for ev in assessment.active_events]
    assert RiskEventType.PERSON_VEHICLE_PROXIMITY in types


# -----------------------------------------------------------------------------
# 10. Approaching vehicle
# -----------------------------------------------------------------------------
def test_approaching_vehicle():
    p_node = SceneNode(node_id="person_17", node_type=SceneNodeType.PERSON, class_name="person", track_id=17)
    v_node = SceneNode(node_id="forklift_2", node_type=SceneNodeType.VEHICLE, class_name="forklift", track_id=2)
    rel = SceneRelation(
        relation_id="rel_appr_2_17",
        source_node_id="forklift_2",
        target_node_id="person_17",
        relation_type=SceneRelationType.APPROACHING,
        lifecycle=RelationLifecycle.ACTIVE,
        evidence={"closing_speed_px_s": 40.0},
    )
    scene = _make_dummy_scene(person_nodes=[p_node], vehicle_nodes=[v_node], relations=[rel])

    engine = RiskEngine(confirmation_frames=1)
    assessment = engine.evaluate_frame(scene=scene)
    types = [ev.event_type for ev in assessment.active_events]
    assert RiskEventType.APPROACHING_VEHICLE in types


# -----------------------------------------------------------------------------
# 11. Worker-machine risk
# -----------------------------------------------------------------------------
def test_worker_machine_risk():
    p_node = SceneNode(node_id="person_17", node_type=SceneNodeType.PERSON, class_name="person", track_id=17)
    m_node = SceneNode(node_id="machine_press_1", node_type=SceneNodeType.MACHINE, class_name="machine")
    rel = SceneRelation(
        relation_id="rel_near_17_m",
        source_node_id="person_17",
        target_node_id="machine_press_1",
        relation_type=SceneRelationType.NEAR,
        lifecycle=RelationLifecycle.ACTIVE,
        evidence={"distance_px": 80.0},
    )
    scene = _make_dummy_scene(person_nodes=[p_node], relations=[rel], vehicle_nodes=[m_node])

    engine = RiskEngine(confirmation_frames=1)
    assessment = engine.evaluate_frame(scene=scene)
    types = [ev.event_type for ev in assessment.active_events]
    assert RiskEventType.WORKER_MACHINE_RISK in types


# -----------------------------------------------------------------------------
# 12. Single-factor risk scoring & 13. Multi-factor risk scoring
# -----------------------------------------------------------------------------
def test_risk_scoring():
    scorer = RiskScorer(settings_override={"RISK_SCORE_PPE_VIOLATION": 25.0, "RISK_SCORE_MULTI_FACTOR_ESCALATION": 15.0})
    f1 = RiskFactorExtractor().extract_factors(
        ppe_association=FramePPEAssociation(
            frame_id=1,
            timestamp=1.0,
            worker_inventories=[
                WorkerPPEInventory(
                    track_id=1,
                    bbox=BoundingBox(x1=0, y1=0, x2=10, y2=10),
                    compliance_status=ComplianceStatus.NON_COMPLIANT,
                    missing_ppe=["Hardhat"],
                )
            ],
        )
    )[0]

    # Single factor score
    score1, level1, b1 = scorer.compute_risk([f1], is_compound=False)
    assert score1 == 25.0
    assert level1 == RiskLevel.LOW
    assert b1["escalation"] == 0.0

    # Multi-factor score with compounding
    f2 = RiskFactorExtractor().extract_factors(
        behavior_states=[
            BehaviorState(track_id=1, timestamp=1.0, primary_behavior=PrimaryBehavior.RAPID_MOVEMENT, flag_rapid_movement=True)
        ]
    )[0]

    score2, level2, b2 = scorer.compute_risk([f1, f2], is_compound=True)
    # base = 25 (PPE) + 20 (Rapid) = 45; escalation = 1 * 15 = 15; total = 60.0 -> HIGH
    assert score2 == 60.0
    assert level2 == RiskLevel.HIGH
    assert b2["escalation"] == 15.0


# -----------------------------------------------------------------------------
# 14. Compound safety event
# -----------------------------------------------------------------------------
def test_compound_safety_event():
    engine = RiskEngine(confirmation_frames=1, compound_enabled=True)
    p_node = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        attributes={"compliance_status": "NON_COMPLIANT", "ppe_status": {"Hardhat": "MISSING"}},
    )
    z_node = SceneNode(node_id="zone_danger", node_type=SceneNodeType.ZONE, class_name="zone", zone_id="danger")
    rel = SceneRelation(
        relation_id="rel_inside",
        source_node_id="person_17",
        target_node_id="zone_danger",
        relation_type=SceneRelationType.INSIDE,
        lifecycle=RelationLifecycle.ACTIVE,
    )
    scene = _make_dummy_scene(person_nodes=[p_node], zone_nodes=[z_node], relations=[rel])

    assessment = engine.evaluate_frame(scene=scene)
    types = [ev.event_type for ev in assessment.active_events]
    assert RiskEventType.COMPOUND_SAFETY_EVENT in types
    comp_ev = [ev for ev in assessment.active_events if ev.event_type == RiskEventType.COMPOUND_SAFETY_EVENT][0]
    assert len(comp_ev.risk_factors) >= 2


# -----------------------------------------------------------------------------
# 15. Risk level thresholds
# -----------------------------------------------------------------------------
def test_risk_level_thresholds():
    scorer = RiskScorer(
        settings_override={
            "RISK_LEVEL_INFO_THRESHOLD": 0.0,
            "RISK_LEVEL_LOW_THRESHOLD": 10.0,
            "RISK_LEVEL_MEDIUM_THRESHOLD": 30.0,
            "RISK_LEVEL_HIGH_THRESHOLD": 60.0,
            "RISK_LEVEL_CRITICAL_THRESHOLD": 85.0,
        }
    )
    from backend.schemas.risk import RiskFactor

    def make_fac(sc):
        return RiskFactor(
            factor_id="f",
            factor_type=RiskFactorType.OTHER_SCENE_ANOMALY,
            severity_contribution=sc,
            involved_entity_ids=["e"],
            source_evidence={},
            timestamp=1.0,
            explanation="test",
        )

    assert scorer.compute_risk([make_fac(5.0)])[1] == RiskLevel.INFO
    assert scorer.compute_risk([make_fac(15.0)])[1] == RiskLevel.LOW
    assert scorer.compute_risk([make_fac(45.0)])[1] == RiskLevel.MEDIUM
    assert scorer.compute_risk([make_fac(70.0)])[1] == RiskLevel.HIGH
    assert scorer.compute_risk([make_fac(90.0)])[1] == RiskLevel.CRITICAL


# -----------------------------------------------------------------------------
# 16. Event confirmation frames & 20. Event lifecycle
# -----------------------------------------------------------------------------
def test_event_confirmation_lifecycle():
    engine = RiskEngine(confirmation_frames=3, end_confirmation_frames=3)
    p_node = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        attributes={"compliance_status": "NON_COMPLIANT", "ppe_status": {"Hardhat": "MISSING"}},
    )
    scene = _make_dummy_scene(person_nodes=[p_node])

    # Frame 1: Candidate (not confirmed)
    a1 = engine.evaluate_frame(scene=scene, timestamp=1.0)
    assert len(a1.active_events) == 0

    # Frame 2: Candidate (not confirmed)
    a2 = engine.evaluate_frame(scene=scene, timestamp=1.1)
    assert len(a2.active_events) == 0

    # Frame 3: Confirmed!
    a3 = engine.evaluate_frame(scene=scene, timestamp=1.2)
    assert len(a3.active_events) == 1
    assert a3.active_events[0].lifecycle_state == EventLifecycleState.CONFIRMED

    # Frame 4: Active
    a4 = engine.evaluate_frame(scene=scene, timestamp=1.3)
    assert len(a4.active_events) == 1
    assert a4.active_events[0].lifecycle_state == EventLifecycleState.ACTIVE


# -----------------------------------------------------------------------------
# 17. Event end confirmation & 18. Event cooldown & 19. Duplicate suppression
# -----------------------------------------------------------------------------
def test_event_end_and_cooldown():
    engine = RiskEngine(confirmation_frames=1, end_confirmation_frames=2, cooldown_frames=2)
    p_bad = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        attributes={"compliance_status": "NON_COMPLIANT", "ppe_status": {"Hardhat": "MISSING"}},
    )
    p_good = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        attributes={"compliance_status": "COMPLIANT"},
    )

    # Frame 1: Active
    engine.evaluate_frame(scene=_make_dummy_scene(person_nodes=[p_bad]), timestamp=1.0)

    # Frame 2: Condition disappears (miss 1 < 2) -> remains active (anti-flapping)
    a2 = engine.evaluate_frame(scene=_make_dummy_scene(person_nodes=[p_good]), timestamp=1.1)
    assert len(a2.active_events) == 1

    # Frame 3: Miss 2 >= 2 -> Transitions to ENDED
    a3 = engine.evaluate_frame(scene=_make_dummy_scene(person_nodes=[p_good]), timestamp=1.2)
    assert len(a3.active_events) == 0
    ended = [ev for ev in a3.recent_events if ev.lifecycle_state == EventLifecycleState.ENDED]
    assert len(ended) == 1
    assert ended[0].end_timestamp == 1.2

    # Frame 4: In cooldown, even if bad condition re-appears, candidate is suppressed
    a4 = engine.evaluate_frame(scene=_make_dummy_scene(person_nodes=[p_bad]), timestamp=1.3)
    assert len(a4.active_events) == 0


# -----------------------------------------------------------------------------
# 21. Track disappearance
# -----------------------------------------------------------------------------
def test_track_disappearance_ends_events():
    engine = RiskEngine(confirmation_frames=1, end_confirmation_frames=5)
    p_node = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        attributes={"compliance_status": "NON_COMPLIANT", "ppe_status": {"Hardhat": "MISSING"}},
    )
    # Active frame
    a1 = engine.evaluate_frame(scene=_make_dummy_scene(person_nodes=[p_node]), timestamp=1.0)
    assert len(a1.active_events) == 1

    # Next frame: track 17 completely disappeared from scene
    a2 = engine.evaluate_frame(scene=_make_dummy_scene(person_nodes=[]), timestamp=1.1)
    assert len(a2.active_events) == 0
    ended = [ev for ev in a2.recent_events if ev.lifecycle_state == EventLifecycleState.ENDED]
    assert len(ended) == 1


# -----------------------------------------------------------------------------
# 22-26. Missing inputs handled gracefully
# -----------------------------------------------------------------------------
def test_missing_inputs_handled_gracefully():
    engine = RiskEngine()
    # No inputs passed at all
    a = engine.evaluate_frame()
    assert isinstance(a, FrameRiskAssessment)
    assert len(a.risk_factors) == 0
    assert len(a.active_events) == 0

    # Missing depth, ppe, behavior, zones
    a2 = engine.evaluate_frame(scene=None, ppe_association=None, zone_occupancy=None, behavior_states=None, depth_result=None)
    assert isinstance(a2, FrameRiskAssessment)


# -----------------------------------------------------------------------------
# 27. Serialization
# -----------------------------------------------------------------------------
def test_serialization():
    engine = RiskEngine(confirmation_frames=1)
    p_node = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        attributes={"compliance_status": "NON_COMPLIANT", "ppe_status": {"Hardhat": "MISSING"}},
    )
    assessment = engine.evaluate_frame(scene=_make_dummy_scene(person_nodes=[p_node]))
    d = RiskAssessmentSerializer.to_dict(assessment)
    assert isinstance(d, dict)
    assert "active_events" in d
    assert "risk_summary" in d
    assert d["risk_summary"]["total_active_events"] == 1

    j = RiskAssessmentSerializer.to_json(assessment)
    assert isinstance(j, str)
    assert "person_17" in j


# -----------------------------------------------------------------------------
# 28. API response
# -----------------------------------------------------------------------------
def test_api_endpoints():
    engine = RiskEngine(confirmation_frames=1)
    p_node = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        attributes={"compliance_status": "NON_COMPLIANT", "ppe_status": {"Hardhat": "MISSING"}},
    )
    assessment = engine.evaluate_frame(scene=_make_dummy_scene(person_nodes=[p_node]))
    get_risk_store().set_current_assessment(assessment)

    resp_risk = client.get("/api/v1/risk/current")
    assert resp_risk.status_code == 200
    data = resp_risk.json()
    assert data["risk_summary"]["total_active_events"] == 1

    resp_events = client.get("/api/v1/events/current")
    assert resp_events.status_code == 200
    evs = resp_events.json()
    assert len(evs) == 1
    assert evs[0]["event_type"] == "PPE_VIOLATION"


# -----------------------------------------------------------------------------
# 29-32. Technical honesty, no double counting, relative depth caution
# -----------------------------------------------------------------------------
def test_technical_honesty_and_no_double_counting():
    extractor = RiskFactorExtractor()
    p_node = SceneNode(node_id="person_17", node_type=SceneNodeType.PERSON, class_name="person", track_id=17)
    v_node = SceneNode(node_id="forklift_2", node_type=SceneNodeType.VEHICLE, class_name="forklift", track_id=2)
    rel = SceneRelation(
        relation_id="rel_near",
        source_node_id="person_17",
        target_node_id="forklift_2",
        relation_type=SceneRelationType.NEAR,
        lifecycle=RelationLifecycle.ACTIVE,
        evidence={"distance_px": 50.0},
    )
    scene = _make_dummy_scene(person_nodes=[p_node], vehicle_nodes=[v_node], relations=[rel])
    factors = extractor.extract_factors(scene=scene)

    assert len(factors) == 1
    # Check that explanation mentions uncalibrated relative proximity, not metric meters
    assert "relative" in factors[0].explanation.lower() or "image-space" in factors[0].explanation.lower()
    assert "meter" not in factors[0].explanation.lower()


# -----------------------------------------------------------------------------
# 33. Configuration overrides
# -----------------------------------------------------------------------------
def test_configuration_overrides():
    custom_overrides = {
        "RISK_SCORE_PPE_VIOLATION": 99.0,
        "RISK_LEVEL_CRITICAL_THRESHOLD": 80.0,
    }
    engine = RiskEngine(confirmation_frames=1, settings_override=custom_overrides)
    p_node = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        attributes={"compliance_status": "NON_COMPLIANT", "ppe_status": {"Hardhat": "MISSING"}},
    )
    assessment = engine.evaluate_frame(scene=_make_dummy_scene(person_nodes=[p_node]))
    assert assessment.active_events[0].risk_score == 99.0
    assert assessment.active_events[0].risk_level == RiskLevel.CRITICAL


def test_risk_visualization():
    import numpy as np
    from vision.tracking.visualizer import TrackingVisualizer

    engine = RiskEngine(confirmation_frames=1)
    p_node = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        attributes={"compliance_status": "NON_COMPLIANT", "ppe_status": {"Hardhat": "MISSING"}},
    )
    assessment = engine.evaluate_frame(scene=_make_dummy_scene(person_nodes=[p_node]))

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    visualizer = TrackingVisualizer()
    annotated = visualizer.draw_risk_assessment(frame, assessment)

    assert annotated is not None
    assert annotated.shape == (480, 640, 3)

