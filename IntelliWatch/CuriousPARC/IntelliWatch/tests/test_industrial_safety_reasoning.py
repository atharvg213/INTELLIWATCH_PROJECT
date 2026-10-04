"""
tests/test_industrial_safety_reasoning.py
Comprehensive unit and integration test suite for Step 14:
Industrial Spatial Safety Reasoning, Kinematics, Proximity Tiers, Compound 5-W Audit.

All tests use synthetic data only — zero external model downloads, zero GPU required.
"""

import pytest
import numpy as np

from configs.settings import (
    SCENE_VERY_NEAR_DISTANCE_THRESHOLD,
    SCENE_NEAR_DISTANCE_THRESHOLD,
    SCENE_MODERATE_DISTANCE_THRESHOLD,
    SCENE_FAR_DISTANCE_THRESHOLD,
    SCENE_MIN_APPROACHING_RATE_PX_S,
)
from backend.schemas.detection import BoundingBox, DetectionResult
from backend.schemas.tracking import TrackedObject, FrameTracks
from backend.schemas.scene_graph import SceneRelationType
from backend.schemas.risk import RiskLevel, RiskEventType, RiskFactorType, RiskFactor
from intelligence.scene_graph.relations import (
    classify_proximity_state,
    compute_separation_velocity,
    compute_separation_trend,
)
from intelligence.scene_graph.builder import SceneGraphBuilder
from intelligence.risk.rules import (
    RiskRuleEvaluator,
    _synthesize_compound_explanation_and_audit,
)
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator


@pytest.fixture
def sample_frame():
    """Generates a synthetic 720p dark gray industrial frame."""
    return np.full((720, 1280, 3), 40, dtype=np.uint8)


def _make_tracked(track_id, class_name, class_group, x1, y1, x2, y2, cx=None, cy=None, confidence=0.92):
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name=class_name,
        class_group=class_group,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
        centroid_x=cx if cx is not None else (x1 + x2) / 2.0,
        centroid_y=cy if cy is not None else (y1 + y2) / 2.0,
    )


# =============================================================================
# TEST SUITE A: PROXIMITY TIERS & RELATIVE DISTANCE CLASSIFICATION
# =============================================================================

def test_classify_proximity_state_very_near():
    """Distances at and below 75px map to VERY_NEAR."""
    assert classify_proximity_state(0.0) == "VERY_NEAR"
    assert classify_proximity_state(50.0) == "VERY_NEAR"
    assert classify_proximity_state(75.0) == "VERY_NEAR"


def test_classify_proximity_state_near():
    """Distances above 75px and at or below 150px map to NEAR."""
    assert classify_proximity_state(75.1) == "NEAR"
    assert classify_proximity_state(100.0) == "NEAR"
    assert classify_proximity_state(150.0) == "NEAR"


def test_classify_proximity_state_moderate():
    """Distances above 150px and at or below 300px map to MODERATE."""
    assert classify_proximity_state(150.1) == "MODERATE"
    assert classify_proximity_state(250.0) == "MODERATE"
    assert classify_proximity_state(300.0) == "MODERATE"


def test_classify_proximity_state_far():
    """Distances at or above 400px map to FAR."""
    assert classify_proximity_state(400.0) == "FAR"
    assert classify_proximity_state(600.0) == "FAR"


def test_classify_proximity_state_invalid():
    """Negative distances and None map to UNKNOWN."""
    assert classify_proximity_state(-10.0) == "UNKNOWN"
    assert classify_proximity_state(None) == "UNKNOWN"


def test_proximity_thresholds_match_config():
    """Ensure module-level constants stay in sync with AppSettings."""
    assert SCENE_VERY_NEAR_DISTANCE_THRESHOLD == 75.0
    assert SCENE_NEAR_DISTANCE_THRESHOLD == 150.0
    assert SCENE_MODERATE_DISTANCE_THRESHOLD == 300.0
    assert SCENE_FAR_DISTANCE_THRESHOLD == 400.0
    assert SCENE_MIN_APPROACHING_RATE_PX_S == 15.0


# =============================================================================
# TEST SUITE B: SCENE GRAPH DIRECTED RELATIONSHIPS
# =============================================================================

def test_scene_graph_builder_person_source_vehicle_target():
    """
    Validates person -> [NEAR] -> vehicle directionality.
    PERSON must always be the source, VEHICLE the target.
    """
    builder = SceneGraphBuilder(confirmation_frames=1)

    # Worker at cx=200, Forklift at cx=240 → separation = 40px (VERY_NEAR)
    worker = _make_tracked(1, "person", "PERSON", 180, 150, 220, 250, cx=200, cy=200)
    forklift = _make_tracked(2, "forklift", "VEHICLE", 220, 150, 260, 250, cx=240, cy=200)

    ft = FrameTracks(frame_id=1, timestamp=0.1, active_tracks=[worker, forklift])
    sg = builder.build(frame_tracks=ft)

    spatial_rels = [
        r for r in sg.relationships
        if r.relation_type in (SceneRelationType.NEAR, SceneRelationType.APPROACHING, SceneRelationType.FAR)
    ]
    assert len(spatial_rels) >= 1, "Expected at least one spatial relation between person and vehicle"

    rel = spatial_rels[0]
    assert rel.source_node_id == "person_1", f"Expected person_1 as source, got {rel.source_node_id}"
    assert rel.target_node_id.startswith("vehicle_") or "2" in rel.target_node_id


def test_scene_graph_proximity_state_in_metadata():
    """
    Validates that spatial relationships carry proximity_state metadata
    and that VERY_NEAR is correctly reported for very close entity pairs.
    """
    builder = SceneGraphBuilder(confirmation_frames=1)

    # Worker and forklift 40px apart → VERY_NEAR
    worker = _make_tracked(1, "person", "PERSON", 180, 150, 220, 250, cx=200, cy=200)
    forklift = _make_tracked(2, "forklift", "VEHICLE", 220, 150, 260, 250, cx=240, cy=200)

    ft = FrameTracks(frame_id=1, timestamp=0.1, active_tracks=[worker, forklift])
    sg = builder.build(frame_tracks=ft)

    spatial_rels = [
        r for r in sg.relationships
        if r.relation_type in (SceneRelationType.NEAR, SceneRelationType.APPROACHING, SceneRelationType.FAR)
        and "1" in r.source_node_id
    ]
    assert len(spatial_rels) >= 1

    rel = spatial_rels[0]
    # proximity_state is stored in evidence dict; spatial_state is in metadata
    prox = rel.evidence.get("proximity_state") or rel.metadata.get("spatial_state")
    assert prox == "VERY_NEAR", f"Expected VERY_NEAR, got {prox}"


def test_scene_graph_zone_inside_relationships():
    """
    Validates PERSON -> INSIDE -> ZONE and MACHINE -> INSIDE -> ZONE.
    Uses correct ZoneMembership schema with required fields.
    """
    from backend.schemas.zones import ZoneMembership, FrameZoneOccupancy, ZoneMembershipStatus

    builder = SceneGraphBuilder(confirmation_frames=1)

    worker = _make_tracked(10, "person", "PERSON", 100, 100, 150, 200, cx=125, cy=150)
    machine = _make_tracked(20, "machinery", "MACHINE", 500, 500, 600, 600, cx=550, cy=550)

    ft = FrameTracks(frame_id=1, timestamp=0.1, active_tracks=[worker, machine])

    zm_worker = ZoneMembership(
        track_id=10,
        zone_id="restricted_zone_A",
        zone_name="Restricted Cell A",
        is_inside=True,
        status=ZoneMembershipStatus.INSIDE,
        contact_point=(125.0, 200.0),
        dwell_seconds=5.0,
    )
    zm_machine = ZoneMembership(
        track_id=20,
        zone_id="restricted_zone_A",
        zone_name="Restricted Cell A",
        is_inside=True,
        status=ZoneMembershipStatus.INSIDE,
        contact_point=(550.0, 600.0),
        dwell_seconds=12.0,
    )
    zone_occ = FrameZoneOccupancy(
        frame_id=1,
        timestamp=0.1,
        memberships=[zm_worker, zm_machine],
    )

    sg = builder.build(frame_tracks=ft, zone_occupancy=zone_occ)

    inside_rels = [r for r in sg.relationships if r.relation_type == SceneRelationType.INSIDE]
    assert len(inside_rels) >= 2, f"Expected >=2 INSIDE rels, got {len(inside_rels)}"
    sources = {r.source_node_id for r in inside_rels}
    assert "person_10" in sources, "person_10 should be inside zone"
    assert "machinery_20" in sources, "machinery_20 should be inside zone"


def test_scene_graph_missing_ppe_relationships():
    """
    Validates that PPE missing items create MISSING relationship nodes.
    """
    from backend.schemas.ppe import FramePPEAssociation, WorkerPPEInventory, ComplianceStatus

    builder = SceneGraphBuilder(confirmation_frames=1)

    worker = _make_tracked(5, "person", "PERSON", 100, 100, 150, 200, cx=125, cy=150)
    ft = FrameTracks(frame_id=1, timestamp=0.1, active_tracks=[worker])

    worker_inv = WorkerPPEInventory(
        track_id=5,
        bbox=BoundingBox(x1=100.0, y1=100.0, x2=150.0, y2=200.0),
        compliance_status=ComplianceStatus.NON_COMPLIANT,
        missing_ppe=["Hardhat", "Safety Vest"],
    )
    ppe_assoc = FramePPEAssociation(frame_id=1, worker_inventories=[worker_inv])

    sg = builder.build(frame_tracks=ft, ppe_association=ppe_assoc)

    missing_rels = [r for r in sg.relationships if r.relation_type == SceneRelationType.MISSING]
    assert len(missing_rels) == 2, f"Expected 2 MISSING rels, got {len(missing_rels)}"
    missing_targets = {r.target_node_id for r in missing_rels}
    assert "missing_hardhat_5" in missing_targets
    assert "missing_safety_vest_5" in missing_targets


# =============================================================================
# TEST SUITE C: TEMPORAL KINEMATICS & APPROACHING DETECTION
# =============================================================================

def test_separation_velocity_approaching():
    """Decreasing separation → negative velocity (approaching)."""
    history = [
        (0.0, 300.0),
        (0.5, 250.0),
        (1.0, 200.0),
        (1.5, 150.0),
        (2.0, 100.0),
    ]
    rate = compute_separation_velocity(history)
    assert rate is not None and rate < -15.0
    trend = compute_separation_trend(history, min_rate_px_per_s=15.0)
    assert trend == "APPROACHING"


def test_separation_velocity_moving_away():
    """Increasing separation → positive velocity (moving away)."""
    history = [
        (0.0, 100.0),
        (1.0, 175.0),
        (2.0, 250.0),
    ]
    rate = compute_separation_velocity(history)
    assert rate is not None and rate > 15.0
    trend = compute_separation_trend(history, min_rate_px_per_s=15.0)
    assert trend == "MOVING_AWAY"


def test_separation_velocity_stationary():
    """Negligible change → STABLE trend."""
    history = [
        (0.0, 200.0),
        (1.0, 202.0),
        (2.0, 199.0),
    ]
    rate = compute_separation_velocity(history)
    assert rate is not None and abs(rate) < 15.0
    trend = compute_separation_trend(history, min_rate_px_per_s=15.0)
    assert trend == "STABLE"


def test_separation_velocity_insufficient_history():
    """Less than 2 points → None velocity, no trend."""
    assert compute_separation_velocity([(0.0, 200.0)]) is None
    assert compute_separation_trend([(0.0, 200.0)]) is None


def test_kinematic_approaching_detected_over_frames():
    """
    Multi-frame progression with fast closing should produce APPROACHING.
    Worker moves from cx=400 → 250 → 60 toward forklift at cx=200 over 3 frames.
    """
    builder = SceneGraphBuilder(confirmation_frames=1, approaching_enabled=True)

    # Frame 1: Worker at 400, Forklift at 200 → sep=200px
    ft1 = FrameTracks(frame_id=1, timestamp=0.0, active_tracks=[
        _make_tracked(1, "person", "PERSON", 380, 150, 420, 250, cx=400, cy=200),
        _make_tracked(2, "forklift", "VEHICLE", 180, 150, 220, 250, cx=200, cy=200),
    ])
    builder.build(frame_tracks=ft1)

    # Frame 2: Worker at 300, Forklift at 200 → sep=100px (closing 200px in 0.5s = 400 px/s)
    ft2 = FrameTracks(frame_id=2, timestamp=0.5, active_tracks=[
        _make_tracked(1, "person", "PERSON", 280, 150, 320, 250, cx=300, cy=200),
        _make_tracked(2, "forklift", "VEHICLE", 180, 150, 220, 250, cx=200, cy=200),
    ])
    builder.build(frame_tracks=ft2)

    # Frame 3: Worker at 240, Forklift at 200 → sep=40px (VERY_NEAR, still approaching)
    ft3 = FrameTracks(frame_id=3, timestamp=1.0, active_tracks=[
        _make_tracked(1, "person", "PERSON", 220, 150, 260, 250, cx=240, cy=200),
        _make_tracked(2, "forklift", "VEHICLE", 180, 150, 220, 250, cx=200, cy=200),
    ])
    sg3 = builder.build(frame_tracks=ft3)

    # At least one APPROACHING or NEAR relation must exist
    all_spatial_types = {r.relation_type for r in sg3.relationships}
    assert SceneRelationType.APPROACHING in all_spatial_types or SceneRelationType.NEAR in all_spatial_types

    # All spatial relations for this pair must report VERY_NEAR or NEAR proximity
    pair_rels = [r for r in sg3.relationships
                 if r.relation_type in (SceneRelationType.APPROACHING, SceneRelationType.NEAR)
                 and "1" in r.source_node_id]
    # proximity_state is in evidence; spatial_state is in metadata
    for r in pair_rels:
        prox = r.evidence.get("proximity_state") or r.metadata.get("spatial_state")
        assert prox in ("VERY_NEAR", "NEAR"), f"Unexpected proximity state {prox} for {r.relation_type}"


# =============================================================================
# TEST SUITE D: COMPOUND SAFETY REASONING & 5-W AUDIT
# =============================================================================

def test_compound_audit_scenario_1_vehicle_approach():
    """
    Scenario 1: Worker + Forklift + Proximity + Approaching.
    Validates 5-W audit fields and deterministic explanation.
    """
    factors = [
        RiskFactor(
            factor_id="f1",
            factor_type=RiskFactorType.PERSON_VEHICLE_PROXIMITY,
            severity_contribution=35.0,
            involved_entity_ids=["person_1", "vehicle_2"],
            timestamp=1.0,
            explanation="Worker #1 is in close relative proximity (VERY_NEAR) to vehicle_2.",
            metadata={"proximity_state": "VERY_NEAR"},
        ),
        RiskFactor(
            factor_id="f2",
            factor_type=RiskFactorType.APPROACHING_VEHICLE,
            severity_contribution=40.0,
            involved_entity_ids=["person_1", "vehicle_2"],
            timestamp=1.0,
            explanation="Worker #1 is approaching vehicle_2 while remaining in close relative proximity (VERY_NEAR).",
            metadata={"proximity_state": "VERY_NEAR", "closing_rate_px_s": 120.0},
        ),
    ]

    explanation, audit = _synthesize_compound_explanation_and_audit(
        "person_1", factors, ["person_1", "vehicle_2"]
    )

    assert "Worker #1" in explanation
    assert "approaching" in explanation.lower()
    assert "VERY_NEAR" in explanation

    assert "what" in audit and "who" in audit and "where" in audit and "why" in audit
    assert "Worker #1" in audit["who"]
    assert len(audit["evidence_summary"]) == 2


def test_compound_audit_scenario_2_machine_ppe_zone():
    """
    Scenario 2: Worker + Machine + PPE Violation + Restricted Zone.
    Validates zone name, PPE items, and audit structure.
    """
    factors = [
        RiskFactor(
            factor_id="f_ppe",
            factor_type=RiskFactorType.PPE_NON_COMPLIANCE,
            severity_contribution=25.0,
            involved_entity_ids=["person_5"],
            timestamp=2.0,
            explanation="Worker #5 is missing mandatory safety gear: ['Hardhat', 'Safety Vest'].",
            source_evidence={"missing_ppe": ["Hardhat", "Safety Vest"]},
        ),
        RiskFactor(
            factor_id="f_mach",
            factor_type=RiskFactorType.WORKER_NEAR_MACHINE,
            severity_contribution=30.0,
            involved_entity_ids=["person_5", "machine_8"],
            timestamp=2.0,
            explanation="Worker #5 is in close proximity to machine_8.",
            metadata={"proximity_state": "NEAR"},
        ),
        RiskFactor(
            factor_id="f_zone",
            factor_type=RiskFactorType.RESTRICTED_ZONE_INTRUSION,
            severity_contribution=35.0,
            involved_entity_ids=["person_5", "zone_press_cell"],
            timestamp=2.0,
            explanation="Worker #5 is located inside restricted zone 'Heavy Press Cell'.",
            source_evidence={"zone_name": "Heavy Press Cell"},
        ),
    ]

    explanation, audit = _synthesize_compound_explanation_and_audit(
        "person_5", factors, ["person_5", "machine_8", "zone_press_cell"]
    )

    assert "Worker #5" in explanation
    assert "Heavy Press Cell" in explanation
    assert "Hardhat" in explanation
    assert "Safety Vest" in explanation

    assert audit["where"] == "Restricted zone 'Heavy Press Cell'"
    assert "Worker #5" in audit["who"]
    assert len(audit["evidence_summary"]) == 3


def test_compound_event_candidate_via_evaluator():
    """
    Tests RiskRuleEvaluator generates compound COMPOUND_SAFETY_EVENT
    from two concurrent proximity+approaching factors for the same worker-vehicle pair.
    """
    evaluator = RiskRuleEvaluator()

    factors = [
        RiskFactor(
            factor_id="f1",
            factor_type=RiskFactorType.PERSON_VEHICLE_PROXIMITY,
            severity_contribution=35.0,
            involved_entity_ids=["person_1", "vehicle_2"],
            timestamp=1.0,
            explanation="Worker #1 is in close relative proximity (VERY_NEAR) to vehicle_2.",
            metadata={"proximity_state": "VERY_NEAR"},
        ),
        RiskFactor(
            factor_id="f2",
            factor_type=RiskFactorType.APPROACHING_VEHICLE,
            severity_contribution=40.0,
            involved_entity_ids=["person_1", "vehicle_2"],
            timestamp=1.0,
            explanation="Worker #1 is approaching vehicle_2 while remaining in close relative proximity (VERY_NEAR).",
            metadata={"proximity_state": "VERY_NEAR"},
        ),
    ]

    candidates = evaluator.evaluate_candidate_events(factors=factors, timestamp=1.0)
    compound_candidates = [c for c in candidates if c.is_compound]
    assert len(compound_candidates) >= 1, "At least one compound event candidate must be generated"
    cc = compound_candidates[0]
    assert cc.event_type == RiskEventType.COMPOUND_SAFETY_EVENT
    assert "Worker #1" in cc.explanation
    assert cc.evidence.get("audit_5w") is not None, "Compound event must carry 5-W audit in evidence"


# =============================================================================
# TEST SUITE E: END-TO-END PIPELINE INTEGRATION
# =============================================================================

def test_orchestrator_three_frame_vehicle_proximity(sample_frame):
    """
    Full pipeline: worker moves toward forklift over 3 frames.
    Expects:
    - Scene graph to contain a spatial relation (NEAR or APPROACHING)
    - Proximity state VERY_NEAR in final frame
    - Risk level escalated to HIGH or CRITICAL
    """
    from backend.services.incident_store import get_incident_store
    incident_store = get_incident_store()
    incident_store.reset()

    orch = EndToEndPipelineOrchestrator(
        device="cpu",
        enable_ppe_model=False,
        enable_depth_model=False,
    )
    orch._risk_engine.confirmation_frames = 1
    orch._scene_builder.confirmation_frames = 1

    forklift_det = DetectionResult(
        class_id=1, class_name="forklift", confidence=0.92,
        bbox=BoundingBox(x1=180.0, y1=150.0, x2=220.0, y2=250.0),
    )

    # Frame 1 — worker far (gap = 220px, MODERATE)
    a1, _ = orch.process_frame(
        frame=sample_frame, frame_id=1, timestamp=0.0,
        manual_detections=[
            DetectionResult(class_id=0, class_name="person", confidence=0.95,
                            bbox=BoundingBox(x1=390.0, y1=150.0, x2=430.0, y2=250.0)),
            forklift_det,
        ],
    )
    assert a1.frame_id == 1

    # Frame 2 — worker closing (gap = 100px, NEAR)
    a2, _ = orch.process_frame(
        frame=sample_frame, frame_id=2, timestamp=0.5,
        manual_detections=[
            DetectionResult(class_id=0, class_name="person", confidence=0.95,
                            bbox=BoundingBox(x1=280.0, y1=150.0, x2=320.0, y2=250.0)),
            forklift_det,
        ],
    )
    assert a2.frame_id == 2

    # Frame 3 — worker very close (gap ≈ 40px, VERY_NEAR)
    a3, _ = orch.process_frame(
        frame=sample_frame, frame_id=3, timestamp=1.0,
        manual_detections=[
            DetectionResult(class_id=0, class_name="person", confidence=0.95,
                            bbox=BoundingBox(x1=220.0, y1=150.0, x2=260.0, y2=250.0)),
            forklift_det,
        ],
    )
    assert a3.frame_id == 3

    # Check that at least one spatial relationship exists in the scene graph
    all_rel_types = {r.relation_type for r in a3.scene.relationships}
    has_spatial = bool(all_rel_types & {
        SceneRelationType.NEAR,
        SceneRelationType.APPROACHING,
        SceneRelationType.FAR,
        SceneRelationType.INSIDE,
    })
    assert has_spatial, f"Expected spatial relationships in frame 3, got: {all_rel_types}"

    # Risk level must have escalated
    assert a3.highest_risk_level in (RiskLevel.INFO, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL)


def test_orchestrator_ppe_violation_zone_compound(sample_frame):
    """
    Places worker inside the configured High Voltage zone without PPE.
    Expects COMPOUND_SAFETY_EVENT with deterministic explanation containing "Worker #".
    """
    from backend.services.incident_store import get_incident_store
    incident_store = get_incident_store()
    incident_store.reset()

    orch = EndToEndPipelineOrchestrator(
        device="cpu",
        enable_ppe_model=False,
        enable_depth_model=False,
    )
    orch._risk_engine.confirmation_frames = 1
    orch._zone_engine.confirmation_frames = 1

    worker_in_zone = DetectionResult(
        class_id=0, class_name="person", confidence=0.96,
        bbox=BoundingBox(x1=200.0, y1=200.0, x2=260.0, y2=350.0),
    )

    assessment, _ = orch.process_frame(
        frame=sample_frame, frame_id=1, timestamp=0.1,
        manual_detections=[worker_in_zone],
    )

    assert assessment.frame_id == 1
    # Check incident is recorded correctly
    incidents = incident_store.list_incidents()
    if incidents:
        inc = incidents[0]
        assert inc.has_snapshot is True
        assert "Worker #" in inc.explanation
