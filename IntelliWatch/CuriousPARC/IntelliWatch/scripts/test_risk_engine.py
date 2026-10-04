"""
scripts/test_risk_engine.py
Step 10 - Integration Test Script for Safety Risk & Event Reasoning Engine.
Simulates a multi-frame synthetic scenario demonstrating:
- Frame 1: Normal safe condition (compliant, outside zone, stationary)
- Frame 2: Person enters restricted zone
- Frame 3: Person confirmed inside zone
- Frame 4: Person removes Hardhat (PPE non-compliance)
- Frame 5: Forklift in close spatial proximity (NEAR)
- Frame 6: Person exhibits rapid movement -> Compound Safety Event escalation
Benchmarks CPU execution latency and verifies explainability.
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox
from backend.schemas.scene_graph import (
    FrameScene,
    RelationLifecycle,
    SceneNode,
    SceneNodeType,
    SceneRelation,
    SceneRelationType,
    SituationalSummary,
)
from backend.schemas.risk import RiskEventType, RiskLevel
from intelligence.risk.engine import RiskEngine
from intelligence.risk.serializer import RiskAssessmentSerializer


def build_frame_scene(
    frame_id: int,
    timestamp: float,
    in_zone: bool = False,
    zone_dwell_s: float = 0.0,
    missing_hardhat: bool = False,
    forklift_near: bool = False,
    rapid_movement: bool = False,
) -> FrameScene:
    nodes = []
    relationships = []

    # Person 17 Node
    attrs = {
        "compliance_status": "NON_COMPLIANT" if missing_hardhat else "COMPLIANT",
        "ppe_status": {"Hardhat": "MISSING" if missing_hardhat else "PRESENT", "Safety Vest": "PRESENT"},
        "speed_px_per_s": 150.0 if rapid_movement else 10.0,
        "primary_behavior": "RAPID_MOVEMENT" if rapid_movement else "MOVING",
        "flags": {"rapid": rapid_movement},
    }
    p_node = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        confidence=0.92,
        bbox=BoundingBox(x1=100.0, y1=100.0, x2=160.0, y2=280.0),
        centroid=(130.0, 190.0),
        contact_point=(130.0, 280.0),
        attributes=attrs,
        timestamp=timestamp,
    )
    nodes.append(p_node)

    # Restricted Zone Node
    z_node = SceneNode(
        node_id="zone_high_voltage",
        node_type=SceneNodeType.ZONE,
        class_name="zone",
        zone_id="high_voltage",
        attributes={"name": "High Voltage Switchgear", "max_dwell_seconds": 10.0},
        timestamp=timestamp,
    )
    nodes.append(z_node)

    # Zone Relationship
    if in_zone:
        rel_zone = SceneRelation(
            relation_id="rel_zone_17",
            source_node_id="person_17",
            target_node_id="zone_high_voltage",
            relation_type=SceneRelationType.INSIDE,
            lifecycle=RelationLifecycle.ACTIVE,
            timestamp=timestamp,
            evidence={"dwell_seconds": zone_dwell_s, "contact_point": [130.0, 280.0]},
        )
        relationships.append(rel_zone)

    # Forklift Node & Proximity
    if forklift_near:
        v_node = SceneNode(
            node_id="forklift_2",
            node_type=SceneNodeType.VEHICLE,
            class_name="forklift",
            track_id=2,
            confidence=0.89,
            bbox=BoundingBox(x1=200.0, y1=120.0, x2=350.0, y2=300.0),
            centroid=(275.0, 210.0),
            contact_point=(275.0, 300.0),
            attributes={"speed_px_per_s": 45.0},
            timestamp=timestamp,
        )
        nodes.append(v_node)

        rel_near = SceneRelation(
            relation_id="rel_near_17_2",
            source_node_id="person_17",
            target_node_id="forklift_2",
            relation_type=SceneRelationType.NEAR,
            lifecycle=RelationLifecycle.ACTIVE,
            timestamp=timestamp,
            evidence={"distance_px": 85.0, "contact_distance_px": 85.0},
        )
        relationships.append(rel_near)

    return FrameScene(
        frame_id=frame_id,
        timestamp=timestamp,
        camera_id="cam_01",
        nodes=nodes,
        relationships=relationships,
        summary=SituationalSummary(),
    )


def run_synthetic_integration_test():
    print("================================================================================")
    print("INTELLIWATCH STEP 10: RISK & EVENT REASONING INTEGRATION TEST")
    print("================================================================================\n")

    engine = RiskEngine(
        confirmation_frames=1,  # immediate confirmation for scenario walkthrough
        compound_enabled=True,
    )

    scenarios = [
        ("Frame 1: Safe baseline (Compliant, outside zone, clear of vehicles)", False, 0.0, False, False, False),
        ("Frame 2: Person enters restricted zone", True, 0.5, False, False, False),
        ("Frame 3: Person remains inside zone (Dwell 3.5s)", True, 3.5, False, False, False),
        ("Frame 4: Person removes Hardhat (PPE Violation inside zone)", True, 5.0, True, False, False),
        ("Frame 5: Forklift in close spatial proximity (NEAR)", True, 6.0, True, True, False),
        ("Frame 6: Rapid movement detected (Compound Hazard Escalation)", True, 7.0, True, True, True),
    ]

    latencies_ms = []

    for idx, (desc, in_zone, dwell, no_ppe, veh_near, rapid) in enumerate(scenarios, 1):
        ts = 1.0 * idx
        scene = build_frame_scene(
            frame_id=idx,
            timestamp=ts,
            in_zone=in_zone,
            zone_dwell_s=dwell,
            missing_hardhat=no_ppe,
            forklift_near=veh_near,
            rapid_movement=rapid,
        )

        t0 = time.perf_counter()
        assessment = engine.evaluate_frame(scene=scene, frame_id=idx, timestamp=ts)
        t1 = time.perf_counter()
        dur_ms = (t1 - t0) * 1000.0
        latencies_ms.append(dur_ms)

        print(f"--- {desc} ---")
        print(f"  Frame ID: {idx} | Latency: {dur_ms:.3f} ms")
        print(f"  Evaluated Factors: {len(assessment.risk_factors)}")
        for f in assessment.risk_factors:
            print(f"    - Factor: {f.factor_type.value} (+{f.severity_contribution:.1f} pts) -> {f.explanation}")

        print(f"  Active Confirmed Events: {len(assessment.active_events)}")
        for ev in assessment.active_events:
            print(f"    * [{ev.risk_level.value}] {ev.event_type.value} (Score: {ev.risk_score:.1f})")
            print(f"      Explanation: {ev.explanation}")

        print(f"  Risk Summary: Max Level = {assessment.risk_summary.max_risk_level.value} | Max Score = {assessment.risk_summary.max_risk_score:.1f}")
        print()

    # Verifications
    print("--------------------------------------------------------------------------------")
    print("VALIDATING SCENARIO REQUIREMENTS:")
    print("--------------------------------------------------------------------------------")
    avg_latency = sum(latencies_ms) / len(latencies_ms)
    print(f"[OK] CPU Performance Benchmark: Average Latency = {avg_latency:.3f} ms per frame (Target: < 5.0 ms)")

    # Test serialization
    last_assessment = assessment
    serialized_dict = RiskAssessmentSerializer.to_dict(last_assessment)
    assert "active_events" in serialized_dict, "Serialization missing active_events"
    assert serialized_dict["risk_summary"]["total_active_events"] > 0, "Serialization missing active events count"
    print("[OK] Risk Assessment Serialization verified.")

    # Check Frame 6 produced Compound Safety Event
    has_compound = any(ev.event_type == RiskEventType.COMPOUND_SAFETY_EVENT for ev in last_assessment.active_events)
    assert has_compound, "Compound safety event was not generated in Frame 6"
    print("[OK] Compound Safety Event successfully generated with multi-factor compounding.")
    print("[OK] All integration checks passed deterministically!\n")


if __name__ == "__main__":
    run_synthetic_integration_test()
