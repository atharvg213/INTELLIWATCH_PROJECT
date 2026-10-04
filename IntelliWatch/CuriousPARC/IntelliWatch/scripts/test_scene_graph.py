"""
scripts/test_scene_graph.py
Integration script for Step 9: Scene Graph & Situational Awareness.
Simulates a multi-frame scenario with:
- Person #1 (with Helmet and Safety Vest)
- Person #2
- Forklift #1
- Machine #1
- Restricted Zone A

Progression:
  Frame 1 (t=0.0s): Person #1 outside Zone A, far from Forklift #1
  Frame 2 (t=0.1s): Person #1 approaches Zone A and Forklift #1
  Frame 3 (t=0.2s): Person #1 enters Zone A, near Forklift #1 (APPROACHING)
  Frame 4 (t=0.3s): Person #1 confirmed inside Zone A, NEAR Forklift #1, exhibits RAPID_MOVEMENT
  Frame 5 (t=0.4s): Person #1 moves away from Forklift #1 (MOVING_AWAY)
  Frame 6 (t=0.5s): Person #1 exits Zone A, distance exceeds FAR threshold

Prints:
  - Active nodes and properties
  - Relationships with lifecycles (CREATED, ACTIVE, ENDED)
  - Situational awareness summary
  - CPU construction benchmark
  - Verifies at least 5 assertions.
"""
import sys
import time
from pathlib import Path

# Add project root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.schemas.behavior import BehaviorState, PrimaryBehavior
from backend.schemas.detection import BoundingBox
from backend.schemas.ppe import (
    AssociatedPPEItem,
    ComplianceStatus,
    FramePPEAssociation,
    PPEItemState,
    WorkerPPEInventory,
)
from backend.schemas.scene_graph import (
    RelationLifecycle,
    SceneNodeType,
    SceneRelationType,
)
from backend.schemas.tracking import FrameTracks, TrackState, TrackedObject
from backend.schemas.zones import (
    FrameZoneOccupancy,
    RestrictedZone,
    ZoneMembership,
    ZoneMembershipStatus,
    ZoneType,
)
from intelligence.scene_graph.builder import SceneGraphBuilder
from intelligence.scene_graph.graph import SceneGraph


def make_box(cx: float, cy: float, w: float, h: float) -> BoundingBox:
    hw, hh = w / 2.0, h / 2.0
    return BoundingBox(x1=cx - hw, y1=cy - hh, x2=cx + hw, y2=cy + hh)


def run_integration_test():
    print("=" * 70)
    print("INTELLIWATCH - STEP 9: SCENE GRAPH & SITUATIONAL AWARENESS")
    print("Multi-Frame Synthetic Scenario Integration Test")
    print("=" * 70)

    # 1. Initialize SceneGraphBuilder
    builder = SceneGraphBuilder(
        near_distance_threshold=150.0,
        far_distance_threshold=350.0,
        confirmation_frames=1,       # 1 frame for test scenario responsiveness
        end_confirmation_frames=1,   # immediate lifecycle transition for demo clarity
        approaching_enabled=True,
    )

    # 2. Define static Restricted Zone A
    zone_a = RestrictedZone(
        zone_id="restricted_zone_a",
        name="Automated Cell A",
        zone_type=ZoneType.RESTRICTED,
        polygon=[[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]],
        enabled=True,
    )

    # Person 1 PPE: Compliant with Helmet & Safety Vest
    p1_ppe = WorkerPPEInventory(
        track_id=1,
        timestamp=0.0,
        bbox=make_box(50.0, 50.0, 40.0, 80.0),
        items=[
            AssociatedPPEItem(
                class_id=1,
                class_name="Hardhat",
                confidence=0.91,
                bbox=make_box(50.0, 30.0, 20.0, 20.0),
                association_score=0.94,
                body_region="head",
            ),
            AssociatedPPEItem(
                class_id=2,
                class_name="Safety Vest",
                confidence=0.88,
                bbox=make_box(50.0, 60.0, 30.0, 35.0),
                association_score=0.91,
                body_region="torso",
            ),
        ],
        ppe_status={"Hardhat": PPEItemState.PRESENT, "Safety Vest": PPEItemState.PRESENT},
        compliance_status=ComplianceStatus.COMPLIANT,
    )
    ppe_frame = FramePPEAssociation(frame_id=1, timestamp=0.0, worker_inventories=[p1_ppe])

    # Simulation progression data
    # (timestamp, p1_x, p1_y, in_zone, p1_behavior)
    frames_data = [
        (0.0, 50.0, 50.0, False, PrimaryBehavior.MOVING),               # Frame 1: outside, far (d=400)
        (0.1, 100.0, 100.0, False, PrimaryBehavior.MOVING),             # Frame 2: approaching zone
        (0.2, 180.0, 180.0, True, PrimaryBehavior.MOVING),              # Frame 3: inside zone, near (d=120)
        (0.3, 200.0, 200.0, True, PrimaryBehavior.RAPID_MOVEMENT),      # Frame 4: inside, rapid movement, near
        (0.4, 280.0, 280.0, True, PrimaryBehavior.MOVING),              # Frame 5: moving away (d=240)
        (0.5, 450.0, 450.0, False, PrimaryBehavior.MOVING),             # Frame 6: outside zone, far
    ]

    forklift_x, forklift_y = 200.0, 200.0
    machine_x, machine_y = 500.0, 200.0
    person2_x, person2_y = 50.0, 400.0

    collected_scenes = []
    total_build_time_ms = 0.0

    # Assertion verification flags
    observed_person_near_forklift = False
    observed_person_inside_zone = False
    observed_wearing_relations = False
    observed_rapid_movement = False
    observed_relation_ended = False

    for idx, (ts, p1_x, p1_y, in_zone, p1_behave) in enumerate(frames_data):
        frame_id = idx + 1
        t_start = time.perf_counter()

        # Build tracks
        p1_track = TrackedObject(
            track_id=1, class_id=0, class_name="person", confidence=0.95,
            bbox=make_box(p1_x, p1_y, 40.0, 80.0), state=TrackState.ACTIVE,
            frame_index=frame_id, timestamp=ts, centroid_x=p1_x, centroid_y=p1_y,
            speed_pixels_per_second=150.0 if p1_behave == PrimaryBehavior.RAPID_MOVEMENT else 40.0,
        )
        p2_track = TrackedObject(
            track_id=2, class_id=0, class_name="person", confidence=0.90,
            bbox=make_box(person2_x, person2_y, 40.0, 80.0), state=TrackState.ACTIVE,
            frame_index=frame_id, timestamp=ts, centroid_x=person2_x, centroid_y=person2_y,
            speed_pixels_per_second=0.0,
        )
        forklift_track = TrackedObject(
            track_id=10, class_id=1, class_name="forklift", confidence=0.89,
            bbox=make_box(forklift_x, forklift_y, 70.0, 70.0), state=TrackState.ACTIVE,
            frame_index=frame_id, timestamp=ts, centroid_x=forklift_x, centroid_y=forklift_y,
            speed_pixels_per_second=0.0,
        )
        machine_track = TrackedObject(
            track_id=20, class_id=2, class_name="machinery", confidence=0.92,
            bbox=make_box(machine_x, machine_y, 90.0, 90.0), state=TrackState.ACTIVE,
            frame_index=frame_id, timestamp=ts, centroid_x=machine_x, centroid_y=machine_y,
            speed_pixels_per_second=0.0,
        )

        tracks = FrameTracks(
            frame_id=frame_id, timestamp=ts,
            active_tracks=[p1_track, p2_track, forklift_track, machine_track],
            total_tracked_count=4,
        )

        # Zone occupancy
        zone_mem = ZoneMembership(
            track_id=1, zone_id=zone_a.zone_id, zone_name=zone_a.name,
            is_inside=in_zone,
            status=ZoneMembershipStatus.INSIDE if in_zone else ZoneMembershipStatus.OUTSIDE,
            contact_point=(p1_x, p1_y + 40.0),
            dwell_seconds=0.1 * idx if in_zone else 0.0,
        )
        zone_occ = FrameZoneOccupancy(frame_id=frame_id, timestamp=ts, memberships=[zone_mem])

        # Behavior
        p1_bstate = BehaviorState(
            track_id=1, timestamp=ts, frame_id=frame_id,
            primary_behavior=p1_behave,
            image_speed_px_per_s=250.0 if p1_behave == PrimaryBehavior.RAPID_MOVEMENT else 40.0,
        )

        # Build Scene Graph
        scene = builder.build(
            frame_tracks=tracks,
            ppe_association=ppe_frame,
            zone_occupancy=zone_occ,
            behavior_states=[p1_bstate],
            configured_zones=[zone_a],
        )

        t_elapsed = (time.perf_counter() - t_start) * 1000.0
        total_build_time_ms += t_elapsed
        collected_scenes.append(scene)

        # Evaluate assertion triggers
        for r in scene.relationships:
            if r.relation_type == SceneRelationType.NEAR and "person_1" in (r.source_node_id, r.target_node_id) and "forklift_10" in (r.source_node_id, r.target_node_id):
                observed_person_near_forklift = True
            if r.relation_type == SceneRelationType.INSIDE and r.source_node_id == "person_1" and r.target_node_id == "zone_restricted_zone_a":
                observed_person_inside_zone = True
            if r.relation_type == SceneRelationType.WEARING and r.source_node_id == "person_1":
                observed_wearing_relations = True
            if r.lifecycle == RelationLifecycle.ENDED:
                observed_relation_ended = True

        if scene.summary.rapid_movement_count > 0:
            observed_rapid_movement = True

        # Print per-frame snapshot
        print(f"\n--- Frame {frame_id:02d} (t={ts:.1f}s) | Latency: {t_elapsed:.2f}ms ---")
        print(f"  Nodes ({len(scene.nodes)}): " + ", ".join([f"{n.node_id} [{n.node_type.value}]" for n in scene.nodes]))
        print(f"  Relationships ({len(scene.relationships)}):")
        for r in scene.relationships:
            print(f"    - {r.source_node_id} --({r.relation_type.value} | {r.lifecycle.value})--> {r.target_node_id} | evidence: {r.evidence}")
        s = scene.summary
        print(f"  Summary: Workers={s.workers_count}, Vehicles={s.vehicles_count}, Machines={s.machines_count}, "
              f"Zones Occupied={s.occupied_zones_count}, Proximity Rels={s.active_proximity_relationships}, "
              f"Total Active={s.total_active_relationships}")

    avg_time_ms = total_build_time_ms / len(frames_data)

    print("\n" + "=" * 70)
    print("PERFORMANCE & CPU BENCHMARK SUMMARY")
    print("=" * 70)
    print(f"  Frames processed        : {len(frames_data)}")
    print(f"  Total processing time   : {total_build_time_ms:.2f} ms")
    print(f"  Average per frame       : {avg_time_ms:.2f} ms/frame")
    print(f"  Estimated throughput   : {1000.0 / max(0.01, avg_time_ms):.1f} FPS (CPU only, zero GPU required)")

    print("\n" + "=" * 70)
    print("INTEGRATION ASSERTIONS")
    print("=" * 70)

    assertions = [
        ("Person 1 WEARING relations confirmed (Helmet & Safety Vest)", observed_wearing_relations),
        ("Person 1 INSIDE Restricted Zone A confirmed", observed_person_inside_zone),
        ("Person 1 NEAR Forklift 1 spatial proximity confirmed", observed_person_near_forklift),
        ("Person 1 RAPID_MOVEMENT situational behavior tracked", observed_rapid_movement),
        ("Relationship lifecycle ENDED observed on departure/exit", observed_relation_ended),
        ("Graph query API resolves neighbors and zone occupants", len(SceneGraph.from_frame_scene(collected_scenes[2]).find_workers_in_zone("restricted_zone_a")) == 1),
    ]

    passed_count = 0
    for desc, passed in assertions:
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {desc}")
        if passed:
            passed_count += 1

    print("-" * 70)
    print(f"Integration results: {passed_count}/{len(assertions)} checks passed.")

    if passed_count == len(assertions):
        print("\nStep 9 integration test: ALL CHECKS PASSED.")
        return 0
    else:
        print("\nStep 9 integration test: FAILURES DETECTED.")
        return 1


if __name__ == "__main__":
    sys.exit(run_integration_test())
