"""
tests/test_scene_graph.py
Unit tests for Step 9: Scene Graph & Situational Awareness.
Uses synthetic data only - zero external model downloads or GPU required.
Covers all 26 specified test criteria.
"""
import json
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.behavior import BehaviorState, PrimaryBehavior
from backend.schemas.depth import DepthStatistics, ObjectDepth
from backend.schemas.detection import BoundingBox
from backend.schemas.ppe import (
    AssociatedPPEItem,
    ComplianceStatus,
    FramePPEAssociation,
    PPEItemState,
    WorkerPPEInventory,
)
from backend.schemas.scene_graph import (
    FrameScene,
    RelationLifecycle,
    SceneNode,
    SceneNodeType,
    SceneRelation,
    SceneRelationType,
    SituationalSummary,
)
from backend.schemas.tracking import FrameTracks, TrackState, TrackedObject
from backend.schemas.zones import (
    FrameZoneOccupancy,
    RestrictedZone,
    ZoneMembership,
    ZoneMembershipStatus,
    ZoneType,
)
from configs.settings import get_settings
from intelligence.scene_graph.builder import SceneGraphBuilder
from intelligence.scene_graph.graph import SceneGraph
from intelligence.scene_graph.relations import (
    classify_node_type,
    compute_euclidean_distance,
    compute_separation_trend,
    is_depth_consistent,
)


# ---------------------------------------------------------------------------
# Helpers for Synthetic Data Creation
# ---------------------------------------------------------------------------

def _make_track(
    track_id: int,
    class_name: str,
    cx: float,
    cy: float,
    w: float = 40.0,
    h: float = 80.0,
    speed: float = 0.0,
    timestamp: float = 0.0,
    frame_id: int = 1,
) -> TrackedObject:
    hw, hh = w / 2.0, h / 2.0
    return TrackedObject(
        track_id=track_id,
        class_id=0 if class_name == "person" else 1,
        class_name=class_name,
        confidence=0.92,
        bbox=BoundingBox(x1=cx - hw, y1=cy - hh, x2=cx + hw, y2=cy + hh),
        state=TrackState.ACTIVE,
        frame_index=frame_id,
        timestamp=timestamp,
        centroid_x=cx,
        centroid_y=cy,
        speed_pixels_per_second=speed,
    )


def _make_frame_tracks(tracks: list[TrackedObject], timestamp: float = 0.0, frame_id: int = 1) -> FrameTracks:
    return FrameTracks(
        frame_id=frame_id,
        timestamp=timestamp,
        active_tracks=tracks,
        total_tracked_count=len(tracks),
    )


def _make_ppe_inv(track_id: int, has_helmet: bool = True, has_vest: bool = True) -> WorkerPPEInventory:
    items = []
    status = {}
    if has_helmet:
        items.append(
            AssociatedPPEItem(
                class_id=1,
                class_name="Hardhat",
                confidence=0.88,
                bbox=BoundingBox(x1=90, y1=90, x2=110, y2=110),
                association_score=0.91,
                body_region="head",
            )
        )
        status["Hardhat"] = PPEItemState.PRESENT
    else:
        status["Hardhat"] = PPEItemState.MISSING

    if has_vest:
        items.append(
            AssociatedPPEItem(
                class_id=2,
                class_name="Safety Vest",
                confidence=0.85,
                bbox=BoundingBox(x1=90, y1=115, x2=110, y2=145),
                association_score=0.89,
                body_region="torso",
            )
        )
        status["Safety Vest"] = PPEItemState.PRESENT
    else:
        status["Safety Vest"] = PPEItemState.MISSING

    compliance = ComplianceStatus.COMPLIANT if (has_helmet and has_vest) else ComplianceStatus.NON_COMPLIANT
    return WorkerPPEInventory(
        track_id=track_id,
        timestamp=0.0,
        bbox=BoundingBox(x1=80, y1=90, x2=120, y2=170),
        items=items,
        ppe_status=status,
        compliance_status=compliance,
    )


def _make_zone(zone_id: str = "zone_high_voltage") -> RestrictedZone:
    return RestrictedZone(
        zone_id=zone_id,
        name="High Voltage Switchgear",
        zone_type=ZoneType.RESTRICTED,
        polygon=[[50.0, 50.0], [200.0, 50.0], [200.0, 200.0], [50.0, 200.0]],
        enabled=True,
    )


def _make_zone_membership(track_id: int, zone_id: str, is_inside: bool = True) -> ZoneMembership:
    return ZoneMembership(
        track_id=track_id,
        zone_id=zone_id,
        zone_name="High Voltage Switchgear",
        is_inside=is_inside,
        status=ZoneMembershipStatus.INSIDE if is_inside else ZoneMembershipStatus.OUTSIDE,
        contact_point=(100.0, 100.0),
        dwell_seconds=3.5 if is_inside else 0.0,
    )


# ---------------------------------------------------------------------------
# Unit Test Suite
# ---------------------------------------------------------------------------

class TestSceneGraphCore:

    def test_01_node_creation(self):
        node = SceneNode(
            node_id="person_1",
            node_type=SceneNodeType.PERSON,
            class_name="person",
            track_id=1,
            confidence=0.95,
            bbox=BoundingBox(x1=10, y1=20, x2=50, y2=100),
            centroid=(30.0, 60.0),
            contact_point=(30.0, 100.0),
            timestamp=1.0,
        )
        assert node.node_id == "person_1"
        assert node.node_type == SceneNodeType.PERSON
        assert node.contact_point == (30.0, 100.0)

    def test_02_tracked_person_node_identity(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        t = _make_track(track_id=17, class_name="person", cx=100.0, cy=100.0)
        scene = builder.build(frame_tracks=_make_frame_tracks([t]))
        node = scene.get_node("person_17")
        assert node is not None
        assert node.track_id == 17
        assert node.node_type == SceneNodeType.PERSON

    def test_03_zone_node_creation(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        zone = _make_zone("zone_assembly")
        scene = builder.build(
            frame_tracks=_make_frame_tracks([]),
            configured_zones=[zone],
        )
        znode = scene.get_node("zone_assembly")
        assert znode is not None
        assert znode.node_type == SceneNodeType.ZONE
        assert znode.attributes["name"] == "High Voltage Switchgear"

    def test_04_ppe_relationship(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        t = _make_track(track_id=1, class_name="person", cx=100.0, cy=100.0)
        ppe_inv = _make_ppe_inv(track_id=1, has_helmet=True, has_vest=True)
        ppe_assoc = FramePPEAssociation(frame_id=1, timestamp=0.0, worker_inventories=[ppe_inv])

        scene = builder.build(
            frame_tracks=_make_frame_tracks([t]),
            ppe_association=ppe_assoc,
        )
        wearing_rels = [r for r in scene.relationships if r.relation_type == SceneRelationType.WEARING]
        assert len(wearing_rels) == 2
        assert all(r.source_node_id == "person_1" for r in wearing_rels)
        # Check evidence contains association score
        assert "association_score" in wearing_rels[0].evidence

    def test_05_zone_relationship(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        t = _make_track(track_id=1, class_name="person", cx=100.0, cy=100.0)
        zone = _make_zone("hv_01")
        mem = _make_zone_membership(track_id=1, zone_id="hv_01", is_inside=True)
        zo = FrameZoneOccupancy(frame_id=1, timestamp=0.0, memberships=[mem])

        scene = builder.build(
            frame_tracks=_make_frame_tracks([t]),
            configured_zones=[zone],
            zone_occupancy=zo,
        )
        inside_rels = [r for r in scene.relationships if r.relation_type == SceneRelationType.INSIDE]
        assert len(inside_rels) == 1
        assert inside_rels[0].source_node_id == "person_1"
        assert inside_rels[0].target_node_id == "zone_hv_01"
        assert inside_rels[0].evidence["status"] == "inside"

    def test_06_behavior_relationship(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        t = _make_track(track_id=1, class_name="person", cx=100.0, cy=100.0)
        b = BehaviorState(
            track_id=1,
            timestamp=0.0,
            frame_id=1,
            primary_behavior=PrimaryBehavior.RAPID_MOVEMENT,
            image_speed_px_per_s=280.0,
        )
        scene = builder.build(
            frame_tracks=_make_frame_tracks([t]),
            behavior_states=[b],
        )
        assert scene.active_behaviors.get("person_1") == "RAPID_MOVEMENT"
        b_rels = [r for r in scene.relationships if r.relation_type == SceneRelationType.BEHAVIOR]
        assert len(b_rels) == 1
        assert b_rels[0].evidence["primary_behavior"] == "RAPID_MOVEMENT"

    def test_07_near_relationship(self):
        # Two entities separated by 80px (< threshold 150px)
        builder = SceneGraphBuilder(near_distance_threshold=150.0, confirmation_frames=1)
        t1 = _make_track(track_id=1, class_name="person", cx=100.0, cy=100.0)
        t2 = _make_track(track_id=2, class_name="forklift", cx=150.0, cy=100.0)

        scene = builder.build(frame_tracks=_make_frame_tracks([t1, t2]))
        near_rels = [r for r in scene.relationships if r.relation_type == SceneRelationType.NEAR]
        assert len(near_rels) == 1
        assert near_rels[0].evidence["distance_px"] <= 150.0

    def test_08_far_relationship(self):
        # Two entities separated by 500px (> threshold 400px)
        builder = SceneGraphBuilder(far_distance_threshold=400.0, confirmation_frames=1)
        t1 = _make_track(track_id=1, class_name="person", cx=100.0, cy=100.0)
        t2 = _make_track(track_id=2, class_name="forklift", cx=650.0, cy=100.0)

        scene = builder.build(frame_tracks=_make_frame_tracks([t1, t2]))
        far_rels = [r for r in scene.relationships if r.relation_type == SceneRelationType.FAR]
        assert len(far_rels) == 1
        assert far_rels[0].evidence["distance_px"] >= 400.0

    def test_09_relationship_confirmation(self):
        # Requires 3 consecutive frames before NEAR confirms
        builder = SceneGraphBuilder(near_distance_threshold=150.0, confirmation_frames=3)
        t1 = _make_track(track_id=1, class_name="person", cx=100.0, cy=100.0)
        t2 = _make_track(track_id=2, class_name="forklift", cx=150.0, cy=100.0)

        s1 = builder.build(_make_frame_tracks([t1, t2], timestamp=0.0, frame_id=1))
        assert len([r for r in s1.relationships if r.relation_type == SceneRelationType.NEAR]) == 0

        s2 = builder.build(_make_frame_tracks([t1, t2], timestamp=0.1, frame_id=2))
        assert len([r for r in s2.relationships if r.relation_type == SceneRelationType.NEAR]) == 0

        s3 = builder.build(_make_frame_tracks([t1, t2], timestamp=0.2, frame_id=3))
        near = [r for r in s3.relationships if r.relation_type == SceneRelationType.NEAR]
        assert len(near) == 1
        assert near[0].lifecycle == RelationLifecycle.CREATED

    def test_10_relationship_deconfirmation(self):
        # Confirmed relationship ends after end_confirmation_frames (2)
        builder = SceneGraphBuilder(
            near_distance_threshold=150.0,
            confirmation_frames=1,
            end_confirmation_frames=2,
        )
        t1 = _make_track(track_id=1, class_name="person", cx=100.0, cy=100.0)
        t2_near = _make_track(track_id=2, class_name="forklift", cx=150.0, cy=100.0)
        t2_far = _make_track(track_id=2, class_name="forklift", cx=600.0, cy=100.0)

        # Frame 1: NEAR confirmed
        s1 = builder.build(_make_frame_tracks([t1, t2_near], timestamp=0.0, frame_id=1))
        assert len([r for r in s1.relationships if r.relation_type == SceneRelationType.NEAR]) == 1

        # Frame 2: Miss 1 (still within persistence window)
        s2 = builder.build(_make_frame_tracks([t1, t2_far], timestamp=0.1, frame_id=2))
        near_s2 = [r for r in s2.relationships if r.relation_type == SceneRelationType.NEAR]
        assert len(near_s2) == 1
        assert near_s2[0].lifecycle == RelationLifecycle.ACTIVE

        # Frame 3: Miss 2 (crosses threshold -> ENDED)
        s3 = builder.build(_make_frame_tracks([t1, t2_far], timestamp=0.2, frame_id=3))
        ended_near = [
            r for r in s3.relationships
            if r.relation_type == SceneRelationType.NEAR and r.lifecycle == RelationLifecycle.ENDED
        ]
        assert len(ended_near) == 1

    def test_11_relationship_lifecycle(self):
        builder = SceneGraphBuilder(near_distance_threshold=150.0, confirmation_frames=1, end_confirmation_frames=1)
        t1 = _make_track(1, "person", 100.0, 100.0)
        t2 = _make_track(2, "forklift", 150.0, 100.0)

        # 1. CREATED
        s1 = builder.build(_make_frame_tracks([t1, t2], 0.0, 1))
        r1 = [r for r in s1.relationships if r.relation_type == SceneRelationType.NEAR][0]
        assert r1.lifecycle == RelationLifecycle.CREATED

        # 2. ACTIVE
        s2 = builder.build(_make_frame_tracks([t1, t2], 0.1, 2))
        r2 = [r for r in s2.relationships if r.relation_type == SceneRelationType.NEAR][0]
        assert r2.lifecycle == RelationLifecycle.ACTIVE

        # 3. ENDED
        t2_far = _make_track(2, "forklift", 800.0, 100.0)
        s3 = builder.build(_make_frame_tracks([t1, t2_far], 0.2, 3))
        r3 = [r for r in s3.relationships if r.relation_type == SceneRelationType.NEAR][0]
        assert r3.lifecycle == RelationLifecycle.ENDED

    def test_12_duplicate_relationship_prevention(self):
        builder = SceneGraphBuilder(near_distance_threshold=150.0, confirmation_frames=1)
        t1 = _make_track(1, "person", 100.0, 100.0)
        t2 = _make_track(2, "forklift", 150.0, 100.0)

        scene = builder.build(_make_frame_tracks([t1, t2]))
        rel_ids = [r.relation_id for r in scene.relationships]
        assert len(rel_ids) == len(set(rel_ids))

    def test_13_approaching_relation(self):
        builder = SceneGraphBuilder(
            near_distance_threshold=200.0,
            confirmation_frames=3,
            approaching_enabled=True,
        )
        # Distance decreases: 180px -> 130px -> 80px (rate ~500px/s)
        positions = [180.0, 130.0, 80.0]
        for idx, pos in enumerate(positions):
            t1 = _make_track(1, "person", 0.0, 100.0)
            t2 = _make_track(2, "forklift", pos, 100.0)
            scene = builder.build(_make_frame_tracks([t1, t2], timestamp=idx * 0.1, frame_id=idx + 1))

        app_rels = [r for r in scene.relationships if r.relation_type == SceneRelationType.APPROACHING]
        assert len(app_rels) == 1
        assert app_rels[0].evidence["trend"] == "APPROACHING"

    def test_14_moving_away_relation(self):
        builder = SceneGraphBuilder(
            near_distance_threshold=200.0,
            confirmation_frames=3,
            approaching_enabled=True,
        )
        # Distance increases: 50px -> 100px -> 160px
        positions = [50.0, 100.0, 160.0]
        for idx, pos in enumerate(positions):
            t1 = _make_track(1, "person", 0.0, 100.0)
            t2 = _make_track(2, "forklift", pos, 100.0)
            scene = builder.build(_make_frame_tracks([t1, t2], timestamp=idx * 0.1, frame_id=idx + 1))

        away_rels = [r for r in scene.relationships if r.relation_type == SceneRelationType.MOVING_AWAY]
        assert len(away_rels) == 1
        assert away_rels[0].evidence["trend"] == "MOVING_AWAY"

    def test_15_multiple_people(self):
        builder = SceneGraphBuilder(near_distance_threshold=150.0, confirmation_frames=1)
        p1 = _make_track(1, "person", 100.0, 100.0)
        p2 = _make_track(2, "person", 140.0, 100.0)
        p3 = _make_track(3, "person", 500.0, 100.0)

        scene = builder.build(_make_frame_tracks([p1, p2, p3]))
        assert scene.summary.workers_count == 3
        # p1 and p2 should be NEAR
        near_rels = [r for r in scene.relationships if r.relation_type == SceneRelationType.NEAR]
        assert len(near_rels) == 1
        assert near_rels[0].source_node_id == "person_1"
        assert near_rels[0].target_node_id == "person_2"

    def test_16_multiple_objects(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        p1 = _make_track(1, "person", 100.0, 100.0)
        f1 = _make_track(10, "forklift", 200.0, 100.0)
        m1 = _make_track(20, "machinery", 300.0, 100.0)

        scene = builder.build(_make_frame_tracks([p1, f1, m1]))
        assert scene.summary.workers_count == 1
        assert scene.summary.vehicles_count == 1
        assert scene.summary.machines_count == 1

    def test_17_multiple_zones(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        z1 = _make_zone("zone_1")
        z2 = _make_zone("zone_2")
        scene = builder.build(_make_frame_tracks([]), configured_zones=[z1, z2])
        assert scene.summary.active_zones_count == 2
        assert "zone_1" in scene.active_zones
        assert "zone_2" in scene.active_zones

    def test_18_missing_ppe_data(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        t = _make_track(1, "person", 100.0, 100.0)
        scene = builder.build(_make_frame_tracks([t]), ppe_association=None)
        assert len(scene.nodes) == 1
        assert "ppe_status" not in scene.nodes[0].attributes

    def test_19_missing_zone_data(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        t = _make_track(1, "person", 100.0, 100.0)
        scene = builder.build(_make_frame_tracks([t]), zone_occupancy=None, configured_zones=None)
        assert scene.summary.occupied_zones_count == 0
        assert len(scene.active_zones) == 0

    def test_20_missing_behavior_data(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        t = _make_track(1, "person", 100.0, 100.0)
        scene = builder.build(_make_frame_tracks([t]), behavior_states=None)
        assert len(scene.active_behaviors) == 0

    def test_21_optional_depth_data(self):
        # Depth gating: two objects close in 2D (50px) but depth differs substantially (0.1 vs 0.7)
        builder = SceneGraphBuilder(
            near_distance_threshold=150.0,
            confirmation_frames=1,
            depth_relation_enabled=True,
            depth_near_threshold=0.20,
        )
        t1 = _make_track(1, "person", 100.0, 100.0)
        t2 = _make_track(2, "forklift", 150.0, 100.0)

        od1 = ObjectDepth(
            track_id=1,
            bbox=t1.bbox,
            depth_stats=DepthStatistics(min_depth=0.1, max_depth=0.15, mean_depth=0.12, median_depth=0.12, percentile_25=0.11, percentile_75=0.13),
            contact_depth=0.10,
        )
        od2 = ObjectDepth(
            track_id=2,
            bbox=t2.bbox,
            depth_stats=DepthStatistics(min_depth=0.65, max_depth=0.75, mean_depth=0.70, median_depth=0.70, percentile_25=0.68, percentile_75=0.72),
            contact_depth=0.70,
        )

        scene = builder.build(_make_frame_tracks([t1, t2]), object_depths=[od1, od2])
        # Depth difference |0.10 - 0.70| = 0.60 > 0.20 -> NEAR must NOT be triggered
        near_rels = [r for r in scene.relationships if r.relation_type == SceneRelationType.NEAR]
        assert len(near_rels) == 0

    def test_22_no_depth_fallback(self):
        # Without depth data, 2D proximity is used gracefully
        builder = SceneGraphBuilder(
            near_distance_threshold=150.0,
            confirmation_frames=1,
            depth_relation_enabled=True,
        )
        t1 = _make_track(1, "person", 100.0, 100.0)
        t2 = _make_track(2, "forklift", 150.0, 100.0)
        scene = builder.build(_make_frame_tracks([t1, t2]), object_depths=None)
        near_rels = [r for r in scene.relationships if r.relation_type == SceneRelationType.NEAR]
        assert len(near_rels) == 1

    def test_23_scene_summary(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        p1 = _make_track(1, "person", 100.0, 100.0)
        f1 = _make_track(2, "forklift", 120.0, 100.0)
        ppe_inv = _make_ppe_inv(1, has_helmet=False, has_vest=True)
        assoc = FramePPEAssociation(frame_id=1, timestamp=0.0, worker_inventories=[ppe_inv])
        zone = _make_zone("z1")
        mem = _make_zone_membership(1, "z1", is_inside=True)
        zo = FrameZoneOccupancy(frame_id=1, timestamp=0.0, memberships=[mem])

        scene = builder.build(
            _make_frame_tracks([p1, f1]),
            ppe_association=assoc,
            zone_occupancy=zo,
            configured_zones=[zone],
        )
        s = scene.summary
        assert s.workers_count == 1
        assert s.vehicles_count == 1
        assert s.occupied_zones_count == 1
        assert s.workers_in_restricted_zones == 1
        assert s.workers_non_compliant_ppe == 1
        assert s.active_proximity_relationships == 1

    def test_24_json_serialization(self):
        builder = SceneGraphBuilder(confirmation_frames=1)
        p1 = _make_track(1, "person", 100.0, 100.0)
        scene = builder.build(_make_frame_tracks([p1]))

        # Pydantic dump
        data = scene.model_dump()
        json_str = json.dumps(data)
        loaded = json.loads(json_str)
        assert loaded["summary"]["workers_count"] == 1

        # SceneGraph to_json
        sg = SceneGraph.from_frame_scene(scene)
        sg_json = sg.to_json()
        assert "person_1" in sg_json

    def test_25_track_disappearance(self):
        builder = SceneGraphBuilder(near_distance_threshold=150.0, confirmation_frames=1)
        t1 = _make_track(1, "person", 100.0, 100.0)
        t2 = _make_track(2, "forklift", 150.0, 100.0)

        # Frame 1: both present
        s1 = builder.build(_make_frame_tracks([t1, t2], timestamp=0.0, frame_id=1))
        assert len([r for r in s1.relationships if r.relation_type == SceneRelationType.NEAR]) == 1

        # Frame 2: track 2 disappears
        s2 = builder.build(_make_frame_tracks([t1], timestamp=0.1, frame_id=2))
        ended_rels = [
            r for r in s2.relationships
            if r.relation_type == SceneRelationType.NEAR and r.lifecycle == RelationLifecycle.ENDED
        ]
        assert len(ended_rels) == 1
        assert ended_rels[0].metadata.get("reason") == "entity_departed"

    def test_26_stale_relationship_cleanup(self):
        builder = SceneGraphBuilder(near_distance_threshold=150.0, confirmation_frames=1)
        t1 = _make_track(1, "person", 100.0, 100.0)
        t2 = _make_track(2, "forklift", 150.0, 100.0)

        builder.build(_make_frame_tracks([t1, t2], timestamp=0.0, frame_id=1))
        assert len(builder._active_relations) >= 1

        # Track disappears
        builder.build(_make_frame_tracks([t1], timestamp=0.1, frame_id=2))
        # Active relations should now be cleared
        assert len(builder._active_relations) == 0

    def test_graph_query_api(self):
        builder = SceneGraphBuilder(near_distance_threshold=150.0, confirmation_frames=1)
        p1 = _make_track(1, "person", 100.0, 100.0)
        f1 = _make_track(2, "forklift", 140.0, 100.0)
        zone = _make_zone("hv_01")
        mem = _make_zone_membership(1, "hv_01", is_inside=True)
        zo = FrameZoneOccupancy(frame_id=1, timestamp=0.0, memberships=[mem])

        scene = builder.build(
            _make_frame_tracks([p1, f1]),
            zone_occupancy=zo,
            configured_zones=[zone],
        )
        graph = SceneGraph.from_frame_scene(scene)

        # Query workers in zone
        workers_in_zone = graph.find_workers_in_zone("hv_01")
        assert len(workers_in_zone) == 1
        assert workers_in_zone[0].node_id == "person_1"

        # Query entities near person_1
        near_entities = graph.find_entities_near("person_1")
        assert len(near_entities) == 1
        assert near_entities[0][0].node_id == "forklift_2"

        # Query neighbors
        neighbors = graph.get_neighbors("person_1")
        neighbor_ids = {n.node_id for n in neighbors}
        assert "forklift_2" in neighbor_ids
        assert "zone_hv_01" in neighbor_ids

    def test_api_scene_current_endpoint(self):
        client = TestClient(app)
        # Query GET /api/v1/scene/current
        resp = client.get("/api/v1/scene/current")
        assert resp.status_code == 200
        data = resp.json()
        assert "nodes" in data
        assert "relationships" in data
        assert "summary" in data

    def test_visualizer_draw_scene_graph(self):
        import numpy as np
        from vision.tracking.visualizer import TrackingVisualizer

        builder = SceneGraphBuilder(near_distance_threshold=150.0, confirmation_frames=1)
        p1 = _make_track(1, "person", 100.0, 100.0)
        f1 = _make_track(2, "forklift", 140.0, 100.0)
        scene = builder.build(_make_frame_tracks([p1, f1]))

        vis = TrackingVisualizer()
        dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        annotated = vis.draw_scene_graph(dummy_frame, scene)
        assert annotated.shape == dummy_frame.shape
        assert annotated.dtype == np.uint8
