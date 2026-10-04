"""
intelligence/scene_graph/builder.py
Constructs deterministic, factual SceneGraph and FrameScene representations.
Consumes outputs from:
- ByteTrack (Step 4)
- PPE Association (Step 5B)
- Restricted Zone Engine (Step 6)
- Monocular Depth (Step 7)
- Behavior & Temporal Engine (Step 8)
Maintains temporal continuity, relationship lifecycles, and anti-flapping confirmation.
"""
from collections import deque
import logging
import math
from typing import Any, Dict, List, Optional, Set, Tuple

from backend.schemas.behavior import BehaviorState, PrimaryBehavior
from backend.schemas.depth import ObjectDepth
from backend.schemas.ppe import ComplianceStatus, FramePPEAssociation
from backend.schemas.scene_graph import (
    FrameScene,
    RelationLifecycle,
    SceneNode,
    SceneNodeType,
    SceneRelation,
    SceneRelationType,
    SituationalSummary,
)
from backend.schemas.tracking import FrameTracks, TrackedObject
from backend.schemas.zones import FrameZoneOccupancy, RestrictedZone
from backend.services.scene_store import get_scene_store
from configs.settings import get_settings
from intelligence.scene_graph.graph import SceneGraph
from intelligence.scene_graph.relations import (
    classify_node_type,
    classify_proximity_state,
    compute_euclidean_distance,
    compute_separation_trend,
    compute_separation_velocity,
    is_depth_consistent,
)

logger = logging.getLogger("intelliwatch.scene_graph.builder")


class _ActiveRelationState:
    """Internal tracker for an active or pending relationship."""

    def __init__(
        self,
        relation_id: str,
        source_id: str,
        target_id: str,
        relation_type: SceneRelationType,
        first_seen_ts: float,
    ):
        self.relation_id = relation_id
        self.source_id = source_id
        self.target_id = target_id
        self.relation_type = relation_type
        self.first_seen_ts = first_seen_ts
        self.last_seen_ts = first_seen_ts
        self.confirmed = False
        self.consecutive_hits = 0
        self.consecutive_misses = 0
        self.evidence: Dict[str, Any] = {}
        self.metadata: Dict[str, Any] = {}


class SceneGraphBuilder:
    """
    Stateful builder that converts multimodal perception outputs into a structured
    SceneGraph / FrameScene snapshot with temporally verified relationships.
    """

    def __init__(
        self,
        near_distance_threshold: Optional[float] = None,
        far_distance_threshold: Optional[float] = None,
        very_near_distance_threshold: Optional[float] = None,
        moderate_distance_threshold: Optional[float] = None,
        confirmation_frames: Optional[int] = None,
        end_confirmation_frames: Optional[int] = None,
        approaching_enabled: Optional[bool] = None,
        depth_relation_enabled: Optional[bool] = None,
        depth_near_threshold: Optional[float] = None,
        camera_id: str = "cam_01",
    ):
        s = get_settings()
        self.very_near_distance_threshold = (
            very_near_distance_threshold if very_near_distance_threshold is not None else getattr(s, "SCENE_VERY_NEAR_DISTANCE_THRESHOLD", 75.0)
        )
        self.near_distance_threshold = (
            near_distance_threshold if near_distance_threshold is not None else s.SCENE_NEAR_DISTANCE_THRESHOLD
        )
        self.moderate_distance_threshold = (
            moderate_distance_threshold if moderate_distance_threshold is not None else getattr(s, "SCENE_MODERATE_DISTANCE_THRESHOLD", 300.0)
        )
        self.far_distance_threshold = (
            far_distance_threshold if far_distance_threshold is not None else s.SCENE_FAR_DISTANCE_THRESHOLD
        )
        self.min_approaching_rate_px_s = getattr(s, "SCENE_MIN_APPROACHING_RATE_PX_S", 15.0)
        self.confirmation_frames = (
            confirmation_frames if confirmation_frames is not None else s.SCENE_RELATION_CONFIRMATION_FRAMES
        )
        self.end_confirmation_frames = (
            end_confirmation_frames if end_confirmation_frames is not None else s.SCENE_RELATION_END_CONFIRMATION_FRAMES
        )
        self.approaching_enabled = (
            approaching_enabled if approaching_enabled is not None else s.SCENE_APPROACHING_ENABLED
        )
        self.depth_relation_enabled = (
            depth_relation_enabled if depth_relation_enabled is not None else s.SCENE_DEPTH_RELATION_ENABLED
        )
        self.depth_near_threshold = (
            depth_near_threshold if depth_near_threshold is not None else s.SCENE_DEPTH_NEAR_THRESHOLD
        )
        self.camera_id = camera_id

        # Internal state
        self._active_relations: Dict[str, _ActiveRelationState] = {}
        # Separation history: (id_a, id_b) -> deque of (timestamp, distance_px)
        self._separation_history: Dict[Tuple[str, str], deque] = {}
        self._prev_active_track_ids: Set[int] = set()

    def build(
        self,
        frame_tracks: FrameTracks,
        ppe_association: Optional[FramePPEAssociation] = None,
        zone_occupancy: Optional[FrameZoneOccupancy] = None,
        behavior_states: Optional[List[BehaviorState]] = None,
        object_depths: Optional[List[ObjectDepth]] = None,
        configured_zones: Optional[List[RestrictedZone]] = None,
    ) -> FrameScene:
        """
        Constructs the FrameScene for the given frame observations.
        """
        timestamp = frame_tracks.timestamp
        frame_id = frame_tracks.frame_id

        # 1. Build lookup tables for optional multimodal inputs
        ppe_by_track: Dict[int, Any] = {}
        if ppe_association:
            inventories = getattr(ppe_association, "worker_inventories", None) or getattr(ppe_association, "inventories", [])
            for inv in inventories:
                ppe_by_track[inv.track_id] = inv

        behavior_by_track: Dict[int, BehaviorState] = {}
        if behavior_states:
            for b in behavior_states:
                behavior_by_track[b.track_id] = b

        depth_by_track: Dict[int, float] = {}
        if object_depths:
            for od in object_depths:
                if od.track_id is not None:
                    # Prefer contact depth if present, else median depth
                    val = od.contact_depth if od.contact_depth is not None else od.depth_stats.median_depth
                    depth_by_track[od.track_id] = val

        # 2. Construct SceneNodes
        nodes: List[SceneNode] = []
        node_lookup: Dict[str, SceneNode] = {}
        current_active_track_ids: Set[int] = set()

        for track in frame_tracks.active_tracks:
            tid = track.track_id
            current_active_track_ids.add(tid)
            ntype = classify_node_type(track.class_name, getattr(track, "class_group", None))
            node_id = f"{track.class_name.lower().replace(' ', '_')}_{tid}"

            # Use the bottom-center image point as the estimated ground contact
            # for physical tracked entities. Non-ground semantic objects retain
            # their centroid and are not treated as ground contacts.
            bbox = track.bbox
            if ntype in (SceneNodeType.PERSON, SceneNodeType.VEHICLE, SceneNodeType.MACHINE, SceneNodeType.OBJECT):
                contact_pt = ((bbox.x1 + bbox.x2) / 2.0, bbox.y2)
            else:
                contact_pt = (track.centroid_x or (bbox.x1 + bbox.x2) / 2.0, track.centroid_y or (bbox.y1 + bbox.y2) / 2.0)

            speed_px_s = getattr(track, "speed_pixels_per_second", None)
            trajectory = getattr(track, "trajectory", None) or []
            motion_observed = len(trajectory) >= 2
            attrs: Dict[str, Any] = {
                "speed_px_per_s": speed_px_s,
                "motion_observed": motion_observed,
            }

            # Tracker velocity_x/y are pixel displacements per frame. Preserve
            # their direction but scale that unit vector by the tracked px/s
            # magnitude before handing it to the homography integration.
            raw_vx = getattr(track, "velocity_x", None)
            raw_vy = getattr(track, "velocity_y", None)
            try:
                raw_vx = float(raw_vx)
                raw_vy = float(raw_vy)
                speed_value = float(speed_px_s)
                direction_magnitude = math.hypot(raw_vx, raw_vy)
                if motion_observed and math.isfinite(speed_value) and speed_value > 0.0 and direction_magnitude > 1e-8:
                    attrs["velocity_x_px_per_s"] = raw_vx / direction_magnitude * speed_value
                    attrs["velocity_y_px_per_s"] = raw_vy / direction_magnitude * speed_value
                elif motion_observed and math.isfinite(speed_value) and speed_value == 0.0:
                    attrs["velocity_x_px_per_s"] = 0.0
                    attrs["velocity_y_px_per_s"] = 0.0
            except (TypeError, ValueError):
                pass

            # Enrich with Depth
            if tid in depth_by_track:
                attrs["relative_depth"] = depth_by_track[tid]

            # Enrich with PPE
            if tid in ppe_by_track:
                inv = ppe_by_track[tid]
                attrs["compliance_status"] = inv.compliance_status.value
                attrs["ppe_status"] = {k: v.value for k, v in inv.ppe_status.items()}

            # Enrich with Behavior
            if tid in behavior_by_track:
                b = behavior_by_track[tid]
                attrs["primary_behavior"] = b.primary_behavior.value
                attrs["stationary_duration_s"] = b.stationary_duration_s
                attrs["flags"] = {
                    "rapid": b.flag_rapid_movement,
                    "sudden": b.flag_sudden_movement,
                    "fall": b.flag_possible_fall,
                    "dir_change": b.flag_direction_change,
                }

            node = SceneNode(
                node_id=node_id,
                node_type=ntype,
                class_name=track.class_name,
                track_id=tid,
                confidence=track.confidence,
                bbox=bbox,
                centroid=(track.centroid_x or (bbox.x1 + bbox.x2) / 2.0, track.centroid_y or (bbox.y1 + bbox.y2) / 2.0),
                contact_point=contact_pt,
                attributes=attrs,
                timestamp=timestamp,
            )
            nodes.append(node)
            node_lookup[node_id] = node

        # 3. Create Zone Nodes
        zone_node_ids: Set[str] = set()
        active_zone_names: List[str] = []
        if configured_zones:
            for z in configured_zones:
                if z.enabled:
                    zid = f"zone_{z.zone_id}" if not z.zone_id.startswith("zone_") else z.zone_id
                    znode = SceneNode(
                        node_id=zid,
                        node_type=SceneNodeType.ZONE,
                        class_name="zone",
                        zone_id=z.zone_id,
                        attributes={"name": z.name, "zone_type": z.zone_type.value, "max_dwell_seconds": z.max_dwell_seconds},
                        timestamp=timestamp,
                    )
                    nodes.append(znode)
                    node_lookup[zid] = znode
                    zone_node_ids.add(zid)
                    active_zone_names.append(z.zone_id)

        # Also register any zone present in zone_occupancy if not already in configured_zones
        if zone_occupancy:
            for m in getattr(zone_occupancy, "memberships", []):
                zid = f"zone_{m.zone_id}" if not m.zone_id.startswith("zone_") else m.zone_id
                if zid not in node_lookup:
                    znode = SceneNode(
                        node_id=zid,
                        node_type=SceneNodeType.ZONE,
                        class_name="zone",
                        zone_id=m.zone_id,
                        attributes={"name": m.zone_name},
                        timestamp=timestamp,
                    )
                    nodes.append(znode)
                    node_lookup[zid] = znode
                    zone_node_ids.add(zid)
                    if m.zone_id not in active_zone_names:
                        active_zone_names.append(m.zone_id)

        # 4. Generate Candidate Relationships for this frame
        # candidate_map: rel_id -> (source_id, target_id, rel_type, evidence, metadata)
        candidate_rels: Dict[str, Tuple[str, str, SceneRelationType, Dict[str, Any], Dict[str, Any]]] = {}

        # A. PERSON -> PPE Nodes & WEARING / MISSING relationships
        if ppe_association:
            inventories = getattr(ppe_association, "worker_inventories", None) or getattr(ppe_association, "inventories", [])
            for inv in inventories:
                worker_node_id = f"person_{inv.track_id}"
                if worker_node_id in node_lookup:
                    # The evaluated inventory is the canonical PPE state for
                    # both Workspace and Scene Graph. A detection can produce a
                    # WEARING edge only when that category resolved PRESENT.
                    ppe_status = getattr(inv, "ppe_status", {}) or {}
                    state_by_item = {
                        str(name).strip().casefold(): str(getattr(state, "value", state)).upper()
                        for name, state in ppe_status.items()
                    }

                    # Positive PPE items worn
                    for idx, ppe_item in enumerate(inv.items):
                        item_state = state_by_item.get(ppe_item.class_name.strip().casefold())
                        if not ppe_item.is_negative and item_state == "PRESENT":
                            ppe_nid = f"ppe_{ppe_item.class_name.lower().replace(' ', '_')}_{inv.track_id}_{idx}"
                            if ppe_nid not in node_lookup:
                                ppe_node = SceneNode(
                                    node_id=ppe_nid,
                                    node_type=SceneNodeType.PPE,
                                    class_name=ppe_item.class_name,
                                    confidence=ppe_item.confidence,
                                    bbox=ppe_item.bbox,
                                    attributes={
                                        "association_score": ppe_item.association_score,
                                        "is_negative": False,
                                        "body_region": ppe_item.body_region,
                                        "ppe_status": item_state,
                                    },
                                    timestamp=timestamp,
                                )
                                nodes.append(ppe_node)
                                node_lookup[ppe_nid] = ppe_node

                            rel_id = f"rel_{worker_node_id}_wearing_{ppe_nid}"
                            candidate_rels[rel_id] = (
                                worker_node_id,
                                ppe_nid,
                                SceneRelationType.WEARING,
                                {
                                    "association_score": ppe_item.association_score,
                                    "body_region": ppe_item.body_region,
                                },
                                {"class_name": ppe_item.class_name},
                            )

                    # Missing items are represented explicitly, never as a
                    # physical PPE node with a WEARING relationship.
                    missing_ppe = [
                        name for name, state in ppe_status.items()
                        if str(getattr(state, "value", state)).upper() == "MISSING"
                    ]
                    for missing_item in missing_ppe:
                        ppe_nid = f"missing_{missing_item.lower().replace(' ', '_')}_{inv.track_id}"
                        if ppe_nid not in node_lookup:
                            ppe_node = SceneNode(
                                node_id=ppe_nid,
                                node_type=SceneNodeType.PPE,
                                class_name=f"Missing {missing_item}",
                                attributes={
                                    "missing": True,
                                    "required_item": missing_item,
                                    "ppe_status": "MISSING",
                                },
                                timestamp=timestamp,
                            )
                            nodes.append(ppe_node)
                            node_lookup[ppe_nid] = ppe_node

                        rel_id = f"rel_{worker_node_id}_missing_{ppe_nid}"
                        candidate_rels[rel_id] = (
                            worker_node_id,
                            ppe_nid,
                            SceneRelationType.MISSING,
                            {"required_item": missing_item},
                            {"class_name": missing_item},
                        )

        # B. ENTITY -> ZONE (INSIDE) relationships (Workers, Vehicles, Machines)
        if zone_occupancy:
            for m in getattr(zone_occupancy, "memberships", []):
                if m.is_inside:
                    actor_node_id = f"person_{m.track_id}"
                    # If not person, could be vehicle or object
                    if actor_node_id not in node_lookup:
                        for n in nodes:
                            if n.track_id == m.track_id:
                                actor_node_id = n.node_id
                                break
                    zid = f"zone_{m.zone_id}" if not m.zone_id.startswith("zone_") else m.zone_id
                    if actor_node_id in node_lookup and zid in node_lookup:
                        rel_id = f"rel_{actor_node_id}_inside_{zid}"
                        candidate_rels[rel_id] = (
                            actor_node_id,
                            zid,
                            SceneRelationType.INSIDE,
                            {
                                "status": m.status.value,
                                "contact_point": list(m.contact_point),
                                "dwell_seconds": m.dwell_seconds,
                            },
                            {"zone_name": m.zone_name},
                        )

        # C. BEHAVIOR relations & tracking
        active_behaviors: Dict[str, str] = {}
        if behavior_states:
            for b in behavior_states:
                actor_id = f"person_{b.track_id}"
                if actor_id not in node_lookup:
                    for n in nodes:
                        if n.track_id == b.track_id:
                            actor_id = n.node_id
                            break
                if actor_id in node_lookup:
                    if b.primary_behavior != PrimaryBehavior.UNKNOWN:
                        active_behaviors[actor_id] = b.primary_behavior.value
                        rel_id = f"rel_{actor_id}_behavior_{b.primary_behavior.value.lower()}"
                        candidate_rels[rel_id] = (
                            actor_id,
                            actor_id,  # self-edge representing behavior state
                            SceneRelationType.BEHAVIOR,
                            {
                                "primary_behavior": b.primary_behavior.value,
                                "speed_px_per_s": b.image_speed_px_per_s,
                                "duration_s": b.behavior_duration_s,
                            },
                            {"flags": [k for k, v in getattr(b, "flags", {}).items() if v]},
                        )

        # D. Spatial Proximity (NEAR / FAR) and Kinematics (APPROACHING / MOVING_AWAY)
        # Compare all pairs of physical tracked entities (nodes with track_id)
        tracked_nodes = [n for n in nodes if n.track_id is not None]
        for i in range(len(tracked_nodes)):
            for j in range(i + 1, len(tracked_nodes)):
                node_a = tracked_nodes[i]
                node_b = tracked_nodes[j]
                if node_a.contact_point is None or node_b.contact_point is None:
                    continue

                dist_px = compute_euclidean_distance(node_a.contact_point, node_b.contact_point)

                # Determine directed source and target
                # If one node is PERSON and the other is VEHICLE or MACHINE, make PERSON the source
                if node_a.node_type == SceneNodeType.PERSON and node_b.node_type in (SceneNodeType.VEHICLE, SceneNodeType.MACHINE, SceneNodeType.OBJECT):
                    src_node, tgt_node = node_a, node_b
                elif node_b.node_type == SceneNodeType.PERSON and node_a.node_type in (SceneNodeType.VEHICLE, SceneNodeType.MACHINE, SceneNodeType.OBJECT):
                    src_node, tgt_node = node_b, node_a
                else:
                    if node_a.node_id <= node_b.node_id:
                        src_node, tgt_node = node_a, node_b
                    else:
                        src_node, tgt_node = node_b, node_a

                # Consistent pair history key (min_id, max_id)
                pair_key = (min(node_a.node_id, node_b.node_id), max(node_a.node_id, node_b.node_id))
                if pair_key not in self._separation_history:
                    self._separation_history[pair_key] = deque(maxlen=10)
                self._separation_history[pair_key].append((timestamp, dist_px))

                # Depth consistency check
                depth_a = depth_by_track.get(src_node.track_id)
                depth_b = depth_by_track.get(tgt_node.track_id)
                depth_ok = True
                if self.depth_relation_enabled:
                    depth_ok = is_depth_consistent(depth_a, depth_b, self.depth_near_threshold)

                # Classify qualitative spatial proximity state
                prox_state = classify_proximity_state(
                    dist_px,
                    very_near_thresh=self.very_near_distance_threshold,
                    near_thresh=self.near_distance_threshold,
                    moderate_thresh=self.moderate_distance_threshold,
                    far_thresh=self.far_distance_threshold,
                )

                # Check NEAR / VERY_NEAR
                if dist_px <= self.near_distance_threshold and depth_ok:
                    rel_id = f"rel_{src_node.node_id}_near_{tgt_node.node_id}"
                    candidate_rels[rel_id] = (
                        src_node.node_id,
                        tgt_node.node_id,
                        SceneRelationType.NEAR,
                        {
                            "distance_px": round(dist_px, 2),
                            "proximity_state": prox_state,
                            "depth_a": depth_a,
                            "depth_b": depth_b,
                            "threshold_px": self.near_distance_threshold,
                            "spatial_explanation": f"Worker is {prox_state.lower().replace('_', ' ')} {tgt_node.class_name} based on relative spatial analysis.",
                        },
                        {"spatial_state": prox_state},
                    )
                elif dist_px >= self.far_distance_threshold:
                    rel_id = f"rel_{src_node.node_id}_far_{tgt_node.node_id}"
                    candidate_rels[rel_id] = (
                        src_node.node_id,
                        tgt_node.node_id,
                        SceneRelationType.FAR,
                        {
                            "distance_px": round(dist_px, 2),
                            "proximity_state": "FAR",
                            "threshold_px": self.far_distance_threshold,
                            "spatial_explanation": f"Worker is far from {tgt_node.class_name}.",
                        },
                        {"spatial_state": "FAR"},
                    )

                # Check APPROACHING / MOVING_AWAY with temporal kinematics
                if self.approaching_enabled:
                    trend = compute_separation_trend(
                        list(self._separation_history[pair_key]),
                        min_samples=self.confirmation_frames,
                        min_rate_px_per_s=self.min_approaching_rate_px_s,
                    )
                    vel = compute_separation_velocity(list(self._separation_history[pair_key]))
                    if trend == "APPROACHING":
                        rel_id = f"rel_{src_node.node_id}_approaching_{tgt_node.node_id}"
                        candidate_rels[rel_id] = (
                            src_node.node_id,
                            tgt_node.node_id,
                            SceneRelationType.APPROACHING,
                            {
                                "distance_px": round(dist_px, 2),
                                "proximity_state": prox_state,
                                "closing_rate_px_s": round(abs(vel), 1) if vel else None,
                                "trend": "APPROACHING",
                                "spatial_explanation": f"{src_node.class_name.capitalize()} is approaching {tgt_node.class_name} (relative spatial state: {prox_state}).",
                            },
                            {"spatial_state": prox_state},
                        )
                    elif trend == "MOVING_AWAY":
                        rel_id = f"rel_{src_node.node_id}_moving_away_{tgt_node.node_id}"
                        candidate_rels[rel_id] = (
                            src_node.node_id,
                            tgt_node.node_id,
                            SceneRelationType.MOVING_AWAY,
                            {
                                "distance_px": round(dist_px, 2),
                                "proximity_state": prox_state,
                                "opening_rate_px_s": round(abs(vel), 1) if vel else None,
                                "trend": "MOVING_AWAY",
                                "spatial_explanation": f"{src_node.class_name.capitalize()} is moving away from {tgt_node.class_name}.",
                            },
                            {"spatial_state": prox_state},
                        )

        # 5. Temporal Confirmation and Relationship Lifecycle Processing
        output_relationships: List[SceneRelation] = []

        # Identify departed tracks and end their relationships immediately
        departed_track_ids = self._prev_active_track_ids - current_active_track_ids
        for rel_id, state in list(self._active_relations.items()):
            src_tid = self._extract_track_id(state.source_id)
            dst_tid = self._extract_track_id(state.target_id)
            if (src_tid in departed_track_ids) or (dst_tid in departed_track_ids):
                # Emit ENDED relation
                ended_rel = SceneRelation(
                    relation_id=rel_id,
                    source_node_id=state.source_id,
                    target_node_id=state.target_id,
                    relation_type=state.relation_type,
                    lifecycle=RelationLifecycle.ENDED,
                    timestamp=timestamp,
                    first_seen_timestamp=state.first_seen_ts,
                    last_seen_timestamp=timestamp,
                    duration_seconds=round(max(0.0, timestamp - state.first_seen_ts), 2),
                    evidence=state.evidence,
                    metadata={"reason": "entity_departed"},
                )
                output_relationships.append(ended_rel)
                del self._active_relations[rel_id]

        # Update observed candidates
        for rel_id, (src, dst, rtype, ev, meta) in candidate_rels.items():
            if rel_id not in self._active_relations:
                # Newly proposed candidate
                state = _ActiveRelationState(
                    relation_id=rel_id,
                    source_id=src,
                    target_id=dst,
                    relation_type=rtype,
                    first_seen_ts=timestamp,
                )
                state.evidence = ev
                state.metadata = meta
                self._active_relations[rel_id] = state

            state = self._active_relations[rel_id]
            state.last_seen_ts = timestamp
            state.consecutive_hits += 1
            state.consecutive_misses = 0
            state.evidence = ev
            state.metadata = meta

            # Certain deterministic relations (WEARING, INSIDE, BEHAVIOR) and already-windowed trends
            # (APPROACHING, MOVING_AWAY) confirm once proposed; pure static proximity requires confirmation_frames
            req_frames = 1 if rtype in (
                SceneRelationType.WEARING,
                SceneRelationType.MISSING,
                SceneRelationType.INSIDE,
                SceneRelationType.BEHAVIOR,
                SceneRelationType.APPROACHING,
                SceneRelationType.MOVING_AWAY,
            ) else self.confirmation_frames
            if state.consecutive_hits >= req_frames:
                lifecycle = RelationLifecycle.CREATED if not state.confirmed else RelationLifecycle.ACTIVE
                state.confirmed = True
                duration = max(0.0, timestamp - state.first_seen_ts)
                rel = SceneRelation(
                    relation_id=rel_id,
                    source_node_id=src,
                    target_node_id=dst,
                    relation_type=rtype,
                    lifecycle=lifecycle,
                    confidence=1.0,
                    timestamp=timestamp,
                    first_seen_timestamp=state.first_seen_ts,
                    last_seen_timestamp=timestamp,
                    duration_seconds=round(duration, 2),
                    evidence=ev,
                    metadata=meta,
                )
                output_relationships.append(rel)

        # Handle active relations that were NOT seen in this frame
        for rel_id, state in list(self._active_relations.items()):
            if rel_id not in candidate_rels:
                if state.relation_type == SceneRelationType.WEARING:
                    # PPE is a frame-grounded fact. Do not keep showing an old
                    # worn item while its current canonical state is UNKNOWN or
                    # MISSING (or the detection is no longer associated).
                    del self._active_relations[rel_id]
                    continue

                state.consecutive_misses += 1
                state.consecutive_hits = 0

                # Check if it was previously confirmed and now crossed end confirmation threshold
                if state.confirmed:
                    if state.consecutive_misses >= self.end_confirmation_frames:
                        # End relation
                        ended_rel = SceneRelation(
                            relation_id=rel_id,
                            source_node_id=state.source_id,
                            target_node_id=state.target_id,
                            relation_type=state.relation_type,
                            lifecycle=RelationLifecycle.ENDED,
                            timestamp=timestamp,
                            first_seen_timestamp=state.first_seen_ts,
                            last_seen_timestamp=timestamp,
                            duration_seconds=round(max(0.0, timestamp - state.first_seen_ts), 2),
                            evidence=state.evidence,
                            metadata={"reason": "condition_ceased"},
                        )
                        output_relationships.append(ended_rel)
                        del self._active_relations[rel_id]
                    else:
                        # Persist as ACTIVE until end confirmation threshold
                        duration = max(0.0, timestamp - state.first_seen_ts)
                        rel = SceneRelation(
                            relation_id=rel_id,
                            source_node_id=state.source_id,
                            target_node_id=state.target_id,
                            relation_type=state.relation_type,
                            lifecycle=RelationLifecycle.ACTIVE,
                            confidence=1.0,
                            timestamp=timestamp,
                            first_seen_timestamp=state.first_seen_ts,
                            last_seen_timestamp=state.last_seen_ts,
                            duration_seconds=round(duration, 2),
                            evidence=state.evidence,
                            metadata=state.metadata,
                        )
                        output_relationships.append(rel)
                else:
                    # Never reached confirmation; discard
                    del self._active_relations[rel_id]

        # Clean stale separation history for absent entities
        self._cleanup_stale_history(current_active_track_ids)
        self._prev_active_track_ids = current_active_track_ids

        # 6. Compute Situational Summary
        summary = self._compute_summary(nodes, output_relationships, active_zone_names)

        # 7. Assemble FrameScene
        frame_scene = FrameScene(
            frame_id=frame_id,
            timestamp=timestamp,
            camera_id=self.camera_id,
            source_camera_id=self.camera_id if self.camera_id.strip().lower() not in ("unassigned", "unknown") else None,
            nodes=nodes,
            relationships=output_relationships,
            summary=summary,
            active_zones=active_zone_names,
            active_behaviors=active_behaviors,
        )

        # Update in-memory scene store singleton for API inspection
        get_scene_store().set_current_scene(frame_scene)

        return frame_scene

    @staticmethod
    def _extract_track_id(node_id: str) -> Optional[int]:
        """Extracts integer track ID from node strings like 'person_17'."""
        parts = node_id.split("_")
        if len(parts) >= 2 and parts[-1].isdigit():
            return int(parts[-1])
        return None

    def _cleanup_stale_history(self, current_tids: Set[int]) -> None:
        """Removes pairwise distance history for entities no longer in scene."""
        stale_pairs = []
        for pair in self._separation_history.keys():
            t1 = self._extract_track_id(pair[0])
            t2 = self._extract_track_id(pair[1])
            if (t1 is not None and t1 not in current_tids) or (t2 is not None and t2 not in current_tids):
                stale_pairs.append(pair)
        for p in stale_pairs:
            del self._separation_history[p]

    def _compute_summary(
        self,
        nodes: List[SceneNode],
        relationships: List[SceneRelation],
        active_zones: List[str],
    ) -> SituationalSummary:
        """Aggregates high-level situational metrics across nodes and relationships."""
        workers = [n for n in nodes if n.node_type == SceneNodeType.PERSON]
        vehicles = [n for n in nodes if n.node_type == SceneNodeType.VEHICLE]
        machines = [n for n in nodes if n.node_type == SceneNodeType.MACHINE]
        objects = [n for n in nodes if n.node_type == SceneNodeType.OBJECT]

        active_rels = [r for r in relationships if r.lifecycle in (RelationLifecycle.ACTIVE, RelationLifecycle.CREATED)]

        # Occupied zones & workers in zones
        inside_rels = [r for r in active_rels if r.relation_type == SceneRelationType.INSIDE]
        occupied_zone_ids = {r.target_node_id for r in inside_rels}
        workers_in_zones = len({r.source_node_id for r in inside_rels if r.source_node_id.startswith("person_")})

        # Non-compliant workers
        non_compliant_workers = sum(
            1 for w in workers if w.attributes.get("compliance_status") == ComplianceStatus.NON_COMPLIANT.value
        )

        # Behavior metrics
        rapid_count = sum(
            1 for n in nodes if n.attributes.get("primary_behavior") == PrimaryBehavior.RAPID_MOVEMENT.value
            or n.attributes.get("flags", {}).get("rapid", False)
        )
        loiter_count = sum(
            1 for n in nodes if n.attributes.get("primary_behavior") == PrimaryBehavior.PROLONGED_STATIONARY.value
        )
        fall_count = sum(
            1 for n in nodes if n.attributes.get("primary_behavior") == PrimaryBehavior.POSSIBLE_FALL.value
            or n.attributes.get("flags", {}).get("fall", False)
        )

        # Proximity metrics
        near_count = sum(1 for r in active_rels if r.relation_type == SceneRelationType.NEAR)
        approaching_count = sum(1 for r in active_rels if r.relation_type == SceneRelationType.APPROACHING)

        return SituationalSummary(
            workers_count=len(workers),
            vehicles_count=len(vehicles),
            machines_count=len(machines),
            objects_count=len(objects),
            active_zones_count=len(active_zones),
            occupied_zones_count=len(occupied_zone_ids),
            workers_in_restricted_zones=workers_in_zones,
            workers_non_compliant_ppe=non_compliant_workers,
            rapid_movement_count=rapid_count,
            stationary_loitering_count=loiter_count,
            possible_fall_count=fall_count,
            active_proximity_relationships=near_count,
            active_approaching_relationships=approaching_count,
            total_active_relationships=len(active_rels),
        )

    def reset(self) -> None:
        """Clears all internal temporal and relationship tracking state."""
        self._active_relations.clear()
        self._separation_history.clear()
        self._prev_active_track_ids.clear()
        logger.info("SceneGraphBuilder reset.")
