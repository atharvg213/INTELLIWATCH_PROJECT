"""
intelligence/risk/factors.py
Step 10 - Deterministic Risk Factor Extraction.
Extracts atomic, verifiable risk factors from multimodal scene understanding outputs
(SceneGraph, PPE association, Zone occupancy, Behavior state machine, Depth).
Strictly adheres to technical honesty: image-space motion is NOT metric velocity;
monocular depth is relative, NOT metric distance; fall detection is a geometric heuristic.
"""
from typing import Any, Dict, List, Optional, Set
import logging

from backend.schemas.risk import RiskFactor, RiskFactorType
from backend.schemas.scene_graph import FrameScene, SceneNodeType, SceneRelationType
from backend.schemas.ppe import ComplianceStatus, FramePPEAssociation
from backend.schemas.zones import FrameZoneOccupancy
from backend.schemas.behavior import BehaviorState, PrimaryBehavior
from backend.schemas.depth import DepthResult
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.risk.factors")


class RiskFactorExtractor:
    """
    Deterministic extractor of verifiable safety risk factors from scene understanding outputs.
    Can ingest either the unified FrameScene (Step 9) or individual pipeline artifacts.
    """

    def __init__(self, settings_override: Optional[Dict[str, Any]] = None):
        self._settings = get_settings()
        self._overrides = settings_override or {}

    def _get_setting(self, key: str, default: Any) -> Any:
        return self._overrides.get(key, getattr(self._settings, key, default))

    def extract_factors(
        self,
        scene: Optional[FrameScene] = None,
        ppe_association: Optional[FramePPEAssociation] = None,
        zone_occupancy: Optional[FrameZoneOccupancy] = None,
        behavior_states: Optional[List[BehaviorState]] = None,
        depth_result: Optional[DepthResult] = None,
        timestamp: Optional[float] = None,
    ) -> List[RiskFactor]:
        """
        Extracts all active atomic risk factors for the current frame.
        """
        factors: List[RiskFactor] = []
        cur_ts = timestamp if timestamp is not None else (scene.timestamp if scene else 0.0)

        # ---------------------------------------------------------------------
        # 1. PPE NON-COMPLIANCE FACTORS
        # ---------------------------------------------------------------------
        ppe_score = float(self._get_setting("RISK_SCORE_PPE_VIOLATION", 25.0))
        seen_ppe_entities: Set[str] = set()

        # Check directly from FramePPEAssociation if provided
        if ppe_association:
            inventories = getattr(ppe_association, "worker_inventories", []) or getattr(ppe_association, "inventories", [])
            for inv in inventories:
                if inv.compliance_status == ComplianceStatus.NON_COMPLIANT:
                    entity_id = f"person_{inv.track_id}"
                    seen_ppe_entities.add(entity_id)
                    missing_items = list(inv.missing_ppe)
                    factors.append(
                        RiskFactor(
                            factor_id=f"factor_ppe_{inv.track_id}_{cur_ts:.3f}",
                            factor_type=RiskFactorType.PPE_NON_COMPLIANCE,
                            severity_contribution=ppe_score,
                            involved_entity_ids=[entity_id],
                            source_evidence={
                                "track_id": inv.track_id,
                                "compliance_status": inv.compliance_status.value,
                                "missing_ppe": missing_items,
                                "present_ppe": list(inv.present_ppe),
                                "unknown_ppe": list(inv.unknown_ppe),
                            },
                            timestamp=cur_ts,
                            explanation=f"Worker #{inv.track_id} is missing mandatory safety gear: {missing_items}.",
                            metadata={"track_id": inv.track_id},
                        )
                    )

        # Check from FrameScene if not already extracted from direct object
        if scene:
            for node in scene.nodes:
                if node.node_type == SceneNodeType.PERSON and node.node_id not in seen_ppe_entities:
                    comp = node.attributes.get("compliance_status")
                    if comp == ComplianceStatus.NON_COMPLIANT.value or comp == "NON_COMPLIANT":
                        ppe_status = node.attributes.get("ppe_status", {})
                        missing = [k for k, v in ppe_status.items() if v in ("MISSING", "missing")]
                        factors.append(
                            RiskFactor(
                                factor_id=f"factor_ppe_{node.node_id}_{cur_ts:.3f}",
                                factor_type=RiskFactorType.PPE_NON_COMPLIANCE,
                                severity_contribution=ppe_score,
                                involved_entity_ids=[node.node_id],
                                source_evidence={
                                    "node_id": node.node_id,
                                    "track_id": node.track_id,
                                    "compliance_status": comp,
                                    "missing_ppe": missing,
                                },
                                timestamp=cur_ts,
                                explanation=f"Worker {node.node_id} is non-compliant with PPE requirements: missing {missing}.",
                                metadata={"track_id": node.track_id},
                            )
                        )
                        seen_ppe_entities.add(node.node_id)

        # ---------------------------------------------------------------------
        # 2. RESTRICTED ZONE INTRUSION & DWELL FACTORS
        # ---------------------------------------------------------------------
        zone_score = float(self._get_setting("RISK_SCORE_ZONE_INTRUSION", 35.0))
        dwell_score = float(self._get_setting("RISK_SCORE_ZONE_DWELL", 40.0))
        seen_zone_pairs: Set[str] = set()

        if zone_occupancy:
            for mem in getattr(zone_occupancy, "memberships", []):
                if mem.is_inside:
                    pair_key = f"{mem.track_id}_{mem.zone_id}"
                    seen_zone_pairs.add(pair_key)
                    person_id = f"person_{mem.track_id}"
                    zone_id = f"zone_{mem.zone_id}" if not mem.zone_id.startswith("zone_") else mem.zone_id

                    # Zone Intrusion
                    factors.append(
                        RiskFactor(
                            factor_id=f"factor_zone_entry_{pair_key}_{cur_ts:.3f}",
                            factor_type=RiskFactorType.RESTRICTED_ZONE_INTRUSION,
                            severity_contribution=zone_score,
                            involved_entity_ids=[person_id, zone_id],
                            source_evidence={
                                "track_id": mem.track_id,
                                "zone_id": mem.zone_id,
                                "zone_name": mem.zone_name,
                                "contact_point": list(mem.contact_point),
                                "dwell_seconds": mem.dwell_seconds,
                            },
                            timestamp=cur_ts,
                            explanation=f"Worker #{mem.track_id} is located inside restricted zone '{mem.zone_name}'.",
                            metadata={"track_id": mem.track_id, "zone_id": mem.zone_id},
                        )
                    )

                    # Zone Dwell Exceeded (if dwell exceeds max allowed dwell)
                    max_dwell = float(self._get_setting("ZONE_MAX_DWELL_SECONDS", 10.0))
                    if mem.dwell_seconds > max_dwell:
                        factors.append(
                            RiskFactor(
                                factor_id=f"factor_zone_dwell_{pair_key}_{cur_ts:.3f}",
                                factor_type=RiskFactorType.RESTRICTED_ZONE_DWELL,
                                severity_contribution=dwell_score,
                                involved_entity_ids=[person_id, zone_id],
                                source_evidence={
                                    "track_id": mem.track_id,
                                    "zone_id": mem.zone_id,
                                    "zone_name": mem.zone_name,
                                    "dwell_seconds": mem.dwell_seconds,
                                    "max_dwell_seconds": max_dwell,
                                },
                                timestamp=cur_ts,
                                explanation=(
                                    f"Worker #{mem.track_id} has exceeded maximum allowed dwell time in zone "
                                    f"'{mem.zone_name}': {mem.dwell_seconds:.1f}s (threshold: {max_dwell:.1f}s)."
                                ),
                                metadata={"track_id": mem.track_id, "zone_id": mem.zone_id},
                            )
                        )

        # Check from FrameScene relations if not already captured
        if scene:
            for rel in scene.relationships:
                if rel.relation_type == SceneRelationType.INSIDE:
                    lval = rel.lifecycle.value if hasattr(rel.lifecycle, "value") else str(rel.lifecycle)
                    if lval in ("ACTIVE", "CREATED"):
                        pair_key = f"{rel.source_node_id}_{rel.target_node_id}"
                        if pair_key not in seen_zone_pairs:
                            seen_zone_pairs.add(pair_key)
                            dwell = rel.evidence.get("dwell_seconds", 0.0)
                            factors.append(
                                RiskFactor(
                                    factor_id=f"factor_zone_entry_{pair_key}_{cur_ts:.3f}",
                                    factor_type=RiskFactorType.RESTRICTED_ZONE_INTRUSION,
                                    severity_contribution=zone_score,
                                    involved_entity_ids=[rel.source_node_id, rel.target_node_id],
                                    source_evidence=dict(rel.evidence),
                                    timestamp=cur_ts,
                                    explanation=f"Entity {rel.source_node_id} is inside restricted zone {rel.target_node_id}.",
                                    metadata={"source": rel.source_node_id, "target": rel.target_node_id},
                                )
                            )
                            # Check dwell from relation evidence
                            max_dwell = float(self._get_setting("ZONE_MAX_DWELL_SECONDS", 10.0))
                            if dwell > max_dwell:
                                factors.append(
                                    RiskFactor(
                                        factor_id=f"factor_zone_dwell_{pair_key}_{cur_ts:.3f}",
                                        factor_type=RiskFactorType.RESTRICTED_ZONE_DWELL,
                                        severity_contribution=dwell_score,
                                        involved_entity_ids=[rel.source_node_id, rel.target_node_id],
                                        source_evidence=dict(rel.evidence),
                                        timestamp=cur_ts,
                                        explanation=(
                                            f"Entity {rel.source_node_id} has exceeded maximum allowed dwell time in "
                                            f"{rel.target_node_id}: {dwell:.1f}s (threshold: {max_dwell:.1f}s)."
                                        ),
                                        metadata={"source": rel.source_node_id, "target": rel.target_node_id},
                                    )
                                )

        # ---------------------------------------------------------------------
        # 3. BEHAVIOR & TEMPORAL MOTION FACTORS
        # ---------------------------------------------------------------------
        rapid_score = float(self._get_setting("RISK_SCORE_RAPID_MOVEMENT", 20.0))
        sudden_score = float(self._get_setting("RISK_SCORE_SUDDEN_MOVEMENT", 20.0))
        fall_score = float(self._get_setting("RISK_SCORE_FALL_LIKE", 50.0))
        stationary_score = float(self._get_setting("RISK_SCORE_PROLONGED_STATIONARY", 25.0))
        seen_behavior_entities: Set[str] = set()

        if behavior_states:
            for b in behavior_states:
                ent_id = f"person_{b.track_id}"
                seen_behavior_entities.add(ent_id)

                if b.flag_rapid_movement or b.primary_behavior == PrimaryBehavior.RAPID_MOVEMENT:
                    speed = b.image_speed_px_per_s or 0.0
                    factors.append(
                        RiskFactor(
                            factor_id=f"factor_rapid_{b.track_id}_{cur_ts:.3f}",
                            factor_type=RiskFactorType.RAPID_MOVEMENT,
                            severity_contribution=rapid_score,
                            involved_entity_ids=[ent_id],
                            source_evidence={"track_id": b.track_id, "image_speed_px_per_s": speed},
                            timestamp=cur_ts,
                            explanation=f"Worker #{b.track_id} is exhibiting rapid image-space movement ({speed:.1f} px/s).",
                            metadata={"track_id": b.track_id},
                        )
                    )

                if b.flag_sudden_movement or b.primary_behavior == PrimaryBehavior.SUDDEN_MOVEMENT:
                    accel = b.image_accel_px_per_s2 or 0.0
                    factors.append(
                        RiskFactor(
                            factor_id=f"factor_sudden_{b.track_id}_{cur_ts:.3f}",
                            factor_type=RiskFactorType.SUDDEN_MOVEMENT,
                            severity_contribution=sudden_score,
                            involved_entity_ids=[ent_id],
                            source_evidence={"track_id": b.track_id, "image_accel_px_per_s2": accel},
                            timestamp=cur_ts,
                            explanation=f"Worker #{b.track_id} experienced sudden image-space acceleration ({accel:.1f} px/s²).",
                            metadata={"track_id": b.track_id},
                        )
                    )

                if b.flag_possible_fall or b.primary_behavior == PrimaryBehavior.POSSIBLE_FALL:
                    factors.append(
                        RiskFactor(
                            factor_id=f"factor_fall_{b.track_id}_{cur_ts:.3f}",
                            factor_type=RiskFactorType.FALL_LIKE_BEHAVIOR,
                            severity_contribution=fall_score,
                            involved_entity_ids=[ent_id],
                            source_evidence={"track_id": b.track_id, "heuristic": "bounding_box_aspect_ratio_collapse"},
                            timestamp=cur_ts,
                            explanation=f"Fall-like behavior detected for worker #{b.track_id} (heuristic aspect ratio collapse).",
                            metadata={"track_id": b.track_id},
                        )
                    )

                if b.primary_behavior == PrimaryBehavior.PROLONGED_STATIONARY:
                    factors.append(
                        RiskFactor(
                            factor_id=f"factor_loitering_{b.track_id}_{cur_ts:.3f}",
                            factor_type=RiskFactorType.PROLONGED_STATIONARY,
                            severity_contribution=stationary_score,
                            involved_entity_ids=[ent_id],
                            source_evidence={"track_id": b.track_id, "stationary_duration_s": b.stationary_duration_s},
                            timestamp=cur_ts,
                            explanation=f"Worker #{b.track_id} has remained stationary for {b.stationary_duration_s:.1f}s (loitering).",
                            metadata={"track_id": b.track_id},
                        )
                    )

        # Check behavior flags on SceneNodes if not covered
        if scene:
            for node in scene.nodes:
                if node.node_id not in seen_behavior_entities:
                    flags = node.attributes.get("flags", {})
                    prim = node.attributes.get("primary_behavior", "")

                    if flags.get("rapid") or prim == PrimaryBehavior.RAPID_MOVEMENT.value:
                        sp = node.attributes.get("speed_px_per_s", 0.0)
                        factors.append(
                            RiskFactor(
                                factor_id=f"factor_rapid_{node.node_id}_{cur_ts:.3f}",
                                factor_type=RiskFactorType.RAPID_MOVEMENT,
                                severity_contribution=rapid_score,
                                involved_entity_ids=[node.node_id],
                                source_evidence={"node_id": node.node_id, "speed_px_per_s": sp},
                                timestamp=cur_ts,
                                explanation=f"Entity {node.node_id} is exhibiting rapid movement ({sp:.1f} px/s).",
                                metadata={"track_id": node.track_id},
                            )
                        )

                    if flags.get("sudden") or prim == PrimaryBehavior.SUDDEN_MOVEMENT.value:
                        factors.append(
                            RiskFactor(
                                factor_id=f"factor_sudden_{node.node_id}_{cur_ts:.3f}",
                                factor_type=RiskFactorType.SUDDEN_MOVEMENT,
                                severity_contribution=sudden_score,
                                involved_entity_ids=[node.node_id],
                                source_evidence={"node_id": node.node_id},
                                timestamp=cur_ts,
                                explanation=f"Entity {node.node_id} exhibited sudden acceleration.",
                                metadata={"track_id": node.track_id},
                            )
                        )

                    if flags.get("fall") or prim == PrimaryBehavior.POSSIBLE_FALL.value:
                        factors.append(
                            RiskFactor(
                                factor_id=f"factor_fall_{node.node_id}_{cur_ts:.3f}",
                                factor_type=RiskFactorType.FALL_LIKE_BEHAVIOR,
                                severity_contribution=fall_score,
                                involved_entity_ids=[node.node_id],
                                source_evidence={"node_id": node.node_id, "heuristic": "aspect_ratio_collapse"},
                                timestamp=cur_ts,
                                explanation=f"Fall-like behavior detected for {node.node_id} (heuristic aspect ratio collapse).",
                                metadata={"track_id": node.track_id},
                            )
                        )

                    if prim == PrimaryBehavior.PROLONGED_STATIONARY.value:
                        dur = node.attributes.get("stationary_duration_s", 0.0)
                        factors.append(
                            RiskFactor(
                                factor_id=f"factor_loitering_{node.node_id}_{cur_ts:.3f}",
                                factor_type=RiskFactorType.PROLONGED_STATIONARY,
                                severity_contribution=stationary_score,
                                involved_entity_ids=[node.node_id],
                                source_evidence={"node_id": node.node_id, "stationary_duration_s": dur},
                                timestamp=cur_ts,
                                explanation=f"Entity {node.node_id} has remained stationary for {dur:.1f}s (loitering).",
                                metadata={"track_id": node.track_id},
                            )
                        )

        # ---------------------------------------------------------------------
        # 4. SPATIAL & RELATIONAL FACTORS (PROXIMITY, APPROACHING, MACHINERY)
        # ---------------------------------------------------------------------
        prox_score = float(self._get_setting("RISK_SCORE_PERSON_VEHICLE_PROXIMITY", 30.0))
        appr_score = float(self._get_setting("RISK_SCORE_APPROACHING_VEHICLE", 45.0))
        mach_score = float(self._get_setting("RISK_SCORE_WORKER_MACHINE", 30.0))

        if scene:
            node_map = {n.node_id: n for n in scene.nodes}
            for rel in scene.relationships:
                lval = rel.lifecycle.value if hasattr(rel.lifecycle, "value") else str(rel.lifecycle)
                if lval not in ("ACTIVE", "CREATED"):
                    continue

                src_node = node_map.get(rel.source_node_id)
                tgt_node = node_map.get(rel.target_node_id)
                if not src_node or not tgt_node:
                    continue

                src_type = src_node.node_type
                tgt_type = tgt_node.node_type

                is_person_vehicle = (
                    (src_type == SceneNodeType.PERSON and tgt_type == SceneNodeType.VEHICLE)
                    or (src_type == SceneNodeType.VEHICLE and tgt_type == SceneNodeType.PERSON)
                )
                is_person_machine = (
                    (src_type == SceneNodeType.PERSON and tgt_type == SceneNodeType.MACHINE)
                    or (src_type == SceneNodeType.MACHINE and tgt_type == SceneNodeType.PERSON)
                )

                # A. NEAR relationship between Person and Vehicle
                if rel.relation_type == SceneRelationType.NEAR and is_person_vehicle:
                    prox_state = rel.evidence.get("proximity_state", "NEAR")
                    factors.append(
                        RiskFactor(
                            factor_id=f"factor_near_veh_{rel.relation_id}_{cur_ts:.3f}",
                            factor_type=RiskFactorType.PERSON_VEHICLE_PROXIMITY,
                            severity_contribution=prox_score,
                            involved_entity_ids=[rel.source_node_id, rel.target_node_id],
                            source_evidence=dict(rel.evidence),
                            timestamp=cur_ts,
                            explanation=(
                                f"Worker ({rel.source_node_id}) is {prox_state.lower().replace('_', ' ')} vehicle "
                                f"({rel.target_node_id}) based on relative spatial analysis."
                            ),
                            metadata={"relation_id": rel.relation_id, "proximity_state": prox_state},
                        )
                    )

                # B. APPROACHING relationship between Person and Vehicle
                if rel.relation_type == SceneRelationType.APPROACHING and is_person_vehicle:
                    prox_state = rel.evidence.get("proximity_state", "NEAR")
                    factors.append(
                        RiskFactor(
                            factor_id=f"factor_appr_veh_{rel.relation_id}_{cur_ts:.3f}",
                            factor_type=RiskFactorType.APPROACHING_VEHICLE,
                            severity_contribution=appr_score,
                            involved_entity_ids=[rel.source_node_id, rel.target_node_id],
                            source_evidence=dict(rel.evidence),
                            timestamp=cur_ts,
                            explanation=(
                                f"Worker ({rel.source_node_id}) and vehicle ({rel.target_node_id}) exhibit an "
                                f"approaching spatial relationship (relative spatial state: {prox_state}). Note: kinematic relationship, not confirmed collision."
                            ),
                            metadata={"relation_id": rel.relation_id, "proximity_state": prox_state},
                        )
                    )

                # C. NEAR relationship between Person and Machine
                if rel.relation_type == SceneRelationType.NEAR and is_person_machine:
                    prox_state = rel.evidence.get("proximity_state", "NEAR")
                    factors.append(
                        RiskFactor(
                            factor_id=f"factor_near_mach_{rel.relation_id}_{cur_ts:.3f}",
                            factor_type=RiskFactorType.WORKER_NEAR_MACHINE,
                            severity_contribution=mach_score,
                            involved_entity_ids=[rel.source_node_id, rel.target_node_id],
                            source_evidence=dict(rel.evidence),
                            timestamp=cur_ts,
                            explanation=(
                                f"Worker ({rel.source_node_id}) is {prox_state.lower().replace('_', ' ')} machine "
                                f"({rel.target_node_id}) based on relative spatial analysis."
                            ),
                            metadata={"relation_id": rel.relation_id, "proximity_state": prox_state},
                        )
                    )

                # D. APPROACHING relationship between Person and Machine
                if rel.relation_type == SceneRelationType.APPROACHING and is_person_machine:
                    prox_state = rel.evidence.get("proximity_state", "NEAR")
                    factors.append(
                        RiskFactor(
                            factor_id=f"factor_appr_mach_{rel.relation_id}_{cur_ts:.3f}",
                            factor_type=RiskFactorType.WORKER_NEAR_MACHINE,
                            severity_contribution=mach_score + 10.0,
                            involved_entity_ids=[rel.source_node_id, rel.target_node_id],
                            source_evidence=dict(rel.evidence),
                            timestamp=cur_ts,
                            explanation=(
                                f"Worker ({rel.source_node_id}) is actively approaching machine "
                                f"({rel.target_node_id}) (spatial state: {prox_state})."
                            ),
                            metadata={"relation_id": rel.relation_id, "trend": "APPROACHING", "proximity_state": prox_state},
                        )
                    )

        return factors
