"""
intelligence/risk/rules.py
Step 10 - Deterministic Risk Factor to Event Mapping Rules.
Maps individual atomic risk factors into structured Candidate Events.
Identifies multi-factor scenarios and constructs compound safety events without double-counting evidence.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict
import logging

from backend.schemas.risk import (
    RiskEvent,
    RiskEventType,
    RiskFactor,
    RiskFactorType,
    EventLifecycleState,
    RiskLevel,
)
from intelligence.risk.scorer import RiskScorer
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.risk.rules")


class CandidateEvent:
    """Intermediate candidate event representation before temporal lifecycle evaluation."""

    def __init__(
        self,
        event_key: str,
        event_type: RiskEventType,
        involved_entities: List[str],
        factors: List[RiskFactor],
        explanation: str,
        evidence: Dict[str, Any],
        is_compound: bool = False,
    ):
        self.event_key = event_key
        self.event_type = event_type
        self.involved_entities = sorted(list(set(involved_entities)))
        self.factors = factors
        self.explanation = explanation
        self.evidence = evidence
        self.is_compound = is_compound


class RiskRuleEvaluator:
    """
    Evaluates business rules over atomic risk factors to construct candidate safety events.
    Handles single-factor safety events and compound multi-hazard events.
    """

    def __init__(self, settings_override: Optional[Dict[str, Any]] = None):
        self._settings = get_settings()
        self._overrides = settings_override or {}
        self._scorer = RiskScorer(settings_override=settings_override)

    def _get_setting(self, key: str, default: Any) -> Any:
        return self._overrides.get(key, getattr(self._settings, key, default))

    def evaluate_candidate_events(
        self,
        factors: List[RiskFactor],
        timestamp: float,
    ) -> List[CandidateEvent]:
        """
        Transforms atomic factors into candidate events.
        """
        candidates: List[CandidateEvent] = []
        compound_enabled = bool(self._get_setting("RISK_COMPOUND_EVENT_ENABLED", True))

        # ---------------------------------------------------------------------
        # 1. Single-factor / Relational event mapping
        # ---------------------------------------------------------------------
        for f in factors:
            ents = f.involved_entity_ids
            ents_key = "_".join(sorted(ents))

            if f.factor_type == RiskFactorType.PPE_NON_COMPLIANCE:
                ekey = f"PPE_VIOLATION:{ents_key}"
                candidates.append(
                    CandidateEvent(
                        event_key=ekey,
                        event_type=RiskEventType.PPE_VIOLATION,
                        involved_entities=ents,
                        factors=[f],
                        explanation=f.explanation,
                        evidence=dict(f.source_evidence),
                    )
                )

            elif f.factor_type == RiskFactorType.RESTRICTED_ZONE_INTRUSION:
                ekey = f"RESTRICTED_ZONE_INTRUSION:{ents_key}"
                candidates.append(
                    CandidateEvent(
                        event_key=ekey,
                        event_type=RiskEventType.RESTRICTED_ZONE_INTRUSION,
                        involved_entities=ents,
                        factors=[f],
                        explanation=f.explanation,
                        evidence=dict(f.source_evidence),
                    )
                )

            elif f.factor_type == RiskFactorType.RESTRICTED_ZONE_DWELL:
                ekey = f"RESTRICTED_ZONE_DWELL:{ents_key}"
                candidates.append(
                    CandidateEvent(
                        event_key=ekey,
                        event_type=RiskEventType.RESTRICTED_ZONE_DWELL,
                        involved_entities=ents,
                        factors=[f],
                        explanation=f.explanation,
                        evidence=dict(f.source_evidence),
                    )
                )

            elif f.factor_type == RiskFactorType.RAPID_MOVEMENT:
                ekey = f"RAPID_MOVEMENT_EVENT:{ents_key}"
                candidates.append(
                    CandidateEvent(
                        event_key=ekey,
                        event_type=RiskEventType.RAPID_MOVEMENT_EVENT,
                        involved_entities=ents,
                        factors=[f],
                        explanation=f.explanation,
                        evidence=dict(f.source_evidence),
                    )
                )

            elif f.factor_type == RiskFactorType.FALL_LIKE_BEHAVIOR:
                ekey = f"FALL_LIKE_EVENT:{ents_key}"
                candidates.append(
                    CandidateEvent(
                        event_key=ekey,
                        event_type=RiskEventType.FALL_LIKE_EVENT,
                        involved_entities=ents,
                        factors=[f],
                        explanation=f.explanation,
                        evidence=dict(f.source_evidence),
                    )
                )

            elif f.factor_type == RiskFactorType.PERSON_VEHICLE_PROXIMITY:
                ekey = f"PERSON_VEHICLE_PROXIMITY:{ents_key}"
                candidates.append(
                    CandidateEvent(
                        event_key=ekey,
                        event_type=RiskEventType.PERSON_VEHICLE_PROXIMITY,
                        involved_entities=ents,
                        factors=[f],
                        explanation=f.explanation,
                        evidence=dict(f.source_evidence),
                    )
                )

            elif f.factor_type == RiskFactorType.APPROACHING_VEHICLE:
                ekey = f"APPROACHING_VEHICLE:{ents_key}"
                candidates.append(
                    CandidateEvent(
                        event_key=ekey,
                        event_type=RiskEventType.APPROACHING_VEHICLE,
                        involved_entities=ents,
                        factors=[f],
                        explanation=f.explanation,
                        evidence=dict(f.source_evidence),
                    )
                )

            elif f.factor_type == RiskFactorType.WORKER_NEAR_MACHINE:
                ekey = f"WORKER_MACHINE_RISK:{ents_key}"
                candidates.append(
                    CandidateEvent(
                        event_key=ekey,
                        event_type=RiskEventType.WORKER_MACHINE_RISK,
                        involved_entities=ents,
                        factors=[f],
                        explanation=f.explanation,
                        evidence=dict(f.source_evidence),
                    )
                )

        # ---------------------------------------------------------------------
        # 2. Compound Multi-Factor Event Detection
        # ---------------------------------------------------------------------
        if compound_enabled:
            # Group factors by primary entity (e.g. person_17)
            entity_factors: Dict[str, List[RiskFactor]] = defaultdict(list)
            for f in factors:
                for ent in f.involved_entity_ids:
                    # Focus compounding on human workers or primary focal entities
                    if ent.startswith("person_"):
                        entity_factors[ent].append(f)

            for focal_entity, ent_factor_list in entity_factors.items():
                # Distinct factor types to avoid duplicate counting of same condition
                distinct_types = {f.factor_type for f in ent_factor_list}
                if len(distinct_types) >= 2:
                    # Collect all involved entities across factors
                    all_involved = set()
                    for f in ent_factor_list:
                        all_involved.update(f.involved_entity_ids)

                    # Deduplicate factors by type (keep highest contribution or first)
                    unique_factors = []
                    seen_types = set()
                    for f in ent_factor_list:
                        if f.factor_type not in seen_types:
                            unique_factors.append(f)
                            seen_types.add(f.factor_type)

                    factor_names = [f.factor_type.value for f in unique_factors]
                    ckey = f"COMPOUND_SAFETY_EVENT:{focal_entity}"
                    explanation, audit_5w = _synthesize_compound_explanation_and_audit(
                        focal_entity=focal_entity,
                        factors=unique_factors,
                        all_involved=all_involved,
                    )
                    compound_evidence = {
                        "focal_entity": focal_entity,
                        "audit_5w": audit_5w,
                        "contributing_factors": [
                            {"type": f.factor_type.value, "explanation": f.explanation, "evidence": f.source_evidence}
                            for f in unique_factors
                        ],
                    }

                    candidates.append(
                        CandidateEvent(
                            event_key=ckey,
                            event_type=RiskEventType.COMPOUND_SAFETY_EVENT,
                            involved_entities=list(all_involved),
                            factors=unique_factors,
                            explanation=explanation,
                            evidence=compound_evidence,
                            is_compound=True,
                        )
                    )

        return candidates


def _synthesize_compound_explanation_and_audit(
    focal_entity: str,
    factors: List[RiskFactor],
    all_involved: Set[str],
) -> Tuple[str, Dict[str, Any]]:
    """
    Synthesizes a human-readable, deterministic explainable description and
    structured 5-W audit breakdown for a compound safety event.
    """
    if focal_entity.startswith("person_") or focal_entity.startswith("worker_"):
        worker_label = focal_entity.replace("person_", "Worker #").replace("worker_", "Worker #")
    else:
        worker_label = focal_entity.replace("_", " #").capitalize()
    factor_types = {f.factor_type for f in factors}

    vehicle_ent = None
    machine_ent = None
    zone_name = None
    missing_items = []
    prox_state = None

    for f in factors:
        for ent in f.involved_entity_ids:
            ent_l = ent.lower()
            if any(k in ent_l for k in ("forklift", "vehicle", "truck", "agv")):
                vehicle_ent = ent.replace("_", " #").capitalize()
            elif any(k in ent_l for k in ("machinery", "machine", "conveyor", "press", "robot")):
                machine_ent = ent.replace("_", " #").capitalize()

        if f.factor_type in (RiskFactorType.RESTRICTED_ZONE_INTRUSION, RiskFactorType.RESTRICTED_ZONE_DWELL):
            zone_name = f.source_evidence.get("zone_name") or f.metadata.get("zone_name")

        if f.factor_type == RiskFactorType.PPE_NON_COMPLIANCE:
            miss = f.source_evidence.get("missing_ppe") or f.source_evidence.get("items")
            if miss and isinstance(miss, list):
                missing_items.extend(miss)

        if "proximity_state" in f.metadata:
            prox_state = f.metadata["proximity_state"]
        elif "proximity_state" in f.source_evidence:
            prox_state = f.source_evidence["proximity_state"]

    clauses = []
    # Kinematics / proximity clause
    if RiskFactorType.APPROACHING_VEHICLE in factor_types and vehicle_ent:
        prox_str = f" while remaining in close relative proximity"
        if prox_state and prox_state != "UNKNOWN":
            prox_str = f" while remaining in close relative proximity ({prox_state})"
        clauses.append(f"is approaching {vehicle_ent}{prox_str}")
    elif RiskFactorType.PERSON_VEHICLE_PROXIMITY in factor_types and vehicle_ent:
        prox_desc = prox_state.lower().replace('_', ' ') if (prox_state and prox_state != "UNKNOWN") else "close relative proximity"
        clauses.append(f"is in {prox_desc} to {vehicle_ent}")

    if RiskFactorType.WORKER_NEAR_MACHINE in factor_types and machine_ent:
        prox_desc = prox_state.lower().replace('_', ' ') if (prox_state and prox_state != "UNKNOWN") else "close proximity"
        clauses.append(f"is in {prox_desc} to {machine_ent}")

    # Zone clause
    if zone_name:
        clauses.append(f"inside restricted zone '{zone_name}'")
    elif RiskFactorType.RESTRICTED_ZONE_INTRUSION in factor_types:
        clauses.append("inside a restricted area")

    # PPE clause
    if missing_items:
        unique_missing = sorted(list(set(missing_items)))
        clauses.append(f"missing mandatory PPE ({', '.join(unique_missing)})")
    elif RiskFactorType.PPE_NON_COMPLIANCE in factor_types:
        clauses.append("has an active PPE non-compliance violation")

    # Rapid movement / behavior
    if RiskFactorType.RAPID_MOVEMENT in factor_types:
        clauses.append("exhibiting rapid movement")

    if clauses:
        if len(clauses) == 1:
            explanation = f"{worker_label} {clauses[0]}."
        elif len(clauses) == 2:
            explanation = f"{worker_label} {clauses[0]} and {clauses[1]}."
        else:
            explanation = f"{worker_label} {', '.join(clauses[:-1])}, and {clauses[-1]}."
    else:
        fnames = [f.factor_type.value for f in factors]
        explanation = f"Multiple concurrent safety risk factors active for {worker_label}: [{', '.join(fnames)}]."

    who_list = [worker_label]
    if vehicle_ent:
        who_list.append(vehicle_ent)
    if machine_ent:
        who_list.append(machine_ent)

    what_str = "Compound safety risk condition"
    if vehicle_ent and RiskFactorType.APPROACHING_VEHICLE in factor_types:
        what_str = f"Worker approaching active vehicle ({vehicle_ent})"
    elif machine_ent:
        what_str = f"Worker operating near machinery ({machine_ent}) with hazard conditions"
    elif zone_name:
        what_str = f"Worker safety violations inside {zone_name}"

    where_str = f"Restricted zone '{zone_name}'" if zone_name else "Facility floor area"
    why_str = f"Compounding hazards elevate risk beyond single-factor tolerances ({len(factors)} concurrent factors)"

    audit = {
        "what": what_str,
        "who": who_list,
        "where": where_str,
        "why": why_str,
        "evidence_summary": [
            {"factor": f.factor_type.value, "detail": f.explanation} for f in factors
        ],
    }

    return explanation, audit
