"""
backend/services/explanation_service.py
Step 17 - Deterministic Explainability, Evidence & Safety Analytics Service.

Provides:
- Five-W Incident Explanations (WHO, WHAT, WHERE, WHEN, WHY)
- Risk Score Calculation Breakdown
- Chronological Incident Lifecycle Timelines
- Risk-over-Time Trajectory Trend Data
- Descriptive Safety Analytics Aggregations
- JSON Evidence Package Generation

Zero generative AI or external LLM dependencies; entirely derived from
structured perception, tracking, compliance, behavior, and risk state.
"""
from collections import Counter, defaultdict
import logging
from typing import Any, Dict, List, Optional, Tuple

from backend.schemas.evidence import IncidentRecord
from backend.schemas.explanation import (
    AnalyticsSummary,
    FiveWExplanation,
    IncidentExplanationResponse,
    IncidentTimelineEntry,
    RiskBreakdownItem,
    RiskScoreBreakdown,
    RiskTrendPoint,
)
from backend.schemas.risk import EventLifecycleState, RiskEventType, RiskLevel
from backend.services.assessment_store import get_assessment_store
from backend.services.incident_store import get_incident_store
from backend.services.temporal_log import get_temporal_log

logger = logging.getLogger("intelliwatch.explanation_service")


# Human-friendly event type mapping
EVENT_TYPE_DESCRIPTIONS: Dict[str, str] = {
    RiskEventType.RESTRICTED_ZONE_INTRUSION.value: "Restricted-zone intrusion detected",
    RiskEventType.RESTRICTED_ZONE_DWELL.value: "Extended dwell timeout in restricted zone",
    RiskEventType.PPE_VIOLATION.value: "Mandatory PPE compliance violation",
    RiskEventType.PERSON_VEHICLE_PROXIMITY.value: "Person-vehicle hazardous proximity condition",
    RiskEventType.APPROACHING_VEHICLE.value: "Vehicle rapidly approaching worker trajectory",
    RiskEventType.WORKER_MACHINE_RISK.value: "Worker in hazardous operating perimeter of heavy machinery",
    RiskEventType.COMPOUND_SAFETY_EVENT.value: "Compound multi-hazard industrial safety event",
    RiskEventType.FALL_LIKE_EVENT.value: "Fall-like geometric anomaly detected (heuristic condition)",
    RiskEventType.RAPID_MOVEMENT_EVENT.value: "Rapid / erratic locomotion observed in operational area",
}

DEFAULT_THRESHOLDS = {
    "INFO": 0.0,
    "LOW": 10.0,
    "MEDIUM": 30.0,
    "HIGH": 60.0,
    "CRITICAL": 85.0,
}


class ExplanationService:
    """
    Operator Explainability, Evidence & Safety Analytics Service.
    Transforms raw and structured system state into auditable, explainable insights.
    """

    def __init__(self):
        self.incident_store = get_incident_store()
        self.temporal_log = get_temporal_log()
        self.assessment_store = get_assessment_store()

    # -------------------------------------------------------------------------
    # 1. Five-W Explanation Builder
    # -------------------------------------------------------------------------
    def build_five_w(self, incident: IncidentRecord) -> FiveWExplanation:
        """
        Builds a deterministic 5W explanation from structured incident state.
        Answers: WHO, WHAT, WHERE, WHEN, WHY without generative models.
        """
        # --- WHO ---
        who_parts = []
        if incident.involved_track_ids:
            tracks_str = ", ".join(f"Track #{tid}" for tid in incident.involved_track_ids)
            who_parts.append(tracks_str)
        elif incident.involved_entity_ids:
            entities_str = ", ".join(eid.replace("_", " ").title() for eid in incident.involved_entity_ids)
            who_parts.append(entities_str)
        else:
            who_parts.append("Unassigned actor / General workcell")

        # Contextual compliance & behavior hints from metadata or factors
        ppe_hints = []
        behavior_hints = []
        for factor in incident.risk_factors:
            ftype = factor.factor_type.value if hasattr(factor.factor_type, "value") else str(factor.factor_type)
            if "PPE" in ftype:
                ppe_hints.append(factor.explanation)
            elif "MOVEMENT" in ftype or "FALL" in ftype:
                behavior_hints.append(factor.explanation)

        if ppe_hints:
            who_parts.append(f"[{'; '.join(ppe_hints)}]")
        if behavior_hints:
            who_parts.append(f"[{'; '.join(behavior_hints)}]")

        who_str = " | ".join(who_parts)

        # --- WHAT ---
        etype_val = incident.event_type.value if hasattr(incident.event_type, "value") else str(incident.event_type)
        what_str = EVENT_TYPE_DESCRIPTIONS.get(etype_val, etype_val.replace("_", " ").title())
        if incident.risk_level in (RiskLevel.HIGH, RiskLevel.CRITICAL):
            what_str += f" ({incident.risk_level.value} Risk)"

        # --- WHERE ---
        if incident.location_or_zone and incident.location_or_zone.strip():
            where_str = f"Zone: {incident.location_or_zone.strip()} ({incident.camera_id})"
        elif incident.metadata and incident.metadata.get("zone_name"):
            where_str = f"Zone: {incident.metadata['zone_name']} ({incident.camera_id})"
        else:
            where_str = f"Operational Scene ({incident.camera_id})"

        # --- WHEN ---
        when_str = f"{incident.timestamp:.2f}s (Frame {incident.frame_id})"

        # --- WHY ---
        why_bullets: List[str] = []
        for f in incident.risk_factors:
            expl = f.explanation.strip()
            if expl and expl not in why_bullets:
                why_bullets.append(expl)

        # Scene relationships as supporting evidence
        for rel in incident.scene_relationships:
            subj = rel.get("source_id", "").replace("_", " ").title()
            predicate = rel.get("relation_type", "").replace("_", " ").lower()
            obj = rel.get("target_id", "").replace("_", " ").title()
            if subj and predicate and obj:
                rel_fact = f"{subj} {predicate} {obj}"
                if rel_fact not in why_bullets:
                    why_bullets.append(rel_fact)

        # Early-warning evidence if present
        for ew in incident.early_warnings:
            ew_type = ew.indicator_type.value if hasattr(ew.indicator_type, "value") else str(ew.indicator_type)
            ew_expl = ew.explanation if hasattr(ew, "explanation") else ""
            ew_fact = f"Preceding indicator: {ew_type.replace('_', ' ').title()}"
            if ew_expl:
                ew_fact += f" ({ew_expl})"
            if ew_fact not in why_bullets:
                why_bullets.append(ew_fact)

        if not why_bullets:
            why_bullets.append(incident.explanation or "Condition evaluated to risk threshold criteria.")

        return FiveWExplanation(
            who=who_str,
            what=what_str,
            where=where_str,
            when=when_str,
            why=why_bullets,
        )

    # -------------------------------------------------------------------------
    # 2. Risk Score Breakdown
    # -------------------------------------------------------------------------
    def build_risk_breakdown(self, incident: IncidentRecord) -> RiskScoreBreakdown:
        """
        Itemizes atomic factor contributions and multi-factor escalation.
        Uses the exact mathematical formulation from RiskScorer.
        """
        items: List[RiskBreakdownItem] = []
        base_sum = 0.0

        for f in incident.risk_factors:
            ftype = f.factor_type.value if hasattr(f.factor_type, "value") else str(f.factor_type)
            items.append(
                RiskBreakdownItem(
                    factor_type=ftype,
                    contribution=float(f.severity_contribution),
                    explanation=f.explanation,
                )
            )
            base_sum += float(f.severity_contribution)

        # Determine escalation
        final_score = float(incident.risk_score)
        escalation = max(0.0, round(final_score - base_sum, 2))

        return RiskScoreBreakdown(
            base_score=round(base_sum, 2),
            factor_contributions=items,
            escalation=escalation,
            final_score=final_score,
            risk_tier=incident.risk_level,
            thresholds=DEFAULT_THRESHOLDS,
        )

    # -------------------------------------------------------------------------
    # 3. Chronological Incident Timeline
    # -------------------------------------------------------------------------
    def build_incident_timeline(self, incident: IncidentRecord) -> List[IncidentTimelineEntry]:
        """
        Constructs a chronological progression of events before, during, and after the incident.
        Integrates entries from TemporalLogService and incident lifecycle transitions.
        """
        timeline: List[IncidentTimelineEntry] = []
        start_ts = incident.timestamp
        end_ts = (
            incident.metadata.get("end_timestamp")
            if incident.metadata
            else incident.timestamp + 2.0
        )
        window_start = max(0.0, start_ts - 5.0)
        window_end = (end_ts or start_ts) + 5.0

        # Query temporal log entries
        all_temporal = self.temporal_log.get_all_entries()
        involved_eids = set(incident.involved_entity_ids)
        for tid in incident.involved_track_ids:
            involved_eids.add(f"person_{tid}")
            involved_eids.add(f"worker_{tid}")
            involved_eids.add(f"track_{tid}")

        for entry in all_temporal:
            # Check if within temporal proximity and entity relevant
            if window_start <= entry.timestamp <= window_end:
                if (
                    not involved_eids
                    or entry.entity_id in involved_eids
                    or entry.entity_id == "system"
                    or any(str(tid) in entry.label or str(tid) in entry.description for tid in incident.involved_track_ids)
                ):
                    cat_val = entry.category.value if hasattr(entry.category, "value") else str(entry.category)
                    timeline.append(
                        IncidentTimelineEntry(
                            timestamp=entry.timestamp,
                            frame_id=entry.frame_id,
                            category=cat_val,
                            label=entry.label,
                            description=entry.description,
                            risk_level=None,
                        )
                    )

        # Ensure the incident confirmation itself is explicitly present
        has_inc_entry = any(
            incident.incident_id in e.description or e.frame_id == incident.frame_id
            for e in timeline
        )
        if not has_inc_entry:
            timeline.append(
                IncidentTimelineEntry(
                    timestamp=incident.timestamp,
                    frame_id=incident.frame_id,
                    category="INCIDENT_CONFIRMED",
                    label=f"Incident Confirmed: {incident.event_type.value}",
                    description=f"{incident.explanation} (Score: {incident.risk_score:.1f}, Tier: {incident.risk_level.value})",
                    risk_level=incident.risk_level.value,
                )
            )

        # Check for incident end transition
        if incident.lifecycle_state in (EventLifecycleState.ENDED, "ENDED") or (incident.metadata and incident.metadata.get("end_timestamp")):
            e_ts = incident.metadata.get("end_timestamp", incident.timestamp + 1.0)
            timeline.append(
                IncidentTimelineEntry(
                    timestamp=e_ts,
                    frame_id=incident.frame_id + int((e_ts - incident.timestamp) * 30),
                    category="INCIDENT_ENDED",
                    label=f"Incident Ended: {incident.event_type.value}",
                    description="Safety risk condition ceased or involved entity departed the hazard zone.",
                    risk_level="INFO",
                )
            )

        # Sort chronologically by timestamp, then frame_id
        timeline.sort(key=lambda x: (x.timestamp, x.frame_id))
        return timeline

    # -------------------------------------------------------------------------
    # 4. Risk Trend Trajectory
    # -------------------------------------------------------------------------
    def build_risk_trend(self, incident: IncidentRecord) -> List[RiskTrendPoint]:
        """
        Produces time-series points (timestamp, frame_id, risk_score, risk_level)
        for visual trend graph rendering.
        """
        points: List[RiskTrendPoint] = []
        # Check if temporal log has risk escalation/decay points
        for entry in self.temporal_log.get_all_entries():
            cat_name = entry.category.value if hasattr(entry.category, "value") else str(entry.category)
            if "RISK" in cat_name:
                # Approximate score from level string
                lvl = entry.new_state or "LOW"
                approx_score = 15.0
                if lvl == "CRITICAL":
                    approx_score = 90.0
                elif lvl == "HIGH":
                    approx_score = 70.0
                elif lvl == "MEDIUM":
                    approx_score = 45.0
                elif lvl == "LOW":
                    approx_score = 20.0
                elif lvl == "INFO":
                    approx_score = 5.0

                points.append(
                    RiskTrendPoint(
                        timestamp=entry.timestamp,
                        frame_id=entry.frame_id,
                        risk_score=approx_score,
                        risk_level=lvl,
                    )
                )

        # Always include the incident confirmation point
        points.append(
            RiskTrendPoint(
                timestamp=incident.timestamp,
                frame_id=incident.frame_id,
                risk_score=incident.risk_score,
                risk_level=incident.risk_level.value,
            )
        )

        # Sort and deduplicate by frame_id
        points.sort(key=lambda p: (p.timestamp, p.frame_id))
        unique_points: List[RiskTrendPoint] = []
        seen_fids = set()
        for p in points:
            if p.frame_id not in seen_fids:
                seen_fids.add(p.frame_id)
                unique_points.append(p)

        return unique_points

    # -------------------------------------------------------------------------
    # 5. Full Incident Explanation Payload
    # -------------------------------------------------------------------------
    def get_incident_explanation(self, incident_id: str) -> Optional[IncidentExplanationResponse]:
        """
        Retrieves an incident by ID and constructs the comprehensive explanation payload.
        """
        incident = self.incident_store.get_incident(incident_id)
        if incident is None:
            return None

        five_w = self.build_five_w(incident)
        risk_breakdown = self.build_risk_breakdown(incident)
        timeline = self.build_incident_timeline(incident)
        risk_trend = self.build_risk_trend(incident)

        # Calculate duration if ended
        duration_s = None
        end_ts = incident.metadata.get("end_timestamp") if incident.metadata else None
        if end_ts and end_ts >= incident.timestamp:
            duration_s = round(end_ts - incident.timestamp, 2)

        snapshot_url = (
            f"/api/v1/incidents/{incident.incident_id}/evidence"
            if incident.has_snapshot
            else None
        )
        evidence_notice = (
            "Visual evidence snapshot captured at incident confirmation."
            if incident.has_snapshot
            else "No visual evidence snapshot available (Metadata-only record)."
        )

        early_warnings_payload = [
            ew.model_dump(mode="json") if hasattr(ew, "model_dump") else dict(ew)
            for ew in incident.early_warnings
        ]

        return IncidentExplanationResponse(
            incident_id=incident.incident_id,
            event_type=incident.event_type,
            risk_level=incident.risk_level,
            risk_score=incident.risk_score,
            lifecycle_state=incident.lifecycle_state,
            timestamp=incident.timestamp,
            end_timestamp=end_ts,
            duration_s=duration_s,
            frame_id=incident.frame_id,
            camera_id=incident.camera_id,
            five_w=five_w,
            risk_breakdown=risk_breakdown,
            timeline=timeline,
            risk_trend=risk_trend,
            scene_relationships=incident.scene_relationships,
            early_warnings=early_warnings_payload,
            has_snapshot=incident.has_snapshot,
            snapshot_url=snapshot_url,
            evidence_notice=evidence_notice,
        )

    # -------------------------------------------------------------------------
    # 6. Safety Analytics Summary
    # -------------------------------------------------------------------------
    def compute_analytics(self) -> AnalyticsSummary:
        """
        Aggregates operational statistics across all recorded incidents.
        Strictly descriptive safety metrics; no employee profiling or rating.
        """
        incidents = self.incident_store.list_incidents(limit=1000)

        total = len(incidents)
        tier_counts: Dict[str, int] = Counter()
        type_counts: Dict[str, int] = Counter()
        zone_counts: Dict[str, int] = Counter()

        active_count = 0
        durations: List[float] = []

        zone_intrusions = 0
        zone_dwell = 0
        ppe_viol = 0
        rapid_mov = 0
        veh_prox = 0
        fall_like = 0
        compound = 0

        for inc in incidents:
            tier_val = inc.risk_level.value if hasattr(inc.risk_level, "value") else str(inc.risk_level)
            type_val = inc.event_type.value if hasattr(inc.event_type, "value") else str(inc.event_type)

            tier_counts[tier_val] += 1
            type_counts[type_val] += 1

            if inc.lifecycle_state in (EventLifecycleState.ACTIVE, EventLifecycleState.CONFIRMED, "ACTIVE", "CONFIRMED"):
                active_count += 1

            if inc.location_or_zone and inc.location_or_zone.strip():
                zone_counts[inc.location_or_zone.strip()] += 1

            # Event category tracking
            if type_val == RiskEventType.RESTRICTED_ZONE_INTRUSION.value:
                zone_intrusions += 1
            elif type_val == RiskEventType.RESTRICTED_ZONE_DWELL.value:
                zone_dwell += 1
            elif type_val == RiskEventType.PPE_VIOLATION.value:
                ppe_viol += 1
            elif type_val == RiskEventType.RAPID_MOVEMENT_EVENT.value:
                rapid_mov += 1
            elif type_val in (RiskEventType.PERSON_VEHICLE_PROXIMITY.value, RiskEventType.APPROACHING_VEHICLE.value):
                veh_prox += 1
            elif type_val == RiskEventType.FALL_LIKE_EVENT.value:
                fall_like += 1
            elif type_val == RiskEventType.COMPOUND_SAFETY_EVENT.value:
                compound += 1

            # Duration tracking
            end_ts = inc.metadata.get("end_timestamp") if inc.metadata else None
            if end_ts and end_ts > inc.timestamp:
                durations.append(end_ts - inc.timestamp)

        avg_dur = round(sum(durations) / len(durations), 2) if durations else None

        most_active_zones = [
            {"zone_name": z_name, "incident_count": count}
            for z_name, count in zone_counts.most_common(5)
        ]

        return AnalyticsSummary(
            total_incidents=total,
            critical_incidents=tier_counts.get("CRITICAL", 0),
            high_risk_incidents=tier_counts.get("HIGH", 0),
            medium_risk_incidents=tier_counts.get("MEDIUM", 0),
            low_risk_incidents=tier_counts.get("LOW", 0),
            info_risk_incidents=tier_counts.get("INFO", 0),
            active_incidents=active_count,
            zone_intrusions=zone_intrusions,
            zone_dwell_exceeded=zone_dwell,
            ppe_violations=ppe_viol,
            rapid_movement_events=rapid_mov,
            vehicle_proximity_events=veh_prox,
            fall_like_events=fall_like,
            compound_events=compound,
            average_incident_duration_s=avg_dur,
            event_type_distribution=dict(type_counts),
            risk_tier_distribution=dict(tier_counts),
            most_active_zones=most_active_zones,
        )


_explanation_service_instance = ExplanationService()


def get_explanation_service() -> ExplanationService:
    """Returns the singleton instance of ExplanationService."""
    return _explanation_service_instance
