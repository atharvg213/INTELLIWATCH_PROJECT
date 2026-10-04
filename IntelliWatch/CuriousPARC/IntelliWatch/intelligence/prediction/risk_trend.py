"""
intelligence/prediction/risk_trend.py
Step 11 - Risk Trend and Temporal Escalation Analysis.
Tracks sliding-window risk scores and factor persistence across frames.
Identifies sustained risk escalation and persistent unsafe operational patterns.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
from collections import defaultdict, deque
import logging

from backend.schemas.risk import FrameRiskAssessment, RiskEvent, RiskFactor, RiskLevel
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.prediction.risk_trend")


class _EntityRiskSnapshot:
    """Historical observation of an entity's risk state at a single frame."""

    def __init__(
        self,
        timestamp: float,
        score: float,
        level: RiskLevel,
        active_event_types: Set[str],
        active_factor_types: Set[str],
    ):
        self.timestamp = timestamp
        self.score = score
        self.level = level
        self.active_event_types = active_event_types
        self.active_factor_types = active_factor_types


class RiskTrendAnalyzer:
    """
    Analyzes temporal trends in risk evaluations across consecutive observations.
    Detects sustained escalation and recurrent violation patterns.
    """

    def __init__(self, settings_override: Optional[Dict[str, Any]] = None):
        self._settings = get_settings()
        self._overrides = settings_override or {}
        # entity_id -> deque of _EntityRiskSnapshot
        self._history: Dict[str, deque] = defaultdict(deque)

    def _get_setting(self, key: str, default: Any) -> Any:
        return self._overrides.get(key, getattr(self._settings, key, default))

    def update_and_evaluate(
        self,
        risk_assessment: FrameRiskAssessment,
    ) -> Dict[str, List[Dict[str, Any]]]:
        """
        Updates sliding history and evaluates risk trend indicators for each active entity.

        Returns:
            Dict mapping indicator_type -> list of evidence dictionaries
        """
        cur_ts = risk_assessment.timestamp
        window_size = int(self._get_setting("PREDICTION_RISK_HISTORY_FRAMES", 6))
        escalation_thresh = float(self._get_setting("PREDICTION_RISK_ESCALATION_THRESHOLD", 20.0))
        persistence_thresh = int(self._get_setting("PREDICTION_PERSISTENCE_THRESHOLD", 4))

        # 1. Map active events and factors by entity
        entity_events: Dict[str, Set[str]] = defaultdict(set)
        entity_factors: Dict[str, Set[str]] = defaultdict(set)
        entity_scores: Dict[str, float] = defaultdict(float)
        entity_levels: Dict[str, RiskLevel] = defaultdict(lambda: RiskLevel.INFO)

        for ev in risk_assessment.active_events:
            for ent in ev.involved_entities:
                entity_events[ent].add(ev.event_type.value)
                if ev.risk_score > entity_scores[ent]:
                    entity_scores[ent] = ev.risk_score
                    entity_levels[ent] = ev.risk_level

        for f in risk_assessment.risk_factors:
            for ent in f.involved_entity_ids:
                entity_factors[ent].add(f.factor_type.value)

        # Update entity history
        observed_entities = set(entity_scores.keys()) | set(entity_factors.keys())
        for ent in observed_entities:
            snap = _EntityRiskSnapshot(
                timestamp=cur_ts,
                score=entity_scores[ent],
                level=entity_levels[ent],
                active_event_types=entity_events[ent],
                active_factor_types=entity_factors[ent],
            )
            dq = self._history[ent]
            dq.append(snap)
            while len(dq) > window_size:
                dq.popleft()

        detected_trends: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

        # 2. Evaluate trends per entity
        for ent, dq in self._history.items():
            if len(dq) < 3:
                continue

            snaps = list(dq)
            first_score = snaps[0].score
            last_score = snaps[-1].score
            score_delta = last_score - first_score

            # A. Check for RISK_ESCALATING
            # Requires positive monotonic or overall trend with score increase exceeding threshold
            is_increasing = score_delta >= escalation_thresh and last_score > 0
            if is_increasing:
                detected_trends["RISK_ESCALATING"].append({
                    "entity_id": ent,
                    "initial_score": first_score,
                    "current_score": last_score,
                    "score_delta": round(score_delta, 1),
                    "window_observations": len(snaps),
                    "duration_seconds": round(snaps[-1].timestamp - snaps[0].timestamp, 2),
                    "current_level": snaps[-1].level.value,
                })

            # B. Check for PERSISTENT_UNSAFE_PATTERN
            # Consecutive frames with Medium/High/Critical risk
            unsafe_count = sum(1 for s in snaps if s.level in (RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL))
            if unsafe_count >= persistence_thresh:
                active_events = list(snaps[-1].active_event_types)
                detected_trends["PERSISTENT_UNSAFE_PATTERN"].append({
                    "entity_id": ent,
                    "unsafe_observations": unsafe_count,
                    "window_size": len(snaps),
                    "active_events": active_events,
                    "current_level": snaps[-1].level.value,
                })

            # C. Check for REPEATED_PPE_VIOLATION
            ppe_hits = sum(1 for s in snaps if "PPE_VIOLATION" in s.active_event_types or "PPE_NON_COMPLIANCE" in s.active_factor_types)
            if ppe_hits >= persistence_thresh:
                detected_trends["REPEATED_PPE_VIOLATION"].append({
                    "entity_id": ent,
                    "violation_count": ppe_hits,
                    "window_size": len(snaps),
                })

            # D. Check for REPEATED_ZONE_VIOLATION
            zone_hits = sum(1 for s in snaps if "RESTRICTED_ZONE_INTRUSION" in s.active_event_types or "RESTRICTED_ZONE_INTRUSION" in s.active_factor_types)
            if zone_hits >= persistence_thresh:
                detected_trends["REPEATED_ZONE_VIOLATION"].append({
                    "entity_id": ent,
                    "violation_count": zone_hits,
                    "window_size": len(snaps),
                })

            # E. Check for PERSISTENT_VEHICLE_PROXIMITY
            veh_prox_hits = sum(1 for s in snaps if "PERSON_VEHICLE_PROXIMITY" in s.active_event_types or "PERSON_VEHICLE_PROXIMITY" in s.active_factor_types)
            if veh_prox_hits >= persistence_thresh:
                detected_trends["PERSISTENT_VEHICLE_PROXIMITY"].append({
                    "entity_id": ent,
                    "proximity_count": veh_prox_hits,
                    "window_size": len(snaps),
                })

            # F. Check for PERSISTENT_MACHINE_PROXIMITY
            mach_prox_hits = sum(1 for s in snaps if "WORKER_MACHINE_RISK" in s.active_event_types or "WORKER_NEAR_MACHINE" in s.active_factor_types)
            if mach_prox_hits >= persistence_thresh:
                detected_trends["PERSISTENT_MACHINE_PROXIMITY"].append({
                    "entity_id": ent,
                    "proximity_count": mach_prox_hits,
                    "window_size": len(snaps),
                })

            # G. Check for REPEATED_COMPOUND_RISK
            compound_hits = sum(1 for s in snaps if "COMPOUND_SAFETY_EVENT" in s.active_event_types)
            if compound_hits >= 2:
                detected_trends["REPEATED_COMPOUND_RISK"].append({
                    "entity_id": ent,
                    "compound_event_count": compound_hits,
                    "window_size": len(snaps),
                })

        return detected_trends

    def purge_entity(self, entity_id: str) -> None:
        """Purges history for departed entity."""
        self._history.pop(entity_id, None)

    def reset(self) -> None:
        """Resets all history."""
        self._history.clear()
