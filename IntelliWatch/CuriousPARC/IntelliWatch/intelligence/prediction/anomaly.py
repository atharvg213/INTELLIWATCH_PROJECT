"""
intelligence/prediction/anomaly.py
Step 11 - Explainable Temporal Anomaly Detection.
Detects erratic behavioral state transitions and temporal sequence anomalies
without unvalidated opaque neural models.
"""
from typing import Any, Dict, List, Optional
from collections import defaultdict, deque
import logging

from backend.schemas.behavior import BehaviorState, PrimaryBehavior
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.prediction.anomaly")


class TemporalAnomalyDetector:
    """
    Lightweight, deterministic detector of temporal behavioral sequence anomalies.
    Flags erratic locomotion shifts and abrupt kinematic collapse sequences.
    """

    def __init__(self, settings_override: Optional[Dict[str, Any]] = None):
        self._settings = get_settings()
        self._overrides = settings_override or {}
        # track_id -> deque of (timestamp, PrimaryBehavior)
        self._behavior_history: Dict[int, deque] = defaultdict(deque)

    def _get_setting(self, key: str, default: Any) -> Any:
        return self._overrides.get(key, getattr(self._settings, key, default))

    def evaluate_behavior_anomalies(
        self,
        behavior_states: Optional[List[BehaviorState]],
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        """
        Evaluates sliding behavior sequences for active tracks to detect temporal anomalies.
        """
        if not behavior_states:
            return []

        anomalies: List[Dict[str, Any]] = []
        window_size = int(self._get_setting("PREDICTION_ANOMALY_WINDOW", 5))

        for b in behavior_states:
            tid = b.track_id
            dq = self._behavior_history[tid]
            dq.append((timestamp, b.primary_behavior))
            while len(dq) > window_size:
                dq.popleft()

            if len(dq) < 3:
                continue

            seq = [item[1] for item in dq]

            # 1. Fall Sequence Anomaly:
            # Transition from active movement through sudden/fall into stationarity
            has_fall = any(s == PrimaryBehavior.POSSIBLE_FALL for s in seq)
            is_now_stationary = seq[-1] in (PrimaryBehavior.STATIONARY, PrimaryBehavior.PROLONGED_STATIONARY)
            was_previously_moving = any(s in (PrimaryBehavior.MOVING, PrimaryBehavior.RAPID_MOVEMENT, PrimaryBehavior.SUDDEN_MOVEMENT) for s in seq[:-1])

            if has_fall and is_now_stationary and was_previously_moving:
                anomalies.append({
                    "track_id": tid,
                    "entity_id": f"person_{tid}",
                    "anomaly_type": "FALL_AND_COLLAPSE_SEQUENCE",
                    "sequence": [s.value for s in seq],
                    "explanation": f"Worker #{tid} exhibited an abrupt movement-to-fall-to-stationary sequence.",
                })
                continue

            # 2. Erratic Locomotion Oscillation:
            # Alternating between moving/rapid and stationary >= 3 times in short window
            transitions = 0
            for i in range(1, len(seq)):
                if (seq[i] in (PrimaryBehavior.STATIONARY, PrimaryBehavior.PROLONGED_STATIONARY)) != (
                    seq[i - 1] in (PrimaryBehavior.STATIONARY, PrimaryBehavior.PROLONGED_STATIONARY)
                ):
                    transitions += 1

            if transitions >= 3:
                anomalies.append({
                    "track_id": tid,
                    "entity_id": f"person_{tid}",
                    "anomaly_type": "ERRATIC_BEHAVIOR_OSCILLATION",
                    "transitions": transitions,
                    "sequence": [s.value for s in seq],
                    "explanation": f"Worker #{tid} is oscillating erratically between locomotion and complete stationarity.",
                })

        return anomalies

    def purge_track(self, track_id: int) -> None:
        """Purges history for departed track."""
        self._behavior_history.pop(track_id, None)

    def reset(self) -> None:
        """Resets all history."""
        self._behavior_history.clear()
