"""
intelligence/prediction/indicators.py
Step 11 - Early-Warning Indicator candidate synthesizer.
Constructs CandidateIndicator objects from trajectory, risk trend, and anomaly outputs.
"""
from typing import Any, Dict, List, Optional
import logging

from backend.schemas.prediction import (
    EarlyWarningIndicatorType,
    IndicatorLifecycleState,
    IndicatorSeverity,
    ProjectedTrajectory,
)

logger = logging.getLogger("intelliwatch.prediction.indicators")


class CandidateIndicator:
    """Internal candidate representation for early-warning indicators."""

    def __init__(
        self,
        indicator_key: str,
        indicator_type: EarlyWarningIndicatorType,
        track_ids: List[int],
        entity_ids: List[str],
        severity: IndicatorSeverity,
        explanation: str,
        evidence: Dict[str, Any],
        target_zone_id: Optional[str] = None,
        projected_trajectory: Optional[ProjectedTrajectory] = None,
        trend: Optional[str] = None,
        supporting_events: Optional[List[str]] = None,
        supporting_risk_factors: Optional[List[str]] = None,
    ):
        self.indicator_key = indicator_key
        self.indicator_type = indicator_type
        self.track_ids = track_ids
        self.entity_ids = entity_ids
        self.severity = severity
        self.explanation = explanation
        self.evidence = evidence
        self.target_zone_id = target_zone_id
        self.projected_trajectory = projected_trajectory
        self.trend = trend
        self.supporting_events = supporting_events or []
        self.supporting_risk_factors = supporting_risk_factors or []
