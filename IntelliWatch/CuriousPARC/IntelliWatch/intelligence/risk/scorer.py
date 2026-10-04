"""
intelligence/risk/scorer.py
Step 10 - Transparent Deterministic Risk Scorer.
Calculates numeric risk scores and categorizes into controlled RiskLevel tiers
(INFO, LOW, MEDIUM, HIGH, CRITICAL) using configurable thresholds.
Avoids arbitrary unexplained numbers; every point addition is explainable and traceable.
"""
from typing import Any, Dict, List, Optional, Tuple
import logging

from backend.schemas.risk import RiskFactor, RiskLevel
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.risk.scorer")


class RiskScorer:
    """
    Transparent, explainable risk scoring calculator.
    Combines individual factor severity contributions with documented escalation
    rules for multi-factor compound hazards.
    """

    def __init__(self, settings_override: Optional[Dict[str, Any]] = None):
        self._settings = get_settings()
        self._overrides = settings_override or {}

    def _get_setting(self, key: str, default: Any) -> Any:
        return self._overrides.get(key, getattr(self._settings, key, default))

    def compute_risk(
        self,
        factors: List[RiskFactor],
        is_compound: bool = False,
    ) -> Tuple[float, RiskLevel, Dict[str, Any]]:
        """
        Computes composite numeric score, RiskLevel, and transparent explanation breakdown.

        Returns:
            Tuple of (final_score, risk_level, score_breakdown)
        """
        if not factors:
            return 0.0, RiskLevel.INFO, {"base_score": 0.0, "factor_contributions": [], "escalation": 0.0}

        # 1. Base sum of individual factor contributions
        factor_items = []
        base_score = 0.0
        for f in factors:
            base_score += f.severity_contribution
            factor_items.append({
                "factor_type": f.factor_type.value,
                "contribution": f.severity_contribution,
                "explanation": f.explanation,
            })

        # 2. Multi-factor compounding escalation
        # If multiple concurrent independent factors are present for the entity / event
        escalation_per_factor = float(self._get_setting("RISK_SCORE_MULTI_FACTOR_ESCALATION", 15.0))
        escalation = 0.0
        if is_compound or len(factors) > 1:
            additional_factor_count = max(0, len(factors) - 1)
            escalation = additional_factor_count * escalation_per_factor

        total_score = base_score + escalation

        # 3. Classify into RiskLevel using configurable thresholds
        info_thresh = float(self._get_setting("RISK_LEVEL_INFO_THRESHOLD", 0.0))
        low_thresh = float(self._get_setting("RISK_LEVEL_LOW_THRESHOLD", 10.0))
        med_thresh = float(self._get_setting("RISK_LEVEL_MEDIUM_THRESHOLD", 30.0))
        high_thresh = float(self._get_setting("RISK_LEVEL_HIGH_THRESHOLD", 60.0))
        crit_thresh = float(self._get_setting("RISK_LEVEL_CRITICAL_THRESHOLD", 85.0))

        if total_score >= crit_thresh:
            level = RiskLevel.CRITICAL
        elif total_score >= high_thresh:
            level = RiskLevel.HIGH
        elif total_score >= med_thresh:
            level = RiskLevel.MEDIUM
        elif total_score >= low_thresh:
            level = RiskLevel.LOW
        else:
            level = RiskLevel.INFO

        breakdown = {
            "base_score": base_score,
            "escalation": escalation,
            "total_score": total_score,
            "risk_level": level.value,
            "thresholds": {
                "INFO": info_thresh,
                "LOW": low_thresh,
                "MEDIUM": med_thresh,
                "HIGH": high_thresh,
                "CRITICAL": crit_thresh,
            },
            "factor_contributions": factor_items,
        }

        return total_score, level, breakdown
