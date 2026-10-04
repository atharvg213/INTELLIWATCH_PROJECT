"""
backend/services/risk_store.py
In-memory singleton storage for the most recently evaluated FrameRiskAssessment.
Provides fast O(1) retrieval for API endpoints.
"""
from typing import Optional
from backend.schemas.risk import FrameRiskAssessment, RiskSummary


class RiskStore:
    """Thread-safe / asynchronous lightweight in-memory cache for current risk assessment."""

    def __init__(self):
        self._current_assessment: Optional[FrameRiskAssessment] = None

    def set_current_assessment(self, assessment: FrameRiskAssessment) -> None:
        self._current_assessment = assessment

    def get_current_assessment(self) -> FrameRiskAssessment:
        if self._current_assessment is not None:
            return self._current_assessment
        return FrameRiskAssessment(
            frame_id=0,
            timestamp=0.0,
            camera_id="cam_01",
            active_events=[],
            recent_events=[],
            risk_factors=[],
            risk_summary=RiskSummary(),
        )

    def reset(self) -> None:
        self._current_assessment = None


_risk_store_instance = RiskStore()


def get_risk_store() -> RiskStore:
    """Returns the singleton instance of the in-memory RiskStore."""
    return _risk_store_instance
