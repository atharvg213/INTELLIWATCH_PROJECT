"""
backend/services/prediction_store.py
In-memory singleton storage for the most recently evaluated FramePredictionAssessment.
Provides fast O(1) retrieval for early-warning intelligence inspection API.
"""
from typing import Optional
from backend.schemas.prediction import FramePredictionAssessment


class PredictionStore:
    """Thread-safe / asynchronous lightweight in-memory cache for current prediction assessment."""

    def __init__(self):
        self._current_assessment: Optional[FramePredictionAssessment] = None

    def set_current_assessment(self, assessment: FramePredictionAssessment) -> None:
        self._current_assessment = assessment

    def get_current_assessment(self) -> FramePredictionAssessment:
        if self._current_assessment is not None:
            return self._current_assessment
        return FramePredictionAssessment(
            frame_id=0,
            timestamp=0.0,
            camera_id="cam_01",
            projected_trajectories=[],
            active_indicators=[],
            recent_indicators=[],
            total_active_indicators=0,
        )

    def reset(self) -> None:
        self._current_assessment = None


_prediction_store_instance = PredictionStore()


def get_prediction_store() -> PredictionStore:
    """Returns the singleton instance of the in-memory PredictionStore."""
    return _prediction_store_instance
