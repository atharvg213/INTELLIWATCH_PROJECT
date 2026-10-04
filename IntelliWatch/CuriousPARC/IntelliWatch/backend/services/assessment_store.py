"""
backend/services/assessment_store.py
In-memory singleton storage for the most recently evaluated FrameAssessment.
Provides fast O(1) retrieval for real-time operator dashboard and inspection APIs.
"""
from typing import Optional
from backend.schemas.assessment import FrameAssessment


class AssessmentStore:
    """Thread-safe / asynchronous lightweight in-memory cache for current frame assessment."""

    def __init__(self):
        self._current_assessment: Optional[FrameAssessment] = None

    def set_current_assessment(self, assessment: FrameAssessment) -> None:
        self._current_assessment = assessment

    def get_current_assessment(self) -> FrameAssessment:
        if self._current_assessment is not None:
            return self._current_assessment
        # Return default empty assessment if no frames have been processed yet
        return FrameAssessment(
            frame_id=0,
            timestamp=0.0,
            camera_id="cam_01",
            frame_width=1280,
            frame_height=720,
            processing_time_ms=0.0,
            detections=[],
            tracks=[],
            worker_inventories=[],
            zone_memberships=[],
            behavior_states=[],
            active_events=[],
            active_early_warnings=[],
        )

    def reset(self) -> None:
        self._current_assessment = None


_assessment_store_instance = AssessmentStore()


def get_assessment_store() -> AssessmentStore:
    """Returns the singleton instance of the AssessmentStore."""
    return _assessment_store_instance
