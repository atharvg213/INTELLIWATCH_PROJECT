"""
intelligence/prediction/serializer.py
Step 11 - Serialization utility for Predictive & Early-Warning Intelligence outputs.
"""
from typing import Any, Dict
from backend.schemas.prediction import (
    EarlyWarningIndicator,
    FramePredictionAssessment,
    ProjectedTrajectory,
)


class PredictionAssessmentSerializer:
    """
    Serializes FramePredictionAssessment, EarlyWarningIndicator, and ProjectedTrajectory to JSON-safe dicts.
    """

    @staticmethod
    def to_dict(assessment: FramePredictionAssessment) -> Dict[str, Any]:
        """Convert FramePredictionAssessment to pure dict."""
        return assessment.model_dump()

    @staticmethod
    def to_json(assessment: FramePredictionAssessment, indent: int = 2) -> str:
        """Convert FramePredictionAssessment to JSON string."""
        return assessment.model_dump_json(indent=indent)

    @staticmethod
    def indicator_to_dict(indicator: EarlyWarningIndicator) -> Dict[str, Any]:
        """Convert EarlyWarningIndicator to pure dict."""
        return indicator.model_dump()

    @staticmethod
    def trajectory_to_dict(trajectory: ProjectedTrajectory) -> Dict[str, Any]:
        """Convert ProjectedTrajectory to pure dict."""
        return trajectory.model_dump()
