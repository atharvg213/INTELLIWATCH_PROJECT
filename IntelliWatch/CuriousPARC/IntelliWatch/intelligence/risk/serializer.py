"""
intelligence/risk/serializer.py
Step 10 - Serialization utility for Risk & Event Reasoning outputs.
Provides clean JSON-serializable dictionaries and schemas.
"""
from typing import Any, Dict
from backend.schemas.risk import FrameRiskAssessment, RiskEvent, RiskFactor


class RiskAssessmentSerializer:
    """
    Serializes FrameRiskAssessment, RiskEvent, and RiskFactor into JSON-safe dictionaries.
    """

    @staticmethod
    def to_dict(assessment: FrameRiskAssessment) -> Dict[str, Any]:
        """Convert a FrameRiskAssessment instance to a pure dict."""
        return assessment.model_dump()

    @staticmethod
    def to_json(assessment: FrameRiskAssessment, indent: int = 2) -> str:
        """Convert a FrameRiskAssessment instance to a formatted JSON string."""
        return assessment.model_dump_json(indent=indent)

    @staticmethod
    def event_to_dict(event: RiskEvent) -> Dict[str, Any]:
        """Convert a single RiskEvent to a pure dict."""
        return event.model_dump()

    @staticmethod
    def factor_to_dict(factor: RiskFactor) -> Dict[str, Any]:
        """Convert a single RiskFactor to a pure dict."""
        return factor.model_dump()
