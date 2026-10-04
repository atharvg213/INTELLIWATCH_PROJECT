from abc import ABC, abstractmethod
from typing import Any
from backend.schemas.events import IndustrialEvent


class BaseRiskEngine(ABC):
    """
    Abstract Base Class for composite safety risk scoring.
    Computes an aggregated risk index (0.0 to 1.0) based on concurrent events.
    """

    @abstractmethod
    def calculate_risk_score(
        self,
        events: list[IndustrialEvent],
        scene_state: dict[str, Any]
    ) -> float:
        """Compute current environment threat / danger index."""
        pass
