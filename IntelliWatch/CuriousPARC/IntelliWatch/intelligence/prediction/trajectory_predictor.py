from abc import ABC, abstractmethod
from typing import Any
from backend.schemas.tracking import FrameTracks


class BasePredictor(ABC):
    """
    Abstract Base Class for near-future motion and collision prediction.
    """

    @abstractmethod
    def predict(
        self,
        tracks: FrameTracks,
        time_horizon_seconds: float = 2.0
    ) -> dict[int, list[Any]]:
        """Project track trajectories forward in time to anticipate hazards."""
        pass
