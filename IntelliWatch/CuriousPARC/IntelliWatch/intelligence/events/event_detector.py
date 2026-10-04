from abc import ABC, abstractmethod
from typing import Any
from backend.schemas.tracking import FrameTracks
from backend.schemas.events import IndustrialEvent


class BaseEventDetector(ABC):
    """
    Abstract Base Class for multi-modal industrial safety event detection.
    """

    @abstractmethod
    def evaluate(
        self,
        tracks: FrameTracks,
        context: dict[str, Any]
    ) -> list[IndustrialEvent]:
        """Evaluate conditions and generate verified industrial events."""
        pass
