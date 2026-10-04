from abc import ABC, abstractmethod
from backend.schemas.tracking import FrameTracks
from backend.schemas.events import IndustrialEvent


class BaseBehaviorAnalyzer(ABC):
    """
    Abstract Base Class for temporal behavior understanding.
    Evaluates: loitering, worker erratic motion, unauthorized running.
    """

    @abstractmethod
    def analyze(self, tracks: FrameTracks) -> list[IndustrialEvent]:
        """Analyze temporal movement dynamics across recent frames."""
        pass
