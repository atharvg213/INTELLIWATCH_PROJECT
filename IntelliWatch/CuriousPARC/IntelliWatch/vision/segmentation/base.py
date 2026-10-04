from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BaseSegmenter(ABC):
    """
    Abstract Base Class for semantic / instance segmentation (e.g. hazardous zones, spills).
    """

    @abstractmethod
    def segment(self, frame: np.ndarray) -> Any:
        """Generate segmentation masks from an input frame."""
        pass
