from abc import ABC, abstractmethod
from typing import Optional
import numpy as np
from backend.schemas.detection import FrameDetections
from backend.schemas.tracking import FrameTracks


class BaseTracker(ABC):
    """
    Abstract Base Class for Multi-Object Trackers.
    Responsible for maintaining identity across frames and recording trajectories.
    """

    @abstractmethod
    def update(
        self,
        detections: FrameDetections,
        frame: Optional[np.ndarray] = None
    ) -> FrameTracks:
        """
        Associate current frame detections with existing persistent tracks.

        Args:
            detections: FrameDetections from the detector module.
            frame: Optional frame image (used by optical flow or appearance-based trackers).

        Returns:
            FrameTracks with persistent track IDs and updated trajectory history.
        """
        pass

    @abstractmethod
    def reset(self) -> None:
        """Reset all tracking state and active tracks."""
        pass
