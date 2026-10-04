from abc import ABC, abstractmethod
from typing import Any
import numpy as np


class BasePoseEstimator(ABC):
    """
    Abstract Base Class for human keypoint / pose estimation.
    Used for fall detection, ergonomical posture analysis, and worker fatigue.
    """

    @abstractmethod
    def estimate_pose(self, frame: np.ndarray) -> Any:
        """Extract human keypoints and skeletal joints from an input frame."""
        pass
