from abc import ABC, abstractmethod
from typing import Optional, Union
import numpy as np
from backend.schemas.detection import FrameDetections
from vision.preprocessing.frame import FrameData


class BaseDetector(ABC):
    """
    Abstract Base Class for all object detectors in IntelliWatch.
    Future implementations (e.g. YOLOv8, YOLOv11) must adhere to this interface.
    """

    @abstractmethod
    def load_model(self, model_path: str) -> None:
        """Load model weights and configure inference engine."""
        pass

    @abstractmethod
    def detect(
        self,
        frame: Union[np.ndarray, FrameData],
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> FrameDetections:
        """
        Execute detection on a single RGB/BGR frame or preprocessed FrameData.

        Args:
            frame: Numpy ndarray or FrameData representing the image frame.
            frame_id: Monotonically increasing frame index (optional if frame is FrameData).
            timestamp: Video timestamp in seconds (optional if frame is FrameData).

        Returns:
            FrameDetections containing bounding boxes, confidence, and class labels.
        """
        pass
