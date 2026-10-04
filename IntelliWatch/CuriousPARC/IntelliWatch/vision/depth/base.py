from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

from backend.schemas.detection import BoundingBox
from backend.schemas.depth import DepthResult, ObjectDepth


class BaseDepthEstimator(ABC):
    """
    Abstract Base Class for monocular depth estimation perception modules.
    Provides standard interfaces for full-frame dense depth mapping and
    object-level spatial depth sampling.
    """

    @abstractmethod
    def estimate_depth(
        self,
        frame: Any,
        normalize: bool = True,
    ) -> Tuple[np.ndarray, DepthResult]:
        """
        Produces a dense 2D depth map and associated statistical DepthResult.

        Args:
            frame: Input frame (np.ndarray of shape (H, W, 3) or FrameData).
            normalize: Whether to normalize relative depth values to [0.0, 1.0].

        Returns:
            Tuple of (dense 2D depth map as float32 ndarray (H, W), DepthResult schema).
        """
        pass

    @abstractmethod
    def sample_object_depth(
        self,
        depth_map: np.ndarray,
        bbox: BoundingBox,
        class_name: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> ObjectDepth:
        """
        Samples depth distribution and representative ground contact depth for an object.

        Args:
            depth_map: Dense 2D depth map of native image dimensions (H, W).
            bbox: BoundingBox in original native image coordinates.
            class_name: Optional object classification label.
            track_id: Optional tracking identifier.

        Returns:
            ObjectDepth schema with statistical percentiles and contact depth.
        """
        pass
