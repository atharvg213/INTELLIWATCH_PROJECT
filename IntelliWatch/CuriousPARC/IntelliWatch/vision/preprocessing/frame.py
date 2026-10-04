from dataclasses import dataclass, field
from typing import Optional, Tuple
import numpy as np


@dataclass
class FrameData:
    """
    Lightweight, high-performance in-memory representation of a video frame.
    Separates heavy NumPy image matrices from serialization schemas.
    """
    frame_index: int
    timestamp: float  # In seconds from start of video/stream
    image: np.ndarray  # OpenCV image array (H, W, C) in BGR or RGB format
    width: int
    height: int
    is_preprocessed: bool = False
    original_shape: Optional[Tuple[int, int]] = None  # (height, width) before resize
    scale_factor: Optional[float] = None  # Resize scale factor
    pad_offset: Optional[Tuple[int, int]] = None  # (pad_top, pad_left) for letterbox

    @property
    def channels(self) -> int:
        return self.image.shape[2] if self.image.ndim == 3 else 1

    @property
    def shape(self) -> Tuple[int, ...]:
        return self.image.shape

    def copy(self) -> "FrameData":
        """Creates a shallow copy with a cloned image buffer."""
        return FrameData(
            frame_index=self.frame_index,
            timestamp=self.timestamp,
            image=self.image.copy(),
            width=self.width,
            height=self.height,
            is_preprocessed=self.is_preprocessed,
            original_shape=self.original_shape,
            scale_factor=self.scale_factor,
            pad_offset=self.pad_offset,
        )
