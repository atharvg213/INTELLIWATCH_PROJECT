import logging
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union
import cv2
import numpy as np

from vision.preprocessing.frame import FrameData

logger = logging.getLogger("intelliwatch.frame_processor")


class BaseFramePreprocessor(ABC):
    """
    Abstract Base Class for frame normalization, letterboxing, and color conversion.
    """

    @abstractmethod
    def preprocess(self, frame_input: Union[np.ndarray, FrameData]) -> Union[np.ndarray, FrameData]:
        """Transform raw CCTV frame into normalized model input."""
        pass


class FramePreprocessor(BaseFramePreprocessor):
    """
    Industrial-grade frame preprocessing utility.
    Supports:
      - Direct resizing or aspect-ratio preserving letterboxing (with padding).
      - Recording padding and scaling factors for accurate future bbox mapping.
      - Optional color space conversion (BGR to RGB).
      - Optional pixel intensity normalization [0.0, 1.0].
    """

    def __init__(
        self,
        target_width: int = 640,
        target_height: int = 640,
        resize_enabled: bool = True,
        preserve_aspect_ratio: bool = True,
        normalize: bool = False,
        to_rgb: bool = False,
        pad_color: Tuple[int, int, int] = (114, 114, 114),
    ):
        if target_width <= 0 or target_height <= 0:
            raise ValueError(f"Target dimensions must be positive, got {target_width}x{target_height}")

        self.target_width = target_width
        self.target_height = target_height
        self.resize_enabled = resize_enabled
        self.preserve_aspect_ratio = preserve_aspect_ratio
        self.normalize = normalize
        self.to_rgb = to_rgb
        self.pad_color = pad_color

    def letterbox(
        self,
        image: np.ndarray,
        target_shape: Tuple[int, int],
        color: Tuple[int, int, int] = (114, 114, 114),
    ) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        """
        Resizes and pads image while preserving aspect ratio.
        Returns:
            (padded_image, scale_factor, (pad_top, pad_left))
        """
        shape = image.shape[:2]  # (height, width)
        target_h, target_w = target_shape

        # Scale ratio (new / old)
        r = min(target_w / shape[1], target_h / shape[0])

        # Compute unpadded new dimensions
        new_unpad_w = int(round(shape[1] * r))
        new_unpad_h = int(round(shape[0] * r))
        dw = target_w - new_unpad_w
        dh = target_h - new_unpad_h

        # Divide padding equally on both sides
        top = dh // 2
        bottom = dh - top
        left = dw // 2
        right = dw - left

        if shape[::-1] != (new_unpad_w, new_unpad_h):
            image = cv2.resize(image, (new_unpad_w, new_unpad_h), interpolation=cv2.INTER_LINEAR)

        padded = cv2.copyMakeBorder(
            image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
        )
        return padded, r, (top, left)

    def preprocess(
        self,
        frame_input: Union[np.ndarray, FrameData]
    ) -> Union[np.ndarray, FrameData]:
        """
        Preprocesses either a raw numpy frame or a structured FrameData instance.
        """
        is_frame_data = isinstance(frame_input, FrameData)
        img = frame_input.image if is_frame_data else frame_input

        if img is None or not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("Input frame must be a non-empty numpy.ndarray.")

        orig_h, orig_w = img.shape[:2]
        processed = img
        scale_factor = 1.0
        pad_offset = (0, 0)

        # 1. Color conversion if requested
        if self.to_rgb and processed.ndim == 3 and processed.shape[2] == 3:
            processed = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)

        # 2. Resizing / Letterboxing
        if self.resize_enabled:
            if self.preserve_aspect_ratio:
                processed, scale_factor, pad_offset = self.letterbox(
                    processed, (self.target_height, self.target_width), self.pad_color
                )
            else:
                processed = cv2.resize(
                    processed, (self.target_width, self.target_height), interpolation=cv2.INTER_LINEAR
                )
                scale_factor = min(self.target_width / orig_w, self.target_height / orig_h)
                pad_offset = (0, 0)

        # 3. Optional Normalization to [0.0, 1.0]
        if self.normalize:
            processed = processed.astype(np.float32) / 255.0

        if is_frame_data:
            return FrameData(
                frame_index=frame_input.frame_index,
                timestamp=frame_input.timestamp,
                image=processed,
                width=processed.shape[1],
                height=processed.shape[0],
                is_preprocessed=True,
                original_shape=(orig_h, orig_w),
                scale_factor=scale_factor,
                pad_offset=pad_offset,
            )
        return processed
