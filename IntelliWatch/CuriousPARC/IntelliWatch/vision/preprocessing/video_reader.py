import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Generator, Optional, Tuple, Union
import cv2
import numpy as np

from vision.preprocessing.frame import FrameData

logger = logging.getLogger("intelliwatch.video_reader")


class VideoProcessingError(Exception):
    """Base exception for video processing operations."""
    pass


class VideoFileNotFoundError(VideoProcessingError):
    """Raised when the specified video file does not exist."""
    pass


class VideoOpenError(VideoProcessingError):
    """Raised when OpenCV fails to open the video file or stream."""
    pass


class VideoReadError(VideoProcessingError):
    """Raised when an unrecoverable error occurs while reading a frame."""
    pass


@dataclass
class VideoMetadata:
    """Detailed metadata for an ingested video file or stream."""
    source_path: str
    fps: float
    width: int
    height: int
    total_frames: int
    duration_sec: float

    def __str__(self) -> str:
        return (
            f"VideoMetadata(source='{self.source_path}', "
            f"resolution={self.width}x{self.height}, "
            f"fps={self.fps:.2f}, total_frames={self.total_frames}, "
            f"duration={self.duration_sec:.2f}s)"
        )


class VideoReader:
    """
    Robust video ingestion component using OpenCV.
    Supports sequential frame reading, frame sampling, seeking, and safe resource cleanup.
    """

    def __init__(self, video_path: Union[str, Path]):
        self.video_path = Path(video_path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._metadata: Optional[VideoMetadata] = None
        self._current_frame_index: int = 0

    @property
    def is_open(self) -> bool:
        return self._cap is not None and self._cap.isOpened()

    @property
    def metadata(self) -> VideoMetadata:
        if self._metadata is None:
            raise VideoOpenError("Video must be opened before accessing metadata.")
        return self._metadata

    def open(self) -> VideoMetadata:
        """
        Validates and opens the video file, extracting stream metadata.
        """
        if not self.video_path.exists():
            msg = f"Video file not found at path: {self.video_path}"
            logger.error(msg)
            raise VideoFileNotFoundError(msg)

        logger.info(f"Opening video source: {self.video_path}")
        self._cap = cv2.VideoCapture(str(self.video_path))

        if not self._cap.isOpened():
            msg = f"OpenCV failed to open video source: {self.video_path}"
            logger.error(msg)
            self._cap.release()
            self._cap = None
            raise VideoOpenError(msg)

        fps = float(self._cap.get(cv2.CAP_PROP_FPS))
        if fps <= 0.0 or np.isnan(fps):
            fps = 30.0  # Fallback to standard 30 FPS if header metadata is missing

        width = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = (total_frames / fps) if fps > 0 and total_frames > 0 else 0.0

        self._metadata = VideoMetadata(
            source_path=str(self.video_path),
            fps=fps,
            width=width,
            height=height,
            total_frames=total_frames,
            duration_sec=duration_sec,
        )
        self._current_frame_index = 0

        logger.info(
            f"Successfully opened video [{self.video_path.name}]: "
            f"{width}x{height} @ {fps:.2f}fps, {total_frames} frames ({duration_sec:.2f}s)"
        )
        return self._metadata

    def read_frame(self) -> Tuple[bool, Optional[FrameData]]:
        """
        Reads the next sequential frame from the video source.
        Returns:
            (success, FrameData) where success is False if EOF or read failure.
        """
        if not self.is_open:
            raise VideoOpenError("Cannot read from an un-opened video reader.")

        frame_idx = int(self._cap.get(cv2.CAP_PROP_POS_FRAMES))
        ret, frame = self._cap.read()

        if not ret or frame is None or frame.size == 0:
            # End of video reached or frame decode error
            return False, None

        timestamp = (frame_idx / self._metadata.fps) if self._metadata and self._metadata.fps > 0 else 0.0
        h, w = frame.shape[:2]

        frame_data = FrameData(
            frame_index=frame_idx,
            timestamp=timestamp,
            image=frame,
            width=w,
            height=h,
            is_preprocessed=False,
            original_shape=(h, w),
        )
        self._current_frame_index = frame_idx + 1
        return True, frame_data

    def read_frames(self, frame_skip: int = 1) -> Generator[FrameData, None, None]:
        """
        Yields frames sequentially with optional frame sampling.

        Args:
            frame_skip: 1 = process every frame, 2 = every 2nd frame, 5 = every 5th frame.
        """
        if frame_skip < 1:
            raise ValueError(f"frame_skip must be >= 1, received {frame_skip}")

        if not self.is_open:
            self.open()

        current_idx = 0
        while self.is_open:
            ret, frame_data = self.read_frame()
            if not ret or frame_data is None:
                break

            if current_idx % frame_skip == 0:
                yield frame_data

            current_idx += 1

        logger.info(f"Finished reading video: reached end of stream at frame {current_idx}")

    def release(self) -> None:
        """Releases the underlying OpenCV video capture handle."""
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            logger.info(f"Released video capture for: {self.video_path}")

    def __enter__(self) -> "VideoReader":
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.release()
