import logging
import time
from pathlib import Path
from typing import Callable, Generator, Optional, Union

from vision.preprocessing.frame import FrameData
from vision.preprocessing.frame_processor import FramePreprocessor
from vision.preprocessing.video_reader import VideoMetadata, VideoReader

logger = logging.getLogger("intelliwatch.pipeline")


class VideoPipeline:
    """
    Decoupled video ingestion and frame preprocessing pipeline.
    Connects:
      VideoReader -> Frame Extraction -> Frame Sampling -> FramePreprocessor -> ProcessedFrame

    Designed to cleanly interchange video sources (files, webcams, RTSP streams)
    without modifying downstream vision/intelligence stages.
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        frame_skip: int = 1,
        preprocessor: Optional[FramePreprocessor] = None,
    ):
        self.video_path = Path(video_path)
        self.frame_skip = frame_skip
        self.preprocessor = preprocessor or FramePreprocessor()
        self.reader = VideoReader(self.video_path)

    @property
    def metadata(self) -> VideoMetadata:
        if not self.reader.is_open:
            self.reader.open()
        return self.reader.metadata

    def stream_frames(self) -> Generator[FrameData, None, None]:
        """
        Streams and yields preprocessed frames sequentially.
        """
        logger.info(
            f"Starting video processing pipeline for: {self.video_path.name} "
            f"(frame_skip={self.frame_skip}, target_dims={self.preprocessor.target_width}x{self.preprocessor.target_height})"
        )

        with self.reader:
            for raw_frame in self.reader.read_frames(frame_skip=self.frame_skip):
                processed_frame = self.preprocessor.preprocess(raw_frame)
                yield processed_frame

        logger.info("Video processing pipeline completed stream successfully.")

    def process_all(
        self,
        callback: Optional[Callable[[FrameData], None]] = None,
        log_interval: int = 30,
    ) -> dict:
        """
        Executes the entire video through the pipeline, invoking an optional
        callback on each frame, and returns performance telemetry.
        """
        start_time = time.perf_counter()
        frames_processed = 0

        logger.info(f"Executing batch pipeline on: {self.video_path}")
        for frame in self.stream_frames():
            frames_processed += 1
            if callback:
                callback(frame)
            if frames_processed % log_interval == 0:
                logger.info(
                    f"Processed frame #{frame.frame_index} (timestamp={frame.timestamp:.2f}s, count={frames_processed})"
                )

        elapsed = time.perf_counter() - start_time
        effective_fps = (frames_processed / elapsed) if elapsed > 0 else 0.0

        summary = {
            "source": str(self.video_path),
            "frames_processed": frames_processed,
            "elapsed_seconds": round(elapsed, 4),
            "effective_fps": round(effective_fps, 2),
            "frame_skip": self.frame_skip,
            "preprocessed_dimensions": f"{self.preprocessor.target_width}x{self.preprocessor.target_height}",
        }
        logger.info(f"Pipeline finished: {frames_processed} frames in {elapsed:.2f}s ({effective_fps:.1f} FPS)")
        return summary
