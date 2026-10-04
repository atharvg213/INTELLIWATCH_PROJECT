import logging
import time
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
import numpy as np

from backend.schemas.detection import FrameDetections
from vision.detection.visualizer import DetectionVisualizer
from vision.detection.yolo_detector import YOLODetector
from vision.preprocessing.frame import FrameData
from vision.preprocessing.frame_processor import FramePreprocessor
from vision.preprocessing.pipeline import VideoPipeline

logger = logging.getLogger("intelliwatch.detection_pipeline")


class DetectionPipeline:
    """
    End-to-End Object Detection Pipeline.
    Connects:
      Video File / Stream
           ↓
      VideoReader (Sequential Ingestion)
           ↓
      Frame Sampling (frame_skip)
           ↓
      FramePreprocessor (Letterbox 640x640)
           ↓
      FrameData (Typed In-Memory Representation)
           ↓
      YOLODetector (Inference & Coordinate Re-projection)
           ↓
      FrameDetections (Pydantic Schema in Original Video Resolution)
           ↓
      DetectionVisualizer (Optional Frame Annotation & Storage)
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        detector: Optional[YOLODetector] = None,
        frame_skip: int = 1,
        preprocessor: Optional[FramePreprocessor] = None,
        visualizer: Optional[DetectionVisualizer] = None,
    ):
        self.video_path = Path(video_path)
        self.frame_skip = frame_skip
        self.preprocessor = preprocessor or FramePreprocessor(
            target_width=640,
            target_height=640,
            resize_enabled=True,
            preserve_aspect_ratio=True,
        )
        self.video_pipeline = VideoPipeline(
            video_path=self.video_path,
            frame_skip=self.frame_skip,
            preprocessor=self.preprocessor,
        )
        self.detector = detector or YOLODetector()
        self.visualizer = visualizer or DetectionVisualizer()

    def stream_detections(
        self,
        render_visualization: bool = False,
    ) -> Generator[Tuple[FrameData, FrameDetections, Optional[np.ndarray]], None, None]:
        """
        Streams preprocessed frames and corresponding object detections.

        Yields:
            (FrameData, FrameDetections, Optional[np.ndarray])
        """
        logger.info(
            f"Starting detection stream on [{self.video_path.name}] "
            f"(skip={self.frame_skip}, model={self.detector.model_path}, conf={self.detector.confidence_threshold})"
        )

        for frame_data in self.video_pipeline.stream_frames():
            detections = self.detector.detect(frame_data)
            annotated_frame = None

            if render_visualization and self.visualizer:
                # Annotate on the current frame image
                annotated_frame = self.visualizer.draw_detections(
                    image=frame_data.image,
                    detections=detections,
                )

            yield frame_data, detections, annotated_frame

    def process_all(
        self,
        callback: Optional[Callable[[FrameData, FrameDetections], None]] = None,
        save_output_dir: Optional[Union[str, Path]] = None,
        max_save_frames: int = 0,
        log_interval: int = 15,
    ) -> Dict:
        """
        Executes detection over all sampled frames in the video,
        optionally saving sample annotated frames and logging progress.

        Returns:
            Dict of execution and performance statistics.
        """
        start_time = time.perf_counter()
        frames_processed = 0
        total_detections = 0
        saved_frames_count = 0
        class_distribution: Dict[str, int] = {}

        out_dir = Path(save_output_dir) if save_output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Executing batch detection pipeline on {self.video_path}")

        for frame_data, detections, _ in self.stream_detections(render_visualization=False):
            frames_processed += 1
            num_dets = len(detections.detections)
            total_detections += num_dets

            for d in detections.detections:
                class_distribution[d.class_name] = class_distribution.get(d.class_name, 0) + 1

            # Save annotated frame if requested
            if out_dir and saved_frames_count < max_save_frames:
                annotated = self.visualizer.draw_detections(frame_data.image, detections)
                out_path = out_dir / f"step3_detection_frame_{frame_data.frame_index:04d}.jpg"
                self.visualizer.save_annotated_frame(annotated, out_path)
                saved_frames_count += 1

            if callback:
                callback(frame_data, detections)

            if frames_processed % log_interval == 0:
                logger.info(
                    f"Processed frame #{frame_data.frame_index} "
                    f"(timestamp={frame_data.timestamp:.2f}s, detections={num_dets}, total_seen={total_detections})"
                )

        elapsed = time.perf_counter() - start_time
        fps = (frames_processed / elapsed) if elapsed > 0 else 0.0

        summary = {
            "source": str(self.video_path),
            "model": self.detector.model_path,
            "confidence_threshold": self.detector.confidence_threshold,
            "frames_processed": frames_processed,
            "total_detections": total_detections,
            "avg_detections_per_frame": round(total_detections / frames_processed, 2) if frames_processed > 0 else 0.0,
            "class_distribution": class_distribution,
            "elapsed_seconds": round(elapsed, 4),
            "effective_fps": round(fps, 2),
            "saved_frames": saved_frames_count,
        }

        logger.info(
            f"Detection completed: {frames_processed} frames, {total_detections} detections "
            f"in {elapsed:.2f}s ({fps:.1f} FPS)"
        )
        return summary
