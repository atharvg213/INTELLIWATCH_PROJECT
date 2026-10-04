import logging
import time
from pathlib import Path
from typing import Callable, Dict, Generator, List, Optional, Tuple, Union
import cv2
import numpy as np

from backend.schemas.detection import FrameDetections
from backend.schemas.tracking import FrameTracks
from configs.settings import get_settings
from vision.detection.yolo_detector import YOLODetector
from vision.preprocessing.frame import FrameData
from vision.preprocessing.frame_processor import FramePreprocessor
from vision.preprocessing.pipeline import VideoPipeline
from vision.tracking.base import BaseTracker
from vision.tracking.bytetrack_tracker import ByteTrackTracker
from vision.tracking.visualizer import TrackingVisualizer

logger = logging.getLogger("intelliwatch.tracking_pipeline")


class TrackingPipeline:
    """
    End-to-End Multi-Object Tracking Pipeline.
    Architectural Dataflow:
      Video File / CCTV Stream
           ↓
      VideoReader (Sequential Ingestion)
           ↓
      Frame Sampling (frame_skip)
           ↓
      FramePreprocessor (Letterbox 640x640)
           ↓
      FrameData (Typed In-Memory Representation)
           ↓
      YOLODetector (Inference & Coordinate Re-projection to Original Video Resolution)
           ↓
      FrameDetections (Decoupled Schema in Original Video Resolution)
           ↓
      ByteTrackTracker (Persistent Association, Trajectories, Motion & Lifecycles)
           ↓
      FrameTracks (Persistent Track IDs, Centroids & Trajectories)
           ↓
      TrackingVisualizer (ID Badges, Trajectory Trails, Video Annotation)
    """

    def __init__(
        self,
        video_path: Union[str, Path],
        detector: Optional[YOLODetector] = None,
        tracker: Optional[BaseTracker] = None,
        frame_skip: int = 1,
        preprocessor: Optional[FramePreprocessor] = None,
        visualizer: Optional[TrackingVisualizer] = None,
    ):
        self.video_path = Path(video_path)
        self.frame_skip = frame_skip
        settings = get_settings()

        self.preprocessor = preprocessor or FramePreprocessor(
            target_width=settings.TARGET_WIDTH,
            target_height=settings.TARGET_HEIGHT,
            resize_enabled=settings.RESIZE_ENABLED,
            preserve_aspect_ratio=settings.PRESERVE_ASPECT_RATIO,
        )

        self.video_pipeline = VideoPipeline(
            video_path=self.video_path,
            frame_skip=self.frame_skip,
            preprocessor=self.preprocessor,
        )

        self.detector = detector or YOLODetector()
        self.tracker = tracker or ByteTrackTracker(
            track_history_length=settings.TRACK_HISTORY_LENGTH,
            track_buffer=settings.TRACK_BUFFER,
            match_thresh=settings.TRACK_MATCH_THRESH,
        )
        self.visualizer = visualizer or TrackingVisualizer()

    def stream_tracks(
        self,
        render_visualization: bool = False,
    ) -> Generator[Tuple[FrameData, FrameDetections, FrameTracks, Optional[np.ndarray]], None, None]:
        """
        Sequentially processes video frames, running detection followed by ByteTrack tracking.
        Maintains persistent tracking state across frames.

        Yields:
            (FrameData, FrameDetections, FrameTracks, Optional[np.ndarray])
        """
        logger.info(
            f"Starting tracking stream on [{self.video_path.name}] "
            f"(skip={self.frame_skip}, detector={self.detector.model_path}, tracker={type(self.tracker).__name__})"
        )

        for frame_data in self.video_pipeline.stream_frames():
            # Step 3: Run YOLO detection and map coordinates back to original video pixels
            detections = self.detector.detect(frame_data)

            # Step 4: Associate detections into persistent tracks using ByteTrack
            tracks = self.tracker.update(detections=detections)

            annotated_frame = None
            if render_visualization and self.visualizer:
                # Render tracking visualization directly onto the unpadded original/working frame
                annotated_frame = self.visualizer.draw_tracks(
                    image=frame_data.image,
                    tracks=tracks,
                )

            yield frame_data, detections, tracks, annotated_frame

    def process_all(
        self,
        save_video_path: Optional[Union[str, Path]] = None,
        save_output_dir: Optional[Union[str, Path]] = None,
        max_save_frames: int = 0,
        log_interval: int = 15,
        callback: Optional[Callable[[FrameData, FrameDetections, FrameTracks], None]] = None,
    ) -> Dict:
        """
        Executes end-to-end tracking over the entire video sequence.
        Optionally writes an annotated MP4 video and sample frame images.

        Returns:
            Dictionary containing comprehensive execution and tracking metrics.
        """
        start_time = time.perf_counter()
        frames_processed = 0
        total_detections = 0
        max_simultaneous_tracks = 0
        saved_frames_count = 0
        observed_track_ids = set()
        class_distribution: Dict[str, int] = {}

        video_writer = None
        video_out_path = Path(save_video_path) if save_video_path else None
        if video_out_path:
            video_out_path.parent.mkdir(parents=True, exist_ok=True)

        out_dir = Path(save_output_dir) if save_output_dir else None
        if out_dir:
            out_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Executing batch tracking pipeline on: {self.video_path}")

        try:
            for frame_data, detections, tracks, _ in self.stream_tracks(render_visualization=False):
                frames_processed += 1
                num_dets = len(detections.detections)
                num_tracks = len(tracks.active_tracks)
                total_detections += num_dets

                if num_tracks > max_simultaneous_tracks:
                    max_simultaneous_tracks = num_tracks

                for t in tracks.active_tracks:
                    observed_track_ids.add(t.track_id)
                    class_distribution[t.class_name] = class_distribution.get(t.class_name, 0) + 1

                # Generate visualization if writing video or saving sample images
                if video_out_path or (out_dir and saved_frames_count < max_save_frames):
                    annotated = self.visualizer.draw_tracks(
                        image=frame_data.image,
                        tracks=tracks,
                    )

                    # Initialize video writer on first frame
                    if video_out_path and video_writer is None:
                        h, w = annotated.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        # Adjust FPS according to frame sampling
                        nominal_fps = max(1.0, 30.0 / max(1, self.frame_skip))
                        video_writer = cv2.VideoWriter(str(video_out_path), fourcc, nominal_fps, (w, h))
                        if not video_writer.isOpened():
                            logger.warning(f"Could not open VideoWriter for {video_out_path}")
                            video_writer = None

                    if video_writer is not None:
                        video_writer.write(annotated)

                    # Save sample image
                    if out_dir and saved_frames_count < max_save_frames:
                        frame_file = out_dir / f"step4_tracking_frame_{frame_data.frame_index:04d}.jpg"
                        self.visualizer.save_annotated_frame(annotated, frame_file)
                        saved_frames_count += 1

                if callback:
                    callback(frame_data, detections, tracks)

                if frames_processed % log_interval == 0:
                    logger.info(
                        f"Frame {frame_data.frame_index:04d} | "
                        f"Detections: {num_dets} | "
                        f"Active Tracks: {num_tracks} | "
                        f"Cumulative Tracks: {len(observed_track_ids)}"
                    )

        finally:
            if video_writer is not None:
                video_writer.release()
                logger.info(f"Tracking video successfully saved to: {video_out_path}")

        elapsed = time.perf_counter() - start_time
        fps = (frames_processed / elapsed) if elapsed > 0 else 0.0

        summary = {
            "source": str(self.video_path),
            "detector_model": self.detector.model_path,
            "tracker_type": type(self.tracker).__name__,
            "frames_processed": frames_processed,
            "total_detections": total_detections,
            "unique_tracks_observed": len(observed_track_ids),
            "max_simultaneous_tracks": max_simultaneous_tracks,
            "class_distribution": class_distribution,
            "elapsed_seconds": round(elapsed, 4),
            "effective_fps": round(fps, 2),
            "saved_frames": saved_frames_count,
            "output_video": str(video_out_path) if video_out_path and video_out_path.exists() else None,
        }

        logger.info(
            f"Tracking completed: {frames_processed} frames, {len(observed_track_ids)} unique tracks "
            f"in {elapsed:.2f}s ({fps:.1f} FPS)"
        )
        return summary
