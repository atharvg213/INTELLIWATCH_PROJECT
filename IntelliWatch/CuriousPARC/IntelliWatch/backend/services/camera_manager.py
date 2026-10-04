"""
backend/services/camera_manager.py
Multi-Camera Stream Management and Asynchronous Processing Service for IntelliWatch.

Responsibilities:
  - Manages multiple concurrent camera streams with strict state isolation.
  - Couples each camera stream with a dedicated EndToEndPipelineOrchestrator instance.
  - Ensures per-camera tracking, PPE inventory, behavior analysis, zone events, and incidents.
  - Supports configurable frame sampling, FPS throttling, and automatic frame dropping.
  - Exposes thread-safe telemetry, JPEG snapshot extraction, and MJPEG streaming.
  - Guarantees that failure or disconnection of one camera never halts other camera streams.
"""
import logging
import threading
import time
from typing import Any, Dict, Generator, List, Optional, Tuple

import cv2
import numpy as np

from backend.schemas.assessment import FrameAssessment
from backend.schemas.camera import (
    CameraDetailResponse,
    CameraListResponse,
    CameraRegisterRequest,
    CameraStatus,
    CameraTelemetrySchema,
    CameraUpdateRequest,
)
from configs.settings import get_settings
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from vision.streaming.stream_reader import RTSPStreamReader, StreamFrame, sanitize_stream_url
from vision.utils.device import resolve_device

logger = logging.getLogger("intelliwatch.camera_manager")


class CameraStreamWorker:
    """
    Dedicated asynchronous processing worker for a single camera stream.
    Runs an isolated EndToEndPipelineOrchestrator consuming frames from RTSPStreamReader.
    """

    def __init__(
        self,
        camera_id: str,
        name: str,
        source: str,
        sampling_interval: int = 1,
        max_processing_fps: Optional[float] = 10.0,
        loop_file: bool = True,
        reconnect_interval_sec: float = 3.0,
    ):
        self.camera_id = camera_id
        self.name = name
        self.source = source
        self.sanitized_source = sanitize_stream_url(source)
        self.sampling_interval = max(1, sampling_interval)
        self.max_processing_fps = max_processing_fps
        self.settings = get_settings()

        # Stream reader (handles capture thread & reconnection)
        self.reader = RTSPStreamReader(
            camera_id=self.camera_id,
            source=self.source,
            queue_size=getattr(self.settings, "STREAM_QUEUE_SIZE", 10),
            reconnect_interval_sec=reconnect_interval_sec,
            loop_file=loop_file,
        )

        # Isolated AI perception & reasoning pipeline orchestrator
        resolved_device = resolve_device(self.settings.DETECTION_DEVICE)
        self.orchestrator = EndToEndPipelineOrchestrator(
            device=resolved_device,
            camera_id=self.camera_id,
            enable_ppe_model=self.settings.PPE_ENABLED,
            enable_industrial_model=False,
            enable_depth_model=False,  # CPU fallback performance optimization
        )

        # Processing thread & lifecycle
        self._processing_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._lock = threading.Lock()

        # Telemetry & Performance counters
        self.total_processed_frames = 0
        self.processing_fps = 0.0
        self.avg_latency_ms = 0.0
        self.active_tracks_count = 0
        self.active_incidents_count = 0
        self.non_compliant_workers_count = 0
        self._start_time: Optional[float] = None
        self._proc_fps_window: list = []

        # Latest frame & assessment buffer (for live snapshot / dashboard streaming)
        self._latest_assessment: Optional[FrameAssessment] = None
        self._latest_annotated_frame: Optional[np.ndarray] = None
        self._latest_jpeg_bytes: Optional[bytes] = None

    @property
    def is_running(self) -> bool:
        """Returns True if processing worker thread is active."""
        return self._processing_thread is not None and self._processing_thread.is_alive()

    @property
    def status(self) -> CameraStatus:
        """Aggregates stream reader and worker operational status."""
        if not self.is_running and not self.reader.is_running:
            return CameraStatus.STOPPED
        return self.reader.status

    def start(self) -> None:
        """Starts stream reader and processing worker threads."""
        with self._lock:
            if self.is_running:
                logger.warning(f"Worker for camera [{self.camera_id}] is already running.")
                return

            self._stop_event.clear()
            self._start_time = time.time()

            # Start ingestion reader
            self.reader.start()

            # Start processing thread
            self._processing_thread = threading.Thread(
                target=self._processing_loop,
                name=f"CameraWorker-{self.camera_id}",
                daemon=True,
            )
            self._processing_thread.start()
            logger.info(f"Camera worker [{self.camera_id}] processing thread started.")

    def stop(self) -> None:
        """Stops stream ingestion and processing worker threads gracefully."""
        self._stop_event.set()

        # Stop reader
        self.reader.stop()

        # Wait for processing thread
        if self._processing_thread and self._processing_thread.is_alive():
            self._processing_thread.join(timeout=3.0)

        with self._lock:
            logger.info(f"Camera worker [{self.camera_id}] stopped.")

    def _processing_loop(self) -> None:
        """Continuous processing loop consuming frames from reader and running AI pipeline."""
        last_proc_time = 0.0

        while not self._stop_event.is_set():
            # Retrieve next frame from reader queue (with 0.2s timeout)
            frame: Optional[StreamFrame] = self.reader.get_frame(timeout=0.2)
            if frame is None:
                continue

            # Apply frame sampling interval
            if self.sampling_interval > 1 and (frame.frame_id % self.sampling_interval != 0):
                continue

            # Apply max processing FPS rate limit
            now = time.time()
            if self.max_processing_fps and self.max_processing_fps > 0:
                min_interval = 1.0 / self.max_processing_fps
                if (now - last_proc_time) < min_interval:
                    continue

            # Execute end-to-end perception and reasoning
            t0 = time.perf_counter()
            try:
                assessment, annotated = self.orchestrator.process_frame(
                    frame=frame.image,
                    frame_id=frame.frame_id,
                    timestamp=frame.timestamp,
                )
                proc_time_ms = (time.perf_counter() - t0) * 1000.0
            except Exception as e:
                logger.error(f"Inference error in camera [{self.camera_id}] frame {frame.frame_id}: {e}", exc_info=True)
                continue

            last_proc_time = time.time()
            self.total_processed_frames += 1

            # Update rolling processing FPS
            self._proc_fps_window.append(last_proc_time)
            cutoff = last_proc_time - 2.0
            while self._proc_fps_window and self._proc_fps_window[0] < cutoff:
                self._proc_fps_window.pop(0)
            if len(self._proc_fps_window) >= 2:
                dt = self._proc_fps_window[-1] - self._proc_fps_window[0]
                self.processing_fps = round((len(self._proc_fps_window) - 1) / max(0.001, dt), 2)

            # Update rolling average latency
            if self.avg_latency_ms <= 0.0:
                self.avg_latency_ms = round(proc_time_ms, 2)
            else:
                self.avg_latency_ms = round((0.90 * self.avg_latency_ms) + (0.10 * proc_time_ms), 2)

            # Update scene metrics
            self.active_tracks_count = len(assessment.tracks)
            self.non_compliant_workers_count = assessment.non_compliant_workers_count
            self.active_incidents_count = len(assessment.risk_assessment.active_events) if assessment.risk_assessment else 0

            # Encode annotated frame to JPEG for live dashboard streaming
            try:
                encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 80]
                success, enc_jpg = cv2.imencode(".jpg", annotated, encode_param)
                jpg_bytes = enc_jpg.tobytes() if success else None
            except Exception as e:
                logger.debug(f"JPEG encoding failed on camera [{self.camera_id}]: {e}")
                jpg_bytes = None

            with self._lock:
                self._latest_assessment = assessment
                self._latest_annotated_frame = annotated
                self._latest_jpeg_bytes = jpg_bytes

    def get_latest_jpeg(self) -> Optional[bytes]:
        """Returns latest annotated frame as JPEG bytes."""
        with self._lock:
            return self._latest_jpeg_bytes

    def get_latest_assessment(self) -> Optional[FrameAssessment]:
        """Returns latest structured FrameAssessment."""
        with self._lock:
            return self._latest_assessment

    def generate_mjpeg_stream(self) -> Generator[bytes, None, None]:
        """
        Yields multipart/x-mixed-replace MJPEG stream frames for live browser rendering.
        """
        while not self._stop_event.is_set():
            jpeg_data = self.get_latest_jpeg()
            if jpeg_data is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n"
                    b"Content-Length: " + str(len(jpeg_data)).encode() + b"\r\n\r\n"
                    + jpeg_data
                    + b"\r\n"
                )
            # Throttle stream output to ~15 FPS max for bandwidth economy
            time.sleep(0.066)

    def get_telemetry(self) -> CameraTelemetrySchema:
        """Returns comprehensive operational telemetry for this camera."""
        reader_tel = self.reader.get_telemetry()
        uptime = (time.time() - self._start_time) if (self._start_time and self.is_running) else 0.0

        return CameraTelemetrySchema(
            camera_id=self.camera_id,
            name=self.name,
            source_sanitized=self.sanitized_source,
            status=self.status,
            is_active=self.is_running,
            width=reader_tel["width"],
            height=reader_tel["height"],
            stream_fps=reader_tel["stream_fps"],
            ingest_fps=reader_tel["ingest_fps"],
            processing_fps=self.processing_fps,
            avg_latency_ms=self.avg_latency_ms,
            total_ingested_frames=reader_tel["total_ingested_frames"],
            total_processed_frames=self.total_processed_frames,
            dropped_frames_count=reader_tel["dropped_frames_count"],
            active_tracks_count=self.active_tracks_count,
            active_incidents_count=self.active_incidents_count,
            non_compliant_workers_count=self.non_compliant_workers_count,
            reconnect_attempts=reader_tel["reconnect_attempts"],
            last_error=reader_tel["last_error"],
            uptime_seconds=round(uptime, 2),
        )


class MultiCameraManager:
    """
    Central registry and coordination service managing multi-camera stream processing.
    """

    def __init__(self):
        self._cameras: Dict[str, CameraStreamWorker] = {}
        self._lock = threading.Lock()

    def register_camera(self, req: CameraRegisterRequest) -> CameraTelemetrySchema:
        """
        Registers and optionally starts a new camera stream.
        """
        with self._lock:
            if req.camera_id in self._cameras:
                raise ValueError(f"Camera with ID '{req.camera_id}' is already registered.")

            worker = CameraStreamWorker(
                camera_id=req.camera_id,
                name=req.name,
                source=req.source,
                sampling_interval=req.sampling_interval,
                max_processing_fps=req.max_processing_fps,
                loop_file=req.loop_file,
                reconnect_interval_sec=req.reconnect_interval_sec,
            )
            self._cameras[req.camera_id] = worker

        if req.auto_start:
            worker.start()

        logger.info(f"Registered camera [{req.camera_id}] '{req.name}' (auto_start={req.auto_start})")
        return worker.get_telemetry()

    def update_camera(self, camera_id: str, req: CameraUpdateRequest) -> CameraTelemetrySchema:
        """Updates properties of an existing camera stream."""
        worker = self.get_camera(camera_id)
        if worker is None:
            raise KeyError(f"Camera '{camera_id}' not found.")

        if req.name is not None:
            worker.name = req.name
        if req.sampling_interval is not None:
            worker.sampling_interval = req.sampling_interval
        if req.max_processing_fps is not None:
            worker.max_processing_fps = req.max_processing_fps

        return worker.get_telemetry()

    def start_camera(self, camera_id: str) -> CameraTelemetrySchema:
        """Starts stream ingestion and processing for the specified camera."""
        worker = self.get_camera(camera_id)
        if worker is None:
            raise KeyError(f"Camera '{camera_id}' not found.")
        worker.start()
        return worker.get_telemetry()

    def stop_camera(self, camera_id: str) -> CameraTelemetrySchema:
        """Stops stream processing for the specified camera."""
        worker = self.get_camera(camera_id)
        if worker is None:
            raise KeyError(f"Camera '{camera_id}' not found.")
        worker.stop()
        return worker.get_telemetry()

    def remove_camera(self, camera_id: str) -> None:
        """Stops and unregisters the specified camera stream."""
        with self._lock:
            worker = self._cameras.pop(camera_id, None)

        if worker is not None:
            worker.stop()
            logger.info(f"Removed camera [{camera_id}].")
        else:
            raise KeyError(f"Camera '{camera_id}' not found.")

    def get_camera(self, camera_id: str) -> Optional[CameraStreamWorker]:
        """Returns the worker instance for a camera, or None."""
        with self._lock:
            return self._cameras.get(camera_id)

    def list_cameras(self) -> CameraListResponse:
        """Returns telemetry for all registered cameras."""
        with self._lock:
            workers = list(self._cameras.values())

        telemetries = [w.get_telemetry() for w in workers]
        connected = sum(1 for t in telemetries if t.status == CameraStatus.CONNECTED)

        return CameraListResponse(
            total_cameras=len(telemetries),
            connected_count=connected,
            cameras=telemetries,
        )

    def get_camera_detail(self, camera_id: str) -> CameraDetailResponse:
        """Returns detailed telemetry and latest assessment for a camera."""
        worker = self.get_camera(camera_id)
        if worker is None:
            raise KeyError(f"Camera '{camera_id}' not found.")

        tel = worker.get_telemetry()
        assessment = worker.get_latest_assessment()
        assessment_dict = assessment.model_dump() if assessment else None

        return CameraDetailResponse(
            camera=tel,
            latest_assessment=assessment_dict,
        )

    def stop_all(self) -> None:
        """Stops all active camera streams during shutdown."""
        with self._lock:
            workers = list(self._cameras.values())
        for w in workers:
            try:
                w.stop()
            except Exception as e:
                logger.warning(f"Error stopping camera [{w.camera_id}]: {e}")
        logger.info(f"Stopped all {len(workers)} camera workers.")


# Global singleton instance
_multi_camera_manager_instance = MultiCameraManager()


def get_camera_manager() -> MultiCameraManager:
    """Returns singleton MultiCameraManager instance."""
    return _multi_camera_manager_instance
