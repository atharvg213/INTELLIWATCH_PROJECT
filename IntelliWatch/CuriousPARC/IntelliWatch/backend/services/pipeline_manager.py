"""
backend/services/pipeline_manager.py
Lifecycle and execution manager for the IntelliWatch video processing pipeline.
Manages EndToEndPipelineOrchestrator, processing state, and operational status telemetry.
"""
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import numpy as np

from backend.schemas.assessment import FrameAssessment
from backend.schemas.evidence import SystemStatusResponse
from backend.services.incident_store import get_incident_store
from configs.settings import get_settings
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator

logger = logging.getLogger("intelliwatch.pipeline_manager")


class PipelineManager:
    """
    Orchestration service coordinating live and file-based video processing.
    """

    def __init__(self):
        self.settings = get_settings()
        self._is_running: bool = False
        self._current_source: Optional[str] = None
        self._orchestrator: Optional[EndToEndPipelineOrchestrator] = None
        self._orchestrators: Dict[str, EndToEndPipelineOrchestrator] = {}
        self._start_time: Optional[float] = None
        self._last_frame_id: int = 0

    def get_orchestrator(self, camera_id: Optional[str] = None) -> EndToEndPipelineOrchestrator:
        """Returns an orchestrator scoped to the assigned camera identity."""
        resolved_camera_id = camera_id.strip() if isinstance(camera_id, str) and camera_id.strip() else "unassigned"
        if resolved_camera_id not in self._orchestrators:
            orchestrator = EndToEndPipelineOrchestrator(
                device=self.settings.DETECTION_DEVICE,
                camera_id=resolved_camera_id,
                enable_ppe_model=self.settings.PPE_ENABLED,
                enable_depth_model=False,
            )
            self._orchestrators[resolved_camera_id] = orchestrator
            if resolved_camera_id == "unassigned":
                self._orchestrator = orchestrator
        return self._orchestrators[resolved_camera_id]

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start(self, source: Optional[str] = None) -> None:
        """Starts pipeline processing service for specified video source."""
        self._current_source = source or self.settings.VIDEO_SOURCE
        self._is_running = True
        self._start_time = time.time()
        logger.info(f"Pipeline service initialized for source: {self._current_source}")

    def stop(self) -> None:
        """Gracefully halts pipeline processing."""
        self._is_running = False
        logger.info("Pipeline service stopped.")

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> Tuple[FrameAssessment, np.ndarray]:
        """Processes a single frame through the end-to-end pipeline."""
        orchestrator = self.get_orchestrator()
        fid = frame_id if frame_id is not None else (self._last_frame_id + 1)
        ts = timestamp if timestamp is not None else (fid / 30.0)

        assessment, annotated = orchestrator.process_frame(
            frame=frame,
            frame_id=fid,
            timestamp=ts,
        )
        self._last_frame_id = fid
        return assessment, annotated

    def get_status(self) -> SystemStatusResponse:
        """Returns comprehensive operational system status."""
        orchestrator = self.get_orchestrator()
        perf = orchestrator.get_performance_stats()
        incident_store = get_incident_store()

        yolo_path = Path(self.settings.DETECTION_MODEL_PATH)
        ppe_path = Path(self.settings.PPE_MODEL_PATH)

        return SystemStatusResponse(
            status="healthy",
            pipeline_state="running" if self._is_running else "idle",
            device=perf["device"],
            is_cpu_mode=(perf["device"] == "cpu"),
            current_source=self._current_source,
            current_frame_id=self._last_frame_id,
            processing_fps=perf["processing_fps"],
            avg_latency_ms=perf["avg_latency_ms"],
            total_frames_processed=perf["total_frames_processed"],
            active_tracks_count=len(orchestrator._tracker.active_tracks) if hasattr(orchestrator._tracker, "active_tracks") else 0,
            active_incidents_count=len(incident_store.get_active_incidents()),
            total_recorded_incidents=incident_store.total_count(),
            models_loaded={
                "yolo_detection": yolo_path.exists(),
                "ppe_detection": ppe_path.exists(),
                "bytetrack": True,
                "zone_engine": True,
                "behavior_engine": True,
                "scene_graph": True,
                "risk_engine": True,
                "prediction_engine": True,
            },
        )

    def reset(self) -> None:
        """Resets orchestrator state."""
        for orchestrator in self._orchestrators.values():
            orchestrator.reset()
        if self._orchestrator and self._orchestrator not in self._orchestrators.values():
            self._orchestrator.reset()
        self._last_frame_id = 0
        self._is_running = False


_pipeline_manager_instance = PipelineManager()


def get_pipeline_manager() -> PipelineManager:
    """Returns singleton PipelineManager."""
    return _pipeline_manager_instance
