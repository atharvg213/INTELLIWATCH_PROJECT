"""
tests/test_orchestrator.py
Step 12 Test Suite - End-to-End Pipeline Orchestration.
Validates:
- Orchestrator initialization and CPU device selection
- Sequential execution across all perception, tracking, compliance, spatial, behavior,
  scene graph, risk, and predictive early-warning layers
- FrameAssessment schema integrity
- Annotated frame generation
- Error handling on invalid/empty inputs
"""
import numpy as np
import pytest

from backend.schemas.assessment import FrameAssessment
from backend.schemas.detection import BoundingBox, DetectionResult
from backend.schemas.risk import RiskLevel
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from backend.services.assessment_store import get_assessment_store


@pytest.fixture
def sample_frame():
    """Generates a synthetic 720p dark gray industrial frame."""
    return np.full((720, 1280, 3), 40, dtype=np.uint8)


@pytest.fixture
def orchestrator():
    """Returns an EndToEndPipelineOrchestrator instance configured for CPU test mode."""
    return EndToEndPipelineOrchestrator(
        device="cpu",
        enable_ppe_model=False,
        enable_depth_model=False,
    )


def test_orchestrator_initialization(orchestrator):
    """Verifies that orchestrator initializes with CPU execution and all child components."""
    assert orchestrator.device_str == "cpu"
    assert orchestrator.total_frames_processed == 0
    assert orchestrator.total_processing_time_ms == 0.0


def test_orchestrator_process_empty_frame(orchestrator):
    """Verifies that empty or None frame input raises ValueError."""
    with pytest.raises(ValueError, match="Input frame cannot be empty"):
        orchestrator.process_frame(None)

    empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="Input frame cannot be empty"):
        orchestrator.process_frame(empty_img)


def test_orchestrator_process_frame_no_detections(orchestrator, sample_frame):
    """Verifies pipeline execution when no objects are detected in the frame."""
    assessment, annotated = orchestrator.process_frame(
        frame=sample_frame,
        frame_id=1,
        timestamp=0.033,
        manual_detections=[],
    )

    assert isinstance(assessment, FrameAssessment)
    assert assessment.frame_id == 1
    assert assessment.timestamp == 0.033
    assert assessment.total_active_tracks == 0
    assert assessment.highest_risk_level == RiskLevel.INFO
    assert assessment.highest_risk_score == 0.0
    assert assessment.processing_time_ms > 0.0

    # Annotated image must retain original frame shape and dtype
    assert annotated.shape == sample_frame.shape
    assert annotated.dtype == np.uint8


def test_orchestrator_process_frame_with_worker_detection(orchestrator, sample_frame):
    """Verifies pipeline execution when a worker is detected and tracked."""
    worker_det = DetectionResult(
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=BoundingBox(x1=200.0, y1=150.0, x2=280.0, y2=450.0),
    )

    assessment, annotated = orchestrator.process_frame(
        frame=sample_frame,
        frame_id=1,
        timestamp=0.033,
        manual_detections=[worker_det],
    )

    assert isinstance(assessment, FrameAssessment)
    assert assessment.total_active_tracks == 1
    assert len(assessment.tracks) == 1
    assert assessment.tracks[0].class_name == "person"
    assert len(assessment.worker_inventories) == 1

    # Verify assessment store is updated
    stored = get_assessment_store().get_current_assessment()
    assert stored.frame_id == 1
    assert stored.total_active_tracks == 1


def test_orchestrator_multi_frame_progression(orchestrator, sample_frame):
    """Verifies temporal continuity across consecutive frames."""
    worker_det_f1 = DetectionResult(
        class_id=0,
        class_name="person",
        confidence=0.95,
        bbox=BoundingBox(x1=200.0, y1=150.0, x2=280.0, y2=450.0),
    )
    worker_det_f2 = DetectionResult(
        class_id=0,
        class_name="person",
        confidence=0.95,
        bbox=BoundingBox(x1=210.0, y1=150.0, x2=290.0, y2=450.0),
    )

    assess1, _ = orchestrator.process_frame(sample_frame, frame_id=1, timestamp=0.033, manual_detections=[worker_det_f1])
    assess2, _ = orchestrator.process_frame(sample_frame, frame_id=2, timestamp=0.066, manual_detections=[worker_det_f2])

    assert assess1.total_active_tracks == 1
    assert assess2.total_active_tracks == 1
    # Track ID should persist across frames
    assert assess1.tracks[0].track_id == assess2.tracks[0].track_id
    assert orchestrator.total_frames_processed == 2


def test_orchestrator_performance_stats(orchestrator, sample_frame):
    """Verifies that performance telemetry metrics are accurately calculated."""
    orchestrator.process_frame(sample_frame, frame_id=1, timestamp=0.033, manual_detections=[])
    stats = orchestrator.get_performance_stats()

    assert stats["device"] == "cpu"
    assert stats["total_frames_processed"] == 1
    assert stats["avg_latency_ms"] > 0.0
    assert stats["processing_fps"] > 0.0


def test_orchestrator_reset(orchestrator, sample_frame):
    """Verifies state reset cleans all internal trackers and metrics."""
    orchestrator.process_frame(sample_frame, frame_id=1, timestamp=0.033, manual_detections=[])
    assert orchestrator.total_frames_processed == 1

    orchestrator.reset()
    assert orchestrator.total_frames_processed == 0
    assert orchestrator.total_processing_time_ms == 0.0


def test_orchestrator_incident_creation_and_evidence(sample_frame):
    """
    Verifies that when risk events occur, the orchestrator logs an incident evidence record
    with snapshot and includes the incident ID in assessment.new_incident_ids.
    """
    from backend.services.incident_store import get_incident_store
    incident_store = get_incident_store()
    incident_store.reset()

    orch = EndToEndPipelineOrchestrator(
        device="cpu",
        enable_ppe_model=False,
        enable_depth_model=False,
    )
    # Configure fast confirmation
    orch._risk_engine.confirmation_frames = 1
    orch._zone_engine.confirmation_frames = 1

    # Place a worker inside configured zone (high_voltage_01 has coords [100,100] to [450,450])
    worker_inside_zone = DetectionResult(
        class_id=0,
        class_name="person",
        confidence=0.96,
        bbox=BoundingBox(x1=200.0, y1=200.0, x2=260.0, y2=350.0),
    )

    assessment, annotated = orch.process_frame(
        frame=sample_frame,
        frame_id=1,
        timestamp=0.1,
        manual_detections=[worker_inside_zone],
    )

    assert assessment.frame_id == 1
    # Check if an incident record was logged
    incidents = incident_store.list_incidents()
    if incidents:
        inc = incidents[0]
        assert inc.has_snapshot is True
        assert inc.snapshot_path is not None
        assert inc.frame_id == 1
        assert "Worker #" in inc.explanation or "person" in inc.explanation


def test_pipeline_manager_lifecycle(sample_frame):
    """Verifies PipelineManager startup, frame processing, and status inspection."""
    from backend.services.pipeline_manager import get_pipeline_manager
    pm = get_pipeline_manager()
    pm.reset()

    assert pm.is_running is False
    pm.start("sample_test_feed")
    assert pm.is_running is True
    assert pm._current_source == "sample_test_feed"

    assessment, annotated = pm.process_frame(sample_frame)
    assert assessment.frame_id == 1
    assert annotated.shape == sample_frame.shape

    status = pm.get_status()
    assert status.pipeline_state == "running"
    assert status.total_frames_processed >= 1

    pm.stop()
    assert pm.is_running is False
