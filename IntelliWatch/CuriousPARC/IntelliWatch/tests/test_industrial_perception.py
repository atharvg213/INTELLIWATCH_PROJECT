"""
tests/test_industrial_perception.py
Comprehensive test suite for Industrial-Aware Perception Layer Upgrade:
  - ClassGroup enum and schema normalization
  - IndustrialDetector (YOLO-World) initialization and vocabulary configuration
  - MultiDetectorAggregator with cross-model IoU deduplication and priority resolution
  - False positive domain filtering for non-industrial COCO classes
  - ByteTrack multi-class tracking with persistent vehicle and machine tracks
  - Scene graph relations for industrial vehicles and machinery
  - Deterministic risk factor extraction for PERSON_VEHICLE_PROXIMITY and WORKER_NEAR_MACHINE
  - Full EndToEndPipelineOrchestrator industrial integration
"""
import pytest
import numpy as np
from pathlib import Path
from unittest.mock import MagicMock, patch

from backend.schemas.detection import (
    BoundingBox,
    ClassGroup,
    DetectionResult,
    FrameDetections,
    infer_class_group,
)
from backend.schemas.tracking import FrameTracks, TrackedObject, TrackState
from backend.schemas.scene_graph import FrameScene, SceneNode, SceneNodeType, SceneRelationType
from backend.schemas.risk import RiskFactorType
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from intelligence.risk.factors import RiskFactorExtractor
from intelligence.scene_graph.builder import SceneGraphBuilder
from intelligence.scene_graph.relations import classify_node_type
from vision.detection.base import BaseDetector
from vision.detection.industrial_detector import IndustrialDetector
from vision.detection.merged_detector import MultiDetectorAggregator, compute_box_iou
from vision.tracking.bytetrack_tracker import ByteTrackTracker


# ---------------------------------------------------------------------------
# 1. Detection Schema & ClassGroup Normalization Tests
# ---------------------------------------------------------------------------
def test_class_group_enum_and_inference():
    """Verify all canonical classes map to the appropriate high-level ClassGroup."""
    assert infer_class_group("person") == ClassGroup.PERSON
    assert infer_class_group("worker") == ClassGroup.PERSON
    assert infer_class_group("operator") == ClassGroup.PERSON

    assert infer_class_group("forklift") == ClassGroup.VEHICLE
    assert infer_class_group("industrial vehicle") == ClassGroup.VEHICLE
    assert infer_class_group("truck") == ClassGroup.VEHICLE
    assert infer_class_group("agv") == ClassGroup.VEHICLE

    assert infer_class_group("machinery") == ClassGroup.MACHINE
    assert infer_class_group("robotic arm") == ClassGroup.MACHINE
    assert infer_class_group("conveyor") == ClassGroup.MACHINE
    assert infer_class_group("press") == ClassGroup.MACHINE

    assert infer_class_group("pallet") == ClassGroup.INFRASTRUCTURE
    assert infer_class_group("safety barrier") == ClassGroup.INFRASTRUCTURE
    assert infer_class_group("electrical cabinet") == ClassGroup.INFRASTRUCTURE

    assert infer_class_group("hardhat") == ClassGroup.PPE
    assert infer_class_group("safety vest") == ClassGroup.PPE

    assert infer_class_group("unknown_random_asset") == ClassGroup.OTHER


def test_detection_result_backward_compatibility():
    """Verify legacy construction without detection_id or class_group works and auto-infers."""
    det = DetectionResult(
        class_id=0,
        class_name="forklift",
        confidence=0.88,
        bbox=BoundingBox(x1=10, y1=20, x2=100, y2=200),
    )
    # Automatically assigned detection_id and inferred class_group
    assert det.detection_id is not None
    assert det.detection_id.startswith("det_")
    assert det.class_group == ClassGroup.VEHICLE
    assert det.confidence == 0.88
    assert det.bbox.width == 90
    assert det.bbox.height == 180


def test_detection_result_explicit_attributes():
    """Verify full construction with explicit provenance metadata."""
    det = DetectionResult(
        detection_id="custom_id_123",
        class_id=1,
        class_name="robotic arm",
        class_group=ClassGroup.MACHINE,
        confidence=0.92,
        bbox=BoundingBox(x1=50, y1=50, x2=150, y2=150),
        source_model="yolo_world",
        frame_index=14,
        timestamp=0.467,
        metadata={"custom_flag": True},
    )
    assert det.detection_id == "custom_id_123"
    assert det.class_group == ClassGroup.MACHINE
    assert det.source_model == "yolo_world"
    assert det.frame_index == 14
    assert det.metadata.get("custom_flag") is True


# ---------------------------------------------------------------------------
# 2. Box IoU & Cross-Model Deduplication Tests
# ---------------------------------------------------------------------------
def test_compute_box_iou():
    """Test mathematical correctness of bounding box IoU calculation."""
    box_a = BoundingBox(x1=0, y1=0, x2=100, y2=100)
    # Identical box -> IoU = 1.0
    assert compute_box_iou(box_a, box_a) == 1.0

    # Non-overlapping box -> IoU = 0.0
    box_b = BoundingBox(x1=200, y1=200, x2=300, y2=300)
    assert compute_box_iou(box_a, box_b) == 0.0

    # 50% horizontal overlap
    box_c = BoundingBox(x1=50, y1=0, x2=150, y2=100)
    # intersection: 50*100 = 5000, union = 10000 + 10000 - 5000 = 15000 -> 1/3
    assert abs(compute_box_iou(box_a, box_c) - (1.0 / 3.0)) < 1e-4


def test_multi_detector_cross_model_deduplication():
    """Verify that industrial detection takes precedence over general COCO misclassification."""
    aggregator = MultiDetectorAggregator(
        general_detector=None,
        industrial_detector=None,
        cross_model_iou_thresh=0.40,
        filter_irrelevant_coco=False,
    )

    # General detector detected "boat" at the forklift location
    general_det = DetectionResult(
        class_id=8,
        class_name="boat",
        confidence=0.25,
        bbox=BoundingBox(x1=100, y1=100, x2=300, y2=300),
        source_model="yolo11n",
    )
    # Industrial detector detected "industrial vehicle" (forklift) at same location
    industrial_det = DetectionResult(
        class_id=0,
        class_name="industrial vehicle",
        class_group=ClassGroup.VEHICLE,
        confidence=0.35,
        bbox=BoundingBox(x1=105, y1=98, x2=302, y2=295),
        source_model="yolo_world",
    )

    merged = aggregator.merge_detections([general_det], [industrial_det])
    # The industrial vehicle should be retained, and the general boat suppressed
    assert len(merged) == 1
    assert merged[0].class_name == "industrial vehicle"
    assert merged[0].source_model == "yolo_world"


def test_multi_detector_person_duplicate_resolution():
    """Verify when both models detect person, the one with higher confidence is kept."""
    aggregator = MultiDetectorAggregator(cross_model_iou_thresh=0.40)
    gen_person = DetectionResult(
        class_id=0,
        class_name="person",
        confidence=0.88,
        bbox=BoundingBox(x1=50, y1=50, x2=100, y2=200),
        source_model="yolo11n",
    )
    ind_person = DetectionResult(
        class_id=0,
        class_name="person",
        confidence=0.72,
        bbox=BoundingBox(x1=52, y1=48, x2=98, y2=198),
        source_model="yolo_world",
    )

    merged = aggregator.merge_detections([gen_person], [ind_person])
    assert len(merged) == 1
    assert merged[0].confidence == 0.88
    assert merged[0].source_model == "yolo11n"


# ---------------------------------------------------------------------------
# 3. False Positive Filtering Tests
# ---------------------------------------------------------------------------
def test_filter_irrelevant_coco_classes():
    """Verify non-industrial COCO classes (bench, boat, sports ball) are filtered out."""
    aggregator = MultiDetectorAggregator(
        cross_model_iou_thresh=0.40,
        filter_irrelevant_coco=True,
        irrelevant_classes=["boat", "bench", "sports ball", "frisbee"],
    )

    dets = [
        DetectionResult(class_id=0, class_name="person", confidence=0.9, bbox=BoundingBox(x1=0, y1=0, x2=10, y2=20), source_model="yolo11n"),
        DetectionResult(class_id=8, class_name="boat", confidence=0.22, bbox=BoundingBox(x1=10, y1=10, x2=50, y2=50), source_model="yolo11n"),
        DetectionResult(class_id=13, class_name="bench", confidence=0.45, bbox=BoundingBox(x1=60, y1=60, x2=90, y2=90), source_model="yolo11n"),
        DetectionResult(class_id=32, class_name="sports ball", confidence=0.30, bbox=BoundingBox(x1=100, y1=100, x2=110, y2=110), source_model="yolo11n"),
    ]

    filtered = aggregator.filter_detections(dets)
    assert len(filtered) == 1
    assert filtered[0].class_name == "person"


# ---------------------------------------------------------------------------
# 4. Multi-Class Tracking with ByteTrack Tests
# ---------------------------------------------------------------------------
def test_bytetrack_tracks_both_person_and_vehicle():
    """Verify ByteTrack maintains persistent tracks for both workers and industrial vehicles."""
    tracker = ByteTrackTracker(track_high_thresh=0.20, track_low_thresh=0.05)

    # Frame 1: 1 Person + 1 Forklift
    d1 = FrameDetections(
        frame_id=0,
        timestamp=0.0,
        detections=[
            DetectionResult(
                class_id=0,
                class_name="person",
                confidence=0.85,
                bbox=BoundingBox(x1=100, y1=100, x2=150, y2=250),
            ),
            DetectionResult(
                class_id=1,
                class_name="forklift",
                confidence=0.75,
                bbox=BoundingBox(x1=400, y1=200, x2=600, y2=450),
            ),
        ],
        frame_width=1280,
        frame_height=720,
    )
    t1 = tracker.update(d1)
    assert len(t1.active_tracks) == 2

    # Frame 2: Objects shifted slightly
    d2 = FrameDetections(
        frame_id=1,
        timestamp=0.033,
        detections=[
            DetectionResult(
                class_id=0,
                class_name="person",
                confidence=0.87,
                bbox=BoundingBox(x1=102, y1=101, x2=152, y2=251),
            ),
            DetectionResult(
                class_id=1,
                class_name="forklift",
                confidence=0.78,
                bbox=BoundingBox(x1=405, y1=202, x2=605, y2=452),
            ),
        ],
        frame_width=1280,
        frame_height=720,
    )
    t2 = tracker.update(d2)
    assert len(t2.active_tracks) == 2

    # Verify track class preservation
    classes_present = {trk.class_name for trk in t2.active_tracks}
    assert "person" in classes_present
    assert "forklift" in classes_present


# ---------------------------------------------------------------------------
# 5. Scene Graph Node Classification & Relationships Tests
# ---------------------------------------------------------------------------
def test_classify_node_type_industrial_classes():
    """Verify scene node type classification recognizes open-vocabulary industrial classes."""
    assert classify_node_type("person") == SceneNodeType.PERSON
    assert classify_node_type("forklift") == SceneNodeType.VEHICLE
    assert classify_node_type("industrial vehicle") == SceneNodeType.VEHICLE
    assert classify_node_type("forklift truck") == SceneNodeType.VEHICLE
    assert classify_node_type("agv") == SceneNodeType.VEHICLE

    assert classify_node_type("machinery") == SceneNodeType.MACHINE
    assert classify_node_type("robotic arm") == SceneNodeType.MACHINE
    assert classify_node_type("conveyor") == SceneNodeType.MACHINE
    assert classify_node_type("industrial machine") == SceneNodeType.MACHINE

    assert classify_node_type("pallet") == SceneNodeType.OBJECT
    assert classify_node_type("safety barrier") == SceneNodeType.OBJECT


def test_scene_graph_builder_person_vehicle_relation():
    """Verify SceneGraphBuilder generates NEAR relationship between Worker and Forklift."""
    builder = SceneGraphBuilder(
        near_distance_threshold=200.0,
        far_distance_threshold=400.0,
        confirmation_frames=1,
    )

    # Worker and Forklift in close proximity (<200 px)
    tracks = FrameTracks(
        frame_id=1,
        timestamp=0.1,
        active_tracks=[
            TrackedObject(
                track_id=1,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=100, x2=150, y2=250),
            ),
            TrackedObject(
                track_id=2,
                class_id=1,
                class_name="forklift",
                confidence=0.8,
                bbox=BoundingBox(x1=180, y1=120, x2=320, y2=280),
            ),
        ],
    )

    scene = builder.build(frame_tracks=tracks)
    assert scene.summary.workers_count == 1
    assert scene.summary.vehicles_count == 1

    # Check for NEAR relationship
    near_rels = [r for r in scene.relationships if r.relation_type == SceneRelationType.NEAR]
    assert len(near_rels) > 0


# ---------------------------------------------------------------------------
# 6. Risk Factor Extraction for Industrial Objects Tests
# ---------------------------------------------------------------------------
def test_risk_factor_person_vehicle_proximity():
    """Verify PERSON_VEHICLE_PROXIMITY risk factor is triggered when worker is near forklift."""
    builder = SceneGraphBuilder(
        near_distance_threshold=200.0,
        far_distance_threshold=400.0,
        confirmation_frames=1,
    )
    extractor = RiskFactorExtractor()

    tracks = FrameTracks(
        frame_id=1,
        timestamp=0.1,
        active_tracks=[
            TrackedObject(
                track_id=1,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=BoundingBox(x1=100, y1=100, x2=150, y2=250),
            ),
            TrackedObject(
                track_id=2,
                class_id=1,
                class_name="forklift",
                confidence=0.8,
                bbox=BoundingBox(x1=160, y1=110, x2=300, y2=270),
            ),
        ],
    )

    scene = builder.build(frame_tracks=tracks)
    factors = extractor.extract_factors(scene=scene, timestamp=0.1)

    factor_types = [f.factor_type for f in factors]
    assert RiskFactorType.PERSON_VEHICLE_PROXIMITY in factor_types


def test_risk_factor_worker_near_machine():
    """Verify WORKER_NEAR_MACHINE risk factor is triggered when worker is near industrial machinery."""
    builder = SceneGraphBuilder(
        near_distance_threshold=200.0,
        far_distance_threshold=400.0,
        confirmation_frames=1,
    )
    extractor = RiskFactorExtractor()

    tracks = FrameTracks(
        frame_id=1,
        timestamp=0.1,
        active_tracks=[
            TrackedObject(
                track_id=1,
                class_id=0,
                class_name="person",
                confidence=0.9,
                bbox=BoundingBox(x1=200, y1=200, x2=250, y2=350),
            ),
            TrackedObject(
                track_id=3,
                class_id=2,
                class_name="machinery",
                confidence=0.85,
                bbox=BoundingBox(x1=280, y1=220, x2=450, y2=380),
            ),
        ],
    )

    scene = builder.build(frame_tracks=tracks)
    factors = extractor.extract_factors(scene=scene, timestamp=0.1)

    factor_types = [f.factor_type for f in factors]
    assert RiskFactorType.WORKER_NEAR_MACHINE in factor_types


# ---------------------------------------------------------------------------
# 7. EndToEndPipelineOrchestrator Integration Tests
# ---------------------------------------------------------------------------
def test_orchestrator_process_frame_with_industrial_detections():
    """Verify EndToEndPipelineOrchestrator executes full pipeline with worker + forklift."""
    orchestrator = EndToEndPipelineOrchestrator(
        enable_ppe_model=False,
        enable_depth_model=False,
    )

    synthetic_frame = np.zeros((720, 1280, 3), dtype=np.uint8)

    manual_dets = [
        DetectionResult(
            class_id=0,
            class_name="person",
            confidence=0.92,
            bbox=BoundingBox(x1=150, y1=200, x2=200, y2=400),
            source_model="yolo11n",
        ),
        DetectionResult(
            class_id=1,
            class_name="industrial vehicle",
            confidence=0.78,
            bbox=BoundingBox(x1=220, y1=220, x2=380, y2=420),
            source_model="yolo_world",
        ),
    ]

    assessment, annotated_frame = orchestrator.process_frame(
        frame=synthetic_frame,
        frame_id=1,
        timestamp=0.033,
        manual_detections=manual_dets,
    )

    assert assessment.frame_id == 1
    assert assessment.total_active_tracks == 2
    assert len(assessment.detections) == 2

    # Verify track class names
    track_classes = {t.class_name for t in assessment.tracks}
    assert "person" in track_classes
    assert "industrial vehicle" in track_classes

    # Verify annotated frame is returned
    assert annotated_frame is not None
    assert annotated_frame.shape == (720, 1280, 3)
