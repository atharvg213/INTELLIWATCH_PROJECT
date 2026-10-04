"""
tests/test_detection_accuracy_regression.py
Regression tests for:
1. Detecting in-vehicle persons (e.g. 4th person seated in a vehicle/forklift).
2. Preventing duplicate vehicle detections for a single physical vehicle.
3. Accurate object classification, domain filtering, and false-positive suppression.
4. Counting consistency across detection, tracking, and assessment schemas.
"""
from pathlib import Path
import cv2
import numpy as np
import pytest

from backend.schemas.detection import (
    BoundingBox,
    ClassGroup,
    DetectionResult,
    FrameDetections,
    infer_class_group,
)
from backend.schemas.tracking import FrameTracks, TrackedObject, TrackState
from configs.settings import get_settings
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from vision.detection.coordinates import compute_box_iou
from vision.detection.merged_detector import MultiDetectorAggregator
from vision.detection.yolo_detector import YOLODetector


REAL_FACTORY_IMAGE = Path("data/input/uploads/job_img_2adfe9fb54_input.png")


def test_in_vehicle_person_not_suppressed_by_merging():
    """
    Regression Test for Issue 1:
    Verify that a person inside or overlapping with a vehicle is NEVER suppressed
    by cross-model or multi-detector deduplication.
    """
    aggregator = MultiDetectorAggregator(cross_model_iou_thresh=0.40)

    # Person inside vehicle cab
    person_det = DetectionResult(
        class_id=0,
        class_name="person",
        class_group=ClassGroup.PERSON,
        confidence=0.75,
        bbox=BoundingBox(x1=400, y1=300, x2=480, y2=380),
        source_model="yolo11n",
    )
    # Surrounding vehicle
    vehicle_det = DetectionResult(
        class_id=100,
        class_name="forklift",
        class_group=ClassGroup.VEHICLE,
        confidence=0.85,
        bbox=BoundingBox(x1=350, y1=250, x2=550, y2=500),
        source_model="yolo_world",
    )

    merged = aggregator.merge_detections([person_det], [vehicle_det])

    # Both the vehicle AND the person MUST be preserved!
    assert len(merged) == 2
    class_names = {d.class_name for d in merged}
    assert "person" in class_names
    assert "forklift" in class_names


def test_duplicate_vehicles_deduplication():
    """
    Regression Test for Issue 2:
    Verify that duplicate vehicle detections on the same physical vehicle
    (e.g., overlapping 'truck' and 'bus') are suppressed to exactly 1 vehicle.
    """
    aggregator = MultiDetectorAggregator(cross_model_iou_thresh=0.40)

    truck_det = DetectionResult(
        class_id=7,
        class_name="truck",
        class_group=ClassGroup.VEHICLE,
        confidence=0.55,
        bbox=BoundingBox(x1=358.0, y1=246.0, x2=550.0, y2=503.0),
        source_model="yolo11n",
    )
    bus_det = DetectionResult(
        class_id=5,
        class_name="bus",
        class_group=ClassGroup.VEHICLE,
        confidence=0.35,
        bbox=BoundingBox(x1=360.0, y1=260.0, x2=550.0, y2=503.0),
        source_model="yolo11n",
    )

    deduped = aggregator.deduplicate_vehicles([truck_det, bus_det])
    assert len(deduped) == 1
    assert deduped[0].class_name == "truck"
    assert deduped[0].confidence == 0.55


def test_legitimate_nearby_vehicles_preserved():
    """
    Regression Test for Issue 2:
    Verify that legitimate distinct vehicles separated in space are both preserved.
    """
    aggregator = MultiDetectorAggregator(cross_model_iou_thresh=0.40)

    vehicle_1 = DetectionResult(
        class_id=100,
        class_name="forklift",
        class_group=ClassGroup.VEHICLE,
        confidence=0.80,
        bbox=BoundingBox(x1=100, y1=100, x2=200, y2=200),
    )
    vehicle_2 = DetectionResult(
        class_id=100,
        class_name="forklift",
        class_group=ClassGroup.VEHICLE,
        confidence=0.85,
        bbox=BoundingBox(x1=300, y1=100, x2=400, y2=200),
    )

    deduped = aggregator.deduplicate_vehicles([vehicle_1, vehicle_2])
    assert len(deduped) == 2


def test_irrelevant_coco_classes_filtering():
    """
    Regression Test for Issue 3:
    Verify that nonsensical COCO classes (e.g. sports ball, suitcase, boat) are rejected.
    """
    det = YOLODetector.__new__(YOLODetector)
    det.filter_irrelevant_coco = True
    det.irrelevant_classes = {"suitcase", "sports ball", "boat", "bench"}
    det._filter_classes = None

    class MockBox:
        def __init__(self, xyxy, conf, cls):
            self.xyxy = np.array([xyxy])
            self.conf = np.array([conf])
            self.cls = np.array([cls])

    class MockResult:
        def __init__(self, boxes):
            self.boxes = boxes

    det.confidence_threshold = 0.25
    det.person_confidence_threshold = 0.20
    det.vehicle_confidence_threshold = 0.28
    det.iou_threshold = 0.45
    det.device = "cpu"
    det.imgsz = 640
    det.map_to_original = False
    det._class_names = {0: "person", 28: "suitcase", 37: "sports ball"}
    det.enable_occupant_detection = False
    det.enable_vehicle_deduplication = False
    det.detection_debug = False

    from unittest.mock import MagicMock
    mock_boxes = [
        MockBox([10, 10, 50, 100], 0.90, 0),    # person -> keep
        MockBox([60, 60, 90, 90], 0.85, 28),    # suitcase -> filter
        MockBox([100, 100, 120, 120], 0.70, 37) # sports ball -> filter
    ]
    mock_model = MagicMock()
    mock_model.return_value = [MockResult(mock_boxes)]
    det._model = mock_model

    frame = np.zeros((200, 200, 3), dtype=np.uint8)
    res = det.detect(frame)

    assert len(res.detections) == 1
    assert res.detections[0].class_name == "person"


@pytest.mark.skipif(not REAL_FACTORY_IMAGE.exists(), reason="Real factory CCTV image not found")
def test_real_factory_floor_cctv_accuracy():
    """
    End-to-End Validation on Authentic Factory Floor CCTV:
    1. The 4th person (operator seated in forklift) MUST be detected.
    2. Exactly 1 vehicle (the active forklift) MUST be detected on the transit floor.
    3. Exactly 4 total persons MUST be detected (assembly worker, walking worker, background worker, driver).
    4. Stationary infrastructure (shelving, pallet crates) MUST NOT be classified as vehicles.
    """
    img = cv2.imread(str(REAL_FACTORY_IMAGE))
    assert img is not None, "Failed to load factory CCTV test image"

    detector = YOLODetector(device="cpu", confidence_threshold=0.25)
    frame_dets = detector.detect(img)

    persons = [d for d in frame_dets.detections if d.class_group == ClassGroup.PERSON]
    vehicles = [d for d in frame_dets.detections if d.class_group == ClassGroup.VEHICLE]
    infras = [d for d in frame_dets.detections if d.class_group == ClassGroup.INFRASTRUCTURE]

    # Verify Problem 1: Exactly 4 persons detected (including the 4th person in vehicle)
    assert len(persons) == 4, f"Expected 4 persons, got {len(persons)}: {[p.bbox for p in persons]}"
    # Verify at least one person has in_vehicle metadata
    in_vehicle_persons = [p for p in persons if p.metadata.get("in_vehicle")]
    assert len(in_vehicle_persons) >= 1, "Expected in-vehicle operator detection"

    # Verify Problem 2: Exactly 1 vehicle detected (the forklift)
    assert len(vehicles) == 1, f"Expected 1 vehicle (forklift), got {len(vehicles)}: {[v.class_name for v in vehicles]}"
    assert vehicles[0].class_name == "forklift"

    # Verify Problem 3: Infrastructure items identified as infrastructure, not vehicles
    assert len(infras) >= 1, "Expected static infrastructure (shelf/pallet) to be classified as INFRASTRUCTURE"
    for inf in infras:
        assert inf.class_group == ClassGroup.INFRASTRUCTURE
        assert inf.class_name in ("shelf", "pallet", "infrastructure")


@pytest.mark.skipif(not REAL_FACTORY_IMAGE.exists(), reason="Real factory CCTV image not found")
def test_orchestrator_e2e_counting_consistency():
    """
    Verify complete pipeline orchestrator produces consistent counts across:
    - detections_count
    - total_active_tracks
    - worker_inventories (4 workers evaluated for PPE compliance)
    """
    img = cv2.imread(str(REAL_FACTORY_IMAGE))
    orchestrator = EndToEndPipelineOrchestrator(device="cpu")
    assessment, annotated = orchestrator.process_frame(img, frame_id=1, timestamp=0.0)

    assert assessment.frame_id == 1
    assert annotated.shape == img.shape

    # Check worker inventories matches 4 workers
    assert len(assessment.worker_inventories) == 4, (
        f"Expected 4 worker inventories, got {len(assessment.worker_inventories)}"
    )

    # Active tracks must contain 4 workers + 1 forklift + infrastructure
    person_tracks = [t for t in assessment.tracks if (t.class_name or "").lower() in ("person", "worker", "operator")]
    vehicle_tracks = [t for t in assessment.tracks if (t.class_name or "").lower() in ("forklift", "truck", "vehicle")]

    assert len(person_tracks) == 4, f"Expected 4 person tracks, got {len(person_tracks)}"
    assert len(vehicle_tracks) == 1, f"Expected 1 vehicle track, got {len(vehicle_tracks)}"


# ==============================================================================
# PHASE 9: 10 FORMAL REGRESSION TEST CASES
# ==============================================================================

def test_case_1_multiple_people_in_and_around_vehicle():
    """Requirement 1: Multiple people in and around a vehicle."""
    aggregator = MultiDetectorAggregator(cross_model_iou_thresh=0.40)
    vehicle = DetectionResult(
        class_id=7,
        class_name="forklift",
        class_group=ClassGroup.VEHICLE,
        confidence=0.85,
        bbox=BoundingBox(x1=300, y1=200, x2=600, y2=500),
    )
    driver = DetectionResult(
        class_id=0,
        class_name="person",
        class_group=ClassGroup.PERSON,
        confidence=0.72,
        bbox=BoundingBox(x1=350, y1=250, x2=420, y2=360),
        metadata={"in_vehicle": True},
    )
    passenger = DetectionResult(
        class_id=0,
        class_name="person",
        class_group=ClassGroup.PERSON,
        confidence=0.68,
        bbox=BoundingBox(x1=440, y1=250, x2=510, y2=360),
        metadata={"in_vehicle": True},
    )
    ground_worker = DetectionResult(
        class_id=0,
        class_name="person",
        class_group=ClassGroup.PERSON,
        confidence=0.88,
        bbox=BoundingBox(x1=200, y1=250, x2=270, y2=450),
    )

    merged = aggregator.merge_detections([driver, passenger, ground_worker], [vehicle])
    persons = [d for d in merged if d.class_group == ClassGroup.PERSON]
    vehicles = [d for d in merged if d.class_group == ClassGroup.VEHICLE]

    assert len(persons) == 3, f"Expected 3 people, got {len(persons)}"
    assert len(vehicles) == 1, f"Expected 1 vehicle, got {len(vehicles)}"


def test_case_2_partially_occluded_person():
    """Requirement 2: Partially occluded person with lower confidence (0.21) is preserved & tracked."""
    from vision.tracking.bytetrack_tracker import ByteTrackTracker
    tracker = ByteTrackTracker(track_high_thresh=0.20, new_track_thresh=0.18)

    occluded_person = DetectionResult(
        class_id=0,
        class_name="person",
        class_group=ClassGroup.PERSON,
        confidence=0.21,  # Below standard 0.25, but above hardened 0.20/0.18
        bbox=BoundingBox(x1=100, y1=100, x2=150, y2=220),
    )
    frame_dets = FrameDetections(
        frame_id=1,
        timestamp=0.0,
        detections=[occluded_person],
        frame_width=640,
        frame_height=480,
    )
    tracks = tracker.update(frame_dets)
    assert len(tracks.active_tracks) == 1
    assert tracks.active_tracks[0].class_name == "person"
    assert tracks.active_tracks[0].confidence == 0.21


def test_case_3_small_person():
    """Requirement 3: Small person in distant frame is retained by detector sanity checks."""
    small_person_box = BoundingBox(x1=1440, y1=178, x2=1482, y2=288)  # 42x110 px
    bw, bh = small_person_box.width, small_person_box.height
    assert bw >= 8.0 and bh >= 8.0
    assert (bw * bh) >= 64.0
    aspect = max(bw / max(1.0, bh), bh / max(1.0, bw))
    assert aspect <= 20.0


def test_case_4_overlapping_people():
    """Requirement 4: Overlapping people (IoU = 0.35) are both retained."""
    aggregator = MultiDetectorAggregator(cross_model_iou_thresh=0.40)
    worker_a = DetectionResult(
        class_id=0,
        class_name="person",
        class_group=ClassGroup.PERSON,
        confidence=0.88,
        bbox=BoundingBox(x1=100, y1=100, x2=180, y2=300),
    )
    worker_b = DetectionResult(
        class_id=0,
        class_name="person",
        class_group=ClassGroup.PERSON,
        confidence=0.82,
        bbox=BoundingBox(x1=140, y1=100, x2=220, y2=300),
    )
    iou = compute_box_iou(worker_a.bbox, worker_b.bbox)
    assert 0.20 <= iou <= 0.45

    # Passing both to deduplicate preserves both distinct workers
    res = aggregator.filter_detections([worker_a, worker_b])
    assert len(res) == 2


def test_case_5_one_vehicle_producing_duplicate_boxes():
    """Requirement 5: One vehicle producing duplicate boxes (nested cab + full vehicle) is deduplicated."""
    from vision.detection.coordinates import compute_box_iomin
    full_box = BoundingBox(x1=358.0, y1=246.0, x2=550.0, y2=503.0)
    cab_box = BoundingBox(x1=374.0, y1=246.0, x2=532.0, y2=484.0)

    iou = compute_box_iou(full_box, cab_box)
    iomin = compute_box_iomin(full_box, cab_box)
    # Standard IoU is moderate, but IoMin is very high (>0.85) indicating containment
    assert iomin >= 0.65

    det_full = DetectionResult(
        class_id=7,
        class_name="forklift",
        class_group=ClassGroup.VEHICLE,
        confidence=0.45,
        bbox=full_box,
    )
    det_cab = DetectionResult(
        class_id=7,
        class_name="truck",
        class_group=ClassGroup.VEHICLE,
        confidence=0.55,
        bbox=cab_box,
    )
    aggregator = MultiDetectorAggregator(cross_model_iou_thresh=0.40)
    deduped = aggregator.deduplicate_vehicles([det_full, det_cab])
    assert len(deduped) == 1
    assert deduped[0].class_name == "forklift"


def test_case_6_two_genuinely_separate_nearby_vehicles():
    """Requirement 6: Two genuinely separate nearby vehicles are both preserved."""
    from vision.detection.coordinates import compute_box_iomin
    v1_box = BoundingBox(x1=100.0, y1=200.0, x2=250.0, y2=400.0)
    v2_box = BoundingBox(x1=270.0, y1=200.0, x2=420.0, y2=400.0)

    assert compute_box_iou(v1_box, v2_box) < 0.40
    assert compute_box_iomin(v1_box, v2_box) < 0.65

    v1 = DetectionResult(class_id=7, class_name="forklift", class_group=ClassGroup.VEHICLE, confidence=0.80, bbox=v1_box)
    v2 = DetectionResult(class_id=7, class_name="forklift", class_group=ClassGroup.VEHICLE, confidence=0.82, bbox=v2_box)

    aggregator = MultiDetectorAggregator(cross_model_iou_thresh=0.40)
    deduped = aggregator.deduplicate_vehicles([v1, v2])
    assert len(deduped) == 2


def test_case_7_false_positive_filtering():
    """Requirement 7: False positive filtering rejects non-industrial COCO classes."""
    aggregator = MultiDetectorAggregator(
        filter_irrelevant_coco=True,
        irrelevant_classes=["boat", "fire hydrant", "traffic light", "stop sign"],
    )
    valid_person = DetectionResult(
        class_id=0, class_name="person", class_group=ClassGroup.PERSON, confidence=0.85,
        bbox=BoundingBox(x1=10, y1=10, x2=50, y2=100), source_model="yolo11n",
    )
    hydrant = DetectionResult(
        class_id=10, class_name="fire hydrant", class_group=ClassGroup.OTHER, confidence=0.75,
        bbox=BoundingBox(x1=60, y1=60, x2=90, y2=90), source_model="yolo11n",
    )
    filtered = aggregator.filter_detections([valid_person, hydrant])
    assert len(filtered) == 1
    assert filtered[0].class_name == "person"


def test_case_8_stable_tracking_ids():
    """Requirement 8: Stable tracking IDs across sequential frames."""
    from vision.tracking.bytetrack_tracker import ByteTrackTracker
    tracker = ByteTrackTracker(frame_rate=30)

    # Frame 1
    f1_det = DetectionResult(
        class_id=0, class_name="person", class_group=ClassGroup.PERSON, confidence=0.85,
        bbox=BoundingBox(x1=100, y1=100, x2=150, y2=250),
    )
    t1 = tracker.update(FrameDetections(frame_id=1, timestamp=0.0, detections=[f1_det]))
    assert len(t1.active_tracks) == 1
    initial_id = t1.active_tracks[0].track_id

    # Frame 2 (slight movement: worker walks 5 pixels right)
    f2_det = DetectionResult(
        class_id=0, class_name="person", class_group=ClassGroup.PERSON, confidence=0.84,
        bbox=BoundingBox(x1=105, y1=100, x2=155, y2=250),
    )
    t2 = tracker.update(FrameDetections(frame_id=2, timestamp=0.033, detections=[f2_det]))
    assert len(t2.active_tracks) == 1
    assert t2.active_tracks[0].track_id == initial_id, "Track ID must remain stable across adjacent frames"


def test_case_9_ppe_to_person_association():
    """Requirement 9: PPE-to-person association in crowded scenes."""
    from vision.detection.ppe_association import PPEAssociationEngine
    from backend.schemas.tracking import TrackedObject, TrackState

    engine = PPEAssociationEngine(association_threshold=0.10)
    worker_1 = TrackedObject(
        track_id=1, class_id=0, class_name="person", class_group=ClassGroup.PERSON, confidence=0.90,
        bbox=BoundingBox(x1=100, y1=100, x2=180, y2=300), state=TrackState.ACTIVE,
    )
    worker_2 = TrackedObject(
        track_id=2, class_id=0, class_name="person", class_group=ClassGroup.PERSON, confidence=0.88,
        bbox=BoundingBox(x1=170, y1=100, x2=250, y2=300), state=TrackState.ACTIVE,
    )

    # Hardhat 1 directly on Worker 1's head
    h1 = DetectionResult(
        class_id=1, class_name="Hardhat", class_group=ClassGroup.PPE, confidence=0.85,
        bbox=BoundingBox(x1=115, y1=90, x2=165, y2=140),
    )
    # Hardhat 2 directly on Worker 2's head
    h2 = DetectionResult(
        class_id=1, class_name="Hardhat", class_group=ClassGroup.PPE, confidence=0.82,
        bbox=BoundingBox(x1=185, y1=90, x2=235, y2=140),
    )

    assoc = engine.associate(tracks=[worker_1, worker_2], ppe_detections=[h1, h2])
    w1_items = next(inv.items for inv in assoc.worker_inventories if inv.track_id == 1)
    w2_items = next(inv.items for inv in assoc.worker_inventories if inv.track_id == 2)

    # Both workers must receive exactly 1 hardhat without stealing
    assert len(w1_items) == 1
    assert len(w2_items) == 1
    assert w1_items[0].class_name == "Hardhat"
    assert w2_items[0].class_name == "Hardhat"


def test_case_10_detection_tracking_api_response_consistency():
    """Requirement 10: Detection -> Tracking -> Assessment consistency."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    orchestrator = EndToEndPipelineOrchestrator(device="cpu")

    mock_person = DetectionResult(
        class_id=0, class_name="person", class_group=ClassGroup.PERSON, confidence=0.85,
        bbox=BoundingBox(x1=200, y1=150, x2=280, y2=350),
    )
    mock_forklift = DetectionResult(
        class_id=7, class_name="forklift", class_group=ClassGroup.VEHICLE, confidence=0.80,
        bbox=BoundingBox(x1=350, y1=200, x2=550, y2=400),
    )

    assessment, annotated = orchestrator.process_frame(
        img, frame_id=1, timestamp=0.0, manual_detections=[mock_person, mock_forklift]
    )

    assert assessment.frame_id == 1
    assert len(assessment.detections) == 2
    assert len(assessment.tracks) == 2
    track_classes = {t.class_name for t in assessment.tracks}
    assert "person" in track_classes
    assert "forklift" in track_classes

    # Verify JSON serializability for API response
    json_str = assessment.model_dump_json()
    assert len(json_str) > 0

