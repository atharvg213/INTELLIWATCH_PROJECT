"""
Unit tests for Step 5B Person-PPE Spatial Association Engine.
Verifies:
  1. Class normalization and category filtering
  2. Spatial containment and anatomical scoring
  3. Single person - multiple PPE item associations
  4. Nearby worker disambiguation (closest/best spatial evidence)
  5. Weak/implausible spatial overlap rejection (unassociated PPE)
  6. Distant/unrelated PPE rejection (unassociated PPE)
  7. Multiple PPE items of the same class (e.g. left and right gloves)
  8. Multiple workers evaluated independently
  9. Graceful handling of empty tracks and empty detections
"""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from backend.schemas.ppe import FramePPEAssociation, WorkerPPEInventory
from backend.schemas.tracking import FrameTracks, TrackedObject
from vision.detection.ppe_association import PPEAssociationEngine


def create_person_track(
    track_id: int,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.90,
) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def create_ppe_det(
    class_id: int,
    class_name: str,
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    confidence: float = 0.85,
) -> DetectionResult:
    return DetectionResult(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


# 1. Class Normalization
def test_class_normalization():
    engine = PPEAssociationEngine()

    assert engine.normalize_class("person") is None
    assert engine.normalize_class("worker") is None

    # Positive items
    assert engine.normalize_class("Hardhat") == ("Hardhat", False)
    assert engine.normalize_class("helmet") == ("Hardhat", False)
    assert engine.normalize_class("Safety Vest") == ("Safety Vest", False)
    assert engine.normalize_class("gloves") == ("Gloves", False)
    assert engine.normalize_class("goggles") == ("Goggles", False)

    # Negative items
    assert engine.normalize_class("NO-Hardhat") == ("Hardhat", True)
    assert engine.normalize_class("NO-Safety Vest") == ("Safety Vest", True)
    assert engine.normalize_class("no-gloves") == ("Gloves", True)
    assert engine.normalize_class("no-goggles") == ("Goggles", True)


# 2. Correct Association for Single Worker with Full PPE
def test_single_worker_all_ppe_association():
    engine = PPEAssociationEngine()

    # Worker: x=200..300, y=100..400 (W=100, H=300)
    worker = create_person_track(track_id=17, x1=200.0, y1=100.0, x2=300.0, y2=400.0)

    # Hardhat near head
    hardhat = create_ppe_det(1, "Hardhat", 220.0, 85.0, 280.0, 135.0)
    # Vest on torso
    vest = create_ppe_det(2, "Safety Vest", 210.0, 150.0, 290.0, 270.0)
    # Gloves near hands
    gloves = create_ppe_det(3, "Gloves", 195.0, 280.0, 225.0, 330.0)
    # Goggles near face
    goggles = create_ppe_det(4, "Goggles", 235.0, 120.0, 265.0, 140.0)

    association = engine.associate(
        tracks=[worker],
        ppe_detections=[hardhat, vest, gloves, goggles],
        frame_id=1,
        timestamp=0.033,
    )

    assert isinstance(association, FramePPEAssociation)
    assert len(association.worker_inventories) == 1
    assert len(association.unassociated_ppe) == 0

    inv = association.worker_inventories[0]
    assert inv.track_id == 17
    assert len(inv.items) == 4

    associated_classes = {item.class_name for item in inv.items}
    assert associated_classes == {"Hardhat", "Safety Vest", "Gloves", "Goggles"}
    for item in inv.items:
        assert item.association_score >= engine.association_threshold


# 3. Unassociated PPE (Distant gear)
def test_distant_ppe_remains_unassociated():
    engine = PPEAssociationEngine()

    # Worker at left side: x=100..200, y=100..400
    worker = create_person_track(track_id=1, x1=100.0, y1=100.0, x2=200.0, y2=400.0)

    # Helmet far away at right side: x=800..850, y=100..150
    distant_helmet = create_ppe_det(1, "Hardhat", 800.0, 100.0, 850.0, 150.0)

    association = engine.associate(
        tracks=[worker],
        ppe_detections=[distant_helmet],
        frame_id=2,
    )

    inv = association.worker_inventories[0]
    assert len(inv.items) == 0
    assert len(association.unassociated_ppe) == 1
    assert association.unassociated_ppe[0].class_name == "Hardhat"


# 4. Nearby Worker Disambiguation
def test_nearby_worker_disambiguation():
    engine = PPEAssociationEngine()

    # Worker A: x=100..200, y=100..400 (center_x=150)
    worker_a = create_person_track(track_id=10, x1=100.0, y1=100.0, x2=200.0, y2=400.0)
    # Worker B: x=250..350, y=100..400 (center_x=300)
    worker_b = create_person_track(track_id=20, x1=250.0, y1=100.0, x2=350.0, y2=400.0)

    # Helmet positioned directly over Worker A's head (x=130..170)
    helmet_a = create_ppe_det(1, "Hardhat", 130.0, 85.0, 170.0, 130.0)

    association = engine.associate(
        tracks=[worker_a, worker_b],
        ppe_detections=[helmet_a],
    )

    inv_map = {inv.track_id: inv for inv in association.worker_inventories}
    assert len(inv_map[10].items) == 1
    assert inv_map[10].items[0].class_name == "Hardhat"
    assert len(inv_map[20].items) == 0
    assert len(association.unassociated_ppe) == 0


def test_worker_roi_ppe_cannot_be_reassigned_to_a_nearby_track():
    engine = PPEAssociationEngine()
    worker_a = create_person_track(track_id=10, x1=100.0, y1=100.0, x2=200.0, y2=400.0)
    worker_b = create_person_track(track_id=20, x1=160.0, y1=100.0, x2=260.0, y2=400.0)
    roi_goggles = DetectionResult(
        class_id=4,
        class_name="Goggles",
        confidence=0.82,
        bbox=BoundingBox(x1=188.0, y1=120.0, x2=208.0, y2=140.0),
        metadata={"crop_parent_track": 10},
    )

    association = engine.associate(
        tracks=[worker_a, worker_b],
        ppe_detections=[roi_goggles],
    )

    inv_map = {inv.track_id: inv for inv in association.worker_inventories}
    assert [item.class_name for item in inv_map[10].items] == ["Goggles"]
    assert inv_map[20].items == []
    assert association.unassociated_ppe == []


def test_low_confidence_positive_roi_detection_survives_strong_spatial_match():
    engine = PPEAssociationEngine()
    worker = create_person_track(track_id=10, x1=100.0, y1=100.0, x2=200.0, y2=400.0)
    vest = DetectionResult(
        class_id=4,
        class_name="Safety Vest",
        confidence=0.16,
        bbox=BoundingBox(x1=110.0, y1=180.0, x2=190.0, y2=280.0),
        metadata={"low_confidence_candidate": True, "crop_parent_track": 10},
    )

    association = engine.associate(tracks=[worker], ppe_detections=[vest])

    inventory = association.worker_inventories[0]
    assert [item.class_name for item in inventory.items] == ["Safety Vest"]
    assert inventory.items[0].confidence == 0.16


def test_duplicate_positive_boxes_do_not_exceed_one_item_capacity():
    engine = PPEAssociationEngine()
    worker = create_person_track(track_id=10, x1=100.0, y1=100.0, x2=200.0, y2=400.0)
    stronger = create_ppe_det(3, "Safety Vest", 110.0, 180.0, 190.0, 280.0, confidence=0.88)
    duplicate = create_ppe_det(3, "Safety Vest", 112.0, 182.0, 188.0, 278.0, confidence=0.52)

    association = engine.associate(tracks=[worker], ppe_detections=[stronger, duplicate])

    items = association.worker_inventories[0].items
    assert len(items) == 1
    assert items[0].class_name == "Safety Vest"
    assert items[0].confidence == 0.88


def test_positive_ppe_detection_wins_over_conflicting_negative_box():
    engine = PPEAssociationEngine()
    worker = create_person_track(track_id=10, x1=100.0, y1=100.0, x2=200.0, y2=400.0)
    vest = create_ppe_det(3, "Safety Vest", 110.0, 180.0, 190.0, 280.0, confidence=0.32)
    no_vest = create_ppe_det(8, "NO-Safety Vest", 110.0, 180.0, 190.0, 280.0, confidence=0.86)

    association = engine.associate(tracks=[worker], ppe_detections=[no_vest, vest])

    items = association.worker_inventories[0].items
    assert len(items) == 1
    assert items[0].class_name == "Safety Vest"
    assert items[0].is_negative is False


def test_low_confidence_negative_candidate_keeps_evidence_for_temporal_review():
    engine = PPEAssociationEngine()
    worker = create_person_track(track_id=10, x1=100.0, y1=100.0, x2=200.0, y2=400.0)
    weak_negative = DetectionResult(
        class_id=8,
        class_name="NO-Safety Vest",
        confidence=0.16,
        bbox=BoundingBox(x1=110.0, y1=180.0, x2=190.0, y2=280.0),
        metadata={"low_confidence_candidate": True, "crop_parent_track": 10},
    )

    association = engine.associate(tracks=[worker], ppe_detections=[weak_negative])

    items = association.worker_inventories[0].items
    assert len(items) == 1
    assert items[0].is_negative is True
    assert items[0].low_confidence_candidate is True
    assert association.unassociated_ppe == []


def test_gloves_at_wrist_edges_remain_in_anatomical_region():
    engine = PPEAssociationEngine()
    worker = create_person_track(track_id=7, x1=200.0, y1=100.0, x2=300.0, y2=400.0)
    glove = create_ppe_det(3, "Gloves", 175.0, 310.0, 198.0, 340.0)

    association = engine.associate(tracks=[worker], ppe_detections=[glove])

    assert [item.class_name for item in association.worker_inventories[0].items] == ["Gloves"]


# 5. Weak Spatial Overlap / Implausible Anatomical Region Rejection
def test_weak_spatial_or_wrong_region_rejected():
    engine = PPEAssociationEngine()

    # Worker: x=100..200, y=100..400 (Head is at y=100..160, Feet at y=360..400)
    worker = create_person_track(track_id=5, x1=100.0, y1=100.0, x2=200.0, y2=400.0)

    # Helmet detected near feet (y=370..395) -> Implausible anatomical position for Hardhat
    misplaced_helmet = create_ppe_det(1, "Hardhat", 130.0, 370.0, 170.0, 395.0)

    association = engine.associate(
        tracks=[worker],
        ppe_detections=[misplaced_helmet],
    )

    assert len(association.worker_inventories[0].items) == 0
    assert len(association.unassociated_ppe) == 1


# 6. Multiple PPE Items of the Same Class (e.g., Left and Right Gloves)
def test_multiple_ppe_items_same_class():
    engine = PPEAssociationEngine()

    worker = create_person_track(track_id=7, x1=200.0, y1=100.0, x2=300.0, y2=400.0)

    # Left hand glove (x=190..215, y=280..320)
    left_glove = create_ppe_det(3, "Gloves", 190.0, 280.0, 215.0, 320.0)
    # Right hand glove (x=285..310, y=280..320)
    right_glove = create_ppe_det(3, "Gloves", 285.0, 280.0, 310.0, 320.0)

    association = engine.associate(
        tracks=[worker],
        ppe_detections=[left_glove, right_glove],
    )

    inv = association.worker_inventories[0]
    assert len(inv.items) == 2
    for item in inv.items:
        assert item.class_name == "Gloves"


# 7. Multiple Tracked Workers Independent Association
def test_multiple_workers_independent_assignment():
    engine = PPEAssociationEngine()

    # Worker 1 at x=100..200
    w1 = create_person_track(track_id=1, x1=100.0, y1=100.0, x2=200.0, y2=400.0)
    # Worker 2 at x=500..600
    w2 = create_person_track(track_id=2, x1=500.0, y1=100.0, x2=600.0, y2=400.0)

    h1 = create_ppe_det(1, "Hardhat", 120.0, 90.0, 180.0, 140.0)
    v2 = create_ppe_det(2, "Safety Vest", 520.0, 160.0, 580.0, 280.0)

    association = engine.associate(
        tracks=[w1, w2],
        ppe_detections=[h1, v2],
    )

    inv_map = {inv.track_id: inv for inv in association.worker_inventories}
    assert len(inv_map[1].items) == 1
    assert inv_map[1].items[0].class_name == "Hardhat"

    assert len(inv_map[2].items) == 1
    assert inv_map[2].items[0].class_name == "Safety Vest"


# 8. Empty / Edge Cases Handling
def test_empty_cases_handling():
    engine = PPEAssociationEngine()

    # No tracks, with PPE detections
    helmet = create_ppe_det(1, "Hardhat", 100.0, 100.0, 150.0, 150.0)
    assoc_no_tracks = engine.associate(tracks=[], ppe_detections=[helmet])
    assert len(assoc_no_tracks.worker_inventories) == 0
    assert len(assoc_no_tracks.unassociated_ppe) == 1

    # Tracks present, no PPE detections
    worker = create_person_track(track_id=1, x1=100.0, y1=100.0, x2=200.0, y2=400.0)
    assoc_no_dets = engine.associate(tracks=[worker], ppe_detections=[])
    assert len(assoc_no_dets.worker_inventories) == 1
    assert len(assoc_no_dets.worker_inventories[0].items) == 0
    assert len(assoc_no_dets.unassociated_ppe) == 0
