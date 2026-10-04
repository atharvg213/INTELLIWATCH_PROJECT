"""
Unit tests for Step 5B PPE Compliance Engine and Temporal Confirmation.
Verifies:
  1. Worker with all required PPE -> COMPLIANT
  2. Worker missing one required item -> NON_COMPLIANT
  3. Worker missing multiple required items -> NON_COMPLIANT
  4. Insufficient visual evidence:
       a) Head clipped at top border -> Hardhat / Goggles evaluated as UNKNOWN -> status UNKNOWN
       b) Hands clipped at bottom/side border -> Gloves evaluated as UNKNOWN -> status UNKNOWN
       c) Worker too small/distant (height < 120px) -> Goggles evaluated as UNKNOWN
  5. Negative violation detection (e.g. 'NO-Hardhat') directly confirms MISSING
  6. Temporal stability:
       a) Single temporary missed detection does NOT emit a violation event
       b) Fluctuation (missed, detected, missed) does NOT emit a violation event
       c) Persistent missed detection for N consecutive frames triggers confirmed IndustrialEvent
       d) Explainability fields (WHO, WHAT, WHICH PPE, WHEN, WHY)
  7. Track lifecycle cleanup:
       Departed track IDs are purged from temporal confirmation memory.
"""
import sys
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from backend.schemas.events import EventType, IndustrialEvent, SeverityLevel
from backend.schemas.ppe import ComplianceStatus, PPEItemState
from backend.schemas.tracking import FrameTracks, TrackedObject
from intelligence.events.ppe_compliance import PPEComplianceEngine
from vision.detection.ppe_association import PPEAssociationEngine


def make_person(track_id: int, x1: float, y1: float, x2: float, y2: float) -> TrackedObject:
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


def make_ppe(class_name: str, x1: float, y1: float, x2: float, y2: float, conf: float = 0.88) -> DetectionResult:
    return DetectionResult(
        class_id=1,
        class_name=class_name,
        confidence=conf,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


# 1. Full Compliance: Person with all required PPE (Hardhat, Vest, Gloves, Goggles)
def test_compliance_all_required_present():
    engine = PPEComplianceEngine(
        required_ppe=["Hardhat", "Safety Vest", "Gloves", "Goggles"],
        confirmation_frames=3,
    )

    worker = make_person(track_id=17, x1=200.0, y1=100.0, x2=300.0, y2=400.0)
    hardhat = make_ppe("Hardhat", 220.0, 85.0, 280.0, 135.0)
    vest = make_ppe("Safety Vest", 210.0, 150.0, 290.0, 270.0)
    gloves = make_ppe("Gloves", 195.0, 280.0, 225.0, 330.0)
    goggles = make_ppe("Goggles", 235.0, 120.0, 265.0, 140.0)

    association, events = engine.process_frame(
        tracks=[worker],
        ppe_detections=[hardhat, vest, gloves, goggles],
        frame_id=1,
        timestamp=0.033,
        frame_width=640,
        frame_height=480,
    )

    assert len(association.worker_inventories) == 1
    inv = association.worker_inventories[0]
    assert inv.compliance_status == ComplianceStatus.COMPLIANT
    assert inv.ppe_status["Hardhat"] == PPEItemState.PRESENT
    assert inv.ppe_status["Safety Vest"] == PPEItemState.PRESENT
    assert inv.ppe_status["Gloves"] == PPEItemState.PRESENT
    assert inv.ppe_status["Goggles"] == PPEItemState.PRESENT
    assert len(inv.missing_ppe) == 0
    assert len(events) == 0


# 2. First-frame misses are UNKNOWN; repeated visible absence confirms MISSING
def test_non_compliance_missing_items():
    engine = PPEComplianceEngine(
        required_ppe=["Hardhat", "Safety Vest", "Gloves", "Goggles"],
        confirmation_frames=3,
    )

    # Worker has Hardhat and Vest, but no Gloves and no Goggles
    worker = make_person(track_id=10, x1=200.0, y1=100.0, x2=300.0, y2=400.0)
    hardhat = make_ppe("Hardhat", 220.0, 85.0, 280.0, 135.0)
    vest = make_ppe("Safety Vest", 210.0, 150.0, 290.0, 270.0)

    association, events = engine.process_frame(
        tracks=[worker],
        ppe_detections=[hardhat, vest],
        frame_id=1,
        timestamp=0.033,
        frame_width=640,
        frame_height=480,
    )

    inv = association.worker_inventories[0]
    assert inv.compliance_status == ComplianceStatus.UNKNOWN
    assert inv.ppe_status["Hardhat"] == PPEItemState.PRESENT
    assert inv.ppe_status["Safety Vest"] == PPEItemState.PRESENT
    assert inv.ppe_status["Gloves"] == PPEItemState.UNKNOWN
    assert inv.ppe_status["Goggles"] == PPEItemState.UNKNOWN
    assert inv.missing_ppe == []
    assert events == []

    for frame_id in (2, 3):
        association, events = engine.process_frame(
            tracks=[worker],
            ppe_detections=[hardhat, vest],
            frame_id=frame_id,
            timestamp=frame_id / 30.0,
            frame_width=640,
            frame_height=480,
        )

    inv = association.worker_inventories[0]
    assert inv.compliance_status == ComplianceStatus.NON_COMPLIANT
    assert inv.ppe_status["Gloves"] == PPEItemState.MISSING
    assert inv.ppe_status["Goggles"] == PPEItemState.MISSING
    assert set(inv.missing_ppe) == {"Gloves", "Goggles"}
    assert len(events) == 1


def test_one_frame_goggles_and_gloves_miss_does_not_erase_present_state():
    engine = PPEComplianceEngine(
        required_ppe=["Hardhat", "Safety Vest", "Gloves", "Goggles"],
        confirmation_frames=3,
    )
    worker = make_person(track_id=17, x1=200.0, y1=100.0, x2=300.0, y2=400.0)
    hardhat = make_ppe("Hardhat", 220.0, 85.0, 280.0, 135.0)
    vest = make_ppe("Safety Vest", 210.0, 150.0, 290.0, 270.0)
    gloves = make_ppe("Gloves", 195.0, 280.0, 225.0, 330.0)
    goggles = make_ppe("Goggles", 235.0, 120.0, 265.0, 140.0)

    first, _ = engine.process_frame(
        tracks=[worker], ppe_detections=[hardhat, vest, gloves, goggles],
        frame_id=1, frame_width=640, frame_height=480,
    )
    assert first.worker_inventories[0].ppe_status["Gloves"] == PPEItemState.PRESENT
    assert first.worker_inventories[0].ppe_status["Goggles"] == PPEItemState.PRESENT

    missed, events = engine.process_frame(
        tracks=[worker], ppe_detections=[hardhat, vest],
        frame_id=2, frame_width=640, frame_height=480,
    )
    assert missed.worker_inventories[0].ppe_status["Gloves"] == PPEItemState.UNKNOWN
    assert missed.worker_inventories[0].ppe_status["Goggles"] == PPEItemState.UNKNOWN
    assert missed.worker_inventories[0].missing_ppe == []
    assert events == []

    recovered, _ = engine.process_frame(
        tracks=[worker], ppe_detections=[hardhat, vest, gloves, goggles],
        frame_id=3, frame_width=640, frame_height=480,
    )
    assert recovered.worker_inventories[0].ppe_status["Gloves"] == PPEItemState.PRESENT
    assert recovered.worker_inventories[0].ppe_status["Goggles"] == PPEItemState.PRESENT


# 3. UNKNOWN vs NON_COMPLIANT: Boundary Truncation
def test_unknown_status_boundary_truncation():
    engine = PPEComplianceEngine(
        required_ppe=["Hardhat", "Safety Vest", "Gloves", "Goggles"],
        boundary_margin_px=5.0,
    )

    # Worker whose head is clipped at the top of the frame (y1 = 2px <= 5px margin)
    truncated_top_worker = make_person(track_id=1, x1=200.0, y1=2.0, x2=300.0, y2=300.0)
    vest = make_ppe("Safety Vest", 210.0, 80.0, 290.0, 200.0)
    gloves = make_ppe("Gloves", 200.0, 220.0, 230.0, 260.0)

    for frame_id in range(1, 5):
        association, _ = engine.process_frame(
            tracks=[truncated_top_worker],
            ppe_detections=[vest, gloves],
            frame_id=frame_id,
            timestamp=frame_id / 30.0,
            frame_width=640,
            frame_height=480,
        )

    inv = association.worker_inventories[0]
    # Hardhat and Goggles cannot be verified because head is out of frame -> UNKNOWN
    assert inv.ppe_status["Hardhat"] == PPEItemState.UNKNOWN
    assert inv.ppe_status["Goggles"] == PPEItemState.UNKNOWN
    assert inv.ppe_status["Safety Vest"] == PPEItemState.PRESENT
    assert inv.ppe_status["Gloves"] == PPEItemState.PRESENT
    # Since none are definitely MISSING, but some are UNKNOWN -> overall status UNKNOWN
    assert inv.compliance_status == ComplianceStatus.UNKNOWN

    hands_clipped_worker = make_person(track_id=9, x1=200.0, y1=100.0, x2=300.0, y2=478.0)
    for frame_id in range(10, 14):
        association, _ = engine.process_frame(
            tracks=[hands_clipped_worker],
            ppe_detections=[],
            frame_id=frame_id,
            timestamp=frame_id / 30.0,
            frame_width=640,
            frame_height=480,
        )
    hands_inventory = association.worker_inventories[0]
    assert hands_inventory.ppe_status["Gloves"] == PPEItemState.UNKNOWN


# 4. UNKNOWN vs NON_COMPLIANT: Distance / Small Box Resolution
def test_unknown_status_small_person_resolution():
    engine = PPEComplianceEngine(
        required_ppe=["Hardhat", "Safety Vest", "Gloves", "Goggles"],
        min_goggles_height=120.0,
        min_gloves_height=100.0,
    )

    # Worker is small (height = 90px < 120px and < 100px)
    distant_worker = make_person(track_id=2, x1=200.0, y1=200.0, x2=240.0, y2=290.0)
    hardhat = make_ppe("Hardhat", 205.0, 195.0, 235.0, 215.0)
    vest = make_ppe("Safety Vest", 205.0, 215.0, 235.0, 255.0)

    for frame_id in range(1, 5):
        association, _ = engine.process_frame(
            tracks=[distant_worker],
            ppe_detections=[hardhat, vest],
            frame_id=frame_id,
            timestamp=frame_id / 30.0,
            frame_width=640,
            frame_height=480,
        )

    inv = association.worker_inventories[0]
    # Goggles and Gloves should be UNKNOWN due to distance
    assert inv.ppe_status["Goggles"] == PPEItemState.UNKNOWN
    assert inv.ppe_status["Gloves"] == PPEItemState.UNKNOWN
    assert inv.compliance_status == ComplianceStatus.UNKNOWN


# 5. Explicit negative detection is MISSING immediately; alerts remain temporal
def test_explicit_negative_detection():
    engine = PPEComplianceEngine(
        required_ppe=["Hardhat", "Safety Vest"],
        confirmation_frames=3,
    )

    worker = make_person(track_id=3, x1=200.0, y1=100.0, x2=300.0, y2=400.0)
    # Detector explicitly detected 'NO-Hardhat'
    no_hardhat = make_ppe("NO-Hardhat", 220.0, 85.0, 280.0, 135.0)
    vest = make_ppe("Safety Vest", 210.0, 150.0, 290.0, 270.0)

    association, events = engine.process_frame(
        tracks=[worker],
        ppe_detections=[no_hardhat, vest],
        frame_id=1,
        frame_width=640,
        frame_height=480,
    )

    inv = association.worker_inventories[0]
    assert inv.ppe_status["Hardhat"] == PPEItemState.MISSING
    assert inv.compliance_status == ComplianceStatus.NON_COMPLIANT
    assert events == []

    for frame_id in (2, 3):
        association, events = engine.process_frame(
            tracks=[worker],
            ppe_detections=[no_hardhat, vest],
            frame_id=frame_id,
            frame_width=640,
            frame_height=480,
        )

    inv = association.worker_inventories[0]
    assert inv.ppe_status["Hardhat"] == PPEItemState.MISSING
    assert inv.ppe_status["Safety Vest"] == PPEItemState.PRESENT
    assert inv.compliance_status == ComplianceStatus.NON_COMPLIANT
    assert len(events) == 1


def test_low_confidence_negative_requires_repeated_frames_before_missing():
    engine = PPEComplianceEngine(required_ppe=["Safety Vest"], confirmation_frames=3)
    worker = make_person(track_id=30, x1=100.0, y1=100.0, x2=200.0, y2=400.0)
    weak_no_vest = DetectionResult(
        class_id=8,
        class_name="NO-Safety Vest",
        confidence=0.16,
        bbox=BoundingBox(x1=110.0, y1=180.0, x2=190.0, y2=280.0),
        metadata={"low_confidence_candidate": True, "crop_parent_track": 30},
    )

    for frame_id in (1, 2):
        association, events = engine.process_frame(
            tracks=[worker], ppe_detections=[weak_no_vest], frame_id=frame_id,
            timestamp=frame_id / 30.0, frame_width=640, frame_height=480,
        )
        inv = association.worker_inventories[0]
        assert inv.ppe_status["Safety Vest"] == PPEItemState.UNKNOWN
        assert events == []

    association, events = engine.process_frame(
        tracks=[worker], ppe_detections=[weak_no_vest], frame_id=3,
        timestamp=3 / 30.0, frame_width=640, frame_height=480,
    )
    inv = association.worker_inventories[0]
    assert inv.ppe_status["Safety Vest"] == PPEItemState.MISSING
    assert inv.missing_ppe == ["Safety Vest"]
    assert len(events) == 1


# 6. Temporal Confirmation: Jitter / Fluctuations Do NOT Trigger Violations
def test_temporal_jitter_does_not_trigger_event():
    engine = PPEComplianceEngine(
        required_ppe=["Hardhat", "Safety Vest"],
        confirmation_frames=3,
    )

    worker = make_person(track_id=17, x1=200.0, y1=100.0, x2=300.0, y2=400.0)
    hardhat = make_ppe("Hardhat", 220.0, 85.0, 280.0, 135.0)
    vest = make_ppe("Safety Vest", 210.0, 150.0, 290.0, 270.0)

    # Frame 1: Vest missing
    _, ev1 = engine.process_frame(
        tracks=[worker],
        ppe_detections=[hardhat],
        frame_id=101,
        timestamp=1.01,
        frame_width=640,
        frame_height=480,
    )
    assert len(ev1) == 0

    # Frame 2: Vest temporarily detected again
    _, ev2 = engine.process_frame(
        tracks=[worker],
        ppe_detections=[hardhat, vest],
        frame_id=102,
        timestamp=1.02,
        frame_width=640,
        frame_height=480,
    )
    assert len(ev2) == 0

    # Frame 3: Vest missing again (broken sequence -> consecutive is 1, not 3)
    _, ev3 = engine.process_frame(
        tracks=[worker],
        ppe_detections=[hardhat],
        frame_id=103,
        timestamp=1.03,
        frame_width=640,
        frame_height=480,
    )
    assert len(ev3) == 0


# 7. Temporal Confirmation: Persistent Missing PPE Confirms Violation Event
def test_persistent_missing_ppe_confirms_event():
    engine = PPEComplianceEngine(
        required_ppe=["Hardhat", "Safety Vest"],
        confirmation_frames=3,
        camera_id="cam_factory_01",
    )

    worker = make_person(track_id=17, x1=200.0, y1=100.0, x2=300.0, y2=400.0)
    hardhat = make_ppe("Hardhat", 220.0, 85.0, 280.0, 135.0)

    # Frame 1: Non-compliant (count = 1) -> No event
    _, ev1 = engine.process_frame([worker], [hardhat], frame_id=1, timestamp=0.1, frame_width=640, frame_height=480)
    assert len(ev1) == 0

    # Frame 2: Non-compliant (count = 2) -> No event
    _, ev2 = engine.process_frame([worker], [hardhat], frame_id=2, timestamp=0.2, frame_width=640, frame_height=480)
    assert len(ev2) == 0

    # Frame 3: Non-compliant (count = 3 >= confirmation_frames) -> Event CONFIRMED!
    _, ev3 = engine.process_frame([worker], [hardhat], frame_id=3, timestamp=0.3, frame_width=640, frame_height=480)
    assert len(ev3) == 1

    event = ev3[0]
    assert isinstance(event, IndustrialEvent)
    assert event.event_type == EventType.PPE_VIOLATION
    assert event.camera_id == "cam_factory_01"
    assert event.tracked_object_ids == [17]
    assert event.severity == SeverityLevel.HIGH
    assert "Worker #17 PPE violation confirmed" in event.explanation
    assert "Safety Vest" in event.explanation
    assert event.metadata["missing_ppe"] == ["Safety Vest"]
    assert event.metadata["detected_ppe"] == ["Hardhat"]
    assert event.metadata["consecutive_violation_count"] == 3


# 8. Track Disappearance Cleans Up Temporal Memory
def test_track_disappearance_cleanup():
    engine = PPEComplianceEngine(confirmation_frames=3)

    worker1 = make_person(track_id=1, x1=100.0, y1=100.0, x2=200.0, y2=300.0)
    worker2 = make_person(track_id=2, x1=400.0, y1=100.0, x2=500.0, y2=300.0)

    # Frame 1: Both workers present
    engine.process_frame([worker1, worker2], [], frame_id=1, frame_width=640, frame_height=480)
    assert 1 in engine._track_states
    assert 2 in engine._track_states

    # Frame 2: Worker 1 disappeared from scene (only worker 2 is tracked)
    engine.process_frame([worker2], [], frame_id=2, frame_width=640, frame_height=480)
    assert 1 not in engine._track_states
    assert 2 in engine._track_states
