"""
Unit tests for Restricted Zone Spatial Reasoning, Point-in-Polygon,
Foot-Point Calculation, Temporal Confirmation, Dwell Time, and Zone Events.
"""
import sys
from pathlib import Path
import pytest
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox
from backend.schemas.events import EventType, IndustrialEvent, SeverityLevel
from backend.schemas.tracking import FrameTracks, TrackPoint, TrackedObject
from backend.schemas.zones import (
    FrameZoneOccupancy,
    RestrictedZone,
    ZoneMembership,
    ZoneMembershipStatus,
    ZoneType,
)
from intelligence.zones.zone_engine import ZoneEngine, compute_contact_point
from vision.tracking.visualizer import TrackingVisualizer


def _create_tracked_worker(
    track_id: int,
    bbox: BoundingBox,
    class_name: str = "person",
    confidence: float = 0.90,
) -> TrackedObject:
    """Helper to instantiate a synthetic TrackedObject."""
    return TrackedObject(
        track_id=track_id,
        class_id=0 if class_name == "person" else 1,
        class_name=class_name,
        confidence=confidence,
        bbox=bbox,
        trajectory=[
            TrackPoint(
                frame_id=1,
                x=bbox.center[0],
                y=bbox.center[1],
                timestamp=0.0,
            )
        ],
    )


# Standard test polygon: 5-sided convex polygon in [100, 100] to [400, 400] region
SAMPLE_POLYGON = [
    [100.0, 100.0],
    [400.0, 100.0],
    [450.0, 300.0],
    [350.0, 400.0],
    [100.0, 400.0],
]


@pytest.fixture
def sample_zone() -> RestrictedZone:
    return RestrictedZone(
        zone_id="test_zone_a",
        name="Test High Voltage Area",
        zone_type=ZoneType.RESTRICTED,
        polygon=SAMPLE_POLYGON,
        enabled=True,
        max_dwell_seconds=5.0,
    )


# ---------------------------------------------------------------------------
# Test 1 & 2: Point Clearly Outside vs Inside Polygon
# ---------------------------------------------------------------------------

def test_point_clearly_outside_polygon(sample_zone):
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)
    is_inside, status = engine.test_point_in_zone((50.0, 50.0), sample_zone.zone_id)
    assert is_inside is False
    assert status == ZoneMembershipStatus.OUTSIDE


def test_point_clearly_inside_polygon(sample_zone):
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)
    is_inside, status = engine.test_point_in_zone((200.0, 200.0), sample_zone.zone_id)
    assert is_inside is True
    assert status == ZoneMembershipStatus.INSIDE


# ---------------------------------------------------------------------------
# Test 3: Point On Polygon Boundary Handled Consistently
# ---------------------------------------------------------------------------

def test_point_on_polygon_boundary(sample_zone):
    """
    Points exactly on polygon boundary edges are classified as BOUNDARY and is_inside=True
    under industrial safety-first conventions.
    """
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)
    # (250.0, 100.0) lies exactly on the line segment [100.0, 100.0] -> [400.0, 100.0]
    is_inside, status = engine.test_point_in_zone((250.0, 100.0), sample_zone.zone_id)
    assert is_inside is True
    assert status == ZoneMembershipStatus.BOUNDARY


# ---------------------------------------------------------------------------
# Test 4: Worker Entering Zone -> ENTRY Event
# ---------------------------------------------------------------------------

def test_worker_entering_zone_generates_entry_event(sample_zone):
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)

    # Worker foot point at (200.0, 200.0) which is inside polygon
    worker = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))
    memberships, events = engine.process_frame([worker], timestamp=10.0, frame_id=1)

    assert len(memberships) == 1
    assert memberships[0].is_inside is True
    assert memberships[0].track_id == 17
    assert memberships[0].zone_id == "test_zone_a"

    assert len(events) == 1
    entry_evt = events[0]
    assert entry_evt.event_type == EventType.ZONE_ENTRY
    assert entry_evt.tracked_object_ids == [17]
    assert entry_evt.metadata["zone_id"] == "test_zone_a"
    assert entry_evt.severity == SeverityLevel.HIGH


# ---------------------------------------------------------------------------
# Test 5: Worker Remaining Inside -> No Duplicate ENTRY Events
# ---------------------------------------------------------------------------

def test_worker_remaining_inside_no_duplicate_entry(sample_zone):
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)
    worker = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))

    # Frame 1: Enters and confirmed
    _, events_f1 = engine.process_frame([worker], timestamp=10.0, frame_id=1)
    assert len(events_f1) == 1
    assert events_f1[0].event_type == EventType.ZONE_ENTRY

    # Frame 2: Remains inside -> no new entry event
    _, events_f2 = engine.process_frame([worker], timestamp=11.0, frame_id=2)
    assert len(events_f2) == 0

    # Frame 3: Still inside -> no new entry event
    _, events_f3 = engine.process_frame([worker], timestamp=12.0, frame_id=3)
    assert len(events_f3) == 0


# ---------------------------------------------------------------------------
# Test 6: Worker Leaving Zone -> EXIT Event
# ---------------------------------------------------------------------------

def test_worker_leaving_zone_generates_exit_event(sample_zone):
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)
    worker_inside = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))
    worker_outside = _create_tracked_worker(17, BoundingBox(x1=10.0, y1=10.0, x2=40.0, y2=50.0))

    # Frame 1: Inside (Entry at t=10.0)
    engine.process_frame([worker_inside], timestamp=10.0, frame_id=1)

    # Frame 2: Outside (Exit at t=14.5)
    memberships, events = engine.process_frame([worker_outside], timestamp=14.5, frame_id=2)

    assert len(memberships) == 1
    assert memberships[0].is_inside is False
    assert len(events) == 1

    exit_evt = events[0]
    assert exit_evt.event_type == EventType.ZONE_EXIT
    assert exit_evt.tracked_object_ids == [17]
    assert exit_evt.metadata["dwell_seconds"] == pytest.approx(4.5, rel=1e-2)
    assert exit_evt.metadata["entry_timestamp"] == 10.0
    assert exit_evt.metadata["exit_timestamp"] == 14.5


# ---------------------------------------------------------------------------
# Test 7: Worker Enters Fewer than Confirmation Frames -> No Confirmed Entry
# ---------------------------------------------------------------------------

def test_worker_enters_fewer_than_confirmation_frames_no_entry(sample_zone):
    # Require 3 frames to confirm
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=3)
    worker_inside = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))
    worker_outside = _create_tracked_worker(17, BoundingBox(x1=10.0, y1=10.0, x2=40.0, y2=50.0))

    # Frame 1: Inside (1 of 3) -> No event
    _, events_f1 = engine.process_frame([worker_inside], timestamp=1.0, frame_id=1)
    assert len(events_f1) == 0

    # Frame 2: Inside (2 of 3) -> No event
    _, events_f2 = engine.process_frame([worker_inside], timestamp=2.0, frame_id=2)
    assert len(events_f2) == 0

    # Frame 3: Leaves zone before reaching confirmation threshold
    _, events_f3 = engine.process_frame([worker_outside], timestamp=3.0, frame_id=3)
    assert len(events_f3) == 0  # No entry was ever confirmed, so no exit event either


# ---------------------------------------------------------------------------
# Test 8: Worker Remains Inside for Confirmation Period -> Confirmed ENTRY
# ---------------------------------------------------------------------------

def test_worker_remains_inside_for_confirmation_period(sample_zone):
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=3)
    worker = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))

    # Frame 1: Inside (count=1)
    _, e1 = engine.process_frame([worker], timestamp=10.0, frame_id=1)
    assert len(e1) == 0

    # Frame 2: Inside (count=2)
    _, e2 = engine.process_frame([worker], timestamp=10.5, frame_id=2)
    assert len(e2) == 0

    # Frame 3: Inside (count=3 == threshold) -> Confirmed ENTRY!
    _, e3 = engine.process_frame([worker], timestamp=11.0, frame_id=3)
    assert len(e3) == 1
    assert e3[0].event_type == EventType.ZONE_ENTRY
    assert e3[0].tracked_object_ids == [17]


# ---------------------------------------------------------------------------
# Test 9: Dwell Time Increases While Inside
# ---------------------------------------------------------------------------

def test_dwell_time_increases_while_inside(sample_zone):
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)
    worker = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))

    # Frame 1 at t=10.0 (entry)
    m1, _ = engine.process_frame([worker], timestamp=10.0, frame_id=1)
    assert m1[0].dwell_seconds == 0.0

    # Frame 2 at t=12.5 -> dwell = 2.5s
    m2, _ = engine.process_frame([worker], timestamp=12.5, frame_id=2)
    assert m2[0].dwell_seconds == pytest.approx(2.5, rel=1e-2)

    # Frame 3 at t=14.0 -> dwell = 4.0s
    m3, _ = engine.process_frame([worker], timestamp=14.0, frame_id=3)
    assert m3[0].dwell_seconds == pytest.approx(4.0, rel=1e-2)


# ---------------------------------------------------------------------------
# Test 10 & 11: Dwell Threshold Exceeded -> One Event, Not Repeated
# ---------------------------------------------------------------------------

def test_dwell_threshold_exceeded_emits_once(sample_zone):
    # sample_zone has max_dwell_seconds = 5.0
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)
    worker = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))

    # Frame 1: Entry at t=10.0
    engine.process_frame([worker], timestamp=10.0, frame_id=1)

    # Frame 2: t=14.0 (dwell=4.0s < 5.0s) -> No dwell event
    _, e2 = engine.process_frame([worker], timestamp=14.0, frame_id=2)
    assert len(e2) == 0

    # Frame 3: t=15.5 (dwell=5.5s >= 5.0s) -> ZONE_DWELL_EXCEEDED emitted!
    _, e3 = engine.process_frame([worker], timestamp=15.5, frame_id=3)
    assert len(e3) == 1
    assert e3[0].event_type == EventType.ZONE_DWELL_EXCEEDED
    assert e3[0].severity == SeverityLevel.CRITICAL
    assert e3[0].metadata["dwell_seconds"] == pytest.approx(5.5, rel=1e-2)

    # Frame 4: t=17.0 (dwell=7.0s) -> MUST NOT repeat dwell event!
    _, e4 = engine.process_frame([worker], timestamp=17.0, frame_id=4)
    assert len(e4) == 0

    # Frame 5: t=18.0 (dwell=8.0s) -> Still no duplicate
    _, e5 = engine.process_frame([worker], timestamp=18.0, frame_id=5)
    assert len(e5) == 0


# ---------------------------------------------------------------------------
# Test 12: Worker Exits -> Dwell State Reset and Re-entry Timer Resets
# ---------------------------------------------------------------------------

def test_worker_exits_resets_dwell_state(sample_zone):
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)
    worker_in = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))
    worker_out = _create_tracked_worker(17, BoundingBox(x1=10.0, y1=10.0, x2=40.0, y2=50.0))

    # Episode 1: Worker enters at t=10.0, exceeds dwell at t=16.0
    engine.process_frame([worker_in], timestamp=10.0, frame_id=1)
    engine.process_frame([worker_in], timestamp=16.0, frame_id=2)

    # Worker exits at t=20.0
    _, exit_events = engine.process_frame([worker_out], timestamp=20.0, frame_id=3)
    assert len(exit_events) == 1
    assert exit_events[0].event_type == EventType.ZONE_EXIT

    # Episode 2: Worker re-enters at t=30.0 -> Timer is reset!
    m_reentry, entry_events = engine.process_frame([worker_in], timestamp=30.0, frame_id=4)
    assert len(entry_events) == 1
    assert entry_events[0].event_type == EventType.ZONE_ENTRY
    assert m_reentry[0].dwell_seconds == 0.0

    # At t=32.0 (dwell=2.0s < 5.0s), no dwell exceeded event
    _, e_mid = engine.process_frame([worker_in], timestamp=32.0, frame_id=5)
    assert len(e_mid) == 0


# ---------------------------------------------------------------------------
# Test 13: Multiple Zones Tracked Independently
# ---------------------------------------------------------------------------

def test_multiple_zones_tracked_independently():
    zone_a = RestrictedZone(
        zone_id="zone_a",
        name="Zone A - High Voltage",
        zone_type=ZoneType.RESTRICTED,
        polygon=[[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]],
        enabled=True,
    )
    zone_b = RestrictedZone(
        zone_id="zone_b",
        name="Zone B - Robotic Hazard",
        zone_type=ZoneType.HAZARD,
        polygon=[[400.0, 100.0], [600.0, 100.0], [600.0, 300.0], [400.0, 300.0]],
        enabled=True,
    )

    engine = ZoneEngine(zones=[zone_a, zone_b], confirmation_frames=1)

    # Worker in Zone A only (foot point = (200, 200))
    worker = _create_tracked_worker(10, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))
    m, events = engine.process_frame([worker], timestamp=1.0, frame_id=1)

    assert len(m) == 2
    mem_a = next(x for x in m if x.zone_id == "zone_a")
    mem_b = next(x for x in m if x.zone_id == "zone_b")

    assert mem_a.is_inside is True
    assert mem_b.is_inside is False

    assert len(events) == 1
    assert events[0].metadata["zone_id"] == "zone_a"


# ---------------------------------------------------------------------------
# Test 14: Multiple Workers States Remain Independent
# ---------------------------------------------------------------------------

def test_multiple_workers_states_remain_independent(sample_zone):
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)

    # Worker 1 inside (foot point = (200, 200))
    w1 = _create_tracked_worker(1, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))
    # Worker 2 outside (foot point = (25, 40))
    w2 = _create_tracked_worker(2, BoundingBox(x1=10.0, y1=10.0, x2=40.0, y2=40.0))

    m, events = engine.process_frame([w1, w2], timestamp=5.0, frame_id=1)

    w1_mem = next(x for x in m if x.track_id == 1)
    w2_mem = next(x for x in m if x.track_id == 2)

    assert w1_mem.is_inside is True
    assert w2_mem.is_inside is False

    assert len(events) == 1
    assert events[0].tracked_object_ids == [1]


# ---------------------------------------------------------------------------
# Test 15: Track Disappears -> Temporal State Cleaned Up and Alert Emitted
# ---------------------------------------------------------------------------

def test_track_disappears_cleans_up_state(sample_zone):
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)
    worker = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))

    # Frame 1: Worker inside (state created)
    engine.process_frame([worker], timestamp=10.0, frame_id=1)
    assert (17, "test_zone_a") in engine._actor_states

    # Frame 2: Worker disappears from active tracks
    _, events = engine.process_frame([], timestamp=15.0, frame_id=2)

    # State purged
    assert (17, "test_zone_a") not in engine._actor_states
    # Exit event emitted for departed track that was confirmed inside
    assert len(events) == 1
    assert events[0].event_type == EventType.ZONE_EXIT
    assert events[0].tracked_object_ids == [17]
    assert events[0].metadata.get("reason") == "track_departed"


# ---------------------------------------------------------------------------
# Test 16: Disabled Zone Ignored
# ---------------------------------------------------------------------------

def test_disabled_zone_ignored():
    disabled_zone = RestrictedZone(
        zone_id="zone_disabled",
        name="Disabled Zone",
        zone_type=ZoneType.RESTRICTED,
        polygon=SAMPLE_POLYGON,
        enabled=False,
    )
    engine = ZoneEngine(zones=[disabled_zone], confirmation_frames=1)
    worker = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))

    m, events = engine.process_frame([worker], timestamp=1.0, frame_id=1)
    assert len(m) == 0
    assert len(events) == 0


# ---------------------------------------------------------------------------
# Test 17: Invalid/Degenerate Polygon Handled Safely
# ---------------------------------------------------------------------------

def test_invalid_degenerate_polygon_handled_safely():
    # Attempting to create polygon with fewer than 3 vertices raises ValueError in schema
    with pytest.raises(ValueError):
        RestrictedZone(
            zone_id="bad_zone",
            name="Bad Polygon",
            polygon=[[10.0, 10.0], [20.0, 20.0]],
        )

    # Engine handles adding/parsing gracefully without crashing
    engine = ZoneEngine(zones=[], confirmation_frames=1)
    worker = _create_tracked_worker(17, BoundingBox(x1=10.0, y1=10.0, x2=30.0, y2=30.0))
    m, events = engine.process_frame([worker], timestamp=1.0, frame_id=1)
    assert len(m) == 0
    assert len(events) == 0


# ---------------------------------------------------------------------------
# Test 18: Person Foot-Point vs Non-Person Centroid Calculation
# ---------------------------------------------------------------------------

def test_foot_point_and_centroid_calculation():
    bbox = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=400.0)

    # Person/worker: Foot point is bottom-center ((100+200)/2, 400) = (150, 400)
    person_track = _create_tracked_worker(1, bbox, class_name="person")
    foot_pt = compute_contact_point(person_track)
    assert foot_pt == (150.0, 400.0)

    worker_track = _create_tracked_worker(2, bbox, class_name="worker")
    assert compute_contact_point(worker_track) == (150.0, 400.0)

    # Non-person (forklift, machinery): Centroid is center ((100+200)/2, (100+400)/2) = (150, 250)
    forklift_track = _create_tracked_worker(3, bbox, class_name="forklift")
    centroid_pt = compute_contact_point(forklift_track)
    assert centroid_pt == (150.0, 250.0)


# ---------------------------------------------------------------------------
# Test 19: Zone Engine BaseEventDetector & BaseZoneMonitor Interface
# ---------------------------------------------------------------------------

def test_zone_engine_interfaces(sample_zone):
    engine = ZoneEngine(zones=[sample_zone], confirmation_frames=1)
    worker = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))
    tracks = FrameTracks(frame_id=1, timestamp=10.0, active_tracks=[worker])

    # BaseZoneMonitor interface: check_intrusion(tracks)
    events_monitor = engine.check_intrusion(tracks)
    assert len(events_monitor) == 1
    assert events_monitor[0].event_type == EventType.ZONE_ENTRY

    engine.reset_state()

    # BaseEventDetector interface: evaluate(tracks, context)
    events_detector = engine.evaluate(tracks, context={"timestamp": 10.0, "frame_id": 1})
    assert len(events_detector) == 1
    assert events_detector[0].event_type == EventType.ZONE_ENTRY


# ---------------------------------------------------------------------------
# Test 20: Visualizer Rendering With Zones
# ---------------------------------------------------------------------------

def test_visualizer_with_zones(sample_zone):
    vis = TrackingVisualizer()
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    worker = _create_tracked_worker(17, BoundingBox(x1=180.0, y1=100.0, x2=220.0, y2=200.0))
    tracks = FrameTracks(frame_id=1, timestamp=1.0, active_tracks=[worker])

    membership = ZoneMembership(
        track_id=17,
        zone_id=sample_zone.zone_id,
        zone_name=sample_zone.name,
        is_inside=True,
        status=ZoneMembershipStatus.INSIDE,
        contact_point=(200.0, 200.0),
        dwell_seconds=3.5,
        entry_timestamp=1.0,
    )

    annotated = vis.draw_tracks(
        image=frame,
        tracks=tracks,
        zones=[sample_zone],
        zone_memberships=[membership],
        show_zones=True,
    )

    assert annotated is not None
    assert annotated.shape == frame.shape
    # Frame was modified (not all zeros)
    assert np.any(annotated > 0)
