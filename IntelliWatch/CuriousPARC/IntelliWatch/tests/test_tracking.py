"""
Unit tests for the Multi-Object Tracking (Step 4) system.
Uses deterministic mock detections to verify:
  1. Tracker initialization
  2. Empty detections handling
  3. Track creation
  4. Persistent identity across consecutive frames
  5. Multiple objects tracking with distinct IDs
  6. Trajectory history updating
  7. Trajectory history length limits (TRACK_HISTORY_LENGTH)
  8. Coordinate preservation in original video resolution
  9. Motion information (image-space displacement and velocity)
  10. Track lifecycle handling across temporary detection loss
  11. Tracker reset
  12. Tracking visualizer rendering
"""
import sys
from pathlib import Path
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from backend.schemas.tracking import FrameTracks, TrackPoint, TrackState, TrackedObject
from configs.settings import get_settings
from vision.tracking.base import BaseTracker
from vision.tracking.bytetrack_tracker import ByteTrackTracker
from vision.tracking.visualizer import TrackingVisualizer


def make_detection(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    class_id: int = 0,
    class_name: str = "person",
    confidence: float = 0.90,
) -> DetectionResult:
    """Helper to construct a deterministic DetectionResult."""
    return DetectionResult(
        class_id=class_id,
        class_name=class_name,
        confidence=confidence,
        bbox=BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2),
    )


# 1. Tracker Initialization
def test_tracker_initialization():
    tracker = ByteTrackTracker(
        track_high_thresh=0.3,
        track_low_thresh=0.1,
        match_thresh=0.7,
        track_buffer=25,
        track_history_length=15,
        frame_rate=25,
    )
    assert isinstance(tracker, BaseTracker)
    assert tracker.track_high_thresh == 0.3
    assert tracker.track_low_thresh == 0.1
    assert tracker.match_thresh == 0.7
    assert tracker.track_buffer == 25
    assert tracker.track_history_length == 15
    assert tracker.frame_rate == 25


# 2. Empty Detections
def test_empty_detections_handling():
    tracker = ByteTrackTracker()
    empty_frame = FrameDetections(
        frame_id=0,
        timestamp=0.0,
        detections=[],
        frame_width=1920,
        frame_height=1080,
    )
    result = tracker.update(empty_frame)

    assert isinstance(result, FrameTracks)
    assert result.frame_id == 0
    assert result.timestamp == 0.0
    assert len(result.active_tracks) == 0
    assert result.total_track_count == 0


# 3. Track Creation
def test_track_creation():
    tracker = ByteTrackTracker()
    det = make_detection(x1=100.0, y1=150.0, x2=200.0, y2=350.0, class_id=0, class_name="person", confidence=0.88)
    frame = FrameDetections(frame_id=1, timestamp=0.033, detections=[det], frame_width=1280, frame_height=720)

    result = tracker.update(frame)

    assert len(result.active_tracks) == 1
    track = result.active_tracks[0]
    assert track.track_id >= 1
    assert track.class_id == 0
    assert track.class_name == "person"
    assert track.confidence == pytest.approx(0.88, abs=1e-2)
    assert track.centroid_x == pytest.approx(150.0, abs=1.0)
    assert track.centroid_y == pytest.approx(250.0, abs=1.0)
    assert track.state in (TrackState.NEW, TrackState.ACTIVE)
    assert len(track.trajectory) == 1


# 4. Persistent Identity Across Consecutive Frames
def test_persistent_identity():
    tracker = ByteTrackTracker(track_buffer=30)

    # Frame 1: Object at (100, 100) -> (200, 300)
    f1 = FrameDetections(
        frame_id=1,
        timestamp=0.033,
        detections=[make_detection(100.0, 100.0, 200.0, 300.0)],
        frame_width=1280,
        frame_height=720,
    )
    r1 = tracker.update(f1)
    assert len(r1.active_tracks) == 1
    initial_id = r1.active_tracks[0].track_id

    # Frame 2: Object moves slightly to (105, 102) -> (205, 302)
    f2 = FrameDetections(
        frame_id=2,
        timestamp=0.066,
        detections=[make_detection(105.0, 102.0, 205.0, 302.0)],
        frame_width=1280,
        frame_height=720,
    )
    r2 = tracker.update(f2)
    assert len(r2.active_tracks) == 1
    assert r2.active_tracks[0].track_id == initial_id
    assert r2.active_tracks[0].state == TrackState.ACTIVE

    # Frame 3: Object moves slightly to (110, 104) -> (210, 304)
    f3 = FrameDetections(
        frame_id=3,
        timestamp=0.099,
        detections=[make_detection(110.0, 104.0, 210.0, 304.0)],
        frame_width=1280,
        frame_height=720,
    )
    r3 = tracker.update(f3)
    assert len(r3.active_tracks) == 1
    assert r3.active_tracks[0].track_id == initial_id


# 5. Multiple Objects with Distinct IDs and Classes
def test_multiple_objects_tracking():
    tracker = ByteTrackTracker()

    dets = [
        make_detection(50.0, 50.0, 150.0, 250.0, class_id=0, class_name="person"),
        make_detection(300.0, 100.0, 400.0, 300.0, class_id=0, class_name="person"),
        make_detection(700.0, 400.0, 950.0, 600.0, class_id=2, class_name="car"),
    ]
    frame = FrameDetections(frame_id=1, timestamp=0.033, detections=dets, frame_width=1920, frame_height=1080)
    result = tracker.update(frame)

    assert len(result.active_tracks) == 3
    track_ids = {t.track_id for t in result.active_tracks}
    assert len(track_ids) == 3  # All IDs must be unique

    classes = {t.class_name for t in result.active_tracks}
    assert "person" in classes
    assert "car" in classes


# 6. Trajectory History Updating
def test_trajectory_history_updating():
    tracker = ByteTrackTracker()

    # Move object across 4 frames
    positions = [
        (100.0, 100.0, 150.0, 200.0),  # cx=125, cy=150
        (106.0, 104.0, 156.0, 204.0),  # cx=131, cy=154
        (112.0, 109.0, 162.0, 209.0),  # cx=137, cy=159
        (119.0, 115.0, 169.0, 215.0),  # cx=144, cy=165
    ]

    last_track = None
    for i, (x1, y1, x2, y2) in enumerate(positions, start=1):
        f = FrameDetections(
            frame_id=i,
            timestamp=i * 0.033,
            detections=[make_detection(x1, y1, x2, y2)],
            frame_width=640,
            frame_height=480,
        )
        tracks = tracker.update(f)
        assert len(tracks.active_tracks) == 1
        last_track = tracks.active_tracks[0]
        assert len(last_track.trajectory) == i

    assert len(last_track.trajectory) == 4
    for pt in last_track.trajectory:
        assert isinstance(pt, TrackPoint)
        assert pt.frame_id in {1, 2, 3, 4}


# 7. Trajectory History Limit
def test_trajectory_history_limit():
    max_history = 5
    tracker = ByteTrackTracker(track_history_length=max_history)

    # Feed 12 frames
    for i in range(1, 13):
        x1 = 100.0 + i * 2.0
        y1 = 100.0 + i * 2.0
        f = FrameDetections(
            frame_id=i,
            timestamp=i * 0.033,
            detections=[make_detection(x1, y1, x1 + 50.0, y1 + 100.0)],
            frame_width=640,
            frame_height=480,
        )
        tracks = tracker.update(f)
        active = tracks.active_tracks[0]
        assert len(active.trajectory) <= max_history

    assert len(active.trajectory) == max_history
    # Most recent point should correspond to frame 12
    assert active.trajectory[-1].frame_id == 12
    # Oldest retained point should correspond to frame 8 (12 - 5 + 1)
    assert active.trajectory[0].frame_id == 8


# 8. Coordinate Preservation in Original Video Pixel Space
def test_coordinate_preservation():
    tracker = ByteTrackTracker()

    orig_x1, orig_y1, orig_x2, orig_y2 = 450.25, 230.75, 580.50, 490.25
    f = FrameDetections(
        frame_id=1,
        timestamp=0.033,
        detections=[make_detection(orig_x1, orig_y1, orig_x2, orig_y2)],
        frame_width=1920,
        frame_height=1080,
    )
    result = tracker.update(f)
    assert len(result.active_tracks) == 1
    t = result.active_tracks[0]

    # Tracker coordinates must match original video coordinates
    assert t.bbox.x1 == pytest.approx(orig_x1, abs=1.5)
    assert t.bbox.y1 == pytest.approx(orig_y1, abs=1.5)
    assert t.bbox.x2 == pytest.approx(orig_x2, abs=1.5)
    assert t.bbox.y2 == pytest.approx(orig_y2, abs=1.5)

    expected_cx = (orig_x1 + orig_x2) / 2.0
    expected_cy = (orig_y1 + orig_y2) / 2.0
    assert t.centroid_x == pytest.approx(expected_cx, abs=1.5)
    assert t.centroid_y == pytest.approx(expected_cy, abs=1.5)


# 9. Motion Information Calculation
def test_motion_information():
    tracker = ByteTrackTracker()

    # Frame 1: centroid at (100, 100)
    f1 = FrameDetections(
        frame_id=1,
        timestamp=1.0,
        detections=[make_detection(75.0, 75.0, 125.0, 125.0)],
        frame_width=1280,
        frame_height=720,
    )
    r1 = tracker.update(f1)
    t1 = r1.active_tracks[0]
    assert t1.velocity_x == 0.0
    assert t1.velocity_y == 0.0
    assert t1.speed_pixels_per_second == 0.0

    # Frame 2: centroid shifts by +10 in X and +5 in Y after 0.1s
    f2 = FrameDetections(
        frame_id=2,
        timestamp=1.1,
        detections=[make_detection(85.0, 80.0, 135.0, 130.0)],
        frame_width=1280,
        frame_height=720,
    )
    r2 = tracker.update(f2)
    t2 = r2.active_tracks[0]

    assert t2.velocity_x == pytest.approx(10.0, abs=1.5)
    assert t2.velocity_y == pytest.approx(5.0, abs=1.5)
    expected_speed = np.hypot(10.0, 5.0) / 0.1
    assert t2.speed_pixels_per_second == pytest.approx(expected_speed, abs=20.0)


# 10. Temporary Detection Loss and Identity Recovery
def test_temporary_loss_and_recovery():
    tracker = ByteTrackTracker(track_buffer=30)

    # Frame 1: Person visible
    f1 = FrameDetections(
        frame_id=1,
        timestamp=0.033,
        detections=[make_detection(100.0, 100.0, 180.0, 260.0)],
    )
    r1 = tracker.update(f1)
    tid = r1.active_tracks[0].track_id

    # Frame 2: Occlusion / detection missed (empty detections)
    f2 = FrameDetections(frame_id=2, timestamp=0.066, detections=[])
    r2 = tracker.update(f2)
    assert len(r2.active_tracks) == 0
    # State internally tracked as LOST
    assert tracker.get_track_state(tid) == TrackState.LOST

    # Frame 3: Person reappears near expected position
    f3 = FrameDetections(
        frame_id=3,
        timestamp=0.099,
        detections=[make_detection(104.0, 102.0, 184.0, 262.0)],
    )
    r3 = tracker.update(f3)
    assert len(r3.active_tracks) == 1
    # Identity must be preserved across the 1-frame occlusion
    assert r3.active_tracks[0].track_id == tid
    assert r3.active_tracks[0].state == TrackState.ACTIVE


# 11. Tracker Reset
def test_tracker_reset():
    tracker = ByteTrackTracker()
    f1 = FrameDetections(
        frame_id=1,
        timestamp=0.033,
        detections=[make_detection(100.0, 100.0, 200.0, 200.0)],
    )
    r1 = tracker.update(f1)
    assert len(r1.active_tracks) == 1

    tracker.reset()
    assert len(tracker._trajectories) == 0
    assert len(tracker._track_classes) == 0
    assert len(tracker._total_tracks_created) == 0


# 12. Tracking Visualizer
def test_tracking_visualizer():
    viz = TrackingVisualizer()
    canvas = np.zeros((480, 640, 3), dtype=np.uint8)

    bbox = BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=200.0)
    traj = [
        TrackPoint(frame_id=1, timestamp=0.033, cx=95.0, cy=120.0),
        TrackPoint(frame_id=2, timestamp=0.066, cx=100.0, cy=125.0),
    ]
    tracked_obj = TrackedObject(
        track_id=17,
        class_id=0,
        class_name="person",
        confidence=0.91,
        bbox=bbox,
        trajectory=traj,
        speed_pixels_per_second=45.0,
    )
    frame_tracks = FrameTracks(
        frame_id=2,
        timestamp=0.066,
        active_tracks=[tracked_obj],
        total_track_count=1,
    )

    annotated = viz.draw_tracks(canvas, frame_tracks)

    assert annotated is not None
    assert annotated.shape == canvas.shape
    assert annotated.dtype == np.uint8
    # Visual elements were drawn: canvas should no longer be completely black
    assert np.any(annotated > 0)
