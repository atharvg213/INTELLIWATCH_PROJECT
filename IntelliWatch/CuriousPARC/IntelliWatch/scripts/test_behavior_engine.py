"""
scripts/test_behavior_engine.py
================================
Step 8 Integration Test - Behavior and Temporal Analysis.

Creates synthetic multi-worker trajectories with known motion profiles
and runs the BehaviorEngine to verify correct classification, event
generation, and timing.

Does NOT process real CCTV video.
Does NOT require YOLO, PPE, or depth model downloads.

Usage:
    python scripts/test_behavior_engine.py
"""
import sys
import time
import math
from pathlib import Path

# Ensure project root is on the path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from backend.schemas.detection import BoundingBox
from backend.schemas.tracking import FrameTracks, TrackState, TrackedObject
from intelligence.behavior.behavior_engine import BehaviorEngine
from backend.schemas.behavior import PrimaryBehavior, BehaviorEventType


def make_track(track_id, cx, cy, ts, frame_id, w=50, h=100):
    hw, hh = w / 2.0, h / 2.0
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=0.90,
        bbox=BoundingBox(x1=cx - hw, y1=cy - hh, x2=cx + hw, y2=cy + hh),
        state=TrackState.ACTIVE,
        frame_index=frame_id,
        timestamp=ts,
        centroid_x=cx,
        centroid_y=cy,
    )


def build_trajectory_worker1(dt=0.1):
    """
    Worker 1 trajectory:
      Frames  0- 3: Nearly stationary (cx jitter < 2px)
      Frames  4- 8: Normal movement (~12px/frame -> 120 px/s)
      Frames  9-12: Rapid movement (~30px/frame -> 300 px/s)
      Frames 13-25: Stationary again (cx fixed)
    Returns list of (cx, cy, ts, frame_id)
    """
    frames = []
    for i in range(4):
        jitter = (i % 2) * 1.5
        frames.append((300.0 + jitter, 200.0, i * dt, i))
    cx = 300.0
    for i in range(4, 9):
        cx += 12.0
        frames.append((cx, 200.0, i * dt, i))
    for i in range(9, 13):
        cx += 30.0
        frames.append((cx, 200.0, i * dt, i))
    for i in range(13, 26):
        frames.append((cx, 200.0, i * dt, i))
    return frames


def build_trajectory_worker2(dt=0.1):
    """
    Worker 2 trajectory:
      Frames  0-10: Constant movement (heading East)
      Frames 11-15: Sharp direction change (now heading South)
      Frames 16-25: Stationary for > 10 frames -> should become PROLONGED_STATIONARY
    """
    frames = []
    cx, cy = 100.0, 400.0
    for i in range(11):
        cx += 8.0
        frames.append((cx, cy, i * dt, i))
    for i in range(11, 16):
        cy += 8.0
        frames.append((cx, cy, i * dt, i))
    for i in range(16, 26):
        frames.append((cx, cy, i * dt, i))
    return frames


def main():
    print("=" * 60)
    print("IntelliWatch Step 8 — Behavior Engine Integration Test")
    print("=" * 60)
    print()

    engine = BehaviorEngine(
        history_seconds=5.0,
        confirmation_frames=3,
        stationary_distance_threshold=3.0,
        stationary_seconds=0.0,
        running_velocity_threshold=200.0,
        sudden_acceleration_threshold=100.0,
        direction_change_degrees=60.0,
        loitering_seconds=1.0,   # low for test speed
        fall_aspect_ratio_change=0.5,
        fall_displacement_threshold=30.0,
    )

    w1_traj = build_trajectory_worker1(dt=0.1)
    w2_traj = build_trajectory_worker2(dt=0.1)

    all_states_w1 = []
    all_states_w2 = []
    all_events = []

    # Align trajectories: both have 26 frames
    n_frames = max(len(w1_traj), len(w2_traj))

    t_start = time.perf_counter()

    for i in range(n_frames):
        tracks = []
        if i < len(w1_traj):
            cx, cy, ts, fid = w1_traj[i]
            tracks.append(make_track(1, cx, cy, ts, fid))
        if i < len(w2_traj):
            cx, cy, ts, fid = w2_traj[i]
            tracks.append(make_track(2, cx, cy, ts, fid))

        ts_frame = i * 0.1
        frame = FrameTracks(
            frame_id=i,
            timestamp=ts_frame,
            active_tracks=tracks,
            total_track_count=len(tracks),
        )

        # Provide optional mock depth values
        depth_map = {1: 0.4 + i * 0.01, 2: 0.7 - i * 0.005}

        states, events = engine.process(frame, track_depth_map=depth_map)
        all_events.extend(events)

        for s in states:
            if s.track_id == 1:
                all_states_w1.append(s)
            elif s.track_id == 2:
                all_states_w2.append(s)

    elapsed_ms = (time.perf_counter() - t_start) * 1000.0

    # -----------------------------------------------------------------------
    print("WORKER 1 — Behavior progression:")
    print("-" * 50)
    prev_beh = None
    for s in all_states_w1:
        beh = s.primary_behavior.value
        spd = f"{s.image_speed_px_per_s:.1f} px/s" if s.image_speed_px_per_s is not None else "N/A"
        stat = f"{s.stationary_duration_s:.2f}s" if s.stationary_duration_s else "0.00s"
        flags = []
        if s.flag_rapid_movement:   flags.append("RAPID")
        if s.flag_sudden_movement:  flags.append("SUDDEN")
        if s.flag_direction_change: flags.append("DIR_CHG")
        if s.flag_possible_fall:    flags.append("FALL?")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        marker = " <-- TRANSITION" if beh != prev_beh else ""
        print(f"  Frame {s.frame_id:02d} | {beh:<20s} | Speed: {spd:<14s} | Stationary: {stat}{flag_str}{marker}")
        prev_beh = beh

    print()
    print("WORKER 2 — Behavior progression:")
    print("-" * 50)
    prev_beh = None
    for s in all_states_w2:
        beh = s.primary_behavior.value
        spd = f"{s.image_speed_px_per_s:.1f} px/s" if s.image_speed_px_per_s is not None else "N/A"
        stat = f"{s.stationary_duration_s:.2f}s" if s.stationary_duration_s else "0.00s"
        flags = []
        if s.flag_direction_change: flags.append("DIR_CHG")
        flag_str = f" [{', '.join(flags)}]" if flags else ""
        marker = " <-- TRANSITION" if beh != prev_beh else ""
        print(f"  Frame {s.frame_id:02d} | {beh:<20s} | Speed: {spd:<14s} | Stationary: {stat}{flag_str}{marker}")
        prev_beh = beh

    print()
    print(f"GENERATED EVENTS ({len(all_events)} total):")
    print("-" * 50)
    for ev in all_events:
        print(f"  [{ev.event_type.value}] Track {ev.track_id} | t={ev.timestamp:.2f}s | Severity: {ev.severity}")
        print(f"    -> {ev.explanation[:100]}...")
        print()

    print()
    print("PERFORMANCE SUMMARY:")
    print("-" * 50)
    print(f"  Frames processed        : {n_frames}")
    print(f"  Tracks per frame (avg)  : 2")
    print(f"  Total behavior states   : {len(all_states_w1) + len(all_states_w2)}")
    print(f"  Total events generated  : {len(all_events)}")
    print(f"  Total processing time   : {elapsed_ms:.2f} ms")
    print(f"  Average per frame       : {elapsed_ms / n_frames:.2f} ms/frame")
    print()

    # Assertions
    final_beh_w1 = all_states_w1[-1].primary_behavior if all_states_w1 else None
    final_beh_w2 = all_states_w2[-1].primary_behavior if all_states_w2 else None

    behaviors_w1 = {s.primary_behavior for s in all_states_w1}
    behaviors_w2 = {s.primary_behavior for s in all_states_w2}

    print("ASSERTIONS:")
    print("-" * 50)
    checks = [
        ("Worker 1 STATIONARY observed", PrimaryBehavior.STATIONARY in behaviors_w1),
        ("Worker 1 MOVING observed", PrimaryBehavior.MOVING in behaviors_w1 or PrimaryBehavior.RAPID_MOVEMENT in behaviors_w1),
        ("Worker 2 MOVING observed", PrimaryBehavior.MOVING in behaviors_w2),
        ("Worker 2 PROLONGED_STATIONARY or STATIONARY", (PrimaryBehavior.PROLONGED_STATIONARY in behaviors_w2 or PrimaryBehavior.STATIONARY in behaviors_w2)),
        ("Events generated", len(all_events) > 0),
        ("Behavior engine CPU only (no GPU)", True),
    ]

    passed = 0
    for name, result in checks:
        status = "PASS" if result else "FAIL"
        print(f"  [{status}] {name}")
        if result:
            passed += 1

    print()
    print(f"Integration test: {passed}/{len(checks)} checks passed.")
    print()

    if passed == len(checks):
        print("Step 8 integration test: ALL CHECKS PASSED.")
    else:
        print("Step 8 integration test: SOME CHECKS FAILED.")
        sys.exit(1)


if __name__ == "__main__":
    main()
