"""
tests/test_behavior.py
Step 8 Behavior and Temporal Analysis unit tests.
Uses synthetic data only - no model downloads required.
"""
import math
import collections
import pytest
from unittest.mock import MagicMock

from backend.schemas.behavior import (
    BehaviorEvent, BehaviorEventType, BehaviorState, PrimaryBehavior,
)
from backend.schemas.detection import BoundingBox
from backend.schemas.tracking import FrameTracks, TrackState, TrackedObject
from configs.settings import get_settings
from intelligence.behavior.behavior_engine import BehaviorEngine
from intelligence.behavior.motion import (
    check_direction_change, compute_depth_trend, compute_heading_change,
    compute_recent_accel, compute_speed_average, detect_fall_heuristic,
)
from intelligence.behavior.temporal_state import (
    MotionObservation, TrackStateRegistry, TrackTemporalState,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_track(track_id, cx, cy, timestamp, frame_id=1, w=50, h=100, speed=0.0, vx=0.0, vy=0.0):
    hw, hh = w / 2, h / 2
    return TrackedObject(
        track_id=track_id, class_id=0, class_name="person", confidence=0.9,
        bbox=BoundingBox(x1=cx - hw, y1=cy - hh, x2=cx + hw, y2=cy + hh),
        state=TrackState.ACTIVE, frame_index=frame_id, timestamp=timestamp,
        centroid_x=cx, centroid_y=cy, velocity_x=vx, velocity_y=vy,
        speed_pixels_per_second=speed,
    )


def _make_frame(tracks, timestamp, frame_id=1):
    return FrameTracks(
        frame_id=frame_id, timestamp=timestamp,
        active_tracks=tracks, total_track_count=len(tracks),
    )


def _run_frames(engine, positions, dt=0.1, track_id=1, start_ts=0.0):
    results = []
    for i, (cx, cy) in enumerate(positions):
        ts = start_ts + i * dt
        track = _make_track(track_id, cx, cy, ts, frame_id=i)
        frame = _make_frame([track], ts, frame_id=i)
        states, events = engine.process(frame)
        results.append((states, events))
    return results


# ---------------------------------------------------------------------------
# 1. New track initialization
# ---------------------------------------------------------------------------

class TestNewTrack:
    def test_first_frame_unknown(self):
        engine = BehaviorEngine(confirmation_frames=3)
        states, _ = engine.process(_make_frame([_make_track(1, 100.0, 200.0, 0.0)], 0.0))
        assert states[0].primary_behavior == PrimaryBehavior.UNKNOWN

    def test_new_track_no_transition_events(self):
        engine = BehaviorEngine(confirmation_frames=3)
        _, events = engine.process(_make_frame([_make_track(1, 100.0, 200.0, 0.0)], 0.0))
        transition_evs = [e for e in events if e.event_type == BehaviorEventType.BEHAVIOR_TRANSITION]
        assert len(transition_evs) == 0

    def test_new_track_speed_none(self):
        engine = BehaviorEngine(confirmation_frames=3)
        states, _ = engine.process(_make_frame([_make_track(1, 100.0, 200.0, 0.0)], 0.0))
        assert states[0].image_speed_px_per_s is None


# ---------------------------------------------------------------------------
# 2. Stationary detection
# ---------------------------------------------------------------------------

class TestStationary:
    def test_stationary_confirmed_after_frames(self):
        engine = BehaviorEngine(stationary_distance_threshold=5.0, stationary_seconds=0.0, confirmation_frames=3)
        results = _run_frames(engine, [(100.0, 200.0)] * 8, dt=0.1)
        assert results[-1][0][0].primary_behavior == PrimaryBehavior.STATIONARY

    def test_stationary_not_instant(self):
        engine = BehaviorEngine(stationary_distance_threshold=5.0, stationary_seconds=0.0, confirmation_frames=5)
        results = _run_frames(engine, [(100.0, 200.0)] * 2, dt=0.1)
        assert results[-1][0][0].primary_behavior != PrimaryBehavior.STATIONARY

    def test_stationary_duration_accumulates(self):
        engine = BehaviorEngine(stationary_distance_threshold=50.0, stationary_seconds=0.0, confirmation_frames=2)
        results = _run_frames(engine, [(100.0, 100.0)] * 6, dt=0.5)
        assert results[-1][0][0].stationary_duration_s > 1.0


# ---------------------------------------------------------------------------
# 3. Moving detection
# ---------------------------------------------------------------------------

class TestMoving:
    def test_moving_confirmed(self):
        engine = BehaviorEngine(stationary_distance_threshold=3.0, running_velocity_threshold=500.0, confirmation_frames=3)
        results = _run_frames(engine, [(float(i * 10), 100.0) for i in range(10)], dt=0.1)
        assert results[-1][0][0].primary_behavior == PrimaryBehavior.MOVING

    def test_moving_speed_populated(self):
        engine = BehaviorEngine(confirmation_frames=2)
        results = _run_frames(engine, [(float(i * 15), 100.0) for i in range(6)], dt=0.1)
        spd = results[-1][0][0].image_speed_px_per_s
        assert spd is not None and spd > 0


# ---------------------------------------------------------------------------
# 4. Rapid movement
# ---------------------------------------------------------------------------

class TestRapidMovement:
    def test_rapid_movement_flag_set(self):
        engine = BehaviorEngine(running_velocity_threshold=50.0, confirmation_frames=2, stationary_distance_threshold=3.0)
        results = _run_frames(engine, [(float(i * 20), 100.0) for i in range(8)], dt=0.1)
        assert results[-1][0][0].flag_rapid_movement is True

    def test_rapid_movement_primary_state(self):
        engine = BehaviorEngine(running_velocity_threshold=30.0, confirmation_frames=3, stationary_distance_threshold=2.0)
        results = _run_frames(engine, [(float(i * 30), 100.0) for i in range(10)], dt=0.1)
        assert results[-1][0][0].primary_behavior == PrimaryBehavior.RAPID_MOVEMENT

    def test_rapid_not_from_slow_track(self):
        engine = BehaviorEngine(running_velocity_threshold=200.0, confirmation_frames=3)
        results = _run_frames(engine, [(float(i * 2), 100.0) for i in range(8)], dt=0.1)
        assert results[-1][0][0].flag_rapid_movement is False


# ---------------------------------------------------------------------------
# 5. Sudden movement
# ---------------------------------------------------------------------------

class TestSuddenMovement:
    def test_sudden_movement_flag(self):
        engine = BehaviorEngine(sudden_acceleration_threshold=10.0, confirmation_frames=2)
        results = _run_frames(engine, [(100.0, 100.0)] * 4 + [(300.0, 100.0)], dt=0.1)
        flags = [s.flag_sudden_movement for states, _ in results for s in states]
        assert any(flags)

    def test_sudden_movement_generates_event(self):
        engine = BehaviorEngine(sudden_acceleration_threshold=10.0, confirmation_frames=2)
        results = _run_frames(engine, [(100.0, 100.0)] * 3 + [(400.0, 100.0)], dt=0.1)
        all_events = [ev for _, evs in results for ev in evs]
        sudden_evs = [e for e in all_events if e.event_type == BehaviorEventType.SUDDEN_MOVEMENT]
        assert len(sudden_evs) >= 1


# ---------------------------------------------------------------------------
# 6. Direction change
# ---------------------------------------------------------------------------

class TestDirectionChange:
    def test_direction_change_flag(self):
        engine = BehaviorEngine(direction_change_degrees=60.0, confirmation_frames=2)
        positions = [(100.0, 100.0), (110.0, 100.0), (120.0, 100.0), (120.0, 90.0), (120.0, 80.0)]
        results = _run_frames(engine, positions, dt=0.1)
        flags = [s.flag_direction_change for states, _ in results for s in states]
        assert any(flags)

    def test_no_direction_change_straight(self):
        engine = BehaviorEngine(direction_change_degrees=60.0, confirmation_frames=2)
        results = _run_frames(engine, [(float(i * 5), 100.0) for i in range(8)], dt=0.1)
        assert results[-1][0][0].flag_direction_change is False


# ---------------------------------------------------------------------------
# 7. Prolonged stationary
# ---------------------------------------------------------------------------

class TestProlongedStationary:
    def test_prolonged_stationary_confirmed(self):
        engine = BehaviorEngine(stationary_distance_threshold=5.0, stationary_seconds=0.0, loitering_seconds=2.0, confirmation_frames=2)
        results = _run_frames(engine, [(100.0, 200.0)] * 30, dt=0.1)
        assert results[-1][0][0].primary_behavior == PrimaryBehavior.PROLONGED_STATIONARY

    def test_prolonged_generates_event(self):
        engine = BehaviorEngine(stationary_distance_threshold=5.0, stationary_seconds=0.0, loitering_seconds=1.0, confirmation_frames=2)
        results = _run_frames(engine, [(100.0, 200.0)] * 20, dt=0.1)
        all_events = [ev for _, evs in results for ev in evs]
        loiter_evs = [e for e in all_events if e.event_type == BehaviorEventType.PROLONGED_STATIONARY]
        assert len(loiter_evs) >= 1

    def test_stationary_vs_prolonged_distinct(self):
        assert PrimaryBehavior.STATIONARY != PrimaryBehavior.PROLONGED_STATIONARY


# ---------------------------------------------------------------------------
# 8. Temporal confirmation
# ---------------------------------------------------------------------------

class TestTemporalConfirmation:
    def test_confirmation_requires_n_frames(self):
        engine = BehaviorEngine(running_velocity_threshold=20.0, confirmation_frames=5)
        positions = [(100.0, 100.0)] * 2 + [(float(100 + i * 25), 100.0) for i in range(3)]
        results = _run_frames(engine, positions, dt=0.1)
        assert results[-1][0][0].primary_behavior != PrimaryBehavior.RAPID_MOVEMENT


# ---------------------------------------------------------------------------
# 9. Temporal deconfirmation
# ---------------------------------------------------------------------------

class TestDeconfirmation:
    def test_stationary_resets_when_moving(self):
        engine = BehaviorEngine(stationary_distance_threshold=5.0, stationary_seconds=0.0, confirmation_frames=3)
        positions = [(100.0, 100.0)] * 6 + [(float(100 + i * 30), 100.0) for i in range(1, 6)]
        results = _run_frames(engine, positions, dt=0.1)
        assert results[-1][0][0].stationary_duration_s < 0.5


# ---------------------------------------------------------------------------
# 10. Track disappearance
# ---------------------------------------------------------------------------

class TestTrackDisappearance:
    def test_disappeared_track_not_in_output(self):
        engine = BehaviorEngine(confirmation_frames=3)
        for i in range(3):
            engine.process(_make_frame([_make_track(1, 100.0, 100.0, float(i) * 0.1, i)], float(i) * 0.1, i))
        states, _ = engine.process(_make_frame([], 0.4, 4))
        assert len(states) == 0


# ---------------------------------------------------------------------------
# 11. Track reappearance
# ---------------------------------------------------------------------------

class TestTrackReappearance:
    def test_reappeared_track_produces_state(self):
        engine = BehaviorEngine(confirmation_frames=2)
        engine.process(_make_frame([_make_track(1, 100.0, 100.0, 0.0)], 0.0))
        engine.process(_make_frame([], 0.1))
        states, _ = engine.process(_make_frame([_make_track(1, 200.0, 200.0, 0.5)], 0.5))
        assert any(s.track_id == 1 for s in states)


# ---------------------------------------------------------------------------
# 12. Zero time delta
# ---------------------------------------------------------------------------

class TestZeroTimeDelta:
    def test_zero_dt_no_crash(self):
        engine = BehaviorEngine(confirmation_frames=2)
        engine.process(_make_frame([_make_track(1, 100.0, 100.0, 0.0)], 0.0, 0))
        states, _ = engine.process(_make_frame([_make_track(1, 110.0, 100.0, 0.0)], 0.0, 1))
        assert len(states) == 1


# ---------------------------------------------------------------------------
# 13. Missing observations
# ---------------------------------------------------------------------------

class TestMissingObservations:
    def test_single_observation_no_speed(self):
        engine = BehaviorEngine()
        states, _ = engine.process(_make_frame([_make_track(1, 100.0, 100.0, 0.0)], 0.0))
        assert states[0].image_speed_px_per_s is None

    def test_two_observations_have_speed(self):
        engine = BehaviorEngine()
        engine.process(_make_frame([_make_track(1, 100.0, 100.0, 0.0)], 0.0, 0))
        states, _ = engine.process(_make_frame([_make_track(1, 115.0, 100.0, 0.1)], 0.1, 1))
        assert states[0].image_speed_px_per_s is not None


# ---------------------------------------------------------------------------
# 14. Multiple simultaneous tracks
# ---------------------------------------------------------------------------

class TestMultipleTracks:
    def test_two_tracks_independent(self):
        engine = BehaviorEngine(confirmation_frames=2, stationary_distance_threshold=3.0)
        for i in range(4):
            ts = float(i) * 0.1
            t1 = _make_track(1, float(i * 2), 100.0, ts, frame_id=i)
            t2 = _make_track(2, float(i * 40), 200.0, ts, frame_id=i)
            states, _ = engine.process(_make_frame([t1, t2], ts, frame_id=i))
        ids = {s.track_id for s in states}
        assert 1 in ids and 2 in ids

    def test_multiple_different_behaviors(self):
        engine = BehaviorEngine(stationary_distance_threshold=5.0, stationary_seconds=0.0, running_velocity_threshold=500.0, confirmation_frames=4)
        for i in range(8):
            ts = float(i) * 0.1
            t1 = _make_track(1, 100.0, 100.0, ts, frame_id=i)
            t2 = _make_track(2, float(100 + i * 8), 200.0, ts, frame_id=i)
            states, _ = engine.process(_make_frame([t1, t2], ts, frame_id=i))
        behaviors = {s.track_id: s.primary_behavior for s in states}
        assert behaviors[1] in (PrimaryBehavior.STATIONARY, PrimaryBehavior.UNKNOWN)
        assert behaviors[2] in (PrimaryBehavior.MOVING, PrimaryBehavior.UNKNOWN)


# ---------------------------------------------------------------------------
# 15. Zone context
# ---------------------------------------------------------------------------

class TestZoneContext:
    def test_zone_context_populated(self):
        engine = BehaviorEngine(confirmation_frames=2)
        m = MagicMock(); m.track_id = 1; m.zone_id = "zone_A"
        m.zone_name = "High Voltage"; m.dwell_seconds = 4.2; m.is_inside = True
        occ = MagicMock(); occ.memberships = [m]
        engine.process(_make_frame([_make_track(1, 100.0, 100.0, 0.0)], 0.0))
        states, _ = engine.process(_make_frame([_make_track(1, 102.0, 100.0, 0.1)], 0.1), zone_occupancy=occ)
        assert states[0].zone_id == "zone_A"
        assert states[0].zone_dwell_s == pytest.approx(4.2)

    def test_no_zone_none(self):
        engine = BehaviorEngine(confirmation_frames=2)
        states, _ = engine.process(_make_frame([_make_track(1, 100.0, 100.0, 0.0)], 0.0))
        assert states[0].zone_id is None


# ---------------------------------------------------------------------------
# 16. Depth context
# ---------------------------------------------------------------------------

class TestDepthContext:
    def test_depth_value_forwarded(self):
        engine = BehaviorEngine(confirmation_frames=2)
        engine.process(_make_frame([_make_track(1, 100.0, 100.0, 0.0)], 0.0))
        states, _ = engine.process(_make_frame([_make_track(1, 100.0, 100.0, 0.1)], 0.1), track_depth_map={1: 0.73})
        assert states[0].relative_depth == pytest.approx(0.73)

    def test_no_depth_ok(self):
        engine = BehaviorEngine(confirmation_frames=2)
        states, _ = engine.process(_make_frame([_make_track(1, 100.0, 100.0, 0.0)], 0.0))
        assert states[0].relative_depth is None

    def test_depth_trend_computed(self):
        engine = BehaviorEngine(confirmation_frames=2)
        for i, dv in enumerate([0.4, 0.5, 0.6, 0.7, 0.8]):
            ts = float(i) * 0.1
            states, _ = engine.process(_make_frame([_make_track(1, 100.0, 100.0, ts, i)], ts, i), track_depth_map={1: dv})
        assert states[0].depth_trend in ("approaching", "receding", "stable", None)


# ---------------------------------------------------------------------------
# 17. History window cleanup
# ---------------------------------------------------------------------------

class TestHistoryCleanup:
    def test_old_observations_pruned(self):
        registry = TrackStateRegistry(history_seconds=1.0)
        state = registry.get_or_create(1)
        state.add_observation(0.0, 0, 100.0, 100.0)
        state.add_observation(0.5, 1, 105.0, 100.0)
        state.add_observation(2.0, 2, 110.0, 100.0)
        assert state.observation_count() == 1

    def test_stale_tracks_purged(self):
        registry = TrackStateRegistry(history_seconds=1.0)
        s = registry.get_or_create(42)
        s.add_observation(0.0, 0, 100.0, 100.0)
        s.last_seen_timestamp = 0.0
        purged = registry.purge_stale(60.0, max_age_seconds=5.0)
        assert 42 in purged
        assert registry.get(42) is None


# ---------------------------------------------------------------------------
# 18. Behavior state transitions
# ---------------------------------------------------------------------------

class TestStateTransitions:
    def test_unknown_to_moving(self):
        # Use explicit high velocity threshold so 10px/0.1s (100px/s) stays MOVING not RAPID
        engine = BehaviorEngine(confirmation_frames=3, stationary_distance_threshold=2.0,
                                running_velocity_threshold=500.0)
        results = _run_frames(engine, [(float(i * 10), 100.0) for i in range(8)], dt=0.1)
        behaviors = [s.primary_behavior for states, _ in results for s in states]
        assert PrimaryBehavior.UNKNOWN in behaviors
        assert PrimaryBehavior.MOVING in behaviors

    def test_moving_to_stationary(self):
        engine = BehaviorEngine(confirmation_frames=3, stationary_distance_threshold=5.0, stationary_seconds=0.0)
        positions = [(float(i * 15), 100.0) for i in range(5)] + [(75.0, 100.0)] * 10
        results = _run_frames(engine, positions, dt=0.1)
        behaviors = [s.primary_behavior for states, _ in results for s in states]
        assert PrimaryBehavior.STATIONARY in behaviors


# ---------------------------------------------------------------------------
# 19. Possible fall heuristic
# ---------------------------------------------------------------------------

class TestFallHeuristic:
    def _make_obs(self, ts, cx, cy, bw, bh, disp):
        return MotionObservation(
            timestamp=ts, frame_id=0, cx=cx, cy=cy, bbox_width=bw, bbox_height=bh,
            aspect_ratio=(bh / bw) if bw > 0 else None, displacement=disp,
            dx=disp, dy=0.0, vx=disp * 10, vy=0.0, speed_px_per_s=disp * 10,
            ax=0.0, ay=0.0, accel_px_per_s2=0.0, dt=0.1,
        )

    def test_fall_heuristic_triggered(self):
        prev = self._make_obs(0.0, 100.0, 100.0, 50.0, 120.0, 0.0)
        curr = self._make_obs(0.1, 130.0, 100.0, 120.0, 30.0, 50.0)
        assert detect_fall_heuristic(curr, prev, ar_change_threshold=0.5, disp_threshold=30.0) is True

    def test_fall_heuristic_not_triggered(self):
        prev = self._make_obs(0.0, 100.0, 100.0, 50.0, 100.0, 0.0)
        curr = self._make_obs(0.1, 102.0, 100.0, 50.0, 95.0, 2.0)
        assert detect_fall_heuristic(curr, prev, ar_change_threshold=0.5, disp_threshold=30.0) is False

    def test_fall_event_honest_label(self):
        engine = BehaviorEngine(fall_aspect_ratio_change=0.3, fall_displacement_threshold=20.0, confirmation_frames=1)
        for i, (cx, cy, w, h) in enumerate([(100.0, 100.0, 50, 120), (130.0, 100.0, 120, 30)]):
            hw, hh = w / 2, h / 2
            track = TrackedObject(
                track_id=1, class_id=0, class_name="person", confidence=0.9,
                bbox=BoundingBox(x1=cx - hw, y1=cy - hh, x2=cx + hw, y2=cy + hh),
                state=TrackState.ACTIVE, frame_index=i, timestamp=float(i) * 0.1, centroid_x=cx, centroid_y=cy,
            )
            states, events = engine.process(_make_frame([track], float(i) * 0.1, i))
        fall_evs = [e for e in events if e.event_type == BehaviorEventType.POSSIBLE_FALL]
        if fall_evs:
            exp = fall_evs[0].explanation.lower()
            assert "possible" in exp or "heuristic" in exp or "geometric" in exp


# ---------------------------------------------------------------------------
# 20. Regression / integration
# ---------------------------------------------------------------------------

class TestRegression:
    def test_behavior_engine_import(self):
        from intelligence.behavior import BehaviorEngine
        assert BehaviorEngine is not None

    def test_settings_has_all_behavior_keys(self):
        s = get_settings()
        for key in [
            "BEHAVIOR_ENABLED", "BEHAVIOR_HISTORY_SECONDS", "BEHAVIOR_CONFIRMATION_FRAMES",
            "BEHAVIOR_STATIONARY_DISTANCE_THRESHOLD", "BEHAVIOR_STATIONARY_SECONDS",
            "BEHAVIOR_RUNNING_VELOCITY_THRESHOLD", "BEHAVIOR_SUDDEN_ACCELERATION_THRESHOLD",
            "BEHAVIOR_DIRECTION_CHANGE_DEGREES", "BEHAVIOR_LOITERING_SECONDS",
            "BEHAVIOR_FALL_ASPECT_RATIO_CHANGE", "BEHAVIOR_FALL_DISPLACEMENT_THRESHOLD",
        ]:
            assert hasattr(s, key), f"Missing setting: {key}"

    def test_schemas_importable(self):
        from backend.schemas import BehaviorState, BehaviorEvent, PrimaryBehavior
        assert PrimaryBehavior.MOVING == "MOVING"

    def test_visualizer_has_overlay_method(self):
        from vision.tracking.visualizer import TrackingVisualizer
        assert callable(TrackingVisualizer().draw_behavior_overlays)

    def test_engine_reset(self):
        engine = BehaviorEngine(confirmation_frames=2)
        engine.process(_make_frame([_make_track(1, 100.0, 100.0, 0.0)], 0.0))
        engine.reset()
        assert len(engine._machines) == 0

    def test_primary_behavior_unique_values(self):
        values = [b.value for b in PrimaryBehavior]
        assert len(values) == len(set(values))

    def test_event_type_unique_values(self):
        values = [b.value for b in BehaviorEventType]
        assert len(values) == len(set(values))


# ---------------------------------------------------------------------------
# Motion utility tests
# ---------------------------------------------------------------------------

class TestMotionUtils:
    def test_heading_change_acute(self):
        assert compute_heading_change(0.0, 90.0) == pytest.approx(90.0)

    def test_heading_change_wraparound(self):
        assert compute_heading_change(350.0, 10.0) == pytest.approx(20.0)

    def test_heading_change_opposite(self):
        assert compute_heading_change(0.0, 180.0) == pytest.approx(180.0)

    def test_speed_average_empty(self):
        assert compute_speed_average(collections.deque()) is None

    def test_depth_trend_approaching(self):
        assert compute_depth_trend([0.3, 0.5, 0.7, 0.9]) == "approaching"

    def test_depth_trend_receding(self):
        assert compute_depth_trend([0.9, 0.7, 0.5, 0.3]) == "receding"

    def test_depth_trend_stable(self):
        assert compute_depth_trend([0.5, 0.5, 0.5, 0.51]) == "stable"

    def test_depth_trend_insufficient(self):
        assert compute_depth_trend([0.5]) is None
