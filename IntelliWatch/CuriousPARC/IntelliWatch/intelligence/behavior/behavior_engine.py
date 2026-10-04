"""
intelligence/behavior/behavior_engine.py
=========================================
Step 8 — Behavior and Temporal Analysis engine.

The BehaviorEngine is a stateful, frame-by-frame processor that consumes
the outputs of existing pipeline stages (tracking, zones, optional depth)
and produces BehaviorState snapshots and BehaviorEvents.

Architecture
------------
  FrameTracks (from ByteTrack Step 4)
       +
  FrameZoneOccupancy (from ZoneEngine Step 6, optional)
       +
  per-track depth values (from DepthPipeline Step 7, optional)
       |
       v
  BehaviorEngine.process(...)
       |
       v
  List[BehaviorState]  +  List[BehaviorEvent]

Design principles
-----------------
- No ML model inference inside this engine.
- No YOLO, PPE, or depth model runs triggered here.
- All thresholds read from settings; no hardcoded values.
- CPU-only; no GPU dependency.
- Temporal confirmation: behaviors require BEHAVIOR_CONFIRMATION_FRAMES
  consecutive frames before being confirmed.
- Technical honesty: all motion quantities are image-space; no physical
  speed or guaranteed fall detection is claimed.
"""

import logging
import math
import uuid
from typing import Any, Dict, List, Optional, Tuple

from backend.schemas.behavior import (
    BehaviorEvent,
    BehaviorEventType,
    BehaviorState,
    PrimaryBehavior,
)
from backend.schemas.tracking import FrameTracks, TrackedObject
from configs.settings import get_settings
from intelligence.behavior.motion import (
    check_direction_change,
    compute_depth_trend,
    compute_recent_accel,
    compute_speed_average,
    detect_fall_heuristic,
)
from intelligence.behavior.temporal_state import (
    MotionObservation,
    TrackStateRegistry,
    TrackTemporalState,
)

logger = logging.getLogger("intelliwatch.behavior.engine")


class _TrackBehaviorMachine:
    """
    Internal per-track behavioral state machine with temporal confirmation.
    Holds pending candidate counts for confirming or de-confirming behavior states.
    """

    def __init__(self, track_id: int, confirmation_frames: int):
        self.track_id = track_id
        self.confirmation_frames = confirmation_frames

        # Current confirmed primary behavior
        self.confirmed_behavior: PrimaryBehavior = PrimaryBehavior.UNKNOWN

        # Candidate being tested for confirmation
        self._candidate: Optional[PrimaryBehavior] = None
        self._candidate_count: int = 0

        # Duration tracking
        self._behavior_start_ts: Optional[float] = None

        # For prolonged stationary: track whether we already fired the event
        self._prolonged_fired: bool = False

    def propose(
        self,
        candidate: PrimaryBehavior,
        timestamp: float,
    ) -> Tuple[bool, Optional[PrimaryBehavior]]:
        """
        Propose a candidate behavior. Uses confirmation hysteresis.

        Returns:
            (state_changed: bool, previous_behavior: Optional[PrimaryBehavior])
        """
        if candidate == self._candidate:
            self._candidate_count += 1
        else:
            self._candidate = candidate
            self._candidate_count = 1

        if self._candidate_count >= self.confirmation_frames:
            if self.confirmed_behavior != candidate:
                # If we are PROLONGED_STATIONARY and candidate is STATIONARY, do not downgrade back to STATIONARY
                if self.confirmed_behavior == PrimaryBehavior.PROLONGED_STATIONARY and candidate == PrimaryBehavior.STATIONARY:
                    return False, None
                prev = self.confirmed_behavior
                self.confirmed_behavior = candidate
                self._behavior_start_ts = timestamp
                # Reset prolonged flag when leaving stationary family
                if candidate not in (PrimaryBehavior.STATIONARY, PrimaryBehavior.PROLONGED_STATIONARY):
                    self._prolonged_fired = False
                return True, prev

        return False, None

    def escalate_to_prolonged(self, timestamp: float) -> bool:
        """
        Directly escalate to PROLONGED_STATIONARY without requiring another N frames.
        Returns True only on the first escalation (to avoid duplicate events).
        """
        if self._prolonged_fired:
            return False
        self._prolonged_fired = True
        self.confirmed_behavior = PrimaryBehavior.PROLONGED_STATIONARY
        self._behavior_start_ts = timestamp
        return True

    def behavior_duration(self, current_timestamp: float) -> float:
        """Returns seconds in current confirmed behavior state."""
        if self._behavior_start_ts is None:
            return 0.0
        return max(0.0, current_timestamp - self._behavior_start_ts)

    def confirmation_count(self) -> int:
        return min(self._candidate_count, self.confirmation_frames)


class BehaviorEngine:
    """
    Frame-by-frame behavioral state machine for all tracked entities.

    Processes one FrameTracks at a time (and optional zone/depth context)
    and produces per-track BehaviorState snapshots plus BehaviorEvents for
    significant state changes.

    This engine implements rule-based temporal behavior reasoning.
    It is NOT a trained anomaly-detection model, and does NOT claim
    semantic intent, psychological state, or guaranteed fall detection.
    """

    def __init__(
        self,
        history_seconds: Optional[float] = None,
        confirmation_frames: Optional[int] = None,
        stationary_distance_threshold: Optional[float] = None,
        stationary_seconds: Optional[float] = None,
        running_velocity_threshold: Optional[float] = None,
        sudden_acceleration_threshold: Optional[float] = None,
        direction_change_degrees: Optional[float] = None,
        loitering_seconds: Optional[float] = None,
        fall_aspect_ratio_change: Optional[float] = None,
        fall_displacement_threshold: Optional[float] = None,
        camera_id: str = "cam_01",
    ):
        s = get_settings()

        self.history_seconds = history_seconds if history_seconds is not None else s.BEHAVIOR_HISTORY_SECONDS
        self.confirmation_frames = confirmation_frames if confirmation_frames is not None else s.BEHAVIOR_CONFIRMATION_FRAMES
        self.stationary_distance_threshold = stationary_distance_threshold if stationary_distance_threshold is not None else s.BEHAVIOR_STATIONARY_DISTANCE_THRESHOLD
        self.stationary_seconds = stationary_seconds if stationary_seconds is not None else s.BEHAVIOR_STATIONARY_SECONDS
        self.running_velocity_threshold = running_velocity_threshold if running_velocity_threshold is not None else s.BEHAVIOR_RUNNING_VELOCITY_THRESHOLD
        self.sudden_acceleration_threshold = sudden_acceleration_threshold if sudden_acceleration_threshold is not None else s.BEHAVIOR_SUDDEN_ACCELERATION_THRESHOLD
        self.direction_change_degrees = direction_change_degrees if direction_change_degrees is not None else s.BEHAVIOR_DIRECTION_CHANGE_DEGREES
        self.loitering_seconds = loitering_seconds if loitering_seconds is not None else s.BEHAVIOR_LOITERING_SECONDS
        self.fall_aspect_ratio_change = fall_aspect_ratio_change if fall_aspect_ratio_change is not None else s.BEHAVIOR_FALL_ASPECT_RATIO_CHANGE
        self.fall_displacement_threshold = fall_displacement_threshold if fall_displacement_threshold is not None else s.BEHAVIOR_FALL_DISPLACEMENT_THRESHOLD
        self.camera_id = camera_id

        # Registries
        self._registry = TrackStateRegistry(history_seconds=self.history_seconds)
        self._machines: Dict[int, _TrackBehaviorMachine] = {}

        # Per-track depth history (for trend computation; optional)
        self._depth_history: Dict[int, List[Optional[float]]] = {}

        logger.info(
            "BehaviorEngine initialized: hist=%.1fs conf_frames=%d "
            "stat_dist=%.1fpx stat_s=%.1fs run_vel=%.1fpx/s "
            "accel=%.1fpx/s2 dir=%.0fdeg loiter=%.1fs",
            self.history_seconds, self.confirmation_frames,
            self.stationary_distance_threshold, self.stationary_seconds,
            self.running_velocity_threshold, self.sudden_acceleration_threshold,
            self.direction_change_degrees, self.loitering_seconds,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def process(
        self,
        frame_tracks: FrameTracks,
        zone_occupancy: Optional[Any] = None,
        track_depth_map: Optional[Dict[int, float]] = None,
        external_fall_tracks: Optional[Set[int]] = None,
    ) -> Tuple[List[BehaviorState], List[BehaviorEvent]]:
        """
        Process a single frame of tracking data and return behavior results.

        Args:
            frame_tracks: FrameTracks output from ByteTrack (Step 4).
            zone_occupancy: Optional FrameZoneOccupancy from ZoneEngine (Step 6).
                            Used to enrich behavior states with zone context.
            track_depth_map: Optional dict mapping track_id -> relative depth value
                             at the track's contact point (from Step 7, DepthPipeline).
                             Values are relative/uncalibrated; NOT metric meters.
            external_fall_tracks: Optional set of track_ids flagged with fall detection
                                  from domain-specific visual detectors (e.g. PPE model).

        Returns:
            (behavior_states, behavior_events)
        """
        timestamp = frame_tracks.timestamp
        frame_id = frame_tracks.frame_id
        states: List[BehaviorState] = []
        events: List[BehaviorEvent] = []

        # Build zone context lookup
        zone_ctx = self._extract_zone_context(zone_occupancy)

        # Process each active track
        active_ids = set()
        for track in frame_tracks.active_tracks:
            tid = track.track_id
            active_ids.add(tid)

            # Update depth history
            depth_val: Optional[float] = None
            if track_depth_map is not None:
                depth_val = track_depth_map.get(tid)
            if tid not in self._depth_history:
                self._depth_history[tid] = []
            self._depth_history[tid].append(depth_val)
            # Bound depth history to 10 entries
            if len(self._depth_history[tid]) > 10:
                self._depth_history[tid] = self._depth_history[tid][-10:]

            # Get/create temporal state and behavior machine
            t_state = self._registry.get_or_create(tid)
            machine = self._get_or_create_machine(tid)

            # Record observation
            bbox = track.bbox
            obs = t_state.add_observation(
                timestamp=timestamp,
                frame_id=frame_id,
                cx=track.centroid_x or ((bbox.x1 + bbox.x2) / 2.0),
                cy=track.centroid_y or ((bbox.y1 + bbox.y2) / 2.0),
                bbox_x1=bbox.x1,
                bbox_y1=bbox.y1,
                bbox_x2=bbox.x2,
                bbox_y2=bbox.y2,
            )

            # Classify behavior
            is_ext_fall = bool(external_fall_tracks and tid in external_fall_tracks)
            state, new_events = self._classify(
                tid=tid,
                timestamp=timestamp,
                frame_id=frame_id,
                t_state=t_state,
                machine=machine,
                obs=obs,
                zone_ctx=zone_ctx.get(tid),
                depth_val=depth_val,
                external_fall=is_ext_fall,
            )
            states.append(state)
            events.extend(new_events)

        # Purge tracks absent for > 2 * history_seconds
        self._registry.purge_stale(timestamp, max_age_seconds=self.history_seconds * 2)

        return states, events

    # ------------------------------------------------------------------
    # Internal classification logic
    # ------------------------------------------------------------------

    def _classify(
        self,
        tid: int,
        timestamp: float,
        frame_id: int,
        t_state: TrackTemporalState,
        machine: "_TrackBehaviorMachine",
        obs: Optional[MotionObservation],
        zone_ctx: Optional[Dict[str, Any]],
        depth_val: Optional[float],
        external_fall: bool = False,
    ) -> Tuple[BehaviorState, List[BehaviorEvent]]:
        """Core per-track classification logic."""
        events: List[BehaviorEvent] = []
        depth_hist = self._depth_history.get(tid, [])

        # Default motion values
        speed = None
        accel = None
        vx = None
        vy = None
        heading = None
        flag_rapid = False
        flag_sudden = False
        flag_dir = False
        flag_fall = bool(external_fall)
        depth_trend_str = None

        if obs is not None:
            speed = obs.speed_px_per_s
            accel = obs.accel_px_per_s2
            vx = obs.vx
            vy = obs.vy
            heading = obs.heading_degrees

            # --- Secondary flags ---

            # Rapid movement flag
            avg_speed = compute_speed_average(t_state.motion_history, n=self.confirmation_frames)
            if avg_speed is not None and avg_speed >= self.running_velocity_threshold:
                flag_rapid = True

            # Sudden movement flag (single-frame acceleration)
            if accel is not None and accel >= self.sudden_acceleration_threshold:
                flag_sudden = True

            # Direction change flag
            if check_direction_change(t_state.motion_history, self.direction_change_degrees):
                flag_dir = True

            # Possible fall heuristic
            prev_obs = None
            if len(t_state.motion_history) >= 2:
                prev_obs = list(t_state.motion_history)[-2]
            if detect_fall_heuristic(
                obs, prev_obs,
                ar_change_threshold=self.fall_aspect_ratio_change,
                disp_threshold=self.fall_displacement_threshold,
            ):
                flag_fall = True

        # Depth trend
        depth_trend_str = compute_depth_trend(depth_hist, threshold=0.05)

        # --- Determine candidate primary behavior ---
        # Update stationary accumulator BEFORE candidate determination
        self._update_stationary_counter(t_state, obs, timestamp)

        candidate = self._determine_candidate(
            t_state=t_state,
            obs=obs,
            timestamp=timestamp,
            flag_rapid=flag_rapid,
            flag_fall=flag_fall,
        )

        # --- Temporal confirmation ---
        changed, prev_behavior = machine.propose(candidate, timestamp)
        confirmed_behavior = machine.confirmed_behavior
        behavior_duration = machine.behavior_duration(timestamp)
        confirm_count = machine.confirmation_count()

        # --- Generate events on significant transitions ---
        if changed and prev_behavior is not None:
            ev = self._make_transition_event(
                tid=tid, timestamp=timestamp, frame_id=frame_id,
                new_behavior=confirmed_behavior, prev_behavior=prev_behavior,
                speed=speed, t_state=t_state, zone_ctx=zone_ctx,
            )
            if ev is not None:
                events.append(ev)

        # Prolonged stationary check — direct escalation, no second confirmation pass
        # Once stationary_duration_s crosses loitering_seconds, immediately escalate.
        if confirmed_behavior in (PrimaryBehavior.STATIONARY,) and t_state.stationary_duration_s >= self.loitering_seconds:
            newly_escalated = machine.escalate_to_prolonged(timestamp)
            confirmed_behavior = PrimaryBehavior.PROLONGED_STATIONARY
            if newly_escalated:
                ev = self._make_prolonged_stationary_event(
                    tid=tid, timestamp=timestamp, frame_id=frame_id,
                    duration=t_state.stationary_duration_s, zone_ctx=zone_ctx,
                )
                events.append(ev)

        # Sudden movement event (not dependent on primary state confirmation)
        if flag_sudden and obs is not None:
            ev = self._make_sudden_movement_event(
                tid=tid, timestamp=timestamp, frame_id=frame_id,
                accel=obs.accel_px_per_s2, speed=speed, zone_ctx=zone_ctx,
            )
            events.append(ev)

        # Possible fall event
        if flag_fall:
            ev = self._make_fall_event(
                tid=tid, timestamp=timestamp, frame_id=frame_id,
                speed=speed, zone_ctx=zone_ctx,
            )
            events.append(ev)

        # Build explanation string
        explanation = self._build_explanation(
            confirmed_behavior, t_state, obs, flag_rapid, flag_sudden, flag_dir, flag_fall,
        )

        # Zone context fields
        zone_id = zone_ctx["zone_id"] if zone_ctx else None
        zone_name = zone_ctx["zone_name"] if zone_ctx else None
        zone_dwell = zone_ctx["dwell_s"] if zone_ctx else None

        state = BehaviorState(
            track_id=tid,
            timestamp=timestamp,
            frame_id=frame_id,
            primary_behavior=confirmed_behavior,
            flag_rapid_movement=flag_rapid,
            flag_sudden_movement=flag_sudden,
            flag_direction_change=flag_dir,
            flag_possible_fall=flag_fall,
            image_speed_px_per_s=round(speed, 2) if speed is not None else None,
            image_accel_px_per_s2=round(accel, 2) if accel is not None else None,
            velocity_x=round(vx, 2) if vx is not None else None,
            velocity_y=round(vy, 2) if vy is not None else None,
            heading_degrees=round(heading, 1) if heading is not None else None,
            stationary_duration_s=round(t_state.stationary_duration_s, 2),
            behavior_duration_s=round(behavior_duration, 2),
            zone_id=zone_id,
            zone_name=zone_name,
            zone_dwell_s=zone_dwell,
            relative_depth=depth_val,
            depth_trend=depth_trend_str,
            confirmation_frames=confirm_count,
            explanation=explanation,
        )
        return state, events

    def _determine_candidate(
        self,
        t_state: TrackTemporalState,
        obs: Optional[MotionObservation],
        timestamp: float,
        flag_rapid: bool,
        flag_fall: bool,
    ) -> PrimaryBehavior:
        """Select the candidate primary behavior for this frame."""
        if obs is None:
            # Not enough history yet
            return PrimaryBehavior.UNKNOWN

        if flag_fall:
            return PrimaryBehavior.POSSIBLE_FALL

        if flag_rapid:
            return PrimaryBehavior.RAPID_MOVEMENT

        # Stationary check: use displacement of the current observation
        if obs.displacement < self.stationary_distance_threshold:
            # Candidate stationary if we have been in this state long enough
            if t_state.stationary_duration_s >= self.stationary_seconds:
                return PrimaryBehavior.STATIONARY
            # Not yet confirmed stationary duration; stay UNKNOWN or current
            return PrimaryBehavior.STATIONARY  # will be gated by confirmation_frames

        return PrimaryBehavior.MOVING

    def _update_stationary_counter(
        self,
        t_state: TrackTemporalState,
        obs: Optional[MotionObservation],
        timestamp: float,
    ) -> None:
        """Increment or reset the stationary duration accumulator."""
        if obs is None:
            return
        if obs.displacement < self.stationary_distance_threshold:
            t_state.stationary_duration_s += obs.dt
        else:
            t_state.stationary_duration_s = 0.0

    def _get_or_create_machine(self, track_id: int) -> "_TrackBehaviorMachine":
        if track_id not in self._machines:
            self._machines[track_id] = _TrackBehaviorMachine(
                track_id=track_id, confirmation_frames=self.confirmation_frames
            )
        return self._machines[track_id]

    # ------------------------------------------------------------------
    # Event builders
    # ------------------------------------------------------------------

    def _make_transition_event(
        self, tid, timestamp, frame_id, new_behavior, prev_behavior,
        speed, t_state, zone_ctx,
    ) -> Optional[BehaviorEvent]:
        """Create a BEHAVIOR_TRANSITION event for significant state changes."""
        significant = {
            PrimaryBehavior.RAPID_MOVEMENT,
            PrimaryBehavior.PROLONGED_STATIONARY,
            PrimaryBehavior.POSSIBLE_FALL,
            PrimaryBehavior.STATIONARY,
        }
        if new_behavior not in significant and prev_behavior not in significant:
            return None

        severity_map = {
            PrimaryBehavior.POSSIBLE_FALL: "high",
            PrimaryBehavior.RAPID_MOVEMENT: "medium",
            PrimaryBehavior.PROLONGED_STATIONARY: "medium",
            PrimaryBehavior.STATIONARY: "low",
            PrimaryBehavior.MOVING: "low",
        }
        severity = severity_map.get(new_behavior, "low")

        explanation = (
            f"Track {tid} transitioned from {prev_behavior.value} to "
            f"{new_behavior.value} after {self.confirmation_frames} confirmed frame(s)."
        )
        if speed is not None:
            explanation += f" Image-space speed: {speed:.1f} px/s."

        return BehaviorEvent(
            event_id=str(uuid.uuid4()),
            event_type=BehaviorEventType.BEHAVIOR_TRANSITION,
            timestamp=timestamp,
            frame_id=frame_id,
            track_id=tid,
            camera_id=self.camera_id,
            primary_behavior=new_behavior,
            previous_behavior=prev_behavior,
            severity=severity,
            explanation=explanation,
            zone_id=zone_ctx["zone_id"] if zone_ctx else None,
            image_speed_px_per_s=speed,
            stationary_duration_s=t_state.stationary_duration_s,
        )

    def _make_prolonged_stationary_event(
        self, tid, timestamp, frame_id, duration, zone_ctx,
    ) -> BehaviorEvent:
        zone_str = f" inside zone '{zone_ctx['zone_name']}'" if zone_ctx else ""
        explanation = (
            f"Track {tid} has remained nearly stationary{zone_str} for "
            f"{duration:.1f} seconds, exceeding the configured loitering threshold "
            f"of {self.loitering_seconds:.1f}s. (Rule-based temporal reasoning; "
            f"image-space motion analysis only.)"
        )
        return BehaviorEvent(
            event_id=str(uuid.uuid4()),
            event_type=BehaviorEventType.PROLONGED_STATIONARY,
            timestamp=timestamp,
            frame_id=frame_id,
            track_id=tid,
            camera_id=self.camera_id,
            primary_behavior=PrimaryBehavior.PROLONGED_STATIONARY,
            severity="medium",
            explanation=explanation,
            zone_id=zone_ctx["zone_id"] if zone_ctx else None,
            stationary_duration_s=duration,
        )

    def _make_sudden_movement_event(
        self, tid, timestamp, frame_id, accel, speed, zone_ctx,
    ) -> BehaviorEvent:
        explanation = (
            f"Track {tid} exhibited a sudden large image-space acceleration "
            f"({accel:.1f} px/s^2), exceeding the configured threshold of "
            f"{self.sudden_acceleration_threshold:.1f} px/s^2. "
            f"This may indicate a rapid stop, start, or impact. "
            f"Image-space analysis only; NOT a physical impact detection."
        )
        return BehaviorEvent(
            event_id=str(uuid.uuid4()),
            event_type=BehaviorEventType.SUDDEN_MOVEMENT,
            timestamp=timestamp,
            frame_id=frame_id,
            track_id=tid,
            camera_id=self.camera_id,
            primary_behavior=PrimaryBehavior.SUDDEN_MOVEMENT,
            severity="medium",
            explanation=explanation,
            zone_id=zone_ctx["zone_id"] if zone_ctx else None,
            image_speed_px_per_s=speed,
        )

    def _make_fall_event(
        self, tid, timestamp, frame_id, speed, zone_ctx,
    ) -> BehaviorEvent:
        explanation = (
            f"Track {tid}: bounding-box geometry heuristic detected a POSSIBLE_FALL "
            f"(sudden aspect-ratio drop combined with large centroid displacement). "
            f"This is a preliminary geometric signal and is NOT medically or semantically "
            f"reliable fall detection. A pose-based step is required for confident fall detection."
        )
        return BehaviorEvent(
            event_id=str(uuid.uuid4()),
            event_type=BehaviorEventType.POSSIBLE_FALL,
            timestamp=timestamp,
            frame_id=frame_id,
            track_id=tid,
            camera_id=self.camera_id,
            primary_behavior=PrimaryBehavior.POSSIBLE_FALL,
            severity="high",
            explanation=explanation,
            zone_id=zone_ctx["zone_id"] if zone_ctx else None,
            image_speed_px_per_s=speed,
        )

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_zone_context(zone_occupancy: Optional[Any]) -> Dict[int, Dict[str, Any]]:
        """
        Extracts a track_id -> zone_context dict from a FrameZoneOccupancy object
        or any compatible structure. Returns empty dict if input is None or invalid.
        """
        ctx: Dict[int, Dict[str, Any]] = {}
        if zone_occupancy is None:
            return ctx
        memberships = getattr(zone_occupancy, "memberships", [])
        for m in memberships:
            if not getattr(m, "is_inside", False):
                continue
            tid = getattr(m, "track_id", None)
            if tid is None:
                continue
            # If multiple zones, last one wins (could be extended to list)
            ctx[tid] = {
                "zone_id": getattr(m, "zone_id", None),
                "zone_name": getattr(m, "zone_name", None),
                "dwell_s": getattr(m, "dwell_seconds", 0.0),
            }
        return ctx

    @staticmethod
    def _build_explanation(
        behavior: PrimaryBehavior,
        t_state: TrackTemporalState,
        obs: Optional[MotionObservation],
        flag_rapid: bool,
        flag_sudden: bool,
        flag_dir: bool,
        flag_fall: bool,
    ) -> str:
        """Builds a human-readable explanation of the current behavior state."""
        parts = [f"Primary behavior: {behavior.value}."]
        if obs is not None:
            parts.append(f"Image-space speed: {obs.speed_px_per_s:.1f} px/s.")
        if t_state.stationary_duration_s > 0.1:
            parts.append(f"Stationary for: {t_state.stationary_duration_s:.1f}s.")
        flags = []
        if flag_rapid:
            flags.append("rapid-movement")
        if flag_sudden:
            flags.append("sudden-movement")
        if flag_dir:
            flags.append("direction-change")
        if flag_fall:
            flags.append("possible-fall-heuristic")
        if flags:
            parts.append(f"Active flags: {', '.join(flags)}.")
        return " ".join(parts)

    def reset(self) -> None:
        """Resets all internal state (useful for tests or stream restarts)."""
        self._registry = TrackStateRegistry(history_seconds=self.history_seconds)
        self._machines.clear()
        self._depth_history.clear()
        logger.info("BehaviorEngine reset.")
