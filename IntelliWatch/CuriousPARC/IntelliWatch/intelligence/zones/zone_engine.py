import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union
import cv2
import numpy as np

from backend.schemas.events import EventType, IndustrialEvent, SeverityLevel
from backend.schemas.tracking import FrameTracks, TrackedObject
from backend.schemas.zones import (
    FrameZoneOccupancy,
    RestrictedZone,
    ZoneMembership,
    ZoneMembershipStatus,
    ZoneType,
)
from configs.settings import get_settings
from intelligence.events.event_detector import BaseEventDetector
from intelligence.zones.zone_monitor import BaseZoneMonitor

logger = logging.getLogger("intelliwatch.zone_engine")


def compute_contact_point(track: TrackedObject) -> Tuple[float, float]:
    """
    Computes representative ground/contact point for spatial zone reasoning.

    - For person/worker tracks: uses bottom-center foot contact point ((x1 + x2) / 2.0, y2).
      The foot-point represents where the person contacts the ground plane, preventing
      false-positive or delayed zone entries caused by upper-body centroid positioning.
    - For non-person entities (machinery, vehicles, general objects): defaults to bounding box
      centroid ((x1 + x2) / 2.0, (y1 + y2) / 2.0).

    Args:
        track: TrackedObject with bounding box and class metadata.

    Returns:
        (x, y) float coordinate of representative ground contact point.
    """
    bbox = track.bbox
    cls = (track.class_name or "").lower().strip()
    if cls in ("person", "worker", "pedestrian"):
        return ((bbox.x1 + bbox.x2) / 2.0, float(bbox.y2))
    # Centroid for non-person objects
    return ((bbox.x1 + bbox.x2) / 2.0, (bbox.y1 + bbox.y2) / 2.0)


class _TrackZoneTemporalState:
    """
    Internal state machine tracking temporal membership, confirmation, and dwell duration
    for a specific (track_id, zone_id) pair.
    """

    def __init__(self, track_id: int, zone_id: str):
        self.track_id: int = track_id
        self.zone_id: str = zone_id
        self.consecutive_inside_frames: int = 0
        self.confirmed_inside: bool = False
        self.first_detected_timestamp: Optional[float] = None
        self.confirmed_entry_timestamp: Optional[float] = None
        self.last_seen_timestamp: float = 0.0
        self.dwell_event_emitted: bool = False


class ZoneEngine(BaseZoneMonitor, BaseEventDetector):
    """
    Deterministic Geometric Spatial Reasoning & Restricted Zone Monitoring Engine.

    Features:
      1. Arbitrary polygon support via OpenCV pointPolygonTest (convex or concave).
      2. Ground contact foot-point estimation for tracked workers.
      3. Independent multi-zone evaluation per tracked object.
      4. Temporal confirmation to filter transient sensor noise.
      5. Dwell-time accumulation and single-trigger DWELL_EXCEEDED threshold alerting.
      6. Clean state lifecycle and departed track cleanup.
      7. Configuration-driven zone loading via JSON without code changes.
    """

    def __init__(
        self,
        zones: Optional[List[RestrictedZone]] = None,
        config_path: Optional[Union[str, Path]] = None,
        confirmation_frames: Optional[int] = None,
        default_max_dwell_seconds: Optional[float] = None,
        camera_id: str = "cam_01",
    ):
        settings = get_settings()

        self.camera_id: str = camera_id
        self.confirmation_frames: int = (
            confirmation_frames
            if confirmation_frames is not None
            else getattr(settings, "ZONE_ENTRY_CONFIRMATION_FRAMES", 3)
        )
        self.default_max_dwell_seconds: float = (
            default_max_dwell_seconds
            if default_max_dwell_seconds is not None
            else getattr(settings, "ZONE_MAX_DWELL_SECONDS", 10.0)
        )

        self._zones: Dict[str, RestrictedZone] = {}
        self._zone_contours: Dict[str, np.ndarray] = {}

        # Load zones from passed arguments or configuration file
        if zones is not None:
            for zone in zones:
                self.add_zone(zone)
        else:
            cfg = config_path if config_path is not None else getattr(settings, "ZONE_CONFIG_PATH", "configs/zones.json")
            self.load_zones_from_file(cfg)

        # Temporal state keyed by (track_id, zone_id)
        self._actor_states: Dict[Tuple[int, str], _TrackZoneTemporalState] = {}

    @property
    def zones(self) -> Dict[str, RestrictedZone]:
        """Dictionary of currently loaded zones keyed by zone_id."""
        return self._zones

    def add_zone(self, zone: RestrictedZone) -> bool:
        """
        Validates and registers a RestrictedZone with pre-compiled OpenCV contour cache.
        Returns True if successfully registered, False if invalid.
        """
        try:
            if len(zone.polygon) < 3:
                logger.warning(f"Zone '{zone.zone_id}' rejected: polygon has fewer than 3 vertices.")
                return False

            contour = np.array(zone.polygon, dtype=np.float32)
            # Check for NaN / Inf
            if not np.all(np.isfinite(contour)):
                logger.warning(f"Zone '{zone.zone_id}' rejected: non-finite vertex coordinates.")
                return False

            self._zones[zone.zone_id] = zone
            self._zone_contours[zone.zone_id] = contour
            logger.info(f"Registered zone '{zone.zone_id}' ({zone.name}) with {len(zone.polygon)} vertices.")
            return True
        except Exception as e:
            logger.warning(f"Failed to register zone '{getattr(zone, 'zone_id', 'unknown')}': {e}")
            return False

    def remove_zone(self, zone_id: str) -> None:
        """Removes a zone from active evaluation."""
        self._zones.pop(zone_id, None)
        self._zone_contours.pop(zone_id, None)
        # Clean up any actor states associated with this zone
        keys_to_remove = [k for k in self._actor_states if k[1] == zone_id]
        for k in keys_to_remove:
            del self._actor_states[k]

    def load_zones_from_file(self, config_path: Union[str, Path]) -> int:
        """
        Loads zone configurations from a JSON file. Gracefully handles missing or malformed files.
        Returns count of successfully registered zones.
        """
        path = Path(config_path)
        if not path.is_file():
            logger.warning(f"Zone configuration file not found at: {path}. Starting with empty zones.")
            return 0

        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not isinstance(data, list):
                logger.warning(f"Zone configuration at {path} must contain a JSON list. Found: {type(data)}")
                return 0

            loaded_count = 0
            for item in data:
                try:
                    zone = RestrictedZone(**item)
                    if self.add_zone(zone):
                        loaded_count += 1
                except Exception as ex:
                    logger.warning(f"Skipping malformed zone definition in {path}: {ex}")

            logger.info(f"Loaded {loaded_count} restricted zones from {path}")
            return loaded_count
        except Exception as e:
            logger.error(f"Error loading zone configuration from {path}: {e}")
            return 0

    def reset_state(self) -> None:
        """Clears all historical temporal tracking buffers."""
        self._actor_states.clear()
        logger.debug("Cleared zone engine temporal state.")

    def test_point_in_zone(
        self,
        point: Tuple[float, float],
        zone_id: str,
    ) -> Tuple[bool, ZoneMembershipStatus]:
        """
        Tests whether a 2D coordinate is inside, on boundary, or outside a specified zone polygon.

        Uses OpenCV cv2.pointPolygonTest:
          - test result > 0: strictly inside
          - test result == 0: exactly on polygon boundary
          - test result < 0: strictly outside

        In industrial safety monitoring, points on boundary lines are considered inside (status=BOUNDARY).
        """
        if zone_id not in self._zone_contours:
            return False, ZoneMembershipStatus.OUTSIDE

        contour = self._zone_contours[zone_id]
        dist = cv2.pointPolygonTest(contour, (float(point[0]), float(point[1])), measureDist=False)

        if dist > 0:
            return True, ZoneMembershipStatus.INSIDE
        elif dist == 0:
            return True, ZoneMembershipStatus.BOUNDARY
        else:
            return False, ZoneMembershipStatus.OUTSIDE

    def process_frame(
        self,
        tracks: Union[FrameTracks, List[TrackedObject]],
        timestamp: Optional[float] = None,
        frame_id: Optional[int] = None,
    ) -> Tuple[List[ZoneMembership], List[IndustrialEvent]]:
        """
        Evaluates spatial zone relationships, temporal confirmation, dwell time, and events.

        Args:
            tracks: FrameTracks container or list of active TrackedObject instances.
            timestamp: Video stream timestamp in seconds. If None, extracted from FrameTracks or defaulted.
            frame_id: Frame sequence number.

        Returns:
            Tuple of (List of ZoneMembership evaluations, List of triggered IndustrialEvent incidents)
        """
        if isinstance(tracks, FrameTracks):
            track_list = tracks.active_tracks
            curr_timestamp = tracks.timestamp if timestamp is None else timestamp
            curr_frame_id = tracks.frame_id if frame_id is None else frame_id
        else:
            track_list = tracks or []
            curr_timestamp = 0.0 if timestamp is None else timestamp
            curr_frame_id = 0 if frame_id is None else frame_id

        memberships: List[ZoneMembership] = []
        events: List[IndustrialEvent] = []

        active_track_ids: Set[int] = set()
        enabled_zones = [z for z in self._zones.values() if z.enabled]

        # 1. Evaluate spatial membership for all active tracks against all enabled zones
        for track in track_list:
            track_id = track.track_id
            active_track_ids.add(track_id)
            contact_pt = compute_contact_point(track)

            for zone in enabled_zones:
                zone_id = zone.zone_id
                state_key = (track_id, zone_id)

                if state_key not in self._actor_states:
                    self._actor_states[state_key] = _TrackZoneTemporalState(track_id, zone_id)

                state = self._actor_states[state_key]
                state.last_seen_timestamp = curr_timestamp

                is_inside, status = self.test_point_in_zone(contact_pt, zone_id)

                if is_inside:
                    state.consecutive_inside_frames += 1
                    if state.first_detected_timestamp is None:
                        state.first_detected_timestamp = curr_timestamp

                    # Check for confirmed entry
                    if state.consecutive_inside_frames >= self.confirmation_frames and not state.confirmed_inside:
                        state.confirmed_inside = True
                        state.confirmed_entry_timestamp = (
                            state.first_detected_timestamp
                            if state.first_detected_timestamp is not None
                            else curr_timestamp
                        )

                        # Emit ZONE_ENTRY event
                        evt_id = f"evt_zone_entry_{track_id}_{zone_id}_{int(curr_timestamp * 1000)}"
                        entry_event = IndustrialEvent(
                            event_id=evt_id,
                            event_type=EventType.ZONE_ENTRY,
                            timestamp=curr_timestamp,
                            camera_id=self.camera_id,
                            tracked_object_ids=[track_id],
                            severity=SeverityLevel.HIGH,
                            explanation=(
                                f"Worker #{track_id} confirmed entry into restricted zone "
                                f"'{zone.name}' ({zone_id}) after {state.consecutive_inside_frames} "
                                f"consecutive frames."
                            ),
                            metadata={
                                "track_id": track_id,
                                "zone_id": zone_id,
                                "zone_name": zone.name,
                                "zone_type": zone.zone_type.value,
                                "entry_timestamp": state.confirmed_entry_timestamp,
                                "confirmation_frames": self.confirmation_frames,
                                "contact_point": list(contact_pt),
                            },
                        )
                        events.append(entry_event)
                        logger.warning(
                            f"ZONE INTRUSION CONFIRMED: Worker #{track_id} in '{zone.name}' "
                            f"(Frame {curr_frame_id}, time {curr_timestamp:.2f}s)"
                        )

                    # Compute current dwell time if confirmed inside
                    current_dwell = 0.0
                    if state.confirmed_inside and state.confirmed_entry_timestamp is not None:
                        current_dwell = max(0.0, curr_timestamp - state.confirmed_entry_timestamp)

                        # Check dwell threshold limit
                        max_dwell = (
                            zone.max_dwell_seconds
                            if zone.max_dwell_seconds is not None
                            else self.default_max_dwell_seconds
                        )
                        if current_dwell >= max_dwell and not state.dwell_event_emitted:
                            state.dwell_event_emitted = True
                            evt_dwell_id = f"evt_zone_dwell_{track_id}_{zone_id}_{int(curr_timestamp * 1000)}"
                            dwell_event = IndustrialEvent(
                                event_id=evt_dwell_id,
                                event_type=EventType.ZONE_DWELL_EXCEEDED,
                                timestamp=curr_timestamp,
                                camera_id=self.camera_id,
                                tracked_object_ids=[track_id],
                                severity=SeverityLevel.CRITICAL,
                                explanation=(
                                    f"Worker #{track_id} exceeded maximum dwell limit in restricted zone "
                                    f"'{zone.name}' ({zone_id}): stayed {current_dwell:.1f}s "
                                    f"(threshold: {max_dwell:.1f}s)."
                                ),
                                metadata={
                                    "track_id": track_id,
                                    "zone_id": zone_id,
                                    "zone_name": zone.name,
                                    "zone_type": zone.zone_type.value,
                                    "entry_timestamp": state.confirmed_entry_timestamp,
                                    "dwell_seconds": round(current_dwell, 2),
                                    "max_dwell_seconds": max_dwell,
                                },
                            )
                            events.append(dwell_event)
                            logger.warning(
                                f"ZONE DWELL EXCEEDED: Worker #{track_id} in '{zone.name}' "
                                f"for {current_dwell:.1f}s >= {max_dwell:.1f}s"
                            )

                    membership = ZoneMembership(
                        track_id=track_id,
                        zone_id=zone_id,
                        zone_name=zone.name,
                        is_inside=True,
                        status=status,
                        contact_point=contact_pt,
                        dwell_seconds=round(current_dwell, 2),
                        entry_timestamp=state.confirmed_entry_timestamp,
                    )
                    memberships.append(membership)

                else:
                    # Outside the zone
                    # If previously confirmed inside, emit ZONE_EXIT
                    if state.confirmed_inside:
                        total_dwell = max(
                            0.0,
                            curr_timestamp - (state.confirmed_entry_timestamp or curr_timestamp)
                        )
                        evt_exit_id = f"evt_zone_exit_{track_id}_{zone_id}_{int(curr_timestamp * 1000)}"
                        exit_event = IndustrialEvent(
                            event_id=evt_exit_id,
                            event_type=EventType.ZONE_EXIT,
                            timestamp=curr_timestamp,
                            camera_id=self.camera_id,
                            tracked_object_ids=[track_id],
                            severity=SeverityLevel.LOW,
                            explanation=(
                                f"Worker #{track_id} exited restricted zone '{zone.name}' ({zone_id}) "
                                f"after total dwell time of {total_dwell:.1f}s."
                            ),
                            metadata={
                                "track_id": track_id,
                                "zone_id": zone_id,
                                "zone_name": zone.name,
                                "zone_type": zone.zone_type.value,
                                "entry_timestamp": state.confirmed_entry_timestamp,
                                "exit_timestamp": curr_timestamp,
                                "dwell_seconds": round(total_dwell, 2),
                            },
                        )
                        events.append(exit_event)
                        logger.info(
                            f"ZONE EXIT: Worker #{track_id} left '{zone.name}' "
                            f"(Dwell: {total_dwell:.1f}s)"
                        )

                    # Reset temporal state upon exiting
                    state.consecutive_inside_frames = 0
                    state.confirmed_inside = False
                    state.first_detected_timestamp = None
                    state.confirmed_entry_timestamp = None
                    state.dwell_event_emitted = False

                    membership = ZoneMembership(
                        track_id=track_id,
                        zone_id=zone_id,
                        zone_name=zone.name,
                        is_inside=False,
                        status=ZoneMembershipStatus.OUTSIDE,
                        contact_point=contact_pt,
                        dwell_seconds=0.0,
                        entry_timestamp=None,
                    )
                    memberships.append(membership)

        # 2. Cleanup states for disappeared / departed tracks
        departed_keys = [k for k in self._actor_states if k[0] not in active_track_ids]
        for k in departed_keys:
            state = self._actor_states[k]
            # If the actor disappeared while confirmed inside, emit exit alert
            if state.confirmed_inside:
                zone = self._zones.get(state.zone_id)
                zone_name = zone.name if zone else state.zone_id
                zone_type_val = zone.zone_type.value if zone else "restricted"
                total_dwell = max(
                    0.0,
                    curr_timestamp - (state.confirmed_entry_timestamp or curr_timestamp)
                )
                evt_exit_id = f"evt_zone_exit_{state.track_id}_{state.zone_id}_{int(curr_timestamp * 1000)}"
                exit_event = IndustrialEvent(
                    event_id=evt_exit_id,
                    event_type=EventType.ZONE_EXIT,
                    timestamp=curr_timestamp,
                    camera_id=self.camera_id,
                    tracked_object_ids=[state.track_id],
                    severity=SeverityLevel.LOW,
                    explanation=(
                        f"Worker #{state.track_id} departed scene while inside restricted zone "
                        f"'{zone_name}' ({state.zone_id}) after dwell time of {total_dwell:.1f}s."
                    ),
                    metadata={
                        "track_id": state.track_id,
                        "zone_id": state.zone_id,
                        "zone_name": zone_name,
                        "zone_type": zone_type_val,
                        "entry_timestamp": state.confirmed_entry_timestamp,
                        "exit_timestamp": curr_timestamp,
                        "dwell_seconds": round(total_dwell, 2),
                        "reason": "track_departed",
                    },
                )
                events.append(exit_event)

            del self._actor_states[k]
            logger.debug(f"Purged zone tracking state for departed track #{k[0]} in zone '{k[1]}'")

        return memberships, events

    def check_intrusion(self, tracks: FrameTracks) -> List[IndustrialEvent]:
        """
        Implementation of BaseZoneMonitor.check_intrusion().
        """
        _, events = self.process_frame(tracks)
        return events

    def evaluate(self, tracks: FrameTracks, context: Dict[str, Any]) -> List[IndustrialEvent]:
        """
        Implementation of BaseEventDetector.evaluate().
        """
        _, events = self.process_frame(
            tracks=tracks,
            timestamp=context.get("timestamp"),
            frame_id=context.get("frame_id"),
        )
        return events
