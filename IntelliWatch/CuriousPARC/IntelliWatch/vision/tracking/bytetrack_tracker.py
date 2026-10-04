import collections
import logging
from types import SimpleNamespace
from typing import Dict, List, Optional, Set, Tuple
import numpy as np

from ultralytics.trackers.byte_tracker import BYTETracker

from backend.schemas.detection import BoundingBox, FrameDetections
from backend.schemas.tracking import FrameTracks, TrackPoint, TrackState, TrackedObject
from configs.settings import get_settings
from vision.tracking.base import BaseTracker

logger = logging.getLogger("intelliwatch.tracking.bytetrack")


class _DetectionContainer:
    """
    Lightweight numpy-compatible container for detections required by Ultralytics BYTETracker.
    Exposes slicing and attribute access (xywh, conf, cls) expected by BYTETracker.init_track.
    """

    def __init__(
        self,
        xywh: Optional[np.ndarray] = None,
        conf: Optional[np.ndarray] = None,
        cls: Optional[np.ndarray] = None,
    ):
        self.xywh = (
            np.empty((0, 4), dtype=np.float32)
            if xywh is None or len(xywh) == 0
            else np.asarray(xywh, dtype=np.float32)
        )
        self.conf = (
            np.empty((0,), dtype=np.float32)
            if conf is None or len(conf) == 0
            else np.asarray(conf, dtype=np.float32)
        )
        self.cls = (
            np.empty((0,), dtype=np.float32)
            if cls is None or len(cls) == 0
            else np.asarray(cls, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_DetectionContainer":
        return _DetectionContainer(self.xywh[idx], self.conf[idx], self.cls[idx])


class ByteTrackTracker(BaseTracker):
    """
    Production Multi-Object Tracker (MOT) implementing ByteTrack association logic.
    Maintains persistent track IDs, bounding boxes, trajectories, motion estimates,
    and track lifecycles across sequential video frames.

    Decoupled from YOLO: Consumes standardized FrameDetections and produces FrameTracks.
    All bounding box and trajectory coordinates remain in the original video pixel coordinate space.
    """

    def __init__(
        self,
        track_high_thresh: Optional[float] = None,
        track_low_thresh: Optional[float] = None,
        new_track_thresh: Optional[float] = None,
        match_thresh: Optional[float] = None,
        track_buffer: Optional[int] = None,
        track_history_length: Optional[int] = None,
        frame_rate: int = 30,
        fuse_score: bool = True,
    ):
        settings = get_settings()

        self.track_high_thresh = (
            track_high_thresh
            if track_high_thresh is not None
            else getattr(settings, "TRACK_HIGH_THRESH", 0.20)
        )
        self.track_low_thresh = (
            track_low_thresh
            if track_low_thresh is not None
            else getattr(settings, "TRACK_LOW_THRESH", 0.05)
        )
        self.new_track_thresh = (
            new_track_thresh
            if new_track_thresh is not None
            else getattr(settings, "TRACK_NEW_THRESH", self.track_high_thresh)
        )
        self.match_thresh = (
            match_thresh
            if match_thresh is not None
            else getattr(settings, "TRACK_MATCH_THRESH", 0.8)
        )
        self.track_buffer = (
            track_buffer
            if track_buffer is not None
            else getattr(settings, "TRACK_BUFFER", 30)
        )
        self.track_history_length = (
            track_history_length
            if track_history_length is not None
            else getattr(settings, "TRACK_HISTORY_LENGTH", 30)
        )
        self.frame_rate = max(1, int(frame_rate or getattr(settings, "FPS_SETTINGS", 30)))
        self.fuse_score = fuse_score

        # Prepare parameters namespace expected by Ultralytics BYTETracker
        self._args = SimpleNamespace(
            track_high_thresh=self.track_high_thresh,
            track_low_thresh=self.track_low_thresh,
            new_track_thresh=self.new_track_thresh,
            track_buffer=self.track_buffer,
            match_thresh=self.match_thresh,
            fuse_score=self.fuse_score,
        )

        self._tracker = BYTETracker(self._args, frame_rate=self.frame_rate)

        # Persistent track state stores
        self._trajectories: Dict[int, collections.deque] = {}
        self._track_classes: Dict[int, Tuple[int, str]] = {}  # track_id -> (class_id, class_name)
        self._track_class_votes: Dict[int, Dict[str, Tuple[int, float]]] = {}  # track_id -> {cname: (cid, score)}
        self._track_states: Dict[int, TrackState] = {}
        self._total_tracks_created: Set[int] = set()

        logger.info(
            f"ByteTrackTracker initialized: high_thresh={self.track_high_thresh}, "
            f"low_thresh={self.track_low_thresh}, match_thresh={self.match_thresh}, "
            f"buffer={self.track_buffer}, history_len={self.track_history_length}"
        )

    def update(
        self,
        detections: FrameDetections,
        frame: Optional[np.ndarray] = None,
    ) -> FrameTracks:
        """
        Associates current frame detections with existing persistent tracks.

        Args:
            detections: FrameDetections in original video pixel coordinates.
            frame: Optional frame image (unused by geometric ByteTrack).

        Returns:
            FrameTracks with persistent track IDs, lifecycle states, and trajectories.
        """
        frame_id = detections.frame_id
        timestamp = detections.timestamp
        w_max = float(detections.frame_width) if detections.frame_width else None
        h_max = float(detections.frame_height) if detections.frame_height else None

        # Handle empty detection frame
        if not detections.detections:
            empty_container = _DetectionContainer()
            self._tracker.update(empty_container)
            self._update_lost_and_removed_states()
            return FrameTracks(
                frame_id=frame_id,
                timestamp=timestamp,
                active_tracks=[],
                total_track_count=len(self._total_tracks_created),
            )

        # Convert detections to xywh format (center x, center y, width, height)
        xywh_list = []
        conf_list = []
        cls_list = []

        for d in detections.detections:
            bbox = d.bbox
            w = max(0.0, bbox.x2 - bbox.x1)
            h = max(0.0, bbox.y2 - bbox.y1)
            cx = bbox.x1 + w / 2.0
            cy = bbox.y1 + h / 2.0
            xywh_list.append([cx, cy, w, h])
            conf_list.append(float(d.confidence))
            cls_list.append(float(d.class_id))

        container = _DetectionContainer(xywh_list, conf_list, cls_list)
        raw_tracks = self._tracker.update(container)

        active_tracks: List[TrackedObject] = []

        if raw_tracks is not None and len(raw_tracks) > 0:
            for row in raw_tracks:
                # Format: [x1, y1, x2, y2, track_id, score, cls, idx]
                x1, y1, x2, y2 = float(row[0]), float(row[1]), float(row[2]), float(row[3])
                track_id = int(row[4])
                score = float(row[5])
                det_idx = int(row[7])

                self._total_tracks_created.add(track_id)

                # Clamp bounding box coordinates to frame boundaries if known
                if w_max is not None:
                    x1 = max(0.0, min(w_max, x1))
                    x2 = max(0.0, min(w_max, x2))
                else:
                    x1 = max(0.0, x1)
                    x2 = max(0.0, x2)

                if h_max is not None:
                    y1 = max(0.0, min(h_max, y1))
                    y2 = max(0.0, min(h_max, y2))
                else:
                    y1 = max(0.0, y1)
                    y2 = max(0.0, y2)

                x2 = max(x1, x2)
                y2 = max(y1, y2)

                # Resolve class label with class consistency
                if 0 <= det_idx < len(detections.detections):
                    matched_det = detections.detections[det_idx]
                    matched_cid = matched_det.class_id
                    matched_cname = matched_det.class_name
                else:
                    matched_cid = int(row[6])
                    matched_cname = f"class_{matched_cid}"

                if track_id not in self._track_classes:
                    self._track_classes[track_id] = (matched_cid, matched_cname)
                    self._track_class_votes[track_id] = {matched_cname: (matched_cid, max(0.5, score))}
                else:
                    # Confidence-weighted class evidence voting to prevent temporary misclassification lock-in
                    votes = self._track_class_votes.setdefault(track_id, {})
                    prev_cid, prev_score = votes.get(matched_cname, (matched_cid, 0.0))
                    votes[matched_cname] = (matched_cid, prev_score + max(0.5, score))

                    best_cname = max(votes, key=lambda k: votes[k][1])
                    best_cid = votes[best_cname][0]
                    self._track_classes[track_id] = (best_cid, best_cname)
                    matched_cid, matched_cname = self._track_classes[track_id]

                # Compute centroid in original video coordinates
                cx = round((x1 + x2) / 2.0, 2)
                cy = round((y1 + y2) / 2.0, 2)

                # Update trajectory history
                if track_id not in self._trajectories:
                    self._trajectories[track_id] = collections.deque(
                        maxlen=self.track_history_length
                    )

                new_pt = TrackPoint(
                    frame_id=frame_id,
                    timestamp=timestamp,
                    cx=cx,
                    cy=cy,
                )
                self._trajectories[track_id].append(new_pt)
                traj_list = list(self._trajectories[track_id])

                # Calculate image-space motion information (NOT physical speed)
                vel_x = 0.0
                vel_y = 0.0
                speed_px_per_sec = 0.0

                if len(traj_list) >= 2:
                    prev_pt = traj_list[-2]
                    vel_x = round(cx - prev_pt.cx, 2)
                    vel_y = round(cy - prev_pt.cy, 2)
                    dt = timestamp - prev_pt.timestamp
                    displacement = float(np.hypot(vel_x, vel_y))
                    if dt > 0.0:
                        speed_px_per_sec = round(displacement / dt, 2)
                    else:
                        speed_px_per_sec = round(displacement * self.frame_rate, 2)

                # Determine track lifecycle state
                state = TrackState.NEW if len(traj_list) == 1 else TrackState.ACTIVE
                self._track_states[track_id] = state

                tracked_obj = TrackedObject(
                    track_id=track_id,
                    class_id=matched_cid,
                    class_name=matched_cname,
                    confidence=round(score, 4),
                    bbox=BoundingBox(
                        x1=round(x1, 2),
                        y1=round(y1, 2),
                        x2=round(x2, 2),
                        y2=round(y2, 2),
                    ),
                    state=state,
                    frame_index=frame_id,
                    timestamp=timestamp,
                    centroid_x=cx,
                    centroid_y=cy,
                    trajectory=traj_list,
                    velocity_x=vel_x,
                    velocity_y=vel_y,
                    speed_pixels_per_second=speed_px_per_sec,
                )
                active_tracks.append(tracked_obj)

        self._update_lost_and_removed_states()

        return FrameTracks(
            frame_id=frame_id,
            timestamp=timestamp,
            active_tracks=active_tracks,
            total_track_count=len(self._total_tracks_created),
        )

    def _update_lost_and_removed_states(self) -> None:
        """Updates internal lifecycle states for tracks in lost or removed pools."""
        for t in getattr(self._tracker, "lost_stracks", []):
            self._track_states[t.track_id] = TrackState.LOST

        for t in getattr(self._tracker, "removed_stracks", []):
            self._track_states[t.track_id] = TrackState.REMOVED

    def get_track_state(self, track_id: int) -> Optional[TrackState]:
        """Returns the current lifecycle state of a track."""
        return self._track_states.get(track_id)

    def reset(self) -> None:
        """Resets all tracking states, internal Kalman filters, and trajectories."""
        self._tracker.reset()
        self._trajectories.clear()
        self._track_classes.clear()
        self._track_class_votes.clear()
        self._track_states.clear()
        self._total_tracks_created.clear()
        logger.info("ByteTrackTracker reset complete.")
