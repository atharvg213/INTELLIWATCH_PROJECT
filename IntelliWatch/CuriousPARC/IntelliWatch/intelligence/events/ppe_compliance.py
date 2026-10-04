import logging
import uuid
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from backend.schemas.events import EventType, IndustrialEvent, SeverityLevel
from backend.schemas.detection import FrameDetections
from backend.schemas.ppe import (
    ComplianceStatus,
    FramePPEAssociation,
    PPEItemState,
    WorkerPPEInventory,
)
from backend.schemas.tracking import FrameTracks, TrackedObject
from configs.settings import get_settings
from intelligence.events.event_detector import BaseEventDetector
from vision.detection.ppe_association import PPEAssociationEngine

logger = logging.getLogger("intelliwatch.ppe_compliance")


class _TrackTemporalState:
    """Internal tracker for temporal violation confirmation and track lifecycle."""

    def __init__(self, track_id: int):
        self.track_id: int = track_id
        self.consecutive_violations: int = 0
        self.missing_observation_counts: Dict[str, int] = {}
        self.last_missing_ppe: List[str] = []
        self.violation_confirmed: bool = False
        self.last_frame_id: int = 0
        self.last_timestamp: float = 0.0


class PPEComplianceEngine(BaseEventDetector):
    """
    Deterministic Industrial PPE Compliance Evaluation and Temporal Confirmation Engine.

    Responsibilities:
      1. Evaluates per-worker inventory against configured mandatory PPE requirements.
      2. Distinguishes MISSING from UNKNOWN (insufficient evidence due to boundary clipping or distance).
      3. Implements lightweight temporal confirmation across frames using persistent track IDs.
      4. Generates structured, explainable IndustrialEvent instances upon confirmed violations.
      5. Automatically cleans up state when tracks disappear from scene.
    """

    def __init__(
        self,
        required_ppe: Optional[List[str]] = None,
        confirmation_frames: Optional[int] = None,
        min_goggles_height: Optional[float] = None,
        min_gloves_height: Optional[float] = None,
        min_general_height: Optional[float] = None,
        boundary_margin_px: float = 5.0,
        association_engine: Optional[PPEAssociationEngine] = None,
        camera_id: str = "cam_01",
    ):
        settings = get_settings()
        self.required_ppe = (
            required_ppe
            if required_ppe is not None
            else getattr(settings, "PPE_REQUIRED_CLASSES", ["Hardhat", "Safety Vest", "Gloves", "Goggles"])
        )
        self.confirmation_frames = (
            confirmation_frames
            if confirmation_frames is not None
            else getattr(settings, "PPE_VIOLATION_CONFIRMATION_FRAMES", 3)
        )
        self.min_goggles_height = (
            min_goggles_height
            if min_goggles_height is not None
            else getattr(settings, "PPE_MIN_PERSON_HEIGHT_GOGGLES", 120.0)
        )
        self.min_gloves_height = (
            min_gloves_height
            if min_gloves_height is not None
            else getattr(settings, "PPE_MIN_PERSON_HEIGHT_GLOVES", 100.0)
        )
        self.min_general_height = (
            min_general_height
            if min_general_height is not None
            else getattr(settings, "PPE_MIN_PERSON_HEIGHT_GENERAL", 50.0)
        )
        self.boundary_margin_px = boundary_margin_px
        self.association_engine = association_engine or PPEAssociationEngine()
        self.camera_id = camera_id

        # Persistent temporal state per track ID
        self._track_states: Dict[int, _TrackTemporalState] = {}

    def reset_state(self) -> None:
        """Clears all historical temporal confirmation buffers."""
        self._track_states.clear()
        logger.debug("Cleared PPE compliance temporal history.")

    def evaluate_worker_compliance(
        self,
        worker: WorkerPPEInventory,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
        temporal_state: Optional[_TrackTemporalState] = None,
    ) -> WorkerPPEInventory:
        """
        Deterministically evaluates compliance for a single worker inventory.

        Distinguishes PRESENT, MISSING, and UNKNOWN:
          - PRESENT: Associated positive detection confirmed.
          - MISSING: Configured-confidence negative evidence or repeated low-confidence/visible-absence evidence.
          - UNKNOWN: Insufficient visual evidence (boundary clipping, small bounding box / distance).
        """
        bbox = worker.bbox
        pw = bbox.width
        ph = bbox.height
        if temporal_state is None:
            temporal_state = self._track_states.setdefault(
                worker.track_id, _TrackTemporalState(worker.track_id)
            )

        # Map detected items by canonical class
        detected_positives: Set[str] = set()
        detected_negatives: Set[str] = set()
        low_confidence_negatives: Set[str] = set()

        for item in worker.items:
            canonical = item.class_name.strip()
            if item.is_negative:
                if item.low_confidence_candidate:
                    low_confidence_negatives.add(canonical)
                else:
                    detected_negatives.add(canonical)
            else:
                detected_positives.add(canonical)

        # Boundary clipping checks (if frame dimensions are known)
        head_truncated = False
        hands_truncated = False

        if frame_width is not None and frame_height is not None:
            # Head truncated if worker box reaches the top edge
            if bbox.y1 <= self.boundary_margin_px:
                head_truncated = True

            # Hands/lower body truncated if worker box reaches bottom or side edges
            if (
                bbox.y2 >= (frame_height - self.boundary_margin_px) or
                bbox.x1 <= self.boundary_margin_px or
                bbox.x2 >= (frame_width - self.boundary_margin_px)
            ):
                hands_truncated = True

        ppe_status: Dict[str, PPEItemState] = {}
        missing_list: List[str] = []
        present_list: List[str] = []
        unknown_list: List[str] = []

        for req in self.required_ppe:
            req_norm = req.strip()

            # 1. Positive detection present
            if req_norm in detected_positives:
                ppe_status[req_norm] = PPEItemState.PRESENT
                present_list.append(req_norm)
                temporal_state.missing_observation_counts[req_norm] = 0
                continue

            # 2. Configured-confidence negative detection (e.g. "NO-Hardhat")
            if req_norm in detected_negatives:
                # NO-* is direct model evidence of absence. Keep the item-level
                # state useful on a single image; event creation remains gated
                # by the independent consecutive-frame confirmation below.
                temporal_state.missing_observation_counts[req_norm] = (
                    temporal_state.missing_observation_counts.get(req_norm, 0) + 1
                )
                ppe_status[req_norm] = PPEItemState.MISSING
                missing_list.append(req_norm)
                continue

            # Low-confidence NO-* predictions are associated only when their
            # geometry strongly fits this worker, then require repeated frames
            # before becoming a missing verdict.
            if req_norm in low_confidence_negatives:
                negative_observations = temporal_state.missing_observation_counts.get(req_norm, 0) + 1
                temporal_state.missing_observation_counts[req_norm] = negative_observations
                if negative_observations >= max(1, self.confirmation_frames):
                    ppe_status[req_norm] = PPEItemState.MISSING
                    missing_list.append(req_norm)
                else:
                    ppe_status[req_norm] = PPEItemState.UNKNOWN
                    unknown_list.append(req_norm)
                continue

            # 3. Not detected: Evaluate visual observability
            is_unknown = False

            # Distance/resolution check
            if ph < self.min_general_height:
                is_unknown = True
            elif req_norm.lower() in ("goggles", "mask") and (ph < self.min_goggles_height or head_truncated):
                is_unknown = True
            elif req_norm.lower() in ("hardhat",) and head_truncated:
                is_unknown = True
            elif req_norm.lower() in ("gloves",) and (ph < self.min_gloves_height or hands_truncated):
                is_unknown = True

            if is_unknown:
                ppe_status[req_norm] = PPEItemState.UNKNOWN
                unknown_list.append(req_norm)
            else:
                # A detector miss in one frame is not proof of absence. Keep
                # the item uncertain until adequate views repeatedly agree.
                observed_absent = temporal_state.missing_observation_counts.get(req_norm, 0) + 1
                temporal_state.missing_observation_counts[req_norm] = observed_absent
                if observed_absent >= max(1, self.confirmation_frames):
                    ppe_status[req_norm] = PPEItemState.MISSING
                    missing_list.append(req_norm)
                else:
                    ppe_status[req_norm] = PPEItemState.UNKNOWN
                    unknown_list.append(req_norm)

            if is_unknown:
                # Occlusion, truncation, or distance breaks a consecutive
                # absence streak; it must not contribute to a missing verdict.
                temporal_state.missing_observation_counts[req_norm] = 0

        # Overall compliance verdict:
        # - Any verified MISSING -> NON_COMPLIANT
        # - Else if any UNKNOWN -> UNKNOWN (insufficient evidence)
        # - Else (all PRESENT) -> COMPLIANT
        if missing_list:
            overall_status = ComplianceStatus.NON_COMPLIANT
            explanation = (
                f"Worker #{worker.track_id} is NON_COMPLIANT: "
                f"Missing required PPE [{', '.join(missing_list)}]."
            )
        elif unknown_list:
            overall_status = ComplianceStatus.UNKNOWN
            explanation = (
                f"Worker #{worker.track_id} compliance is UNKNOWN: "
                f"Insufficient visual evidence for [{', '.join(unknown_list)}] (truncation/resolution)."
            )
        else:
            overall_status = ComplianceStatus.COMPLIANT
            explanation = (
                f"Worker #{worker.track_id} is COMPLIANT: "
                f"All required PPE items [{', '.join(present_list)}] verified PRESENT."
            )

        worker.ppe_status = ppe_status
        worker.compliance_status = overall_status
        worker.missing_ppe = missing_list
        worker.present_ppe = present_list
        worker.unknown_ppe = unknown_list
        worker.explanation = explanation

        return worker

    def process_frame(
        self,
        tracks: Union[FrameTracks, List[TrackedObject]],
        ppe_detections: Union[FrameDetections, List[Any]],
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        frame_width: Optional[int] = None,
        frame_height: Optional[int] = None,
    ) -> Tuple[FramePPEAssociation, List[IndustrialEvent]]:
        """
        Executes end-to-end association, compliance evaluation, and temporal confirmation.

        Returns:
            Tuple of (FramePPEAssociation with evaluated inventories, List of confirmed IndustrialEvent)
        """
        # 1. Run Person-PPE spatial association
        association = self.association_engine.associate(
            tracks=tracks,
            ppe_detections=ppe_detections,
            frame_id=frame_id,
            timestamp=timestamp,
        )

        # Extract frame dimensions if available from schemas
        f_w = frame_width
        f_h = frame_height
        if f_w is None and isinstance(ppe_detections, FrameDetections):
            f_w = ppe_detections.frame_width
            f_h = ppe_detections.frame_height

        current_frame_id = association.frame_id
        current_timestamp = association.timestamp
        active_track_ids: Set[int] = set()

        events: List[IndustrialEvent] = []

        # 2. Evaluate compliance for each worker
        for worker in association.worker_inventories:
            active_track_ids.add(worker.track_id)
            track_id = worker.track_id
            if track_id not in self._track_states:
                self._track_states[track_id] = _TrackTemporalState(track_id)

            state = self._track_states[track_id]
            self.evaluate_worker_compliance(
                worker,
                frame_width=f_w,
                frame_height=f_h,
                temporal_state=state,
            )

            # 3. Temporal Confirmation Logic
            state.last_frame_id = current_frame_id
            state.last_timestamp = current_timestamp

            logger.debug(
                "[PPE_STATE] frame=%s person=%s item_states=%s compliance=%s",
                current_frame_id,
                track_id,
                {name: value.value for name, value in worker.ppe_status.items()},
                worker.compliance_status.value,
            )

            if worker.compliance_status == ComplianceStatus.NON_COMPLIANT:
                absent_confirmed = any(
                    state.missing_observation_counts.get(item_name, 0) >= max(1, self.confirmation_frames)
                    for item_name in worker.missing_ppe
                )
                if absent_confirmed and state.consecutive_violations == 0:
                    # The per-item absence streak has already supplied the
                    # configured temporal confirmation for this violation.
                    state.consecutive_violations = max(0, self.confirmation_frames - 1)
                state.consecutive_violations += 1
                state.last_missing_ppe = list(worker.missing_ppe)

                # Check if threshold reached
                if state.consecutive_violations >= self.confirmation_frames and not state.violation_confirmed:
                    state.violation_confirmed = True
                    event_id = f"evt_ppe_{track_id}_{int(current_timestamp * 1000)}"

                    event = IndustrialEvent(
                        event_id=event_id,
                        event_type=EventType.PPE_VIOLATION,
                        timestamp=current_timestamp,
                        camera_id=self.camera_id,
                        tracked_object_ids=[track_id],
                        severity=SeverityLevel.HIGH,
                        explanation=(
                            f"Worker #{track_id} PPE violation confirmed: "
                            f"missing required gear [{', '.join(worker.missing_ppe)}] "
                            f"persisting for {state.consecutive_violations} consecutive frames."
                        ),
                        metadata={
                            "track_id": track_id,
                            "status": ComplianceStatus.NON_COMPLIANT.value,
                            "missing_ppe": worker.missing_ppe,
                            "detected_ppe": worker.present_ppe,
                            "unknown_ppe": worker.unknown_ppe,
                            "confirmation_frames": self.confirmation_frames,
                            "consecutive_violation_count": state.consecutive_violations,
                        },
                    )
                    events.append(event)
                    logger.warning(
                        f"CONFIRMED PPE VIOLATION on Worker #{track_id}: "
                        f"Missing {worker.missing_ppe} (Frame {current_frame_id})"
                    )

            elif worker.compliance_status == ComplianceStatus.COMPLIANT:
                # Reset violation counter when compliant
                state.consecutive_violations = 0
                state.violation_confirmed = False

            elif worker.compliance_status == ComplianceStatus.UNKNOWN:
                # If evidence is temporarily unknown, reset consecutive violation sequence
                state.consecutive_violations = 0

        # 4. Clean up state for disappeared tracks
        dead_tracks = [tid for tid in self._track_states if tid not in active_track_ids]
        for tid in dead_tracks:
            del self._track_states[tid]
            logger.debug(f"Purged temporal compliance history for departed track #{tid}")

        return association, events

    def evaluate(
        self,
        tracks: FrameTracks,
        context: Dict[str, Any],
    ) -> List[IndustrialEvent]:
        """
        Implementation of BaseEventDetector.evaluate().
        Expects 'ppe_detections' in context.
        """
        ppe_dets = context.get("ppe_detections")
        if ppe_dets is None:
            logger.warning("No 'ppe_detections' found in context for PPEComplianceEngine.")
            return []

        frame_w = context.get("frame_width")
        frame_h = context.get("frame_height")

        _, events = self.process_frame(
            tracks=tracks,
            ppe_detections=ppe_dets,
            frame_width=frame_w,
            frame_height=frame_h,
        )
        return events
