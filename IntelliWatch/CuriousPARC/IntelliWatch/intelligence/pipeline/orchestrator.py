"""
intelligence/pipeline/orchestrator.py
End-to-End Orchestration Layer for IntelliWatch Industrial Scene Understanding.

Sequentially connects all perception and intelligence layers:
  Frame Preprocessing
  -> YOLO11n Object Detection
  -> ByteTrack Multi-Object Tracking
  -> PPE Detection & Spatial Association & Compliance Evaluation
  -> Zone Spatial Reasoning & Contact-Point Tracking
  -> Monocular Relative Depth Perception
  -> Behavior & Temporal Analysis
  -> Scene Graph Construction & Situational Awareness
  -> Risk & Event Reasoning
  -> Predictive & Early-Warning Intelligence
  -> Incident Evidence Logging
  -> Frame Assessment Assembly & Telemetry
"""
import logging
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from backend.schemas.assessment import FrameAssessment
from backend.schemas.behavior import BehaviorState
from backend.schemas.depth import DepthStatistics, ObjectDepth
from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from backend.schemas.events import IndustrialEvent
from backend.schemas.evidence import IncidentRecord
from backend.schemas.ppe import FramePPEAssociation, WorkerPPEInventory
from backend.schemas.prediction import EarlyWarningIndicator, FramePredictionAssessment
from backend.schemas.risk import (
    EventLifecycleState,
    FrameRiskAssessment,
    RiskEvent,
    RiskEventType,
    RiskLevel,
)
from backend.schemas.scene_graph import FrameScene, SituationalSummary, enrich_scene_graph
from backend.schemas.tracking import FrameTracks, TrackPoint, TrackedObject
from backend.schemas.zones import FrameZoneOccupancy, RestrictedZone, ZoneMembership

from backend.services.assessment_store import get_assessment_store
from backend.services.incident_store import get_incident_store
from backend.services.prediction_store import get_prediction_store
from backend.services.risk_store import get_risk_store
from backend.services.scene_store import get_scene_store
from backend.services.temporal_log import get_temporal_log

from configs.settings import get_settings
from intelligence.behavior.behavior_engine import BehaviorEngine
from intelligence.events.ppe_compliance import PPEComplianceEngine
from intelligence.prediction.engine import PredictionEngine
from intelligence.risk.engine import RiskEngine
from intelligence.scene_graph.builder import SceneGraphBuilder
from intelligence.zones.zone_engine import ZoneEngine
from vision.detection.base import BaseDetector
from vision.detection.industrial_detector import IndustrialDetector
from vision.detection.merged_detector import MultiDetectorAggregator
from vision.detection.ppe_association import PPEAssociationEngine
from vision.detection.ppe_detector import PPEDetector
from vision.detection.yolo_detector import YOLODetector
from vision.tracking.bytetrack_tracker import ByteTrackTracker
from vision.tracking.visualizer import TrackingVisualizer

logger = logging.getLogger("intelliwatch.orchestrator")


class EndToEndPipelineOrchestrator:
    """
    End-to-End Orchestrator executing the complete IntelliWatch intelligence pipeline.
    """

    def __init__(
        self,
        device: Optional[str] = None,
        camera_id: str = "cam_01",
        enable_ppe_model: bool = True,
        enable_industrial_model: Optional[bool] = None,
        enable_depth_model: bool = False,  # Off by default on CPU for fast execution
        detector: Optional[Union[YOLODetector, MultiDetectorAggregator]] = None,
        industrial_detector: Optional[IndustrialDetector] = None,
        tracker: Optional[ByteTrackTracker] = None,
        ppe_detector: Optional[PPEDetector] = None,
        zone_engine: Optional[ZoneEngine] = None,
        behavior_engine: Optional[BehaviorEngine] = None,
        scene_builder: Optional[SceneGraphBuilder] = None,
        risk_engine: Optional[RiskEngine] = None,
        prediction_engine: Optional[PredictionEngine] = None,
        visualizer: Optional[TrackingVisualizer] = None,
    ):
        self.settings = get_settings()
        self.camera_id = camera_id

        # Determine compute device with hardware execution verification and CPU fallback
        from vision.utils.device import resolve_device
        self.device_str = resolve_device(device)

        logger.info(f"Initializing EndToEndPipelineOrchestrator on device: {self.device_str}")

        # 1. Detection & Tracking
        self.enable_industrial_model = (
            enable_industrial_model
            if enable_industrial_model is not None
            else getattr(self.settings, "INDUSTRIAL_DETECTION_ENABLED", True)
        )
        self._custom_detector_provided = detector is not None
        self._yolo_detector = detector
        self._industrial_detector = industrial_detector
        self._multi_detector: Optional[MultiDetectorAggregator] = None
        self._tracker = tracker or ByteTrackTracker()

        # 2. PPE Perceivers & Reasoners
        self.enable_ppe_model = enable_ppe_model
        self._ppe_detector = ppe_detector
        self._ppe_association_engine = PPEAssociationEngine()
        self._ppe_compliance_engine = PPEComplianceEngine(
            required_ppe=self.settings.PPE_REQUIRED_CLASSES,
            confirmation_frames=self.settings.PPE_VIOLATION_CONFIRMATION_FRAMES,
            camera_id=self.camera_id,
        )

        # 3. Zone Spatial Reasoner
        if zone_engine is not None:
            self._zone_engine = zone_engine
        else:
            try:
                from backend.services.zone_service import get_zone_engine
                self._zone_engine = get_zone_engine()
            except Exception:
                self._zone_engine = ZoneEngine(
                    config_path=self.settings.ZONE_CONFIG_PATH,
                    confirmation_frames=self.settings.ZONE_ENTRY_CONFIRMATION_FRAMES,
                    default_max_dwell_seconds=self.settings.ZONE_MAX_DWELL_SECONDS,
                    camera_id=self.camera_id,
                )

        # 4. Depth Estimator (Optional on CPU)
        self.enable_depth_model = enable_depth_model
        self._depth_estimator = None

        # 5. Behavior Engine
        self._behavior_engine = behavior_engine or BehaviorEngine(
            history_seconds=5.0,
            confirmation_frames=3,
        )

        # 6. Scene Graph Builder
        self._scene_builder = scene_builder or SceneGraphBuilder(
            near_distance_threshold=self.settings.SCENE_NEAR_DISTANCE_THRESHOLD,
            far_distance_threshold=self.settings.SCENE_FAR_DISTANCE_THRESHOLD,
            confirmation_frames=self.settings.SCENE_RELATION_CONFIRMATION_FRAMES,
            end_confirmation_frames=self.settings.SCENE_RELATION_END_CONFIRMATION_FRAMES,
            approaching_enabled=self.settings.SCENE_APPROACHING_ENABLED,
            depth_relation_enabled=self.settings.SCENE_DEPTH_RELATION_ENABLED,
            depth_near_threshold=self.settings.SCENE_DEPTH_NEAR_THRESHOLD,
            camera_id=self.camera_id,
        )

        # 7. Risk & Event Engine
        self._risk_engine = risk_engine or RiskEngine(
            confirmation_frames=self.settings.RISK_EVENT_CONFIRMATION_FRAMES,
            end_confirmation_frames=self.settings.RISK_EVENT_END_CONFIRMATION_FRAMES,
            cooldown_frames=self.settings.RISK_EVENT_COOLDOWN_FRAMES,
            compound_enabled=self.settings.RISK_COMPOUND_EVENT_ENABLED,
            camera_id=self.camera_id,
        )

        # 8. Predictive & Early-Warning Engine
        self._prediction_engine = prediction_engine or PredictionEngine(
            confirmation_frames=self.settings.PREDICTION_CONFIRMATION_FRAMES,
            end_confirmation_frames=self.settings.PREDICTION_END_CONFIRMATION_FRAMES,
            horizon_frames=self.settings.PREDICTION_HORIZON_FRAMES,
            camera_id=self.camera_id,
        )

        # 9. Visualizer
        self._visualizer = visualizer or TrackingVisualizer()

        # Telemetry & Performance counters
        self.total_frames_processed = 0
        self.total_processing_time_ms = 0.0
        self._recorded_incident_ids = set()
        self._track_to_display_id: Dict[int, int] = {}
        self._incident_id_map: Dict[str, str] = {}

    def get_display_person_id(self, track_id: int) -> int:
        """Returns stable, monotonic 1-based display ID for a persistent track within this job."""
        if track_id not in self._track_to_display_id:
            self._track_to_display_id[track_id] = len(self._track_to_display_id) + 1
        return self._track_to_display_id[track_id]

    def _get_detector(self) -> Union[BaseDetector, YOLODetector, MultiDetectorAggregator]:
        """Lazy-loads object detector (either custom, multi-detector, or YOLO)."""
        if self._custom_detector_provided and self._yolo_detector is not None:
            return self._yolo_detector

        if self._yolo_detector is None:
            self._yolo_detector = YOLODetector(
                model_path=self.settings.DETECTION_MODEL_PATH,
                confidence_threshold=self.settings.DETECTION_CONFIDENCE_THRESHOLD,
                iou_threshold=self.settings.DETECTION_IOU_THRESHOLD,
                device=self.device_str,
            )

        if not self.enable_industrial_model:
            return self._yolo_detector

        if self._multi_detector is None:
            if self._industrial_detector is None:
                ind_path = Path(self.settings.INDUSTRIAL_MODEL_PATH)
                if not ind_path.exists():
                    fallback = Path("weights") / ind_path.name
                    if fallback.exists():
                        ind_path = fallback
                if ind_path.exists():
                    try:
                        self._industrial_detector = IndustrialDetector(
                            model_path=str(ind_path),
                            confidence_threshold=self.settings.INDUSTRIAL_CONFIDENCE_THRESHOLD,
                            iou_threshold=self.settings.INDUSTRIAL_IOU_THRESHOLD,
                            vocabulary=self.settings.INDUSTRIAL_VOCABULARY,
                            device=self.device_str,
                        )
                    except Exception as e:
                        logger.warning(f"Could not load IndustrialDetector from {ind_path}: {e}")
                        self._industrial_detector = None
                else:
                    logger.warning(f"Industrial model weights not found at {ind_path}. Running general detector only.")

            self._multi_detector = MultiDetectorAggregator(
                general_detector=self._yolo_detector,
                industrial_detector=self._industrial_detector,
                cross_model_iou_thresh=self.settings.INDUSTRIAL_IOU_THRESHOLD,
                filter_irrelevant_coco=self.settings.FILTER_IRRELEVANT_COCO_CLASSES,
                irrelevant_classes=self.settings.IRRELEVANT_COCO_CLASSES,
            )

        return self._multi_detector

    def _get_ppe_detector(self) -> Optional[PPEDetector]:
        """Lazy-loads PPE detector if enabled."""
        if not self.enable_ppe_model:
            return None
        if self._ppe_detector is None:
            ppe_path = Path(self.settings.PPE_MODEL_PATH)
            if not ppe_path.exists():
                project_root = Path(__file__).resolve().parent.parent.parent
                cand1 = project_root / ppe_path
                if cand1.exists():
                    ppe_path = cand1
                else:
                    cand2 = project_root / "weights" / ppe_path.name
                    if cand2.exists():
                        ppe_path = cand2
            if ppe_path.exists():
                self._ppe_detector = PPEDetector(
                    model_path=str(ppe_path),
                    confidence_threshold=self.settings.PPE_CONFIDENCE_THRESHOLD,
                    iou_threshold=self.settings.PPE_IOU_THRESHOLD,
                    device=self.device_str,
                )
            else:
                logger.warning(f"PPE model weights not found at {ppe_path}. Skipping model inference.")
        return self._ppe_detector

    def process_frame(
        self,
        frame: np.ndarray,
        frame_id: int = 0,
        timestamp: float = 0.0,
        manual_detections: Optional[List[DetectionResult]] = None,
        manual_ppe_detections: Optional[List[DetectionResult]] = None,
        show_zones: bool = True,
        is_single_image: bool = False,
        imgsz: Optional[int] = None,
    ) -> Tuple[FrameAssessment, np.ndarray]:
        """
        Executes end-to-end perception and reasoning for a single video frame.

        Args:
            frame: Input image in BGR format (np.ndarray of shape (H, W, 3)).
            frame_id: Frame sequence index.
            timestamp: Video stream timestamp in seconds.
            manual_detections: Optional pre-computed detections (e.g. for deterministic testing).
            manual_ppe_detections: Optional pre-computed PPE detections.
            show_zones: Whether to render configured safety zones.
            is_single_image: Whether processing a single static image.
            imgsz: Optional inference resolution.

        Returns:
            Tuple of (FrameAssessment schema, Annotated frame image ndarray).
        """
        if frame is None or frame.size == 0:
            raise ValueError("Input frame cannot be empty or None.")

        t_start = time.perf_counter()
        h, w = frame.shape[:2]

        # ------------------------------------------------------------------
        # 1. Object Detection (YOLO11n)
        # ------------------------------------------------------------------
        if manual_detections is not None:
            detections = manual_detections
        else:
            detector = self._get_detector()
            det_imgsz = imgsz
            if det_imgsz is None and is_single_image and max(w, h) >= 1000:
                det_imgsz = 960
            det_result = detector.detect(frame, imgsz=det_imgsz) if det_imgsz else detector.detect(frame)
            detections = det_result.detections if hasattr(det_result, "detections") else det_result

        # ------------------------------------------------------------------
        # 2. Multi-Object Tracking (ByteTrack)
        # ------------------------------------------------------------------
        if isinstance(detections, FrameDetections):
            frame_dets = detections
        else:
            frame_dets = FrameDetections(
                frame_id=frame_id,
                timestamp=timestamp,
                detections=detections,
                frame_width=w,
                frame_height=h,
            )

        frame_tracks: FrameTracks = self._tracker.update(
            detections=frame_dets,
            frame=frame,
        )
        active_tracks = frame_tracks.active_tracks

        # Assign stable 1-based display IDs to person tracks
        for t in active_tracks:
            if (t.class_name or "").lower() in ("person", "worker", "operator"):
                disp_id = self.get_display_person_id(t.track_id)
                t.metadata["display_id"] = disp_id
                t.metadata["display_label"] = f"Person #{disp_id}"

        # ------------------------------------------------------------------
        # 3. PPE Detection & Spatial Association & Compliance
        # ------------------------------------------------------------------
        person_tracks = [t for t in active_tracks if (t.class_name or "").lower() in ("person", "worker", "operator")]
        ppe_detections = []

        if manual_ppe_detections is not None:
            ppe_detections = manual_ppe_detections
        elif self.enable_ppe_model:
            ppe_det_model = self._get_ppe_detector()
            if ppe_det_model is not None and len(person_tracks) > 0:
                # Keep a small candidate band below the model threshold.
                # Association requires a strong anatomical fit; low-confidence
                # negative candidates additionally need repeated frames in the
                # compliance engine before they count as missing PPE.
                candidate_floor = min(0.10, float(ppe_det_model.confidence_threshold))
                raw_ppe = ppe_det_model.detect(frame, candidate_floor=candidate_floor)
                ppe_detections = list(raw_ppe.detections if hasattr(raw_ppe, "detections") else raw_ppe)

                # Localized Worker-ROI inference for small/occluded workers to boost gear recall
                if getattr(self.settings, "PPE_LOCALIZED_INFERENCE_ENABLED", True):
                    from vision.detection.coordinates import compute_box_iou
                    for p_track in person_tracks:
                        box = p_track.bbox
                        bw = box.x2 - box.x1
                        bh = box.y2 - box.y1
                        pad_x = bw * 0.15
                        pad_y = bh * 0.15
                        x1 = max(0, int(box.x1 - pad_x))
                        y1 = max(0, int(box.y1 - pad_y))
                        x2 = min(w, int(box.x2 + pad_x))
                        y2 = min(h, int(box.y2 + pad_y))

                        if (x2 - x1) >= 20 and (y2 - y1) >= 20:
                            worker_crop = frame[y1:y2, x1:x2]
                            crop_res = ppe_det_model.detect(worker_crop, candidate_floor=candidate_floor)
                            crop_dets = crop_res.detections if hasattr(crop_res, "detections") else crop_res

                            for c_det in crop_dets:
                                g_bbox = BoundingBox(
                                    x1=round(float(x1 + c_det.bbox.x1), 2),
                                    y1=round(float(y1 + c_det.bbox.y1), 2),
                                    x2=round(float(x1 + c_det.bbox.x2), 2),
                                    y2=round(float(y1 + c_det.bbox.y2), 2),
                                )
                                duplicate_indexes = [
                                    idx for idx, detection in enumerate(ppe_detections)
                                    if detection.class_name.lower() == c_det.class_name.lower()
                                    and compute_box_iou(detection.bbox, g_bbox) >= 0.40
                                    and (
                                        (detection.metadata or {}).get("crop_parent_track") is None
                                        or (detection.metadata or {}).get("crop_parent_track") == p_track.track_id
                                    )
                                ]
                                if any(
                                    ppe_detections[idx].confidence >= c_det.confidence
                                    for idx in duplicate_indexes
                                ):
                                    continue

                                # An ROI prediction replaces only weaker
                                # overlapping full-frame candidates. Keep the
                                # stronger box and its worker ownership.
                                if duplicate_indexes:
                                    duplicate_indexes = set(duplicate_indexes)
                                    ppe_detections = [
                                        detection for idx, detection in enumerate(ppe_detections)
                                        if idx not in duplicate_indexes
                                    ]

                                proj_det = DetectionResult(
                                    class_id=c_det.class_id,
                                    class_name=c_det.class_name,
                                    confidence=c_det.confidence,
                                    bbox=g_bbox,
                                    class_group=c_det.class_group,
                                    source_model="yolo11n_ppe_roi",
                                    metadata={
                                        **(c_det.metadata or {}),
                                        "crop_parent_track": p_track.track_id,
                                    },
                                )
                                ppe_detections.append(proj_det)

        if getattr(self.settings, "DETECTION_DEBUG", False) or logger.isEnabledFor(logging.DEBUG):
            logger.info(
                f"[DETECTION_DIAGNOSTICS] Frame {frame_id}: "
                f"raw_dets={len(frame_dets.detections)}, "
                f"active_tracks={len(active_tracks)}, "
                f"person_tracks={len(person_tracks)}, "
                f"ppe_dets={len(ppe_detections)}"
            )

        # Association & Compliance evaluation
        ppe_association, ppe_events = self._ppe_compliance_engine.process_frame(
            tracks=active_tracks,
            ppe_detections=ppe_detections,
            frame_id=frame_id,
            timestamp=timestamp,
            frame_width=w,
            frame_height=h,
        )
        worker_inventories = ppe_association.worker_inventories
        for inv in worker_inventories:
            disp_id = self.get_display_person_id(inv.track_id)
            inv.metadata["display_id"] = disp_id
            inv.metadata["display_label"] = f"Person #{disp_id}"
            if inv.explanation:
                inv.explanation = inv.explanation.replace(f"Worker #{inv.track_id}", f"Person #{disp_id}").replace(f"Track #{inv.track_id}", f"Person #{disp_id}")

        non_compliant_count = sum(
            1 for inv in worker_inventories
            if getattr(inv, "compliance_status", None) == "NON_COMPLIANT"
            or str(getattr(inv, "compliance_status", "")).endswith("NON_COMPLIANT")
        )

        # ------------------------------------------------------------------
        # 4. Zone Spatial Reasoning
        # ------------------------------------------------------------------
        zone_memberships, zone_events = self._zone_engine.process_frame(
            tracks=frame_tracks,
            timestamp=timestamp,
            frame_id=frame_id,
        )
        occupied_zones_count = len({m.zone_id for m in zone_memberships if m.is_inside})
        zone_occupancy = FrameZoneOccupancy(
            frame_id=frame_id,
            timestamp=timestamp,
            memberships=zone_memberships,
        )

        # ------------------------------------------------------------------
        # 5. Relative Depth Perception (Conservative / Explainable)
        # ------------------------------------------------------------------
        # Note: Non-metric relative depth. Higher values = closer to camera.
        object_depths: List[ObjectDepth] = []
        depth_stats = None

        # ------------------------------------------------------------------
        # 6. Behavior & Temporal Analysis
        # ------------------------------------------------------------------
        # Collect model-assisted fall detections from PPE detector (class 0 'Fall-Detected')
        fall_track_ids: Set[int] = set()
        for inv in worker_inventories:
            for item in inv.items:
                if (item.class_name or "").lower() in ("fall-detected", "fall") and item.confidence >= 0.35:
                    fall_track_ids.add(inv.track_id)

        for unassoc in ppe_association.unassociated_ppe:
            if (unassoc.class_name or "").lower() in ("fall-detected", "fall") and unassoc.confidence >= 0.35:
                u_area = unassoc.bbox.width * unassoc.bbox.height
                if u_area > 0:
                    for pt in person_tracks:
                        inter = PPEAssociationEngine.compute_box_intersection_area(pt.bbox, unassoc.bbox)
                        if (inter / u_area) >= 0.20:
                            fall_track_ids.add(pt.track_id)

        behavior_states, behavior_events = self._behavior_engine.process(
            frame_tracks,
            track_depth_map=None,
            external_fall_tracks=fall_track_ids,
        )

        # ------------------------------------------------------------------
        # 7. Scene Graph Construction & Situational Awareness
        # ------------------------------------------------------------------
        configured_zones = list(self._zone_engine.zones.values())
        frame_scene: FrameScene = self._scene_builder.build(
            frame_tracks=frame_tracks,
            ppe_association=ppe_association,
            zone_occupancy=zone_occupancy,
            behavior_states=behavior_states,
            configured_zones=configured_zones,
            object_depths=object_depths,
        )
        situational_summary = frame_scene.summary

        # ------------------------------------------------------------------
        # 8. Safety Risk & Event Reasoning
        # ------------------------------------------------------------------
        risk_assessment: FrameRiskAssessment = self._risk_engine.evaluate_frame(
            scene=frame_scene,
            ppe_association=ppe_association,
            zone_occupancy=zone_occupancy,
            behavior_states=behavior_states,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        active_events = risk_assessment.active_events
        risk_summary = risk_assessment.risk_summary
        highest_risk_level = risk_summary.max_risk_level
        highest_risk_score = risk_summary.max_risk_score

        # ------------------------------------------------------------------
        # 9. Predictive & Early-Warning Intelligence
        # ------------------------------------------------------------------
        prediction_assessment: FramePredictionAssessment = self._prediction_engine.evaluate_frame(
            frame_tracks=frame_tracks,
            scene=frame_scene,
            risk_assessment=risk_assessment,
            behavior_states=behavior_states,
            zones=configured_zones,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        active_early_warnings = prediction_assessment.active_indicators

        # ------------------------------------------------------------------
        # 10. Incident Evidence Logging
        # ------------------------------------------------------------------
        new_incident_ids: List[str] = []
        incident_store = get_incident_store()

        # Capture evidence whenever an event is newly confirmed or active
        for ev in active_events:
            ev_state = getattr(ev, "lifecycle_state", EventLifecycleState.CONFIRMED)
            # Only record confirmed or active events
            if ev_state not in (EventLifecycleState.CONFIRMED, EventLifecycleState.ACTIVE):
                continue

            ev_id = ev.event_id
            if ev_id not in self._recorded_incident_ids:
                # Find involved tracks
                involved_tracks = []
                for ent_id in ev.involved_entities:
                    if ent_id.startswith("person_") or ent_id.startswith("track_"):
                        try:
                            tid = int(ent_id.split("_")[1])
                            involved_tracks.append(tid)
                        except (ValueError, IndexError):
                            pass

                # Extract location/zone if available
                loc_zone = None
                for ent_id in ev.involved_entities:
                    if "zone" in ent_id:
                        loc_zone = ent_id
                        break

                # Extract scene relationships relevant to involved entities
                relevant_rels = []
                if frame_scene:
                    for rel in frame_scene.relationships:
                        if rel.source_node_id in ev.involved_entities or rel.target_node_id in ev.involved_entities:
                            rel_type_val = rel.relation_type.value if hasattr(rel.relation_type, "value") else str(rel.relation_type)
                            relevant_rels.append({
                                "source": rel.source_node_id,
                                "relation": rel_type_val,
                                "target": rel.target_node_id,
                                "evidence": rel.evidence,
                            })

                # Format human-friendly explanation with stable Person #X labels
                exp = ev.explanation
                for tid in involved_tracks:
                    disp_id = self.get_display_person_id(tid)
                    exp = exp.replace(f"Worker #{tid}", f"Person #{disp_id}").replace(f"Track #{tid}", f"Person #{disp_id}")

                inc_id = f"inc_{uuid.uuid4().hex[:8]}"
                self._incident_id_map[ev_id] = inc_id

                incident_record = IncidentRecord(
                    incident_id=inc_id,
                    timestamp=timestamp,
                    frame_id=frame_id,
                    camera_id=self.camera_id,
                    event_type=ev.event_type,
                    risk_level=ev.risk_level,
                    risk_score=ev.risk_score,
                    lifecycle_state=ev_state,
                    involved_track_ids=involved_tracks,
                    involved_entity_ids=ev.involved_entities,
                    location_or_zone=loc_zone,
                    risk_factors=ev.risk_factors,
                    explanation=exp,
                    scene_relationships=relevant_rels,
                    early_warnings=active_early_warnings,
                    metadata={
                        "frame_timestamp": timestamp,
                        "source_event_id": ev.event_id,
                        "first_frame_id": frame_id,
                        "last_frame_id": frame_id,
                        "duration_seconds": 0.0,
                        "occurrence_count": 1,
                    },
                )

                # Save incident with current raw frame as evidence snapshot
                incident_store.record_incident(incident_record, frame_image=frame)
                new_incident_ids.append(incident_record.incident_id)
                self._recorded_incident_ids.add(ev_id)
            else:
                # Update ongoing incident duration and occurrence count
                existing_inc_id = self._incident_id_map.get(ev_id)
                if existing_inc_id:
                    existing_inc = incident_store.get_incident(existing_inc_id)
                    if existing_inc:
                        existing_inc.lifecycle_state = ev_state
                        existing_inc.metadata["last_frame_id"] = frame_id
                        existing_inc.metadata["duration_seconds"] = round(timestamp - existing_inc.timestamp, 2)
                        existing_inc.metadata["occurrence_count"] = existing_inc.metadata.get("occurrence_count", 1) + 1

        # ------------------------------------------------------------------
        # Step 15 — Temporal Intelligence log update
        # ------------------------------------------------------------------
        temporal_log = get_temporal_log()
        # Assemble a preliminary assessment with current data for the log
        _prelim = FrameAssessment(
            frame_id=frame_id, timestamp=timestamp, camera_id=self.camera_id,
            frame_width=w, frame_height=h, processing_time_ms=0.0,
            tracks=active_tracks, behavior_states=behavior_states,
            scene=frame_scene, risk_assessment=risk_assessment,
            active_events=active_events, highest_risk_level=highest_risk_level,
            prediction_assessment=prediction_assessment,
            active_early_warnings=active_early_warnings,
            new_incident_ids=new_incident_ids,
        )
        temporal_events = temporal_log.update(_prelim)
        entity_profiles = temporal_log.get_all_entity_profiles(_prelim)

        t_end = time.perf_counter()
        elapsed_ms = (t_end - t_start) * 1000.0

        self.total_frames_processed += 1
        self.total_processing_time_ms += elapsed_ms

        # Assemble comprehensive assessment schema
        assessment = FrameAssessment(
            frame_id=frame_id,
            timestamp=timestamp,
            camera_id=self.camera_id,
            frame_width=w,
            frame_height=h,
            processing_time_ms=elapsed_ms,
            detections=detections,
            ppe_detections=ppe_detections,
            tracks=active_tracks,
            total_active_tracks=len(active_tracks),
            worker_inventories=worker_inventories,
            non_compliant_workers_count=non_compliant_count,
            zone_memberships=zone_memberships,
            occupied_zones_count=occupied_zones_count,
            depth_statistics=depth_stats,
            object_depths=object_depths,
            behavior_states=behavior_states,
            scene=frame_scene,
            situational_summary=situational_summary,
            risk_assessment=risk_assessment,
            active_events=active_events,
            highest_risk_level=highest_risk_level,
            highest_risk_score=highest_risk_score,
            prediction_assessment=prediction_assessment,
            active_early_warnings=active_early_warnings,
            new_incident_ids=new_incident_ids,
            temporal_events=temporal_events,
            entity_profiles=entity_profiles,
            metadata={
                "frame_width": w,
                "frame_height": h,
                "aspect_ratio": round(w / max(1, h), 3),
                "is_single_image": is_single_image,
                "raw_detections_count": len(detections) if detections else 0,
                "raw_person_count": getattr(det_result, "metadata", {}).get("raw_person_count", len(person_tracks)) if 'det_result' in locals() and hasattr(det_result, "metadata") else len(person_tracks),
                "filtered_person_count": getattr(det_result, "metadata", {}).get("filtered_person_count", len(person_tracks)) if 'det_result' in locals() and hasattr(det_result, "metadata") else len(person_tracks),
                "suppressed_person_detections": getattr(det_result, "metadata", {}).get("suppressed_person_detections", []) if 'det_result' in locals() and hasattr(det_result, "metadata") else [],
                "active_tracks_count": len(active_tracks),
                "person_tracks_count": len(person_tracks),
            },
        )

        # Enrich scene graph with canonical entities and risk reasoning
        enrich_scene_graph(
            scene=frame_scene,
            risk_assessment=risk_assessment,
            is_live=not is_single_image,
            source_mode="LIVE" if not is_single_image else "LATEST SCENE SNAPSHOT",
        )

        # Update in-memory singleton stores for real-time dashboard endpoints
        get_assessment_store().set_current_assessment(assessment)
        get_scene_store().set_current_scene(frame_scene)
        get_risk_store().set_current_assessment(risk_assessment)
        get_prediction_store().set_current_assessment(prediction_assessment)

        # Step 22 — Centralized Safety Alert Generation & Evidence Capture
        try:
            from intelligence.alerting import get_alert_engine
            get_alert_engine().process_assessment(
                assessment=assessment,
                frame_image=frame,
                camera_name=f"Camera {self.camera_id}",
            )
        except Exception as alert_err:
            logger.warning(f"Alert processing error on camera [{self.camera_id}]: {alert_err}")

        # ------------------------------------------------------------------
        # 12. Visualization Overlay
        # ------------------------------------------------------------------
        annotated_frame = self._visualizer.draw_tracks(
            image=frame,
            tracks=frame_tracks,
            show_trajectories=not is_single_image,
            show_labels=True,
            show_conf=True,
            show_speed=not is_single_image,
            show_banner=not is_single_image,
            worker_inventories=worker_inventories,
            show_ppe=True,
            zones=configured_zones if show_zones else None,
            zone_memberships=zone_memberships if show_zones else None,
            show_zones=show_zones,
        )

        # Overlay Scene Graph relationships & Situational HUD (only if active entities exist)
        if len(active_tracks) > 0:
            annotated_frame = self._visualizer.draw_scene_graph(
                frame=annotated_frame,
                scene=frame_scene,
                draw_hud=True,
                draw_edges=True,
            )

            # Overlay Risk Assessment & Event Alert HUD
            annotated_frame = self._visualizer.draw_risk_assessment(
                frame=annotated_frame,
                assessment=risk_assessment,
                draw_hud=True,
            )

        # Overlay Early-Warning Indicator Badges (if any active)
        if active_early_warnings:
            self._draw_early_warnings(annotated_frame, active_early_warnings)

        return assessment, annotated_frame

    def _draw_early_warnings(
        self,
        frame: np.ndarray,
        indicators: List[EarlyWarningIndicator],
    ) -> None:
        """Overlays compact early-warning badges at bottom-right corner."""
        h, w = frame.shape[:2]
        lines = ["EARLY WARNINGS:"]
        for ind in indicators[:2]:
            ind_type = ind.indicator_type.value if hasattr(ind.indicator_type, "value") else str(ind.indicator_type)
            sev = ind.severity.value if hasattr(ind.severity, "value") else str(ind.severity)
            lines.append(f" ! [{sev}] {ind_type}")

        box_w = 300
        box_h = 16 + len(lines) * 16
        bx1 = max(0, w - box_w - 12)
        by1 = max(0, h - box_h - 12)
        bx2 = min(w - 1, bx1 + box_w)
        by2 = min(h - 1, by1 + box_h)

        overlay = frame.copy()
        cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
        cv2.rectangle(frame, (bx1, by1), (bx2, by2), (0, 165, 255), 1)

        for idx, line in enumerate(lines):
            ly = by1 + 14 + idx * 16
            color = (0, 165, 255) if idx == 0 else (240, 240, 240)
            cv2.putText(
                frame, line, (bx1 + 8, ly),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                color, 1, cv2.LINE_AA,
            )

    def get_performance_stats(self) -> Dict[str, Any]:
        """Returns measured performance statistics."""
        avg_ms = (
            self.total_processing_time_ms / max(1, self.total_frames_processed)
            if self.total_frames_processed > 0 else 0.0
        )
        fps = 1000.0 / avg_ms if avg_ms > 0 else 0.0
        return {
            "device": self.device_str,
            "total_frames_processed": self.total_frames_processed,
            "avg_latency_ms": round(avg_ms, 2),
            "processing_fps": round(fps, 1),
            "ppe_model_enabled": self.enable_ppe_model,
            "depth_model_enabled": self.enable_depth_model,
        }

    def reset(self) -> None:
        """Resets state across all stateful components."""
        self._tracker.reset()
        self._behavior_engine.reset()
        self._scene_builder.reset()
        self._risk_engine.reset()
        self._prediction_engine.reset()
        get_temporal_log().reset()
        self._recorded_incident_ids.clear()
        self._track_to_display_id.clear()
        self._incident_id_map.clear()
        self.total_frames_processed = 0
        self.total_processing_time_ms = 0.0

        # Reset zone and PPE compliance engine states to prevent cross-job state leakage
        if hasattr(self._zone_engine, "reset_state"):
            self._zone_engine.reset_state()
        if hasattr(self._ppe_compliance_engine, "reset_state"):
            self._ppe_compliance_engine.reset_state()

        try:
            from intelligence.alerting import get_alert_engine
            get_alert_engine().reset()
        except Exception:
            pass

        try:
            get_incident_store().reset()
        except Exception:
            pass
