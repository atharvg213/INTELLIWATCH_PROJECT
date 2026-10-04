"""
vision/detection/merged_detector.py
Multi-Detector Aggregator & Cross-Model Deduplication Engine (Step 6 & Step 11).

Combines:
  1. YOLODetector (YOLO11n): High-speed detection of workers and standard COCO entities.
  2. IndustrialDetector (YOLO-World): Open-vocabulary detection of industrial assets
     (forklifts, industrial vehicles, machinery, robotic arms, conveyors, pallets, safety barriers, electrical cabinets).
  3. Domain Filter: Rejects nonsensical COCO classes (e.g. boat, sports ball, bench) in industrial environments.
  4. Cross-Model IoU Merging & Priority Resolution: Resolves overlapping detections between detectors.
"""
import logging
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np

from backend.schemas.detection import BoundingBox, ClassGroup, DetectionResult, FrameDetections, infer_class_group
from configs.settings import get_settings
from vision.detection.base import BaseDetector
from vision.detection.industrial_detector import IndustrialDetector
from vision.detection.yolo_detector import YOLODetector
from vision.preprocessing.frame import FrameData

logger = logging.getLogger("intelliwatch.merged_detector")


from vision.detection.coordinates import compute_box_iomin, compute_box_iou


class MultiDetectorAggregator(BaseDetector):
    """
    Unified multi-detector perception engine combining general COCO YOLO11n
    and industrial open-vocabulary YOLO-World with domain-specific filtering
    and cross-model deduplication.
    """

    def __init__(
        self,
        general_detector: Optional[YOLODetector] = None,
        industrial_detector: Optional[IndustrialDetector] = None,
        cross_model_iou_thresh: float = 0.40,
        filter_irrelevant_coco: Optional[bool] = None,
        irrelevant_classes: Optional[List[str]] = None,
    ):
        settings = get_settings()
        self.general_detector = general_detector
        self.industrial_detector = industrial_detector
        self.cross_model_iou_thresh = cross_model_iou_thresh

        self.filter_irrelevant_coco = (
            filter_irrelevant_coco
            if filter_irrelevant_coco is not None
            else getattr(settings, "FILTER_IRRELEVANT_COCO_CLASSES", True)
        )
        self.irrelevant_classes: Set[str] = set(
            c.lower().strip()
            for c in (
                irrelevant_classes
                if irrelevant_classes is not None
                else getattr(settings, "IRRELEVANT_COCO_CLASSES", [])
            )
        )

    def load_model(self, model_path: str) -> None:
        """Delegates loading to general detector if provided."""
        if self.general_detector:
            self.general_detector.load_model(model_path)

    def filter_detections(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """Filters out nonsensical non-industrial COCO classes."""
        if not self.filter_irrelevant_coco or not self.irrelevant_classes:
            return detections

        filtered: List[DetectionResult] = []
        for det in detections:
            cname = det.class_name.lower().strip()
            # If from general COCO detector and in irrelevant list, drop it
            if det.source_model == "yolo11n" and cname in self.irrelevant_classes:
                logger.debug(f"Filtering out irrelevant COCO class '{cname}' (conf: {det.confidence:.2f})")
                continue
            filtered.append(det)
        return filtered

    def merge_detections(
        self,
        general_dets: List[DetectionResult],
        industrial_dets: List[DetectionResult],
    ) -> List[DetectionResult]:
        """
        Merges general and industrial detections with IoU deduplication and priority resolution:
          - If boxes overlap significantly (IoU >= threshold):
              * Industrial classification takes precedence over generic COCO vehicles/objects (e.g. boat/car/truck -> forklift).
              * For persons, the detection with higher confidence is kept.
              * Otherwise, the higher confidence detection is retained.
          - Non-overlapping detections are both kept.
        """
        if not general_dets:
            return self.deduplicate_vehicles(list(industrial_dets))
        if not industrial_dets:
            return self.deduplicate_vehicles(list(general_dets))

        # Industrial detections take precedence for vehicles and machinery
        suppressed_general_indices: Set[int] = set()
        suppressed_industrial_indices: Set[int] = set()

        for g_idx, g_det in enumerate(general_dets):
            for i_idx, i_det in enumerate(industrial_dets):
                if i_idx in suppressed_industrial_indices or g_idx in suppressed_general_indices:
                    continue

                iou = compute_box_iou(g_det.bbox, i_det.bbox)
                iomin = compute_box_iomin(g_det.bbox, i_det.bbox)
                if iou >= self.cross_model_iou_thresh or (iomin >= 0.65 and (g_det.class_group == i_det.class_group or g_det.class_group == ClassGroup.VEHICLE)):
                    # Overlapping detection found! Resolve priority:
                    g_is_person = g_det.class_group == ClassGroup.PERSON
                    i_is_person = i_det.class_group == ClassGroup.PERSON

                    if g_is_person and i_is_person:
                        # Both detect person: choose higher confidence
                        if g_det.confidence >= i_det.confidence:
                            suppressed_industrial_indices.add(i_idx)
                        else:
                            suppressed_general_indices.add(g_idx)
                    elif (g_is_person and not i_is_person) or (i_is_person and not g_is_person):
                        # CRITICAL FIX: A person inside or near a vehicle/machine is a legitimate
                        # detection and must NOT be suppressed by the vehicle/machine bounding box!
                        continue
                    elif i_det.class_group in (ClassGroup.VEHICLE, ClassGroup.MACHINE, ClassGroup.INFRASTRUCTURE):
                        # Industrial detector classified as vehicle/machine/infra: prioritize industrial
                        suppressed_general_indices.add(g_idx)
                    elif g_det.confidence > i_det.confidence + 0.20:
                        # General detector has substantially higher confidence
                        suppressed_industrial_indices.add(i_idx)
                    else:
                        # Default to industrial model for domain-specific perception
                        suppressed_general_indices.add(g_idx)

        merged: List[DetectionResult] = []
        for g_idx, g_det in enumerate(general_dets):
            if g_idx not in suppressed_general_indices:
                merged.append(g_det)

        for i_idx, i_det in enumerate(industrial_dets):
            if i_idx not in suppressed_industrial_indices:
                merged.append(i_det)

        merged = self.deduplicate_vehicles(merged)
        return self.deduplicate_persons(merged)

    def deduplicate_persons(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Suppresses duplicate nested person detections on the same physical person
        (e.g., upper-body / torso sub-boxes nested inside full-body person detections with IoMin >= 0.75),
        preserving legitimate separate nearby workers.
        """
        persons = [d for d in detections if d.class_group == ClassGroup.PERSON]
        if len(persons) <= 1:
            return detections

        persons.sort(key=lambda d: d.bbox.width * d.bbox.height, reverse=True)
        suppressed: Set[str] = set()

        for i, p_larger in enumerate(persons):
            if p_larger.detection_id in suppressed:
                continue
            for p_smaller in persons[i + 1:]:
                if p_smaller.detection_id in suppressed:
                    continue
                iomin = compute_box_iomin(p_larger.bbox, p_smaller.bbox)
                if iomin >= 0.75:
                    suppressed.add(p_smaller.detection_id)

        return [d for d in detections if d.detection_id not in suppressed]

    def deduplicate_vehicles(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Suppresses duplicate vehicle detections on the same physical vehicle
        (e.g., overlapping 'truck' and 'bus' or 'car' and 'truck' with IoU >= threshold or IoMin >= 0.65),
        preserving legitimate separate nearby vehicles.
        """
        if len(detections) <= 1:
            return detections

        suppressed: Set[int] = set()
        vehicle_indices = [
            idx for idx, d in enumerate(detections)
            if d.class_group == ClassGroup.VEHICLE
        ]

        for i_pos, idx1 in enumerate(vehicle_indices):
            if idx1 in suppressed:
                continue
            d1 = detections[idx1]
            for idx2 in vehicle_indices[i_pos + 1:]:
                if idx2 in suppressed:
                    continue
                d2 = detections[idx2]
                iou = compute_box_iou(d1.bbox, d2.bbox)
                iomin = compute_box_iomin(d1.bbox, d2.bbox)
                if iou >= self.cross_model_iou_thresh or iomin >= 0.65:
                    # Resolve which vehicle detection to keep:
                    # 1. Specialized industrial vehicle/forklift takes priority over generic truck/bus/car
                    d1_is_forklift = "forklift" in d1.class_name.lower() or "industrial" in d1.class_name.lower()
                    d2_is_forklift = "forklift" in d2.class_name.lower() or "industrial" in d2.class_name.lower()
                    if d1_is_forklift and not d2_is_forklift:
                        suppressed.add(idx2)
                    elif d2_is_forklift and not d1_is_forklift:
                        suppressed.add(idx1)
                        break
                    # 2. In-vehicle occupant indicator takes precedence
                    elif d1.metadata.get("in_vehicle") and not d2.metadata.get("in_vehicle"):
                        suppressed.add(idx2)
                    elif d2.metadata.get("in_vehicle") and not d1.metadata.get("in_vehicle"):
                        suppressed.add(idx1)
                        break
                    # 3. Higher confidence
                    elif d1.confidence >= d2.confidence:
                        suppressed.add(idx2)
                    else:
                        suppressed.add(idx1)
                        break

        return [d for idx, d in enumerate(detections) if idx not in suppressed]

    def detect(
        self,
        frame: Union[np.ndarray, FrameData],
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> FrameDetections:
        """
        Executes multi-detector perception on the input frame.
        """
        fid = frame.frame_index if isinstance(frame, FrameData) and frame_id is None else (frame_id or 0)
        ts = frame.timestamp if isinstance(frame, FrameData) and timestamp is None else (timestamp or 0.0)

        # 1. Run General Detector (YOLO11n)
        general_dets: List[DetectionResult] = []
        source_w, source_h = None, None
        if self.general_detector is not None:
            raw_gen = self.general_detector.detect(frame, frame_id=fid, timestamp=ts, imgsz=imgsz)
            general_dets = raw_gen.detections
            source_w = raw_gen.frame_width
            source_h = raw_gen.frame_height

        # 2. Filter irrelevant COCO classes
        filtered_general = self.filter_detections(general_dets)

        # 3. Run Industrial Detector (YOLO-World)
        industrial_dets: List[DetectionResult] = []
        if self.industrial_detector is not None:
            raw_ind = self.industrial_detector.detect(frame, frame_id=fid, timestamp=ts)
            industrial_dets = raw_ind.detections
            if source_w is None:
                source_w = raw_ind.frame_width
                source_h = raw_ind.frame_height

        # 4. Merge detections with IoU deduplication
        final_detections = self.merge_detections(filtered_general, industrial_dets)

        return FrameDetections(
            frame_id=fid,
            timestamp=round(ts, 4),
            detections=final_detections,
            frame_width=source_w,
            frame_height=source_h,
        )
