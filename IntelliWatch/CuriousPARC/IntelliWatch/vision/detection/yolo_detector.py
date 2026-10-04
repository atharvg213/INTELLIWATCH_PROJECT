import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import numpy as np
from ultralytics import YOLO

from backend.schemas.detection import (
    BoundingBox,
    ClassGroup,
    DetectionResult,
    FrameDetections,
    infer_class_group,
)
from configs.settings import get_settings
from vision.detection.base import BaseDetector
from vision.detection.coordinates import compute_box_iou, compute_box_iomin, unpad_and_rescale_bbox
from vision.preprocessing.frame import FrameData
from vision.utils.device import resolve_device

logger = logging.getLogger("intelliwatch.yolo_detector")


class ModelLoadError(Exception):
    """Raised when YOLO weights fail to load."""
    pass


class InferenceError(Exception):
    """Raised when model inference fails on a frame."""
    pass


class YOLODetector(BaseDetector):
    """
    Production-grade YOLO detector integrated with IntelliWatch perception schemas.
    Supports:
      - YOLO11 / YOLOv8 pretrained or custom checkpoints.
      - Class-aware confidence thresholding & Non-Maximum Suppression (IoU) configuration.
      - Dynamic class name/ID filtering.
      - Automatic coordinate re-projection to original CCTV aspect ratio and resolution.
      - High-resolution in-vehicle occupant detection for partially occluded operators.
      - Generalized cross-vehicle class NMS and IoMin enclosure deduplication.
    """
    # Default class attributes for uninitialized instances
    _model: Optional[YOLO] = None
    _class_names: Dict[int, str] = {}
    filter_irrelevant_coco: bool = False
    irrelevant_classes: Set[str] = set()
    enable_occupant_detection: bool = True
    enable_vehicle_deduplication: bool = True

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        person_confidence_threshold: Optional[float] = None,
        vehicle_confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = "cpu",
        classes: Optional[List[Union[str, int]]] = None,
        imgsz: int = 640,
        map_to_original: bool = True,
        filter_irrelevant_coco: Optional[bool] = None,
        irrelevant_classes: Optional[List[str]] = None,
        vehicle_deduplication_enabled: Optional[bool] = None,
        vehicle_deduplication_iou: Optional[float] = None,
        vehicle_deduplication_iomin: Optional[float] = None,
        person_deduplication_enabled: Optional[bool] = None,
        person_deduplication_iomin: Optional[float] = None,
        enable_occupant_detection: Optional[bool] = None,
        occupant_crop_padding: Optional[float] = None,
        occupant_confidence_threshold: Optional[float] = None,
        detection_debug: Optional[bool] = None,
    ):
        settings = get_settings()
        self.model_path = model_path or getattr(settings, "DETECTION_MODEL_PATH", settings.MODEL_PATH)
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(settings, "DETECTION_CONFIDENCE_THRESHOLD", 0.25)
        )
        self.person_confidence_threshold = (
            person_confidence_threshold
            if person_confidence_threshold is not None
            else getattr(settings, "DETECTION_PERSON_CONFIDENCE_THRESHOLD", 0.20)
        )
        self.vehicle_confidence_threshold = (
            vehicle_confidence_threshold
            if vehicle_confidence_threshold is not None
            else getattr(settings, "DETECTION_VEHICLE_CONFIDENCE_THRESHOLD", 0.38)
        )
        self.iou_threshold = iou_threshold if iou_threshold is not None else settings.IOU_THRESHOLD
        self.device = resolve_device(device or getattr(settings, "DETECTION_DEVICE", "cpu"))
        self.imgsz = imgsz
        self.map_to_original = map_to_original

        self.filter_irrelevant_coco = (
            filter_irrelevant_coco
            if filter_irrelevant_coco is not None
            else getattr(settings, "FILTER_IRRELEVANT_COCO_CLASSES", True)
        )
        self.irrelevant_classes = set(
            c.lower().strip()
            for c in (
                irrelevant_classes
                if irrelevant_classes is not None
                else getattr(settings, "IRRELEVANT_COCO_CLASSES", [])
            )
        )
        self.vehicle_deduplication_enabled = (
            vehicle_deduplication_enabled
            if vehicle_deduplication_enabled is not None
            else getattr(settings, "VEHICLE_DEDUPLICATION_ENABLED", True)
        )
        self.vehicle_deduplication_iou = (
            vehicle_deduplication_iou
            if vehicle_deduplication_iou is not None
            else getattr(settings, "VEHICLE_DEDUPLICATION_IOU", 0.40)
        )
        self.vehicle_deduplication_iomin = (
            vehicle_deduplication_iomin
            if vehicle_deduplication_iomin is not None
            else getattr(settings, "VEHICLE_DEDUPLICATION_IOMIN", 0.65)
        )
        self.person_deduplication_enabled = (
            person_deduplication_enabled
            if person_deduplication_enabled is not None
            else getattr(settings, "PERSON_DEDUPLICATION_ENABLED", True)
        )
        self.person_deduplication_iomin = (
            person_deduplication_iomin
            if person_deduplication_iomin is not None
            else getattr(settings, "PERSON_DEDUPLICATION_IOMIN", 0.75)
        )
        self.enable_occupant_detection = (
            enable_occupant_detection
            if enable_occupant_detection is not None
            else getattr(settings, "OCCUPANT_DETECTION_ENABLED", True)
        )
        self.occupant_crop_padding = (
            occupant_crop_padding
            if occupant_crop_padding is not None
            else getattr(settings, "OCCUPANT_CROP_PADDING", 0.15)
        )
        self.occupant_confidence_threshold = (
            occupant_confidence_threshold
            if occupant_confidence_threshold is not None
            else getattr(settings, "OCCUPANT_CONFIDENCE_THRESHOLD", 0.15)
        )
        self.detection_debug = (
            detection_debug
            if detection_debug is not None
            else getattr(settings, "DETECTION_DEBUG", False)
        )

        self._filter_classes: Optional[Set[Union[str, int]]] = set(classes) if classes else None
        self._model: Optional[YOLO] = None
        self._class_names: Dict[int, str] = {}

        if self.model_path:
            self.load_model(self.model_path)

    @property
    def class_names(self) -> Dict[int, str]:
        """Returns map of class IDs to human-readable names."""
        return self._class_names

    @property
    def is_loaded(self) -> bool:
        """Returns True if model weights are loaded and ready."""
        return self._model is not None

    def load_model(self, model_path: str) -> None:
        """
        Loads YOLO model weights via Ultralytics.
        """
        logger.info(f"Loading YOLO model weights from: {model_path} on device [{self.device}]")
        try:
            resolved_path = Path(model_path)
            # If path points to non-existent local file, check weights/ subdirectory
            if not resolved_path.exists():
                fallback = Path("weights") / resolved_path.name
                if fallback.exists():
                    resolved_path = fallback

            self._model = YOLO(str(resolved_path))
            self.model_path = str(resolved_path)
            self._class_names = dict(self._model.names) if hasattr(self._model, "names") else {}

            if self.device and self.device != "cpu":
                try:
                    self._model.to(self.device)
                except Exception as e:
                    logger.warning(f"Could not immediately transfer YOLO model to {self.device}: {e}")

            logger.info(
                f"Successfully loaded YOLO model [{self.model_path}] on [{self.device}] with {len(self._class_names)} classes."
            )
        except Exception as e:
            msg = f"Failed to load YOLO model from '{model_path}': {e}"
            logger.error(msg)
            raise ModelLoadError(msg) from e

    def set_class_filter(self, classes: Optional[List[Union[str, int]]]) -> None:
        """
        Configures an active class filter.
        Only detections whose class name or ID is in the set will be returned.
        Pass None to disable filtering and return all detections.
        """
        self._filter_classes = set(classes) if classes else None
        logger.info(f"Active class filter set to: {self._filter_classes}")

    def detect(
        self,
        frame: Union[np.ndarray, FrameData],
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        imgsz: Optional[int] = None,
    ) -> FrameDetections:
        """
        Executes YOLO inference on a frame or preprocessed FrameData.

        Returns:
            FrameDetections containing bounding boxes transformed back to
            the original video coordinates.
        """
        if self._model is None:
            raise ModelLoadError("Model must be loaded before running inference.")

        if frame is None:
            raise ValueError("Input frame cannot be None.")

        # Determine input representation
        is_frame_data = isinstance(frame, FrameData)
        if is_frame_data:
            img = frame.image
            fid = frame.frame_index if frame_id is None else frame_id
            ts = frame.timestamp if timestamp is None else timestamp
            scale_factor = frame.scale_factor
            pad_offset = frame.pad_offset
            orig_shape = frame.original_shape
            source_w = orig_shape[1] if orig_shape else frame.width
            source_h = orig_shape[0] if orig_shape else frame.height
        else:
            img = frame
            fid = 0 if frame_id is None else frame_id
            ts = 0.0 if timestamp is None else timestamp
            scale_factor = None
            pad_offset = None
            orig_shape = None
            source_w = frame.shape[1] if frame.ndim >= 2 else None
            source_h = frame.shape[0] if frame.ndim >= 2 else None

        if not isinstance(img, np.ndarray) or img.size == 0:
            raise ValueError("Frame image must be a non-empty numpy.ndarray.")

        # Execute YOLO inference with automatic CPU fallback
        # Use lowest class threshold to allow high recall for persons, with class-aware post-filtering
        person_thresh = getattr(self, "person_confidence_threshold", self.confidence_threshold)
        vehicle_thresh = getattr(self, "vehicle_confidence_threshold", self.confidence_threshold)
        min_call_conf = min(self.confidence_threshold, person_thresh, 0.18)
        eff_imgsz = imgsz if imgsz is not None else self.imgsz
        try:
            results = self._model(
                source=img,
                conf=min_call_conf,
                iou=self.iou_threshold,
                imgsz=eff_imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            if self.device != "cpu":
                logger.warning(
                    f"YOLO inference failed on device [{self.device}]: {e}. "
                    "Safely falling back to CPU."
                )
                self.device = "cpu"
                try:
                    results = self._model(
                        source=img,
                        conf=min_call_conf,
                        iou=self.iou_threshold,
                        imgsz=eff_imgsz,
                        device="cpu",
                        verbose=False,
                    )
                except Exception as inner_e:
                    msg = f"Inference failed on frame {fid} even on CPU: {inner_e}"
                    logger.error(msg)
                    raise InferenceError(msg) from inner_e
            else:
                msg = f"Inference failed on frame {fid}: {e}"
                logger.error(msg)
                raise InferenceError(msg) from e

        detections: List[DetectionResult] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self._class_names.get(cls_id, f"class_{cls_id}")
                cname_lower = cls_name.lower().strip()

                # Filter out irrelevant COCO classes for industrial environments
                if getattr(self, "filter_irrelevant_coco", False) and cname_lower in getattr(self, "irrelevant_classes", set()):
                    continue

                # Apply class filtering if active
                if self._filter_classes is not None:
                    if cls_id not in self._filter_classes and cls_name not in self._filter_classes:
                        continue

                cgroup = infer_class_group(cls_name)

                # Class-aware confidence thresholding
                if cgroup == ClassGroup.PERSON:
                    if conf < person_thresh:
                        continue
                elif cgroup == ClassGroup.VEHICLE:
                    if conf < vehicle_thresh:
                        continue
                elif conf < self.confidence_threshold:
                    continue

                coords = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                raw_bbox = BoundingBox(
                    x1=float(coords[0]),
                    y1=float(coords[1]),
                    x2=float(coords[2]),
                    y2=float(coords[3]),
                )

                # Map back to original CCTV coordinates if letterboxing was applied
                if self.map_to_original and (scale_factor is not None or pad_offset is not None):
                    final_bbox = unpad_and_rescale_bbox(
                        bbox=raw_bbox,
                        scale_factor=scale_factor,
                        pad_offset=pad_offset,
                        original_shape=orig_shape,
                    )
                else:
                    # Clamp raw bbox to image dimensions
                    max_w = float(source_w or img.shape[1])
                    max_h = float(source_h or img.shape[0])
                    final_bbox = BoundingBox(
                        x1=round(max(0.0, min(max_w, raw_bbox.x1)), 2),
                        y1=round(max(0.0, min(max_h, raw_bbox.y1)), 2),
                        x2=round(max(0.0, min(max_w, max(raw_bbox.x1, raw_bbox.x2))), 2),
                        y2=round(max(0.0, min(max_h, max(raw_bbox.y1, raw_bbox.y2))), 2),
                    )

                # Geometric sanity check: reject degenerate bounding boxes and extreme slivers
                bw = final_bbox.width
                bh = final_bbox.height
                if bw < 8.0 or bh < 8.0 or (bw * bh) < 64.0:
                    continue
                if max(bw / max(1.0, bh), bh / max(1.0, bw)) > 20.0:
                    continue

                detections.append(
                    DetectionResult(
                        class_id=cls_id,
                        class_name=cls_name,
                        class_group=cgroup,
                        confidence=round(conf, 4),
                        bbox=final_bbox,
                        source_model="yolo11n",
                        frame_index=fid,
                        timestamp=round(ts, 4),
                    )
                )

        raw_person_count = sum(1 for d in detections if d.class_group == ClassGroup.PERSON)
        raw_vehicle_count = sum(1 for d in detections if d.class_group == ClassGroup.VEHICLE)

        # Step 1: Vehicle deduplication (cross-class vehicle IoU and IoMin sub-box suppression)
        if getattr(self, "vehicle_deduplication_enabled", True):
            detections = self._deduplicate_vehicles(detections)

        # Step 1b: Person deduplication (nested upper-body / sub-box IoMin suppression on same individual)
        if getattr(self, "person_deduplication_enabled", True):
            detections = self._deduplicate_persons(detections)

        # Step 2: High-resolution in-vehicle occupant perception and FP vehicle refinement
        if getattr(self, "enable_occupant_detection", True) and isinstance(self._model, YOLO):
            occupants, filtered_fp_ids, vehicle_drivers = self._detect_in_vehicle_occupants(img, detections, fid, ts)
            if filtered_fp_ids:
                for d in detections:
                    if d.detection_id in filtered_fp_ids:
                        d.class_name = "pallet"
                        d.class_group = ClassGroup.INFRASTRUCTURE
                        d.metadata["reclassified_from"] = "vehicle_fp"
            detections.extend(occupants)

            # Refine confirmed vehicles with drivers and reclassify unattended sub-threshold vehicles
            for d in detections:
                if d.class_group == ClassGroup.VEHICLE:
                    if vehicle_drivers.get(d.detection_id):
                        d.class_name = "forklift"
                        d.metadata["vehicle_type"] = "industrial_vehicle"
                        d.metadata["has_driver"] = True
                    elif d.confidence < vehicle_thresh:
                        d.class_name = "pallet"
                        d.class_group = ClassGroup.INFRASTRUCTURE
                        d.metadata["reclassified_from"] = "vehicle_fp"

        final_p = sum(1 for d in detections if d.class_group == ClassGroup.PERSON)
        final_v = sum(1 for d in detections if d.class_group == ClassGroup.VEHICLE)

        if getattr(self, "detection_debug", False) or logger.isEnabledFor(logging.DEBUG):
            logger.info(
                f"Detection frame {fid}: raw_persons={raw_person_count}->final_persons={final_p}, "
                f"raw_vehicles={raw_vehicle_count}->final_vehicles={final_v}, total={len(detections)}"
            )

        return FrameDetections(
            frame_id=fid,
            timestamp=round(ts, 4),
            detections=detections,
            frame_width=source_w,
            frame_height=source_h,
            metadata={
                "raw_person_count": raw_person_count,
                "raw_vehicle_count": raw_vehicle_count,
                "filtered_person_count": final_p,
                "filtered_vehicle_count": final_v,
                "suppressed_person_detections": getattr(self, "_last_suppressed_persons", []),
            },
        )

    def _deduplicate_persons(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Deduplicates overlapping or enclosed person class detections on the same physical person
        (e.g., nested upper-body or torso sub-boxes where compute_box_iomin >= threshold).
        Preserves genuinely separate nearby individuals where IoMin is below threshold.
        """
        self._last_suppressed_persons = []
        persons = [d for d in detections if d.class_group == ClassGroup.PERSON]
        if len(persons) <= 1:
            return detections

        # Sort persons by area descending to prioritize complete body captures over partial crops
        persons.sort(key=lambda d: d.bbox.width * d.bbox.height, reverse=True)
        suppressed_p_ids: Set[str] = set()
        thresh = getattr(self, "person_deduplication_iomin", 0.75)

        for i, p_larger in enumerate(persons):
            if p_larger.detection_id in suppressed_p_ids:
                continue
            for p_smaller in persons[i + 1:]:
                if p_smaller.detection_id in suppressed_p_ids:
                    continue
                iomin = compute_box_iomin(p_larger.bbox, p_smaller.bbox)
                if iomin >= thresh:
                    suppressed_p_ids.add(p_smaller.detection_id)
                    b_dict = (
                        p_smaller.bbox.model_dump()
                        if hasattr(p_smaller.bbox, "model_dump")
                        else p_smaller.bbox.dict()
                    )
                    self._last_suppressed_persons.append({
                        "detection_id": p_smaller.detection_id,
                        "confidence": p_smaller.confidence,
                        "bbox": b_dict,
                        "status": "suppressed",
                        "visual_state": "RED",
                        "suppression_reason": f"IoMin containment {round(iomin, 3)} >= {thresh} inside detection {p_larger.detection_id}",
                        "surviving_id": p_larger.detection_id,
                        "detector_source": p_smaller.source_model,
                    })

        return [d for d in detections if d.detection_id not in suppressed_p_ids]

    def _deduplicate_vehicles(self, detections: List[DetectionResult]) -> List[DetectionResult]:
        """
        Deduplicates overlapping or enclosed vehicle class detections on the same physical vehicle
        (e.g., overlapping 'truck' and 'bus', or nested vehicle boxes with IoU >= threshold or IoMin >= threshold).
        Preserves genuinely separate nearby vehicles where IoU and IoMin are low.
        """
        vehicles = [d for d in detections if d.class_group == ClassGroup.VEHICLE]
        if len(vehicles) <= 1:
            return detections

        # Sort vehicles by confidence descending
        vehicles.sort(key=lambda x: x.confidence, reverse=True)
        suppressed_v_ids: Set[str] = set()

        for i, v1 in enumerate(vehicles):
            if v1.detection_id in suppressed_v_ids:
                continue
            for v2 in vehicles[i + 1:]:
                if v2.detection_id in suppressed_v_ids:
                    continue
                iou = compute_box_iou(v1.bbox, v2.bbox)
                iomin = compute_box_iomin(v1.bbox, v2.bbox)

                if iou >= self.vehicle_deduplication_iou or iomin >= self.vehicle_deduplication_iomin:
                    area1 = v1.bbox.width * v1.bbox.height
                    area2 = v2.bbox.width * v2.bbox.height

                    # If v2 is significantly larger (>1.4x) and has solid confidence (>= 0.30),
                    # it captures the complete physical vehicle rather than a sub-component
                    if area2 > 1.4 * area1 and v2.confidence >= 0.30:
                        suppressed_v_ids.add(v1.detection_id)
                        break
                    else:
                        suppressed_v_ids.add(v2.detection_id)

        return [d for d in detections if d.detection_id not in suppressed_v_ids]

    def _detect_in_vehicle_occupants(
        self,
        img: np.ndarray,
        detections: List[DetectionResult],
        fid: int,
        ts: float,
    ) -> Tuple[List[DetectionResult], Set[str], Dict[str, bool]]:
        """
        Executes localized ROI inference on candidate vehicles to detect seated,
        partially occluded operators/workers that are missed by whole-frame downscaled inference.
        Also validates candidate vehicles against spurious static background detections.
        """
        if not isinstance(img, np.ndarray) or img.ndim < 2:
            return [], set(), {}

        h, w = img.shape[:2]
        new_occupants: List[DetectionResult] = []
        filtered_fp_vehicle_ids: Set[str] = set()
        vehicle_has_driver: Dict[str, bool] = {}

        # Candidate vehicles to inspect
        vehicles = [d for d in detections if d.class_group == ClassGroup.VEHICLE]
        for v in vehicles:
            vx1, vy1, vx2, vy2 = int(v.bbox.x1), int(v.bbox.y1), int(v.bbox.x2), int(v.bbox.y2)
            vw, vh = vx2 - vx1, vy2 - vy1
            if vw < 30 or vh < 30:
                continue

            # Expand vehicle ROI with configurable padding to capture cabs and roll-cages
            pad_x = int(vw * self.occupant_crop_padding)
            pad_y = int(vh * self.occupant_crop_padding)
            crop_x1 = max(0, vx1 - pad_x)
            crop_y1 = max(0, vy1 - pad_y)
            crop_x2 = min(w, vx2 + pad_x)
            crop_y2 = min(h, vy2 + pad_y)

            crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            if crop.size == 0 or crop.shape[0] < 20 or crop.shape[1] < 20:
                continue

            try:
                crop_results = self._model(
                    source=crop,
                    conf=self.occupant_confidence_threshold,
                    iou=self.iou_threshold,
                    imgsz=self.imgsz,
                    device=self.device,
                    verbose=False,
                )
            except Exception as e:
                logger.debug(f"Vehicle crop inference skipped: {e}")
                continue

            crop_people: List[Tuple[float, BoundingBox]] = []
            crop_vehicle_confs: List[float] = []

            if crop_results and len(crop_results) > 0 and crop_results[0].boxes is not None:
                for cbox in crop_results[0].boxes:
                    ccls = int(cbox.cls[0])
                    cname = self._class_names.get(ccls, "")
                    cconf = float(cbox.conf[0])
                    cgrp = infer_class_group(cname)

                    if cgrp == ClassGroup.PERSON:
                        coords = cbox.xyxy[0].tolist()
                        fx1 = round(crop_x1 + float(coords[0]), 2)
                        fy1 = round(crop_y1 + float(coords[1]), 2)
                        fx2 = round(crop_x1 + float(coords[2]), 2)
                        fy2 = round(crop_y1 + float(coords[3]), 2)
                        crop_people.append((cconf, BoundingBox(x1=fx1, y1=fy1, x2=fx2, y2=fy2)))
                    elif cgrp == ClassGroup.VEHICLE:
                        crop_vehicle_confs.append(cconf)

            # NMS among multiple occupants in the same vehicle (e.g. driver and passenger)
            crop_people.sort(key=lambda x: x[0], reverse=True)
            distinct_crop_people: List[Tuple[float, BoundingBox]] = []
            for pconf, pbox in crop_people:
                if any(compute_box_iou(pbox, kb) >= 0.45 for _, kb in distinct_crop_people):
                    continue
                distinct_crop_people.append((pconf, pbox))

            has_occupant = len(distinct_crop_people) > 0
            for pconf, pbox in distinct_crop_people:
                # Prevent duplicate if this person was already detected on full frame
                is_dup = any(
                    compute_box_iou(pbox, ed.bbox) >= 0.45 or compute_box_iomin(pbox, ed.bbox) >= 0.65
                    for ed in detections if ed.class_group == ClassGroup.PERSON
                )
                if not is_dup:
                    new_occupants.append(
                        DetectionResult(
                            class_id=0,
                            class_name="person",
                            class_group=ClassGroup.PERSON,
                            confidence=round(pconf, 4),
                            bbox=pbox,
                            source_model="yolo11n_occupant_roi",
                            frame_index=fid,
                            timestamp=round(ts, 4),
                            metadata={"in_vehicle": True, "vehicle_id": v.detection_id},
                        )
                    )
                    vehicle_has_driver[v.detection_id] = True

            # Generalized false-positive vehicle rejection:
            # If full-frame vehicle had moderate confidence (<0.50), has NO detected occupant,
            # and in the full-resolution crop no vehicle structure was verified (crop_veh_conf < 0.22),
            # the detection was a spurious artifact on static infrastructure (storage racks, wire bins).
            max_crop_veh_conf = max(crop_vehicle_confs) if crop_vehicle_confs else 0.0
            if v.confidence < 0.50 and not has_occupant and max_crop_veh_conf < 0.28:
                filtered_fp_vehicle_ids.add(v.detection_id)
                logger.debug(
                    f"Suppressed false positive vehicle {v.class_name} (conf={v.confidence:.2f}, crop_conf={max_crop_veh_conf:.2f})"
                )

        return new_occupants, filtered_fp_vehicle_ids, vehicle_has_driver

