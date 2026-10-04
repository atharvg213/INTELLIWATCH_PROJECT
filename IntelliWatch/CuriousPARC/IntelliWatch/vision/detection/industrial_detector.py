"""
vision/detection/industrial_detector.py
Production Industrial Open-Vocabulary Object Detector for IntelliWatch.

Leverages YOLO-World (e.g. YOLOv8s-World-v2) with configurable industrial open-vocabulary
concepts (forklifts, industrial vehicles, machinery, robotic arms, conveyors, pallets,
safety barriers, electrical cabinets) on CPU.
"""
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import numpy as np
from ultralytics import YOLO

from backend.schemas.detection import BoundingBox, ClassGroup, DetectionResult, FrameDetections, infer_class_group
from configs.settings import get_settings
from vision.detection.base import BaseDetector
from vision.detection.coordinates import unpad_and_rescale_bbox
from vision.preprocessing.frame import FrameData
from vision.utils.device import resolve_device

logger = logging.getLogger("intelliwatch.industrial_detector")


class IndustrialModelLoadError(Exception):
    """Raised when industrial model weights fail to load."""
    pass


class IndustrialInferenceError(Exception):
    """Raised when industrial model inference fails."""
    pass


class IndustrialDetector(BaseDetector):
    """
    Industrial Open-Vocabulary Object Detector using YOLO-World.
    Specialized for factory, warehouse, and construction environments.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        vocabulary: Optional[List[str]] = None,
        class_thresholds: Optional[Dict[str, float]] = None,
        device: Optional[str] = "cpu",
        imgsz: int = 640,
        map_to_original: bool = True,
    ):
        settings = get_settings()
        self.model_path = model_path or getattr(settings, "INDUSTRIAL_MODEL_PATH", "weights/yolov8s-worldv2.pt")
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(settings, "INDUSTRIAL_CONFIDENCE_THRESHOLD", 0.15)
        )
        self.iou_threshold = (
            iou_threshold
            if iou_threshold is not None
            else getattr(settings, "INDUSTRIAL_IOU_THRESHOLD", 0.45)
        )
        self.vocabulary = list(
            vocabulary
            if vocabulary is not None
            else getattr(settings, "INDUSTRIAL_VOCABULARY", [
                "forklift",
                "industrial vehicle",
                "machinery",
                "robotic arm",
                "conveyor",
                "pallet",
                "safety barrier",
                "electrical cabinet",
            ])
        )
        self.class_thresholds = class_thresholds or {}
        self.device = resolve_device(device or getattr(settings, "INDUSTRIAL_DEVICE", "cpu"))
        self.imgsz = imgsz
        self.map_to_original = map_to_original

        self._model: Optional[YOLO] = None
        self._class_names: Dict[int, str] = {}

        if self.model_path:
            self.load_model(self.model_path)

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    @property
    def class_names(self) -> Dict[int, str]:
        return self._class_names

    def load_model(self, model_path: str) -> None:
        """Loads the YOLO-World checkpoint and configures the target industrial vocabulary."""
        logger.info(f"Loading Industrial YOLO-World model from: {model_path} on device [{self.device}]")
        try:
            resolved_path = Path(model_path)
            if not resolved_path.exists():
                fallback = Path("weights") / resolved_path.name
                if fallback.exists():
                    resolved_path = fallback

            if not resolved_path.exists():
                raise FileNotFoundError(f"Industrial model weights not found at: {resolved_path}")

            self._model = YOLO(str(resolved_path))
            self.model_path = str(resolved_path)

            # Apply custom industrial vocabulary
            if self.vocabulary:
                self.set_vocabulary(self.vocabulary)
            else:
                self._class_names = dict(self._model.names) if hasattr(self._model, "names") else {}

            logger.info(
                f"Successfully loaded Industrial Detector [{self.model_path}] with {len(self._class_names)} classes: {list(self._class_names.values())}"
            )
        except Exception as e:
            msg = f"Failed to load Industrial Detector model from '{model_path}': {e}"
            logger.error(msg)
            raise IndustrialModelLoadError(msg) from e

    def set_vocabulary(self, vocabulary: List[str]) -> None:
        """Dynamically reconfigures the open-vocabulary target classes."""
        self.vocabulary = list(vocabulary)
        if self._model is not None:
            self._model.set_classes(self.vocabulary)
            self._class_names = {i: name for i, name in enumerate(self.vocabulary)}
            logger.info(f"Industrial vocabulary updated ({len(self.vocabulary)} classes): {self.vocabulary}")

    def detect(
        self,
        frame: Union[np.ndarray, FrameData],
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> FrameDetections:
        """
        Executes industrial object detection on a frame.
        """
        if self._model is None:
            raise IndustrialModelLoadError("Industrial detector model must be loaded before running inference.")

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

        # Determine minimum confidence to use in predict call
        min_conf = self.confidence_threshold
        if self.class_thresholds:
            min_conf = min(min_conf, min(self.class_thresholds.values()))

        try:
            results = self._model(
                source=img,
                conf=min_conf,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            if self.device != "cpu":
                logger.warning(
                    f"Industrial inference failed on device [{self.device}]: {e}. "
                    "Safely falling back to CPU."
                )
                self.device = "cpu"
                try:
                    results = self._model(
                        source=img,
                        conf=min_conf,
                        iou=self.iou_threshold,
                        imgsz=self.imgsz,
                        device="cpu",
                        verbose=False,
                    )
                except Exception as inner_e:
                    msg = f"Industrial inference failed on frame {fid} even on CPU: {inner_e}"
                    logger.error(msg)
                    raise IndustrialInferenceError(msg) from inner_e
            else:
                msg = f"Industrial inference failed on frame {fid}: {e}"
                logger.error(msg)
                raise IndustrialInferenceError(msg) from e

        detections: List[DetectionResult] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self._class_names.get(cls_id, f"industrial_class_{cls_id}")

                # Check class-specific confidence threshold
                required_conf = self.class_thresholds.get(cls_name, self.confidence_threshold)
                if conf < required_conf:
                    continue

                coords = box.xyxy[0].tolist()
                raw_bbox = BoundingBox(
                    x1=float(coords[0]),
                    y1=float(coords[1]),
                    x2=float(coords[2]),
                    y2=float(coords[3]),
                )

                if self.map_to_original and (scale_factor is not None or pad_offset is not None):
                    final_bbox = unpad_and_rescale_bbox(
                        bbox=raw_bbox,
                        scale_factor=scale_factor,
                        pad_offset=pad_offset,
                        original_shape=orig_shape,
                    )
                else:
                    max_w = float(source_w or img.shape[1])
                    max_h = float(source_h or img.shape[0])
                    final_bbox = BoundingBox(
                        x1=round(max(0.0, min(max_w, raw_bbox.x1)), 2),
                        y1=round(max(0.0, min(max_h, raw_bbox.y1)), 2),
                        x2=round(max(0.0, min(max_w, max(raw_bbox.x1, raw_bbox.x2))), 2),
                        y2=round(max(0.0, min(max_h, max(raw_bbox.y1, raw_bbox.y2))), 2),
                    )

                cgroup = infer_class_group(cls_name)

                detections.append(
                    DetectionResult(
                        class_id=cls_id,
                        class_name=cls_name,
                        class_group=cgroup,
                        confidence=round(conf, 4),
                        bbox=final_bbox,
                        source_model="yolo_world",
                        frame_index=fid,
                        timestamp=round(ts, 4),
                        metadata={"vocabulary_index": cls_id},
                    )
                )

        return FrameDetections(
            frame_id=fid,
            timestamp=round(ts, 4),
            detections=detections,
            frame_width=source_w,
            frame_height=source_h,
        )
