import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Union
import numpy as np
from ultralytics import YOLO

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from configs.settings import get_settings
from vision.detection.base import BaseDetector
from vision.detection.coordinates import unpad_and_rescale_bbox
from vision.preprocessing.frame import FrameData
from vision.utils.device import resolve_device

logger = logging.getLogger("intelliwatch.ppe_detector")


class PPEModelLoadError(Exception):
    """Raised when PPE model weights fail to load."""
    pass


class PPEInferenceError(Exception):
    """Raised when PPE inference fails on a frame."""
    pass


class PPEDetector(BaseDetector):
    """
    Dedicated Personal Protective Equipment (PPE) Detector.
    Perceives safety gear (hard hats, safety vests, gloves, goggles, masks, etc.)
    and negative violation indicators independently from general scene objects.

    Architecture:
      - Modular perception layer decoupled from general YOLO11n.
      - Default weights: SafetyVision YOLOv8n (weights/ppe_yolov8n.pt, ~6.2MB).
      - Optimized for CPU-only inference (pure CPU, zero GPU requirement).
      - Loaded once per session and reused across frames.
      - Automatically re-projects letterboxed bounding boxes back to original video pixel space.
    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        confidence_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        device: Optional[str] = None,
        classes: Optional[List[Union[str, int]]] = None,
        imgsz: int = 640,
        map_to_original: bool = True,
    ):
        settings = get_settings()
        self.model_path = model_path or getattr(settings, "PPE_MODEL_PATH", "weights/ppe_yolov8n.pt")
        self.confidence_threshold = (
            confidence_threshold
            if confidence_threshold is not None
            else getattr(settings, "PPE_CONFIDENCE_THRESHOLD", 0.25)
        )
        self.iou_threshold = (
            iou_threshold
            if iou_threshold is not None
            else getattr(settings, "PPE_IOU_THRESHOLD", 0.45)
        )
        self.device = resolve_device(device or getattr(settings, "PPE_DEVICE", "cpu"))
        self.imgsz = imgsz
        self.map_to_original = map_to_original

        self._filter_classes: Optional[Set[Union[str, int]]] = set(classes) if classes else None
        self._model: Optional[YOLO] = None
        self._class_names: Dict[int, str] = {}

        if self.model_path:
            self.load_model(self.model_path)

    @property
    def class_names(self) -> Dict[int, str]:
        """Returns mapping of class IDs to human-readable PPE category names."""
        return self._class_names

    @property
    def is_ready(self) -> bool:
        """Returns True if PPE model weights are loaded and ready."""
        return self._model is not None

    def load_model(self, model_path: str) -> None:
        """
        Loads PPE model weights via Ultralytics once per session.
        """
        logger.info(f"Loading PPE model weights from: {model_path} on device [{self.device}]")
        try:
            resolved_path = Path(model_path)
            # If path points to non-existent local file, check weights/ subdirectory
            if not resolved_path.exists():
                fallback = Path("weights") / resolved_path.name
                if fallback.exists():
                    resolved_path = fallback

            if not resolved_path.exists():
                raise FileNotFoundError(f"PPE model weights not found at {model_path} or {fallback}")

            self._model = YOLO(str(resolved_path))
            self.model_path = str(resolved_path)
            self._class_names = dict(self._model.names) if hasattr(self._model, "names") else {}

            if self.device and self.device != "cpu":
                try:
                    self._model.to(self.device)
                except Exception as e:
                    logger.warning(f"Could not immediately transfer PPE model to {self.device}: {e}")

            logger.info(
                f"Successfully loaded PPE model [{self.model_path}] on [{self.device}] "
                f"with {len(self._class_names)} classes: {list(self._class_names.values())}"
            )
        except Exception as e:
            msg = f"Failed to load PPE model from '{model_path}': {e}"
            logger.error(msg)
            raise PPEModelLoadError(msg) from e

    def set_class_filter(self, classes: Optional[List[Union[str, int]]]) -> None:
        """Configures an active class filter for specific PPE categories."""
        if classes is None:
            self._filter_classes = None
            logger.info("PPE class filter cleared. Detecting all supported PPE categories.")
        else:
            self._filter_classes = set(classes)
            logger.info(f"PPE class filter configured: {self._filter_classes}")

    def set_confidence_threshold(self, threshold: float) -> None:
        """Updates the runtime confidence threshold."""
        if not (0.0 <= threshold <= 1.0):
            raise ValueError(f"Confidence threshold must be between 0.0 and 1.0, got {threshold}")
        self.confidence_threshold = threshold
        logger.info(f"Updated PPE confidence threshold to {threshold}")

    def detect(
        self,
        frame: Union[np.ndarray, FrameData],
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        candidate_floor: Optional[float] = None,
    ) -> FrameDetections:
        """
        Performs PPE inference on a single video frame.

        Args:
            frame: Either a raw NumPy image (BGR) or a structured FrameData instance.
            frame_id: Optional frame index override.
            timestamp: Optional timestamp override.
            candidate_floor: Optional lower-confidence candidate floor. Candidates
                below the configured threshold must still pass strong PPE association.

        Returns:
            FrameDetections containing bounding boxes transformed back to
            the original video coordinates.
        """
        if self._model is None:
            raise PPEModelLoadError("PPE model must be loaded before running inference.")

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

        inference_confidence = self.confidence_threshold
        if candidate_floor is not None:
            inference_confidence = float(candidate_floor)
            if not 0.0 <= inference_confidence <= self.confidence_threshold:
                raise ValueError(
                    "candidate_floor must be between 0 and the configured PPE confidence threshold."
                )

        # Execute PPE model inference with automatic CPU fallback
        try:
            results = self._model(
                source=img,
                conf=inference_confidence,
                iou=self.iou_threshold,
                imgsz=self.imgsz,
                device=self.device,
                verbose=False,
            )
        except Exception as e:
            if self.device != "cpu":
                logger.warning(
                    f"PPE inference failed on device [{self.device}]: {e}. "
                    "Safely falling back to CPU."
                )
                self.device = "cpu"
                try:
                    results = self._model(
                        source=img,
                        conf=inference_confidence,
                        iou=self.iou_threshold,
                        imgsz=self.imgsz,
                        device="cpu",
                        verbose=False,
                    )
                except Exception as inner_e:
                    msg = f"PPE inference failed on frame {fid} even on CPU: {inner_e}"
                    logger.error(msg)
                    raise PPEInferenceError(msg) from inner_e
            else:
                msg = f"PPE inference failed on frame {fid}: {e}"
                logger.error(msg)
                raise PPEInferenceError(msg) from e

        detections: List[DetectionResult] = []

        if results and len(results) > 0 and results[0].boxes is not None:
            boxes = results[0].boxes
            for box in boxes:
                conf = float(box.conf[0])
                if conf < inference_confidence:
                    continue

                cls_id = int(box.cls[0])
                cls_name = self._class_names.get(cls_id, f"class_{cls_id}")

                # Apply class filtering if active
                if self._filter_classes is not None:
                    if cls_id not in self._filter_classes and cls_name not in self._filter_classes:
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

                metadata = {}
                if conf < self.confidence_threshold:
                    metadata = {
                        "low_confidence_candidate": True,
                        "configured_confidence_threshold": self.confidence_threshold,
                    }

                detections.append(
                    DetectionResult(
                        class_id=cls_id,
                        class_name=cls_name,
                        confidence=round(conf, 4),
                        bbox=final_bbox,
                        source_model="safetyvision_ppe",
                        frame_index=fid,
                        timestamp=round(ts, 4),
                        metadata=metadata,
                    )
                )

        return FrameDetections(
            frame_id=fid,
            timestamp=ts,
            detections=detections,
            frame_width=source_w,
            frame_height=source_h,
        )
