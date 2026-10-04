import logging
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import torch

from backend.schemas.detection import BoundingBox
from backend.schemas.depth import DepthResult, DepthStatistics, DepthType, ObjectDepth
from configs.settings import get_settings
from vision.depth.base import BaseDepthEstimator
from vision.preprocessing.frame import FrameData

logger = logging.getLogger("intelliwatch.depth_estimator")


class DepthAnythingEstimator(BaseDepthEstimator):
    """
    Monocular Depth Estimator powered by Depth Anything V2 Small.

    Model Properties:
      - Architecture: Depth-Anything-V2-Small (Vision Transformer / DPT head)
      - Parameter Count: ~24.8M
      - Weight Size: ~98 MB
      - License: Apache 2.0
      - Output Type: Monocular RELATIVE depth (uncalibrated disparity/relative distance).
        Notice: Higher values represent closer proximity / inverse distance.
        This model does NOT produce metric measurements (meters).

    Technical Design:
      1. Model and processor are loaded once during initialization and held in memory.
      2. Frame is converted from OpenCV BGR to RGB and processed through AutoImageProcessor.
      3. Depth tensor is predicted and bilinearly interpolated back to native frame resolution (H, W).
      4. Object-level depth distribution and ground contact point depth are sampled robustly.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
        input_size: Optional[int] = None,
        processor: Optional[Any] = None,
        model: Optional[Any] = None,
    ):
        settings = get_settings()

        self.model_name = (
            model_name
            if model_name is not None
            else getattr(settings, "DEPTH_MODEL_NAME", "depth-anything/Depth-Anything-V2-Small-hf")
        )
        self.device_str = (
            device
            if device is not None
            else getattr(settings, "DEPTH_DEVICE", "cpu")
        )
        self.device = torch.device(self.device_str)
        self.input_size = (
            input_size
            if input_size is not None
            else getattr(settings, "DEPTH_INPUT_SIZE", 518)
        )

        # Allow dependency injection for unit testing / mocking
        if processor is not None and model is not None:
            self.processor = processor
            self.model = model
            self.model.to(self.device)
            self.model.eval()
            logger.info("Initialized DepthAnythingEstimator with injected model & processor.")
        else:
            self._load_model()

    def _load_model(self) -> None:
        """Loads Hugging Face AutoImageProcessor and AutoModelForDepthEstimation once."""
        t0 = time.perf_counter()
        logger.info(f"Loading Depth Anything V2 model: {self.model_name} on {self.device_str}...")

        from transformers import AutoImageProcessor, AutoModelForDepthEstimation

        self.processor = AutoImageProcessor.from_pretrained(
            self.model_name,
            use_fast=True,
        )
        self.model = AutoModelForDepthEstimation.from_pretrained(self.model_name)
        self.model.to(self.device)
        self.model.eval()

        load_sec = time.perf_counter() - t0
        logger.info(f"Depth model successfully loaded in {load_sec:.2f}s.")

    def estimate_depth(
        self,
        frame: Union[np.ndarray, FrameData],
        normalize: bool = True,
    ) -> Tuple[np.ndarray, DepthResult]:
        """
        Executes monocular depth inference on a single frame.

        Args:
            frame: Video frame (OpenCV BGR numpy array or FrameData instance).
            normalize: If True, maps relative depth to [0.0, 1.0].

        Returns:
            Tuple of:
              - dense_depth: 2D numpy array of shape (native_height, native_width), dtype float32
              - depth_result: DepthResult schema with global summary statistics
        """
        if frame is None:
            raise ValueError("Input frame cannot be None.")

        # 1. Extract metadata and raw pixel buffer
        if isinstance(frame, FrameData):
            raw_img = frame.image
            orig_h = frame.height
            orig_w = frame.width
            frame_id = frame.frame_index
            timestamp = frame.timestamp
        elif isinstance(frame, np.ndarray):
            raw_img = frame
            orig_h, orig_w = raw_img.shape[:2]
            frame_id = None
            timestamp = 0.0
        else:
            raise TypeError(f"Unsupported frame type: {type(frame)}. Expected FrameData or np.ndarray.")

        if raw_img.size == 0 or orig_h == 0 or orig_w == 0:
            raise ValueError("Input frame image is empty.")

        t0 = time.perf_counter()

        # 2. Convert BGR to RGB for transformer processor
        if raw_img.ndim == 3 and raw_img.shape[2] == 3:
            rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        elif raw_img.ndim == 2:
            rgb_img = cv2.cvtColor(raw_img, cv2.COLOR_GRAY2RGB)
        else:
            rgb_img = raw_img

        # 3. Preprocess inputs and run inference
        inputs = self.processor(images=rgb_img, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model(**inputs)
            # Depth Anything outputs predicted_depth attribute
            if hasattr(outputs, "predicted_depth"):
                depth_tensor = outputs.predicted_depth
            else:
                depth_tensor = outputs[0]

        # 4. Resize depth map back to native frame resolution (orig_h, orig_w)
        # Using PyTorch interpolate for high-precision bicubic/bilinear scaling
        if depth_tensor.ndim == 2:
            depth_tensor = depth_tensor.unsqueeze(0).unsqueeze(0)
        elif depth_tensor.ndim == 3:
            depth_tensor = depth_tensor.unsqueeze(1)

        depth_interpolated = torch.nn.functional.interpolate(
            depth_tensor,
            size=(orig_h, orig_w),
            mode="bilinear",
            align_corners=False,
        )

        depth_map = depth_interpolated.squeeze().cpu().numpy().astype(np.float32)

        # 5. Normalization & Global Statistics
        raw_min = float(np.min(depth_map))
        raw_max = float(np.max(depth_map))

        if normalize:
            denom = raw_max - raw_min
            if denom > 1e-8:
                depth_map = (depth_map - raw_min) / denom
            else:
                depth_map = np.zeros_like(depth_map)

        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        min_val = float(np.min(depth_map))
        max_val = float(np.max(depth_map))
        mean_val = float(np.mean(depth_map))
        median_val = float(np.median(depth_map))

        result = DepthResult(
            frame_id=frame_id,
            timestamp=timestamp,
            width=orig_w,
            height=orig_h,
            depth_type=DepthType.RELATIVE,
            is_metric=False,
            min_depth=round(min_val, 4),
            max_depth=round(max_val, 4),
            mean_depth=round(mean_val, 4),
            median_depth=round(median_val, 4),
            processing_time_ms=round(elapsed_ms, 2),
            object_depths=[],
        )

        return depth_map, result

    def sample_object_depth(
        self,
        depth_map: np.ndarray,
        bbox: BoundingBox,
        class_name: Optional[str] = None,
        track_id: Optional[int] = None,
    ) -> ObjectDepth:
        """
        Extracts depth statistics and contact-point depth for a specific bounding box.

        Args:
            depth_map: 2D numpy array of dense depth values (H, W).
            bbox: BoundingBox in native frame coordinates.
            class_name: Class label of detected object.
            track_id: Optional tracking identity.

        Returns:
            ObjectDepth schema with statistical percentiles and contact depth.
        """
        h, w = depth_map.shape[:2]

        x1 = max(0, min(w - 1, int(round(bbox.x1))))
        y1 = max(0, min(h - 1, int(round(bbox.y1))))
        x2 = max(x1 + 1, min(w, int(round(bbox.x2))))
        y2 = max(y1 + 1, min(h, int(round(bbox.y2))))

        roi = depth_map[y1:y2, x1:x2]

        if roi.size == 0:
            stats = DepthStatistics(
                min_depth=0.0,
                max_depth=0.0,
                mean_depth=0.0,
                median_depth=0.0,
                percentile_25=0.0,
                percentile_75=0.0,
                depth_unit="relative",
            )
            return ObjectDepth(
                track_id=track_id,
                class_name=class_name,
                bbox=bbox,
                depth_stats=stats,
                contact_point=None,
                contact_depth=None,
                is_metric=False,
            )

        min_d = float(np.min(roi))
        max_d = float(np.max(roi))
        mean_d = float(np.mean(roi))
        med_d = float(np.median(roi))
        p25_d = float(np.percentile(roi, 25))
        p75_d = float(np.percentile(roi, 75))

        stats = DepthStatistics(
            min_depth=round(min_d, 4),
            max_depth=round(max_d, 4),
            mean_depth=round(mean_d, 4),
            median_depth=round(med_d, 4),
            percentile_25=round(p25_d, 4),
            percentile_75=round(p75_d, 4),
            depth_unit="relative",
        )

        # Contact-point depth calculation
        cls = (class_name or "").lower().strip()
        if cls in ("person", "worker", "pedestrian"):
            # Foot contact point at bottom center
            c_x = (bbox.x1 + bbox.x2) / 2.0
            c_y = float(bbox.y2)
            c_pt = (round(c_x, 1), round(c_y, 1))

            # Sample a 5x5 window around feet, clamped to bounds
            px = max(0, min(w - 1, int(round(c_x))))
            py = max(0, min(h - 1, int(round(c_y - 2))))
            w_half = 2
            patch = depth_map[
                max(0, py - w_half): min(h, py + w_half + 1),
                max(0, px - w_half): min(w, px + w_half + 1),
            ]
            contact_depth = float(np.median(patch)) if patch.size > 0 else float(depth_map[py, px])
        else:
            # Centroid for non-person objects
            c_x, c_y = bbox.center
            c_pt = (round(c_x, 1), round(c_y, 1))
            px = max(0, min(w - 1, int(round(c_x))))
            py = max(0, min(h - 1, int(round(c_y))))
            contact_depth = float(depth_map[py, px])

        return ObjectDepth(
            track_id=track_id,
            class_name=class_name,
            bbox=bbox,
            depth_stats=stats,
            contact_point=c_pt,
            contact_depth=round(contact_depth, 4),
            is_metric=False,
        )
