import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from backend.schemas.detection import DetectionResult, FrameDetections

logger = logging.getLogger("intelliwatch.visualizer")


class DetectionVisualizer:
    """
    Visualization utility to overlay bounding boxes, class labels,
    and confidence scores on video frames for debugging and verification.
    """

    # High-visibility colors (BGR format for OpenCV)
    DEFAULT_PALETTE: Dict[str, Tuple[int, int, int]] = {
        "person": (0, 220, 100),       # Vibrant Green
        "forklift": (0, 140, 255),     # Orange
        "truck": (0, 165, 255),        # Deep Orange
        "car": (255, 180, 0),          # Sky Blue
        "machinery": (220, 50, 220),   # Magenta
        "helmet": (0, 235, 255),       # Yellow
        "hardhat": (0, 235, 255),      # Yellow
        "vest": (0, 255, 200),         # Fluorescent Lime
        "safety vest": (0, 165, 255),  # Fluorescent Orange
        "gloves": (255, 215, 0),       # Cyan
        "goggles": (255, 160, 0),      # Electric Sky Blue
        "mask": (0, 255, 128),         # Lime Green
        "no-hardhat": (0, 0, 255),     # Bright Red (Violation)
        "no-safety vest": (0, 0, 255), # Bright Red (Violation)
        "no-gloves": (0, 0, 255),      # Bright Red (Violation)
        "no-goggles": (0, 0, 255),     # Bright Red (Violation)
        "fall-detected": (0, 0, 200),  # Dark Red
    }

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        palette: Optional[Dict[str, Tuple[int, int, int]]] = None,
    ):
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.palette = {k.lower(): v for k, v in self.DEFAULT_PALETTE.items()}
        if palette:
            for k, v in palette.items():
                self.palette[k.lower()] = v

    def _get_class_color(self, class_name: str) -> Tuple[int, int, int]:
        """Returns designated color or generates a deterministic color from class name."""
        norm_name = class_name.lower().strip()
        if norm_name in self.palette:
            return self.palette[norm_name]
        # Generate consistent color from hash
        h = int(hashlib.md5(class_name.encode("utf-8")).hexdigest(), 16)
        r = (h & 0xFF0000) >> 16
        g = (h & 0x00FF00) >> 8
        b = h & 0x0000FF
        # Ensure it's not too dark
        return (max(60, b), max(60, g), max(60, r))

    def draw_detections(
        self,
        image: np.ndarray,
        detections: Union[FrameDetections, List[DetectionResult]],
        show_labels: bool = True,
        show_conf: bool = True,
        show_banner: bool = True,
    ) -> np.ndarray:
        """
        Draws bounding boxes and labels onto a copy of the input frame.

        Args:
            image: Original or preprocessed frame (BGR).
            detections: FrameDetections schema or list of DetectionResult objects.
            show_labels: Whether to render class names.
            show_conf: Whether to render confidence percentage.
            show_banner: Whether to overlay top informational metadata banner.

        Returns:
            New annotated numpy ndarray.
        """
        if image is None or image.size == 0:
            raise ValueError("Input image cannot be empty.")

        annotated = image.copy()
        h, w = annotated.shape[:2]

        if isinstance(detections, FrameDetections):
            det_list = detections.detections
            frame_id = detections.frame_id
            timestamp = detections.timestamp
        else:
            det_list = detections
            frame_id = None
            timestamp = None

        # 1. Draw individual detections
        for det in det_list:
            bbox = det.bbox
            x1, y1 = int(round(bbox.x1)), int(round(bbox.y1))
            x2, y2 = int(round(bbox.x2)), int(round(bbox.y2))

            # Clamp coordinates to canvas
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(0, min(w - 1, x2))
            y2 = max(0, min(h - 1, y2))

            color = self._get_class_color(det.class_name)

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness)

            if show_labels:
                label = det.class_name
                if show_conf:
                    label += f" {det.confidence:.2f}"

                # Calculate text size for background badge
                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
                )
                badge_y1 = max(0, y1 - th - 6)
                badge_y2 = y1
                badge_x2 = min(w, x1 + tw + 6)

                # Draw solid label background for maximum contrast
                cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

                # Text color (black for bright backgrounds, white otherwise)
                brightness = (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299)
                text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 3, badge_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    text_color,
                    self.font_thickness,
                    cv2.LINE_AA,
                )

        # 2. Draw top telemetry banner if frame info is available
        if show_banner and frame_id is not None:
            banner_text = f"Frame: {frame_id:04d} | Time: {timestamp:.2f}s | Detections: {len(det_list)}"
            (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (0, 0), (tw + 20, th + 14), (20, 20, 20), -1)
            cv2.putText(
                annotated,
                banner_text,
                (10, th + 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (220, 220, 220),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def save_annotated_frame(self, image: np.ndarray, output_path: Union[str, Path]) -> Path:
        """Saves annotated image to the target path, creating directories as needed."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), image)
        logger.info(f"Saved annotated frame to: {out}")
        return out
