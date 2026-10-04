import logging
from pathlib import Path
from typing import List, Optional, Tuple, Union
import cv2
import numpy as np

from backend.schemas.depth import ObjectDepth

logger = logging.getLogger("intelliwatch.depth_visualizer")


class DepthVisualizer:
    """
    Visualization engine for monocular depth estimation maps and object spatial tags.
    """

    COLORMAPS = {
        "inferno": cv2.COLORMAP_INFERNO,
        "turbo": cv2.COLORMAP_TURBO,
        "viridis": cv2.COLORMAP_VIRIDIS,
        "plasma": cv2.COLORMAP_PLASMA,
        "magma": cv2.COLORMAP_MAGMA,
    }

    def __init__(
        self,
        default_colormap: str = "inferno",
        font_scale: float = 0.5,
        font_thickness: int = 1,
    ):
        self.default_colormap = default_colormap.lower()
        self.font_scale = font_scale
        self.font_thickness = font_thickness

    def colorize_depth(
        self,
        depth_map: np.ndarray,
        colormap: Optional[str] = None,
    ) -> np.ndarray:
        """
        Converts a 2D float depth map into a vibrant, high-contrast BGR colorized image.

        Args:
            depth_map: 2D numpy array of shape (H, W).
            colormap: Colormap name ('inferno', 'turbo', 'viridis', etc.).

        Returns:
            3-channel BGR image array (H, W, 3) in uint8 format.
        """
        if depth_map is None or depth_map.size == 0:
            raise ValueError("Input depth map cannot be empty.")

        cmap_name = (colormap or self.default_colormap).lower()
        cv_cmap = self.COLORMAPS.get(cmap_name, cv2.COLORMAP_INFERNO)

        # Normalize to 0-255 uint8
        d_min = float(np.min(depth_map))
        d_max = float(np.max(depth_map))
        denom = d_max - d_min

        if denom > 1e-8:
            norm_uint8 = np.clip(((depth_map - d_min) / denom) * 255.0, 0, 255).astype(np.uint8)
        else:
            norm_uint8 = np.zeros_like(depth_map, dtype=np.uint8)

        colorized = cv2.applyColorMap(norm_uint8, cv_cmap)
        return colorized

    def overlay_depth_on_image(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        alpha: float = 0.45,
        colormap: Optional[str] = None,
    ) -> np.ndarray:
        """
        Blends a colorized depth map over the original RGB/BGR image.
        """
        if image.shape[:2] != depth_map.shape[:2]:
            depth_map = cv2.resize(depth_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

        color_depth = self.colorize_depth(depth_map, colormap=colormap)
        blended = cv2.addWeighted(image, 1.0 - alpha, color_depth, alpha, 0)
        return blended

    def draw_object_depths(
        self,
        image: np.ndarray,
        object_depths: List[ObjectDepth],
        show_contact: bool = True,
    ) -> np.ndarray:
        """
        Annotates object bounding boxes, median relative depths, and ground contact depths.
        Explicitly tags '(rel)' to ensure technical honesty.
        """
        annotated = image.copy()
        h, w = annotated.shape[:2]

        for obj in object_depths:
            bbox = obj.bbox
            x1 = max(0, min(w - 1, int(round(bbox.x1))))
            y1 = max(0, min(h - 1, int(round(bbox.y1))))
            x2 = max(0, min(w - 1, int(round(bbox.x2))))
            y2 = max(0, min(h - 1, int(round(bbox.y2))))

            # Box color in cyan
            box_col = (255, 200, 0)  # BGR
            cv2.rectangle(annotated, (x1, y1), (x2, y2), box_col, 2)

            # Badge with median relative depth
            id_prefix = f"ID {obj.track_id} | " if obj.track_id is not None else ""
            cls_name = obj.class_name or "object"
            med_depth = obj.depth_stats.median_depth
            depth_badge = f"{id_prefix}{cls_name} | Rel Depth: {med_depth:.2f}"

            (tw, th), _ = cv2.getTextSize(
                depth_badge, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.85, self.font_thickness
            )
            by1 = max(0, y1 - th - 6)
            by2 = y1
            bx2 = min(w - 1, x1 + tw + 8)

            cv2.rectangle(annotated, (x1, by1), (bx2, by2), (30, 30, 30), -1)
            cv2.putText(
                annotated,
                depth_badge,
                (x1 + 4, by2 - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                self.font_scale * 0.85,
                (0, 240, 255),
                self.font_thickness,
                cv2.LINE_AA,
            )

            # Contact point dot and depth
            if show_contact and obj.contact_point is not None and obj.contact_depth is not None:
                cx, cy = obj.contact_point
                pt_x = max(0, min(w - 1, int(round(cx))))
                pt_y = max(0, min(h - 1, int(round(cy))))

                cv2.circle(annotated, (pt_x, pt_y), 5, (0, 0, 255), -1)
                cv2.circle(annotated, (pt_x, pt_y), 7, (255, 255, 255), 1)

                c_label = f"Ground: {obj.contact_depth:.2f} (rel)"
                (cw, ch), _ = cv2.getTextSize(c_label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
                c_bx2 = min(w - 1, pt_x + cw + 6)
                c_by2 = min(h - 1, pt_y + ch + 6)

                cv2.rectangle(annotated, (pt_x, pt_y), (c_bx2, c_by2), (20, 20, 20), -1)
                cv2.putText(
                    annotated,
                    c_label,
                    (pt_x + 3, c_by2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        return annotated

    def create_side_by_side(
        self,
        image: np.ndarray,
        depth_map: np.ndarray,
        object_depths: Optional[List[ObjectDepth]] = None,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        colormap: Optional[str] = None,
    ) -> np.ndarray:
        """
        Creates a side-by-side composite canvas:
        [ Left: RGB with Object Depth Badges ] | [ Right: Dense Depth Map with Legend ]
        """
        h, w = image.shape[:2]
        color_depth = self.colorize_depth(depth_map, colormap=colormap)
        if color_depth.shape[:2] != (h, w):
            color_depth = cv2.resize(color_depth, (w, h), interpolation=cv2.INTER_LINEAR)

        left_pane = image.copy()
        if object_depths:
            left_pane = self.draw_object_depths(left_pane, object_depths)

        # Header banner height
        banner_h = 40
        canvas_w = w * 2
        canvas_h = h + banner_h

        canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)

        # Render top banner
        cv2.rectangle(canvas, (0, 0), (canvas_w, banner_h), (25, 25, 25), -1)

        fid_str = f"Frame: {frame_id:04d} | " if frame_id is not None else ""
        t_str = f"Time: {timestamp:.2f}s | " if timestamp is not None else ""
        banner_title = (
            f"IntelliWatch Spatial Understanding | {fid_str}{t_str}"
            f"Monocular Relative Depth (Depth-Anything-V2-Small) | Metric: NO"
        )
        cv2.putText(
            canvas,
            banner_title,
            (16, 26),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        # Place left and right panes
        canvas[banner_h:banner_h + h, 0:w] = left_pane
        canvas[banner_h:banner_h + h, w:canvas_w] = color_depth

        # Divider line
        cv2.line(canvas, (w, banner_h), (w, canvas_h), (80, 80, 80), 2)

        # Labels on panes
        cv2.putText(canvas, "RGB Video & Sampled Depths", (16, banner_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(canvas, "Dense Monocular Relative Depth Map (Warm = Closer)", (w + 16, banner_h + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2, cv2.LINE_AA)

        return canvas

    def save_visualization(
        self,
        image: np.ndarray,
        output_path: Union[str, Path],
    ) -> Path:
        """Saves annotated canvas to destination path."""
        p = Path(output_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(p), image)
        logger.info(f"Saved depth visualization to: {p}")
        return p
