import logging
from pathlib import Path
from typing import Any, List, Optional, Tuple, Union
import numpy as np

from backend.schemas.detection import FrameDetections
from backend.schemas.depth import DepthResult, ObjectDepth
from backend.schemas.tracking import FrameTracks, TrackedObject
from vision.depth.depth_estimator import DepthAnythingEstimator
from vision.depth.visualizer import DepthVisualizer
from vision.preprocessing.frame import FrameData

logger = logging.getLogger("intelliwatch.depth_pipeline")


class DepthPipeline:
    """
    Orchestrator pipeline integrating monocular depth estimation with 2D object detections
    and multi-object tracks.
    """

    def __init__(
        self,
        estimator: Optional[DepthAnythingEstimator] = None,
        visualizer: Optional[DepthVisualizer] = None,
        model_name: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.estimator = estimator or DepthAnythingEstimator(
            model_name=model_name,
            device=device,
        )
        self.visualizer = visualizer or DepthVisualizer()

    def process_frame(
        self,
        frame: Union[np.ndarray, FrameData],
        detections: Optional[Union[FrameDetections, List[Any]]] = None,
        tracks: Optional[Union[FrameTracks, List[TrackedObject]]] = None,
        visualize: bool = False,
        output_path: Optional[Union[str, Path]] = None,
    ) -> Tuple[np.ndarray, DepthResult]:
        """
        Processes a single video frame through dense depth estimation and object-level sampling.

        Args:
            frame: FrameData or raw OpenCV BGR image.
            detections: Optional 2D detections in native coordinates.
            tracks: Optional tracked objects with persistent IDs.
            visualize: Whether to render side-by-side visualization.
            output_path: Destination file to save visualization if visualize=True.

        Returns:
            Tuple of (dense 2D depth map (H, W), populated DepthResult schema).
        """
        # 1. Dense Monocular Depth Estimation
        depth_map, depth_result = self.estimator.estimate_depth(frame)

        object_depths: List[ObjectDepth] = []

        # 2. Sample depth for persistent tracks (preferred if tracking is active)
        if tracks is not None:
            track_items = tracks.active_tracks if isinstance(tracks, FrameTracks) else tracks
            for t in track_items:
                obj_d = self.estimator.sample_object_depth(
                    depth_map=depth_map,
                    bbox=t.bbox,
                    class_name=t.class_name,
                    track_id=t.track_id,
                )
                object_depths.append(obj_d)

        # 3. If no tracks provided, sample depth for raw detections
        elif detections is not None:
            det_items = detections.detections if isinstance(detections, FrameDetections) else detections
            for d in det_items:
                obj_d = self.estimator.sample_object_depth(
                    depth_map=depth_map,
                    bbox=d.bbox,
                    class_name=d.class_name,
                    track_id=None,
                )
                object_depths.append(obj_d)

        depth_result.object_depths = object_depths

        # 4. Optional Visualization Export
        if visualize and output_path is not None:
            raw_img = frame.image if isinstance(frame, FrameData) else frame
            fid = depth_result.frame_id
            t_sec = depth_result.timestamp
            composite = self.visualizer.create_side_by_side(
                image=raw_img,
                depth_map=depth_map,
                object_depths=object_depths,
                frame_id=fid,
                timestamp=t_sec,
            )
            self.visualizer.save_visualization(composite, output_path)
            depth_result.depth_map_path = str(output_path)

        return depth_map, depth_result
