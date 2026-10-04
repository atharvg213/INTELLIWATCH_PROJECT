"""
IntelliWatch - Monocular Depth Estimation & Spatial Reasoning Verification Script (Step 7)

Demonstrates the 2D-to-3D Spatial Perception Layer:
  1. Initialization of Depth-Anything-V2-Small model once in CPU memory.
  2. Dense relative monocular depth map generation.
  3. Spatial alignment preserving native CCTV resolution (e.g. 1376x768).
  4. Object-level depth distribution sampling (mean, median, percentiles).
  5. Ground contact point relative depth sampling (worker feet).
  6. High-contrast side-by-side visualization with colorized relative depth.
  7. CPU execution benchmark reporting inference latency and FPS.
"""
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging_config import setup_logging
from configs.settings import get_settings
from vision.depth.depth_estimator import DepthAnythingEstimator
from vision.depth.depth_pipeline import DepthPipeline
from vision.depth.visualizer import DepthVisualizer
from vision.detection.yolo_detector import YOLODetector


def run_depth_benchmark(
    source_path: Path,
    num_frames: int = 5,
    device: str = "cpu",
    output_path: Path = Path("data/output/test_depth_output.jpg"),
    run_detection: bool = True,
):
    """
    Executes depth estimation benchmark on a single image or short frame sequence.
    """
    print("=" * 65)
    print("IntelliWatch - Monocular Depth Estimation Benchmark (Step 7)")
    print(f"Target Device: {device.upper()} (CPU-Optimized)")
    print(f"Input Source: {source_path}")
    print("=" * 65)

    settings = get_settings()

    # 1. Load model once
    t_load_start = time.perf_counter()
    estimator = DepthAnythingEstimator(
        model_name=settings.DEPTH_MODEL_NAME,
        device=device,
        input_size=settings.DEPTH_INPUT_SIZE,
    )
    load_time_sec = time.perf_counter() - t_load_start
    print(f"\nModel Loaded: {settings.DEPTH_MODEL_NAME}")
    print(f"Model Load Time: {load_time_sec:.2f}s (Loaded once)")

    # 2. Initialize Visualizer & Pipeline
    visualizer = DepthVisualizer(default_colormap="inferno")
    pipeline = DepthPipeline(estimator=estimator, visualizer=visualizer)

    # 3. Optional YOLO Detector for Object-Level Depth Sampling
    detector = None
    if run_detection:
        try:
            detector = YOLODetector()
            print("YOLO11n Detector: Active for Object Depth Association")
        except Exception as e:
            print(f"YOLO11n Detector unavailable ({e}), proceeding without detection.")

    # 4. Ingest Frame(s)
    frames: list[np.ndarray] = []
    if source_path.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]:
        img = cv2.imread(str(source_path))
        if img is None:
            raise FileNotFoundError(f"Failed to read image at: {source_path}")
        # Repeat image for multi-frame benchmark if requested
        frames = [img.copy() for _ in range(num_frames)]
    else:
        # Video file: sample up to num_frames
        cap = cv2.VideoCapture(str(source_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Failed to open video file at: {source_path}")
        while len(frames) < num_frames:
            ret, frame = cap.read()
            if not ret or frame is None:
                break
            frames.append(frame)
        cap.release()

    if not frames:
        raise ValueError("No frames available for depth benchmark.")

    h_orig, w_orig = frames[0].shape[:2]
    print(f"Native Resolution: {w_orig}x{h_orig}")
    print(f"Frames to Test: {len(frames)}")

    latencies_ms: list[float] = []
    last_depth_map = None
    last_result = None
    last_frame = None

    print("\n--- Running Inference ---")
    for idx, frame in enumerate(frames):
        detections = None
        if detector is not None:
            try:
                detections = detector.detect(frame, frame_id=idx, timestamp=idx * 0.033)
            except Exception as ex:
                pass

        t0 = time.perf_counter()
        depth_map, result = pipeline.process_frame(
            frame=frame,
            detections=detections,
            visualize=False,
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies_ms.append(elapsed_ms)

        last_depth_map = depth_map
        last_result = result
        last_frame = frame

        det_count = len(result.object_depths)
        print(f"  Frame {idx + 1}/{len(frames)}: {elapsed_ms:.1f}ms | Objects: {det_count} | Depth Range: [{result.min_depth:.2f}, {result.max_depth:.2f}]")

    # 5. Object Depth Sampling Summary
    if last_result and last_result.object_depths:
        print("\n--- Sampled Object Depths (Last Frame) ---")
        for obj in last_result.object_depths:
            cls = obj.class_name or "object"
            stats = obj.depth_stats
            print(f"  Object: {cls} | Median Depth: {stats.median_depth:.2f} (rel) | Range: [{stats.min_depth:.2f}, {stats.max_depth:.2f}]")
            if obj.contact_depth is not None:
                print(f"    Contact Point: {obj.contact_point} -> Contact Depth: {obj.contact_depth:.2f} (rel)")

    # 6. Save Composite Visual Verification Snapshot
    if last_frame is not None and last_depth_map is not None and last_result is not None:
        composite = visualizer.create_side_by_side(
            image=last_frame,
            depth_map=last_depth_map,
            object_depths=last_result.object_depths,
            frame_id=last_result.frame_id or 1,
            timestamp=last_result.timestamp,
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        visualizer.save_visualization(composite, output_path)
        print(f"\nSaved depth visualization to: {output_path}")

    # 7. Print Final Benchmark Summary
    avg_latency = float(np.mean(latencies_ms))
    approx_fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0

    print("\n" + "=" * 65)
    print("IntelliWatch Monocular Depth Benchmark Results")
    print("=" * 65)
    print(f"Depth model: {settings.DEPTH_MODEL_NAME}")
    print(f"Device: {device.upper()}")
    print(f"Input: {settings.DEPTH_INPUT_SIZE}x{settings.DEPTH_INPUT_SIZE}")
    print(f"Output: {w_orig}x{h_orig}")
    print(f"Frames tested: {len(frames)}")
    print(f"Average inference: {avg_latency:.1f} ms/frame")
    print(f"Approx FPS: {approx_fps:.1f}")
    print("Relative depth: YES")
    print("Metric depth: NO")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="IntelliWatch Monocular Depth Benchmark")
    parser.add_argument(
        "--source",
        type=str,
        default="data/samples/sample_test.jpg",
        help="Path to sample image or video file",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=5,
        help="Number of frames to benchmark (default: 5)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Execution device: 'cpu' (default: cpu)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/test_depth_output.jpg",
        help="Output path for annotated composite canvas",
    )
    parser.add_argument(
        "--no-detection",
        action="store_true",
        help="Disable YOLO object detection integration",
    )
    args = parser.parse_args()

    setup_logging()
    run_depth_benchmark(
        source_path=Path(args.source),
        num_frames=args.frames,
        device=args.device,
        output_path=Path(args.output),
        run_detection=not args.no_detection,
    )


if __name__ == "__main__":
    main()
