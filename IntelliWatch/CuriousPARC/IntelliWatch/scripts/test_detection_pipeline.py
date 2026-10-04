"""
Video Object Detection Pipeline Verification Tool (Step 3)
Connects VideoReader -> Sampling -> Preprocessing -> YOLODetector -> Coordinate Mapping -> Visualizer.
Runs inference across sampled video frames and saves sample annotated frames to data/output/.
"""
import argparse
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging_config import setup_logging
from configs.settings import get_settings
from vision.detection.pipeline import DetectionPipeline
from vision.detection.visualizer import DetectionVisualizer
from vision.detection.yolo_detector import YOLODetector
from vision.preprocessing.frame_processor import FramePreprocessor


def main():
    parser = argparse.ArgumentParser(description="IntelliWatch Video Detection Verification Tool")
    parser.add_argument(
        "video_path",
        nargs="?",
        default=str(PROJECT_ROOT / "data" / "samples" / "synthetic_test.mp4"),
        help="Path to the video file to process (defaults to data/samples/synthetic_test.mp4)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/yolo11n.pt",
        help="Path to YOLO weights (default: weights/yolo11n.pt)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=5,
        help="Frame sampling interval: 1 = all frames, 5 = every 5th frame (default: 5)",
    )
    parser.add_argument(
        "--conf",
        "--confidence",
        type=float,
        default=0.25,
        dest="confidence",
        help="Detection confidence threshold between 0.0 and 1.0 (default: 0.25)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional filter for specific classes, e.g. --classes person truck",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=3,
        help="Number of annotated sample frames to save to output directory (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "output"),
        help="Directory where annotated frames will be saved (default: data/output)",
    )

    args = parser.parse_args()
    logger = setup_logging()
    video_file = Path(args.video_path)

    # Automatically generate synthetic video if default path is requested and missing
    if not video_file.exists() and "synthetic_test.mp4" in video_file.name:
        from scripts.create_synthetic_sample import generate_synthetic_video
        generate_synthetic_video(video_file)

    if not video_file.exists():
        logger.error(f"Specified video file does not exist: {video_file}")
        sys.exit(1)

    print("=" * 75)
    print("IntelliWatch - Object Detection Pipeline Verification (Step 3)")
    print("=" * 75)
    print(f"Target Video Source  : {video_file}")
    print(f"YOLO Model Weights   : {args.model_path}")
    print(f"Inference Device     : {args.device}")
    print(f"Confidence Threshold : {args.confidence}")
    print(f"Class Filter         : {args.classes or 'ALL (No Filter)'}")
    print(f"Frame Sampling Skip  : {args.frame_skip}")
    print(f"Annotated Frames Out : {args.save_frames} -> {args.output_dir}")
    print("-" * 75)

    # Initialize Detector
    detector = YOLODetector(
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        device=args.device,
        classes=args.classes,
        map_to_original=True,
    )
    visualizer = DetectionVisualizer()

    # Initialize Pipeline
    pipeline = DetectionPipeline(
        video_path=video_file,
        detector=detector,
        frame_skip=args.frame_skip,
        visualizer=visualizer,
    )

    print(f"[1] Model Loaded Successfully: {len(detector.class_names)} COCO classes available.")
    print("[2] Executing Object Detection Pipeline across frames...")

    summary = pipeline.process_all(
        save_output_dir=args.output_dir,
        max_save_frames=args.save_frames,
        log_interval=10,
    )

    print("-" * 75)
    print("[3] Object Detection Pipeline Summary:")
    print(f"    - Frames Ingested & Evaluated : {summary['frames_processed']}")
    print(f"    - Total Objects Detected      : {summary['total_detections']}")
    print(f"    - Average Detections / Frame  : {summary['avg_detections_per_frame']}")
    print(f"    - Class Counts Breakdown      : {summary['class_distribution']}")
    print(f"    - Total Elapsed Time          : {summary['elapsed_seconds']}s")
    print(f"    - Effective Throughput (FPS)  : {summary['effective_fps']} FPS")
    print(f"    - Visual Frames Saved         : {summary['saved_frames']} in {args.output_dir}")
    print("=" * 75)
    print("DETECTION PIPELINE VERIFICATION COMPLETE!")
    print("=" * 75)


if __name__ == "__main__":
    main()
