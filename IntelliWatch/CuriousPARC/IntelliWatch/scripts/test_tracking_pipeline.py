"""
Video Multi-Object Tracking Pipeline Verification Tool (Step 4)
Connects: VideoReader -> Frame Sampling -> Preprocessing -> YOLO11n -> ByteTrack -> TrackingVisualizer.
Executes multi-object tracking across video frames, maintains persistent track IDs,
renders trajectories, records an annotated output video, and prints tracking statistics.
"""
import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging_config import setup_logging
from configs.settings import get_settings
from vision.detection.yolo_detector import YOLODetector
from vision.tracking.bytetrack_tracker import ByteTrackTracker
from vision.tracking.pipeline import TrackingPipeline
from vision.tracking.visualizer import TrackingVisualizer


def main():
    parser = argparse.ArgumentParser(description="IntelliWatch Multi-Object Tracking Verification Tool")
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
        default=1,
        help="Frame sampling interval: 1 = all frames, 2 = every 2nd frame (default: 1)",
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
        help="Optional filter for specific classes, e.g. --classes person car",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--output-video",
        type=str,
        default=str(PROJECT_ROOT / "data" / "output" / "tracking_test.mp4"),
        help="Path where annotated tracking video is saved (default: data/output/tracking_test.mp4)",
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=5,
        help="Number of representative annotated frames to save (default: 5)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "output"),
        help="Directory where sample annotated frames will be saved (default: data/output)",
    )

    args = parser.parse_args()
    logger = setup_logging()
    video_file = Path(args.video_path)

    # Automatically generate synthetic test video if default is requested and missing
    if not video_file.exists() and "synthetic_test.mp4" in video_file.name:
        from scripts.create_synthetic_sample import generate_synthetic_video
        generate_synthetic_video(video_file)

    if not video_file.exists():
        logger.error(f"Specified video file does not exist: {video_file}")
        sys.exit(1)

    print("=" * 75)
    print("IntelliWatch - Multi-Object Tracking Pipeline Verification (Step 4)")
    print("=" * 75)
    print(f"Target Video Source     : {video_file}")
    print(f"YOLO Detector Model     : {args.model_path}")
    print(f"Tracking Algorithm      : ByteTrack (Modular Multi-Object Tracker)")
    print(f"Inference Device        : {args.device}")
    print(f"Confidence Threshold    : {args.confidence}")
    print(f"Frame Skip Interval     : {args.frame_skip}")
    print(f"Annotated Output Video  : {args.output_video}")
    print(f"Sample Frame Snapshots  : {args.save_frames} -> {args.output_dir}")
    print("-" * 75)

    # 1. Initialize Detector and Tracker components
    detector = YOLODetector(
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        device=args.device,
        classes=args.classes,
        map_to_original=True,
    )

    tracker = ByteTrackTracker(
        track_high_thresh=args.confidence,
        track_low_thresh=0.05,
        track_buffer=30,
        track_history_length=30,
    )

    visualizer = TrackingVisualizer()

    # 2. Build Pipeline
    pipeline = TrackingPipeline(
        video_path=video_file,
        detector=detector,
        tracker=tracker,
        frame_skip=args.frame_skip,
        visualizer=visualizer,
    )

    print("Tracking test started\n")

    def per_frame_monitor(frame_data, detections, tracks):
        if frame_data.frame_index <= 5 or frame_data.frame_index % 10 == 0:
            print(
                f"Frame: {frame_data.frame_index}\n"
                f"Detections: {len(detections.detections)}\n"
                f"Active tracks: {len(tracks.active_tracks)}\n"
            )

    summary = pipeline.process_all(
        save_video_path=args.output_video,
        save_output_dir=args.output_dir,
        max_save_frames=args.save_frames,
        log_interval=15,
        callback=per_frame_monitor,
    )

    print("-" * 75)
    print("Tracking test completed\n")
    print(f"Unique tracks observed        : {summary['unique_tracks_observed']}")
    print(f"Maximum simultaneous tracks  : {summary['max_simultaneous_tracks']}")
    print(f"Total detections evaluated    : {summary['total_detections']}")
    print(f"Total frames processed        : {summary['frames_processed']}")
    print(f"Effective throughput (FPS)    : {summary['effective_fps']} FPS")
    print(f"Class breakdown               : {summary['class_distribution']}")
    print(f"Output Video                  : {summary['output_video']}")
    print("=" * 75)
    print("MULTI-OBJECT TRACKING PIPELINE VERIFICATION COMPLETE!")
    print("=" * 75)


if __name__ == "__main__":
    main()
