"""
Video Pipeline Verification Script
Accepts a video path, opens it using VideoPipeline, extracts metadata,
processes frames with sampling and preprocessing, and optionally saves
visual sample frames to data/output/.
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
from vision.preprocessing.frame_processor import FramePreprocessor
from vision.preprocessing.pipeline import VideoPipeline
from vision.preprocessing.video_reader import VideoReader


def main():
    parser = argparse.ArgumentParser(description="IntelliWatch Video Pipeline Verification Tool")
    parser.add_argument(
        "video_path",
        nargs="?",
        default=str(PROJECT_ROOT / "data" / "samples" / "synthetic_test.mp4"),
        help="Path to the video file to process (defaults to data/samples/synthetic_test.mp4)",
    )
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Frame sampling interval: 1 = all frames, 5 = every 5th frame (default: 1)",
    )
    parser.add_argument(
        "--target-width",
        type=int,
        default=640,
        help="Target preprocessed width (default: 640)",
    )
    parser.add_argument(
        "--target-height",
        type=int,
        default=640,
        help="Target preprocessed height (default: 640)",
    )
    parser.add_argument(
        "--no-resize",
        action="store_true",
        help="Disable frame resizing",
    )
    parser.add_argument(
        "--save-frames",
        type=int,
        default=3,
        help="Number of processed sample frames to save to data/output/ for visual check (default: 3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "output"),
        help="Directory where sample frames will be saved (default: data/output)",
    )

    args = parser.parse_args()
    logger = setup_logging()
    video_file = Path(args.video_path)

    # If the default synthetic video does not exist, create it automatically
    if not video_file.exists() and "synthetic_test.mp4" in video_file.name:
        from scripts.create_synthetic_sample import generate_synthetic_video
        generate_synthetic_video(video_file)

    print("=" * 70)
    print("IntelliWatch - Video Input & Frame Processing Verification")
    print("=" * 70)
    print(f"Target Video Source : {video_file}")
    print(f"Frame Skip Interval : {args.frame_skip}")
    print(f"Target Dimensions   : {args.target_width}x{args.target_height} (Resize: {not args.no_resize})")
    print(f"Sample Frames to Save: {args.save_frames}")
    print("-" * 70)

    # 1. Step 1: Ingest and report metadata
    reader = VideoReader(video_file)
    metadata = reader.open()
    reader.release()

    print("[1] Video Metadata Inspection:")
    print(f"    - Resolution   : {metadata.width} x {metadata.height}")
    print(f"    - Stream FPS   : {metadata.fps:.2f}")
    print(f"    - Total Frames : {metadata.total_frames}")
    print(f"    - Duration     : {metadata.duration_sec:.2f} seconds")
    print("-" * 70)

    # 2. Step 2: Configure preprocessor and pipeline
    preprocessor = FramePreprocessor(
        target_width=args.target_width,
        target_height=args.target_height,
        resize_enabled=not args.no_resize,
        preserve_aspect_ratio=True,
    )
    pipeline = VideoPipeline(
        video_path=video_file,
        frame_skip=args.frame_skip,
        preprocessor=preprocessor,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    saved_count = 0

    def frame_callback(frame_data):
        nonlocal saved_count
        if saved_count < args.save_frames:
            out_file = output_dir / f"step2_sample_frame_{frame_data.frame_index:04d}.jpg"
            cv2.imwrite(str(out_file), frame_data.image)
            print(f"    -> Saved visual verification frame: {out_file.name} (shape: {frame_data.shape})")
            saved_count += 1

    print("[2] Processing Video Frames...")
    summary = pipeline.process_all(callback=frame_callback, log_interval=15)

    print("-" * 70)
    print("[3] Execution Summary:")
    print(f"    - Total Frames Processed: {summary['frames_processed']}")
    print(f"    - Total Processing Time : {summary['elapsed_seconds']}s")
    print(f"    - Effective Throughput  : {summary['effective_fps']} FPS")
    print(f"    - Output Saved Frames   : {saved_count} frames in {output_dir}")
    print("=" * 70)
    print("VIDEO PIPELINE VERIFICATION SUCCESSFUL!")
    print("=" * 70)


if __name__ == "__main__":
    main()
