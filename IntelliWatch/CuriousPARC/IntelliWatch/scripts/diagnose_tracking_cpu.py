"""
Short diagnostic benchmark for Step 4 YOLO11n + ByteTrack on CPU.
Runs 15 frames to measure CPU inference time, ByteTrack association time,
and verify tracking pipeline correctness without hanging.
"""
import sys
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import get_settings
from vision.detection.yolo_detector import YOLODetector
from vision.preprocessing.frame_processor import FramePreprocessor
from vision.preprocessing.video_reader import VideoReader
from vision.tracking.bytetrack_tracker import ByteTrackTracker
from vision.tracking.visualizer import TrackingVisualizer


def generate_short_test_video(path: Path, num_frames: int = 20) -> Path:
    """Creates a short 20-frame test video from sample_test.jpg."""
    path.parent.mkdir(parents=True, exist_ok=True)
    img_sample = PROJECT_ROOT / "data" / "samples" / "sample_test.jpg"
    if not img_sample.exists():
        raise FileNotFoundError(f"Missing sample image: {img_sample}")

    base_img = cv2.imread(str(img_sample))
    h, w = base_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(path), fourcc, 30.0, (w, h))

    for i in range(num_frames):
        # Smooth horizontal motion across the CCTV frame
        shift_x = int(i * 4)
        shift_y = int(3 * np.sin(i * 0.2))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        frame = cv2.warpAffine(base_img, M, (w, h))
        out.write(frame)

    out.release()
    return path


def run_cpu_diagnostic(max_frames: int = 15):
    settings = get_settings()
    video_path = PROJECT_ROOT / "data" / "samples" / "short_diagnostic.mp4"
    generate_short_test_video(video_path, num_frames=max_frames)

    print("=" * 70)
    print("IntelliWatch - Step 4 CPU-Only Diagnostic Benchmark")
    print("=" * 70)
    print(f"Device Configuration: CPU (CUDA is neither used nor required)")
    print(f"Test Video: {video_path.name} ({max_frames} frames)")

    # 1. Initialize components
    init_start = time.perf_counter()
    from vision.preprocessing.pipeline import VideoPipeline

    preprocessor = FramePreprocessor(
        target_width=settings.TARGET_WIDTH,
        target_height=settings.TARGET_HEIGHT,
        resize_enabled=True,
        preserve_aspect_ratio=True,
    )

    video_pipe = VideoPipeline(
        video_path=video_path,
        frame_skip=1,
        preprocessor=preprocessor,
    )

    detector = YOLODetector(
        model_path="weights/yolo11n.pt",
        confidence_threshold=0.25,
        device="cpu",
    )

    tracker = ByteTrackTracker(
        track_high_thresh=0.25,
        track_low_thresh=0.05,
        track_buffer=30,
        track_history_length=30,
        frame_rate=30,
    )

    visualizer = TrackingVisualizer()
    init_elapsed = time.perf_counter() - init_start
    print(f"Component Initialization Time: {init_elapsed:.3f}s (YOLO11n + ByteTrack loaded once)")

    # 2. Sequential frame benchmark
    preprocess_times = []
    yolo_times = []
    track_times = []
    viz_times = []
    active_track_history = []
    unique_ids_seen = set()

    benchmark_start = time.perf_counter()

    for idx, frame_data in enumerate(video_pipe.stream_frames()):
        if idx >= max_frames:
            break

        # Stage B: YOLO11n Inference on CPU
        t2 = time.perf_counter()
        detections = detector.detect(frame_data)
        t3 = time.perf_counter()
        yolo_times.append(t3 - t2)

        # Stage C: ByteTrack Association on CPU
        t4 = time.perf_counter()
        tracks = tracker.update(detections)
        t5 = time.perf_counter()
        track_times.append(t5 - t4)

        # Stage D: Visualization
        t6 = time.perf_counter()
        annotated = visualizer.draw_tracks(frame_data.image, tracks)
        t7 = time.perf_counter()
        viz_times.append(t7 - t6)

        num_tracks = len(tracks.active_tracks)
        active_track_history.append(num_tracks)
        for t in tracks.active_tracks:
            unique_ids_seen.add(t.track_id)

        print(
            f"Frame {idx:02d} | Dets: {len(detections.detections)} | "
            f"Active Tracks: {num_tracks} | "
            f"YOLO: {(t3 - t2)*1000:.1f}ms | Track: {(t5 - t4)*1000:.2f}ms"
        )

    total_elapsed = time.perf_counter() - benchmark_start
    num_processed = len(yolo_times)
    overall_fps = num_processed / total_elapsed if total_elapsed > 0 else 0.0

    avg_yolo = np.mean(yolo_times) * 1000
    avg_track = np.mean(track_times) * 1000
    avg_viz = np.mean(viz_times) * 1000

    print("-" * 70)
    print("BENCHMARK RESULTS (CPU-ONLY):")
    print(f"  Frames processed         : {num_processed}")
    print(f"  Total processing time    : {total_elapsed:.3f}s")
    print(f"  Overall pipeline speed   : {overall_fps:.1f} FPS")
    print(f"  Average YOLO11n (CPU)    : {avg_yolo:.1f} ms/frame")
    print(f"  Average ByteTrack (CPU)  : {avg_track:.2f} ms/frame")
    print(f"  Average Visualization    : {avg_viz:.1f} ms/frame")
    print(f"  Unique track IDs created : {len(unique_ids_seen)} (IDs: {sorted(list(unique_ids_seen))})")
    print("=" * 70)

    # Save representative annotated sample frame
    out_sample = PROJECT_ROOT / "data" / "output" / "diagnostic_step4_cpu.jpg"
    visualizer.save_annotated_frame(annotated, out_sample)
    print(f"Sample frame saved to: {out_sample}")


if __name__ == "__main__":
    run_cpu_diagnostic(max_frames=15)
