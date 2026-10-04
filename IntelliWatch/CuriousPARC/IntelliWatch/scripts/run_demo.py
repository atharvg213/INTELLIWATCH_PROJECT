"""
scripts/run_demo.py
Step 12 Demo & Video Processing Runner.

Executes the full IntelliWatch AI Industrial Safety & Scene Understanding pipeline
across video frames, generates annotated output, logs confirmed safety incidents,
and outputs structured event and risk assessments.

Usage:
    python scripts/run_demo.py
    python scripts/run_demo.py --input data/samples/cctv_worker_moving.mp4 --max-frames 60
    python scripts/run_demo.py --input data/samples/synthetic_test.mp4 --output data/output/demo_output.mp4
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging_config import setup_logging
from configs.settings import get_settings
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from backend.services.incident_store import get_incident_store


def parse_args():
    parser = argparse.ArgumentParser(
        description="IntelliWatch AI Industrial Safety & Scene Understanding - End-to-End Demo Runner"
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        default=None,
        help="Path to input video file (defaults to data/samples/cctv_worker_moving.mp4 or synthetic_test.mp4)",
    )
    parser.add_argument(
        "--output-video",
        "-o",
        type=str,
        default=str(PROJECT_ROOT / "data" / "output" / "demo_annotated.mp4"),
        help="Path to save annotated output video (default: data/output/demo_annotated.mp4)",
    )
    parser.add_argument(
        "--output-json",
        type=str,
        default=str(PROJECT_ROOT / "data" / "output" / "demo_results.json"),
        help="Path to save structured assessment summary JSON (default: data/output/demo_results.json)",
    )
    parser.add_argument(
        "--max-frames",
        "-m",
        type=int,
        default=60,
        help="Maximum number of frames to process (default: 60)",
    )
    parser.add_argument(
        "--sample-interval",
        "-s",
        type=int,
        default=1,
        help="Frame sampling interval: 1 = every frame, 2 = every 2nd frame (default: 1)",
    )
    parser.add_argument(
        "--confidence",
        "-c",
        type=float,
        default=0.25,
        help="Detection confidence threshold between 0.0 and 1.0 (default: 0.25)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--disable-ppe-model",
        action="store_true",
        help="Disable second-stage PPE detector model to maximize processing speed on low-end CPUs",
    )
    return parser.parse_args()


def get_or_create_demo_video(input_arg: str = None) -> Path:
    """Resolves input video path or creates deterministic synthetic video if needed."""
    if input_arg:
        target = Path(input_arg)
        if not target.is_absolute():
            target = PROJECT_ROOT / target
        if target.exists():
            return target
        else:
            print(f"Warning: Specified input video not found: {target}")

    # Check default candidate sample files
    candidates = [
        PROJECT_ROOT / "data" / "samples" / "cctv_worker_moving.mp4",
        PROJECT_ROOT / "data" / "samples" / "short_diagnostic.mp4",
        PROJECT_ROOT / "data" / "samples" / "synthetic_test.mp4",
    ]
    for c in candidates:
        if c.exists():
            return c

    # If no sample video exists, deterministically generate synthetic video
    print("No sample video found. Generating synthetic industrial CCTV sample...")
    synth_path = PROJECT_ROOT / "data" / "samples" / "synthetic_test.mp4"
    from scripts.create_synthetic_sample import generate_synthetic_video
    generate_synthetic_video(synth_path, width=640, height=360, fps=20, num_frames=60)
    return synth_path


def main():
    args = parse_args()
    logger = setup_logging()

    video_path = get_or_create_demo_video(args.input)
    output_video_path = Path(args.output_video)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)
    output_json_path = Path(args.output_json)
    output_json_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("INTELLIWATCH - AI INDUSTRIAL SAFETY & SCENE UNDERSTANDING")
    print("End-to-End Demo & Video Processing Pipeline (Step 12)")
    print("=" * 80)
    print(f"Input Video Source      : {video_path}")
    print(f"Annotated Output Video  : {output_video_path}")
    print(f"Structured JSON Export  : {output_json_path}")
    print(f"Target Max Frames       : {args.max_frames}")
    print(f"Frame Sampling Interval : {args.sample_interval}")
    print(f"Perception Device       : {args.device.upper()}")
    print("=" * 80)

    # Open video capture
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Failed to open video source: {video_path}")
        sys.exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 360
    total_src_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Source Stream Specs     : {src_w}x{src_h} @ {fps:.1f} FPS ({total_src_frames} total frames)")
    print("-" * 80)

    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_video_path), fourcc, fps, (src_w, src_h))

    # Initialize orchestrator
    orchestrator = EndToEndPipelineOrchestrator(
        device=args.device,
        enable_ppe_model=not args.disable_ppe_model,
        enable_depth_model=False,
    )
    incident_store = get_incident_store()

    frame_idx = 0
    processed_count = 0
    assessments_summary = []
    t_start_total = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                break

            frame_idx += 1
            if frame_idx % args.sample_interval != 0:
                continue

            current_timestamp = frame_idx / fps

            # Process frame through full intelligence pipeline
            assessment, annotated_frame = orchestrator.process_frame(
                frame=frame,
                frame_id=frame_idx,
                timestamp=current_timestamp,
            )

            # Write annotated frame
            if writer.isOpened():
                # Ensure dimensions match writer
                if (annotated_frame.shape[1], annotated_frame.shape[0]) != (src_w, src_h):
                    annotated_frame = cv2.resize(annotated_frame, (src_w, src_h))
                writer.write(annotated_frame)

            processed_count += 1

            # Log frame progress
            active_events_cnt = len(assessment.active_events)
            active_warn_cnt = len(assessment.active_early_warnings)
            tracks_cnt = assessment.total_active_tracks
            risk_tier = assessment.highest_risk_level.value

            print(
                f"[Frame {frame_idx:04d} | {current_timestamp:.2f}s] "
                f"Tracks: {tracks_cnt:02d} | "
                f"Risk: {risk_tier:<8} (Score: {assessment.highest_risk_score:.1f}) | "
                f"Events: {active_events_cnt} | "
                f"Warnings: {active_warn_cnt} | "
                f"Latency: {assessment.processing_time_ms:.1f}ms"
            )

            # Store structured summary item
            assessments_summary.append({
                "frame_id": frame_idx,
                "timestamp": round(current_timestamp, 3),
                "tracks_count": tracks_cnt,
                "highest_risk_level": risk_tier,
                "highest_risk_score": round(assessment.highest_risk_score, 1),
                "active_events": [
                    {
                        "event_type": ev.event_type.value,
                        "risk_level": ev.risk_level.value,
                        "risk_score": ev.risk_score,
                        "entities": ev.involved_entities,
                        "explanation": ev.explanation,
                    }
                    for ev in assessment.active_events
                ],
                "active_early_warnings": [
                    {
                        "indicator_type": w.indicator_type.value,
                        "severity": w.severity.value,
                        "explanation": w.explanation,
                    }
                    for w in assessment.active_early_warnings
                ],
                "new_incident_ids": assessment.new_incident_ids,
            })

            if processed_count >= args.max_frames:
                print(f"\nReached configured max frames limit ({args.max_frames}). Stopping demo.")
                break

    finally:
        cap.release()
        if writer.isOpened():
            writer.release()

    t_end_total = time.perf_counter()
    total_elapsed_s = t_end_total - t_start_total
    perf = orchestrator.get_performance_stats()

    # Save structured summary JSON
    summary_report = {
        "metadata": {
            "source_video": str(video_path),
            "output_video": str(output_video_path),
            "processed_frames": processed_count,
            "total_runtime_seconds": round(total_elapsed_s, 2),
            "device": perf["device"],
            "average_latency_ms": perf["avg_latency_ms"],
            "effective_fps": perf["processing_fps"],
            "total_recorded_incidents": incident_store.total_count(),
        },
        "incidents": [
            inc.model_dump(mode="json")
            for inc in incident_store.list_incidents(limit=20)
        ],
        "frame_assessments": assessments_summary,
    }

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(summary_report, f, indent=2)

    print("\n" + "=" * 80)
    print("DEMO RUN COMPLETE — EXECUTION SUMMARY")
    print("=" * 80)
    print(f"Frames Processed       : {processed_count}")
    print(f"Total Processing Time  : {total_elapsed_s:.2f} seconds")
    print(f"Average Latency / Frame: {perf['avg_latency_ms']} ms")
    print(f"Processing Throughput  : {perf['processing_fps']} FPS (CPU Mode)")
    print(f"Recorded Incidents     : {incident_store.total_count()}")
    print(f"Annotated Video Output : {output_video_path}")
    print(f"Structured JSON Export : {output_json_path}")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
