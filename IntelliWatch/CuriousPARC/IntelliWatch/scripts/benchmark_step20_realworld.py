"""
Step 20 Real-World Benchmarking and End-to-End Pipeline Evaluation Script.

Evaluates:
  1. Hardware & Runtime Diagnostics (RTX 5050 Laptop GPU, sm_120, PyTorch cu126, safe fallback).
  2. End-to-End Pipeline on real 60-frame moving CCTV worker video.
  3. Latency & Throughput profiling across all 60 frames.
  4. Accuracy metrics on authentic imagery without inventing unsupported metrics.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Configure paths
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.schemas.detection import BoundingBox, DetectionResult
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from vision.detection.ppe_association import PPEAssociationEngine
from vision.detection.ppe_detector import PPEDetector
from vision.detection.yolo_detector import YOLODetector
from vision.utils.device import get_device_diagnostics, resolve_device

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_step20")


def run_benchmark():
    logger.info("=== STEP 20: REAL-WORLD ACCURACY & PERFORMANCE BENCHMARK ===")

    # 1. Hardware Diagnostics
    diag = get_device_diagnostics()
    logger.info(f"PyTorch Version: {diag['torch_version']}")
    logger.info(f"CUDA Available: {diag['cuda_available']}")
    logger.info(f"Device Name: {diag['device_name']}")
    logger.info(f"Compute Capability: {diag['compute_capability']}")
    logger.info(f"CUDA Kernel Executable: {diag['cuda_kernel_executable']}")
    logger.info(f"Error Reason: {diag['error_reason']}")
    logger.info(f"Optimal / Resolved Device: {diag['optimal_device']}")

    # 2. Authentic Image Perception Evaluation
    image_path = ROOT_DIR / "data" / "samples" / "industrial_cctv.jpg"
    assert image_path.exists(), f"Image {image_path} missing!"

    img = cv2.imread(str(image_path))
    h, w = img.shape[:2]
    logger.info(f"Evaluating Perception Models on authentic image: {w}x{h}")

    yolo = YOLODetector(device="cpu")
    t0 = time.perf_counter()
    yolo_dets = yolo.detect(img)
    yolo_time_ms = (time.perf_counter() - t0) * 1000

    ppe_model = PPEDetector(device="cpu")
    t0 = time.perf_counter()
    ppe_dets = ppe_model.detect(img)
    ppe_time_ms = (time.perf_counter() - t0) * 1000

    logger.info(f"YOLO11n detected {len(yolo_dets.detections)} objects in {yolo_time_ms:.2f} ms")
    for d in yolo_dets.detections:
        logger.info(f"  - {d.class_name}: conf={d.confidence:.3f}, bbox=[{d.bbox.x1:.1f}, {d.bbox.y1:.1f}, {d.bbox.x2:.1f}, {d.bbox.y2:.1f}]")

    logger.info(f"PPE YOLOv8n detected {len(ppe_dets.detections)} items in {ppe_time_ms:.2f} ms")
    for d in ppe_dets.detections:
        logger.info(f"  - {d.class_name}: conf={d.confidence:.3f}, bbox=[{d.bbox.x1:.1f}, {d.bbox.y1:.1f}, {d.bbox.x2:.1f}, {d.bbox.y2:.1f}]")

    # 3. Video Sequence End-to-End Processing & Throughput Profiling
    video_path = ROOT_DIR / "data" / "samples" / "cctv_worker_moving.mp4"
    assert video_path.exists(), f"Video {video_path} missing!"

    cap = cv2.VideoCapture(str(video_path))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps_video = cap.get(cv2.CAP_PROP_FPS)
    logger.info(f"Processing real CCTV video sequence: {total_video_frames} frames @ {fps_video} FPS")

    orchestrator = EndToEndPipelineOrchestrator(
        enable_depth_model=False,
        enable_industrial_model=False,
        enable_ppe_model=True,
        device="cpu",
    )

    frame_idx = 0
    latencies = []
    tracks_seen = set()
    ppe_inventories_evaluated = 0
    violations_detected = 0

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        t_start = time.perf_counter()
        assessment, annotated = orchestrator.process_frame(
            frame=frame,
            frame_id=frame_idx,
            timestamp=frame_idx / 30.0,
        )
        latency_ms = (time.perf_counter() - t_start) * 1000
        latencies.append(latency_ms)

        for tr in assessment.tracks:
            tracks_seen.add(tr.track_id)

        for inv in assessment.worker_inventories:
            ppe_inventories_evaluated += 1
            if getattr(inv, "compliance_status", None) == "NON_COMPLIANT":
                violations_detected += 1

        frame_idx += 1

    cap.release()

    # Latency & Throughput metrics
    # Exclude warmup frame 0 for steady-state calculation
    steady_latencies = latencies[1:] if len(latencies) > 1 else latencies
    mean_latency = float(np.mean(steady_latencies))
    median_latency = float(np.median(steady_latencies))
    p95_latency = float(np.percentile(steady_latencies, 95))
    fps_throughput = 1000.0 / mean_latency if mean_latency > 0 else 0.0

    report = {
        "hardware_diagnostics": diag,
        "perception_evaluation": {
            "image_resolution": [w, h],
            "yolo_detection_count": len(yolo_dets.detections),
            "yolo_latency_ms": round(yolo_time_ms, 2),
            "yolo_detections": [
                {"class": d.class_name, "confidence": round(d.confidence, 4)}
                for d in yolo_dets.detections
            ],
            "ppe_detection_count": len(ppe_dets.detections),
            "ppe_latency_ms": round(ppe_time_ms, 2),
            "ppe_detections": [
                {"class": d.class_name, "confidence": round(d.confidence, 4)}
                for d in ppe_dets.detections
            ],
        },
        "video_pipeline_benchmark": {
            "total_frames_processed": frame_idx,
            "unique_persistent_tracks": len(tracks_seen),
            "track_ids": sorted(list(tracks_seen)),
            "worker_inventories_evaluated": ppe_inventories_evaluated,
            "violations_detected": violations_detected,
            "latency_ms": {
                "warmup_frame_0": round(latencies[0], 2) if latencies else 0.0,
                "mean_steady_state": round(mean_latency, 2),
                "median": round(median_latency, 2),
                "p95": round(p95_latency, 2),
            },
            "throughput_fps": round(fps_throughput, 2),
        },
    }

    out_file = ROOT_DIR / "data" / "output" / "step20_benchmark_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info("=== BENCHMARK RESULTS SUMMARY ===")
    logger.info(f"Total Frames Processed: {frame_idx}")
    logger.info(f"Unique Track IDs (Persistence): {sorted(list(tracks_seen))}")
    logger.info(f"Steady-State Latency: Mean={mean_latency:.2f}ms, Median={median_latency:.2f}ms, p95={p95_latency:.2f}ms")
    logger.info(f"Throughput: {fps_throughput:.2f} FPS")
    logger.info(f"Report saved to: {out_file}")


if __name__ == "__main__":
    run_benchmark()
