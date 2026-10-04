"""
scripts/evaluate_step19_accuracy.py
====================================
Step 19 Evaluation and Benchmarking Tool:
Hardware Diagnostics, GPU/CPU Fallback, and Perception Accuracy Verification.

Executes and measures:
1. Hardware diagnostics & CUDA kernel compatibility verification.
2. EndToEndPipelineOrchestrator device initialization and CPU fallback telemetry.
3. Worker class compatibility and PPE assessment verification.
4. ByteTrack class evidence voting evaluation.
5. Multi-frame sequence benchmark (FPS, latency, incident detection).
"""
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from vision.detection.ppe_detector import PPEDetector
from vision.detection.yolo_detector import YOLODetector
from vision.tracking.bytetrack_tracker import ByteTrackTracker
from vision.utils.device import get_device_diagnostics, resolve_device


def run_hardware_diagnostics():
    print("=" * 80)
    print("STEP 19 PHASE 1: HARDWARE & ACCELERATION DIAGNOSTICS")
    print("=" * 80)
    diag = get_device_diagnostics()
    for k, v in diag.items():
        print(f"  {k:<26}: {v}")
    resolved_auto = resolve_device("auto")
    print(f"  {'resolved_auto_device':<26}: {resolved_auto}")
    print("=" * 80)
    return diag


def run_orchestrator_worker_ppe_test():
    print("\n" + "=" * 80)
    print("STEP 19 PHASE 2: WORKER CLASS & PPE ASSOCIATION ACCURACY")
    print("=" * 80)
    orch = EndToEndPipelineOrchestrator(device="auto", enable_ppe_model=True)

    dummy_frame = np.full((720, 1280, 3), 40, dtype=np.uint8)
    worker_box = BoundingBox(x1=300.0, y1=150.0, x2=450.0, y2=550.0)
    hardhat_box = BoundingBox(x1=340.0, y1=130.0, x2=410.0, y2=200.0)
    vest_box = BoundingBox(x1=320.0, y1=240.0, x2=430.0, y2=420.0)

    # Note: testing with class_name="worker" instead of "person"
    manual_dets = [
        DetectionResult(class_id=0, class_name="worker", confidence=0.91, bbox=worker_box)
    ]
    manual_ppe = [
        DetectionResult(class_id=1, class_name="Hardhat", confidence=0.89, bbox=hardhat_box),
        DetectionResult(class_id=2, class_name="Safety Vest", confidence=0.87, bbox=vest_box),
    ]

    t0 = time.perf_counter()
    assessment, annotated = orch.process_frame(
        frame=dummy_frame,
        frame_id=1,
        timestamp=0.033,
        manual_detections=manual_dets,
        manual_ppe_detections=manual_ppe,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    print(f"  Execution Latency         : {latency_ms:.2f} ms")
    print(f"  Active Tracks             : {len(assessment.tracks)}")
    print(f"  Worker PPE Inventories    : {len(assessment.worker_inventories)}")
    if assessment.worker_inventories:
        inv = assessment.worker_inventories[0]
        items_detected = [i.class_name for i in inv.items]
        print(f"  Track #{inv.track_id} PPE Items: {items_detected}")
        print(f"  Compliance Status         : {inv.compliance_status.value}")

    assert len(assessment.worker_inventories) == 1, "Expected 1 worker PPE inventory"
    print("  [SUCCESS] Worker class correctly routed to PPE compliance!")
    return assessment


def run_tracker_class_voting_benchmark():
    print("\n" + "=" * 80)
    print("STEP 19 PHASE 3: TRACKER CLASS EVIDENCE VOTING BENCHMARK")
    print("=" * 80)
    tracker = ByteTrackTracker(track_high_thresh=0.20, track_low_thresh=0.05)

    # Simulate 5 frames where frame 1 had a noisy label 'chair', and frames 2-5 confirm 'person'
    history = []
    for f in range(1, 6):
        box = BoundingBox(x1=100.0 + f * 2, y1=100.0, x2=160.0 + f * 2, y2=250.0)
        cname = "chair" if f == 1 else "person"
        cid = 10 if f == 1 else 0
        conf = 0.50 if f == 1 else 0.90

        det = FrameDetections(
            frame_id=f,
            timestamp=f * 0.033,
            detections=[DetectionResult(class_id=cid, class_name=cname, confidence=conf, bbox=box)],
        )
        tracks = tracker.update(det)
        active_cls = tracks.active_tracks[0].class_name if tracks.active_tracks else "none"
        history.append((f, cname, active_cls))
        print(f"  Frame {f}: Incoming detection='{cname}' (conf={conf:.2f}) -> Tracked class='{active_cls}'")

    assert history[-1][2] == "person", "Class should have converged to 'person'"
    print("  [SUCCESS] Class voting resolved transient classification error!")


def run_pipeline_latency_benchmark(num_frames: int = 30):
    print("\n" + "=" * 80)
    print(f"STEP 19 PHASE 4: PIPELINE LATENCY & THROUGHPUT BENCHMARK ({num_frames} frames)")
    print("=" * 80)
    orch = EndToEndPipelineOrchestrator(device="auto", enable_ppe_model=False)

    dummy_frame = np.full((720, 1280, 3), 35, dtype=np.uint8)
    latencies = []

    for f in range(1, num_frames + 1):
        worker_box = BoundingBox(x1=200.0 + f * 5, y1=200.0, x2=280.0 + f * 5, y2=450.0)
        manual_dets = [
            DetectionResult(class_id=0, class_name="person", confidence=0.88, bbox=worker_box)
        ]
        t0 = time.perf_counter()
        orch.process_frame(
            frame=dummy_frame,
            frame_id=f,
            timestamp=f * 0.033,
            manual_detections=manual_dets,
        )
        lat = (time.perf_counter() - t0) * 1000
        latencies.append(lat)

    warmup = latencies[:5]
    bench = latencies[5:]
    avg_lat = sum(bench) / len(bench)
    fps = 1000.0 / avg_lat

    print(f"  Warmup Latency (first 5) : {sum(warmup)/len(warmup):.2f} ms")
    print(f"  Steady-State Latency     : {avg_lat:.2f} ms per frame")
    print(f"  Effective Throughput     : {fps:.1f} FPS")
    print("=" * 80)


def main():
    run_hardware_diagnostics()
    run_orchestrator_worker_ppe_test()
    run_tracker_class_voting_benchmark()
    run_pipeline_latency_benchmark()
    print("\nStep 19 Evaluation and Verification Complete.")


if __name__ == "__main__":
    main()
