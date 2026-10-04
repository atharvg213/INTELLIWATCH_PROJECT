"""
scripts/evaluate_step18_final.py
Step 18 Comprehensive System Evaluation & Benchmarking Tool.

Executes and measures:
1. Real Image Validation (data/samples/industrial_cctv.jpg, data/samples/ppe_sample.jpg)
2. Real Video Validation (data/samples/cctv_worker_moving.mp4)
3. Performance Benchmarks on CPU (YOLO11n, Industrial Detector, Tracker, Full Pipeline, Reasoning)
4. Failure/Edge-Case Validation
"""
import json
import os
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
import cv2
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import get_settings
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from backend.services.incident_store import get_incident_store
from backend.services.temporal_log import get_temporal_log
from backend.services.explanation_service import get_explanation_service
from backend.services.zone_service import get_zone_service
from vision.detection.yolo_detector import YOLODetector
from vision.detection.industrial_detector import IndustrialDetector
from vision.tracking.bytetrack_tracker import ByteTrackTracker


def run_image_validation(image_path: Path):
    print("\n" + "=" * 80)
    print(f"PHASE 4: REAL IMAGE VALIDATION -> {image_path.name}")
    print("=" * 80)
    if not image_path.exists():
        print(f"Error: {image_path} does not exist.")
        return None

    img = cv2.imread(str(image_path))
    h, w, c = img.shape
    print(f"Image Resolution : {w}x{h} ({c} channels)")

    orchestrator = EndToEndPipelineOrchestrator(
        device="cpu",
        enable_ppe_model=True,
        enable_industrial_model=True,
        enable_depth_model=False,
    )

    t0 = time.perf_counter()
    assessment, annotated_frame = orchestrator.process_frame(
        frame=img,
        frame_id=1,
        timestamp=0.033,
    )
    latency_ms = (time.perf_counter() - t0) * 1000

    classes_detected = [d.class_name for d in assessment.detections]
    class_counts = Counter(classes_detected)
    worker_count = class_counts.get("person", 0) + class_counts.get("worker", 0)
    industrial_objects = {k: v for k, v in class_counts.items() if k not in ("person", "worker")}

    rels_count = len(assessment.scene.relationships) if assessment.scene else 0
    risk_score = assessment.highest_risk_score
    risk_tier = assessment.highest_risk_level.value

    print(f"Processing Latency   : {latency_ms:.1f} ms")
    print(f"Total Detections     : {len(assessment.detections)}")
    print(f"Worker Detections    : {worker_count}")
    print(f"Industrial Objects   : {industrial_objects}")
    print(f"Active Tracks        : {len(assessment.tracks)}")
    print(f"PPE Assessments      : {len(assessment.worker_inventories)}")
    print(f"Zone Occupancies     : {len(assessment.zone_memberships)}")
    print(f"Scene Relationships  : {rels_count}")
    print(f"Risk Events          : {len(assessment.active_events)}")
    print(f"Risk Score           : {risk_score:.1f} ({risk_tier})")
    print(f"Early Warnings       : {len(assessment.active_early_warnings)}")

    return {
        "file": image_path.name,
        "resolution": f"{w}x{h}",
        "latency_ms": latency_ms,
        "detections_count": len(assessment.detections),
        "worker_count": worker_count,
        "industrial_objects": industrial_objects,
        "tracks": len(assessment.tracks),
        "risk_tier": risk_tier,
        "risk_score": risk_score,
        "scene_relationships": rels_count,
    }


def run_video_validation(video_path: Path, max_frames: int = 60):
    print("\n" + "=" * 80)
    print(f"PHASE 5: REAL VIDEO VALIDATION -> {video_path.name}")
    print("=" * 80)
    if not video_path.exists():
        print(f"Error: {video_path} does not exist.")
        return None

    cap = cv2.VideoCapture(str(video_path))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / fps

    print(f"Video File       : {video_path.name}")
    print(f"Resolution       : {w}x{h}")
    print(f"Source FPS       : {fps:.2f}")
    print(f"Total Frames     : {total_frames} ({duration_s:.2f}s)")

    # Reset singletons
    get_incident_store().reset()
    get_temporal_log().__init__()

    orchestrator = EndToEndPipelineOrchestrator(
        device="cpu",
        enable_ppe_model=True,
        enable_industrial_model=True,
        enable_depth_model=False,
    )

    frames_to_process = min(total_frames, max_frames)
    processed_count = 0
    latencies = []
    unique_tracks = set()
    worker_detections = 0
    vehicle_detections = 0
    machine_detections = 0
    behavior_transitions = []
    prev_behaviors = {}
    risk_events_total = 0
    prediction_indicators_total = 0

    t_start = time.perf_counter()

    while processed_count < frames_to_process:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = processed_count / fps
        t0 = time.perf_counter()
        assessment, _ = orchestrator.process_frame(
            frame=frame,
            frame_id=processed_count + 1,
            timestamp=timestamp,
        )
        dt = (time.perf_counter() - t0) * 1000
        latencies.append(dt)
        processed_count += 1

        # Track IDs
        for trk in assessment.tracks:
            unique_tracks.add(trk.track_id)

        # Object classifications
        for det in assessment.detections:
            cname = det.class_name.lower()
            if cname in ("person", "worker"):
                worker_detections += 1
            elif cname in ("forklift", "truck", "car", "industrial vehicle", "vehicle"):
                vehicle_detections += 1
            elif cname in ("machinery", "machine", "robotic arm", "conveyor"):
                machine_detections += 1

        # Behavior transitions
        for b in assessment.behavior_states:
            tid = b.track_id
            curr_b = b.primary_behavior.value if hasattr(b.primary_behavior, 'value') else str(b.primary_behavior)
            if tid in prev_behaviors and prev_behaviors[tid] != curr_b:
                behavior_transitions.append((tid, prev_behaviors[tid], curr_b, timestamp))
            prev_behaviors[tid] = curr_b

        # Risk and prediction counts
        risk_events_total += len(assessment.active_events)
        prediction_indicators_total += len(assessment.active_early_warnings)

    cap.release()
    total_elapsed = time.perf_counter() - t_start
    avg_latency = np.mean(latencies) if latencies else 0.0
    proc_fps = processed_count / total_elapsed if total_elapsed > 0 else 0.0

    incidents = get_incident_store().list_incidents()

    print(f"\n--- REAL VIDEO RESULTS SUMMARY ---")
    print(f"Processed Frames        : {processed_count} / {total_frames}")
    print(f"Total Processing Time   : {total_elapsed:.2f} s")
    print(f"Processing FPS (CPU)    : {proc_fps:.2f} FPS")
    print(f"Average Latency / Frame : {avg_latency:.1f} ms")
    print(f"Unique Track IDs        : {len(unique_tracks)} (IDs: {sorted(list(unique_tracks))})")
    print(f"Worker Detections       : {worker_detections}")
    print(f"Vehicle Detections      : {vehicle_detections} (Honest: Real video contains NO vehicles)")
    print(f"Machine Detections      : {machine_detections}")
    print(f"Behavior Transitions    : {len(behavior_transitions)}")
    for tid, b_from, b_to, ts in behavior_transitions[:5]:
        print(f"  • Track #{tid}: {b_from} -> {b_to} at {ts:.2f}s")
    print(f"Total Risk Events Logged: {risk_events_total}")
    print(f"Prediction Indicators   : {prediction_indicators_total}")
    print(f"Incidents Recorded      : {len(incidents)}")

    return {
        "file": video_path.name,
        "resolution": f"{w}x{h}",
        "source_fps": fps,
        "total_frames": total_frames,
        "processed_frames": processed_count,
        "proc_fps": proc_fps,
        "avg_latency_ms": avg_latency,
        "unique_tracks": len(unique_tracks),
        "worker_detections": worker_detections,
        "vehicle_detections": vehicle_detections,
        "machine_detections": machine_detections,
        "behavior_transitions": len(behavior_transitions),
        "risk_events_total": risk_events_total,
        "prediction_indicators_total": prediction_indicators_total,
        "incidents_count": len(incidents),
    }


def run_benchmarks(sample_frame: np.ndarray, num_warmup: int = 2, num_runs: int = 10):
    print("\n" + "=" * 80)
    print("PHASE 9: PERFORMANCE BENCHMARK (CPU-ONLY)")
    print("=" * 80)

    # 1. YOLO11n General Detector
    print("[1/5] Benchmarking YOLO11n General Detector...")
    t0 = time.perf_counter()
    yolo_detector = YOLODetector(device="cpu", confidence_threshold=0.25)
    load_time_yolo = (time.perf_counter() - t0) * 1000

    for _ in range(num_warmup):
        _ = yolo_detector.detect(sample_frame)

    yolo_latencies = []
    for _ in range(num_runs):
        t0 = time.perf_counter()
        _ = yolo_detector.detect(sample_frame)
        yolo_latencies.append((time.perf_counter() - t0) * 1000)

    avg_yolo = np.mean(yolo_latencies)
    std_yolo = np.std(yolo_latencies)
    fps_yolo = 1000.0 / avg_yolo
    print(f"      Load Time : {load_time_yolo:.1f} ms")
    print(f"      Latency   : {avg_yolo:.1f} ± {std_yolo:.1f} ms ({fps_yolo:.1f} FPS)")

    # 2. YOLOv8s-World-v2 Industrial Detector
    print("[2/5] Benchmarking YOLOv8s-World-v2 Industrial Detector...")
    try:
        t0 = time.perf_counter()
        ind_detector = IndustrialDetector(device="cpu")
        load_time_ind = (time.perf_counter() - t0) * 1000

        for _ in range(num_warmup):
            _ = ind_detector.detect(sample_frame)

        ind_latencies = []
        for _ in range(num_runs):
            t0 = time.perf_counter()
            _ = ind_detector.detect(sample_frame)
            ind_latencies.append((time.perf_counter() - t0) * 1000)

        avg_ind = float(np.mean(ind_latencies))
        std_ind = float(np.std(ind_latencies))
        fps_ind = 1000.0 / avg_ind
        print(f"      Load Time : {load_time_ind:.1f} ms")
        print(f"      Latency   : {avg_ind:.1f} ± {std_ind:.1f} ms ({fps_ind:.1f} FPS)")
    except Exception as exc:
        print(f"      [SKIPPED] Industrial model weights unavailable: {exc}")
        load_time_ind, avg_ind, std_ind, fps_ind = 0.0, 0.0, 0.0, 0.0

    # 3. ByteTrack Multi-Object Tracker
    print("[3/5] Benchmarking ByteTrack Multi-Object Tracker...")
    tracker = ByteTrackTracker(frame_rate=30)
    from backend.schemas.detection import DetectionResult, BoundingBox, FrameDetections
    dummy_dets = FrameDetections(
        frame_id=1,
        timestamp=0.033,
        detections=[
            DetectionResult(class_name="person", class_id=0, confidence=0.88, bbox=BoundingBox(x1=100+i*50, y1=100, x2=200+i*50, y2=400))
            for i in range(5)
        ]
    )
    tracker_latencies = []
    for _ in range(num_runs * 2):
        t0 = time.perf_counter()
        _ = tracker.update(dummy_dets)
        tracker_latencies.append((time.perf_counter() - t0) * 1000)

    avg_tracker = np.mean(tracker_latencies)
    print(f"      Latency   : {avg_tracker:.2f} ms ({1000.0/avg_tracker:.1f} FPS)")

    # 4. Complete Orchestrator Pipeline (YOLO11n + Industrial + Tracker + Reasoning)
    print("[4/5] Benchmarking Complete End-to-End Pipeline (CPU)...")
    t0 = time.perf_counter()
    orch = EndToEndPipelineOrchestrator(
        device="cpu",
        enable_ppe_model=True,
        enable_industrial_model=True,
        enable_depth_model=False,
    )
    load_time_orch = (time.perf_counter() - t0) * 1000

    for f_idx in range(num_warmup):
        _ = orch.process_frame(sample_frame, frame_id=f_idx+1, timestamp=(f_idx+1)*0.033)

    pipeline_latencies = []
    for f_idx in range(num_runs):
        t0 = time.perf_counter()
        _ = orch.process_frame(sample_frame, frame_id=f_idx+num_warmup+1, timestamp=(f_idx+num_warmup+1)*0.033)
        pipeline_latencies.append((time.perf_counter() - t0) * 1000)

    avg_pipeline = np.mean(pipeline_latencies)
    std_pipeline = np.std(pipeline_latencies)
    fps_pipeline = 1000.0 / avg_pipeline
    print(f"      Load Time : {load_time_orch:.1f} ms")
    print(f"      Latency   : {avg_pipeline:.1f} ± {std_pipeline:.1f} ms ({fps_pipeline:.2f} FPS)")

    # 5. Reasoning Overhead
    reasoning_overhead_ms = max(0.0, avg_pipeline - (avg_yolo + avg_ind + avg_tracker))
    print(f"[5/5] Reasoning & Spatial Graph Overhead: ~{reasoning_overhead_ms:.1f} ms")

    return {
        "yolo11n": {"load_ms": float(load_time_yolo), "latency_ms": float(avg_yolo), "fps": float(fps_yolo)},
        "industrial": {"load_ms": float(load_time_ind), "latency_ms": float(avg_ind), "fps": float(fps_ind)},
        "tracker": {"latency_ms": float(avg_tracker), "fps": float(1000.0 / avg_tracker)},
        "full_pipeline": {"load_ms": float(load_time_orch), "latency_ms": float(avg_pipeline), "fps": float(fps_pipeline)},
        "reasoning_overhead_ms": float(reasoning_overhead_ms),
    }


def run_failure_cases():
    print("\n" + "=" * 80)
    print("PHASE 10: FAILURE-CASE & EDGE-CASE VALIDATION")
    print("=" * 80)

    orch = EndToEndPipelineOrchestrator(device="cpu", enable_depth_model=False)
    results = {}

    # Case 1: Empty frame (None or 0-dim)
    try:
        empty_frame = np.zeros((0, 0, 3), dtype=np.uint8)
        _, _ = orch.process_frame(empty_frame, frame_id=1, timestamp=0.0)
        results["empty_frame"] = "Failed (should have raised error)"
    except Exception as e:
        results["empty_frame"] = f"Passed (caught expected {type(e).__name__}: {str(e)[:40]})"

    # Case 2: Uniform blank image (no detections expected)
    try:
        blank_frame = np.full((480, 640, 3), 128, dtype=np.uint8)
        ass, _ = orch.process_frame(blank_frame, frame_id=1, timestamp=0.0)
        assert len(ass.detections) == 0
        assert ass.highest_risk_score == 0.0
        results["blank_image_no_detections"] = "Passed (zero detections, zero risk, no crashes)"
    except Exception as e:
        results["blank_image_no_detections"] = f"Failed ({e})"

    # Case 3: Invalid Video Path
    from vision.preprocessing.video_reader import VideoReader, VideoFileNotFoundError
    try:
        reader = VideoReader("non_existent_file_xyz.mp4")
        with reader:
            pass
        results["invalid_video_path"] = "Failed (should have raised FileNotFoundError)"
    except VideoFileNotFoundError:
        results["invalid_video_path"] = "Passed (caught VideoFileNotFoundError)"
    except Exception as e:
        results["invalid_video_path"] = f"Failed (unexpected {type(e).__name__}: {e})"

    # Case 4: Non-existent Incident Explanation
    expl_service = get_explanation_service()
    res = expl_service.get_incident_explanation("non_existent_inc_999")
    if res is None:
        results["missing_incident_explanation"] = "Passed (returned None gracefully, ready for 404)"
    else:
        results["missing_incident_explanation"] = "Failed (returned non-None for invalid id)"

    # Case 5: Zone coordinates out of bounds / degenerate
    zone_svc = get_zone_service()
    try:
        from backend.schemas.zones import ZoneCreateRequest
        req = ZoneCreateRequest(name="degenerate", zone_type="restricted", polygon=[[10, 10], [20, 20]])  # Only 2 points
        zone_svc.create_zone(req)
        results["degenerate_zone_polygon"] = "Failed (allowed 2-point polygon)"
    except ValueError as e:
        results["degenerate_zone_polygon"] = f"Passed (caught ValueError: {str(e)[:45]})"

    # Case 6: Evidence Snapshot Missing Honesty
    from backend.schemas.evidence import IncidentRecord
    from backend.schemas.risk import RiskLevel
    rec = IncidentRecord(
        incident_id="inc_test_no_snap",
        event_type="RESTRICTED_ZONE_INTRUSION",
        risk_level=RiskLevel.HIGH,
        risk_score=75.0,
        frame_id=10,
        timestamp=0.33,
        explanation="Test explanation",
        has_snapshot=False,
        snapshot_path=None,
    )
    get_incident_store().record_incident(rec, frame_image=None)
    expl_resp = expl_service.get_incident_explanation(rec.incident_id)
    if expl_resp and expl_resp.has_snapshot is False and expl_resp.snapshot_url is None:
        results["evidence_snapshot_missing_honesty"] = "Passed (explicitly reports has_snapshot=False, snapshot_url=None)"
    else:
        results["evidence_snapshot_missing_honesty"] = "Failed (incorrectly claimed snapshot)"

    for k, v in results.items():
        print(f"  • {k:35s}: {v}")

    return results


def main():
    print("=" * 80)
    print("INTELLIWATCH STEP 18: SYSTEM EVALUATION & BENCHMARKING")
    print("=" * 80)

    # 1. Real Image Validation
    img1_path = PROJECT_ROOT / "data" / "samples" / "industrial_cctv.jpg"
    img2_path = PROJECT_ROOT / "data" / "samples" / "ppe_sample.jpg"
    res_img1 = run_image_validation(img1_path)
    res_img2 = run_image_validation(img2_path)

    # 2. Real Video Validation
    vid_path = PROJECT_ROOT / "data" / "samples" / "cctv_worker_moving.mp4"
    res_vid = run_video_validation(vid_path, max_frames=60)

    # 3. CPU Benchmarks
    sample_img = cv2.imread(str(img1_path))
    res_bench = run_benchmarks(sample_img, num_warmup=2, num_runs=8)

    # 4. Failure Cases
    res_fail = run_failure_cases()

    # Save summary report to JSON
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "image_validation": [res_img1, res_img2],
        "video_validation": res_vid,
        "benchmarks_cpu": res_bench,
        "failure_cases": res_fail,
    }
    out_file = PROJECT_ROOT / "data" / "output" / "step18_evaluation_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(f"\nSaved evaluation report to: {out_file}")


if __name__ == "__main__":
    main()
