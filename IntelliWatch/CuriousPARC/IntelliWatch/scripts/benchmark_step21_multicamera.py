"""
scripts/benchmark_step21_multicamera.py
Step 21 Multi-Camera RTSP Stream Processing Benchmark.

Evaluates:
  1. Multiple concurrent simulated camera streams (3 streams).
  2. Per-camera Ingest FPS, Processing FPS, End-to-End Latency, and Dropped Frames.
  3. Stream fault isolation: forcefully stopping/disconnecting 1 camera stream while
     verifying the remaining 2 streams continue uninterrupted.
  4. Records measured performance metrics and writes report to data/output/step21_multicamera_benchmark_report.json.
"""
import json
import logging
import os
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from backend.schemas.camera import CameraRegisterRequest, CameraStatus
from backend.services.camera_manager import get_camera_manager

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("benchmark_step21")

SAMPLE_VIDEO = ROOT_DIR / "data" / "samples" / "cctv_worker_moving.mp4"


def run_multicamera_benchmark():
    logger.info("=== STEP 21: MULTI-CAMERA STREAM INGESTION & PROCESSING BENCHMARK ===")
    assert SAMPLE_VIDEO.exists(), f"Sample video {SAMPLE_VIDEO} missing!"

    manager = get_camera_manager()
    manager.stop_all()

    # 1. Register 3 concurrent simulated CCTV camera streams
    configs = [
        {
            "camera_id": "cam_factory_01",
            "name": "Assembly Line East",
            "source": str(SAMPLE_VIDEO),
            "sampling_interval": 1,
            "max_processing_fps": 10.0,
        },
        {
            "camera_id": "cam_factory_02",
            "name": "Robotic Welding Bay",
            "source": str(SAMPLE_VIDEO),
            "sampling_interval": 2,
            "max_processing_fps": 8.0,
        },
        {
            "camera_id": "cam_factory_03",
            "name": "Forklift Loading Dock",
            "source": str(SAMPLE_VIDEO),
            "sampling_interval": 1,
            "max_processing_fps": 6.0,
        },
    ]

    logger.info("Registering and starting 3 concurrent camera streams...")
    for cfg in configs:
        req = CameraRegisterRequest(
            camera_id=cfg["camera_id"],
            name=cfg["name"],
            source=cfg["source"],
            sampling_interval=cfg["sampling_interval"],
            max_processing_fps=cfg["max_processing_fps"],
            auto_start=True,
            loop_file=True,
        )
        tel = manager.register_camera(req)
        logger.info(f"Registered [{tel.camera_id}] ({tel.name}) on {tel.source_sanitized}")

    # 2. Warm up and let all 3 cameras process concurrently
    logger.info("Running all 3 cameras concurrently for 4 seconds...")
    time.sleep(4.0)

    # Capture mid-run telemetry
    mid_telemetries = {c["camera_id"]: manager.get_camera(c["camera_id"]).get_telemetry() for c in configs}
    for cid, t in mid_telemetries.items():
        logger.info(
            f"Camera [{cid}]: Ingest={t.ingest_fps:.1f} FPS, AI Proc={t.processing_fps:.1f} FPS, "
            f"Latency={t.avg_latency_ms:.1f}ms, Dropped={t.dropped_frames_count}, Processed={t.total_processed_frames}"
        )

    # 3. Stream Fault Isolation Test: Simulate network failure / disconnect on cam_factory_02
    logger.info("Injecting disconnection failure on [cam_factory_02]...")
    worker_fail = manager.get_camera("cam_factory_02")
    worker_fail.reader.stop()  # Simulates RTSP stream drop

    # Verify cam_factory_01 and cam_factory_03 continue processing
    logger.info("Monitoring surviving cameras [cam_factory_01, cam_factory_03] for 3 seconds...")
    frames_before_c1 = manager.get_camera("cam_factory_01").total_processed_frames
    frames_before_c3 = manager.get_camera("cam_factory_03").total_processed_frames
    time.sleep(3.0)
    frames_after_c1 = manager.get_camera("cam_factory_01").total_processed_frames
    frames_after_c3 = manager.get_camera("cam_factory_03").total_processed_frames

    c1_delta = frames_after_c1 - frames_before_c1
    c3_delta = frames_after_c3 - frames_before_c3
    logger.info(f"Surviving camera [cam_factory_01] processed +{c1_delta} frames during failure.")
    logger.info(f"Surviving camera [cam_factory_03] processed +{c3_delta} frames during failure.")
    assert c1_delta >= 1, "Camera 1 stalled when Camera 2 disconnected!"
    assert c3_delta >= 1, "Camera 3 stalled when Camera 2 disconnected!"

    # 4. Final Telemetry Collection
    final_list = manager.list_cameras()
    report = {
        "benchmark_name": "Step 21 Real-Time RTSP & Multi-Camera Stream Ingestion",
        "total_cameras_tested": len(configs),
        "execution_device": "cpu (automatic fallback)",
        "concurrent_streams": [
            {
                "camera_id": c.camera_id,
                "name": c.name,
                "source": c.source_sanitized,
                "status": c.status.value,
                "is_active": c.is_active,
                "ingest_fps": c.ingest_fps,
                "processing_fps": c.processing_fps,
                "avg_latency_ms": c.avg_latency_ms,
                "total_ingested_frames": c.total_ingested_frames,
                "total_processed_frames": c.total_processed_frames,
                "dropped_frames_count": c.dropped_frames_count,
                "active_tracks": c.active_tracks_count,
                "active_incidents": c.active_incidents_count,
                "non_compliant_workers": c.non_compliant_workers_count,
            }
            for c in final_list.cameras
        ],
        "fault_isolation_verification": {
            "faulted_camera": "cam_factory_02",
            "cam_factory_01_processed_after_fault": c1_delta,
            "cam_factory_03_processed_after_fault": c3_delta,
            "isolation_passed": bool(c1_delta >= 1 and c3_delta >= 1),
        },
    }

    out_file = ROOT_DIR / "data" / "output" / "step21_multicamera_benchmark_report.json"
    out_file.parent.mkdir(parents=True, exist_ok=True)
    with open(out_file, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"Multi-camera benchmark report written to: {out_file}")

    # Clean shutdown
    manager.stop_all()
    logger.info("=== STEP 21 BENCHMARK COMPLETE ===")


if __name__ == "__main__":
    run_multicamera_benchmark()
