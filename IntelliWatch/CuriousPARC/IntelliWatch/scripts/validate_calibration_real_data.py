"""
Real Data & Media Switching Validation Script for Camera Calibration.
Validates:
1. industrial_cctv.jpg
2. ppe_sample.jpg
3. cctv_worker_moving.mp4
4. VIDEO -> IMAGE -> VIDEO media switching without stale state contamination.
5. Micro-benchmarking calibration homography overhead.
"""

import sys
import time
from pathlib import Path

# Setup sys.path
root_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(root_dir))

from fastapi.testclient import TestClient
from backend.main import app
from backend.services.calibration_service import get_calibration_service, CameraCalibrationConfig, Point2D

client = TestClient(app)
cal_svc = get_calibration_service()

def log(msg):
    print(f"[VALIDATION] {msg}")

def test_media_switching_and_calibration():
    samples_dir = root_dir / "data" / "samples"
    cctv_jpg = samples_dir / "industrial_cctv.jpg"
    ppe_jpg = samples_dir / "ppe_sample.jpg"
    video_mp4 = samples_dir / "cctv_worker_moving.mp4"

    assert cctv_jpg.exists(), f"Missing {cctv_jpg}"
    assert ppe_jpg.exists(), f"Missing {ppe_jpg}"
    assert video_mp4.exists(), f"Missing {video_mp4}"
    log("All sample assets verified on disk.")

    # 1. Configure ground-plane calibration for camera 'cam_cctv_01'
    cal_cfg = CameraCalibrationConfig(
        camera_id="cam_cctv_01",
        calibration_enabled=True,
        points=[
            Point2D(x=200, y=300),
            Point2D(x=1000, y=300),
            Point2D(x=1200, y=700),
            Point2D(x=100, y=700)
        ],
        real_world_width_m=8.0,
        real_world_depth_m=14.0,
        coordinate_system="METRIC_GROUND_PLANE"
    )
    saved_cfg = cal_svc.save_calibration("cam_cctv_01", cal_cfg)
    assert saved_cfg.calibration_status == "CALIBRATED"
    log("Ground plane calibration configured and saved for 'cam_cctv_01'.")

    # 2. Step 1: Run Video Inference
    log("--- Step 1: Video Inference (cctv_worker_moving.mp4) ---")
    with open(video_mp4, "rb") as f:
        res = client.post("/api/v1/analyze/video", files={"file": ("cctv_worker_moving.mp4", f, "video/mp4")})
    assert res.status_code in (200, 202), res.text
    video_job_id = res.json()["job_id"]
    log(f"Video job submitted: {video_job_id}. Waiting for completion...")

    # Poll video job
    for _ in range(60):
        status_res = client.get(f"/api/v1/analyze/status/{video_job_id}")
        assert status_res.status_code == 200
        st = status_res.json()
        if st["status"] == "COMPLETED":
            break
        time.sleep(0.5)
    assert st["status"] == "COMPLETED", f"Video job did not complete: {st}"
    log(f"Video job {video_job_id} COMPLETED. Total frames: {st.get('frames_processed')}")

    # Query video scene graph with camera_id='cam_cctv_01' (calibrated)
    sg_vid_cal = client.get(f"/api/v1/scene/graph?job_id={video_job_id}&camera_id=cam_cctv_01")
    assert sg_vid_cal.status_code == 200
    vid_scene_cal = sg_vid_cal.json()
    assert vid_scene_cal["calibration_status"] == "CALIBRATED"
    assert vid_scene_cal["spatial_basis"] == "GROUND_PLANE_APPROXIMATION"
    log("Video scene graph with camera_id='cam_cctv_01' has calibrated spatial basis.")

    # Check entities in video
    entities_cal = vid_scene_cal.get("entities", [])
    has_ground_pos = False
    for ent in entities_cal:
        if ent.get("position", {}).get("ground_plane"):
            has_ground_pos = True
            gp = ent["position"]["ground_plane"]
            log(f"Entity {ent['id']} ground coordinates: X={gp['x_m']}m, Y={gp['y_m']}m")
    assert has_ground_pos, "Expected at least one entity with ground plane coordinates in calibrated video scene."

    # 3. Step 2: Media Switching -> Switch to Image Inference (industrial_cctv.jpg)
    log("--- Step 2: Media Switching -> Image Inference (industrial_cctv.jpg) ---")
    with open(cctv_jpg, "rb") as f:
        res = client.post("/api/v1/analyze/image", files={"file": ("industrial_cctv.jpg", f, "image/jpeg")})
    assert res.status_code in (200, 202), res.text
    image_job_id = res.json()["job_id"]
    log(f"Image job submitted: {image_job_id}. Polling for completion...")

    for _ in range(30):
        st = client.get(f"/api/v1/analyze/status/{image_job_id}").json()
        if st["status"] == "COMPLETED":
            break
        time.sleep(0.2)
    assert st["status"] == "COMPLETED"

    # Query image scene graph WITHOUT camera calibration (uncalibrated fallback)
    sg_img_uncal = client.get(f"/api/v1/scene/graph?job_id={image_job_id}&camera_id=cam_uncalibrated_99")
    assert sg_img_uncal.status_code == 200
    img_scene_uncal = sg_img_uncal.json()

    # Verify no video state leaked into image state
    assert img_scene_uncal["source_mode"] in ("LATEST SCENE SNAPSHOT", "single_image")
    assert img_scene_uncal["spatial_basis"] == "IMAGE_SPACE"
    assert img_scene_uncal["calibration_status"] == "UNCONFIGURED"

    for ent in img_scene_uncal.get("entities", []):
        assert ent.get("position", {}).get("ground_plane") is None, "Uncalibrated image must not have ground_plane coords"
        if ent.get("type") in ("PERSON", "VEHICLE"):
            # Single frame must honestly report velocity unavailable
            assert ent.get("state", {}).get("speed_m_per_s") is None
            assert ent.get("state", {}).get("speed_px_per_s") is None

    for rel in img_scene_uncal.get("relationships", []):
        assert rel.get("distance_m") is None, "Uncalibrated relationship must have distance_m=None"
        assert rel.get("closing_rate_m_s") is None, "Uncalibrated relationship must have closing_rate_m_s=None"
        assert rel.get("spatial_basis") == "IMAGE_SPACE"

    log("Verified uncalibrated image scene graph has pure IMAGE_SPACE basis and zero state contamination.")

    # 4. Step 3: Media Switching -> Switch back to Video Inference (cctv_worker_moving.mp4)
    log("--- Step 3: Media Switching -> Switch back to Video Scene ---")
    sg_vid_refetch = client.get(f"/api/v1/scene/graph?job_id={video_job_id}&camera_id=cam_cctv_01")
    assert sg_vid_refetch.status_code == 200
    vid_scene_2 = sg_vid_refetch.json()
    assert vid_scene_2["source_mode"] in ("VIDEO ANALYSIS", "video_job")
    assert vid_scene_2["spatial_basis"] == "GROUND_PLANE_APPROXIMATION"
    assert vid_scene_2["calibration_status"] == "CALIBRATED"
    log("Verified video scene maintains consistent calibrated ground plane state upon switching back.")

    # 5. Performance overhead micro-benchmark
    log("--- Step 4: Performance Overhead Benchmark ---")
    calibrator = cal_svc.get_calibrator("cam_cctv_01")
    assert calibrator is not None

    n_iterations = 10000
    t0 = time.perf_counter()
    for _ in range(n_iterations):
        gx, gy = calibrator.image_to_ground(450.0, 520.0)
    elapsed = time.perf_counter() - t0
    avg_us = (elapsed / n_iterations) * 1e6
    log(f"Transformed {n_iterations} points in {elapsed*1000:.2f}ms ({avg_us:.3f} microseconds per point).")
    assert avg_us < 50.0, f"Transformation too slow: {avg_us} us"

    # Cleanup calibration
    cal_svc.delete_calibration("cam_cctv_01")
    log("Cleaned up temporary calibration.")
    log("ALL REAL DATA AND MEDIA SWITCHING CHECKS PASSED.")

if __name__ == "__main__":
    test_media_switching_and_calibration()
