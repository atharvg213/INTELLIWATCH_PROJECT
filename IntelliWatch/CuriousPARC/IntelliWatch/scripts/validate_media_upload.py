"""
scripts/validate_media_upload.py
Validation script testing the running FastAPI server for dashboard media analysis.
Tests:
- Health check
- Real image upload and single-frame analysis
- Annotated media retrieval
- Real video upload and asynchronous background job execution
- Status polling with measured frame counts and FPS
- Incident recording and evidence snapshot verification
"""
import time
from pathlib import Path
import httpx

PROJECT_ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=" * 70)
    print("INTELLIWATCH - LIVE MEDIA UPLOAD & DIRECT ANALYSIS VALIDATION")
    print("=" * 70)

    client = httpx.Client(base_url="http://127.0.0.1:8000", timeout=60.0)

    # 1. Health check
    h = client.get("/health")
    assert h.status_code == 200, f"Health check failed: {h.text}"
    print(f"[PASS] 1. Health Check: {h.json()}")

    # 2. Upload Real Image: industrial_cctv.jpg
    img_path = PROJECT_ROOT / "data" / "samples" / "industrial_cctv.jpg"
    assert img_path.exists(), f"Sample image not found: {img_path}"
    print(f"\nUploading real image: {img_path.name} ({img_path.stat().st_size} bytes)...")

    with open(img_path, "rb") as f:
        files = {"file": (img_path.name, f, "image/jpeg")}
        res_img = client.post("/api/v1/analyze/image", files=files)

    assert res_img.status_code == 200, f"Image upload failed: {res_img.text}"
    img_data = res_img.json()
    print(f"[PASS] 2. Image Analysis Response:")
    print(f"       Job ID          : {img_data['job_id']}")
    print(f"       Status          : {img_data['status']}")
    print(f"       Detections      : {img_data['detections_count']}")
    print(f"       Workers         : {img_data['workers_count']}")
    print(f"       Non-Compliant   : {img_data['non_compliant_count']}")
    print(f"       Risk Level      : {img_data['highest_risk_level']} (Score: {img_data['highest_risk_score']})")
    print(f"       Single Frame    : {img_data['is_single_frame']}")
    print(f"       Temporal Notice : '{img_data['temporal_notice']}'")

    # 3. Retrieve Annotated Image Media
    ann_url = img_data["annotated_media_url"]
    ann_res = client.get(ann_url)
    assert ann_res.status_code == 200, f"Failed to retrieve annotated image: {ann_res.status_code}"
    assert len(ann_res.content) > 1000, "Annotated image content is unexpectedly empty"
    print(f"[PASS] 3. Annotated Image Retrievable: {ann_url} ({len(ann_res.content)} bytes)")

    # 4. Upload Real Video: short_diagnostic.mp4
    vid_path = PROJECT_ROOT / "data" / "samples" / "short_diagnostic.mp4"
    assert vid_path.exists(), f"Sample video not found: {vid_path}"
    print(f"\nUploading real video: {vid_path.name} ({vid_path.stat().st_size} bytes)...")

    with open(vid_path, "rb") as f:
        files = {"file": (vid_path.name, f, "video/mp4")}
        res_vid = client.post("/api/v1/analyze/video", files=files)

    assert res_vid.status_code == 202, f"Video upload failed: {res_vid.text}"
    vid_data = res_vid.json()
    job_id = vid_data["job_id"]
    print(f"[PASS] 4. Video Analysis Queued:")
    print(f"       Job ID : {job_id}")
    print(f"       Status : {vid_data['status']}")

    # 5. Poll video status
    print("\nPolling video job telemetry...")
    status_data = None
    for i in range(40):
        time.sleep(1.0)
        st_res = client.get(f"/api/v1/analyze/status/{job_id}")
        assert st_res.status_code == 200
        st = st_res.json()
        print(f"       [T+{i+1:02d}s] Status: {st['status']:<10} | Progress: {st['progress_pct']:>5.1f}% | Frames: {st['current_frame']:>3}/{st['total_frames']} | FPS: {st['processing_fps']:.1f} | Risk: {st['current_risk_level']}")
        if st["status"] in ["COMPLETED", "FAILED"]:
            status_data = st
            break

    assert status_data is not None, "Video job polling timed out"
    assert status_data["status"] == "COMPLETED", f"Video processing failed: {status_data.get('error_message')}"
    print("[PASS] 5. Video Processing Completed Successfully!")

    # 6. Retrieve Video Result Summary
    res_final = client.get(f"/api/v1/analyze/result/{job_id}")
    assert res_final.status_code == 200
    final_data = res_final.json()
    print(f"[PASS] 6. Final Video Analysis Summary:")
    print(f"       Total Frames       : {final_data['total_frames']}")
    print(f"       Processed Frames   : {final_data['processed_frames']}")
    print(f"       Average FPS        : {final_data['average_fps']} FPS")
    print(f"       Average Latency    : {final_data['average_latency_ms']} ms")
    print(f"       Highest Risk Tier  : {final_data['highest_risk_tier']}")
    print(f"       Recorded Incidents : {final_data['total_incidents']}")
    print(f"       Annotated Video    : {final_data['annotated_video_url']}")

    # 7. Check Incidents API
    inc_res = client.get("/api/v1/incidents")
    assert inc_res.status_code == 200
    incidents = inc_res.json()
    print(f"[PASS] 7. Incident Evidence Registry: {len(incidents)} total incidents logged.")
    if len(incidents) > 0:
        first_inc = incidents[0]
        print(f"       Latest Incident: {first_inc['incident_id']} - [{first_inc['risk_level']}] {first_inc['event_type']}")
        if first_inc["has_snapshot"]:
            ev_res = client.get(f"/api/v1/incidents/{first_inc['incident_id']}/evidence")
            assert ev_res.status_code == 200
            print(f"       Evidence Snapshot Retrievable: {len(ev_res.content)} bytes")

    print("\n" + "=" * 70)
    print("ALL MEDIA UPLOAD & ANALYSIS VALIDATION CHECKS PASSED PERFECTLY!")
    print("=" * 70)

if __name__ == "__main__":
    main()
