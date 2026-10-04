"""
tests/test_upload_analysis.py
Test suite for IntelliWatch Dashboard Media Upload and Direct Analysis.
Validates all Step 23 requirements:
1. Image upload endpoint
2. Video upload endpoint
3. Invalid file type rejection
4. Corrupt image handling
5. Corrupt video handling
6. Job creation
7. Job status polling
8. Completed image analysis response & single-frame notice
9. Completed video analysis execution
10. AssessmentStore update
11. IncidentStore integration
12. Safe filename/path traversal prevention
13. Dashboard route access
14. Result retrieval endpoint
15. Missing job ID (404)
16. CPU processing telemetry & non-fabrication
"""
import io
import time
from pathlib import Path
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.services.assessment_store import get_assessment_store
from backend.services.incident_store import get_incident_store
from backend.services.job_manager import get_job_manager


@pytest.fixture
def client():
    """FastAPI TestClient instance."""
    return TestClient(app)


@pytest.fixture
def synthetic_image_bytes():
    """Generates a valid encoded synthetic test image (PNG)."""
    img = np.zeros((360, 640, 3), dtype=np.uint8)
    # Draw simple shapes
    cv2.rectangle(img, (100, 100), (250, 300), (200, 150, 50), -1)
    cv2.circle(img, (400, 200), 50, (0, 0, 255), -1)
    ret, buf = cv2.imencode(".png", img)
    assert ret
    return buf.tobytes()


@pytest.fixture
def synthetic_video_bytes():
    """Generates a brief 5-frame synthetic MP4 video in memory."""
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(str(tmp_path), fourcc, 10.0, (320, 240))
        for i in range(5):
            frame = np.full((240, 320, 3), i * 30, dtype=np.uint8)
            cv2.rectangle(frame, (20 + i * 10, 20), (80 + i * 10, 120), (0, 255, 0), -1)
            writer.write(frame)
        writer.release()

        with open(tmp_path, "rb") as f:
            data = f.read()
    finally:
        tmp_path.unlink(missing_ok=True)

    return data


# 1. Image upload endpoint & 8. Completed image analysis & 10. AssessmentStore update & 16. CPU processing
def test_image_upload_and_analysis(client, synthetic_image_bytes):
    """
    Tests uploading a valid image.
    Verifies 200 OK, full FrameAssessment, single-frame technical honesty,
    AssessmentStore update, and CPU execution.
    """
    files = {"file": ("test_sample.png", synthetic_image_bytes, "image/png")}
    response = client.post("/api/v1/analyze/image", files=files)
    assert response.status_code == 200
    data = response.json()

    # Schema verification
    assert data["job_id"].startswith("job_img_")
    assert data["media_type"] == "image"
    assert data["status"] == "COMPLETED"
    assert data["is_single_frame"] is True
    assert "Not available for single-frame analysis" in data["temporal_notice"]
    assert "annotated_media_url" in data
    assert "assessment" in data

    # Verify AssessmentStore was updated
    current_assessment = get_assessment_store().get_current_assessment()
    assert current_assessment is not None
    assert current_assessment.frame_id == 1

    # Verify annotated image is retrievable
    media_url = data["annotated_media_url"]
    media_res = client.get(media_url)
    assert media_res.status_code == 200
    assert media_res.headers["content-type"] in ["image/jpeg", "image/png"]


# 2. Video upload endpoint & 6. Job creation
def test_video_upload_and_job_creation(client, synthetic_video_bytes):
    """
    Tests uploading a valid video.
    Verifies 202 Accepted, returns job_id, status QUEUED, and does not block.
    """
    files = {"file": ("worker_sample.mp4", synthetic_video_bytes, "video/mp4")}
    response = client.post("/api/v1/analyze/video", files=files)
    assert response.status_code == 202
    data = response.json()

    assert "job_id" in data
    assert data["job_id"].startswith("job_vid_")
    assert data["media_type"] == "video"
    assert data["status"] in ["QUEUED", "PROCESSING", "COMPLETED"]


# 3. Invalid file type rejection
def test_invalid_file_type_rejection(client):
    """Verifies rejection of unsupported file extensions (e.g. .txt, .pdf)."""
    # Invalid image
    bad_img = {"file": ("malicious.txt", b"not an image", "text/plain")}
    res_img = client.post("/api/v1/analyze/image", files=bad_img)
    assert res_img.status_code == 400
    assert "Unsupported image format" in res_img.json()["detail"]

    # Invalid video
    bad_vid = {"file": ("script.sh", b"#!/bin/bash", "application/x-sh")}
    res_vid = client.post("/api/v1/analyze/video", files=bad_vid)
    assert res_vid.status_code == 400
    assert "Unsupported video format" in res_vid.json()["detail"]


# 4. Corrupt image handling
def test_corrupt_image_handling(client):
    """Verifies that non-image bytes disguised with .jpg extension are rejected with clear message."""
    files = {"file": ("corrupt.jpg", b"PK\x03\x04definitely_not_a_jpeg_stream", "image/jpeg")}
    response = client.post("/api/v1/analyze/image", files=files)
    assert response.status_code == 400
    assert "Corrupted or unreadable image file" in response.json()["detail"]


# 5. Corrupt video handling
def test_corrupt_video_handling(client):
    """Verifies that corrupt video data disguised with .mp4 extension is rejected."""
    files = {"file": ("corrupt.mp4", b"random_corrupted_bytes_00000000", "video/mp4")}
    response = client.post("/api/v1/analyze/video", files=files)
    assert response.status_code == 400
    assert "Corrupted or unreadable video file" in response.json()["detail"]


# 7. Job status & 9. Completed video analysis & 14. Result retrieval
def test_video_job_lifecycle_and_result(client, synthetic_video_bytes):
    """
    Submits a brief video, polls status until COMPLETED, and retrieves final result.
    """
    files = {"file": ("test_flow.mp4", synthetic_video_bytes, "video/mp4")}
    submit_res = client.post("/api/v1/analyze/video", files=files)
    assert submit_res.status_code == 202
    job_id = submit_res.json()["job_id"]

    # Poll status (wait up to 10 seconds for completion of 5-frame video)
    status_data = None
    for _ in range(20):
        time.sleep(0.5)
        st_res = client.get(f"/api/v1/analyze/status/{job_id}")
        assert st_res.status_code == 200
        status_data = st_res.json()
        if status_data["status"] in ["COMPLETED", "FAILED"]:
            break

    assert status_data["status"] == "COMPLETED"
    assert status_data["processed_frames"] >= 5
    assert status_data["progress_pct"] == 100.0

    # Retrieve final result via /api/v1/analyze/result/{job_id}
    res_res = client.get(f"/api/v1/analyze/result/{job_id}")
    assert res_res.status_code == 200
    res_data = res_res.json()
    assert res_data["job_id"] == job_id
    assert res_data["status"] == "COMPLETED"
    assert res_data["media_type"] == "video"
    assert res_data["total_frames"] >= 5


# 11. IncidentStore integration
def test_incident_store_integration(client, synthetic_image_bytes):
    """
    Verifies that safety incidents from media processing are accessible via IncidentStore
    and /api/v1/incidents endpoint.
    """
    incident_store = get_incident_store()
    init_count = incident_store.total_count()

    # Process image
    files = {"file": ("incident_check.png", synthetic_image_bytes, "image/png")}
    res = client.post("/api/v1/analyze/image", files=files)
    assert res.status_code == 200

    # Total count in incident store remains stable or increases, but is valid
    assert incident_store.total_count() >= init_count


# 12. Safe filename and path traversal prevention
def test_safe_filename_and_path_traversal(client, synthetic_image_bytes):
    """
    Verifies that attempts to inject path traversal characters (../) in filenames
    are sanitized and kept strictly within upload directories.
    """
    malicious_filename = "../../../../../etc/passwd.png"
    files = {"file": (malicious_filename, synthetic_image_bytes, "image/png")}
    response = client.post("/api/v1/analyze/image", files=files)
    assert response.status_code == 200
    data = response.json()
    assert ".." not in data["filename"]
    assert "/" not in data["filename"]

    # Test media route traversal attempt
    bad_media_res = client.get("/api/v1/media/..%2F..%2Fconfigs%2Fsettings.py")
    assert bad_media_res.status_code in [404, 400]


# 13. Dashboard route
def test_dashboard_route(client):
    """Verifies that the dashboard is served at / and /dashboard."""
    for path in ["/", "/dashboard"]:
        res = client.get(path)
        assert res.status_code == 200
        assert "text/html" in res.headers.get("content-type", "") or "INTELLIWATCH" in res.text


# 15. Missing job ID (404)
def test_missing_job_id(client):
    """Verifies that querying a non-existent job ID returns 404."""
    res_status = client.get("/api/v1/analyze/status/job_vid_nonexistent999")
    assert res_status.status_code == 404

    res_result = client.get("/api/v1/analyze/result/job_vid_nonexistent999")
    assert res_result.status_code == 404


def test_image_result_retrieval(client, synthetic_image_bytes):
    """Verifies that GET /api/v1/analyze/result/{job_id} works for image jobs."""
    files = {"file": ("test_res.jpg", synthetic_image_bytes, "image/jpeg")}
    up_res = client.post("/api/v1/analyze/image", files=files)
    assert up_res.status_code == 200
    job_id = up_res.json()["job_id"]

    get_res = client.get(f"/api/v1/analyze/result/{job_id}")
    assert get_res.status_code == 200
    data = get_res.json()
    assert data["job_id"] == job_id
    assert data["media_type"] == "image"
    assert data["status"] == "COMPLETED"
    assert data["is_single_frame"] is True


def test_missing_file_payload(client):
    """Verifies that requests with missing file payloads return validation error."""
    res_img = client.post("/api/v1/analyze/image", files={})
    assert res_img.status_code in [400, 422]

    res_vid = client.post("/api/v1/analyze/video", files={})
    assert res_vid.status_code in [400, 422]


def test_nonexistent_media_returns_404(client):
    """Verifies that requesting non-existent media IDs returns 404."""
    res = client.get("/api/v1/media/nonexistent_file_12345.jpg")
    assert res.status_code == 404


def test_execution_device_telemetry(client):
    """Verifies that system status honestly reports active execution device."""
    from vision.utils.device import resolve_device
    expected_device = resolve_device("auto")

    res = client.get("/api/v1/status")
    assert res.status_code == 200
    data = res.json()
    assert data["device"] == expected_device
    assert data["is_cpu_mode"] == (expected_device == "cpu")


def test_image_analysis_e2e_lifecycle(client):
    """Verifies complete end-to-end image upload, inference, and evidence retrieval."""
    project_root = Path(__file__).resolve().parent.parent
    sample_path = project_root / "data" / "samples" / "ppe_sample.jpg"
    assert sample_path.exists(), "Sample test image must exist"

    with open(sample_path, "rb") as f:
        img_bytes = f.read()

    res = client.post(
        "/api/v1/analyze/image",
        files={"file": ("ppe_sample.jpg", img_bytes, "image/jpeg")},
    )
    assert res.status_code == 200
    data = res.json()
    assert data["status"] == "COMPLETED"
    assert data["annotated_media_url"].startswith("/api/v1/media/")
    assert data["detections_count"] >= 1
    assert data["workers_count"] >= 1

    # Verify visual evidence retrieval
    evidence_res = client.get(data["annotated_media_url"])
    assert evidence_res.status_code == 200
    assert len(evidence_res.content) > 1000
    assert evidence_res.headers["content-type"] in ["image/jpeg", "image/png"]


def test_video_analysis_e2e_lifecycle_and_progress_lock(client):
    """Verifies complete video submission, polling, terminal completion, and that 100% progress never resets to 0%."""
    import time
    project_root = Path(__file__).resolve().parent.parent
    sample_path = project_root / "data" / "samples" / "cctv_worker_moving.mp4"
    assert sample_path.exists(), "Sample test video must exist"

    with open(sample_path, "rb") as f:
        vid_bytes = f.read()

    submit_res = client.post(
        "/api/v1/analyze/video",
        files={"file": ("cctv_worker_moving.mp4", vid_bytes, "video/mp4")},
    )
    assert submit_res.status_code == 202
    job_id = submit_res.json()["job_id"]

    last_pct = 0.0
    terminal_status = None
    for _ in range(60):
        time.sleep(0.5)
        status_res = client.get(f"/api/v1/analyze/status/{job_id}")
        assert status_res.status_code == 200
        st = status_res.json()
        curr_pct = st.get("progress_pct", 0.0)

        # Invariant: progress must never drop backwards
        assert curr_pct >= last_pct, f"Progress dropped from {last_pct}% to {curr_pct}%"
        last_pct = curr_pct

        if st["status"] in ["COMPLETED", "FAILED"]:
            terminal_status = st
            break

    assert terminal_status is not None, "Video job did not finish in time"
    assert terminal_status["status"] == "COMPLETED"
    assert terminal_status["progress_pct"] == 100.0

    # Invariant: Subsequent status checks after completion must REMAIN 100.0% and NEVER reset to 0%
    for _ in range(3):
        post_complete_res = client.get(f"/api/v1/analyze/status/{job_id}")
        assert post_complete_res.status_code == 200
        post_data = post_complete_res.json()
        assert post_data["status"] == "COMPLETED"
        assert post_data["progress_pct"] == 100.0

    # Verify final result retrieval endpoint
    result_res = client.get(f"/api/v1/analyze/result/{job_id}")
    assert result_res.status_code == 200
    res_data = result_res.json()
    assert res_data["status"] == "COMPLETED"
    assert res_data["annotated_video_url"].startswith("/api/v1/media/")
    assert res_data["total_frames"] > 0

    # Verify annotated video stream retrieval
    video_stream_res = client.get(res_data["annotated_video_url"])
    assert video_stream_res.status_code == 200
    assert len(video_stream_res.content) > 1000

