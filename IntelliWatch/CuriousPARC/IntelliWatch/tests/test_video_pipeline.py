"""
Unit and Integration Tests for Step 2: Video Input + Frame Processing
Tests video reading, metadata parsing, sequential frame extraction,
frame sampling, aspect-ratio letterboxing, normalization, and error handling.
"""
import sys
import tempfile
from pathlib import Path
import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.preprocessing.frame import FrameData
from vision.preprocessing.frame_processor import FramePreprocessor
from vision.preprocessing.pipeline import VideoPipeline
from vision.preprocessing.video_reader import (
    VideoFileNotFoundError,
    VideoMetadata,
    VideoOpenError,
    VideoReader,
)


@pytest.fixture(scope="session")
def synthetic_video_path(tmp_path_factory) -> Path:
    """
    Creates a temporary 30-frame 320x240 MP4 video for isolated testing.
    Does not require internet access or external downloads.
    """
    temp_dir = tmp_path_factory.mktemp("test_videos")
    video_path = temp_dir / "unit_test_video.mp4"

    width, height, fps, num_frames = 320, 240, 30, 30
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(video_path), fourcc, fps, (width, height))

    for i in range(num_frames):
        # Draw frame with a unique color gradient and timestamp
        frame = np.full((height, width, 3), (i * 8) % 255, dtype=np.uint8)
        cv2.circle(frame, (width // 2, height // 2), 20 + i, (0, 255, 0), -1)
        out.write(frame)

    out.release()
    return video_path


def test_video_file_not_found():
    """Verify that a non-existent path raises VideoFileNotFoundError."""
    reader = VideoReader("non_existent_path_to_video.mp4")
    with pytest.raises(VideoFileNotFoundError):
        reader.open()


def test_video_metadata_extraction(synthetic_video_path):
    """Verify correct video dimension, fps, and duration extraction."""
    reader = VideoReader(synthetic_video_path)
    meta = reader.open()

    assert isinstance(meta, VideoMetadata)
    assert meta.width == 320
    assert meta.height == 240
    assert meta.fps == 30.0
    assert meta.total_frames == 30
    assert meta.duration_sec == pytest.approx(1.0, rel=0.1)

    reader.release()
    assert not reader.is_open


def test_sequential_frame_reading(synthetic_video_path):
    """Verify sequential frame reading and end-of-video detection."""
    with VideoReader(synthetic_video_path) as reader:
        frames = list(reader.read_frames(frame_skip=1))
        assert len(frames) == 30
        for idx, frame in enumerate(frames):
            assert isinstance(frame, FrameData)
            assert frame.frame_index == idx
            assert frame.width == 320
            assert frame.height == 240
            assert frame.image.shape == (240, 320, 3)
            assert frame.timestamp == pytest.approx(idx / 30.0, abs=1e-3)


def test_frame_sampling(synthetic_video_path):
    """Verify that frame_skip correctly samples every Nth frame."""
    with VideoReader(synthetic_video_path) as reader:
        # frame_skip=5 on 30 frames yields frames 0, 5, 10, 15, 20, 25 (6 frames)
        sampled = list(reader.read_frames(frame_skip=5))
        assert len(sampled) == 6
        expected_indices = [0, 5, 10, 15, 20, 25]
        actual_indices = [f.frame_index for f in sampled]
        assert actual_indices == expected_indices


def test_frame_preprocessor_resizing():
    """Verify direct resizing and letterbox padding with aspect ratio preservation."""
    raw_frame = np.zeros((300, 600, 3), dtype=np.uint8)
    frame_data = FrameData(
        frame_index=1,
        timestamp=0.033,
        image=raw_frame,
        width=600,
        height=300,
    )

    # 1. Aspect Ratio Preserving Letterbox (to 640x640)
    preproc_letterbox = FramePreprocessor(
        target_width=640,
        target_height=640,
        resize_enabled=True,
        preserve_aspect_ratio=True,
    )
    result = preproc_letterbox.preprocess(frame_data)
    assert result.shape == (640, 640, 3)
    assert result.is_preprocessed is True
    assert result.scale_factor is not None
    assert result.pad_offset is not None

    # 2. Direct Resize (without letterbox)
    preproc_direct = FramePreprocessor(
        target_width=640,
        target_height=640,
        resize_enabled=True,
        preserve_aspect_ratio=False,
    )
    result_direct = preproc_direct.preprocess(frame_data)
    assert result_direct.shape == (640, 640, 3)
    assert result_direct.pad_offset == (0, 0)


def test_frame_preprocessor_normalization():
    """Verify pixel normalization scales uint8 values to [0.0, 1.0] float32."""
    raw_frame = np.full((100, 100, 3), 255, dtype=np.uint8)
    preproc = FramePreprocessor(
        resize_enabled=False,
        normalize=True,
    )
    normalized = preproc.preprocess(raw_frame)
    assert normalized.dtype == np.float32
    assert np.allclose(normalized, 1.0)


def test_invalid_frame_handling():
    """Verify that empty or invalid inputs raise appropriate ValueErrors."""
    preproc = FramePreprocessor()
    with pytest.raises(ValueError):
        preproc.preprocess(None)

    empty_array = np.array([])
    with pytest.raises(ValueError):
        preproc.preprocess(empty_array)


def test_video_pipeline_end_to_end(synthetic_video_path):
    """Verify complete pipeline ingestion and batch telemetry."""
    preproc = FramePreprocessor(target_width=640, target_height=640)
    pipeline = VideoPipeline(
        video_path=synthetic_video_path,
        frame_skip=2,  # Process every 2nd frame (15 frames total)
        preprocessor=preproc,
    )

    summary = pipeline.process_all(log_interval=10)
    assert summary["frames_processed"] == 15
    assert summary["frame_skip"] == 2
    assert summary["elapsed_seconds"] >= 0.0
    assert summary["effective_fps"] > 0.0


def test_video_metadata_api(synthetic_video_path):
    """Verify the minimal /api/v1/video/metadata inspection endpoint."""
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    # Test GET metadata
    resp = client.get(f"/api/v1/video/metadata?video_path={synthetic_video_path}")
    assert resp.status_code == 200
    data = resp.json()
    assert data["width"] == 320
    assert data["height"] == 240
    assert data["fps"] == 30.0
    assert data["total_frames"] == 30

    # Test POST inspect
    resp_post = client.post("/api/v1/video/inspect", json={"video_path": str(synthetic_video_path)})
    assert resp_post.status_code == 200
    assert resp_post.json()["total_frames"] == 30

    # Test non-existent file
    resp_404 = client.get("/api/v1/video/metadata?video_path=nonexistent.mp4")
    assert resp_404.status_code == 404
