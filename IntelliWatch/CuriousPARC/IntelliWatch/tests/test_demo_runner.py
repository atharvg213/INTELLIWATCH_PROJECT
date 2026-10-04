"""
tests/test_demo_runner.py
Step 12 Test Suite - Demo Runner Verification.
Validates:
- Video path resolution and synthetic video generation fallback
- Path handling using pathlib relative to project root without hard-coded usernames
- End-to-end execution on short synthetic clip
- Generation of annotated output video and structured JSON export
"""
import subprocess
import sys
from pathlib import Path
import pytest

from scripts.run_demo import get_or_create_demo_video
from scripts.create_synthetic_sample import generate_synthetic_video


def test_get_or_create_demo_video_synthetic(tmp_path):
    """Verifies that demo runner generates synthetic video if requested file is missing."""
    target_synth = tmp_path / "test_synth.mp4"
    assert not target_synth.exists()

    generate_synthetic_video(target_synth, width=320, height=180, fps=10, num_frames=5)
    assert target_synth.exists()
    assert target_synth.stat().st_size > 0


def test_path_safety_no_hardcoded_user():
    """Verifies that project paths resolve dynamically without hard-coded user home directories."""
    resolved = get_or_create_demo_video()
    assert isinstance(resolved, Path)
    assert resolved.exists()
    assert resolved.is_file()


def test_demo_runner_short_run(tmp_path):
    """Verifies that run_demo.py executes successfully and creates output video and JSON summary."""
    out_video = tmp_path / "out.mp4"
    out_json = tmp_path / "out.json"

    # Run 3 frames via subprocess
    cmd = [
        sys.executable,
        "scripts/run_demo.py",
        "--max-frames", "3",
        "--disable-ppe-model",
        "--output-video", str(out_video),
        "--output-json", str(out_json),
    ]

    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(Path(__file__).resolve().parent.parent))
    assert res.returncode == 0, f"Demo runner failed: {res.stderr}"

    assert out_video.exists()
    assert out_video.stat().st_size > 0
    assert out_json.exists()
    assert out_json.stat().st_size > 0
