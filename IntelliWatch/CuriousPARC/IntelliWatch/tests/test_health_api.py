"""
Test FastAPI application and /health endpoint.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi.testclient import TestClient
from backend.main import app

client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data == {
        "status": "ok",
        "project": "IntelliWatch"
    }


def test_public_config_endpoint():
    response = client.get("/api/v1/config")
    assert response.status_code == 200
    data = response.json()
    assert data["project"] == "IntelliWatch"
    assert "version" in data
    assert "environment" in data
    assert "zones" in data
    assert data["zones"]["enabled"] is True
    assert data["zones"]["confirmation_frames"] == 3
    assert data["zones"]["max_dwell_seconds"] == 10.0
    assert data["zones"]["config_path"] == "configs/zones.json"
    assert "depth" in data
    assert data["depth"]["enabled"] is True
    assert data["depth"]["device"] == "cpu"
