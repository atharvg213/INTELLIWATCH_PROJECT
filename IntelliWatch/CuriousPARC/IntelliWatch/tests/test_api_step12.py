"""
tests/test_api_step12.py
Step 12 Test Suite - API Endpoints for Assessment, Evidence & Telemetry.
Validates:
- GET /api/v1/assessment/current
- GET /api/v1/incidents
- GET /api/v1/incidents/{incident_id}
- GET /api/v1/incidents/{incident_id}/evidence
- GET /api/v1/status
- GET / and /dashboard static console response
"""
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.evidence import IncidentRecord
from backend.schemas.risk import EventLifecycleState, RiskEventType, RiskLevel
from backend.services.incident_store import get_incident_store


@pytest.fixture
def client():
    """Returns FastAPI TestClient."""
    return TestClient(app)


def test_get_current_assessment_endpoint(client):
    """Verifies that GET /api/v1/assessment/current returns a valid FrameAssessment schema."""
    response = client.get("/api/v1/assessment/current")
    assert response.status_code == 200
    data = response.json()

    assert "frame_id" in data
    assert "timestamp" in data
    assert "detections" in data
    assert "tracks" in data
    assert "worker_inventories" in data
    assert "zone_memberships" in data
    assert "highest_risk_level" in data


def test_get_system_status_endpoint(client):
    """Verifies that GET /api/v1/status returns operational status and CPU mode telemetry."""
    response = client.get("/api/v1/status")
    assert response.status_code == 200
    data = response.json()

    assert data["status"] == "healthy"
    assert "device" in data
    assert "is_cpu_mode" in data
    assert "processing_fps" in data
    assert "models_loaded" in data
    assert isinstance(data["models_loaded"], dict)


def test_incidents_crud_endpoints(client):
    """Verifies full incident listing, retrieval by ID, and snapshot evidence retrieval."""
    incident_store = get_incident_store()
    incident_store.reset()

    # Create and store test incident with synthetic image snapshot
    inc = IncidentRecord(
        incident_id="inc_api_test_01",
        timestamp=2.5,
        frame_id=75,
        camera_id="cam_01",
        event_type=RiskEventType.PPE_VIOLATION,
        risk_level=RiskLevel.HIGH,
        risk_score=70.0,
        lifecycle_state=EventLifecycleState.CONFIRMED,
        involved_track_ids=[3],
        involved_entity_ids=["person_3"],
        explanation="Worker #3 missing Hardhat in operational area.",
    )
    frame = np.full((100, 100, 3), 150, dtype=np.uint8)
    incident_store.record_incident(inc, frame_image=frame)

    # 1. List incidents
    list_res = client.get("/api/v1/incidents")
    assert list_res.status_code == 200
    inc_list = list_res.json()
    assert len(inc_list) >= 1
    assert any(i["incident_id"] == "inc_api_test_01" for i in inc_list)

    # 2. Get specific incident
    detail_res = client.get("/api/v1/incidents/inc_api_test_01")
    assert detail_res.status_code == 200
    detail = detail_res.json()
    assert detail["incident_id"] == "inc_api_test_01"
    assert detail["risk_level"] == "HIGH"
    assert detail["has_snapshot"] is True

    # 3. Retrieve evidence snapshot
    evidence_res = client.get("/api/v1/incidents/inc_api_test_01/evidence")
    assert evidence_res.status_code == 200
    assert evidence_res.headers["content-type"] == "image/jpeg"

    # 4. Unknown incident returns 404
    missing_res = client.get("/api/v1/incidents/inc_nonexistent_99")
    assert missing_res.status_code == 404


def test_dashboard_route_serves_html(client):
    """Verifies that GET / and GET /dashboard serve the operator dashboard HTML."""
    res_root = client.get("/")
    assert res_root.status_code == 200

    res_dash = client.get("/dashboard")
    assert res_dash.status_code == 200
    assert "INTELLIWATCH" in res_dash.text or "text/html" in res_dash.headers.get("content-type", "")
