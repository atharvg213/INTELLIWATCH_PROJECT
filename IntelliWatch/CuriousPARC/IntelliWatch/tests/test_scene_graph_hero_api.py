"""
tests/test_scene_graph_hero_api.py
Comprehensive test suite validating the Scene Graph Hero API endpoints, canonical entity representation,
risk engine integration, relationship lifecycle aliases, and job-specific scene retrieval.
"""
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.detection import BoundingBox
from backend.schemas.risk import (
    EventLifecycleState,
    FrameRiskAssessment,
    RiskEvent,
    RiskEventType,
    RiskFactor,
    RiskFactorType,
    RiskLevel,
    RiskSummary,
)
from backend.schemas.scene_graph import (
    FrameScene,
    RelationLifecycle,
    SceneNode,
    SceneNodeType,
    SceneRelation,
    SceneRelationType,
    SituationalSummary,
    enrich_scene_graph,
)
from backend.services.assessment_store import get_assessment_store
from backend.services.risk_store import get_risk_store
from backend.services.scene_store import get_scene_store


@pytest.fixture
def client():
    return TestClient(app)


def test_scene_current_schema_and_entities(client):
    """Verifies that GET /api/v1/scene/current returns the canonical Hero schema."""
    # Setup test scene
    p1 = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        confidence=0.94,
        bbox=BoundingBox(x1=100.0, y1=150.0, x2=200.0, y2=400.0),
        centroid=(150.0, 275.0),
        contact_point=(150.0, 400.0),
        attributes={
            "compliance_status": "NON-COMPLIANT",
            "ppe_status": {"Hardhat": "MISSING", "Safety Vest": "PRESENT"},
            "primary_behavior": "MOVING",
            "speed_px_per_s": 42.5,
        },
        timestamp=10.0,
    )

    f1 = SceneNode(
        node_id="forklift_3",
        node_type=SceneNodeType.VEHICLE,
        class_name="forklift",
        track_id=3,
        confidence=0.89,
        bbox=BoundingBox(x1=300.0, y1=150.0, x2=550.0, y2=450.0),
        centroid=(425.0, 300.0),
        contact_point=(425.0, 450.0),
        attributes={"speed_px_per_s": 65.0},
        timestamp=10.0,
    )

    z1 = SceneNode(
        node_id="zone_hv_01",
        node_type=SceneNodeType.ZONE,
        class_name="zone",
        zone_id="hv_01",
        attributes={"name": "High Voltage Substation", "zone_type": "RESTRICTED", "max_dwell_seconds": 30},
        timestamp=10.0,
    )

    rel_inside = SceneRelation(
        relation_id="rel_person_17_inside_zone_hv_01",
        source_node_id="person_17",
        target_node_id="zone_hv_01",
        relation_type=SceneRelationType.INSIDE,
        lifecycle=RelationLifecycle.ACTIVE,
        confidence=1.0,
        timestamp=10.0,
        duration_seconds=12.5,
        evidence={"dwell_seconds": 12.5},
    )

    rel_approach = SceneRelation(
        relation_id="rel_forklift_3_approaching_person_17",
        source_node_id="forklift_3",
        target_node_id="person_17",
        relation_type=SceneRelationType.APPROACHING,
        lifecycle=RelationLifecycle.ACTIVE,
        confidence=1.0,
        timestamp=10.0,
        duration_seconds=2.0,
        evidence={"distance_px": 142.0, "closing_rate_px_s": 22.0},
    )

    test_scene = FrameScene(
        frame_id=100,
        timestamp=10.0,
        camera_id="cam_01",
        nodes=[p1, f1, z1],
        relationships=[rel_inside, rel_approach],
        summary=SituationalSummary(workers_count=1, vehicles_count=1, active_zones_count=1),
    )

    # Setup matching risk event
    rf = RiskFactor(
        factor_id="rf_01",
        factor_type=RiskFactorType.APPROACHING_VEHICLE,
        severity_contribution=35.0,
        involved_entity_ids=["person_17", "forklift_3"],
        timestamp=10.0,
        explanation="Forklift trajectory converges toward worker at closing velocity 22 px/s.",
    )

    ev = RiskEvent(
        event_id="ev_01",
        event_type=RiskEventType.APPROACHING_VEHICLE,
        lifecycle_state=EventLifecycleState.ACTIVE,
        timestamp=10.0,
        start_timestamp=8.0,
        involved_entities=["person_17", "forklift_3"],
        risk_level=RiskLevel.CRITICAL,
        risk_score=85.0,
        risk_factors=[rf],
        explanation="Imminent collision trajectory detected between Forklift #3 and Worker #17 inside restricted area.",
    )

    test_risk = FrameRiskAssessment(
        frame_id=100,
        timestamp=10.0,
        camera_id="cam_01",
        active_events=[ev],
        risk_factors=[rf],
        risk_summary=RiskSummary(critical=1, total_active_events=1, max_risk_level=RiskLevel.CRITICAL, max_risk_score=85.0),
    )

    get_scene_store().set_current_scene(test_scene)
    get_risk_store().set_current_assessment(test_risk)

    resp = client.get("/api/v1/scene/current")
    assert resp.status_code == 200
    data = resp.json()

    # Backwards compatibility check
    assert "nodes" in data
    assert "relationships" in data
    assert "summary" in data

    # Hero Scene Graph fields
    assert "entities" in data
    assert "source_mode" in data
    assert "is_live" in data
    assert len(data["entities"]) == 3

    # Find worker entity
    worker = next(e for e in data["entities"] if e["id"] == "person_17")
    assert worker["type"] == "PERSON"
    assert worker["label"] in ("Person #17", "Worker #17")
    assert worker["confidence"] == 0.94
    assert worker["position"]["centroid"] == [150.0, 275.0]
    assert worker["state"]["compliance_status"] == "NON-COMPLIANT"
    assert worker["risk"]["level"] == "CRITICAL"
    assert worker["risk"]["score"] == 85.0
    assert "Imminent collision trajectory" in worker["risk"]["reasons"][0]

    # Find zone entity
    zone = next(e for e in data["entities"] if e["id"] == "zone_hv_01")
    assert zone["type"] == "ZONE"
    assert zone["state"]["zone_name"] == "High Voltage Substation"
    assert zone["state"]["workers_inside_count"] == 1
    assert "person_17" in zone["state"]["occupants"]

    # Verify relationships aliases
    rels = data["relationships"]
    assert len(rels) == 2
    for r in rels:
        assert "source_id" in r
        assert "target_id" in r
        assert "type" in r
        assert "state" in r
        assert r["state"] == "ACTIVE"


def test_scene_graph_alias_route(client):
    """Verifies that /api/v1/scene/graph alias route returns the exact same structure."""
    resp = client.get("/api/v1/scene/graph")
    assert resp.status_code == 200
    data = resp.json()
    assert "entities" in data
    assert "relationships" in data


def test_scene_job_not_found(client):
    """Verifies 404 behavior for unknown job ID."""
    resp = client.get("/api/v1/scene/current?job_id=non_existent_job_123")
    assert resp.status_code == 404
    assert "not found" in resp.json()["detail"].lower()

    resp_alias = client.get("/api/v1/analyze/scene/non_existent_job_123")
    assert resp_alias.status_code == 404


def test_real_image_analysis_scene_graph(client):
    """
    Uploads a real industrial image, runs the full neural pipeline, and verifies
    that the resulting scene graph contains real detected entities and factual relationships.
    """
    sample_path = app.state if hasattr(app, "state") else None
    from pathlib import Path
    img_path = Path("data/samples/industrial_cctv.jpg")
    if not img_path.exists():
        pytest.skip("Sample image not found")

    with open(img_path, "rb") as f:
        file_bytes = f.read()

    resp = client.post(
        "/api/v1/analyze/image",
        files={"file": ("industrial_cctv.jpg", file_bytes, "image/jpeg")},
    )
    assert resp.status_code == 200
    res_data = resp.json()
    job_id = res_data["job_id"]
    assert job_id

    # Verify scene for this specific job
    scene_resp = client.get(f"/api/v1/scene/current?job_id={job_id}")
    assert scene_resp.status_code == 200
    scene_data = scene_resp.json()

    assert "entities" in scene_data
    assert "relationships" in scene_data
    assert "summary" in scene_data
    assert scene_data["source_mode"] == "LATEST SCENE SNAPSHOT"
    assert scene_data["is_live"] is False

    # Check that any entities in the graph have real fields
    for entity in scene_data["entities"]:
        assert entity["id"]
        assert entity["type"] in ("PERSON", "VEHICLE", "MACHINE", "OBJECT", "PPE", "ZONE", "OTHER")
        assert "label" in entity
        assert "position" in entity
        assert "state" in entity
        assert "risk" in entity

    # Verify alias endpoint works identically
    alias_resp = client.get(f"/api/v1/analyze/scene/{job_id}")
    assert alias_resp.status_code == 200
    assert alias_resp.json()["entities"] == scene_data["entities"]

    # Image inference must explicitly disclose single-frame limitations
    assert scene_data["time_to_hazard"]["available"] is False
    assert "single-frame" in scene_data["time_to_hazard"]["status_text"].lower()


def test_scene_graph_time_to_hazard_deterministic_calculation(client):
    """
    Verifies deterministic TTH calculation for approaching entities:
    TTH = max(0.0, d - 60.0) / v_closing
    """
    p1 = SceneNode(
        node_id="person_17",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=17,
        confidence=0.95,
        centroid=(100.0, 200.0),
        attributes={
            "compliance_status": "NON-COMPLIANT",
            "ppe_status": {"Hardhat": "MISSING", "Safety Vest": "PRESENT"},
            "speed_px_per_s": 25.0,
        },
        timestamp=5.0,
    )
    f1 = SceneNode(
        node_id="forklift_3",
        node_type=SceneNodeType.VEHICLE,
        class_name="forklift",
        track_id=3,
        confidence=0.91,
        centroid=(242.0, 200.0),
        attributes={"speed_px_per_s": 45.0},
        timestamp=5.0,
    )

    rel_approach = SceneRelation(
        relation_id="rel_forklift_3_approaching_person_17",
        source_node_id="forklift_3",
        target_node_id="person_17",
        relation_type=SceneRelationType.APPROACHING,
        lifecycle=RelationLifecycle.ACTIVE,
        confidence=0.89,
        timestamp=5.0,
        duration_seconds=1.5,
        evidence={"distance_px": 142.0, "closing_rate_px_s": 22.0},
    )

    scene = FrameScene(
        frame_id=50,
        timestamp=5.0,
        camera_id="cam_01",
        nodes=[p1, f1],
        relationships=[rel_approach],
    )
    # Mark as video to enable temporal kinematic reasoning
    scene.is_video = True

    enriched = enrich_scene_graph(scene, None, is_live=False, source_mode="VIDEO_STREAM")

    # Formula: round(max(0.0, 142.0 - 60.0) / 22.0, 1) = round(82.0 / 22.0, 1) = 3.7 s
    assert enriched.relationships[0].tth_seconds == 3.7
    assert enriched.time_to_hazard is not None
    assert enriched.time_to_hazard["available"] is True
    assert enriched.time_to_hazard["time_to_hazard_seconds"] == 3.7
    assert enriched.time_to_hazard["source_id"] == "forklift_3"
    assert enriched.time_to_hazard["target_id"] == "person_17"
    assert len(enriched.time_to_hazard["timeline"]) == 3
    assert enriched.time_to_hazard["timeline"][0]["phase"] == "CURRENT"
    assert enriched.time_to_hazard["timeline"][2]["phase"] == "HAZARD_POINT"
    assert enriched.time_to_hazard["timeline"][2]["time_offset_s"] == 3.7

    # Compounding factors must only contain facts
    factors = enriched.time_to_hazard["compounding_factors"]
    assert any("PPE violation" in f for f in factors)
    assert any("approach at 22" in f for f in factors)


def test_scene_graph_single_frame_honest_disclosure(client):
    """
    Verifies that single-frame static image scene graph truthfully returns
    Unavailable status for TTH and velocity, without fabricated values.
    """
    p1 = SceneNode(
        node_id="person_01",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=None,  # No temporal tracking in single frame
        confidence=0.92,
        centroid=(200.0, 300.0),
        attributes={"compliance_status": "COMPLIANT"},
        timestamp=0.0,
    )
    scene = FrameScene(
        frame_id=1,
        timestamp=0.0,
        camera_id="cam_static",
        nodes=[p1],
        relationships=[],
    )
    enriched = enrich_scene_graph(scene, None, is_live=False, source_mode="STATIC_IMAGE")

    assert enriched.time_to_hazard["available"] is False
    assert enriched.time_to_hazard["time_to_hazard_seconds"] is None
    assert "single-frame analysis has no motion vectors" in enriched.time_to_hazard["status_text"]

    # Person state must state velocity is not available
    person_entity = enriched.entities[0]
    assert person_entity["state"]["speed_px_per_s"] is None
    assert "single-frame" in person_entity["state"]["velocity_status"].lower()


