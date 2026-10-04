"""
tests/test_camera_calibration.py
Deterministic test suite validating Camera Ground-Plane Perspective Calibration,
homography calculations, geometric validation, metric spatial estimation,
Scene Graph integration, TTH compatibility, and honest unavailable fallbacks.
"""
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.calibration import (
    CalibrationSaveRequest,
    CalibrationStatus,
    CalibrationValidateRequest,
    CameraCalibrationConfig,
    Point2D,
)
from backend.schemas.detection import BoundingBox
from backend.schemas.scene_graph import (
    FrameScene,
    RelationLifecycle,
    SceneNode,
    SceneNodeType,
    SceneRelation,
    SceneRelationType,
    enrich_scene_graph,
)
from backend.services.calibration_service import (
    CalibrationService,
    GroundPlaneCalibrator,
    check_quadrilateral_validity,
    get_calibration_service,
)


@pytest.fixture
def client():
    return TestClient(app)


@pytest.fixture
def clean_calibration_service(tmp_path):
    """Provides an isolated CalibrationService instance with temporary persistence."""
    temp_json = tmp_path / "test_calibrations.json"
    service = CalibrationService(config_path=temp_json)
    return service


def test_1_valid_four_point_calibration():
    """1. Valid four-point calibration correctly compiles homography and computes reprojection error."""
    points = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=300.0, y=100.0),
        Point2D(x=300.0, y=300.0),
        Point2D(x=100.0, y=300.0),
    ]
    is_valid, msg = check_quadrilateral_validity(points)
    assert is_valid is True

    config = CameraCalibrationConfig(
        camera_id="cam_test_01",
        calibration_enabled=True,
        points=points,
        real_world_width_m=10.0,
        real_world_depth_m=10.0,
    )
    calibrator = GroundPlaneCalibrator(config)
    assert calibrator.is_valid is True
    assert calibrator.H is not None
    assert calibrator.H_inv is not None
    assert calibrator.reprojection_error_px is not None
    assert calibrator.reprojection_error_px < 0.05
    assert calibrator.config.calibration_status == CalibrationStatus.CALIBRATED


def test_2_invalid_point_count():
    """2. Rejects calibrations with point count != 4."""
    # 3 points
    p_3 = [Point2D(x=100, y=100), Point2D(x=200, y=100), Point2D(x=150, y=200)]
    valid, msg = check_quadrilateral_validity(p_3)
    assert valid is False
    assert "Exactly 4" in msg

    # 5 points
    p_5 = p_3 + [Point2D(x=120, y=220), Point2D(x=110, y=250)]
    valid5, msg5 = check_quadrilateral_validity(p_5)
    assert valid5 is False
    assert "Exactly 4" in msg5


def test_3_invalid_dimensions(clean_calibration_service):
    """3. Rejects negative or zero real-world width and depth."""
    points = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=300.0, y=100.0),
        Point2D(x=300.0, y=300.0),
        Point2D(x=100.0, y=300.0),
    ]
    # Zero width
    with pytest.raises(ValueError):
        req = CalibrationSaveRequest(
            points=points,
            real_world_width_m=0.0,
            real_world_depth_m=10.0,
        )

    # Negative depth
    with pytest.raises(ValueError):
        req = CalibrationSaveRequest(
            points=points,
            real_world_width_m=10.0,
            real_world_depth_m=-5.0,
        )


def test_4_degenerate_and_collinear_points():
    """4. Rejects collinear or non-convex / self-intersecting points."""
    # 3 collinear points
    collinear = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=200.0, y=200.0),
        Point2D(x=300.0, y=300.0),
        Point2D(x=100.0, y=300.0),
    ]
    valid, msg = check_quadrilateral_validity(collinear)
    assert valid is False
    assert "collinear" in msg.lower()

    # Non-convex dart quadrilateral
    dart = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=300.0, y=100.0),
        Point2D(x=200.0, y=200.0),
        Point2D(x=300.0, y=300.0),
    ]
    valid_d, msg_d = check_quadrilateral_validity(dart)
    assert valid_d is False
    assert "convex" in msg_d.lower()

    # Self-intersecting / hourglass quadrilateral
    hourglass = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=300.0, y=300.0),
        Point2D(x=300.0, y=100.0),
        Point2D(x=100.0, y=300.0),
    ]
    valid_h, msg_h = check_quadrilateral_validity(hourglass)
    assert valid_h is False
    assert ("convex" in msg_h.lower() or "degenerate" in msg_h.lower())


def test_5_point_transformation_invertibility():
    """5. Point transformation between image and ground plane is strictly invertible."""
    points = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=500.0, y=100.0),
        Point2D(x=600.0, y=400.0),
        Point2D(x=50.0, y=400.0),
    ]
    config = CameraCalibrationConfig(
        camera_id="cam_persp",
        calibration_enabled=True,
        points=points,
        real_world_width_m=15.0,
        real_world_depth_m=20.0,
    )
    calibrator = GroundPlaneCalibrator(config)
    assert calibrator.is_valid is True

    # Test point inside perspective region
    test_u, test_v = 300.0, 250.0
    ground_pt = calibrator.image_to_ground(test_u, test_v)
    assert ground_pt is not None

    # Invert back to image
    img_pt = calibrator.ground_to_image(ground_pt[0], ground_pt[1])
    assert img_pt is not None
    assert abs(img_pt[0] - test_u) < 0.2
    assert abs(img_pt[1] - test_v) < 0.2


def test_6_known_calibration_transformation():
    """6. Known calibration maps exact coordinates on a known 10m x 10m square."""
    # 200px x 200px square = 10m x 10m ground plane
    # Scale = 20 px per meter, or 0.05 m per pixel
    points = [
        Point2D(x=100.0, y=100.0), # P1 -> (0, 0)m
        Point2D(x=300.0, y=100.0), # P2 -> (10, 0)m
        Point2D(x=300.0, y=300.0), # P3 -> (10, 10)m
        Point2D(x=100.0, y=300.0), # P4 -> (0, 10)m
    ]
    config = CameraCalibrationConfig(
        camera_id="cam_known",
        calibration_enabled=True,
        points=points,
        real_world_width_m=10.0,
        real_world_depth_m=10.0,
    )
    calibrator = GroundPlaneCalibrator(config)

    # Corner 1
    p1_m = calibrator.image_to_ground(100.0, 100.0)
    assert p1_m == (0.0, 0.0)

    # Corner 2
    p2_m = calibrator.image_to_ground(300.0, 100.0)
    assert p2_m == (10.0, 0.0)

    # Corner 3
    p3_m = calibrator.image_to_ground(300.0, 300.0)
    assert p3_m == (10.0, 10.0)

    # Center
    center_m = calibrator.image_to_ground(200.0, 200.0)
    assert center_m == (5.0, 5.0)


def test_7_metric_distance_calculation():
    """7. Ground-plane distance in meters is calculated accurately between image points."""
    points = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=300.0, y=100.0),
        Point2D(x=300.0, y=300.0),
        Point2D(x=100.0, y=300.0),
    ]
    config = CameraCalibrationConfig(
        camera_id="cam_dist",
        calibration_enabled=True,
        points=points,
        real_world_width_m=10.0,
        real_world_depth_m=10.0,
    )
    calibrator = GroundPlaneCalibrator(config)

    # Distance between P1 and P2 (top edge, 200px = 10m)
    dist = calibrator.compute_ground_distance((100.0, 100.0), (300.0, 100.0))
    assert dist == 10.0

    # Distance between center and P3: (5,5) to (10,10) = sqrt(25+25) = 7.071m
    dist_diag = calibrator.compute_ground_distance((200.0, 200.0), (300.0, 300.0))
    assert dist_diag == pytest.approx(7.071, abs=0.01)


def test_8_metric_velocity_calculation():
    """8. Ground-plane velocity in m/s is derived accurately from image displacement and planar scale."""
    points = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=300.0, y=100.0),
        Point2D(x=300.0, y=300.0),
        Point2D(x=100.0, y=300.0),
    ]
    config = CameraCalibrationConfig(
        camera_id="cam_vel",
        calibration_enabled=True,
        points=points,
        real_world_width_m=10.0,
        real_world_depth_m=10.0,
    )
    calibrator = GroundPlaneCalibrator(config)

    # 20 px/s speed at (200, 200) with velocity along X: (20, 0)
    # Scale is 0.05 m/px -> 20 px/s = 1.0 m/s
    speed_m_s = calibrator.compute_ground_speed(200.0, 200.0, 20.0, (20.0, 0.0))
    assert speed_m_s == 1.0
    # A speed magnitude without direction cannot be converted through perspective.
    assert calibrator.compute_ground_speed(200.0, 200.0, 20.0) is None

    # Zero speed returns 0.0
    zero_spd = calibrator.compute_ground_speed(200.0, 200.0, 0.0)
    assert zero_spd == 0.0

    # None speed returns None
    none_spd = calibrator.compute_ground_speed(200.0, 200.0, None)
    assert none_spd is None


def test_9_missing_calibration_fallback():
    """9. System strictly retains pixel space and reports honest unavailable when uncalibrated."""
    p1 = SceneNode(
        node_id="person_1",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=1,
        bbox=BoundingBox(x1=50, y1=50, x2=100, y2=150),
        centroid=(75.0, 100.0),
        contact_point=(75.0, 150.0),
        attributes={"speed_px_per_s": 25.0},
    )
    v1 = SceneNode(
        node_id="forklift_2",
        node_type=SceneNodeType.VEHICLE,
        class_name="forklift",
        track_id=2,
        bbox=BoundingBox(x1=200, y1=100, x2=350, y2=250),
        centroid=(275.0, 175.0),
        contact_point=(275.0, 250.0),
        attributes={"speed_px_per_s": 40.0},
    )
    rel = SceneRelation(
        relation_id="rel_p1_v1",
        source_node_id="person_1",
        target_node_id="forklift_2",
        relation_type=SceneRelationType.APPROACHING,
        evidence={"distance_px": 223.6, "closing_rate_px_s": 15.0},
    )

    scene = FrameScene(
        scene_id="uncalibrated_scene",
        camera_id="non_existent_cam",
        timestamp=1.0,
        nodes=[p1, v1],
        relationships=[rel],
        is_video=True,
    )

    enriched = enrich_scene_graph(scene, None, is_live=False, source_mode="VIDEO_STREAM")

    # Honest disclosure checks
    assert enriched.calibration_status == "UNCONFIGURED"
    assert enriched.spatial_basis == "IMAGE_SPACE"

    # Entities ground plane must be None
    ent1 = next(e for e in enriched.entities if e["id"] == "person_1")
    assert ent1["position"]["ground_plane"] is None
    assert ent1["position"]["spatial_basis"] == "IMAGE_SPACE"
    assert ent1["state"]["speed_m_per_s"] is None
    assert "camera calibration required" in ent1["state"]["ground_speed_status"].lower()

    # Relationship metric values must be None
    r_out = enriched.relationships[0]
    assert r_out.distance_m is None
    assert r_out.closing_rate_m_s is None
    assert r_out.spatial_basis == "IMAGE_SPACE"
    assert r_out.evidence["distance_m"] is None
    assert r_out.evidence["closing_rate_m_s"] is None
    assert r_out.evidence["spatial_basis"] == "IMAGE_SPACE"


def test_10_missing_contact_point_fallback(clean_calibration_service):
    """10. When contact point is missing, ground-plane estimation remains unavailable without inventing centroids."""
    # Register active calibration
    points = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=300.0, y=100.0),
        Point2D(x=300.0, y=300.0),
        Point2D(x=100.0, y=300.0),
    ]
    req = CalibrationSaveRequest(
        points=points,
        real_world_width_m=10.0,
        real_world_depth_m=10.0,
    )
    get_calibration_service().save_calibration("cam_active", req)

    # Node without contact point
    p_no_contact = SceneNode(
        node_id="person_ghost",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=99,
        bbox=BoundingBox(x1=50, y1=50, x2=100, y2=150),
        centroid=(75.0, 100.0),
        contact_point=None,  # Intentionally missing
        attributes={
            "speed_px_per_s": 20.0,
            "velocity_x_px_per_s": 20.0,
            "velocity_y_px_per_s": 0.0,
            "motion_observed": True,
        },
    )

    scene = FrameScene(
        scene_id="scene_no_contact",
        camera_id="cam_active",
        timestamp=1.0,
        nodes=[p_no_contact],
        relationships=[],
        is_video=True,
    )

    enriched = enrich_scene_graph(scene, None, is_live=False, source_mode="VIDEO_STREAM")
    ent = enriched.entities[0]

    # Must remain unavailable because contact point is absent
    assert ent["position"]["contact_point"] is None
    assert ent["position"]["ground_plane"] is None
    assert ent["position"]["spatial_basis"] == "IMAGE_SPACE"
    assert ent["state"]["speed_m_per_s"] is None


def test_11_scene_graph_calibrated_integration():
    """11. Scene Graph entities and relationships are populated with metric coordinates when calibrated."""
    points = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=300.0, y=100.0),
        Point2D(x=300.0, y=300.0),
        Point2D(x=100.0, y=300.0),
    ]
    req = CalibrationSaveRequest(
        points=points,
        real_world_width_m=10.0,
        real_world_depth_m=10.0,
    )
    get_calibration_service().save_calibration("cam_calibrated_01", req)

    p1 = SceneNode(
        node_id="person_10",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=10,
        bbox=BoundingBox(x1=100, y1=50, x2=100, y2=100),
        centroid=(100.0, 75.0),
        contact_point=(100.0, 100.0),  # At P1 -> (0, 0)m
        attributes={
            "speed_px_per_s": 20.0,
            "velocity_x_px_per_s": 20.0,
            "velocity_y_px_per_s": 0.0,
            "motion_observed": True,
        },
    )
    v1 = SceneNode(
        node_id="forklift_20",
        node_type=SceneNodeType.VEHICLE,
        class_name="forklift",
        track_id=20,
        bbox=BoundingBox(x1=300, y1=50, x2=300, y2=100),
        centroid=(300.0, 75.0),
        contact_point=(300.0, 100.0),  # At P2 -> (10, 0)m
        attributes={
            "speed_px_per_s": 0.0,
            "velocity_x_px_per_s": 0.0,
            "velocity_y_px_per_s": 0.0,
            "motion_observed": True,
        },
    )
    rel = SceneRelation(
        relation_id="rel_p10_v20",
        source_node_id="person_10",
        target_node_id="forklift_20",
        relation_type=SceneRelationType.APPROACHING,
        evidence={"distance_px": 200.0, "closing_rate_px_s": 20.0},
    )

    scene = FrameScene(
        scene_id="cal_scene",
        camera_id="cam_calibrated_01",
        timestamp=2.0,
        nodes=[p1, v1],
        relationships=[rel],
        is_video=True,
    )

    enriched = enrich_scene_graph(scene, None, is_live=False, source_mode="VIDEO_STREAM")

    assert enriched.calibration_status in ("CALIBRATED", "CONFIGURED")
    assert enriched.spatial_basis == "GROUND_PLANE_APPROXIMATION"

    # Person ground position
    e_p = next(e for e in enriched.entities if e["id"] == "person_10")
    assert e_p["position"]["ground_plane"] == {"x_m": 0.0, "y_m": 0.0}
    assert e_p["position"]["spatial_basis"] == "GROUND_PLANE_APPROXIMATION"
    assert e_p["state"]["speed_m_per_s"] == 1.0

    # Forklift ground position
    e_v = next(e for e in enriched.entities if e["id"] == "forklift_20")
    assert e_v["position"]["ground_plane"] == {"x_m": 10.0, "y_m": 0.0}

    # Relationship evidence
    r = enriched.relationships[0]
    assert r.distance_px == 200.0
    assert r.distance_m == 10.0
    assert r.closing_rate_px_s == 20.0
    assert r.closing_rate_m_s == 1.0  # Projected pair separation decreases by 1m over one second.
    assert r.spatial_basis == "GROUND_PLANE_APPROXIMATION"
    assert r.evidence["spatial_basis"] == "GROUND_PLANE_APPROXIMATION"


def test_12_tth_compatibility_with_calibration():
    """12. TTH remains explicitly image-space even when ground-plane distances exist."""
    points = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=300.0, y=100.0),
        Point2D(x=300.0, y=300.0),
        Point2D(x=100.0, y=300.0),
    ]
    req = CalibrationSaveRequest(
        points=points,
        real_world_width_m=10.0,
        real_world_depth_m=10.0,
    )
    get_calibration_service().save_calibration("cam_tth", req)

    p1 = SceneNode(
        node_id="person_tth",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=1,
        bbox=BoundingBox(x1=100, y1=50, x2=100, y2=100),
        centroid=(100.0, 75.0),
        contact_point=(100.0, 100.0),
        attributes={
            "speed_px_per_s": 20.0,
            "velocity_x_px_per_s": 20.0,
            "velocity_y_px_per_s": 0.0,
            "motion_observed": True,
        },
    )
    v1 = SceneNode(
        node_id="forklift_tth",
        node_type=SceneNodeType.VEHICLE,
        class_name="forklift",
        track_id=2,
        bbox=BoundingBox(x1=300, y1=50, x2=300, y2=100),
        centroid=(300.0, 75.0),
        contact_point=(300.0, 100.0),
        attributes={
            "speed_px_per_s": 0.0,
            "velocity_x_px_per_s": 0.0,
            "velocity_y_px_per_s": 0.0,
            "motion_observed": True,
        },
    )
    rel = SceneRelation(
        relation_id="rel_tth_test",
        source_node_id="person_tth",
        target_node_id="forklift_tth",
        relation_type=SceneRelationType.APPROACHING,
        evidence={"distance_px": 200.0, "closing_rate_px_s": 20.0},
    )

    # A. Multi-frame video
    scene_vid = FrameScene(
        scene_id="scene_vid",
        camera_id="cam_tth",
        timestamp=2.0,
        nodes=[p1, v1],
        relationships=[rel],
        is_video=True,
    )
    enriched_vid = enrich_scene_graph(scene_vid, None, is_live=False, source_mode="VIDEO_STREAM")

    assert enriched_vid.time_to_hazard["available"] is True
    # (200px - 60px) / 20 px/s = 7.0s
    assert enriched_vid.time_to_hazard["time_to_hazard_seconds"] == 7.0
    assert enriched_vid.time_to_hazard["spatial_basis"] == "IMAGE_SPACE"
    assert "image-space" in enriched_vid.time_to_hazard["status_text"].lower()
    assert "distance_m" not in enriched_vid.time_to_hazard
    assert "closing_rate_m_s" not in enriched_vid.time_to_hazard

    # B. Single-frame image must remain unavailable despite calibration
    scene_img = FrameScene(
        scene_id="scene_img",
        camera_id="cam_tth",
        timestamp=2.0,
        nodes=[p1, v1],
        relationships=[rel],
        is_video=False,
    )
    enriched_img = enrich_scene_graph(scene_img, None, is_live=False, source_mode="STATIC_IMAGE")
    assert enriched_img.time_to_hazard["available"] is False
    assert enriched_img.time_to_hazard["time_to_hazard_seconds"] is None
    assert "single-frame" in enriched_img.time_to_hazard["status_text"].lower()


def test_13_calibration_api_crud_and_validation(client):
    """13. Calibration API endpoints support complete validation and CRUD lifecycle."""
    camera_id = "cam_api_test"

    # Step 1: Validate proposed points
    val_payload = {
        "camera_id": camera_id,
        "points": [
            {"x": 100, "y": 100},
            {"x": 400, "y": 100},
            {"x": 450, "y": 350},
            {"x": 50, "y": 350},
        ],
        "real_world_width_m": 8.0,
        "real_world_depth_m": 12.0,
    }
    resp_val = client.post(f"/api/v1/cameras/{camera_id}/calibration/validate", json=val_payload)
    assert resp_val.status_code == 200
    val_data = resp_val.json()
    assert val_data["is_valid"] is True
    assert val_data["reprojection_error_px"] is not None
    assert len(val_data["sample_projections"]) == 4

    # Step 2: Save calibration
    save_payload = {
        "camera_id": camera_id,
        "calibration_enabled": True,
        "points": val_payload["points"],
        "real_world_width_m": 8.0,
        "real_world_depth_m": 12.0,
    }
    resp_save = client.put(f"/api/v1/cameras/{camera_id}/calibration", json=save_payload)
    assert resp_save.status_code == 200
    save_data = resp_save.json()
    assert save_data["calibration_status"] == "CALIBRATED"
    assert save_data["real_world_width_m"] == 8.0

    # Step 3: Retrieve saved calibration
    resp_get = client.get(f"/api/v1/cameras/{camera_id}/calibration")
    assert resp_get.status_code == 200
    get_data = resp_get.json()
    assert get_data["calibration_status"] == "CALIBRATED"
    assert len(get_data["points"]) == 4

    # Step 4: List all calibrations
    resp_list = client.get("/api/v1/calibrations")
    assert resp_list.status_code == 200
    assert any(c["camera_id"] == camera_id for c in resp_list.json())

    # Step 5: Delete calibration
    resp_del = client.delete(f"/api/v1/cameras/{camera_id}/calibration")
    assert resp_del.status_code == 200
    assert resp_del.json()["deleted"] is True

    # Step 6: Verify reset to unconfigured
    resp_recheck = client.get(f"/api/v1/cameras/{camera_id}/calibration")
    assert resp_recheck.status_code == 200
    assert resp_recheck.json()["calibration_status"] == "UNCONFIGURED"


def test_14_real_image_inference_regression(client):
    """14. Verifies real image analysis does not regress and maintains valid scene graph."""
    img_path = Path("data/samples/industrial_cctv.jpg")
    if not img_path.exists():
        pytest.skip("Sample image industrial_cctv.jpg not found")

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

    # Retrieve verified scene graph for this specific job
    scene_resp = client.get(f"/api/v1/scene/current?job_id={job_id}")
    assert scene_resp.status_code == 200
    sg = scene_resp.json()
    assert "entities" in sg
    assert "relationships" in sg
    assert "time_to_hazard" in sg
    # Single image must not fabricate metric motion vectors
    assert sg["time_to_hazard"]["available"] is False
    assert sg["time_to_hazard"]["time_to_hazard_seconds"] is None
    assert "single-frame" in sg["time_to_hazard"]["status_text"].lower()
