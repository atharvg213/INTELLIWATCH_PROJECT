"""Stage 6C acceptance regressions for camera-scoped calibration and metric provenance."""

from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.analysis import MediaType
from backend.schemas.assessment import FrameAssessment
from backend.schemas.calibration import (
    CalibrationSaveRequest,
    CalibrationStatus,
    CameraCalibrationConfig,
    Point2D,
)
from backend.schemas.detection import BoundingBox
from backend.schemas.scene_graph import (
    FrameScene,
    SceneNode,
    SceneNodeType,
    SceneRelation,
    SceneRelationType,
    enrich_scene_graph,
)
from backend.schemas.tracking import FrameTracks, TrackPoint, TrackState, TrackedObject
from backend.services.calibration_service import CalibrationService, GroundPlaneCalibrator
import backend.services.calibration_service as calibration_service_module
import backend.services.job_manager as job_manager_module
import intelligence.scene_graph.builder as scene_builder_module
from intelligence.scene_graph.builder import SceneGraphBuilder


IMAGE_POINTS = [
    Point2D(x=100.0, y=100.0),
    Point2D(x=300.0, y=100.0),
    Point2D(x=300.0, y=300.0),
    Point2D(x=100.0, y=300.0),
]


@pytest.fixture
def service(tmp_path, monkeypatch):
    calibration_service = CalibrationService(tmp_path / "calibrations.json")
    monkeypatch.setattr(calibration_service_module, "get_calibration_service", lambda: calibration_service)
    return calibration_service


@pytest.fixture
def client():
    return TestClient(app)


def _save_calibration(calibration_service, camera_id="cam_A"):
    request = CalibrationSaveRequest(
        points=IMAGE_POINTS,
        real_world_width_m=10.0,
        real_world_depth_m=10.0,
    )
    calibration_service.save_calibration(camera_id, request)


def _person(contact_point=(100.0, 100.0), attributes=None):
    return SceneNode(
        node_id="person_1",
        node_type=SceneNodeType.PERSON,
        class_name="person",
        track_id=1,
        bbox=BoundingBox(x1=90, y1=50, x2=110, y2=100),
        centroid=(100.0, 75.0),
        contact_point=contact_point,
        attributes=attributes or {},
    )


def _vehicle(contact_point=(300.0, 100.0), attributes=None):
    return SceneNode(
        node_id="forklift_2",
        node_type=SceneNodeType.VEHICLE,
        class_name="forklift",
        track_id=2,
        bbox=BoundingBox(x1=290, y1=50, x2=310, y2=100),
        centroid=(300.0, 75.0),
        contact_point=contact_point,
        attributes=attributes or {},
    )


def _approaching_relation(distance_px=200.0, closing_rate_px_s=20.0):
    return SceneRelation(
        relation_id="rel_person_forklift",
        source_node_id="person_1",
        target_node_id="forklift_2",
        relation_type=SceneRelationType.APPROACHING,
        evidence={"distance_px": distance_px, "closing_rate_px_s": closing_rate_px_s},
    )


def test_projection_uses_original_pixel_coordinates_and_rejects_extrapolation():
    config = CameraCalibrationConfig(
        camera_id="cam_projection",
        points=IMAGE_POINTS,
        real_world_width_m=10.0,
        real_world_depth_m=10.0,
    )
    calibrator = GroundPlaneCalibrator(config)

    projected = calibrator.image_to_ground(200.0, 200.0)
    assert projected == (5.0, 5.0)
    assert all(value == pytest.approx(value) and abs(value) < 500.0 for value in projected)
    assert calibrator.image_to_ground(301.0, 200.0) is None


def test_only_matching_source_camera_can_apply_calibration(service):
    _save_calibration(service, "cam_A")
    matching = FrameScene(
        timestamp=1.0,
        camera_id="cam_A",
        source_camera_id="cam_A",
        nodes=[_person()],
    )
    matching_result = enrich_scene_graph(matching, source_mode="VIDEO_STREAM")
    assert matching_result.calibration_status == "CALIBRATED"
    assert matching_result.entities[0]["position"]["ground_plane"] == {"x_m": 0.0, "y_m": 0.0}

    different = FrameScene(
        timestamp=1.0,
        camera_id="cam_A",
        source_camera_id="cam_B",
        nodes=[_person()],
    )
    different_result = enrich_scene_graph(different, source_mode="VIDEO_STREAM", camera_id="cam_A")
    assert different_result.source_camera_id == "cam_B"
    assert different_result.calibration_status == "UNCONFIGURED"
    assert different_result.spatial_basis == "IMAGE_SPACE"
    assert different_result.entities[0]["position"]["ground_plane"] is None


def test_explicit_unassigned_identity_cannot_fall_back_to_legacy_camera_id(service):
    _save_calibration(service, "cam_A")
    scene = FrameScene(
        timestamp=1.0,
        camera_id="cam_A",
        source_camera_id=None,
        nodes=[_person()],
    )

    enriched = enrich_scene_graph(scene, source_mode="VIDEO_STREAM", camera_id="cam_A")

    assert enriched.source_camera_id is None
    assert enriched.calibration_status == "UNAVAILABLE"
    assert enriched.spatial_basis == "IMAGE_SPACE"
    assert enriched.entities[0]["position"]["ground_plane"] is None


def test_invalid_and_deleted_calibration_never_produce_ground_metrics(service):
    invalid_points = [
        Point2D(x=100.0, y=100.0),
        Point2D(x=150.0, y=100.0),
        Point2D(x=200.0, y=100.0),
        Point2D(x=250.0, y=100.0),
    ]
    invalid_config = CameraCalibrationConfig(
        camera_id="cam_A",
        points=invalid_points,
        real_world_width_m=10.0,
        real_world_depth_m=10.0,
        calibration_status=CalibrationStatus.INVALID,
    )
    service._calibrations["cam_A"] = invalid_config
    service._calibrators["cam_A"] = GroundPlaneCalibrator(invalid_config)
    scene = FrameScene(timestamp=1.0, camera_id="cam_A", source_camera_id="cam_A", nodes=[_person()])

    invalid = enrich_scene_graph(scene.model_copy(deep=True), source_mode="VIDEO_STREAM")
    assert invalid.calibration_status == "INVALID"
    assert invalid.spatial_basis == "IMAGE_SPACE"
    assert invalid.entities[0]["position"]["ground_plane"] is None
    assert invalid.entities[0]["state"]["speed_m_per_s"] is None

    assert service.delete_calibration("cam_A") is True
    deleted = enrich_scene_graph(scene.model_copy(deep=True), source_mode="VIDEO_STREAM")
    assert deleted.calibration_status == "UNCONFIGURED"
    assert deleted.spatial_basis == "IMAGE_SPACE"
    assert deleted.entities[0]["position"]["ground_plane"] is None


def test_distance_speed_and_tth_keep_their_own_coordinate_provenance(service):
    _save_calibration(service, "cam_A")
    person = _person(attributes={
        "speed_px_per_s": 20.0,
        "velocity_x_px_per_s": 20.0,
        "velocity_y_px_per_s": 0.0,
        "motion_observed": True,
    })
    forklift = _vehicle(attributes={
        "speed_px_per_s": 0.0,
        "velocity_x_px_per_s": 0.0,
        "velocity_y_px_per_s": 0.0,
        "motion_observed": True,
    })
    scene = FrameScene(
        timestamp=1.0,
        camera_id="cam_A",
        source_camera_id="cam_A",
        nodes=[person, forklift],
        relationships=[_approaching_relation()],
        is_video=True,
    )

    enriched = enrich_scene_graph(scene, source_mode="VIDEO_STREAM")
    person_entity = next(entity for entity in enriched.entities if entity["id"] == "person_1")
    relation = enriched.relationships[0]

    assert person_entity["position"]["ground_plane"] == {"x_m": 0.0, "y_m": 0.0}
    assert person_entity["position"]["spatial_basis"] == "GROUND_PLANE_APPROXIMATION"
    assert person_entity["state"]["speed_px_per_s"] == 20.0
    assert person_entity["state"]["speed_m_per_s"] == 1.0
    assert "approximate ground-plane" in person_entity["state"]["ground_speed_status"].lower()

    assert relation.distance_px == 200.0
    assert relation.distance_m == 10.0
    assert relation.spatial_basis == "GROUND_PLANE_APPROXIMATION"
    assert relation.closing_rate_px_s == 20.0
    assert relation.closing_rate_m_s == 1.0

    assert enriched.time_to_hazard["available"] is True
    assert enriched.time_to_hazard["time_to_hazard_seconds"] == 7.0
    assert enriched.time_to_hazard["spatial_basis"] == "IMAGE_SPACE"
    assert "image-space" in enriched.time_to_hazard["status_text"].lower()
    assert "distance_m" not in enriched.time_to_hazard
    assert "closing_rate_m_s" not in enriched.time_to_hazard
    assert "image-space" in relation.tth_status.lower()
    assert all(prediction["spatial_basis"] == "IMAGE_SPACE" for prediction in enriched.temporal_predictions)
    assert all("distance_m" not in prediction and "closing_rate_m_s" not in prediction for prediction in enriched.temporal_predictions)


def test_scalar_pixel_speed_without_direction_stays_unavailable_in_meters(service):
    _save_calibration(service, "cam_A")
    scene = FrameScene(
        timestamp=1.0,
        camera_id="cam_A",
        source_camera_id="cam_A",
        nodes=[
            _person(attributes={"speed_px_per_s": 25.0, "motion_observed": True}),
            _vehicle(),
        ],
        relationships=[_approaching_relation()],
        is_video=True,
    )

    enriched = enrich_scene_graph(scene, source_mode="VIDEO_STREAM")
    state = enriched.entities[0]["state"]
    assert state["speed_px_per_s"] == 25.0
    assert state["speed_m_per_s"] is None
    assert "direction is missing" in state["ground_speed_status"].lower()
    assert enriched.relationships[0].distance_m == 10.0
    assert enriched.relationships[0].closing_rate_m_s is None
    assert enriched.time_to_hazard["spatial_basis"] == "IMAGE_SPACE"


def test_scene_builder_preserves_and_projects_contact_point_and_directed_motion(service, monkeypatch):
    _save_calibration(service, "cam_A")
    monkeypatch.setattr(
        scene_builder_module,
        "get_scene_store",
        lambda: SimpleNamespace(set_current_scene=lambda _scene: None),
    )
    track = TrackedObject(
        track_id=1,
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=BoundingBox(x1=140.0, y1=120.0, x2=160.0, y2=220.0),
        state=TrackState.ACTIVE,
        frame_index=2,
        timestamp=2.0,
        centroid_x=150.0,
        centroid_y=170.0,
        trajectory=[
            TrackPoint(frame_id=1, timestamp=1.0, cx=140.0, cy=170.0),
            TrackPoint(frame_id=2, timestamp=2.0, cx=150.0, cy=170.0),
        ],
        velocity_x=10.0,
        velocity_y=0.0,
        speed_pixels_per_second=10.0,
    )
    frame_tracks = FrameTracks(frame_id=2, timestamp=2.0, active_tracks=[track])

    scene = SceneGraphBuilder(camera_id="cam_A").build(frame_tracks)
    node = scene.nodes[0]
    assert node.contact_point == (150.0, 220.0)
    assert node.attributes["motion_observed"] is True
    assert node.attributes["velocity_x_px_per_s"] == 10.0
    assert node.attributes["velocity_y_px_per_s"] == 0.0

    enriched = enrich_scene_graph(scene, source_mode="VIDEO_STREAM")
    entity = enriched.entities[0]
    assert entity["position"]["contact_point"] == [150.0, 220.0]
    assert entity["position"]["ground_plane"] == {"x_m": 2.5, "y_m": 6.0}


def test_scene_graph_query_camera_id_cannot_override_job_source(client, service, monkeypatch):
    _save_calibration(service, "cam_A")
    job_id = "job_vid_source_cam_B"
    scene = FrameScene(
        timestamp=1.0,
        camera_id="cam_B",
        source_camera_id="cam_B",
        nodes=[_person()],
        is_video=True,
    )
    job = {
        "job_id": job_id,
        "camera_id": "cam_B",
        "source_camera_id": "cam_B",
        "media_type": MediaType.VIDEO,
        "assessment": FrameAssessment(camera_id="cam_B", scene=scene),
    }
    manager = SimpleNamespace(get_job=lambda requested_id: job if requested_id == job_id else None)
    monkeypatch.setattr(job_manager_module, "get_job_manager", lambda: manager)

    response = client.get(f"/api/v1/scene/graph?job_id={job_id}&camera_id=cam_A")

    assert response.status_code == 200
    data = response.json()
    assert data["source_camera_id"] == "cam_B"
    assert data["camera_id"] == "cam_B"
    assert data["calibration_status"] == "UNCONFIGURED"
    assert data["spatial_basis"] == "IMAGE_SPACE"
    assert data["entities"][0]["position"]["ground_plane"] is None
