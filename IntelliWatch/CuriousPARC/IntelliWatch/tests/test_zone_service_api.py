"""
tests/test_zone_service_api.py
Step 16 Integration & API Test Suite - Safety Zones Configuration & Risk Integration.

Validates:
- ZoneService CRUD operations and validation rules
- JSON configuration persistence across reloads / restarts
- FastAPI endpoints: GET, POST, PUT, DELETE /api/v1/zones
- ZoneEngine point-in-polygon integration with newly created zones
- Zone entry, exit, dwell, and RiskEngine event generation
"""
import json
import pytest
from pathlib import Path
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.events import EventType
from backend.schemas.risk import RiskEventType, RiskLevel
from backend.schemas.tracking import BoundingBox, FrameTracks, TrackedObject
from backend.schemas.zones import (
    RestrictedZone,
    ZoneCreateRequest,
    ZonePoint,
    ZoneType,
    ZoneUpdateRequest,
)
from backend.services.zone_service import ZoneService, get_zone_service
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from intelligence.risk.engine import RiskEngine
from intelligence.zones.zone_engine import ZoneEngine


@pytest.fixture
def tmp_config_path(tmp_path):
    """Provides an isolated JSON config path initialized with sample zones."""
    cfg = tmp_path / "test_zones.json"
    initial_zones = [
        {
            "zone_id": "test_zone_alpha",
            "name": "Alpha Restricted Area",
            "zone_type": "restricted",
            "polygon": [[100.0, 100.0], [300.0, 100.0], [300.0, 300.0], [100.0, 300.0]],
            "enabled": True,
            "max_dwell_seconds": 10.0,
        }
    ]
    with open(cfg, "w", encoding="utf-8") as f:
        json.dump(initial_zones, f)
    return cfg


@pytest.fixture
def isolated_zone_service(tmp_config_path):
    """Creates a ZoneService instance backed by the temporary config path."""
    return ZoneService(config_path=tmp_config_path)


class TestZoneServiceCRUD:

    def test_01_list_initial_zones(self, isolated_zone_service):
        zones = isolated_zone_service.list_zones()
        assert len(zones) == 1
        assert zones[0].zone_id == "test_zone_alpha"
        assert zones[0].zone_type == ZoneType.RESTRICTED

    def test_02_create_zone_success(self, isolated_zone_service):
        req = ZoneCreateRequest(
            name="Robotic Cell Hazard",
            zone_type=ZoneType.HAZARD,
            polygon=[[400.0, 100.0], [600.0, 100.0], [600.0, 400.0], [400.0, 400.0]],
            max_dwell_seconds=5.0,
        )
        created = isolated_zone_service.create_zone(req)
        assert created.zone_id == "robotic_cell_hazard"
        assert created.name == "Robotic Cell Hazard"
        assert created.zone_type == ZoneType.HAZARD
        assert len(created.polygon) == 4

        # Verify it is present in service and underlying engine
        assert isolated_zone_service.get_zone("robotic_cell_hazard") is not None
        assert "robotic_cell_hazard" in isolated_zone_service.engine.zones

    def test_03_create_zone_validation_empty_name(self, isolated_zone_service):
        with pytest.raises(Exception):
            ZoneCreateRequest(
                name="   ",
                polygon=[[10, 10], [50, 10], [50, 50]],
            )

    def test_04_create_zone_validation_fewer_than_3_points(self, isolated_zone_service):
        with pytest.raises(Exception):
            ZoneCreateRequest(
                name="Degenerate Zone",
                polygon=[[10, 10], [50, 10]],
            )

    def test_05_create_zone_duplicate_points_normalized(self, isolated_zone_service):
        """Consecutive duplicate vertices are cleaned automatically."""
        req = ZoneCreateRequest(
            name="Redundant Vertex Zone",
            polygon=[[10, 10], [10, 10], [50, 10], [50, 50], [10, 50], [10, 10]],
        )
        created = isolated_zone_service.create_zone(req)
        assert len(created.polygon) == 4

    def test_06_update_zone_fields(self, isolated_zone_service):
        upd = ZoneUpdateRequest(
            name="Alpha High-Voltage Restricted",
            max_dwell_seconds=3.0,
            enabled=False,
        )
        res = isolated_zone_service.update_zone("test_zone_alpha", upd)
        assert res is not None
        assert res.name == "Alpha High-Voltage Restricted"
        assert res.max_dwell_seconds == 3.0
        assert res.enabled is False
        assert len(isolated_zone_service.list_zones(enabled_only=True)) == 0

    def test_07_update_zone_polygon(self, isolated_zone_service):
        upd = ZoneUpdateRequest(
            polygon=[[150.0, 150.0], [350.0, 150.0], [350.0, 350.0], [150.0, 350.0]]
        )
        res = isolated_zone_service.update_zone("test_zone_alpha", upd)
        assert res is not None
        assert res.polygon[0] == [150.0, 150.0]

    def test_08_update_nonexistent_zone_returns_none(self, isolated_zone_service):
        res = isolated_zone_service.update_zone("non_existent_id", ZoneUpdateRequest(name="New"))
        assert res is None

    def test_09_delete_zone_success(self, isolated_zone_service):
        assert isolated_zone_service.delete_zone("test_zone_alpha") is True
        assert isolated_zone_service.get_zone("test_zone_alpha") is None
        assert "test_zone_alpha" not in isolated_zone_service.engine.zones

    def test_10_delete_nonexistent_zone_returns_false(self, isolated_zone_service):
        assert isolated_zone_service.delete_zone("missing_zone") is False

    def test_11_persistence_survives_reload(self, tmp_config_path):
        """Zone changes saved to disk survive service recreation (simulating server restart)."""
        service1 = ZoneService(config_path=tmp_config_path)
        service1.create_zone(
            ZoneCreateRequest(
                name="Persistent Perimeter",
                zone_type=ZoneType.SAFETY,
                polygon=[[50, 50], [150, 50], [150, 150], [50, 150]],
            )
        )

        # Create new service pointing to same file
        service2 = ZoneService(config_path=tmp_config_path)
        zones = service2.list_zones()
        assert len(zones) == 2
        p_zone = service2.get_zone("persistent_perimeter")
        assert p_zone is not None
        assert p_zone.zone_type == ZoneType.SAFETY


class TestZoneFastAPIEndpoints:

    @pytest.fixture(autouse=True)
    def setup_client(self):
        self.client = TestClient(app)

    def test_20_get_zones_api(self):
        res = self.client.get("/api/v1/zones")
        assert res.status_code == 200
        data = res.json()
        assert isinstance(data, list)

    def test_21_post_and_get_zone_api(self):
        payload = {
            "name": "API Testing Safe Zone",
            "zone_type": "safety",
            "polygon": [
                {"x": 100, "y": 100},
                {"x": 200, "y": 100},
                {"x": 200, "y": 200},
                {"x": 100, "y": 200},
            ],
            "enabled": True,
            "max_dwell_seconds": 12.0,
        }
        res = self.client.post("/api/v1/zones", json=payload)
        assert res.status_code == 201
        created = res.json()
        zid = created["zone_id"]
        assert created["name"] == "API Testing Safe Zone"
        assert created["zone_type"] == "safety"
        assert len(created["polygon"]) == 4

        # Retrieve single zone
        res_get = self.client.get(f"/api/v1/zones/{zid}")
        assert res_get.status_code == 200
        assert res_get.json()["zone_id"] == zid

        # Clean up
        self.client.delete(f"/api/v1/zones/{zid}")

    def test_22_put_zone_api(self):
        # Create zone
        payload = {
            "name": "Machine Inspection Zone",
            "zone_type": "machine",
            "polygon": [[10, 10], [90, 10], [90, 90], [10, 90]],
        }
        res = self.client.post("/api/v1/zones", json=payload)
        zid = res.json()["zone_id"]

        # Update zone
        upd_payload = {
            "name": "Updated Machine Zone",
            "max_dwell_seconds": 25.0,
        }
        res_put = self.client.put(f"/api/v1/zones/{zid}", json=upd_payload)
        assert res_put.status_code == 200
        assert res_put.json()["name"] == "Updated Machine Zone"
        assert res_put.json()["max_dwell_seconds"] == 25.0

        # Clean up
        self.client.delete(f"/api/v1/zones/{zid}")

    def test_23_delete_zone_api(self):
        payload = {
            "name": "Temporary Delete Me Zone",
            "zone_type": "monitored",
            "polygon": [[10, 10], [90, 10], [90, 90]],
        }
        res = self.client.post("/api/v1/zones", json=payload)
        zid = res.json()["zone_id"]

        del_res = self.client.delete(f"/api/v1/zones/{zid}")
        assert del_res.status_code == 200
        assert del_res.json()["status"] == "deleted"

        # Verify 404 on subsequent get
        res_404 = self.client.get(f"/api/v1/zones/{zid}")
        assert res_404.status_code == 404

    def test_24_invalid_payload_returns_422(self):
        invalid_payload = {
            "name": "",
            "polygon": [[10, 10]],
        }
        res = self.client.post("/api/v1/zones", json=invalid_payload)
        assert res.status_code == 422


class TestZoneEngineAndRiskIntegration:

    def test_30_created_zone_triggers_entry_and_risk(self, isolated_zone_service):
        """
        Validates that a zone created via ZoneService feeds into ZoneEngine and
        generates ZONE_ENTRY and RiskEngine ZONE_INTRUSION_EVENT.
        """
        # Create restricted zone at (200, 200) -> (400, 400)
        zone = isolated_zone_service.create_zone(
            ZoneCreateRequest(
                zone_id="test_intrusion_zone",
                name="Dangerous Restricted Zone",
                zone_type=ZoneType.RESTRICTED,
                polygon=[[200.0, 200.0], [400.0, 200.0], [400.0, 400.0], [200.0, 400.0]],
                enabled=True,
                max_dwell_seconds=5.0,
            )
        )

        engine = isolated_zone_service.engine
        engine.confirmation_frames = 1  # 1 frame confirmation for testing

        # Person foot-point at (300, 300) -> bbox y2 = 300, cx = 300
        worker_track = TrackedObject(
            track_id=1,
            class_id=0,
            class_name="person",
            confidence=0.92,
            bbox=BoundingBox(x1=280, y1=200, x2=320, y2=300),
            frame_index=1,
            timestamp=0.1,
        )
        frame_tracks = FrameTracks(
            frame_id=1,
            timestamp=0.1,
            active_tracks=[worker_track],
            total_tracked_count=1,
        )

        # Process through ZoneEngine
        memberships, zone_events = engine.process_frame(frame_tracks, timestamp=0.1, frame_id=1)
        assert len(memberships) >= 1
        target_mem = next(m for m in memberships if m.zone_id == "test_intrusion_zone")
        assert target_mem.is_inside is True
        assert target_mem.status.value == "inside"

        # Check Zone Event for target zone
        entry_events = [
            e for e in zone_events
            if e.event_type == EventType.ZONE_ENTRY and e.metadata.get("zone_id") == "test_intrusion_zone"
        ]
        assert len(entry_events) == 1
        assert entry_events[0].metadata["zone_id"] == "test_intrusion_zone"

        # Clean up
        isolated_zone_service.delete_zone("test_intrusion_zone")
