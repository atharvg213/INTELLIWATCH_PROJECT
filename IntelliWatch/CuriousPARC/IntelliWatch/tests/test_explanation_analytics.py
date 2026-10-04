"""
tests/test_explanation_analytics.py
Step 17 - Explainability, Evidence & Safety Analytics Unit & Integration Tests.

Validates:
- Deterministic Five-W generation (WHO, WHAT, WHERE, WHEN, WHY)
- Risk score breakdown (base factors, escalation, final score, risk tier)
- Evidence handling (snapshots vs honest metadata-only reporting)
- Chronological incident timelines
- Risk trend time-series data
- Event filtering in IncidentStore
- Safety analytics summary calculations
- FastAPI endpoints for explanations, timelines, packages, and analytics
"""
import shutil
import tempfile
from pathlib import Path
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.api.routes import router
from backend.schemas.evidence import IncidentRecord
from backend.schemas.prediction import EarlyWarningIndicator, EarlyWarningIndicatorType, IndicatorLifecycleState, IndicatorSeverity
from backend.schemas.risk import (
    EventLifecycleState,
    RiskEventType,
    RiskFactor,
    RiskFactorType,
    RiskLevel,
)
from backend.services.explanation_service import ExplanationService, get_explanation_service
from backend.services.incident_store import IncidentStore
from backend.services.temporal_log import TemporalLogService
from fastapi import FastAPI


@pytest.fixture
def clean_incident_setup():
    """Provides isolated IncidentStore and ExplanationService instances with temporary storage."""
    temp_dir = Path(tempfile.mkdtemp())
    inc_store = IncidentStore(storage_dir=temp_dir)
    temp_log = TemporalLogService()

    service = ExplanationService()
    service.incident_store = inc_store
    service.temporal_log = temp_log

    yield inc_store, temp_log, service

    shutil.rmtree(temp_dir, ignore_errors=True)


def create_sample_incident(
    incident_id: str = "inc_test_01",
    event_type: RiskEventType = RiskEventType.RESTRICTED_ZONE_INTRUSION,
    risk_level: RiskLevel = RiskLevel.CRITICAL,
    risk_score: float = 95.0,
    timestamp: float = 14.5,
    frame_id: int = 145,
    track_ids: list = None,
    zone: str = "High Voltage Area",
    with_snapshot: bool = True,
    temp_dir: Path = None,
) -> IncidentRecord:
    """Helper to construct a realistic, verifiable IncidentRecord."""
    factors = [
        RiskFactor(
            factor_id="rf_01",
            factor_type=RiskFactorType.RESTRICTED_ZONE_INTRUSION,
            severity_contribution=40.0,
            involved_entity_ids=["person_17"],
            source_evidence={"zone_id": "zone_hv_01"},
            timestamp=timestamp,
            explanation="Worker entered High Voltage Area without authorization.",
        ),
        RiskFactor(
            factor_id="rf_02",
            factor_type=RiskFactorType.PPE_NON_COMPLIANCE,
            severity_contribution=25.0,
            involved_entity_ids=["person_17"],
            source_evidence={"missing": ["helmet"]},
            timestamp=timestamp,
            explanation="Mandatory hard hat is missing.",
        ),
        RiskFactor(
            factor_id="rf_03",
            factor_type=RiskFactorType.RAPID_MOVEMENT,
            severity_contribution=15.0,
            involved_entity_ids=["person_17"],
            source_evidence={"speed_px_per_s": 210.0},
            timestamp=timestamp,
            explanation="Worker moving rapidly (210 px/s) inside restricted perimeter.",
        ),
    ]

    scene_rels = [
        {"source_id": "worker_17", "relation_type": "INSIDE", "target_id": "zone_hv_01"},
        {"source_id": "worker_17", "relation_type": "NEAR", "target_id": "machine_03"},
    ]

    early_warnings = [
        EarlyWarningIndicator(
            indicator_id="ew_01",
            indicator_type=EarlyWarningIndicatorType.TRAJECTORY_TOWARD_RESTRICTED_ZONE,
            severity=IndicatorSeverity.HIGH,
            lifecycle_state=IndicatorLifecycleState.CONFIRMED,
            timestamp=timestamp,
            first_observed_timestamp=timestamp - 1.0,
            track_ids=[17],
            entity_ids=["person_17"],
            target_zone_id="zone_hv_01",
            explanation="Projected trajectory directed toward High Voltage Area",
        )
    ]

    snap_path = None
    if with_snapshot and temp_dir:
        import cv2
        dummy_img = np.zeros((480, 640, 3), dtype=np.uint8)
        snap_file = temp_dir / f"{incident_id}.jpg"
        cv2.imwrite(str(snap_file), dummy_img)
        snap_path = str(snap_file)

    return IncidentRecord(
        incident_id=incident_id,
        timestamp=timestamp,
        frame_id=frame_id,
        camera_id="cam_01",
        event_type=event_type,
        risk_level=risk_level,
        risk_score=risk_score,
        lifecycle_state=EventLifecycleState.CONFIRMED,
        involved_track_ids=track_ids or [17],
        involved_entity_ids=["person_17"],
        location_or_zone=zone,
        risk_factors=factors,
        explanation="Worker #17 entered High Voltage Area without helmet moving rapidly.",
        scene_relationships=scene_rels,
        early_warnings=early_warnings,
        has_snapshot=with_snapshot and snap_path is not None,
        snapshot_path=snap_path,
        metadata={"end_timestamp": timestamp + 4.0},
    )


class TestFiveWAndExplanation:
    """Tests deterministic Five-W structure without LLMs."""

    def test_01_five_w_content(self, clean_incident_setup):
        inc_store, temp_log, service = clean_incident_setup
        incident = create_sample_incident(temp_dir=inc_store.storage_dir)
        inc_store.record_incident(incident)

        five_w = service.build_five_w(incident)

        # WHO
        assert "Track #17" in five_w.who
        assert "hard hat is missing" in five_w.who
        assert "moving rapidly" in five_w.who

        # WHAT
        assert "Restricted-zone intrusion" in five_w.what
        assert "CRITICAL Risk" in five_w.what

        # WHERE
        assert "High Voltage Area" in five_w.where
        assert "cam_01" in five_w.where

        # WHEN
        assert "14.50s" in five_w.when
        assert "Frame 145" in five_w.when

        # WHY (deterministic bullet points)
        assert len(five_w.why) >= 3
        assert any("High Voltage Area" in b for b in five_w.why)
        assert any("hard hat is missing" in b for b in five_w.why)
        assert any("moving rapidly" in b for b in five_w.why)

    def test_02_five_w_unassigned_actor_fallback(self, clean_incident_setup):
        inc_store, temp_log, service = clean_incident_setup
        inc = IncidentRecord(
            incident_id="inc_empty_01",
            timestamp=5.0,
            frame_id=50,
            camera_id="cam_02",
            event_type=RiskEventType.COMPOUND_SAFETY_EVENT,
            risk_level=RiskLevel.MEDIUM,
            risk_score=45.0,
            explanation="General workcell risk threshold exceeded.",
        )
        five_w = service.build_five_w(inc)

        assert "Unassigned actor" in five_w.who
        assert "Compound" in five_w.what
        assert "cam_02" in five_w.where
        assert len(five_w.why) >= 1


class TestRiskScoreBreakdown:
    """Tests transparent risk breakdown matches RiskScorer calculation."""

    def test_03_risk_breakdown_math(self, clean_incident_setup):
        inc_store, temp_log, service = clean_incident_setup
        incident = create_sample_incident(temp_dir=inc_store.storage_dir)
        inc_store.record_incident(incident)

        breakdown = service.build_risk_breakdown(incident)

        # Base score = 40 + 25 + 15 = 80
        assert breakdown.base_score == 80.0
        assert len(breakdown.factor_contributions) == 3

        # Escalation = 95 - 80 = 15
        assert breakdown.escalation == 15.0
        assert breakdown.final_score == 95.0
        assert breakdown.risk_tier == RiskLevel.CRITICAL
        assert "CRITICAL" in breakdown.thresholds


class TestEvidenceAndTimeline:
    """Tests evidence presentation honesty and chronological timeline ordering."""

    def test_04_evidence_snapshot_available(self, clean_incident_setup):
        inc_store, temp_log, service = clean_incident_setup
        incident = create_sample_incident(with_snapshot=True, temp_dir=inc_store.storage_dir)
        inc_store.record_incident(incident)

        expl = service.get_incident_explanation(incident.incident_id)
        assert expl is not None
        assert expl.has_snapshot is True
        assert f"/api/v1/incidents/{incident.incident_id}/evidence" in expl.snapshot_url
        assert "Visual evidence snapshot captured" in expl.evidence_notice

    def test_05_evidence_snapshot_missing_honesty(self, clean_incident_setup):
        inc_store, temp_log, service = clean_incident_setup
        incident = create_sample_incident(with_snapshot=False, temp_dir=inc_store.storage_dir)
        inc_store.record_incident(incident)

        expl = service.get_incident_explanation(incident.incident_id)
        assert expl is not None
        assert expl.has_snapshot is False
        assert expl.snapshot_url is None
        assert "No visual evidence snapshot available" in expl.evidence_notice

    def test_06_chronological_timeline(self, clean_incident_setup):
        inc_store, temp_log, service = clean_incident_setup
        incident = create_sample_incident(temp_dir=inc_store.storage_dir)
        inc_store.record_incident(incident)

        timeline = service.build_incident_timeline(incident)
        assert len(timeline) >= 2

        # Verify strict chronological sorting
        for i in range(len(timeline) - 1):
            assert timeline[i].timestamp <= timeline[i + 1].timestamp


class TestAnalyticsSummaryAndFiltering:
    """Tests descriptive analytics metrics and incident store filtering."""

    def test_07_analytics_summary_metrics(self, clean_incident_setup):
        inc_store, temp_log, service = clean_incident_setup

        # Add 3 distinct incidents
        inc1 = create_sample_incident("inc_01", RiskEventType.RESTRICTED_ZONE_INTRUSION, RiskLevel.CRITICAL, 95.0, 10.0, 100, [1], "Zone A", False, inc_store.storage_dir)
        inc2 = create_sample_incident("inc_02", RiskEventType.PPE_VIOLATION, RiskLevel.MEDIUM, 40.0, 20.0, 200, [2], "Zone B", False, inc_store.storage_dir)
        inc3 = create_sample_incident("inc_03", RiskEventType.RESTRICTED_ZONE_INTRUSION, RiskLevel.HIGH, 75.0, 30.0, 300, [3], "Zone A", False, inc_store.storage_dir)

        inc_store.record_incident(inc1)
        inc_store.record_incident(inc2)
        inc_store.record_incident(inc3)

        analytics = service.compute_analytics()

        assert analytics.total_incidents == 3
        assert analytics.critical_incidents == 1
        assert analytics.high_risk_incidents == 1
        assert analytics.medium_risk_incidents == 1
        assert analytics.low_risk_incidents == 0
        assert analytics.zone_intrusions == 2
        assert analytics.ppe_violations == 1
        assert len(analytics.most_active_zones) >= 1
        assert analytics.most_active_zones[0]["zone_name"] == "Zone A"
        assert analytics.most_active_zones[0]["incident_count"] == 2

    def test_08_incident_filtering(self, clean_incident_setup):
        inc_store, temp_log, service = clean_incident_setup

        inc1 = create_sample_incident("inc_01", RiskEventType.RESTRICTED_ZONE_INTRUSION, RiskLevel.CRITICAL, 95.0, 10.0, 100, [17], "High Voltage", False, inc_store.storage_dir)
        inc2 = create_sample_incident("inc_02", RiskEventType.PPE_VIOLATION, RiskLevel.LOW, 20.0, 20.0, 200, [42], "Assembly Floor", False, inc_store.storage_dir)

        inc_store.record_incident(inc1)
        inc_store.record_incident(inc2)

        # Filter by risk_level
        crit = inc_store.list_incidents(risk_level="CRITICAL")
        assert len(crit) == 1
        assert crit[0].incident_id == "inc_01"

        # Filter by event_type
        ppe = inc_store.list_incidents(event_type="PPE_VIOLATION")
        assert len(ppe) == 1
        assert ppe[0].incident_id == "inc_02"

        # Filter by track_id
        t17 = inc_store.list_incidents(track_id=17)
        assert len(t17) == 1
        assert t17[0].incident_id == "inc_01"

        # Filter by zone
        hv = inc_store.list_incidents(zone="Voltage")
        assert len(hv) == 1
        assert hv[0].incident_id == "inc_01"


class TestFastAPIExplanationEndpoints:
    """Tests FastAPI REST endpoints for explainability, timelines, packages, and analytics."""

    @pytest.fixture(autouse=True)
    def setup_app(self, clean_incident_setup):
        inc_store, temp_log, service = clean_incident_setup
        self.inc_store = inc_store
        self.service = service

        # Patch global singletons for route handler calls
        import backend.services.incident_store as is_module
        import backend.services.explanation_service as es_module
        orig_is = is_module._incident_store_instance
        orig_es = es_module._explanation_service_instance

        is_module._incident_store_instance = inc_store
        es_module._explanation_service_instance = service

        app = FastAPI()
        app.include_router(router)
        self.client = TestClient(app)

        yield

        is_module._incident_store_instance = orig_is
        es_module._explanation_service_instance = orig_es

    def test_09_api_get_explanation_success(self):
        inc = create_sample_incident("api_inc_01", temp_dir=self.inc_store.storage_dir)
        self.inc_store.record_incident(inc)

        res = self.client.get(f"/api/v1/incidents/{inc.incident_id}/explanation")
        assert res.status_code == 200
        data = res.json()
        assert data["incident_id"] == "api_inc_01"
        assert "Track #17" in data["five_w"]["who"]
        assert data["risk_breakdown"]["final_score"] == 95.0
        assert len(data["timeline"]) >= 1

    def test_10_api_get_explanation_not_found(self):
        res = self.client.get("/api/v1/incidents/nonexistent_id/explanation")
        assert res.status_code == 404

    def test_11_api_get_timeline(self):
        inc = create_sample_incident("api_inc_02", temp_dir=self.inc_store.storage_dir)
        self.inc_store.record_incident(inc)

        res = self.client.get(f"/api/v1/incidents/{inc.incident_id}/timeline")
        assert res.status_code == 200
        timeline = res.json()
        assert isinstance(timeline, list)
        assert len(timeline) >= 1

    def test_12_api_get_evidence_package(self):
        inc = create_sample_incident("api_inc_03", temp_dir=self.inc_store.storage_dir)
        self.inc_store.record_incident(inc)

        res = self.client.get(f"/api/v1/incidents/{inc.incident_id}/package")
        assert res.status_code == 200
        pkg = res.json()
        assert pkg["incident_id"] == "api_inc_03"
        assert "format" in pkg
        assert "five_w_explanation" in pkg
        assert "risk_score_breakdown" in pkg
        assert "disclaimer" in pkg
        assert "no generative AI" in pkg["disclaimer"]

    def test_13_api_get_analytics_summary(self):
        inc = create_sample_incident("api_inc_04", temp_dir=self.inc_store.storage_dir)
        self.inc_store.record_incident(inc)

        res = self.client.get("/api/v1/analytics/summary")
        assert res.status_code == 200
        data = res.json()
        assert data["total_incidents"] == 1
        assert data["critical_incidents"] == 1
        assert "RESTRICTED_ZONE_INTRUSION" in data["event_type_distribution"]
