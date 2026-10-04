"""
Test Pydantic schema validation and instantiation.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from backend.schemas.tracking import TrackPoint, TrackedObject, FrameTracks
from backend.schemas.events import SeverityLevel, EventType, IndustrialEvent
from backend.schemas.health import HealthResponse


def test_health_schema():
    health = HealthResponse()
    assert health.status == "ok"
    assert health.project == "IntelliWatch"


def test_detection_schemas():
    bbox = BoundingBox(x1=50.0, y1=100.0, x2=150.0, y2=300.0)
    assert bbox.width == 100.0
    assert bbox.height == 200.0
    assert bbox.center == (100.0, 200.0)

    det = DetectionResult(class_id=0, class_name="worker", confidence=0.92, bbox=bbox)
    assert det.class_name == "worker"
    assert det.confidence == 0.92

    frame = FrameDetections(frame_id=10, timestamp=0.33, detections=[det])
    assert frame.frame_id == 10
    assert len(frame.detections) == 1


def test_tracking_schemas():
    bbox = BoundingBox(x1=10.0, y1=10.0, x2=50.0, y2=50.0)
    pt = TrackPoint(frame_id=1, x=30.0, y=30.0, timestamp=0.033)
    tracked = TrackedObject(
        track_id=101,
        class_id=1,
        class_name="forklift",
        confidence=0.88,
        bbox=bbox,
        trajectory=[pt]
    )
    assert tracked.track_id == 101
    assert len(tracked.trajectory) == 1

    tracks = FrameTracks(frame_id=1, timestamp=0.033, active_tracks=[tracked])
    assert len(tracks.active_tracks) == 1


def test_event_schema():
    event = IndustrialEvent(
        event_id="ev-12345",
        event_type=EventType.PPE_VIOLATION,
        timestamp=1700000000.0,
        camera_id="cam_factory_01",
        tracked_object_ids=[101],
        severity=SeverityLevel.HIGH,
        explanation="Worker detected without required safety helmet near active crane.",
        metadata={"missing_item": "helmet", "zone": "crane_loading"}
    )
    assert event.severity == SeverityLevel.HIGH
    assert event.event_type == EventType.PPE_VIOLATION
    assert event.metadata["missing_item"] == "helmet"


def test_ppe_schemas():
    from backend.schemas.ppe import (
        AssociatedPPEItem,
        ComplianceStatus,
        FramePPEAssociation,
        PPEItemState,
        WorkerPPEInventory,
    )

    bbox = BoundingBox(x1=100.0, y1=100.0, x2=200.0, y2=400.0)
    item = AssociatedPPEItem(
        class_id=1,
        class_name="Hardhat",
        confidence=0.95,
        bbox=BoundingBox(x1=120.0, y1=85.0, x2=180.0, y2=130.0),
        association_score=0.82,
        body_region="head",
    )
    assert item.class_name == "Hardhat"
    assert item.body_region == "head"

    worker_inv = WorkerPPEInventory(
        track_id=17,
        timestamp=1.5,
        bbox=bbox,
        items=[item],
        ppe_status={"Hardhat": PPEItemState.PRESENT, "Gloves": PPEItemState.MISSING},
        compliance_status=ComplianceStatus.NON_COMPLIANT,
        missing_ppe=["Gloves"],
        present_ppe=["Hardhat"],
    )
    assert worker_inv.track_id == 17
    assert worker_inv.compliance_status == ComplianceStatus.NON_COMPLIANT
    assert worker_inv.missing_ppe == ["Gloves"]

    frame_assoc = FramePPEAssociation(
        frame_id=1,
        timestamp=1.5,
        worker_inventories=[worker_inv],
        unassociated_ppe=[],
    )
    assert len(frame_assoc.worker_inventories) == 1


def test_zone_schemas():
    from backend.schemas.zones import (
        FrameZoneOccupancy,
        RestrictedZone,
        ZoneMembership,
        ZoneMembershipStatus,
        ZoneType,
    )

    zone = RestrictedZone(
        zone_id="zone_01",
        name="High Voltage Area",
        zone_type=ZoneType.RESTRICTED,
        polygon=[[10.0, 10.0], [50.0, 10.0], [50.0, 50.0], [10.0, 50.0]],
        enabled=True,
        max_dwell_seconds=8.0,
    )
    assert zone.zone_id == "zone_01"
    assert zone.zone_type == ZoneType.RESTRICTED
    assert len(zone.polygon) == 4
    assert zone.max_dwell_seconds == 8.0

    mem = ZoneMembership(
        track_id=17,
        zone_id="zone_01",
        zone_name="High Voltage Area",
        is_inside=True,
        status=ZoneMembershipStatus.INSIDE,
        contact_point=(25.0, 30.0),
        dwell_seconds=4.5,
        entry_timestamp=10.0,
    )
    assert mem.track_id == 17
    assert mem.is_inside is True
    assert mem.dwell_seconds == 4.5

    occupancy = FrameZoneOccupancy(
        frame_id=10,
        timestamp=14.5,
        memberships=[mem],
        occupancy_by_zone={"zone_01": [17]},
    )
    assert occupancy.frame_id == 10
    assert occupancy.occupancy_by_zone["zone_01"] == [17]


def test_depth_schemas():
    from backend.schemas.depth import (
        DepthResult,
        DepthStatistics,
        DepthType,
        ObjectDepth,
    )

    stats = DepthStatistics(
        min_depth=0.05,
        max_depth=0.95,
        mean_depth=0.55,
        median_depth=0.58,
        percentile_25=0.30,
        percentile_75=0.75,
        depth_unit="relative",
    )
    assert stats.median_depth == 0.58

    bbox = BoundingBox(x1=50.0, y1=50.0, x2=150.0, y2=200.0)
    obj_depth = ObjectDepth(
        track_id=5,
        class_name="person",
        bbox=bbox,
        depth_stats=stats,
        contact_point=(100.0, 200.0),
        contact_depth=0.82,
        is_metric=False,
    )
    assert obj_depth.is_metric is False
    assert obj_depth.contact_depth == 0.82

    result = DepthResult(
        frame_id=100,
        timestamp=3.33,
        width=1280,
        height=720,
        depth_type=DepthType.RELATIVE,
        is_metric=False,
        min_depth=0.0,
        max_depth=1.0,
        mean_depth=0.45,
        median_depth=0.47,
        object_depths=[obj_depth],
    )
    assert result.width == 1280
    assert result.height == 720
    assert len(result.object_depths) == 1


def test_risk_schemas():
    from backend.schemas.risk import (
        RiskLevel,
        RiskFactorType,
        RiskEventType,
        EventLifecycleState,
        RiskFactor,
        RiskEvent,
        RiskSummary,
        FrameRiskAssessment,
    )

    factor = RiskFactor(
        factor_id="factor_01",
        factor_type=RiskFactorType.PPE_NON_COMPLIANCE,
        severity_contribution=25.0,
        involved_entity_ids=["person_17"],
        source_evidence={"missing": ["Hardhat"]},
        timestamp=1.5,
        explanation="Missing hardhat",
    )
    assert factor.severity_contribution == 25.0
    assert factor.factor_type == RiskFactorType.PPE_NON_COMPLIANCE

    event = RiskEvent(
        event_id="evt_01",
        event_type=RiskEventType.PPE_VIOLATION,
        lifecycle_state=EventLifecycleState.CONFIRMED,
        timestamp=1.5,
        start_timestamp=1.0,
        involved_entities=["person_17"],
        risk_level=RiskLevel.MEDIUM,
        risk_score=35.0,
        risk_factors=[factor],
        explanation="PPE violation confirmed",
    )
    assert event.event_type == RiskEventType.PPE_VIOLATION
    assert event.risk_level == RiskLevel.MEDIUM

    summary = RiskSummary(medium=1, total_active_events=1, max_risk_level=RiskLevel.MEDIUM, max_risk_score=35.0)
    assessment = FrameRiskAssessment(
        frame_id=10,
        timestamp=1.5,
        active_events=[event],
        risk_factors=[factor],
        risk_summary=summary,
    )
    assert assessment.frame_id == 10
    assert len(assessment.active_events) == 1
    assert assessment.risk_summary.max_risk_level == RiskLevel.MEDIUM


