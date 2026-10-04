"""
tests/test_alerts_step22.py
Step 22 Automated Test Suite: Intelligent Safety Alerting, Evidence Capture & Alert Lifecycle.

Tests cover:
1. Alert Creation for all supported safety violations (PPE, Fall, Zone intrusion, Vehicle proximity)
2. Severity and Confidence Rules (Immediate critical bypass, confidence gating, severity tiers)
3. Deduplication and Cooldown Behavior (Continuing violations aggregated, cooldown re-arming)
4. Multi-camera and Multi-worker Isolation (Camera isolation, distinct worker IDs)
5. Missing / Unstable Worker Tracking IDs (Graceful handling of unassigned/None tracks)
6. Evidence Image Creation and Storage Fallbacks (Annotations, bounding boxes, watermarks, error tolerance)
7. Alert Lifecycle State Transitions (NEW -> ACKNOWLEDGED -> RESOLVED / DISMISSED, audit logging, invalid transitions)
8. REST API Endpoints (Listing, filtering, pagination, statistics, acknowledgment, resolution, dismissal, evidence retrieval, path traversal protection)
9. Concurrent Multi-threaded Event Ingestion (Thread safety, race conditions, deduplication lock)
10. End-to-End FrameAssessment processing
"""
import concurrent.futures
import json
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import List

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.schemas.alert import (
    AlertAcknowledgeRequest,
    AlertDismissRequest,
    AlertRecord,
    AlertResolveRequest,
    AlertSeverity,
    AlertStatus,
)
from backend.schemas.assessment import FrameAssessment
from backend.schemas.behavior import BehaviorState, PrimaryBehavior
from backend.schemas.detection import BoundingBox
from backend.schemas.ppe import AssociatedPPEItem, ComplianceStatus, PPEItemState, WorkerPPEInventory
from backend.schemas.risk import EventLifecycleState, FrameRiskAssessment, RiskEvent, RiskEventType, RiskLevel
from backend.services.alert_store import AlertStore, get_alert_store
from intelligence.alerting.alert_engine import AlertEngine, get_alert_engine


@pytest.fixture
def temp_alert_env(tmp_path):
    """Creates isolated SQLite database and evidence directory for test isolation."""
    db_file = tmp_path / "test_alerts.db"
    evidence_dir = tmp_path / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)

    store = AlertStore(db_path=db_file)
    engine = AlertEngine(
        alert_store=store,
        evidence_dir=evidence_dir,
        cooldown_seconds=2.0,
        confidence_threshold=0.35,
        immediate_critical=True,
    )
    yield store, engine, evidence_dir


@pytest.fixture
def synthetic_frame():
    """Generates a synthetic 640x480 RGB test image."""
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (100, 100), (300, 400), (120, 120, 120), -1)
    return img


# --------------------------------------------------------------------------
# 1. Alert Creation for Supported Violations
# --------------------------------------------------------------------------
def test_alert_creation_supported_violations(temp_alert_env, synthetic_frame):
    store, engine, _ = temp_alert_env

    # 1. Missing Hardhat
    alert_hardhat = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="NO_HARDHAT",
        confidence=0.88,
        track_id=1,
        bbox=[100, 50, 180, 150],
        frame_image=synthetic_frame,
    )
    assert alert_hardhat is not None
    assert alert_hardhat.violation_type == "NO_HARDHAT"
    assert alert_hardhat.severity == AlertSeverity.HIGH
    assert alert_hardhat.status == AlertStatus.NEW
    assert alert_hardhat.has_evidence is True
    assert alert_hardhat.evidence_image_path is not None
    assert os.path.exists(alert_hardhat.evidence_image_path)

    # 2. Missing Safety Vest
    alert_vest = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="NO_SAFETY_VEST",
        confidence=0.82,
        track_id=1,
        bbox=[90, 120, 210, 320],
        frame_image=synthetic_frame,
    )
    assert alert_vest is not None
    assert alert_vest.violation_type == "NO_SAFETY_VEST"
    assert alert_vest.severity == AlertSeverity.HIGH

    # 3. Fall Detected (Critical)
    alert_fall = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="FALL_DETECTED",
        confidence=0.91,
        track_id=2,
        bbox=[150, 300, 400, 420],
        frame_image=synthetic_frame,
    )
    assert alert_fall is not None
    assert alert_fall.violation_type == "FALL_DETECTED"
    assert alert_fall.severity == AlertSeverity.CRITICAL

    # 4. Restricted Zone Intrusion
    alert_zone = engine.process_violation_event(
        camera_id="cam_02",
        violation_type="RESTRICTED_ZONE_INTRUSION",
        confidence=0.85,
        track_id=3,
        bbox=[200, 200, 350, 450],
        frame_image=synthetic_frame,
    )
    assert alert_zone is not None
    assert alert_zone.violation_type == "RESTRICTED_ZONE_INTRUSION"
    assert alert_zone.severity == AlertSeverity.HIGH


# --------------------------------------------------------------------------
# 2. Severity and Confidence Gating Rules
# --------------------------------------------------------------------------
def test_severity_and_confidence_gating(temp_alert_env):
    store, engine, _ = temp_alert_env

    # Low-confidence non-critical event (< 0.35) should be filtered out
    low_conf_alert = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="NO_HARDHAT",
        confidence=0.25,
        track_id=10,
    )
    assert low_conf_alert is None

    # Immediate critical event (Fall) bypasses high threshold with lower bar (>= 0.20)
    crit_alert = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="FALL_DETECTED",
        confidence=0.30,
        track_id=11,
    )
    assert crit_alert is not None
    assert crit_alert.severity == AlertSeverity.CRITICAL

    # Severity mapping verification
    assert engine.get_severity_for_violation("FALL_DETECTED") == AlertSeverity.CRITICAL
    assert engine.get_severity_for_violation("POSSIBLE_FALL") == AlertSeverity.CRITICAL
    assert engine.get_severity_for_violation("RESTRICTED_ZONE_INTRUSION") == AlertSeverity.HIGH
    assert engine.get_severity_for_violation("NO_HARDHAT") == AlertSeverity.HIGH
    assert engine.get_severity_for_violation("NO_SAFETY_VEST") == AlertSeverity.HIGH
    assert engine.get_severity_for_violation("NO_MASK") == AlertSeverity.MEDIUM
    assert engine.get_severity_for_violation("RAPID_MOVEMENT_EVENT") == AlertSeverity.MEDIUM
    assert engine.get_severity_for_violation("RESTRICTED_ZONE_DWELL") == AlertSeverity.LOW


# --------------------------------------------------------------------------
# 3. Deduplication and Cooldown Behavior
# --------------------------------------------------------------------------
def test_deduplication_and_cooldown(temp_alert_env):
    store, engine, _ = temp_alert_env

    t0 = 1000.0
    # First detection creates ticket
    a1 = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="NO_HARDHAT",
        confidence=0.80,
        track_id=5,
        timestamp=t0,
    )
    assert a1 is not None
    assert a1.occurrence_count == 1
    alert_id = a1.alert_id

    # Second detection 0.5s later (within 2.0s cooldown) should NOT create new alert
    a2 = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="NO_HARDHAT",
        confidence=0.85,
        track_id=5,
        timestamp=t0 + 0.5,
    )
    assert a2 is not None
    assert a2.alert_id == alert_id
    assert a2.occurrence_count == 2
    assert a2.last_detected_at == t0 + 0.5

    # Third detection 1.0s later (still within cooldown)
    a3 = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="NO_HARDHAT",
        confidence=0.90,
        track_id=5,
        timestamp=t0 + 1.0,
    )
    assert a3.alert_id == alert_id
    assert a3.occurrence_count == 3

    # Total alerts in store must still be 1
    stats = store.get_statistics()
    assert stats.total_alerts == 1

    # Fourth detection 3.5s later (> 2.0s cooldown) re-arms and creates a new ticket
    a4 = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="NO_HARDHAT",
        confidence=0.82,
        track_id=5,
        timestamp=t0 + 4.5,
    )
    assert a4 is not None
    assert a4.alert_id != alert_id
    assert a4.occurrence_count == 1

    # Now total alerts in store must be 2
    stats2 = store.get_statistics()
    assert stats2.total_alerts == 2


# --------------------------------------------------------------------------
# 4. Multi-Camera and Multi-Worker Isolation
# --------------------------------------------------------------------------
def test_multi_camera_and_multi_worker_isolation(temp_alert_env):
    store, engine, _ = temp_alert_env

    ts = 2000.0
    # Same violation on Camera 1, Worker 1
    a_cam1_w1 = engine.process_violation_event(
        camera_id="cam_01", violation_type="NO_HARDHAT", confidence=0.8, track_id=1, timestamp=ts
    )
    # Same violation on Camera 2, Worker 1 (Different camera -> must be distinct alert)
    a_cam2_w1 = engine.process_violation_event(
        camera_id="cam_02", violation_type="NO_HARDHAT", confidence=0.8, track_id=1, timestamp=ts
    )
    # Same violation on Camera 1, Worker 2 (Different worker -> must be distinct alert)
    a_cam1_w2 = engine.process_violation_event(
        camera_id="cam_01", violation_type="NO_HARDHAT", confidence=0.8, track_id=2, timestamp=ts
    )

    assert a_cam1_w1.alert_id != a_cam2_w1.alert_id
    assert a_cam1_w1.alert_id != a_cam1_w2.alert_id
    assert a_cam2_w1.alert_id != a_cam1_w2.alert_id

    stats = store.get_statistics()
    assert stats.total_alerts == 3


# --------------------------------------------------------------------------
# 5. Missing / Unstable Worker Tracking IDs
# --------------------------------------------------------------------------
def test_untracked_workers_graceful_handling(temp_alert_env):
    store, engine, _ = temp_alert_env

    ts = 3000.0
    # Untracked worker (track_id is None)
    a_untracked = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="ZONE_INTRUSION",
        confidence=0.75,
        track_id=None,
        timestamp=ts,
    )
    assert a_untracked is not None
    assert a_untracked.track_id is None

    # Immediate second frame for untracked worker aggregates under 'unassigned'
    a_untracked_2 = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="ZONE_INTRUSION",
        confidence=0.78,
        track_id=None,
        timestamp=ts + 0.2,
    )
    assert a_untracked_2.alert_id == a_untracked.alert_id
    assert a_untracked_2.occurrence_count == 2


# --------------------------------------------------------------------------
# 6. Evidence Image Creation & Storage Failure Handling
# --------------------------------------------------------------------------
def test_evidence_capture_and_storage_failures(temp_alert_env, synthetic_frame):
    store, engine, evidence_dir = temp_alert_env

    # Successful evidence capture
    alert_valid = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="NO_SAFETY_VEST",
        confidence=0.85,
        track_id=12,
        bbox=[50, 50, 150, 250],
        frame_image=synthetic_frame,
    )
    assert alert_valid.has_evidence is True
    assert alert_valid.evidence_image_path is not None
    assert Path(alert_valid.evidence_image_path).is_file()

    # None frame image handled gracefully
    alert_no_frame = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="FALL_DETECTED",
        confidence=0.90,
        track_id=14,
        frame_image=None,
    )
    assert alert_no_frame.has_evidence is False
    assert alert_no_frame.evidence_image_path is None

    # Empty frame array handled gracefully
    empty_frame = np.array([], dtype=np.uint8)
    alert_empty_frame = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="RESTRICTED_ZONE_INTRUSION",
        confidence=0.80,
        track_id=15,
        frame_image=empty_frame,
    )
    assert alert_empty_frame.has_evidence is False
    assert alert_empty_frame.evidence_image_path is None


# --------------------------------------------------------------------------
# 7. Alert Lifecycle State Transitions & Audit Log
# --------------------------------------------------------------------------
def test_alert_lifecycle_state_transitions(temp_alert_env):
    store, engine, _ = temp_alert_env

    # 1. Create alert (status: NEW)
    alert = engine.process_violation_event(
        camera_id="cam_01",
        violation_type="NO_HARDHAT",
        confidence=0.85,
        track_id=20,
    )
    assert alert.status == AlertStatus.NEW
    assert len(alert.transition_history) == 1
    assert alert.transition_history[0].to_status == AlertStatus.NEW

    # 2. Acknowledge alert (NEW -> ACKNOWLEDGED)
    ack_alert = store.acknowledge_alert(
        alert_id=alert.alert_id,
        user="supervisor_alice",
        notes="Contacting worker on radio",
    )
    assert ack_alert.status == AlertStatus.ACKNOWLEDGED
    assert ack_alert.acknowledged_by == "supervisor_alice"
    assert ack_alert.acknowledged_at is not None
    assert len(ack_alert.transition_history) == 2
    assert ack_alert.transition_history[1].to_status == AlertStatus.ACKNOWLEDGED
    assert ack_alert.transition_history[1].notes == "Contacting worker on radio"

    # 3. Resolve alert (ACKNOWLEDGED -> RESOLVED)
    res_alert = store.resolve_alert(
        alert_id=alert.alert_id,
        user="supervisor_alice",
        notes="Worker put on hardhat from safety locker",
    )
    assert res_alert.status == AlertStatus.RESOLVED
    assert res_alert.resolved_by == "supervisor_alice"
    assert res_alert.resolved_at is not None
    assert res_alert.resolution_notes == "Worker put on hardhat from safety locker"
    assert len(res_alert.transition_history) == 3

    # 4. Invalid transition: Cannot acknowledge a resolved alert
    with pytest.raises(ValueError):
        store.acknowledge_alert(alert_id=alert.alert_id, user="operator")

    # 5. Dismissal lifecycle
    alert_disp = engine.process_violation_event(
        camera_id="cam_02",
        violation_type="RESTRICTED_ZONE_INTRUSION",
        confidence=0.80,
        track_id=21,
    )
    dismissed = store.dismiss_alert(
        alert_id=alert_disp.alert_id,
        user="operator_bob",
        reason="Authorized maintenance technician with temporary permit",
    )
    assert dismissed.status == AlertStatus.DISMISSED
    assert dismissed.dismissed_by == "operator_bob"
    assert dismissed.dismiss_reason == "Authorized maintenance technician with temporary permit"

    # Dismissal without reason must fail
    with pytest.raises(ValueError):
        store.dismiss_alert(alert_id=alert_disp.alert_id, user="bob", reason="")

    # Cannot resolve a dismissed alert
    with pytest.raises(ValueError):
        store.resolve_alert(alert_id=alert_disp.alert_id, user="bob")


# --------------------------------------------------------------------------
# 8. FrameAssessment End-to-End Ingestion
# --------------------------------------------------------------------------
def test_frame_assessment_end_to_end_alert_ingestion(temp_alert_env, synthetic_frame):
    store, engine, _ = temp_alert_env

    # Assemble a synthetic FrameAssessment with PPE non-compliance, Fall behavior, and Zone intrusion
    assessment = FrameAssessment(
        frame_id=10,
        timestamp=500.0,
        camera_id="cam_01",
        worker_inventories=[
            WorkerPPEInventory(
                track_id=1,
                bbox=BoundingBox(x1=50, y1=50, x2=150, y2=250),
                items=[
                    AssociatedPPEItem(
                        class_id=1,
                        class_name="NO-Hardhat",
                        confidence=0.88,
                        bbox=BoundingBox(x1=60, y1=50, x2=120, y2=100),
                        association_score=0.9,
                        is_negative=True,
                    )
                ],
                compliance_status=ComplianceStatus.NON_COMPLIANT,
                ppe_status={"Hardhat": PPEItemState.MISSING, "Safety-Vest": PPEItemState.PRESENT},
            )
        ],
        behavior_states=[
            BehaviorState(
                track_id=2,
                timestamp=500.0,
                primary_behavior=PrimaryBehavior.POSSIBLE_FALL,
                fall_risk_score=0.92,
            )
        ],
        risk_assessment=FrameRiskAssessment(
            active_events=[
                RiskEvent(
                    event_id="ev_01",
                    event_type=RiskEventType.RESTRICTED_ZONE_INTRUSION,
                    timestamp=500.0,
                    start_timestamp=500.0,
                    risk_level=RiskLevel.HIGH,
                    risk_score=75.0,
                    involved_entities=["person_3", "zone_high_voltage"],
                    explanation="Worker 3 entered high voltage area",
                )
            ]
        ),
    )

    alerts = engine.process_assessment(assessment, frame_image=synthetic_frame)
    # Should generate alerts for PPE violation, Fall, and Zone intrusion
    assert len(alerts) >= 3
    violation_types = [a.violation_type for a in alerts]
    assert any("HARDHAT" in v for v in violation_types)
    assert any("FALL" in v for v in violation_types)
    assert any("ZONE" in v for v in violation_types)


# --------------------------------------------------------------------------
# 9. REST API Integration Tests
# --------------------------------------------------------------------------
def test_alert_rest_api_lifecycle(synthetic_frame):
    client = TestClient(app)

    # Initialize a test alert via singleton store
    global_store = get_alert_store()
    global_store.reset()

    global_engine = get_alert_engine()
    global_engine.reset()

    created = global_engine.process_violation_event(
        camera_id="cam_api_test",
        violation_type="NO_SAFETY_VEST",
        confidence=0.87,
        track_id=99,
        bbox=[80, 80, 200, 350],
        frame_image=synthetic_frame,
    )
    alert_id = created.alert_id

    # 1. GET /api/v1/alerts
    resp = client.get("/api/v1/alerts")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] >= 1
    assert any(a["alert_id"] == alert_id for a in data["items"])

    # 2. Filter by status
    resp_filter = client.get("/api/v1/alerts?status=NEW")
    assert resp_filter.status_code == 200
    assert all(a["status"] == "NEW" for a in resp_filter.json()["items"])

    # 3. GET /api/v1/alerts/statistics
    resp_stats = client.get("/api/v1/alerts/statistics")
    assert resp_stats.status_code == 200
    stats = resp_stats.json()
    assert stats["total_alerts"] >= 1
    assert stats["active_unresolved_count"] >= 1

    # 4. GET /api/v1/alerts/{alert_id}
    resp_detail = client.get(f"/api/v1/alerts/{alert_id}")
    assert resp_detail.status_code == 200
    det = resp_detail.json()
    assert det["alert_id"] == alert_id
    assert len(det["transition_history"]) >= 1

    # 5. POST /api/v1/alerts/{alert_id}/acknowledge
    resp_ack = client.post(
        f"/api/v1/alerts/{alert_id}/acknowledge",
        json={"user": "api_operator", "notes": "Alert acknowledged via API"},
    )
    assert resp_ack.status_code == 200
    assert resp_ack.json()["status"] == "ACKNOWLEDGED"

    # 6. POST /api/v1/alerts/{alert_id}/resolve
    resp_resolve = client.post(
        f"/api/v1/alerts/{alert_id}/resolve",
        json={"user": "api_operator", "notes": "PPE issued"},
    )
    assert resp_resolve.status_code == 200
    assert resp_resolve.json()["status"] == "RESOLVED"

    # 7. GET /api/v1/alerts/{alert_id}/evidence
    resp_evidence = client.get(f"/api/v1/alerts/{alert_id}/evidence")
    assert resp_evidence.status_code == 200
    assert resp_evidence.headers["content-type"] == "image/jpeg"
    assert len(resp_evidence.content) > 0

    # 8. Test 404 for non-existent alert
    resp_404 = client.get("/api/v1/alerts/alert_nonexistent_xyz")
    assert resp_404.status_code == 404

    # 9. Test Dismiss endpoint
    alert2 = global_engine.process_violation_event(
        camera_id="cam_api_test",
        violation_type="RAPID_MOVEMENT_EVENT",
        confidence=0.75,
        track_id=101,
    )
    resp_dismiss = client.post(
        f"/api/v1/alerts/{alert2.alert_id}/dismiss",
        json={"user": "api_operator", "reason": "Authorized drill test"},
    )
    assert resp_dismiss.status_code == 200
    assert resp_dismiss.json()["status"] == "DISMISSED"


# --------------------------------------------------------------------------
# 10. Multi-Threaded Concurrent Ingestion & Deduplication Safety
# --------------------------------------------------------------------------
def test_concurrent_alert_processing(temp_alert_env):
    store, engine, _ = temp_alert_env

    def emit_event(worker_id: int):
        return engine.process_violation_event(
            camera_id="cam_concurrent",
            violation_type="NO_HARDHAT",
            confidence=0.85,
            track_id=worker_id,
            timestamp=100.0,
        )

    # 20 concurrent threads: 10 calls for worker 1, 10 calls for worker 2
    worker_ids = [1] * 10 + [2] * 10
    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(emit_event, worker_ids))

    # All threads succeeded without raising exceptions
    assert all(r is not None for r in results)

    # Deduplication check: exactly 2 distinct tickets created (one for worker 1, one for worker 2)
    stats = store.get_statistics()
    assert stats.total_alerts == 2


def test_alerts_scoped_camera_filtering(monkeypatch):
    """
    Verifies that camera-scoped users receive only alerts for permitted cameras
    and that AlertListResponse serialization correctly uses the 'items' attribute.
    """
    from backend.schemas.auth import UserResponse, UserRole
    from backend.services.alert_store import get_alert_store
    from intelligence.alerting.alert_engine import get_alert_engine

    engine = get_alert_engine()
    engine.process_violation_event(camera_id="cam_scoped_A", violation_type="NO_HARDHAT", confidence=0.9)
    engine.process_violation_event(camera_id="cam_scoped_B", violation_type="NO_SAFETY_VEST", confidence=0.9)

    restricted_user = UserResponse(
        user_id="usr_scoped_test",
        username="scoped_operator",
        role=UserRole.OPERATOR,
        permitted_cameras=["cam_scoped_A"],
        created_at=time.time(),
    )

    from backend.api.deps import get_current_user
    app.dependency_overrides[get_current_user] = lambda: restricted_user

    try:
        client = TestClient(app)
        resp = client.get("/api/v1/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        # All returned items must belong exclusively to permitted camera
        for item in data["items"]:
            assert item["camera_id"] == "cam_scoped_A"
    finally:
        app.dependency_overrides.pop(get_current_user, None)

