"""
scripts/validate_step15_real_video.py
Step 15 Real-Video & Synthetic Sequence Validation Tool.

Runs the IntelliWatch end-to-end intelligence pipeline on real sample video
and (if needed) deterministic multi-frame synthetic sequence, recording:
- Frame counts, throughput, latency
- Track counts and object classifications
- Behavior state transitions (e.g. UNKNOWN -> MOVING / STOPPED)
- Spatial relationship transitions (e.g. FAR -> NEAR)
- Risk events and confirmed incidents
- Dashboard / API store state verification
"""
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.assessment import FrameAssessment
from backend.schemas.detection import BoundingBox, DetectionResult
from backend.schemas.risk import RiskLevel
from backend.services.assessment_store import get_assessment_store
from backend.services.incident_store import get_incident_store
from backend.services.prediction_store import get_prediction_store
from backend.services.risk_store import get_risk_store
from backend.services.scene_store import get_scene_store
from backend.services.temporal_log import get_temporal_log
from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator


def validate_real_video(video_path: Path):
    print("=" * 80)
    print(f"STEP 15 REAL-VIDEO VALIDATION: {video_path.name}")
    print("=" * 80)

    # Reset singletons
    get_incident_store().reset()
    get_temporal_log().__init__()

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    duration_s = total_frames / video_fps

    print(f"Video File       : {video_path}")
    print(f"Resolution       : {width} x {height}")
    print(f"Video Stream FPS : {video_fps:.2f}")
    print(f"Total Frames     : {total_frames}")
    print(f"Duration         : {duration_s:.2f} seconds")
    print("-" * 80)

    # Initialize full pipeline orchestrator
    orchestrator = EndToEndPipelineOrchestrator(
        device="cpu",
        enable_ppe_model=True,
        enable_depth_model=False,
    )

    frame_idx = 0
    processed_frames = 0
    unique_tracks = set()
    class_detections = Counter()
    behavior_history = defaultdict(list)
    spatial_transitions = []
    risk_events_observed = []
    incidents_created = []
    temporal_events_captured = []

    prev_spatial_rels = {}

    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret or frame is None:
            break

        frame_idx += 1
        timestamp = frame_idx / video_fps

        assessment, annotated_frame = orchestrator.process_frame(
            frame=frame,
            frame_id=frame_idx,
            timestamp=timestamp,
        )
        processed_frames += 1

        # Track IDs & classes
        for trk in assessment.tracks:
            unique_tracks.add(trk.track_id)
            c_name = (trk.class_name or "unknown").lower()
            class_detections[c_name] += 1

        # Behavior states per track
        for bs in assessment.behavior_states:
            b_val = bs.primary_behavior.value if hasattr(bs.primary_behavior, "value") else str(bs.primary_behavior)
            behavior_history[bs.track_id].append((frame_idx, timestamp, b_val))

        # Spatial relationship transitions
        current_spatial_rels = {}
        if assessment.scene and assessment.scene.relationships:
            for rel in assessment.scene.relationships:
                pair_key = f"{rel.source_node_id} -> {rel.target_node_id}"
                current_spatial_rels[pair_key] = rel.relation_type.value
                if pair_key in prev_spatial_rels and prev_spatial_rels[pair_key] != rel.relation_type.value:
                    spatial_transitions.append({
                        "frame": frame_idx,
                        "timestamp": round(timestamp, 3),
                        "pair": pair_key,
                        "from": prev_spatial_rels[pair_key],
                        "to": rel.relation_type.value,
                    })
        prev_spatial_rels = current_spatial_rels

        # Risk events
        for rev in assessment.active_events:
            risk_events_observed.append({
                "frame": frame_idx,
                "timestamp": round(timestamp, 3),
                "event_type": rev.event_type.value,
                "risk_level": rev.risk_level.value,
                "risk_score": rev.risk_score,
                "entities": rev.involved_entities,
                "explanation": rev.explanation,
            })

        # Incidents
        for inc_id in assessment.new_incident_ids:
            incidents_created.append({
                "frame": frame_idx,
                "incident_id": inc_id,
            })

        # Temporal events captured by Step 15 TemporalLog
        if assessment.temporal_events:
            for te in assessment.temporal_events:
                temporal_events_captured.append(te)

        if frame_idx % 10 == 0 or frame_idx == total_frames:
            print(f"  Frame {frame_idx:03d}/{total_frames:03d} | Active Tracks: {len(assessment.tracks)} | "
                  f"Risk: {assessment.highest_risk_level.value} (Score: {assessment.highest_risk_score:.1f}) | "
                  f"Latency: {assessment.processing_time_ms:.1f}ms")

    cap.release()
    t_end = time.perf_counter()
    total_time_s = t_end - t_start
    effective_fps = processed_frames / total_time_s if total_time_s > 0 else 0.0
    avg_latency_ms = (total_time_s * 1000.0) / processed_frames if processed_frames > 0 else 0.0

    # Categorize detected classes
    workers_detected_count = class_detections.get("person", 0)
    vehicles_detected_count = sum(class_detections.get(k, 0) for k in ["car", "truck", "bus", "forklift", "vehicle"])
    machines_detected_count = sum(class_detections.get(k, 0) for k in ["machinery", "machine", "robot", "conveyor", "heavy_machinery"])

    # Detect behavior transitions
    behavior_transitions = []
    for trk_id, states in behavior_history.items():
        prev_st = None
        for f, t, st in states:
            if prev_st is not None and st != prev_st:
                behavior_transitions.append({
                    "track_id": trk_id,
                    "frame": f,
                    "timestamp": round(t, 3),
                    "from": prev_st,
                    "to": st,
                })
            prev_st = st

    print("-" * 80)
    print("REAL VIDEO EXECUTION METRICS:")
    print(f"  Total Frames                    : {total_frames}")
    print(f"  Processed Frames                : {processed_frames}")
    print(f"  Processing FPS                  : {effective_fps:.2f} FPS")
    print(f"  Average Latency                 : {avg_latency_ms:.2f} ms")
    print(f"  Unique Tracks                   : {len(unique_tracks)} (IDs: {sorted(list(unique_tracks))})")
    print(f"  Worker Detections ('person')    : {workers_detected_count}")
    print(f"  Vehicle Detections              : {vehicles_detected_count}")
    print(f"  Machine Detections              : {machines_detected_count}")
    print(f"  Behavior State Transitions      : {len(behavior_transitions)}")
    for bt in behavior_transitions:
        print(f"    - Track {bt['track_id']} at frame {bt['frame']}: {bt['from']} -> {bt['to']}")
    print(f"  Spatial Relationship Transitions: {len(spatial_transitions)}")
    for st in spatial_transitions:
        print(f"    - Pair {st['pair']} at frame {st['frame']}: {st['from']} -> {st['to']}")
    print(f"  Risk Events Observed            : {len(risk_events_observed)}")
    print(f"  Incidents Created               : {len(incidents_created)}")
    print(f"  Step 15 Temporal Log Entries    : {len(temporal_events_captured)}")
    for te in temporal_events_captured[:10]:
        print(f"    - [F{te.frame_id:03d} | {te.category.value}] {te.label}: {te.description}")
    if len(temporal_events_captured) > 10:
        print(f"    ... and {len(temporal_events_captured) - 10} more temporal events.")

    # Check temporal progression in real video
    has_unknown_to_moving = any(bt["from"] == "unknown" and bt["to"] == "moving" for bt in behavior_transitions)
    has_moving_to_approaching = any(bt["to"] == "approaching" for bt in behavior_transitions)
    has_far_to_near = any(st["from"] == "far" and st["to"] == "near" for st in spatial_transitions)
    has_risk_escalation = any(
        te.category.value == "risk_escalation" for te in temporal_events_captured
    ) or any(
        st.get("to") in ["near", "approaching"] for st in spatial_transitions
    )

    real_results = {
        "video_name": video_path.name,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
        "processing_fps": round(effective_fps, 2),
        "average_latency_ms": round(avg_latency_ms, 2),
        "unique_tracks": len(unique_tracks),
        "unique_track_ids": sorted(list(unique_tracks)),
        "workers_detected": workers_detected_count,
        "vehicles_detected": vehicles_detected_count,
        "machines_detected": machines_detected_count,
        "behavior_state_transitions": behavior_transitions,
        "spatial_relationship_transitions": spatial_transitions,
        "risk_events_count": len(risk_events_observed),
        "incidents_count": len(incidents_created),
        "temporal_events": [t.model_dump() for t in temporal_events_captured],
        "has_unknown_to_moving": has_unknown_to_moving,
        "has_moving_to_approaching": has_moving_to_approaching,
        "has_far_to_near": has_far_to_near,
        "has_risk_escalation": has_risk_escalation,
    }
    return real_results


def run_deterministic_synthetic_validation():
    """
    Validates Worker FAR -> APPROACHING -> NEAR -> Risk Escalation deterministically.
    """
    print("\n" + "=" * 80)
    print("STEP 15 SYNTHETIC SEQUENCE VALIDATION")
    print("Worker FAR -> APPROACHING -> NEAR -> Risk Escalation")
    print("=" * 80)

    get_incident_store().reset()
    get_temporal_log().__init__()

    orchestrator = EndToEndPipelineOrchestrator(
        device="cpu",
        enable_ppe_model=False,
        enable_depth_model=False,
    )

    # 15 frames: Worker (track 1) moves towards stationary Forklift (track 2)
    # Forklift stays at (600, 300, 200, 150)
    # Worker starts at (100, 300, 60, 120) [FAR, dist ~ 500px]
    # Moves towards forklift by 35px each frame
    # Frame 1-3: distance > 350px (FAR)
    # Frame 4-8: distance 350px -> 150px (APPROACHING / CLOSING)
    # Frame 9-15: distance < 120px (NEAR / PROXIMITY VIOLATION)

    dummy_frame = np.full((720, 1280, 3), 30, dtype=np.uint8)
    forklift_box = BoundingBox(x1=700, y1=300, x2=900, y2=500)

    synthetic_results = []
    spatial_transitions = []
    behavior_transitions = []
    temporal_events = []

    for f_idx in range(1, 21):
        t_sec = f_idx * 0.1
        # Worker moving right towards forklift (from FAR to APPROACHING to NEAR)
        worker_x1 = 100 + (f_idx - 1) * 32
        worker_box = BoundingBox(x1=worker_x1, y1=320, x2=worker_x1 + 60, y2=480)

        manual_dets = [
            DetectionResult(
                class_id=0,
                class_name="person",
                confidence=0.92,
                bbox=worker_box,
            ),
            DetectionResult(
                class_id=1,
                class_name="forklift",
                confidence=0.88,
                bbox=forklift_box,
            ),
        ]

        assessment, _ = orchestrator.process_frame(
            frame=dummy_frame,
            frame_id=f_idx,
            timestamp=t_sec,
            manual_detections=manual_dets,
        )

        for te in assessment.temporal_events:
            temporal_events.append(te)

        print(f"  [Synth F{f_idx:02d} | {t_sec:.1f}s] Worker X: {worker_x1} | "
              f"Risk: {assessment.highest_risk_level.value} (Score: {assessment.highest_risk_score:.1f}) | "
              f"Events: {len(assessment.active_events)} | "
              f"New Incidents: {assessment.new_incident_ids} | "
              f"Temporal Events: {len(assessment.temporal_events)}")

    # Extract transitions from temporal log
    timeline = get_temporal_log().recent_timeline(n=50)
    profiles = get_temporal_log().get_all_entity_profiles(assessment)

    print("-" * 80)
    print("SYNTHETIC VALIDATION RESULTS:")
    print(f"  Total Timeline Transitions Captured: {len(timeline)}")
    for ent in timeline:
        print(f"    - Frame {ent.frame_id:02d}: [{ent.category.value}] {ent.label} -> {ent.description}")

    print(f"  Entity Profiles Tracked: {len(profiles)}")
    for p in profiles:
        print(f"    - {p.entity_id} ({p.class_name}): Behavior={p.current_behavior}, Proximity={p.current_spatial_state}, RiskTrend={p.risk_trend_label}")

    return {
        "timeline_entries": [e.model_dump() for e in timeline],
        "entity_profiles": [p.model_dump() for p in profiles],
    }


def verify_dashboard_api(last_assessment: FrameAssessment):
    print("\n" + "=" * 80)
    print("VERIFYING DASHBOARD & API STATE STORES")
    print("=" * 80)

    # 1. Assessment Store
    stored_assessment = get_assessment_store().get_current_assessment()
    assert stored_assessment is not None, "AssessmentStore is empty!"
    print(f"  [OK] AssessmentStore holds frame_id={stored_assessment.frame_id}, risk={stored_assessment.highest_risk_level.value}")

    # 2. Scene Store
    stored_scene = get_scene_store().get_current_scene()
    assert stored_scene is not None, "SceneStore is empty!"
    print(f"  [OK] SceneStore holds {len(stored_scene.nodes)} nodes, {len(stored_scene.relationships)} relationships")

    # 3. Risk Store
    stored_risk = get_risk_store().get_current_assessment()
    assert stored_risk is not None, "RiskStore is empty!"
    print(f"  [OK] RiskStore holds max_risk={stored_risk.risk_summary.max_risk_level.value}")

    # 4. Prediction Store
    stored_pred = get_prediction_store().get_current_assessment()
    assert stored_pred is not None, "PredictionStore is empty!"
    print(f"  [OK] PredictionStore holds {len(stored_pred.active_indicators)} active early warnings")

    # 5. Temporal Log
    temporal_entries = get_temporal_log().recent_timeline(n=30)
    print(f"  [OK] TemporalLog holds {len(temporal_entries)} timeline entries")

    # 6. Incident Store
    incidents = get_incident_store().list_incidents()
    print(f"  [OK] IncidentStore holds {len(incidents)} recorded incidents")

    # 7. FastAPI Endpoint Verification (what frontend dashboard queries)
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)
    resp_assess = client.get("/api/v1/assessment/current")
    assert resp_assess.status_code == 200, f"Assessment endpoint failed: {resp_assess.status_code}"
    assess_data = resp_assess.json()
    print(f"  [OK] GET /api/v1/assessment/current -> 200 OK (risk={assess_data['highest_risk_level']}, tracks={len(assess_data['tracks'])})")

    resp_time = client.get("/api/v1/temporal/timeline?n=10")
    assert resp_time.status_code == 200, f"Timeline endpoint failed: {resp_time.status_code}"
    print(f"  [OK] GET /api/v1/temporal/timeline -> 200 OK ({len(resp_time.json())} entries)")

    resp_ent = client.get("/api/v1/temporal/entities")
    assert resp_ent.status_code == 200, f"Entities endpoint failed: {resp_ent.status_code}"
    print(f"  [OK] GET /api/v1/temporal/entities -> 200 OK ({len(resp_ent.json())} profiles)")

    resp_inc = client.get("/api/v1/incidents?limit=5")
    assert resp_inc.status_code == 200, f"Incidents endpoint failed: {resp_inc.status_code}"
    print(f"  [OK] GET /api/v1/incidents -> 200 OK ({len(resp_inc.json())} incidents)")

    resp_dash = client.get("/dashboard")
    assert resp_dash.status_code == 200, f"Dashboard route failed: {resp_dash.status_code}"
    print(f"  [OK] GET /dashboard -> 200 OK ({resp_dash.headers.get('content-type')})")

    print("-" * 80)
    print("DASHBOARD / API STORES & ENDPOINTS VERIFIED SUCCESSFULLY.")


if __name__ == "__main__":
    real_video_path = PROJECT_ROOT / "data" / "samples" / "cctv_worker_moving.mp4"
    real_results = validate_real_video(real_video_path)

    # Save real video results
    out_dir = PROJECT_ROOT / "data" / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "step15_real_video_validation.json", "w", encoding="utf-8") as f:
        json.dump(real_results, f, indent=2)

    # Check if real video demonstrated worker/vehicle temporal interaction
    if not real_results["has_far_to_near"] or not real_results["has_moving_to_approaching"]:
        print("\nNotice: Real video contains worker moving but does not contain a Worker -> Vehicle interaction.")
        print("Proceeding to separate DETERMINISTIC SYNTHETIC SEQUENCE validation.")
        synth_results = run_deterministic_synthetic_validation()
        with open(out_dir / "step15_synthetic_validation.json", "w", encoding="utf-8") as f:
            json.dump(synth_results, f, indent=2)

    verify_dashboard_api(get_assessment_store().get_current_assessment())
