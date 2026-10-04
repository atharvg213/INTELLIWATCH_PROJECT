import sys
from pathlib import Path

# Ensure project root is on sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def verify():
    print("=" * 60)
    print("IntelliWatch Setup & Architecture Verification")
    print("=" * 60)

    # 1. Verify Configuration
    try:
        from configs.settings import get_settings
        settings = get_settings()
        print(f"[OK] Configuration loaded: {settings.PROJECT_NAME} v{settings.VERSION}")
    except Exception as e:
        print(f"[FAIL] Configuration failed: {e}")
        return False

    # 2. Verify Logging
    try:
        from configs.logging_config import setup_logging
        logger = setup_logging()
        print("[OK] Logging configured successfully.")
    except Exception as e:
        print(f"[FAIL] Logging setup failed: {e}")
        return False

    # 3. Verify Schemas
    try:
        from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
        from backend.schemas.tracking import TrackPoint, TrackedObject, FrameTracks
        from backend.schemas.events import SeverityLevel, EventType, IndustrialEvent

        bbox = BoundingBox(x1=10, y1=20, x2=100, y2=200)
        det = DetectionResult(class_id=0, class_name="worker", confidence=0.95, bbox=bbox)
        frame_det = FrameDetections(frame_id=1, timestamp=0.033, detections=[det])

        track_pt = TrackPoint(frame_id=1, x=55.0, y=110.0, timestamp=0.033)
        tracked = TrackedObject(track_id=1, class_id=0, class_name="worker", confidence=0.95, bbox=bbox, trajectory=[track_pt])
        frame_trk = FrameTracks(frame_id=1, timestamp=0.033, active_tracks=[tracked])

        event = IndustrialEvent(
            event_id="evt-001",
            event_type=EventType.ZONE_INTRUSION,
            timestamp=0.033,
            tracked_object_ids=[1],
            severity=SeverityLevel.HIGH,
            explanation="Worker entered restricted machinery zone without authorization."
        )
        print("[OK] All Pydantic data schemas validated and instantiated.")
    except Exception as e:
        print(f"[FAIL] Schema instantiation failed: {e}")
        return False

    # 4. Verify Vision & Intelligence Base Interfaces
    try:
        from vision.detection.base import BaseDetector
        from vision.detection.yolo_detector import YOLODetector
        from vision.tracking.base import BaseTracker
        from vision.tracking.bytetrack_tracker import ByteTrackTracker
        from vision.depth.base import BaseDepthEstimator
        from vision.preprocessing.frame_processor import FramePreprocessor
        from intelligence.zones.zone_monitor import BaseZoneMonitor
        from intelligence.behavior.behavior_analyzer import BaseBehaviorAnalyzer
        from intelligence.scene_graph.scene_graph import BaseSceneGraph
        from intelligence.events.event_detector import BaseEventDetector
        from intelligence.risk.risk_engine import BaseRiskEngine
        from intelligence.prediction.trajectory_predictor import BasePredictor

        detector = YOLODetector()
        tracker = ByteTrackTracker()
        preproc = FramePreprocessor()
        print("[OK] Vision and Intelligence interfaces verified.")
    except Exception as e:
        print(f"[FAIL] Interface imports failed: {e}")
        return False

    # 5. Verify FastAPI App & Health Route
    try:
        from fastapi.testclient import TestClient
        from backend.main import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        data = response.json()
        assert data.get("status") == "ok", f"Expected status 'ok', got {data.get('status')}"
        assert data.get("project") == "IntelliWatch", f"Expected project 'IntelliWatch', got {data.get('project')}"
        print(f"[OK] GET /health endpoint verified: {data}")
    except Exception as e:
        print(f"[FAIL] Health endpoint check failed: {e}")
        return False

    print("=" * 60)
    print("ALL STEP 1 FOUNDATION CHECKS PASSED SUCCESSFULLY!")
    print("=" * 60)
    return True

if __name__ == "__main__":
    success = verify()
    sys.exit(0 if success else 1)
