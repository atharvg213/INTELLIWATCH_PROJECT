import sys
from pathlib import Path
import cv2

# Set cwd and path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from intelligence.pipeline.orchestrator import EndToEndPipelineOrchestrator
from configs.settings import get_settings

settings = get_settings()
print(f"PPE_MODEL_PATH: {settings.PPE_MODEL_PATH}")
print(f"PPE_ENABLED: {settings.PPE_ENABLED}")
print(f"PPE_REQUIRED_CLASSES: {settings.PPE_REQUIRED_CLASSES}")

orchestrator = EndToEndPipelineOrchestrator(
    device="cpu",
    camera_id="cam_test",
    enable_ppe_model=True,
    enable_depth_model=False,
)

ppe_detector = orchestrator._get_ppe_detector()
print(f"PPE detector loaded: {ppe_detector is not None}")
if ppe_detector:
    print(f"PPE classes: {ppe_detector.class_names}")

# Test on a real image
image_files = list((project_root / "data" / "input" / "uploads").glob("*.jpg")) + list((project_root / "data" / "input" / "uploads").glob("*.png"))
if image_files:
    test_img_path = image_files[0]
    print(f"\nTesting image: {test_img_path.name}")
    img = cv2.imread(str(test_img_path))
    if img is not None:
        assessment, ann = orchestrator.process_frame(
            frame=img,
            frame_id=0,
            timestamp=0.0,
            is_single_image=True,
        )
        print(f"Detections: {len(assessment.detections)}")
        print(f"Active tracks: {len(assessment.tracks)}")
        print(f"PPE detections: {len(assessment.ppe_detections)}")
        for p in assessment.ppe_detections:
            print(f"  PPE item: {p.class_name} ({p.confidence:.2f})")
        print(f"Worker inventories: {len(assessment.worker_inventories)}")
        for inv in assessment.worker_inventories:
            print(f"  Worker #{inv.track_id}: status={inv.compliance_status}, present={inv.present_ppe}, missing={inv.missing_ppe}, unknown={inv.unknown_ppe}")
            print(f"    Explanation: {inv.explanation}")
        print(f"Incidents generated: {len(assessment.new_incident_ids)}")

print("\n--- Diagnostic complete ---")
