"""
PPE Compliance & Spatial Association Verification Tool (Step 5B)

Demonstrates the Scene Understanding / Intelligence Layer:
  1. Multi-Object Tracking (Tracked Persons)
  2. Spatial Person-PPE Association
  3. Worker PPE Inventory (Present, Missing, Unknown)
  4. Deterministic Compliance Evaluation (UNKNOWN vs NON_COMPLIANT)
  5. Lightweight Temporal Confirmation across frames
  6. Structured, Explainable IndustrialEvent generation
  7. Visual Verification Overlay
"""
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from backend.schemas.tracking import FrameTracks, TrackPoint, TrackedObject
from configs.logging_config import setup_logging
from configs.settings import get_settings
from intelligence.events.ppe_compliance import PPEComplianceEngine
from vision.detection.ppe_association import PPEAssociationEngine
from vision.tracking.visualizer import TrackingVisualizer


def run_synthetic_verification(output_image_path: Path):
    """
    Executes a multi-frame synthetic simulation demonstrating:
      - Worker #1: Truncated lower body -> Gloves UNKNOWN, overall status UNKNOWN
      - Worker #2: Adequate visibility but Gloves and Goggles absent -> NON_COMPLIANT (Confirmed violation)
      - Worker #3: All required PPE present -> COMPLIANT
    """
    print("=" * 75)
    print("IntelliWatch - PPE Spatial Association & Compliance Test (Step 5B)")
    print("Mode: Synthetic Multi-Frame Simulation (Clearly Labeled Synthetic)")
    print("=" * 75)

    canvas_w = 1280
    canvas_h = 720
    canvas = np.full((canvas_h, canvas_w, 3), 40, dtype=np.uint8)

    # Draw simple background lines/floor for realistic visualizer canvas
    cv2.line(canvas, (0, 600), (canvas_w, 600), (60, 60, 60), 2)
    cv2.putText(
        canvas,
        "IntelliWatch Synthetic Verification Canvas",
        (30, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (100, 100, 100),
        2,
        cv2.LINE_AA,
    )

    engine = PPEComplianceEngine(
        required_ppe=["Hardhat", "Safety Vest", "Gloves", "Goggles"],
        confirmation_frames=3,
        camera_id="cam_facility_01",
    )

    # Define 3 workers:
    # Worker 1: Lower body touches bottom border (y2 = 718px near canvas_h=720px) -> Gloves UNKNOWN
    w1_box = BoundingBox(x1=100.0, y1=350.0, x2=240.0, y2=718.0)
    # Worker 2: Fully visible in center of view, missing Gloves & Goggles
    w2_box = BoundingBox(x1=450.0, y1=200.0, x2=620.0, y2=600.0)
    # Worker 3: Fully visible on right, wearing all 4 required items
    w3_box = BoundingBox(x1=850.0, y1=200.0, x2=1020.0, y2=600.0)

    # Worker 1 gear: Hardhat, Vest, Goggles (No gloves, but bottom is clipped!)
    w1_gear = [
        DetectionResult(class_id=1, class_name="Hardhat", confidence=0.92, bbox=BoundingBox(x1=135.0, y1=335.0, x2=205.0, y2=395.0)),
        DetectionResult(class_id=2, class_name="Safety Vest", confidence=0.89, bbox=BoundingBox(x1=120.0, y1=420.0, x2=220.0, y2=540.0)),
        DetectionResult(class_id=4, class_name="Goggles", confidence=0.86, bbox=BoundingBox(x1=150.0, y1=385.0, x2=190.0, y2=410.0)),
    ]

    # Worker 2 gear: Hardhat, Vest only (Gloves and Goggles absent)
    w2_gear = [
        DetectionResult(class_id=1, class_name="Hardhat", confidence=0.94, bbox=BoundingBox(x1=495.0, y1=180.0, x2=575.0, y2=250.0)),
        DetectionResult(class_id=2, class_name="Safety Vest", confidence=0.91, bbox=BoundingBox(x1=475.0, y1=270.0, x2=595.0, y2=440.0)),
    ]

    # Worker 3 gear: All 4 items present (Hardhat, Vest, Left/Right Gloves, Goggles)
    w3_gear = [
        DetectionResult(class_id=1, class_name="Hardhat", confidence=0.95, bbox=BoundingBox(x1=895.0, y1=180.0, x2=975.0, y2=250.0)),
        DetectionResult(class_id=2, class_name="Safety Vest", confidence=0.93, bbox=BoundingBox(x1=875.0, y1=270.0, x2=995.0, y2=440.0)),
        DetectionResult(class_id=3, class_name="Gloves", confidence=0.87, bbox=BoundingBox(x1=840.0, y1=430.0, x2=875.0, y2=480.0)),
        DetectionResult(class_id=3, class_name="Gloves", confidence=0.88, bbox=BoundingBox(x1=995.0, y1=430.0, x2=1030.0, y2=480.0)),
        DetectionResult(class_id=4, class_name="Goggles", confidence=0.90, bbox=BoundingBox(x1=915.0, y1=235.0, x2=955.0, y2=260.0)),
    ]

    # Distant unassociated helmet on floor
    unassoc_gear = [
        DetectionResult(class_id=1, class_name="Hardhat", confidence=0.80, bbox=BoundingBox(x1=320.0, y1=620.0, x2=370.0, y2=660.0))
    ]

    all_ppe_dets = w1_gear + w2_gear + w3_gear + unassoc_gear

    tracked_persons = [
        TrackedObject(track_id=1, class_id=0, class_name="person", confidence=0.93, bbox=w1_box),
        TrackedObject(track_id=2, class_id=0, class_name="person", confidence=0.95, bbox=w2_box),
        TrackedObject(track_id=3, class_id=0, class_name="person", confidence=0.96, bbox=w3_box),
    ]

    tracks = FrameTracks(
        frame_id=101,
        timestamp=3.33,
        active_tracks=tracked_persons,
        total_track_count=3,
    )

    ppe_frame_detections = FrameDetections(
        frame_id=101,
        timestamp=3.33,
        detections=all_ppe_dets,
        frame_width=canvas_w,
        frame_height=canvas_h,
    )

    print("\n--- Simulating 3 Sequential Frames for Temporal Confirmation ---\n")
    confirmed_events = []

    last_association = None
    for frame_idx in range(1, 4):
        f_id = 100 + frame_idx
        f_ts = round(3.0 + frame_idx * 0.033, 3)

        tracks.frame_id = f_id
        tracks.timestamp = f_ts
        ppe_frame_detections.frame_id = f_id
        ppe_frame_detections.timestamp = f_ts

        last_association, frame_events = engine.process_frame(
            tracks=tracks,
            ppe_detections=ppe_frame_detections,
            frame_width=canvas_w,
            frame_height=canvas_h,
        )
        if frame_events:
            confirmed_events.extend(frame_events)

        print(f"Frame {f_id} processed: {len(frame_events)} new confirmed event(s) emitted.")

    print("\n" + "=" * 75)
    print("Worker-Level PPE Compliance Inventories (Evaluated at Frame 103):")
    print("=" * 75)

    for worker in last_association.worker_inventories:
        print(f"\nWorker #{worker.track_id}")
        for item_name in engine.required_ppe:
            state = worker.ppe_status.get(item_name, "UNKNOWN")
            print(f"  {item_name:12s}: {state}")
        print(f"\n  Compliance Status : {worker.compliance_status.value}")
        if worker.missing_ppe:
            print(f"  Missing Gear      : {', '.join(worker.missing_ppe)}")
        if worker.unknown_ppe:
            print(f"  Unknown Evidence  : {', '.join(worker.unknown_ppe)}")
        print(f"  Explanation       : {worker.explanation}")

    print("\n" + "-" * 75)
    print(f"Unassociated PPE Detections: {len(last_association.unassociated_ppe)}")
    for unp in last_association.unassociated_ppe:
        print(f"  - {unp.class_name} at bbox [{unp.bbox.x1:.1f}, {unp.bbox.y1:.1f}, {unp.bbox.x2:.1f}, {unp.bbox.y2:.1f}]")

    print("\n" + "=" * 75)
    print(f"Confirmed Safety Violation Events ({len(confirmed_events)} event(s)):")
    print("=" * 75)

    for ev in confirmed_events:
        print(f"\nEvent ID     : {ev.event_id}")
        print(f"Event Type   : {ev.event_type.value}")
        print(f"Track ID     : {ev.tracked_object_ids}")
        print(f"Timestamp    : {ev.timestamp:.3f}s")
        print(f"Severity     : {ev.severity.value.upper()}")
        print(f"Explanation  : {ev.explanation}")
        print(f"Metadata     : {ev.metadata}")

    # Generate Visualization
    visualizer = TrackingVisualizer()
    annotated = visualizer.draw_tracks(
        image=canvas,
        tracks=tracks,
        worker_inventories=last_association,
        show_ppe=True,
    )

    output_image_path.parent.mkdir(parents=True, exist_ok=True)
    visualizer.save_annotated_frame(annotated, output_image_path)
    print(f"\nAnnotated verification image saved to: {output_image_path}")
    print("=" * 75)
    print("PPE COMPLIANCE VERIFICATION COMPLETE!")
    print("=" * 75)


def run_real_pipeline_verification(image_path: Path, output_image_path: Path):
    """
    Runs end-to-end perception models on a real image:
      YOLODetector (Person) + PPEDetector (PPE) + PPEAssociationEngine + PPEComplianceEngine.
    """
    print("=" * 75)
    print("IntelliWatch - PPE Compliance Real-Image Verification (Step 5B)")
    print(f"Input Image: {image_path}")
    print("=" * 75)

    from vision.detection.ppe_detector import PPEDetector
    from vision.detection.yolo_detector import YOLODetector

    img = cv2.imread(str(image_path))
    if img is None:
        print(f"Could not load image: {image_path}")
        sys.exit(1)

    h, w = img.shape[:2]

    # Initialize detectors on CPU
    yolo_detector = YOLODetector(device="cpu", classes=["person"])
    ppe_detector = PPEDetector(device="cpu")

    t0 = time.perf_counter()
    person_dets = yolo_detector.detect(img)
    ppe_dets = ppe_detector.detect(img)
    det_time = (time.perf_counter() - t0) * 1000

    print(f"Inference completed in {det_time:.1f}ms on CPU:")
    print(f"  - Persons detected: {len(person_dets.detections)}")
    print(f"  - PPE objects detected: {len(ppe_dets.detections)}")

    # Convert detected persons into TrackedObject representation
    tracked_persons = []
    for idx, d in enumerate(person_dets.detections, start=1):
        tracked_persons.append(
            TrackedObject(
                track_id=idx,
                class_id=d.class_id,
                class_name="person",
                confidence=d.confidence,
                bbox=d.bbox,
            )
        )

    tracks = FrameTracks(
        frame_id=1,
        timestamp=0.0,
        active_tracks=tracked_persons,
        total_track_count=len(tracked_persons),
    )

    compliance_engine = PPEComplianceEngine(
        required_ppe=["Hardhat", "Safety Vest", "Gloves", "Goggles"],
        confirmation_frames=1,  # Single frame evaluation for static image
    )

    association, events = compliance_engine.process_frame(
        tracks=tracks,
        ppe_detections=ppe_dets,
        frame_width=w,
        frame_height=h,
    )

    print("\nWorker PPE Status:")
    for worker in association.worker_inventories:
        print(f"\nWorker #{worker.track_id}:")
        for k, v in worker.ppe_status.items():
            print(f"  {k:12s}: {v}")
        print(f"  Compliance: {worker.compliance_status.value}")
        print(f"  Explanation: {worker.explanation}")

    print(f"\nUnassociated PPE: {len(association.unassociated_ppe)}")

    # Visualize
    viz = TrackingVisualizer()
    annotated = viz.draw_tracks(
        image=img,
        tracks=tracks,
        worker_inventories=association,
        show_ppe=True,
    )

    viz.save_annotated_frame(annotated, output_image_path)
    print(f"Saved annotated image: {output_image_path}")
    print("=" * 75)


def main():
    parser = argparse.ArgumentParser(description="IntelliWatch PPE Compliance Verification Tool")
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Optional path to a test image to evaluate with real YOLO models",
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Force run the synthetic multi-frame verification scenario",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(PROJECT_ROOT / "data" / "output" / "ppe_compliance_test.jpg"),
        help="Path where annotated result image will be saved",
    )

    args = parser.parse_args()
    setup_logging()

    out_path = Path(args.output)

    if args.image:
        image_path = Path(args.image)
        run_real_pipeline_verification(image_path, out_path)
    else:
        # Check if sample image exists and synthetic not forced
        sample_img = PROJECT_ROOT / "data" / "samples" / "ppe_sample.jpg"
        if sample_img.exists() and not args.synthetic:
            # First run synthetic verification to demonstrate exact compliance states
            run_synthetic_verification(out_path)
            # Then also verify real image pipeline
            real_out = PROJECT_ROOT / "data" / "output" / "ppe_real_sample_test.jpg"
            print("\n")
            run_real_pipeline_verification(sample_img, real_out)
        else:
            run_synthetic_verification(out_path)


if __name__ == "__main__":
    main()
