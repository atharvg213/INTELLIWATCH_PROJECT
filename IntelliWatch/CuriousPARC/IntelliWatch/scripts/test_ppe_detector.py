"""
Single-Image PPE Detection Verification Tool (Step 5A)
Loads PPEDetector on CPU, runs inference on a test image,
prints detected PPE categories and confidences, and saves annotated output.
"""
import argparse
import sys
import time
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging_config import setup_logging
from configs.settings import get_settings
from vision.detection.ppe_detector import PPEDetector
from vision.detection.visualizer import DetectionVisualizer


def main():
    parser = argparse.ArgumentParser(description="IntelliWatch PPE Detection Verification Tool")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=str(PROJECT_ROOT / "data" / "samples" / "ppe_sample.jpg"),
        help="Path to the test image to evaluate (defaults to data/samples/ppe_sample.jpg)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/ppe_yolov8n.pt",
        help="Path to PPE YOLO weights (default: weights/ppe_yolov8n.pt)",
    )
    parser.add_argument(
        "--conf",
        "--confidence",
        type=float,
        default=0.25,
        dest="confidence",
        help="PPE confidence threshold between 0.0 and 1.0 (default: 0.25)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional filter for specific PPE categories, e.g. --classes Hardhat 'Safety Vest'",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' (default: cpu)",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        default=str(PROJECT_ROOT / "data" / "output" / "ppe_test.jpg"),
        help="Destination path for annotated output image (default: data/output/ppe_test.jpg)",
    )

    args = parser.parse_args()
    logger = setup_logging()
    image_file = Path(args.image_path)
    output_path = Path(args.output_path)

    if not image_file.exists():
        # Fallback to sample_test.jpg if ppe_sample.jpg is missing
        fallback = PROJECT_ROOT / "data" / "samples" / "sample_test.jpg"
        if fallback.exists():
            image_file = fallback
            logger.warning(f"Specified image not found. Using fallback: {image_file}")
        else:
            logger.error(f"Test image not found: {image_file}")
            sys.exit(1)

    print("=" * 70)
    print("IntelliWatch - PPE Detection Model Verification (Step 5A)")
    print("=" * 70)
    print(f"Device        : {args.device} (CPU execution)")
    print(f"Model         : {args.model_path}")
    print(f"Confidence    : {args.confidence}")
    print(f"Classes Filter: {args.classes or 'ALL PPE Categories'}")
    print(f"Test Image    : {image_file}")
    print(f"Output Image  : {output_path}")
    print("-" * 70)

    # 1. Initialize PPEDetector once (loaded on CPU)
    t0 = time.perf_counter()
    detector = PPEDetector(
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        device=args.device,
        classes=args.classes,
        map_to_original=True,
    )
    init_time = (time.perf_counter() - t0) * 1000
    print(f"PPE Detector loaded on [{detector.device}] in {init_time:.1f}ms")
    print(f"Recognized Categories: {list(detector.class_names.values())}")

    # 2. Load input image
    raw_img = cv2.imread(str(image_file))
    if raw_img is None or raw_img.size == 0:
        logger.error(f"Could not load image: {image_file}")
        sys.exit(1)

    # 3. Run inference
    t1 = time.perf_counter()
    frame_detections = detector.detect(raw_img)
    inference_time = (time.perf_counter() - t1) * 1000

    print("\nInference completed.\n")
    print(f"Detections found: {len(frame_detections.detections)}")
    for d in frame_detections.detections:
        bbox = d.bbox
        print(f"  - {d.class_name}: {d.confidence:.2f} | bbox: [{bbox.x1:.1f}, {bbox.y1:.1f}, {bbox.x2:.1f}, {bbox.y2:.1f}]")

    print(f"\nInference time: {inference_time:.1f} ms on CPU")

    # 4. Render and save annotated output
    visualizer = DetectionVisualizer()
    annotated = visualizer.draw_detections(
        image=raw_img,
        detections=frame_detections,
        show_labels=True,
        show_conf=True,
        show_banner=True,
    )

    visualizer.save_annotated_frame(annotated, output_path)
    print(f"Output: {output_path}")
    print("=" * 70)
    print("PPE DETECTION VERIFICATION COMPLETE!")
    print("=" * 70)


if __name__ == "__main__":
    main()
