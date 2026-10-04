"""
Single Image Object Detection Verification Tool (Step 3)
Runs YOLODetector on a single image file, prints structured detections,
and saves the annotated output frame to data/output/.
"""
import argparse
import sys
from pathlib import Path
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.logging_config import setup_logging
from vision.detection.visualizer import DetectionVisualizer
from vision.detection.yolo_detector import YOLODetector
from vision.preprocessing.frame import FrameData
from vision.preprocessing.frame_processor import FramePreprocessor


def main():
    parser = argparse.ArgumentParser(description="IntelliWatch Single-Image Detection Verification Tool")
    parser.add_argument(
        "image_path",
        nargs="?",
        default=str(PROJECT_ROOT / "data" / "samples" / "sample_test.jpg"),
        help="Path to the image to run detection on (default: data/samples/sample_test.jpg)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="weights/yolo11n.pt",
        help="Path to YOLO weights (default: weights/yolo11n.pt)",
    )
    parser.add_argument(
        "--conf",
        "--confidence",
        type=float,
        default=0.25,
        dest="confidence",
        help="Confidence threshold between 0.0 and 1.0 (default: 0.25)",
    )
    parser.add_argument(
        "--classes",
        nargs="*",
        default=None,
        help="Optional filter for specific classes, e.g. --classes person truck",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Inference device: 'cpu' or 'cuda' (default: cpu)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(PROJECT_ROOT / "data" / "output"),
        help="Directory to save annotated image (default: data/output)",
    )

    args = parser.parse_args()
    logger = setup_logging()
    image_file = Path(args.image_path)

    if not image_file.exists():
        logger.error(f"Image not found at path: {image_file}")
        sys.exit(1)

    image = cv2.imread(str(image_file))
    if image is None:
        logger.error(f"Failed to decode image from: {image_file}")
        sys.exit(1)

    h, w = image.shape[:2]
    print("=" * 75)
    print("IntelliWatch - Single Image Object Detection Verification (Step 3)")
    print("=" * 75)
    print(f"Target Image Source   : {image_file} ({w}x{h})")
    print(f"Model Weights         : {args.model_path}")
    print(f"Inference Device      : {args.device}")
    print(f"Confidence Threshold  : {args.confidence}")
    print(f"Class Filter          : {args.classes or 'ALL (No Filter)'}")
    print("-" * 75)

    # 1. Initialize Detector & Visualizer
    detector = YOLODetector(
        model_path=args.model_path,
        confidence_threshold=args.confidence,
        device=args.device,
        classes=args.classes,
        map_to_original=True,
    )
    visualizer = DetectionVisualizer()

    # 2. Preprocess with letterbox (640x640) to test coordinate transformation
    preprocessor = FramePreprocessor(target_width=640, target_height=640)
    raw_frame_data = FrameData(
        frame_index=0,
        timestamp=0.0,
        image=image,
        width=w,
        height=h,
    )
    preprocessed_frame = preprocessor.preprocess(raw_frame_data)

    print(
        f"[1] Preprocessing Applied: Letterbox scaled {w}x{h} -> 640x640 "
        f"(scale={preprocessed_frame.scale_factor:.3f}, pad={preprocessed_frame.pad_offset})"
    )

    # 3. Run Inference
    detections = detector.detect(preprocessed_frame)
    num_dets = len(detections.detections)

    print(f"[2] Inference Complete: {num_dets} object(s) detected.")
    print("-" * 75)
    print(f"{'CLASS':<15} | {'CONF':<8} | {'ORIGINAL COORDINATES (x1, y1, x2, y2)':<35} | {'SIZE (WxH)':<12}")
    print("-" * 75)

    for d in detections.detections:
        box = d.bbox
        coords_str = f"({box.x1:.1f}, {box.y1:.1f}, {box.x2:.1f}, {box.y2:.1f})"
        size_str = f"{box.width:.1f}x{box.height:.1f}"
        print(f"{d.class_name:<15} | {d.confidence:<8.2f} | {coords_str:<35} | {size_str:<12}")

    # 4. Draw annotations on the original-resolution image using mapped coordinates
    annotated_orig = visualizer.draw_detections(image, detections)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"step3_image_{image_file.stem}_annotated.jpg"
    visualizer.save_annotated_frame(annotated_orig, out_path)

    print("-" * 75)
    print(f"[3] Annotated original image saved to: {out_path}")
    print("=" * 75)
    print("SINGLE IMAGE DETECTION VERIFICATION SUCCESSFUL!")
    print("=" * 75)


if __name__ == "__main__":
    main()
