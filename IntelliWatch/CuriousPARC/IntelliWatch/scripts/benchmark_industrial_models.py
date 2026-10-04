"""
Benchmark Industrial Object Detection Candidates on CPU (Step 2 & 3).
Evaluates baseline YOLO11n vs YOLO-World candidates across test images on CPU.
Measures:
- Model load time
- Inference latency per image (warmup + multi-run average)
- Memory / model size
- Detected objects and confidence scores
- Industrial vs false-positive detection balance
"""
import os
import sys
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ultralytics import YOLO

INDUSTRIAL_CLASSES = [
    "forklift",
    "industrial machine",
    "robotic arm",
    "conveyor",
    "pallet",
    "safety barrier",
    "electrical cabinet",
    "industrial vehicle",
    "person",
]

TEST_IMAGES = [
    PROJECT_ROOT / "data" / "samples" / "industrial_cctv.jpg",
    PROJECT_ROOT / "data" / "samples" / "ppe_sample.jpg",
    PROJECT_ROOT / "data" / "samples" / "sample_test.jpg",
]

CANDIDATES = [
    {
        "name": "YOLO11n (Baseline COCO)",
        "path": str(PROJECT_ROOT / "weights" / "yolo11n.pt"),
        "is_world": False,
    },
    {
        "name": "YOLOv8s-World-v2",
        "path": str(PROJECT_ROOT / "yolov8s-worldv2.pt"),
        "is_world": True,
    },
    {
        "name": "YOLOv8s-World-v1",
        "path": str(PROJECT_ROOT / "yolov8s-world.pt"),
        "is_world": True,
    },
]


def benchmark_model(candidate, images, num_runs=3, conf_thresh=0.25):
    print("=" * 70)
    print(f"BENCHMARKING: {candidate['name']}")
    print(f"Model File: {candidate['path']}")
    file_size_mb = os.path.getsize(candidate['path']) / (1024 * 1024)
    print(f"Model Size: {file_size_mb:.2f} MB")
    
    # Measure Load Time
    t0 = time.perf_counter()
    model = YOLO(candidate["path"])
    if candidate["is_world"]:
        model.set_classes(INDUSTRIAL_CLASSES)
    load_time_sec = time.perf_counter() - t0
    print(f"Model Load Time: {load_time_sec:.3f} s")
    
    results_per_image = []

    for img_path in images:
        if not img_path.exists():
            print(f"Warning: {img_path} not found.")
            continue
        
        img = cv2.imread(str(img_path))
        h, w = img.shape[:2]
        
        # Warmup
        _ = model.predict(img, device="cpu", verbose=False, conf=conf_thresh)
        
        # Multi-run latency measurement
        latencies = []
        for _ in range(num_runs):
            t_start = time.perf_counter()
            preds = model.predict(img, device="cpu", verbose=False, conf=conf_thresh)
            latencies.append((time.perf_counter() - t_start) * 1000.0)
            
        avg_latency = float(np.mean(latencies))
        fps = 1000.0 / avg_latency if avg_latency > 0 else 0.0
        
        # Parse detections
        detections = []
        boxes = preds[0].boxes
        if boxes is not None and len(boxes) > 0:
            for b in boxes:
                cls_id = int(b.cls.item())
                cls_name = model.names[cls_id] if model.names and cls_id in model.names else str(cls_id)
                confidence = float(b.conf.item())
                xyxy = [float(x) for x in b.xyxy[0].tolist()]
                detections.append({
                    "class": cls_name,
                    "confidence": round(confidence, 3),
                    "bbox": [round(x, 1) for x in xyxy],
                })
        
        print(f"\n--- Image: {img_path.name} ({w}x{h}) ---")
        print(f"Avg CPU Latency ({num_runs} runs): {avg_latency:.1f} ms ({fps:.2f} FPS)")
        print(f"Total Detections: {len(detections)}")
        for d in detections:
            print(f"  - {d['class']}: {d['confidence']:.3f} at {d['bbox']}")
            
        results_per_image.append({
            "image": img_path.name,
            "latency_ms": avg_latency,
            "fps": fps,
            "detections": detections,
        })
        
    return {
        "candidate": candidate["name"],
        "size_mb": file_size_mb,
        "load_time_sec": load_time_sec,
        "image_results": results_per_image,
    }


def main():
    print("IntelliWatch Industrial Perception Candidate Benchmark")
    print("Platform: CPU (PyTorch Ultralytics)")
    print(f"Industrial Target Vocabulary: {INDUSTRIAL_CLASSES}\n")
    
    all_summaries = []
    for cand in CANDIDATES:
        if os.path.exists(cand["path"]):
            summary = benchmark_model(cand, TEST_IMAGES, num_runs=3, conf_thresh=0.20)
            all_summaries.append(summary)
        else:
            print(f"Skipping {cand['name']}: {cand['path']} does not exist.")

    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY COMPARISON")
    print("=" * 70)
    print(f"{'Model':<25} | {'Size (MB)':<10} | {'Load (s)':<9} | {'Avg Latency (ms)':<17} | {'Avg FPS':<8}")
    print("-" * 75)
    for s in all_summaries:
        avg_lat = np.mean([r["latency_ms"] for r in s["image_results"]]) if s["image_results"] else 0
        avg_fps = 1000.0 / avg_lat if avg_lat > 0 else 0
        print(f"{s['candidate']:<25} | {s['size_mb']:<10.1f} | {s['load_time_sec']:<9.2f} | {avg_lat:<17.1f} | {avg_fps:<8.2f}")


if __name__ == "__main__":
    main()
