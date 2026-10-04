import sys
sys.path.insert(0, ".")
import cv2
import json
import time
import torch
from vision.detection.yolo_detector import YOLODetector

img_path = 'data/input/uploads/job_img_f05e32633a_input.jpg'
img = cv2.imread(img_path)

results = {}

for res_name, imgsz in [("640x640", 640), ("960x960", 960)]:
    results[res_name] = {}
    for dev in ["cpu", "cuda"]:
        detector = YOLODetector(device=dev)
        
        # Measure warm-up (first inference call)
        if dev == "cuda":
            torch.cuda.synchronize()
        t_warm_0 = time.perf_counter()
        _ = detector.detect(img, imgsz=imgsz)
        if dev == "cuda":
            torch.cuda.synchronize()
        warmup_ms = (time.perf_counter() - t_warm_0) * 1000.0

        # Run 2 more warmup runs
        for _ in range(2):
            _ = detector.detect(img, imgsz=imgsz)
            if dev == "cuda":
                torch.cuda.synchronize()

        # Measure steady-state (10 runs)
        times = []
        for _ in range(10):
            if dev == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            d_res = detector.detect(img, imgsz=imgsz)
            if dev == "cuda":
                torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)

        avg_latency_ms = (sum(times) / len(times)) * 1000.0
        fps = 1000.0 / avg_latency_ms

        results[res_name][dev] = {
            "device": dev,
            "imgsz": imgsz,
            "warmup_ms": round(warmup_ms, 2),
            "steady_state_latency_ms": round(avg_latency_ms, 2),
            "fps": round(fps, 2),
            "detections_count": len(d_res.detections),
        }
        print(f"[{res_name}] {dev.upper()}: Warmup={warmup_ms:.2f}ms, Steady={avg_latency_ms:.2f}ms, FPS={fps:.2f}, Detections={len(d_res.detections)}")

with open('scratch/cpu_gpu_performance_results.json', 'w') as f:
    json.dump(results, f, indent=2)

print("Benchmark saved to scratch/cpu_gpu_performance_results.json")
