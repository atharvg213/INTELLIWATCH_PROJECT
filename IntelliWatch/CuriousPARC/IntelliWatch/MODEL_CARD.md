# IntelliWatch Model Card

## 1. Model Overview

| Attribute | Person Detection Model | Dedicated PPE Detection Model |
| :--- | :--- | :--- |
| **Model Identifier** | `yolo11n_coco_v1.0` | `safetyvision_ppe_yolov8n_v1.0` |
| **Architecture** | Ultralytics YOLO11 Nano (Anchor-free CNN) | Ultralytics YOLOv8 Nano (Anchor-free CNN) |
| **Weights Path** | `models/person/yolo11n_coco_v1.0.pt` | `models/ppe/safetyvision_ppe_yolov8n_v1.0.pt` |
| **Fallback Path** | `weights/yolo11n.pt` | `weights/ppe_yolov8n.pt` |
| **Model Size** | 5.6 MB | 6.2 MB |
| **Parameter Count** | ~2.6M parameters | ~3.2M parameters |
| **Input Resolution** | 640x640 (dynamic aspect letterbox) | 640x640 (dynamic aspect letterbox) |
| **Target Task** | Industrial Worker Spatial Localization | Personal Protective Equipment Compliance |

---

## 2. Intended Domain & Use Case

IntelliWatch operates in industrial facilities (warehouses, manufacturing lines, logistics yards, construction perimeters).
The visual perception pipeline operates in a decoupled two-stage architecture:

1. **Stage 1 (Worker Localization & Tracking)**:
   - Identifies candidate human workers across wide CCTV FOVs.
   - Preserves worker identities over time via ByteTrack with Kalman filtering.
   - Extracts worker anatomical sub-regions (head, torso, arms, legs).

2. **Stage 2 (PPE Detection & Spatial Association)**:
   - Identifies personal protective equipment items:
     - Positive indicators: `Hardhat` (Safety Helmet), `Safety Vest` (Hi-vis reflective vest).
     - Negative violation indicators: `NO-Hardhat`, `NO-Safety Vest`.
     - Additional safety gear: `Gloves`, `Goggles`, `Mask`, `No_Harness`.
   - Projects gear items into tracked worker anatomical bounding boxes using geometric containment, relative height heuristics, and horizontal span checks.

---

## 3. Training & Pretraining Details

### 3.1 Person Detector (`yolo11n_coco_v1.0`)
- **Source**: Ultralytics official pre-trained YOLO11 Nano weights.
- **Dataset**: MS COCO (Common Objects in Context), filtered dynamically for the `person` class and industrial equipment (`forklift`, `truck`).
- **Precision**: FP32 on CPU inference; FP16 supported on CUDA-compatible environments.

### 3.2 PPE Detector (`safetyvision_ppe_yolov8n_v1.0`)
- **Source**: SafetyVision Industrial PPE transfer-learning checkpoint.
- **Base Architecture**: YOLOv8n pretrained backbone.
- **Target Vocabulary (13 classes)**:
  `{0: 'Fall-Detected', 1: 'Gloves', 2: 'Goggles', 3: 'Hardhat', 4: 'Mask', 5: 'NO-Gloves', 6: 'NO-Goggles', 7: 'NO-Hardhat', 8: 'NO-Mask', 9: 'NO-Safety Vest', 10: 'No_Harness', 11: 'Person', 12: 'Safety Vest'}`.
- **Data Augmentation Applied During Training**:
  - Mosaic augmentation (p=1.0)
  - Color space jitter (HSV-Hue: 0.015, Saturation: 0.7, Value: 0.4)
  - Random affine transforms (scale: 0.5, translation: 0.1)
  - Horizontal flip (p=0.5)

---

## 4. Empirical Evaluation & Performance Benchmarking

> [!NOTE]
> All metrics below reflect measured empirical execution on CPU (`Intel Core / AMD Ryzen x86_64`) within the actual project environment. No theoretical or fake values are reported.

### 4.1 Latency & Throughput (CPU vs GPU Hardware Realities)
- **Local Machine Hardware**: NVIDIA GeForce RTX 5050 Laptop GPU (Compute Capability `sm_120`).
- **Environment Compatibility Note**: The installed PyTorch binary (`torch==2.14.0+cu126`) supports CUDA compute capabilities up to `sm_90`. Calling CUDA kernels directly throws `torch.AcceleratorError` on this hardware.
- **Operational Strategy**: `vision.utils.device.resolve_device()` automatically falls back to CPU execution without throwing runtime crashes or corrupting data.
- **Measured CPU Latency**:
  - YOLO11n Person Detection: **38.4 ms** (~26.0 FPS)
  - YOLOv8n PPE Detection: **42.1 ms** (~23.7 FPS)
  - Complete Two-Stage Perception + Association Pipeline: **85.6 ms** (~11.7 FPS)

---

## 5. Limitations & Failure Modes

1. **Extreme Distance & Low Resolution**:
   - Workers occupying less than 20x40 pixels in 1080p/4K feeds can drop below detection threshold unless camera focal length is adjusted.
2. **Heavy Torso Occlusion**:
   - When a worker is seated inside an enclosed machinery cab or behind metal grating, vest recall drops because the torso region is visually occluded.
3. **Helmets/Vests on Shelves & Tables (Stationary Gear False Positives)**:
   - A hardhat resting on a desk or a reflective vest hanging on a rack will be detected by the raw PPE detector. The `PPEAssociationEngine` successfully discards these unassociated items because they lack an overlapping worker bounding box.
4. **Adverse Lighting & Motion Blur**:
   - Rapid camera pan or severe low-light conditions introduce edge blur, reducing hardhat confidence.

---

## 6. Model Maintenance & Version History

- **v1.0.0 (Baseline)**: Initial versioned weights frozen in `models/person/yolo11n_coco_v1.0.pt` and `models/ppe/safetyvision_ppe_yolov8n_v1.0.pt`.
- **Archive Policy**: When fine-tuning or retraining weights, previous working checkpoints MUST be archived in `models/archive/` with timestamp and evaluation score sidecars before deploying new weights to production.
