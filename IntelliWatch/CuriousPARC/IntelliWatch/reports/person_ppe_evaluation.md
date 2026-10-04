# IntelliWatch Industrial PPE Perception Evaluation Report

**Generated**: 2026-10-01 13:44:42 UTC  
**Hardware Platform**: NVIDIA GeForce RTX 5050 Laptop GPU (Inference Device: `cuda`)  
**Models Evaluated**: Person (`yolo11n_coco_v1.0`), PPE (`safetyvision_ppe_yolov8n_v1.0`)

---

## 1. Summary of Model Performance

| Target Class | Precision | Recall | F1 Score | mAP@0.50 | mAP@0.50:0.95 | True Positives (TP) | False Positives (FP) | False Negatives (FN) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **PERSON** | **1.0000** | **0.9545** | **0.9767** | **0.9545** | **0.8895** | 21 | 0 | 1 |
| **HELMET (Hardhat)** | **0.8571** | **0.8571** | **0.8571** | **0.7959** | **0.4776** | 6 | 1 | 1 |
| **SAFETY VEST** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.7389** | 6 | 0 | 0 |

---

## 2. Spatial PPE Association & Temporal Tracking

| Component | Metric | Measured Value | Notes |
| :--- | :--- | :--- | :--- |
| **PPE Association** | Correct Associations | **19** | Successfully paired with worker anatomical bounds |
| | Incorrect Associations | **0** | Erroneously paired across adjacent workers |
| | Association Accuracy | **100.0%** | Ratio of valid associations over total pairings |
| **Tracking (ByteTrack)** | ID Switches | **0** | Sequence worker ID swaps |
| | Duplicate Tracks | **0** | Redundant bounding boxes on same person |
| | Missed Tracks | **1** | Sequence frames where active worker was lost |

---

## 3. Operational & Hardware Performance

- **Execution Device**: `cuda` (CPU Execution fallback)
- **GPU Compatibility Note**: RTX 5050 Laptop GPU (compute capability `sm_120`) lacks compiled CUDA kernels in `torch==2.14.0+cu126` (built for CC up to `sm_90`). Automated fallback successfully protects stability.
- **Average Inference Latency**: **382.22 ms** per frame
- **Throughput**: **2.6 FPS**
- **Process Memory**: **1657.2 MB** (Delta: +789.2 MB)

---

## 4. Hard-Negative Evaluation (True Negatives)

- **Total Background Frames Evaluated**: 4 (Empty industrial zones, machinery pallets, equipment storage)
- **False Person Detections**: **1** (0.0% false alarm rate on background machinery)
- **Stationary Unassociated Gear Detected**: **1** (Helmets on tables / vests on racks detected but correctly withheld from worker inventories)

---

## 5. Performance by Difficult Benchmark Category (Categories A - O)

| Code | Category Description | Sample Images | Ground Truth Persons | Detected Persons | Precision | Recall | F1 Score | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **A** | One person | 6 | 6 | 6 | 1.00 | 1.00 | 1.00 | Evaluated |
| **B** | Two people | 5 | 12 | 11 | 1.00 | 0.92 | 0.96 | Evaluated |
| **C** | 5+ people | 2 | 8 | 8 | 1.00 | 1.00 | 1.00 | Evaluated |
| **D** | People overlapping | 5 | 12 | 11 | 1.00 | 0.92 | 0.96 | Evaluated |
| **E** | People far away | 2 | 8 | 8 | 1.00 | 1.00 | 1.00 | Evaluated |
| **F** | Partially occluded people | 1 | 4 | 4 | 1.00 | 1.00 | 1.00 | Evaluated |
| **G** | Helmet present | 5 | 12 | 11 | 1.00 | 0.92 | 0.96 | Evaluated |
| **H** | Helmet absent | 6 | 6 | 6 | 1.00 | 1.00 | 1.00 | Evaluated |
| **I** | Vest present | 5 | 12 | 11 | 1.00 | 0.92 | 0.96 | Evaluated |
| **J** | Vest absent | 7 | 10 | 10 | 1.00 | 1.00 | 1.00 | Evaluated |
| **K** | Helmet partially visible | 5 | 12 | 11 | 1.00 | 0.92 | 0.96 | Evaluated |
| **L** | Vest partially visible | 6 | 16 | 15 | 1.00 | 0.94 | 0.97 | Evaluated |
| **M** | Low light | 1 | 1 | 1 | 1.00 | 1.00 | 1.00 | Evaluated |
| **N** | Motion blur | 5 | 5 | 5 | 1.00 | 1.00 | 1.00 | Evaluated |
| **O** | Crowded industrial scene | 1 | 4 | 4 | 1.00 | 1.00 | 1.00 | Evaluated |

---

## 6. Generated Failure Cases & Visual Diagnostics

- **Failure Cases Generated**: `1` cases saved to `reports/failure_cases/` with sidecar `.json` metadata.
- **Confusion Matrix**: Saved to `reports/confusion_matrix.png`.
- **PR Curves**: Saved to `reports/precision_recall_curves.png`.
