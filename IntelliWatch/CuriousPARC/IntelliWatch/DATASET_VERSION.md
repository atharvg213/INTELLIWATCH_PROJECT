# IntelliWatch Dataset Versioning

## Dataset Identifier: `intelliwatch_ppe_v1.0.0`
- **Release Date**: October 2026
- **Format**: Ultralytics YOLO Annotation Format (`.txt` per `.jpg` image)
- **Target Perception Task**: Industrial Worker Detection & PPE Compliance Verification
- **Root Directory**: `datasets/intelliwatch_ppe/`

---

## 1. Directory Structure

```text
datasets/intelliwatch_ppe/
├── data.yaml                # YOLO training/validation dataset specification
├── README.md                # Dataset documentation, annotation guidelines, class schema
├── images/
│   ├── train/               # Training image frames
│   ├── val/                 # Validation image frames (unseen scene segments)
│   ├── test/                # Fixed held-out test frames (categories A-O)
│   └── hard_cases/          # True negatives, empty zones, gear on shelves
└── labels/
    ├── train/               # Normalized YOLO labels [class x y w h]
    ├── val/
    ├── test/
    └── hard_cases/
```

---

## 2. Split Breakdown & Leakage Prevention Rules

To prevent temporal and spatial data leakage:
1. **Scene-Level Partitioning**: Adjacent video frames from the same video track are NEVER split across train and test sets.
2. **Held-Out Test Set**: The test set contains completely held-out scenes, including the high-resolution factory logistics yard (`job_img_2adfe9fb54`), multi-elevation workers (`job_img_302fb486f9`), and unseen video sequence tail frames.
3. **Hard-Negative Subsets**: Frames featuring industrial machinery, pallets, and empty safety zones without humans are explicitly isolated to evaluate background false-alarm rates.

| Split | Images | Target Purpose | Primary Scene Sources |
| :--- | :--- | :--- | :--- |
| **train** | 16 | Model fine-tuning & representation learning | Moving worker seq 1-15, synthetic benchmarks |
| **val** | 8 | Hyperparameter tuning & validation | Moving worker seq 16-25, controlled transitions |
| **test** | 12 | Fixed benchmark evaluation (Categories A-O) | Factory Yard, Construction Scaffolding, Worker Close-up, Moving worker seq 35-50 |
| **hard_cases** | 4 | Hard-negative & unassociated gear testing | Empty machinery zone, pallet racks, storage fixtures |
| **Total** | **40** | High-density annotated industrial domain frames | Grouped multi-source CCTV captures |

---

## 3. Label Specification

Coordinates are normalized to `[0.0, 1.0]` relative to image width and height:
`class_id x_center y_center width height`

Class mapping:
- `0`: `person` (Worker standing, walking, seated, or operating equipment)
- `1`: `hardhat` (Protective hard hat or safety helmet worn on head)
- `2`: `safety_vest` (High-visibility reflective safety vest or jacket worn on torso)
- `3`: `no_hardhat` (Worker head visibly lacking a helmet in a designated zone)
- `4`: `no_safety_vest` (Worker torso visibly lacking a safety vest)
