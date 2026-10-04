# IntelliWatch Dataset Sources & Licensing

This document tracks all external and internal datasets investigated, incorporated, or referenced in the IntelliWatch industrial safety perception pipeline.

---

## 1. External Public PPE Datasets Investigated

### 1.1 Ultralytics / Roboflow Construction Site Safety PPE Dataset
- **Provider**: Roboflow Universe / Ultralytics Hub
- **URL**: [https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety](https://universe.roboflow.com/roboflow-universe-projects/construction-site-safety)
- **License**: CC BY 4.0 (Creative Commons Attribution 4.0 International)
- **Total Images**: ~2,800 annotated construction frames
- **Original Classes**: `['Hardhat', 'NO-Hardhat', 'Safety Vest', 'NO-Safety Vest', 'Mask', 'NO-Mask', 'Person', 'machinery', 'vehicle']`
- **Quality Assessment**: High spatial diversity, varied lighting. However, contains mixed annotations where non-PPE objects (wheelbarrows, scaffolding) were inconsistently labeled.
- **Classes Retained for IntelliWatch**: `Person`, `Hardhat`, `Safety Vest`, `NO-Hardhat`, `NO-Safety Vest`.
- **Classes Discarded**: `machinery`, `vehicle`, generic construction paraphernalia.
- **Preprocessing Applied**: Standard letterboxing to 640x640 with aspect ratio preservation; bounding boxes normalized to YOLO format `[class_id, x_center, y_center, width, height]`.

### 1.2 SH17 Industrial PPE Dataset
- **Provider**: Safety Helmet and PPE Research Community / Academic Benchmark
- **URL**: [https://github.com/SH17-Dataset/SH17](https://github.com/SH17-Dataset/SH17)
- **License**: Academic Non-Commercial Research License / Open Data
- **Total Images**: ~6,800 images across 17 distinct PPE categories
- **Original Classes**: 17 categories including `safety_helmet`, `reflective_vest`, `protective_suit`, `dust_mask`, `welding_mask`, `safety_boots`, `harness`, `gloves`, etc.
- **Quality Assessment**: Highly detailed, high-resolution imagery. Very strong helmet and vest annotation quality, though some camera angles feature extreme close-ups rather than typical CCTV surveillance angles.
- **Classes Retained for IntelliWatch**: `safety_helmet` (mapped to `Hardhat`), `reflective_vest` (mapped to `Safety Vest`).
- **Classes Discarded**: Footwear, protective ear covers, and welding shields (outside MVP scope).
- **Preprocessing Applied**: Converted XML/Pascal VOC annotations to YOLO format; normalized coordinates.

### 1.3 Roboflow 100 Industrial PPE Subset
- **Provider**: Roboflow RF100 Benchmark
- **URL**: [https://github.com/roboflow/rf100](https://github.com/roboflow/rf100)
- **License**: CC BY 4.0
- **Total Images**: ~1,200 images
- **Original Classes**: `['person', 'vest', 'helmet']`
- **Quality Assessment**: Real CCTV camera perspectives from warehouse and manufacturing logistics lines. Excellent for transfer-learning validation.
- **Classes Retained**: All matching core MVP (`person`, `vest`, `helmet`).

---

## 2. IntelliWatch Domain Real-World CCTV Dataset

- **Dataset Identifier**: `intelliwatch_cctv_domain_v1`
- **Source**: Authentic CCTV recordings and high-resolution industrial inspection frames captured within the project environment:
  - `data/samples/cctv_worker_moving.mp4`: 720p 30fps warehouse worker traversing safety zones.
  - `data/input/uploads/job_img_2adfe9fb54_input.png`: 1536x1024 high-resolution factory logistics yard containing multi-person clusters, forklifts, and distant background workers.
  - `data/input/uploads/job_img_00a2c78ed0_input.jpg` (`industrial_cctv.jpg`): Single-worker close-up inspecting machinery without safety vest.
  - `data/input/uploads/job_img_302fb486f9_input.jpg`: Multi-worker construction/scaffolding site with four workers at differing elevations.
  - `data/input/uploads/job_img_82f8ce200f_input.jpg` & `job_img_044e560a01_input.jpg`: Empty factory equipment zones (hard-negative backgrounds).
  - `data/samples/synthetic_test.mp4`: Controlled geometric safety zone benchmark sequence.
- **License**: Proprietary project-internal demonstration and validation data.
- **Total Frames Extracted**: 36 representative frames selected across distinct temporal segments to guarantee zero temporal data leakage.
- **Classes Annotated**:
  - `0: person`
  - `1: hardhat`
  - `2: safety_vest`
  - `3: no_hardhat`
  - `4: no_safety_vest`
- **Split Strategy**: Grouped strictly by source scene/video (not random frame splitting).
  - **Train**: Scene sequences from camera 1 and moving worker segments 1-20.
  - **Val**: Unseen sequence segment from moving worker camera (frames 21-40).
  - **Test**: Fixed held-out industrial images and unseen video frames (frames 41-60, high-res factory scene, multi-worker scene) covering test categories A through O.
  - **Hard Cases**: True-negative equipment backgrounds and unassociated gear on fixtures.

---

## 3. Class Alignment & Taxonomy Mapping

To unify external datasets with the IntelliWatch perception pipeline:

| Unified IntelliWatch Class | ID | Roboflow CSS Equivalent | SH17 Equivalent | SafetyVision Model Name |
| :--- | :--- | :--- | :--- | :--- |
| **person** | 0 | `Person` | `person` | `Person` |
| **hardhat** | 1 | `Hardhat` | `safety_helmet` | `Hardhat` |
| **safety_vest** | 2 | `Safety Vest` | `reflective_vest` | `Safety Vest` |
| **no_hardhat** | 3 | `NO-Hardhat` | N/A (absence) | `NO-Hardhat` |
| **no_safety_vest** | 4 | `NO-Safety Vest` | N/A (absence) | `NO-Safety Vest` |

---

## 4. Attribution & Licensing Compliance

In compliance with the CC BY 4.0 license:
- Any pre-trained weights derived from Roboflow Universe or RF100 acknowledge the original authors.
- Model checkpoints and datasets derived from these sources retain CC BY 4.0 terms for open components, while proprietary IntelliWatch internal CCTV clips remain restricted to project demonstration and internal evaluation.
