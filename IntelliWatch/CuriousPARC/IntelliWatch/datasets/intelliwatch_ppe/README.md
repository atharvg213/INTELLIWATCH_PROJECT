# IntelliWatch Industrial PPE Domain Dataset (`intelliwatch_ppe_v1.0.0`)

## Overview
This dataset provides a curated, leak-free industrial benchmark for training and evaluating computer vision models within the IntelliWatch safety perception ecosystem.

## Splits
- `images/train/`: 10 authentic CCTV images containing moving workers across temporal chunk 1.
- `images/val/`: 8 images representing unseen intermediate CCTV sequence frames (temporal chunk 2).
- `images/test/`: 12 benchmark images rigorously categorized across industrial test categories A through O.
- `images/hard_cases/`: 4 true-negative industrial environments and stationary gear fixtures.

## Class Taxonomy
- `0: person`: Industrial human worker (standing, walking, seated, operating vehicles).
- `1: hardhat`: Industrial protective helmet / hard hat worn on head.
- `2: safety_vest`: High-visibility reflective vest or jacket worn on torso.
- `3: no_hardhat`: Worker head visibly lacking required helmet protection.
- `4: no_safety_vest`: Worker torso visibly lacking high-visibility vest protection.

## Test Category Coverage (A - O)
All test images are mapped in `test_set_categories.json` to verify:
- **A. One person**: `test_cctv_single_worker.jpg`, `test_worker_seq_*.jpg`
- **B. Two people**: `test_two_workers_overlap_*.jpg`, `test_scaffolding.jpg`
- **C. 5+ people**: `test_factory_yard.jpg`, `test_scaffolding.jpg`
- **D. People overlapping**: `test_two_workers_overlap_*.jpg`, `test_factory_yard.jpg`
- **E. People far away**: `test_factory_yard.jpg` (1441x178), `test_scaffolding.jpg` (582x224)
- **F. Partially occluded people**: `test_factory_yard.jpg` (in-vehicle operator 433x298)
- **G. Helmet present**: `test_scaffolding.jpg`, `test_two_workers_overlap_*.jpg`
- **H. Helmet absent**: `test_cctv_single_worker.jpg`, `test_worker_seq_*.jpg`
- **I. Vest present**: `test_factory_yard.jpg`, `test_two_workers_overlap_*.jpg`
- **J. Vest absent**: `test_scaffolding.jpg`, `test_cctv_single_worker.jpg`, `test_worker_seq_*.jpg`
- **K. Helmet partially visible**: `test_two_workers_overlap_*.jpg`, `test_scaffolding.jpg`
- **L. Vest partially visible**: `test_two_workers_overlap_*.jpg`, `test_factory_yard.jpg`
- **M. Low light**: `test_cctv_single_worker.jpg`
- **N. Motion blur**: `test_worker_seq_*.jpg`
- **O. Crowded industrial scene**: `test_factory_yard.jpg`
