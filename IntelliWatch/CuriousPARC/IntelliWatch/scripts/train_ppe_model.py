"""
scripts/train_ppe_model.py
==========================
Reproducible Training & Fine-Tuning Pipeline for IntelliWatch Industrial PPE Models.

Features:
- Configures transfer learning from pretrained YOLO weights (`yolo11n.pt` or `ppe_yolov8n.pt`).
- Uses the unified `datasets/intelliwatch_ppe/data.yaml` specification.
- Configures industrial CCTV augmentations (lighting variation, contrast, scale, perspective).
- Detects environment hardware:
  - If executed on compatible CUDA GPUs (e.g. sm_80, sm_86, sm_90 or cloud A100/H100/T4), runs GPU training.
  - If executed on local RTX 5050 (sm_120) with torch cu126, warns operator and supports dry-run or CPU verification.
- Saves versioned checkpoints to `models/archive/` and logs training metadata.
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from vision.utils.device import get_device_diagnostics, resolve_device


def train_model(
    data_yaml: str = "datasets/intelliwatch_ppe/data.yaml",
    base_model: str = "models/ppe/safetyvision_ppe_yolov8n_v1.0.pt",
    epochs: int = 50,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "auto",
    workers: int = 2,
    learning_rate: float = 0.001,
    dry_run: bool = False,
):
    print("=" * 80)
    print("INTELLIWATCH REPRODUCIBLE MODEL TRAINING & FINE-TUNING PIPELINE")
    print("=" * 80)

    diag = get_device_diagnostics()
    resolved_dev = resolve_device(device)
    print(f"Host Hardware: {diag['device_name']} (CUDA={diag['cuda_available']}, Capability={diag.get('cuda_capability')})")
    print(f"Target Execution Device: {resolved_dev}")

    if diag.get("cuda_available") and diag.get("cuda_capability") == (12, 0) and resolved_dev == "cpu":
        print("\n[HARDWARE NOTICE]")
        print("Local GPU is NVIDIA GeForce RTX 5050 Laptop GPU (sm_120).")
        print("Installed PyTorch cu126 only supports compute capabilities up to sm_90.")
        print("For rapid local verification, training will proceed on CPU.")
        print("For production fine-tuning, run this exact script on a compatible cloud GPU:")
        print(f"  python scripts/train_ppe_model.py --device 0 --epochs {epochs} --batch {batch_size}\n")

    data_path = Path(data_yaml)
    if not data_path.is_absolute():
        data_path = PROJECT_ROOT / data_path
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset configuration not found at: {data_path}")

    model_path = Path(base_model)
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    if not model_path.exists():
        fallback = PROJECT_ROOT / "weights" / "ppe_yolov8n.pt"
        if fallback.exists():
            model_path = fallback
        else:
            raise FileNotFoundError(f"Base model checkpoint not found at: {model_path}")

    print(f"Base Model Weights : {model_path.name}")
    print(f"Dataset Config     : {data_path.as_posix()}")
    print(f"Epochs             : {epochs}")
    print(f"Batch Size         : {batch_size}")
    print(f"Image Resolution   : {imgsz}x{imgsz}")
    print(f"Initial LR         : {learning_rate}")

    # Industrial Surveillance Augmentation Hyperparameters
    # Designed specifically for fixed and PTZ surveillance angles:
    # - HSV jitter for day/night/fluorescent warehouse lighting
    # - Perspective and scale for varied camera mounting heights
    # - Flplr for bilateral worker traversal
    # - Prohibit flipud (workers do not walk upside down)
    augmentation_cfg = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 5.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 2.0,
        "perspective": 0.0005,
        "flipud": 0.0,
        "fliplr": 0.5,
        "mosaic": 1.0,
        "mixup": 0.15,
        "copy_paste": 0.1,
    }

    if dry_run:
        print("\n[DRY RUN VERIFICATION ONLY] Configuration validated successfully. Skipping training loop.")
        return {"status": "dry_run_success", "model": str(model_path)}

    model = YOLO(str(model_path))

    run_name = f"intelliwatch_finetune_{time.strftime('%Y%m%d_%H%M%S')}"
    project_dir = PROJECT_ROOT / "models" / "archive"

    print("\nStarting Ultralytics training engine...")
    results = model.train(
        data=str(data_path),
        epochs=epochs,
        batch=batch_size,
        imgsz=imgsz,
        device=resolved_dev,
        workers=workers,
        lr0=learning_rate,
        project=str(project_dir),
        name=run_name,
        optimizer="AdamW",
        val=True,
        save=True,
        save_period=10,
        **augmentation_cfg,
    )

    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print(f"Outputs and checkpoints saved to: {project_dir / run_name}")
    print("=" * 80)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IntelliWatch Reproducible Training Pipeline")
    parser.add_argument("--data", type=str, default="datasets/intelliwatch_ppe/data.yaml", help="Path to data.yaml")
    parser.add_argument("--model", type=str, default="models/ppe/safetyvision_ppe_yolov8n_v1.0.pt", help="Base model weights")
    parser.add_argument("--epochs", type=int, default=5, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=4, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Input image size")
    parser.add_argument("--device", type=str, default="auto", help="Execution device (auto, cpu, 0)")
    parser.add_argument("--dry-run", action="store_true", help="Validate config without training")
    args = parser.parse_args()

    train_model(
        data_yaml=args.data,
        base_model=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        dry_run=args.dry_run,
    )
