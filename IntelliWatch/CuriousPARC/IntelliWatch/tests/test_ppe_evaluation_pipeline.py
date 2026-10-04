"""
tests/test_ppe_evaluation_pipeline.py
=====================================
Automated Regression Tests for IntelliWatch PPE Evaluation, Benchmarking,
Dataset Versioning, Model Metadata, and Telemetry Endpoints.
"""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from scripts.evaluate_ppe_model import compute_iou, evaluate_detection_metrics

PROJECT_ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def client():
    return TestClient(app)


# -------------------------------------------------------------
# 1. Dataset & Metadata Documentation Verification
# -------------------------------------------------------------
def test_dataset_documentation_files_exist():
    """Validates that DATASET_SOURCES.md, DATASET_VERSION.md, and MODEL_CARD.md exist."""
    sources_md = PROJECT_ROOT / "DATASET_SOURCES.md"
    version_md = PROJECT_ROOT / "DATASET_VERSION.md"
    model_card_md = PROJECT_ROOT / "MODEL_CARD.md"

    assert sources_md.exists(), "DATASET_SOURCES.md must exist"
    assert version_md.exists(), "DATASET_VERSION.md must exist"
    assert model_card_md.exists(), "MODEL_CARD.md must exist"

    sources_content = sources_md.read_text(encoding="utf-8")
    assert "Ultralytics" in sources_content or "Roboflow" in sources_content
    assert "CC BY 4.0" in sources_content
    assert "Hardhat" in sources_content
    assert "Safety Vest" in sources_content

    version_content = version_md.read_text(encoding="utf-8")
    assert "intelliwatch_ppe_v1.0.0" in version_content
    assert "data.yaml" in version_content
    assert "Leakage Prevention" in version_content

    model_card = model_card_md.read_text(encoding="utf-8")
    assert "yolo11n_coco_v1.0" in model_card
    assert "safetyvision_ppe_yolov8n_v1.0" in model_card
    assert "RTX 5050" in model_card
    assert "CPU" in model_card


def test_models_metadata_json_schema():
    """Validates models/metadata.json schema and models."""
    meta_path = PROJECT_ROOT / "models" / "metadata.json"
    assert meta_path.exists(), "models/metadata.json must exist"

    with open(meta_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    assert data.get("project") == "IntelliWatch"
    assert "person_detector" in data.get("models", {})
    assert "ppe_detector" in data.get("models", {})

    p_model = data["models"]["person_detector"]
    assert p_model["architecture"] == "YOLO11n"
    assert "weights_file" in p_model

    ppe_model = data["models"]["ppe_detector"]
    assert ppe_model["architecture"] == "YOLOv8n"
    assert "Hardhat" in ppe_model["primary_targets"]
    assert "Safety Vest" in ppe_model["primary_targets"]


# -------------------------------------------------------------
# 2. Dataset Structure & Leakage Prevention Checks
# -------------------------------------------------------------
def test_dataset_directory_structure_and_categories():
    """Validates the split structure and categories A through O."""
    ds_root = PROJECT_ROOT / "datasets" / "intelliwatch_ppe"
    assert ds_root.exists(), "datasets/intelliwatch_ppe/ must exist"

    data_yaml = ds_root / "data.yaml"
    assert data_yaml.exists(), "data.yaml must exist"
    yaml_text = data_yaml.read_text(encoding="utf-8")
    assert "nc: 5" in yaml_text
    assert "person" in yaml_text
    assert "hardhat" in yaml_text
    assert "safety_vest" in yaml_text

    for split in ["train", "val", "test", "hard_cases"]:
        img_dir = ds_root / "images" / split
        lbl_dir = ds_root / "labels" / split
        assert img_dir.exists(), f"images/{split} must exist"
        assert lbl_dir.exists(), f"labels/{split} must exist"

        imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
        lbls = list(lbl_dir.glob("*.txt"))
        assert len(imgs) > 0, f"Split {split} must contain images"
        assert len(lbls) == len(imgs), f"Every image in {split} must have a corresponding label file"

    # Verify categories A through O in test_set_categories.json
    cat_file = ds_root / "test_set_categories.json"
    assert cat_file.exists(), "test_set_categories.json must exist"
    with open(cat_file, "r", encoding="utf-8") as f:
        cat_data = json.load(f)

    all_covered = set()
    for meta in cat_data.values():
        all_covered.update(meta.get("categories", []))

    required_cats = {"A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O"}
    assert required_cats.issubset(all_covered), f"Missing categories from A-O: {required_cats - all_covered}"


# -------------------------------------------------------------
# 3. Evaluation Math & Logic Verification
# -------------------------------------------------------------
def test_compute_iou_accuracy():
    """Validates bounding box IoU calculation accuracy."""
    box_a = [0.0, 0.0, 10.0, 10.0]
    box_b = [0.0, 0.0, 10.0, 10.0]
    assert compute_iou(box_a, box_b) == 1.0

    box_c = [10.0, 10.0, 20.0, 20.0]
    assert compute_iou(box_a, box_c) == 0.0

    box_half = [0.0, 0.0, 10.0, 5.0]  # area 50
    # union of box_a (100) and box_half (50) is 100, inter is 50 -> IoU = 0.5
    assert abs(compute_iou(box_a, box_half) - 0.5) < 1e-4


def test_evaluate_detection_metrics_perfect_match():
    """Validates metrics calculation when predictions perfectly match ground truth."""
    gts = [
        {"image_name": "img1.jpg", "class_name": "person", "bbox": [10.0, 10.0, 50.0, 50.0]},
        {"image_name": "img2.jpg", "class_name": "person", "bbox": [20.0, 20.0, 60.0, 60.0]},
    ]
    preds = [
        {"image_name": "img1.jpg", "class_name": "person", "confidence": 0.95, "bbox": [10.0, 10.0, 50.0, 50.0]},
        {"image_name": "img2.jpg", "class_name": "person", "confidence": 0.90, "bbox": [20.0, 20.0, 60.0, 60.0]},
    ]

    m = evaluate_detection_metrics(preds, gts, "person")
    assert m["precision"] == 1.0
    assert m["recall"] == 1.0
    assert m["f1"] == 1.0
    assert m["tp"] == 2
    assert m["fp"] == 0
    assert m["fn"] == 0
    assert m["map50"] == 1.0


def test_evaluate_detection_metrics_with_false_positive_and_negative():
    """Validates precision and recall drop when predictions diverge from ground truth."""
    gts = [
        {"image_name": "img1.jpg", "class_name": "hardhat", "bbox": [10.0, 10.0, 50.0, 50.0]},
        {"image_name": "img2.jpg", "class_name": "hardhat", "bbox": [20.0, 20.0, 60.0, 60.0]},
    ]
    preds = [
        {"image_name": "img1.jpg", "class_name": "hardhat", "confidence": 0.95, "bbox": [10.0, 10.0, 50.0, 50.0]},  # TP
        {"image_name": "img1.jpg", "class_name": "hardhat", "confidence": 0.85, "bbox": [100.0, 100.0, 150.0, 150.0]},  # FP
        # img2 missed -> FN
    ]

    m = evaluate_detection_metrics(preds, gts, "hardhat")
    assert m["tp"] == 1
    assert m["fp"] == 1
    assert m["fn"] == 1
    assert m["precision"] == 0.5
    assert m["recall"] == 0.5
    assert m["f1"] == 0.5


# -------------------------------------------------------------
# 4. Evaluation Deliverables & Failure Cases Checks
# -------------------------------------------------------------
def test_evaluation_deliverables_exist_and_valid():
    """Validates existence and schema of evaluation deliverables."""
    json_path = PROJECT_ROOT / "reports" / "person_ppe_evaluation.json"
    md_path = PROJECT_ROOT / "reports" / "person_ppe_evaluation.md"
    cm_path = PROJECT_ROOT / "reports" / "confusion_matrix.png"
    pr_path = PROJECT_ROOT / "reports" / "precision_recall_curves.png"
    bench_path = PROJECT_ROOT / "reports" / "model_benchmark.json"

    assert json_path.exists(), "person_ppe_evaluation.json must exist"
    assert md_path.exists(), "person_ppe_evaluation.md must exist"
    assert cm_path.exists(), "confusion_matrix.png must exist"
    assert pr_path.exists(), "precision_recall_curves.png must exist"
    assert bench_path.exists(), "model_benchmark.json must exist"

    with open(json_path, "r", encoding="utf-8") as f:
        ev = json.load(f)

    assert "hardware" in ev
    assert "operational" in ev
    assert "metrics" in ev
    assert "person" in ev["metrics"]
    assert "hardhat" in ev["metrics"]
    assert "safety_vest" in ev["metrics"]
    assert "categories_breakdown" in ev
    assert "failure_cases_count" in ev

    # Check benchmark file
    with open(bench_path, "r", encoding="utf-8") as f:
        bench = json.load(f)
    assert "YOLO11n_Standalone" in bench["candidates"]
    assert "SafetyVision_YOLOv8n_Standalone" in bench["candidates"]
    assert "IntelliWatch_TwoStage" in bench["candidates"]
    assert "YOLOv8x_Industrial_Candidate" in bench["candidates"]
    assert "Untested" in bench["candidates"]["YOLOv8x_Industrial_Candidate"]["status"]


def test_failure_cases_gallery_and_metadata():
    """Validates that reports/failure_cases/ contains visual images and sidecar JSONs."""
    fail_dir = PROJECT_ROOT / "reports" / "failure_cases"
    assert fail_dir.exists(), "reports/failure_cases/ must exist"

    fail_imgs = list(fail_dir.glob("*.jpg"))
    fail_jsons = list(fail_dir.glob("*.json"))

    assert len(fail_imgs) > 0, "Failure cases images must be generated"
    assert len(fail_jsons) == len(fail_imgs), "Every failure image must have a sidecar JSON"

    for j in fail_jsons:
        with open(j, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "failure_id" in data
        assert "image_name" in data
        assert "expected" in data
        assert "predicted" in data
        assert "failure_reasons" in data
        assert "evaluation_context" in data


# -------------------------------------------------------------
# 5. Backend Telemetry & Diagnostic API Endpoints
# -------------------------------------------------------------
def test_system_metrics_endpoint_real_data(client):
    """Validates /api/v1/system/metrics returns empirical metrics without fake formula."""
    res = client.get("/api/v1/system/metrics")
    assert res.status_code == 200
    data = res.json()

    assert data["status"] == "ok"
    assert "cameras_total" in data
    assert "cameras_online" in data
    assert "active_incidents" in data
    assert "compliance_rate" in data
    assert "execution_device" in data
    assert data["execution_device"] in ("cpu", "cuda")
    assert data["compliance_rate"] is not None
    assert 0.0 <= data["compliance_rate"] <= 100.0


def test_model_evaluation_endpoint(client):
    """Validates /api/v1/model/evaluation exposes real held-out test metrics."""
    res = client.get("/api/v1/model/evaluation")
    assert res.status_code == 200
    data = res.json()

    assert "metrics" in data
    assert "person" in data["metrics"]
    assert "hardhat" in data["metrics"]
    assert "safety_vest" in data["metrics"]
    assert data["metrics"]["person"]["precision"] > 0.0
    assert data["metrics"]["hardhat"]["precision"] > 0.0
    assert data["metrics"]["safety_vest"]["precision"] > 0.0


def test_model_benchmark_endpoint(client):
    """Validates /api/v1/model/benchmark exposes model candidate comparisons."""
    res = client.get("/api/v1/model/benchmark")
    assert res.status_code == 200
    data = res.json()

    assert "candidates" in data
    assert "IntelliWatch_TwoStage" in data["candidates"]
    assert data["candidates"]["IntelliWatch_TwoStage"]["status"] == "Evaluated"
    assert "YOLOv8x_Industrial_Candidate" in data["candidates"]
    assert "Untested" in data["candidates"]["YOLOv8x_Industrial_Candidate"]["status"]


def test_model_diagnostics_endpoint(client):
    """Validates /api/v1/model/diagnostics exposes hardware status and active models."""
    res = client.get("/api/v1/model/diagnostics")
    assert res.status_code == 200
    data = res.json()

    assert "device_diagnostics" in data
    assert "resolved_device" in data
    assert "hardware_status" in data
    assert "models_active" in data
    assert data["models_active"]["person"] == "yolo11n_coco_v1.0"
    assert data["models_active"]["ppe"] == "safetyvision_ppe_yolov8n_v1.0"


def test_existing_core_endpoints_remain_functional(client):
    """Regression test ensuring existing endpoints are preserved without breaking changes."""
    # 1. Health
    h_res = client.get("/api/v1/health")
    assert h_res.status_code == 200
    assert h_res.json()["status"] == "ok"

    # 2. Config
    c_res = client.get("/api/v1/config")
    assert c_res.status_code == 200
    assert c_res.json()["project"] == "IntelliWatch"

    # 3. Cameras list
    cam_res = client.get("/api/v1/cameras")
    assert cam_res.status_code == 200

    # 4. Alerts list
    a_res = client.get("/api/v1/alerts")
    assert a_res.status_code == 200

    # 5. Zones list
    z_res = client.get("/api/v1/zones")
    assert z_res.status_code == 200
