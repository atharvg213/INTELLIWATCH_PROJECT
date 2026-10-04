"""
Test configuration loading and default settings.
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from configs.settings import AppSettings, get_settings


def test_default_settings():
    settings = AppSettings()
    assert settings.PROJECT_NAME == "IntelliWatch"
    assert settings.VERSION == "0.1.0"
    assert settings.API_PORT == 8000
    assert settings.API_HOST == "0.0.0.0"
    assert settings.CONFIDENCE_THRESHOLD == 0.50
    assert settings.FPS_SETTINGS == 30
    assert settings.OUTPUT_DIR == "data/output"


def test_settings_singleton():
    s1 = get_settings()
    s2 = get_settings()
    assert s1 is s2


def test_ppe_compliance_settings():
    settings = get_settings()
    assert settings.PPE_COMPLIANCE_ENABLED is True
    assert "Hardhat" in settings.PPE_REQUIRED_CLASSES
    assert "Safety Vest" in settings.PPE_REQUIRED_CLASSES
    assert settings.PPE_VIOLATION_CONFIRMATION_FRAMES == 3
    assert settings.PPE_ASSOCIATION_IOU_THRESHOLD == 0.10
    assert settings.PPE_CENTER_CONTAINMENT_THRESHOLD == 0.50


def test_zone_settings():
    settings = get_settings()
    assert settings.ZONE_DETECTION_ENABLED is True
    assert settings.ZONE_ENTRY_CONFIRMATION_FRAMES == 3
    assert settings.ZONE_MAX_DWELL_SECONDS == 10.0
    assert settings.ZONE_CONFIG_PATH == "configs/zones.json"


def test_depth_settings():
    settings = get_settings()
    assert settings.DEPTH_ENABLED is True
    assert "Depth-Anything-V2-Small" in settings.DEPTH_MODEL_NAME
    assert settings.DEPTH_DEVICE == "cpu"
    assert settings.DEPTH_INPUT_SIZE == 518
    assert settings.DEPTH_OUTPUT_MODE == "relative"

