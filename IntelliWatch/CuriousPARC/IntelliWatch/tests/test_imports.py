"""
Test module import integrity across all project layers.
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def test_config_imports():
    from configs.settings import AppSettings, get_settings
    from configs.logging_config import setup_logging
    assert AppSettings is not None
    assert get_settings is not None
    assert setup_logging is not None


def test_vision_interface_imports():
    from vision.detection.base import BaseDetector
    from vision.detection.yolo_detector import YOLODetector
    from vision.tracking.base import BaseTracker
    from vision.tracking.bytetrack_tracker import ByteTrackTracker
    from vision.depth.base import BaseDepthEstimator
    from vision.segmentation.base import BaseSegmenter
    from vision.pose.base import BasePoseEstimator
    from vision.preprocessing.frame_processor import BaseFramePreprocessor, FramePreprocessor

    assert issubclass(YOLODetector, BaseDetector)
    assert issubclass(ByteTrackTracker, BaseTracker)
    assert issubclass(FramePreprocessor, BaseFramePreprocessor)


def test_intelligence_interface_imports():
    from intelligence.zones.zone_monitor import BaseZoneMonitor
    from intelligence.behavior.behavior_analyzer import BaseBehaviorAnalyzer
    from intelligence.scene_graph.scene_graph import BaseSceneGraph
    from intelligence.events.event_detector import BaseEventDetector
    from intelligence.events.ppe_compliance import PPEComplianceEngine
    from intelligence.risk.risk_engine import BaseRiskEngine
    from intelligence.prediction.trajectory_predictor import BasePredictor
    from vision.detection.ppe_association import PPEAssociationEngine

    assert BaseZoneMonitor is not None
    assert BaseBehaviorAnalyzer is not None
    assert BaseSceneGraph is not None
    assert BaseEventDetector is not None
    assert issubclass(PPEComplianceEngine, BaseEventDetector)
    assert PPEAssociationEngine is not None
    assert BaseRiskEngine is not None
    assert BasePredictor is not None



def test_backend_imports():
    from backend.main import app
    from backend.api.routes import router
    from backend.services.pipeline_manager import PipelineManager

    assert app is not None
    assert router is not None
    assert PipelineManager is not None
