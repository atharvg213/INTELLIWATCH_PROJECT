"""
backend/services/scene_store.py
In-memory singleton storage for the most recently evaluated FrameScene.
Provides quick access for real-time inspection endpoints.
"""
from typing import Optional
from backend.schemas.scene_graph import FrameScene, SituationalSummary


class SceneStore:
    """Thread-safe / asynchronous lightweight in-memory cache for current scene graph."""

    def __init__(self):
        self._current_scene: Optional[FrameScene] = None

    def set_current_scene(self, scene: FrameScene) -> None:
        self._current_scene = scene

    def get_current_scene(self) -> FrameScene:
        if self._current_scene is not None:
            return self._current_scene
        # Return a valid empty placeholder scene if no frames have been processed yet
        return FrameScene(
            frame_id=0,
            timestamp=0.0,
            camera_id="unassigned",
            source_camera_id=None,
            nodes=[],
            relationships=[],
            summary=SituationalSummary(),
            active_zones=[],
            active_behaviors={},
        )

    def reset(self) -> None:
        self._current_scene = None


_scene_store_instance = SceneStore()


def get_scene_store() -> SceneStore:
    return _scene_store_instance
