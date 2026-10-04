"""
intelligence/scene_graph/scene_graph.py
Abstract base class and module entry points for IntelliWatch scene graphs.
Preserves backwards compatibility with early architectural interfaces.
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional
from backend.schemas.tracking import FrameTracks
from backend.schemas.scene_graph import FrameScene
from intelligence.scene_graph.graph import SceneGraph
from intelligence.scene_graph.builder import SceneGraphBuilder


class BaseSceneGraph(ABC):
    """
    Abstract Base Class for entity-relationship spatial graphs.
    Models relationships such as:
    - [Worker] is NEAR [Machinery]
    - [Forklift] is APPROACHING [Pedestrian]
    - [Worker] is INSIDE [HazardousZone]
    """

    @abstractmethod
    def build_graph(
        self,
        tracks: FrameTracks,
        spatial_data: Optional[Any] = None
    ) -> Dict[str, Any]:
        """Construct structured scene graph representation from tracks."""
        pass


__all__ = [
    "BaseSceneGraph",
    "SceneGraph",
    "SceneGraphBuilder",
]
