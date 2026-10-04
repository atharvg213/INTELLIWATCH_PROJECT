"""
intelligence/scene_graph/graph.py
SceneGraph representation and high-level query interface.
Provides relational querying, neighbor traversal, and conversion to/from FrameScene.
"""
import json
from typing import Any, Dict, List, Optional, Tuple

from backend.schemas.scene_graph import (
    FrameScene,
    RelationLifecycle,
    SceneNode,
    SceneNodeType,
    SceneRelation,
    SceneRelationType,
    SituationalSummary,
)


class SceneGraph:
    """
    In-memory graph of nodes and directed edges for a specific frame or scenario.
    Provides fast O(1) node lookup and indexed relational queries.
    """

    def __init__(
        self,
        nodes: Optional[List[SceneNode]] = None,
        relationships: Optional[List[SceneRelation]] = None,
    ):
        self._nodes: Dict[str, SceneNode] = {}
        self._outgoing: Dict[str, List[SceneRelation]] = {}
        self._incoming: Dict[str, List[SceneRelation]] = {}
        self._relationships: List[SceneRelation] = []

        if nodes:
            for n in nodes:
                self.add_node(n)
        if relationships:
            for r in relationships:
                self.add_relation(r)

    def add_node(self, node: SceneNode) -> None:
        """Adds or updates a node in the graph."""
        self._nodes[node.node_id] = node
        if node.node_id not in self._outgoing:
            self._outgoing[node.node_id] = []
        if node.node_id not in self._incoming:
            self._incoming[node.node_id] = []

    def add_relation(self, relation: SceneRelation) -> None:
        """Adds a directed relation edge to the graph."""
        self._relationships.append(relation)
        src = relation.source_node_id
        dst = relation.target_node_id

        if src not in self._outgoing:
            self._outgoing[src] = []
        self._outgoing[src].append(relation)

        if dst not in self._incoming:
            self._incoming[dst] = []
        self._incoming[dst].append(relation)

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        """Lookup node by unique identifier."""
        return self._nodes.get(node_id)

    @property
    def nodes(self) -> List[SceneNode]:
        """Returns all nodes in the graph."""
        return list(self._nodes.values())

    @property
    def relationships(self) -> List[SceneRelation]:
        """Returns all relationship edges in the graph."""
        return list(self._relationships)

    def get_nodes_by_type(self, node_type: SceneNodeType) -> List[SceneNode]:
        """Filter nodes by semantic type (e.g. PERSON, VEHICLE, ZONE)."""
        return [n for n in self._nodes.values() if n.node_type == node_type]

    def get_relations_for_node(self, node_id: str) -> List[SceneRelation]:
        """Returns all edges where node_id is either source or target."""
        outgoing = self._outgoing.get(node_id, [])
        incoming = self._incoming.get(node_id, [])
        # Combine preserving order without duplicates
        seen_ids = set()
        res = []
        for r in outgoing + incoming:
            if r.relation_id not in seen_ids:
                seen_ids.add(r.relation_id)
                res.append(r)
        return res

    def get_relations_by_type(
        self,
        relation_type: SceneRelationType,
        active_only: bool = True,
    ) -> List[SceneRelation]:
        """Filter relations by category."""
        return [
            r for r in self._relationships
            if r.relation_type == relation_type
            and (not active_only or r.lifecycle in (RelationLifecycle.ACTIVE, RelationLifecycle.CREATED))
        ]

    def get_relations_between(self, source_id: str, target_id: str) -> List[SceneRelation]:
        """Returns directed edges from source to target."""
        return [r for r in self._outgoing.get(source_id, []) if r.target_node_id == target_id]

    def get_neighbors(self, node_id: str) -> List[SceneNode]:
        """Returns all unique nodes directly connected via incoming or outgoing edges."""
        neighbor_ids = set()
        for r in self._outgoing.get(node_id, []):
            neighbor_ids.add(r.target_node_id)
        for r in self._incoming.get(node_id, []):
            neighbor_ids.add(r.source_node_id)
        neighbor_ids.discard(node_id)
        return [self._nodes[nid] for nid in neighbor_ids if nid in self._nodes]

    def find_workers_in_zone(self, zone_id: str) -> List[SceneNode]:
        """Convenience query: find all PERSON nodes confirmed INSIDE a given zone."""
        zone_node_id = zone_id if zone_id.startswith("zone_") else f"zone_{zone_id}"
        incoming = self._incoming.get(zone_node_id, [])
        worker_ids = [
            r.source_node_id for r in incoming
            if r.relation_type == SceneRelationType.INSIDE
            and r.lifecycle in (RelationLifecycle.ACTIVE, RelationLifecycle.CREATED)
        ]
        return [self._nodes[wid] for wid in worker_ids if wid in self._nodes and self._nodes[wid].node_type == SceneNodeType.PERSON]

    def find_entities_near(
        self,
        node_id: str,
        max_distance: Optional[float] = None,
    ) -> List[Tuple[SceneNode, float]]:
        """
        Finds entities connected by a NEAR relationship to the given node.
        Returns list of (SceneNode, distance_px) tuples sorted by distance.
        """
        results: List[Tuple[SceneNode, float]] = []
        rels = self.get_relations_for_node(node_id)
        for r in rels:
            if r.relation_type == SceneRelationType.NEAR and r.lifecycle in (RelationLifecycle.ACTIVE, RelationLifecycle.CREATED):
                other_id = r.target_node_id if r.source_node_id == node_id else r.source_node_id
                other_node = self._nodes.get(other_id)
                if other_node:
                    dist = float(r.evidence.get("distance_px", 0.0))
                    if max_distance is None or dist <= max_distance:
                        results.append((other_node, dist))
        results.sort(key=lambda x: x[1])
        return results

    def to_frame_scene(
        self,
        frame_id: Optional[int],
        timestamp: float,
        camera_id: str = "cam_01",
        summary: Optional[SituationalSummary] = None,
        active_zones: Optional[List[str]] = None,
        active_behaviors: Optional[Dict[str, str]] = None,
    ) -> FrameScene:
        """Exports this graph state to a serializable FrameScene schema."""
        return FrameScene(
            frame_id=frame_id,
            timestamp=timestamp,
            camera_id=camera_id,
            nodes=self.nodes,
            relationships=self.relationships,
            summary=summary or SituationalSummary(),
            active_zones=active_zones or [],
            active_behaviors=active_behaviors or {},
        )

    @classmethod
    def from_frame_scene(cls, scene: FrameScene) -> "SceneGraph":
        """Instantiates a SceneGraph directly from a FrameScene schema object."""
        return cls(nodes=scene.nodes, relationships=scene.relationships)

    def to_dict(self) -> Dict[str, Any]:
        """Serializes nodes and relationships to plain Python dictionary."""
        return {
            "nodes": [n.model_dump() for n in self.nodes],
            "relationships": [r.model_dump() for r in self.relationships],
        }

    def to_json(self, indent: Optional[int] = None) -> str:
        """Serializes graph to valid JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
