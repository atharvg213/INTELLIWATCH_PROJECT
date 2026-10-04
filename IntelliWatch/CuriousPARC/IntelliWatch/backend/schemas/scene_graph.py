"""
backend/schemas/scene_graph.py
Step 9 - Scene Graph and Situational Awareness schemas.
Represents factual relationships between people, machinery, objects, zones, PPE, and behavior.
"""
from enum import Enum
import logging
import math
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field

from backend.schemas.detection import BoundingBox

logger = logging.getLogger("intelliwatch.scene_graph")


class SceneNodeType(str, Enum):
    """Semantic classification of a scene graph node."""
    PERSON = "PERSON"
    VEHICLE = "VEHICLE"
    MACHINE = "MACHINE"
    OBJECT = "OBJECT"
    PPE = "PPE"
    ZONE = "ZONE"
    OTHER = "OTHER"


class SceneRelationType(str, Enum):
    """Semantic relationship between two scene graph nodes."""
    WEARING = "WEARING"                   # Person -> PPE
    MISSING = "MISSING"                   # Person -> Missing PPE item
    INSIDE = "INSIDE"                     # Entity -> Zone
    OUTSIDE = "OUTSIDE"                   # Entity -> Zone
    BEHAVIOR = "BEHAVIOR"                 # Entity -> Behavior representation
    NEAR = "NEAR"                         # Entity <-> Entity spatial proximity
    FAR = "FAR"                           # Entity <-> Entity distant
    APPROACHING = "APPROACHING"           # Entity -> Entity distance decreasing over time
    MOVING_AWAY = "MOVING_AWAY"           # Entity -> Entity distance increasing over time
    ASSOCIATED_WITH = "ASSOCIATED_WITH"   # General association
    INTERACTING_WITH = "INTERACTING_WITH" # Physical or operational interaction


class RelationLifecycle(str, Enum):
    """Lifecycle phase of a tracked relationship across consecutive frames."""
    CREATED = "CREATED"   # Newly confirmed in the current frame
    ACTIVE = "ACTIVE"     # Confirmed and persisting across frames
    ENDED = "ENDED"       # Ended in the current frame (condition ceased or actor departed)


class SceneNode(BaseModel):
    """
    Structured entity node in the industrial scene graph.
    Represents tracked entities, physical equipment, PPE, or spatial zones.
    """
    node_id: str = Field(..., description="Unique alphanumeric identifier (e.g. 'person_17', 'zone_high_voltage')")
    node_type: SceneNodeType = Field(..., description="Semantic category of the node")
    class_name: str = Field(..., description="Detector or category label (e.g. 'person', 'forklift', 'Hardhat')")
    track_id: Optional[int] = Field(default=None, description="ByteTrack persistent identifier for tracked entities")
    confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0, description="Model detection confidence if applicable")
    bbox: Optional[BoundingBox] = Field(default=None, description="2D bounding box in original image pixels")
    centroid: Optional[Tuple[float, float]] = Field(default=None, description="(cx, cy) pixel coordinates")
    contact_point: Optional[Tuple[float, float]] = Field(default=None, description="Physical floor contact point (feet/base)")
    zone_id: Optional[str] = Field(default=None, description="Zone identifier if node represents a zone")
    attributes: Dict[str, Any] = Field(default_factory=dict, description="Arbitrary verifiable properties (e.g. depth, compliance)")
    timestamp: float = Field(default=0.0, description="Video timestamp in seconds")


class SceneRelation(BaseModel):
    """
    Directed factual relationship between two scene nodes with evidence and lifecycle state.
    """
    relation_id: str = Field(..., description="Unique deterministic identifier for this relationship instance")
    source_node_id: str = Field(..., description="Origin node ID (e.g. 'person_17')")
    target_node_id: str = Field(..., description="Destination node ID (e.g. 'zone_high_voltage' or 'forklift_2')")
    relation_type: SceneRelationType = Field(..., description="Semantic relationship category")
    lifecycle: RelationLifecycle = Field(default=RelationLifecycle.ACTIVE, description="Current lifecycle state")
    confidence: float = Field(default=1.0, ge=0.0, le=1.0, description="Explainable evidence confidence score")
    timestamp: float = Field(default=0.0, description="Timestamp of the current observation")
    first_seen_timestamp: Optional[float] = Field(default=None, description="Timestamp when relation was first observed")
    last_seen_timestamp: Optional[float] = Field(default=None, description="Timestamp when relation was most recently observed")
    duration_seconds: float = Field(default=0.0, description="Elapsed seconds this relationship has persisted")
    evidence: Dict[str, Any] = Field(
        default_factory=dict,
        description="Factual evidence supporting this relation (e.g. distance_px, contact_point, score)",
    )
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional contextual metadata")

    # Canonical aliases for API uniformity
    id: Optional[str] = Field(default=None, description="Direct alias for relation_id")
    source_id: Optional[str] = Field(default=None, description="Direct alias for source_node_id")
    target_id: Optional[str] = Field(default=None, description="Direct alias for target_node_id")
    type: Optional[str] = Field(default=None, description="Direct alias for relation_type string value")
    state: Optional[str] = Field(default=None, description="Direct alias for lifecycle state string value")
    source: Optional[str] = Field(default=None, description="Canonical source node ID")
    target: Optional[str] = Field(default=None, description="Canonical target node ID")
    relationship: Optional[str] = Field(default=None, description="Canonical relationship string")
    active: bool = Field(default=True, description="True if relationship is currently active")
    category: Optional[str] = Field(default=None, description="Category: SPATIAL, TEMPORAL, SAFETY, EQUIPMENT")
    tth_seconds: Optional[float] = Field(default=None, description="Deterministic estimated time to hazard in seconds")
    tth_status: Optional[str] = Field(default=None, description="Technical availability or reason for TTH status")
    distance_px: Optional[float] = Field(default=None, description="Image-space distance in pixels")
    distance_m: Optional[float] = Field(default=None, description="Calibrated approximate ground-plane distance in meters")
    closing_rate_px_s: Optional[float] = Field(default=None, description="Image-space closing rate in pixels per second")
    closing_rate_m_s: Optional[float] = Field(default=None, description="Calibrated approximate ground-plane closing rate in meters per second")
    spatial_basis: str = Field(default="IMAGE_SPACE", description="Basis for spatial reasoning: 'IMAGE_SPACE' or 'GROUND_PLANE_APPROXIMATION'")

    def model_post_init(self, __context: Any) -> None:
        if self.id is None:
            self.id = self.relation_id
        if self.source_id is None:
            self.source_id = self.source_node_id
        if self.target_id is None:
            self.target_id = self.target_node_id
        if self.type is None:
            self.type = self.relation_type.value if hasattr(self.relation_type, "value") else str(self.relation_type)
        if self.state is None:
            self.state = self.lifecycle.value if hasattr(self.lifecycle, "value") else str(self.lifecycle)
        if self.source is None:
            self.source = self.source_id
        if self.target is None:
            self.target = self.target_id
        if self.relationship is None:
            self.relationship = self.type
        self.active = str(self.state).upper() in ("ACTIVE", "CREATED")

        r_upper = str(self.type).upper()
        if r_upper in ("IS_WEARING", "WEARING", "MISSING", "NO_HARDHAT", "NO_SAFETY_VEST", "NO_MASK", "NO_GLOVES"):
            self.category = "EQUIPMENT"
        elif r_upper in ("APPROACHING", "MOVING_AWAY", "TRAJECTORY_CONVERGING", "CLOSER_THAN"):
            self.category = "TEMPORAL"
        elif r_upper in ("INSIDE", "INSIDE_ZONE", "ENTERED", "OCCUPIED_ZONE", "DWELLING"):
            self.category = "SAFETY"
        elif r_upper in ("NEAR", "FAR", "PROXIMITY", "ADJACENT"):
            self.category = "SPATIAL"
        else:
            self.category = "SAFETY"


class SituationalSummary(BaseModel):
    """
    Machine-readable aggregate summary of current scene status.
    """
    workers_count: int = Field(default=0, description="Count of active tracked persons in scene")
    vehicles_count: int = Field(default=0, description="Count of active tracked vehicles (forklifts, trucks)")
    machines_count: int = Field(default=0, description="Count of active machinery entities")
    objects_count: int = Field(default=0, description="Count of general objects")
    active_zones_count: int = Field(default=0, description="Count of configured and active restricted zones")
    occupied_zones_count: int = Field(default=0, description="Count of restricted zones currently occupied by workers")
    workers_in_restricted_zones: int = Field(default=0, description="Count of workers confirmed inside any restricted zone")
    workers_non_compliant_ppe: int = Field(default=0, description="Count of workers missing mandatory PPE")
    rapid_movement_count: int = Field(default=0, description="Count of entities exhibiting rapid movement")
    stationary_loitering_count: int = Field(default=0, description="Count of entities exhibiting prolonged stationarity")
    possible_fall_count: int = Field(default=0, description="Count of entities flagged with possible fall heuristic")
    active_proximity_relationships: int = Field(default=0, description="Count of active NEAR spatial relationships")
    active_approaching_relationships: int = Field(default=0, description="Count of active APPROACHING relationships")
    total_active_relationships: int = Field(default=0, description="Total count of active edges in the scene graph")


class FrameScene(BaseModel):
    """
    Complete structured situational representation of a single video frame.
    """
    scene_id: Optional[str] = Field(default=None, description="Unique scene or job identifier")
    frame_id: Optional[int] = Field(default=None, description="Monotonically increasing frame sequence counter")
    timestamp: float = Field(..., description="Video stream elapsed time in seconds")
    camera_id: str = Field(default="cam_01", description="Identifier of the video capture source")
    source_camera_id: Optional[str] = Field(default=None, description="Camera identity assigned to the media that produced this scene")
    nodes: List[SceneNode] = Field(default_factory=list, description="All entity and spatial nodes in this frame")
    relationships: List[SceneRelation] = Field(default_factory=list, description="All active and newly ended relationships")
    summary: SituationalSummary = Field(default_factory=SituationalSummary, description="High-level machine-readable summary")
    active_zones: List[str] = Field(default_factory=list, description="List of zone IDs evaluated in this scene")
    active_behaviors: Dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of node_id -> confirmed primary behavior string",
    )
    entities: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Canonical rich entity representation for Hero Scene Graph visualization",
    )
    temporal_predictions: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Deterministic temporal hazard predictions and conflict timelines",
    )
    time_to_hazard: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Hero primary time-to-hazard assessment and conflict forecast",
    )
    risk_events: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Active confirmed safety events with explainable compounding factors",
    )
    is_live: bool = Field(default=False, description="True if scene graph represents live video streaming")
    is_video: bool = Field(default=False, description="True if scene graph was generated from video stream or job")
    source_mode: str = Field(default="LATEST SCENE SNAPSHOT", description="'LIVE' or 'LATEST SCENE SNAPSHOT'")
    calibration_status: Optional[str] = Field(default="UNCONFIGURED", description="Camera calibration status ('CALIBRATED', 'UNCONFIGURED', 'UNAVAILABLE', 'INVALID', 'DEGENERATE', 'DISABLED')")
    spatial_basis: Optional[str] = Field(default="IMAGE_SPACE", description="Overall spatial reasoning basis ('IMAGE_SPACE' or 'GROUND_PLANE_APPROXIMATION')")

    def get_node(self, node_id: str) -> Optional[SceneNode]:
        """Lookup node by its unique node_id."""
        for n in self.nodes:
            if n.node_id == node_id:
                return n
        return None

    def get_relations_for_node(self, node_id: str) -> List[SceneRelation]:
        """Find all relationships where the node is either source or target."""
        return [r for r in self.relationships if r.source_node_id == node_id or r.target_node_id == node_id]

    def get_relations_between(self, source_id: str, target_id: str) -> List[SceneRelation]:
        """Find relationships directed from source to target."""
        return [r for r in self.relationships if r.source_node_id == source_id and r.target_node_id == target_id]


def resolve_scene_source_camera_id(scene: Any) -> Optional[str]:
    """Resolve persisted camera identity, keeping an explicit unassigned value authoritative."""
    missing = object()
    source_camera_id = getattr(scene, "source_camera_id", missing)
    fields_set = getattr(scene, "model_fields_set", None)
    if fields_set is None:
        fields_set = getattr(scene, "__fields_set__", None)

    source_was_set = source_camera_id is not missing and (
        source_camera_id is not None
        or fields_set is None
        or "source_camera_id" in fields_set
    )
    if source_was_set:
        candidate = source_camera_id
    else:
        # Compatibility for legacy scenes that explicitly stored camera_id but
        # predate source_camera_id. Never use a schema default such as cam_01.
        if fields_set is not None and "camera_id" not in fields_set:
            return None
        candidate = getattr(scene, "camera_id", None)

    if not isinstance(candidate, str):
        return None
    candidate = candidate.strip()
    if not candidate or candidate.lower() in ("unassigned", "unknown"):
        return None
    return candidate


def _image_velocity_vector_px_s(attributes: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    """Return a directed image velocity only when both pixel/second components exist."""
    if attributes.get("motion_observed") is False:
        return None
    vx = attributes.get("velocity_x_px_per_s")
    vy = attributes.get("velocity_y_px_per_s")
    if vx is None or vy is None:
        return None
    try:
        vx, vy = float(vx), float(vy)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(vx) or not math.isfinite(vy):
        return None
    return vx, vy


def enrich_scene_graph(
    scene: FrameScene,
    risk_assessment: Optional[Any] = None,
    is_live: bool = False,
    source_mode: Optional[str] = None,
    prediction_assessment: Optional[Any] = None,
    camera_id: Optional[str] = None,
) -> FrameScene:
    """
    Enriches a FrameScene with a canonical entities list and standard relationship
    fields for the Hero Scene Graph experience, combining spatial perception with
    factual risk factors, behaviors, zone memberships, and deterministic time-to-hazard
    predictions without fabricating data.
    """
    mode_str = str(source_mode or ("LIVE" if is_live else "LATEST SCENE SNAPSHOT")).upper()
    is_video = is_live or getattr(scene, "is_video", False) or "VIDEO" in mode_str
    is_single_image = not is_video

    # 0. Resolve the scene's source identity before looking up calibration.
    calibrator = None
    cal_status = "UNCONFIGURED"
    scene_camera_id = resolve_scene_source_camera_id(scene)

    requested_camera_id = camera_id.strip() if isinstance(camera_id, str) and camera_id.strip() else None
    if requested_camera_id and scene_camera_id and requested_camera_id != scene_camera_id:
        logger.warning(
            "Ignoring camera_id override [%s] for scene source [%s]",
            requested_camera_id,
            scene_camera_id,
        )

    scene.camera_id = scene_camera_id or "unassigned"
    scene.source_camera_id = scene_camera_id
    if scene_camera_id is None:
        cal_status = "UNAVAILABLE"
    invalid_calibrator = False
    disabled_calibrator = False
    try:
        if scene_camera_id is not None:
            from backend.services.calibration_service import get_calibration_service
            cal_svc = get_calibration_service()
            calibrator = cal_svc.get_calibrator(scene_camera_id)
            cfg = cal_svc.get_calibration(scene_camera_id)
            if calibrator is not None and getattr(calibrator, "camera_id", scene_camera_id) != scene_camera_id:
                calibrator = None
            if cfg is not None and getattr(cfg, "camera_id", scene_camera_id) != scene_camera_id:
                cfg = None

            if calibrator is not None and not getattr(calibrator, "is_valid", True):
                invalid_calibrator = True
                calibrator = None
            if calibrator is not None and not getattr(calibrator, "enabled", True):
                disabled_calibrator = True
                calibrator = None

            if calibrator is not None:
                cal_status = "CALIBRATED"
            elif cfg is None:
                cal_status = "INVALID" if invalid_calibrator else ("DISABLED" if disabled_calibrator else "UNCONFIGURED")
            elif not getattr(cfg, "calibration_enabled", getattr(cfg, "enabled", True)):
                cal_status = "DISABLED"
            else:
                cfg_status = getattr(cfg, "calibration_status", None)
                cfg_status = str(cfg_status.value if hasattr(cfg_status, "value") else cfg_status or "INVALID")
                cal_status = cfg_status if cfg_status in ("INVALID", "DEGENERATE", "DISABLED") else "INVALID"
    except Exception as e:
        calibrator = None
        cal_status = "UNAVAILABLE"
        logger.debug(f"Failed to lookup camera calibrator: {e}")

    scene.calibration_status = cal_status
    scene.spatial_basis = "GROUND_PLANE_APPROXIMATION" if calibrator else "IMAGE_SPACE"

    # 1. Update relationships with aliases, ground-plane distance/closing rates, and deterministic TTH
    for rel in scene.relationships:
        if not rel.source_id:
            rel.source_id = rel.source_node_id
        if not rel.target_id:
            rel.target_id = rel.target_node_id
        if not rel.type:
            rel.type = rel.relation_type.value if hasattr(rel.relation_type, "value") else str(rel.relation_type)
        if not rel.state:
            rel.state = rel.lifecycle.value if hasattr(rel.lifecycle, "value") else str(rel.lifecycle)

        meta = rel.metadata or {}
        ev = rel.evidence or {}
        dist_px = meta.get("distance_px") if meta.get("distance_px") is not None else ev.get("distance_px")
        # Only fields that declare pixel/second units may feed the image-space TTH formula.
        rel_spd = ev.get("closing_rate_px_s") if ev.get("closing_rate_px_s") is not None else ev.get("closing_speed_px_per_s")
        try:
            dist_px_value = float(dist_px) if dist_px is not None else None
            rel_spd_value = float(rel_spd) if rel_spd is not None else None
        except (TypeError, ValueError):
            dist_px_value = None
            rel_spd_value = None
        if dist_px_value is not None and not math.isfinite(dist_px_value):
            dist_px_value = None
        if rel_spd_value is not None and not math.isfinite(rel_spd_value):
            rel_spd_value = None

        # Ground plane real-world measurements
        dist_m = None
        closing_rate_m_s = None
        rel_spatial_basis = "IMAGE_SPACE"

        if calibrator:
            src_node_obj = scene.get_node(rel.source_id)
            tgt_node_obj = scene.get_node(rel.target_id)
            if src_node_obj and tgt_node_obj and src_node_obj.contact_point and tgt_node_obj.contact_point:
                ground_types = (SceneNodeType.PERSON, SceneNodeType.VEHICLE, SceneNodeType.MACHINE, SceneNodeType.OBJECT)
                if src_node_obj.node_type not in ground_types or tgt_node_obj.node_type not in ground_types:
                    src_node_obj = None
                    tgt_node_obj = None
            if src_node_obj and tgt_node_obj and src_node_obj.contact_point and tgt_node_obj.contact_point:
                dist_m = calibrator.compute_ground_distance(src_node_obj.contact_point, tgt_node_obj.contact_point)
                if dist_m is not None:
                    rel_spatial_basis = "GROUND_PLANE_APPROXIMATION"
                    rel_type_name = str(rel.type or rel.relation_type.value).upper()
                    if rel_type_name in ("APPROACHING", "CLOSER_THAN", "TRAJECTORY_CONVERGING") or rel.category == "TEMPORAL":
                        src_velocity = _image_velocity_vector_px_s(src_node_obj.attributes or {})
                        tgt_velocity = _image_velocity_vector_px_s(tgt_node_obj.attributes or {})
                        if src_velocity is not None and tgt_velocity is not None:
                            closing_rate_m_s = calibrator.compute_ground_closing_rate(
                                src_node_obj.contact_point,
                                src_velocity,
                                tgt_node_obj.contact_point,
                                tgt_velocity,
                            )

        rel.distance_px = dist_px_value
        rel.distance_m = dist_m
        rel.closing_rate_px_s = rel_spd_value
        rel.closing_rate_m_s = closing_rate_m_s
        rel.spatial_basis = rel_spatial_basis

        ev["distance_px"] = rel.distance_px
        ev["closing_rate_px_s"] = rel.closing_rate_px_s
        ev["distance_m"] = dist_m
        ev["closing_rate_m_s"] = closing_rate_m_s
        ev["spatial_basis"] = rel_spatial_basis
        rel.evidence = ev

        # TTH stays an image-space estimate: the 60px contact boundary and the
        # closing rate are both pixel-space quantities, even with calibration.
        if is_single_image:
            rel.tth_seconds = None
            rel.tth_status = "Unavailable — single-frame analysis has no motion vectors"
        else:
            rel_t = str(rel.type).upper()
            if rel_t in ("APPROACHING", "CLOSER_THAN", "TRAJECTORY_CONVERGING") or rel.category == "TEMPORAL":
                if rel_spd_value is not None and rel_spd_value > 2.0 and dist_px_value is not None:
                    # Deterministic TTH: (distance - 60px contact boundary) / closing velocity
                    tth_val = round(max(0.0, dist_px_value - 60.0) / rel_spd_value, 1)
                    rel.tth_seconds = tth_val
                    rel.tth_status = f"Image-space estimate: approximately {tth_val}s (pixels and pixels/second)"
                elif dist_px_value is not None:
                    rel.tth_seconds = None
                    rel.tth_status = "Unavailable — closing velocity below motion threshold"
                else:
                    rel.tth_seconds = None
                    rel.tth_status = "Unavailable — insufficient spatial distance"
            else:
                rel.tth_seconds = None
                rel.tth_status = "Not applicable for static/equipment relationship"

    # 2. Map zones and occupant IDs
    zone_occupants: Dict[str, List[str]] = {}
    for rel in scene.relationships:
        rtype = rel.relation_type.value if hasattr(rel.relation_type, "value") else str(rel.relation_type)
        if rtype in ("INSIDE", "INSIDE_ZONE") and rel.lifecycle in (
            RelationLifecycle.ACTIVE,
            "ACTIVE",
            RelationLifecycle.CREATED,
            "CREATED",
        ):
            zone_occupants.setdefault(rel.target_node_id, []).append(rel.source_node_id)

    # 3. Extract active events and factors from risk assessment
    active_events = getattr(risk_assessment, "active_events", []) if risk_assessment else []
    risk_factors = getattr(risk_assessment, "risk_factors", []) if risk_assessment else []
    risk_rank = {"INFO": 0, "LOW": 1, "MEDIUM": 2, "HIGH": 3, "CRITICAL": 4}

    # 4. Construct canonical entities list
    entities: List[Dict[str, Any]] = []

    for node in scene.nodes:
        node_id = node.node_id
        ntype = node.node_type.value if hasattr(node.node_type, "value") else str(node.node_type)

        # Human-readable label
        if node.node_type == SceneNodeType.PERSON:
            label = f"Person #{node.track_id}" if node.track_id is not None else f"Person ({node_id})"
        elif node.node_type == SceneNodeType.VEHICLE:
            label = f"{node.class_name.capitalize()} #{node.track_id}" if node.track_id is not None else node.class_name.capitalize()
        elif node.node_type == SceneNodeType.ZONE:
            label = node.attributes.get("name") or (f"Zone {node.zone_id}" if node.zone_id else "Restricted Zone")
        elif node.node_type == SceneNodeType.PPE:
            label = node.class_name
        else:
            label = f"{node.class_name.capitalize()} #{node.track_id}" if node.track_id is not None else node.class_name.capitalize()

        attrs = node.attributes or {}
        raw_speed = attrs.get("speed_px_per_s")
        try:
            raw_speed = float(raw_speed) if raw_speed is not None else None
        except (TypeError, ValueError):
            raw_speed = None
        if raw_speed is not None and (not math.isfinite(raw_speed) or raw_speed < 0.0):
            raw_speed = None

        # A new track's initial 0 px/s is not a measured stationary observation.
        if is_single_image:
            speed_val = None
            velocity_status = "Not available for single-frame analysis"
        elif raw_speed is None:
            speed_val = None
            velocity_status = "Unavailable — image-space speed was not reported"
        elif raw_speed == 0.0 and attrs.get("motion_observed") is not True:
            speed_val = None
            velocity_status = "Unavailable — waiting for a second tracked frame"
        else:
            speed_val = raw_speed
            velocity_status = f"{round(raw_speed, 1)} px/s" if raw_speed is not None else "Unavailable"

        state_dict: Dict[str, Any] = {
            "primary_behavior": attrs.get("primary_behavior"),
            "stationary_duration_s": attrs.get("stationary_duration_s"),
            "speed_px_per_s": speed_val,
            "velocity_status": velocity_status,
            "relative_depth": attrs.get("relative_depth"),
            "compliance_status": attrs.get("compliance_status"),
            "ppe_status": attrs.get("ppe_status"),
            "flags": attrs.get("flags"),
        }

        # Zone spatial interpretation (Requirement 11)
        if node.node_type == SceneNodeType.ZONE:
            state_dict["zone_name"] = attrs.get("name")
            state_dict["zone_type"] = attrs.get("zone_type")
            state_dict["max_dwell_seconds"] = attrs.get("max_dwell_seconds")
            occupants = zone_occupants.get(node_id, [])
            state_dict["occupants"] = occupants
            state_dict["workers_inside_count"] = len(occupants)
            state_dict["spatial_basis"] = "IMAGE_SPACE"
            state_dict["calibration_note"] = "Zone perimeter remains defined in image-space pixels."

        # Factual Risk Engine integration
        entity_risk_level = "INFO"
        entity_risk_score = 0.0
        entity_factors: List[str] = []
        entity_reasons: List[str] = []
        entity_events: List[Dict[str, Any]] = []

        track_str = f"person_{node.track_id}" if node.track_id is not None else None

        for ev in active_events:
            involved = getattr(ev, "involved_entities", []) or []
            if node_id in involved or (track_str and track_str in involved) or (node.track_id is not None and str(node.track_id) in involved):
                ev_level = ev.risk_level.value if hasattr(ev.risk_level, "value") else str(ev.risk_level)
                ev_type = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
                ev_score = float(ev.risk_score) if ev.risk_score is not None else 0.0

                entity_events.append({
                    "event_id": ev.event_id,
                    "event_type": ev_type,
                    "risk_level": ev_level,
                    "risk_score": ev_score,
                    "explanation": ev.explanation,
                })
                if ev.explanation and ev.explanation not in entity_reasons:
                    entity_reasons.append(ev.explanation)
                if risk_rank.get(ev_level, 0) > risk_rank.get(entity_risk_level, 0):
                    entity_risk_level = ev_level
                if ev_score > entity_risk_score:
                    entity_risk_score = ev_score
                for f in getattr(ev, "risk_factors", []):
                    ft = f.factor_type.value if hasattr(f.factor_type, "value") else str(f.factor_type)
                    if ft not in entity_factors:
                        entity_factors.append(ft)

        for rf in risk_factors:
            involved = getattr(rf, "involved_entity_ids", []) or []
            if node_id in involved or (track_str and track_str in involved) or (node.track_id is not None and str(node.track_id) in involved):
                ft = rf.factor_type.value if hasattr(rf.factor_type, "value") else str(rf.factor_type)
                if ft not in entity_factors:
                    entity_factors.append(ft)
                if rf.explanation and rf.explanation not in entity_reasons:
                    entity_reasons.append(rf.explanation)

        # Factual compliance explanation if marked non-compliant
        if node.node_type == SceneNodeType.PERSON and state_dict.get("compliance_status") == "NON-COMPLIANT":
            if "PPE_NON_COMPLIANCE" not in entity_factors:
                entity_factors.append("PPE_NON_COMPLIANCE")
            if not entity_reasons:
                missing_items = [k for k, v in (state_dict.get("ppe_status") or {}).items() if "MISSING" in str(v).upper()]
                if missing_items:
                    entity_reasons.append(f"Missing mandatory PPE: {', '.join(missing_items)}")

        bbox_data = None
        if node.bbox:
            bbox_data = node.bbox.model_dump() if hasattr(node.bbox, "model_dump") else (node.bbox.dict() if hasattr(node.bbox, "dict") else node.bbox)

        position = {
            "bbox": bbox_data,
            "centroid": list(node.centroid) if node.centroid else None,
            "contact_point": list(node.contact_point) if node.contact_point else None,
        }

        # Calibrated ground-plane position and metric speed estimation (Requirements 4, 10)
        ground_pt = None
        speed_m_per_s = None
        ground_speed_status = "Metric speed unavailable — camera calibration required" if not is_single_image else "Not available for single-frame analysis"
        ent_spatial_basis = "IMAGE_SPACE"

        ground_projectable = node.node_type in (
            SceneNodeType.PERSON,
            SceneNodeType.VEHICLE,
            SceneNodeType.MACHINE,
            SceneNodeType.OBJECT,
        )
        if calibrator and ground_projectable and node.contact_point:
            ground_pt = calibrator.image_to_ground(node.contact_point[0], node.contact_point[1])
            if ground_pt is not None:
                ent_spatial_basis = "GROUND_PLANE_APPROXIMATION"
                position["ground_plane"] = {"x_m": ground_pt[0], "y_m": ground_pt[1]}
                position["ground_coordinates"] = {"x_m": ground_pt[0], "y_m": ground_pt[1]}
                position["x_m"] = ground_pt[0]
                position["y_m"] = ground_pt[1]

                # Approximate ground speed needs both a px/s magnitude and a
                # directed image-space vector. Scalar-only pixel speeds cannot
                # be converted safely through a perspective homography.
                image_velocity = _image_velocity_vector_px_s(attrs)
                motion_observed = attrs.get("motion_observed") is True or (
                    image_velocity is not None and math.hypot(*image_velocity) > 1e-8
                )
                can_project_speed = (
                    speed_val is not None
                    and motion_observed
                    and (speed_val == 0.0 or image_velocity is not None)
                )
                if not is_single_image and can_project_speed:
                    speed_m_per_s = calibrator.compute_ground_speed(
                        node.contact_point[0],
                        node.contact_point[1],
                        speed_val,
                        image_velocity,
                    )
                    if speed_m_per_s is not None:
                        ground_speed_status = f"{speed_m_per_s} m/s (approximate ground-plane projection)"
                    else:
                        ground_speed_status = "Unavailable — directed motion could not be projected inside the calibration area"
                elif not is_single_image and speed_val is None:
                    ground_speed_status = "Unavailable — tracking has not provided a measured motion sample"
                elif not is_single_image and image_velocity is None:
                    ground_speed_status = "Unavailable — tracker direction is missing; image speed remains in px/s"
            else:
                position["ground_plane"] = None
                position["ground_coordinates"] = None
                position["x_m"] = None
                position["y_m"] = None
                if not is_single_image:
                    ground_speed_status = "Unavailable — contact point is outside the calibrated image area"
        else:
            position["ground_plane"] = None
            position["ground_coordinates"] = None
            position["x_m"] = None
            position["y_m"] = None
            if calibrator and not is_single_image:
                if not ground_projectable:
                    ground_speed_status = "Unavailable — entity type has no ground-plane contact position"
                elif node.contact_point is None:
                    ground_speed_status = "Unavailable — entity has no ground contact point"

        position["spatial_basis"] = ent_spatial_basis
        state_dict["speed_m_per_s"] = speed_m_per_s
        state_dict["ground_speed_status"] = ground_speed_status
        state_dict["calibration_status"] = scene.calibration_status
        state_dict["spatial_basis"] = ent_spatial_basis

        # Movement determination
        movement = "DETECTED"
        if state_dict.get("primary_behavior"):
            movement = str(state_dict.get("primary_behavior")).upper()
        elif state_dict.get("stationary_duration_s") and state_dict.get("stationary_duration_s") > 0:
            movement = "STATIONARY"
        elif state_dict.get("speed_px_per_s") and state_dict.get("speed_px_per_s") > 0:
            movement = "MOVING"

        state_dict["movement"] = movement
        state_dict["risk_level"] = entity_risk_level

        entity_obj = {
            "id": node_id,
            "type": ntype,
            "label": label,
            "class_name": node.class_name,
            "track_id": node.track_id,
            "confidence": round(float(node.confidence), 3) if node.confidence is not None else None,
            "position": position,
            "bbox": bbox_data,
            "state": state_dict,
            "risk": {
                "level": entity_risk_level,
                "score": round(entity_risk_score, 2),
                "factors": entity_factors,
                "reasons": entity_reasons,
                "active_events": entity_events,
            },
        }
        entities.append(entity_obj)

    # 5. Extract Temporal Predictions & Time-To-Hazard (TTH)
    temporal_predictions: List[Dict[str, Any]] = []

    # A. Ingest confirmed early warning indicators from PredictionEngine
    if prediction_assessment and not is_single_image:
        for ind in getattr(prediction_assessment, "active_indicators", []):
            ind_type = ind.indicator_type.value if hasattr(ind.indicator_type, "value") else str(ind.indicator_type)
            ev_data = ind.evidence or {}
            tth = ev_data.get("time_to_hazard_seconds")
            closing_spd = ev_data.get("closing_speed_px_per_s")
            dist_px = ev_data.get("current_distance_px")

            if tth is None and closing_spd and dist_px and closing_spd > 2.0:
                tth = round(max(0.0, dist_px - 60.0) / closing_spd, 1)

            # Determine source & target labels
            src_node = next((e for e in entities if ind.track_ids and e.get("track_id") == ind.track_ids[0]), None)
            tgt_node = next((e for e in entities if e["id"] in ind.entity_ids and e != src_node), None)
            if not tgt_node and ind.target_zone_id:
                tgt_node = next((e for e in entities if e.get("id") == f"zone_{ind.target_zone_id}"), None)

            src_label = src_node["label"] if src_node else (f"Track #{ind.track_ids[0]}" if ind.track_ids else "Entity")
            tgt_label = tgt_node["label"] if tgt_node else (f"Zone {ind.target_zone_id}" if ind.target_zone_id else "Hazard Area")

            # Deterministic compounding factors
            factors = []
            if src_node and src_node.get("state", {}).get("compliance_status") == "NON-COMPLIANT":
                missing = [k for k, v in (src_node["state"].get("ppe_status") or {}).items() if "MISSING" in str(v).upper()]
                factors.append(f"{src_label} PPE violation ({', '.join(missing) if missing else 'Missing PPE'})")
            if src_node and src_node.get("state", {}).get("inside_zones"):
                factors.append(f"{src_label} confirmed inside restricted zone")
            if dist_px is not None:
                factors.append(f"Current distance: {dist_px} px")
            if closing_spd is not None:
                factors.append(f"Closing velocity: {closing_spd} px/s")

            if ev_data.get("will_intersect_boundary"):
                factors.append("Trajectory crosses restricted perimeter boundary")

            # Timeline
            timeline = [
                {"phase": "CURRENT", "time_offset_s": 0.0, "severity": "MODERATE", "label": "Detected converging motion"},
            ]
            if tth is not None and tth > 0.0:
                timeline.append({
                    "phase": "WARNING",
                    "time_offset_s": round(tth * 0.5, 1),
                    "severity": "HIGH",
                    "label": "Buffer perimeter incursion",
                })
                timeline.append({
                    "phase": "HAZARD_POINT",
                    "time_offset_s": tth,
                    "severity": "CRITICAL" if str(ind.severity).upper() in ("HIGH", "CRITICAL") else "HIGH",
                    "label": "Predicted boundary contact / collision",
                })

            sev = ind.severity.value if hasattr(ind.severity, "value") else str(ind.severity)
            temporal_predictions.append({
                "id": ind.indicator_id,
                "type": ind_type,
                "source_id": src_node["id"] if src_node else None,
                "source_label": src_label,
                "target_id": tgt_node["id"] if tgt_node else None,
                "target_label": tgt_label,
                "time_to_hazard_seconds": tth,
                "tth_status": f"Image-space estimate: approximately {tth}s" if tth is not None else "Unavailable — converging velocity below threshold",
                "confidence": 0.88,
                "risk_level": sev,
                "compounding_factors_count": max(1, len(factors)),
                "compounding_factors": factors or ["Elevated trajectory convergence toward hazard"],
                "timeline": timeline,
                "explanation": ind.explanation,
                "distance_px": dist_px,
                "closing_rate_px_s": closing_spd,
                "spatial_basis": "IMAGE_SPACE",
            })

    # B. Augment from scene approaching relationships
    if not is_single_image:
        for rel in scene.relationships:
            if rel.tth_seconds is not None:
                # Check if already covered
                exists = any(p.get("source_id") == rel.source_id and p.get("target_id") == rel.target_id for p in temporal_predictions)
                if not exists:
                    s_node = next((e for e in entities if e["id"] == rel.source_id), None)
                    t_node = next((e for e in entities if e["id"] == rel.target_id), None)
                    s_lbl = s_node["label"] if s_node else rel.source_id
                    t_lbl = t_node["label"] if t_node else rel.target_id

                    factors = []
                    if s_node and s_node.get("state", {}).get("compliance_status") == "NON-COMPLIANT":
                        factors.append(f"{s_lbl} PPE violation detected")
                    elif t_node and t_node.get("state", {}).get("compliance_status") == "NON-COMPLIANT":
                        factors.append(f"{t_lbl} PPE violation detected")

                    if t_node and t_node.get("type") in ("ZONE", "VEHICLE"):
                        factors.append(f"Proximity within {t_lbl} hazard boundary")
                    elif s_node and s_node.get("type") in ("ZONE", "VEHICLE"):
                        factors.append(f"Proximity within {s_lbl} hazard boundary")

                    spd_val = rel.closing_rate_px_s
                    if spd_val is not None:
                        factors.append(f"Relative motion indicates approach at {round(float(spd_val), 1)} px/s")
                    if not factors:
                        factors.append("Kinematic convergence detected between actors")

                    temporal_predictions.append({
                        "id": f"pred_{rel.id}",
                        "type": "PREDICTED_CONFLICT",
                        "source_id": rel.source_id,
                        "source_label": s_lbl,
                        "target_id": rel.target_id,
                        "target_label": t_lbl,
                        "time_to_hazard_seconds": rel.tth_seconds,
                        "tth_status": rel.tth_status or f"Image-space estimate: approximately {rel.tth_seconds}s",
                        "confidence": 0.87,
                        "risk_level": "CRITICAL" if rel.tth_seconds <= 2.0 else "HIGH",
                        "compounding_factors_count": len(factors),
                        "compounding_factors": factors,
                        "timeline": [
                            {"phase": "CURRENT", "time_offset_s": 0.0, "severity": "MODERATE", "label": "Detected approaching trajectory"},
                            {"phase": "WARNING", "time_offset_s": round(rel.tth_seconds * 0.5, 1), "severity": "HIGH", "label": "Buffer perimeter incursion"},
                            {"phase": "HAZARD_POINT", "time_offset_s": rel.tth_seconds, "severity": "CRITICAL", "label": "Predicted physical contact"},
                        ],
                        "explanation": f"Converging image-space motion detected between {s_lbl} and {t_lbl} with estimated TTH {rel.tth_seconds}s.",
                        "distance_px": rel.distance_px,
                        "closing_rate_px_s": rel.closing_rate_px_s,
                        "spatial_basis": "IMAGE_SPACE",
                    })

    # 6. Hero Time-To-Hazard Panel Object
    hero_tth: Dict[str, Any]
    if temporal_predictions:
        valid_tth_preds = [p for p in temporal_predictions if p.get("time_to_hazard_seconds") is not None]
        if valid_tth_preds:
            best_pred = min(
                valid_tth_preds,
                key=lambda x: (0 if x.get("risk_level") == "CRITICAL" else 1, x["time_to_hazard_seconds"]),
            )
            hero_tth = {
                "available": True,
                "time_to_hazard_seconds": best_pred["time_to_hazard_seconds"],
                "source_id": best_pred.get("source_id"),
                "source_label": best_pred.get("source_label"),
                "target_id": best_pred.get("target_id"),
                "target_label": best_pred.get("target_label"),
                "hazard_type": best_pred.get("type", "PREDICTED CONFLICT"),
                "confidence": best_pred.get("confidence", 0.87),
                "risk_level": best_pred.get("risk_level", "CRITICAL"),
                "compounding_factors_count": best_pred.get("compounding_factors_count", len(best_pred.get("compounding_factors", []))),
                "compounding_factors": best_pred.get("compounding_factors", []),
                "timeline": best_pred.get("timeline", []),
                "status_text": f"IMAGE-SPACE ESTIMATE: PREDICTED CONFLICT IN {best_pred['time_to_hazard_seconds']}s",
                "spatial_basis": "IMAGE_SPACE",
                "calibration_status": scene.calibration_status,
                "distance_px": best_pred.get("distance_px"),
                "closing_rate_px_s": best_pred.get("closing_rate_px_s"),
            }
        else:
            hero_tth = {
                "available": False,
                "time_to_hazard_seconds": None,
                "source_label": temporal_predictions[0].get("source_label"),
                "target_label": temporal_predictions[0].get("target_label"),
                "hazard_type": temporal_predictions[0].get("type", "MONITORED"),
                "confidence": temporal_predictions[0].get("confidence"),
                "risk_level": temporal_predictions[0].get("risk_level", "NORMAL"),
                "compounding_factors_count": len(temporal_predictions[0].get("compounding_factors", [])),
                "compounding_factors": temporal_predictions[0].get("compounding_factors", []),
                "timeline": temporal_predictions[0].get("timeline", []),
                "status_text": "Unavailable — converging velocity below threshold",
                "spatial_basis": "IMAGE_SPACE",
                "calibration_status": scene.calibration_status,
            }
    elif is_single_image:
        hero_tth = {
            "available": False,
            "time_to_hazard_seconds": None,
            "source_label": "Single Frame Perception",
            "target_label": None,
            "hazard_type": "STATIC SCENE ANALYSIS",
            "confidence": None,
            "risk_level": "NORMAL",
            "compounding_factors_count": 0,
            "compounding_factors": [],
            "timeline": [],
            "status_text": "Unavailable — single-frame analysis has no motion vectors",
            "spatial_basis": "IMAGE_SPACE",
            "calibration_status": scene.calibration_status,
        }
    else:
        hero_tth = {
            "available": False,
            "time_to_hazard_seconds": None,
            "source_label": None,
            "target_label": None,
            "hazard_type": "MONITORED",
            "confidence": None,
            "risk_level": "NORMAL",
            "compounding_factors_count": 0,
            "compounding_factors": [],
            "timeline": [],
            "status_text": "Unavailable — insufficient motion history or entities are stationary",
            "spatial_basis": "IMAGE_SPACE",
            "calibration_status": scene.calibration_status,
        }

    # 7. Formulate Risk Events with Compounding Factors
    formatted_risk_events = []
    for ev in active_events:
        ev_level = ev.risk_level.value if hasattr(ev.risk_level, "value") else str(ev.risk_level)
        ev_type = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
        factors = []
        for rf in getattr(ev, "risk_factors", []):
            rf_type = rf.factor_type.value if hasattr(rf.factor_type, "value") else str(rf.factor_type)
            factors.append(rf.explanation or rf_type)

        formatted_risk_events.append({
            "event_id": ev.event_id,
            "event_type": ev_type,
            "risk_level": ev_level,
            "risk_score": float(ev.risk_score) if ev.risk_score is not None else 0.0,
            "explanation": ev.explanation,
            "involved_entities": getattr(ev, "involved_entities", []),
            "compounding_factors": factors,
            "compounding_factors_count": len(factors),
        })

    # Attach per-entity temporal hazard reference
    for ent in entities:
        ent_id = ent["id"]
        rel_pred = next((p for p in temporal_predictions if p.get("source_id") == ent_id or p.get("target_id") == ent_id), None)
        ent["temporal_hazard"] = rel_pred

    if not scene.scene_id:
        scene.scene_id = f"scene_{scene.camera_id}_{scene.frame_id or int(scene.timestamp)}"

    scene.entities = entities
    scene.temporal_predictions = temporal_predictions
    scene.time_to_hazard = hero_tth
    scene.risk_events = formatted_risk_events
    scene.is_live = is_live
    scene.source_mode = source_mode or ("LIVE" if is_live else "LATEST SCENE SNAPSHOT")
    return scene
