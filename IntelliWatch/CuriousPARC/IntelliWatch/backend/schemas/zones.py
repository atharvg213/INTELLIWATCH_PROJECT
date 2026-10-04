from enum import Enum
import math
from typing import Any, Dict, List, Optional, Tuple, Union
from pydantic import BaseModel, Field, field_validator, model_validator


class ZoneType(str, Enum):
    RESTRICTED = "restricted"
    MONITORED = "monitored"
    HAZARD = "hazard"
    SAFE = "safe"
    SAFETY = "safety"
    MACHINE = "machine"


class ZoneMembershipStatus(str, Enum):
    OUTSIDE = "outside"
    INSIDE = "inside"
    BOUNDARY = "boundary"


class ZonePoint(BaseModel):
    """2D vertex coordinate."""
    x: float
    y: float


def normalize_polygon_points(
    raw_polygon: Any,
    deduplicate_consecutive: bool = True,
) -> List[List[float]]:
    """
    Normalizes arbitrary vertex representations into a clean List[List[float]].
    Accepts:
      - [[x, y], [x, y], ...]
      - [{'x': x, 'y': y}, ...]
      - [ZonePoint(x, y), ...]
      - [(x, y), ...]

    Enforces:
      - Numeric, finite values (no NaN / Inf)
      - Consecutive duplicate vertex removal
      - At least 3 non-degenerate vertices
    """
    if not raw_polygon:
        raise ValueError("Polygon cannot be empty.")

    if not isinstance(raw_polygon, (list, tuple)):
        raise ValueError(f"Polygon must be a sequence of points, got {type(raw_polygon).__name__}.")

    normalized: List[List[float]] = []
    for idx, pt in enumerate(raw_polygon):
        if isinstance(pt, dict):
            if "x" not in pt or "y" not in pt:
                raise ValueError(f"Vertex at index {idx} dictionary must have 'x' and 'y' keys, got {pt}.")
            vx, vy = pt["x"], pt["y"]
        elif hasattr(pt, "x") and hasattr(pt, "y"):
            vx, vy = getattr(pt, "x"), getattr(pt, "y")
        elif isinstance(pt, (list, tuple)):
            if len(pt) != 2:
                raise ValueError(f"Vertex at index {idx} must be a 2-element [x, y] coordinate, got {pt}.")
            vx, vy = pt[0], pt[1]
        else:
            raise ValueError(f"Invalid vertex format at index {idx}: {pt}")

        try:
            fx, fy = float(vx), float(vy)
        except (ValueError, TypeError):
            raise ValueError(f"Vertex at index {idx} coordinates must be numeric, got ({vx}, {vy}).")

        if not math.isfinite(fx) or not math.isfinite(fy):
            raise ValueError(f"Vertex at index {idx} has non-finite coordinate values: ({fx}, {fy}).")

        point = [round(fx, 2), round(fy, 2)]

        if deduplicate_consecutive and normalized:
            prev = normalized[-1]
            if abs(prev[0] - point[0]) < 1e-4 and abs(prev[1] - point[1]) < 1e-4:
                continue  # Skip consecutive identical point

        normalized.append(point)

    # If the last point equals the first point, strip the redundant closing vertex
    if len(normalized) >= 4:
        if abs(normalized[0][0] - normalized[-1][0]) < 1e-4 and abs(normalized[0][1] - normalized[-1][1]) < 1e-4:
            normalized.pop()

    if len(normalized) < 3:
        raise ValueError(f"Polygon must have at least 3 distinct vertices, got {len(normalized)}.")

    return normalized


class RestrictedZone(BaseModel):
    """
    Polygonal restricted geofence representation for industrial safety.
    """
    zone_id: str = Field(..., description="Unique alphanumeric identifier for the zone")
    name: str = Field(..., description="Human-readable label for the zone")
    zone_type: ZoneType = Field(default=ZoneType.RESTRICTED, description="Operational category of the zone")
    polygon: List[List[float]] = Field(
        ...,
        description="Ordered list of [x, y] vertex coordinates in image space defining the polygon"
    )
    enabled: bool = Field(default=True, description="Whether zone monitoring is actively evaluated")
    max_dwell_seconds: Optional[float] = Field(
        default=None,
        description="Max allowed worker dwell duration before triggering ZONE_DWELL_EXCEEDED"
    )
    source_resolution: Optional[Tuple[int, int]] = Field(
        default=None,
        description="Source media resolution (width, height) for which polygon was defined"
    )

    @field_validator("polygon", mode="before")
    @classmethod
    def validate_and_normalize_polygon(cls, v: Any) -> List[List[float]]:
        return normalize_polygon_points(v)


class ZoneCreateRequest(BaseModel):
    """
    Operator request schema for creating a new safety zone.
    """
    zone_id: Optional[str] = Field(default=None, description="Optional custom unique alphanumeric ID")
    name: str = Field(..., min_length=1, max_length=100, description="Human-readable label for the zone")
    zone_type: ZoneType = Field(default=ZoneType.RESTRICTED, description="Operational category of the zone")
    polygon: Any = Field(..., description="List of vertices as [[x,y],...] or [{'x':x,'y':y},...]")
    enabled: bool = Field(default=True, description="Whether zone monitoring is actively evaluated")
    max_dwell_seconds: Optional[float] = Field(default=None, ge=0.0, description="Max allowed dwell time in seconds")
    source_resolution: Optional[Tuple[int, int]] = Field(default=None, description="Source frame dimensions (width, height)")

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("Zone name cannot be empty or whitespace only.")
        return s

    @field_validator("polygon", mode="before")
    @classmethod
    def validate_request_polygon(cls, v: Any) -> List[List[float]]:
        return normalize_polygon_points(v)


class ZoneUpdateRequest(BaseModel):
    """
    Operator request schema for updating an existing safety zone.
    """
    name: Optional[str] = Field(default=None, min_length=1, max_length=100)
    zone_type: Optional[ZoneType] = None
    polygon: Optional[Any] = None
    enabled: Optional[bool] = None
    max_dwell_seconds: Optional[float] = Field(default=None, ge=0.0)
    source_resolution: Optional[Tuple[int, int]] = None

    @field_validator("name")
    @classmethod
    def validate_update_name(cls, v: Optional[str]) -> Optional[str]:
        if v is not None:
            s = v.strip()
            if not s:
                raise ValueError("Zone name cannot be empty or whitespace only.")
            return s
        return v

    @field_validator("polygon", mode="before")
    @classmethod
    def validate_update_polygon(cls, v: Any) -> Optional[List[List[float]]]:
        if v is not None:
            return normalize_polygon_points(v)
        return None


class ZoneMembership(BaseModel):
    """
    Evaluated spatial relationship between a tracked actor and a specific restricted zone.
    """
    track_id: int = Field(..., description="Track identifier of the object")
    zone_id: str = Field(..., description="Target zone identifier")
    zone_name: str = Field(..., description="Target zone human-readable name")
    is_inside: bool = Field(..., description="True if actor contact point is inside or on boundary")
    status: ZoneMembershipStatus = Field(..., description="Detailed geometric relationship (inside/outside/boundary)")
    contact_point: Tuple[float, float] = Field(..., description="(x, y) ground contact point tested")
    dwell_seconds: float = Field(default=0.0, description="Elapsed seconds actor has remained inside zone")
    entry_timestamp: Optional[float] = Field(default=None, description="Timestamp when confirmed entry occurred")


class FrameZoneOccupancy(BaseModel):
    """
    Aggregated zone membership state for a single video frame.
    """
    frame_id: Optional[int] = Field(default=None, description="Sequential frame index")
    timestamp: float = Field(..., description="Stream timestamp of the frame in seconds")
    memberships: List[ZoneMembership] = Field(default_factory=list, description="All evaluated track-zone relationships")
    occupancy_by_zone: Dict[str, List[int]] = Field(
        default_factory=dict,
        description="Mapping from zone_id to list of track_ids currently confirmed inside"
    )
