"""
backend/services/zone_service.py
Step 16 - Centralized Interactive Zone Configuration & Persistence Service.

Provides:
- Thread-safe CRUD operations for safety zones
- Persistence to configs/zones.json (survives server restart)
- Validation of names, polygons, coordinates, and resolution bounds
- Single source of truth driving the underlying ZoneEngine
- Graceful error handling for missing/corrupt configuration
"""
import json
import logging
import os
import re
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import uuid

from backend.schemas.zones import (
    RestrictedZone,
    ZoneCreateRequest,
    ZoneType,
    ZoneUpdateRequest,
    normalize_polygon_points,
)
from configs.settings import get_settings
from intelligence.zones.zone_engine import ZoneEngine

logger = logging.getLogger("intelliwatch.zone_service")

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def slugify_zone_name(name: str) -> str:
    """Creates a clean alphanumeric identifier from a zone name."""
    clean = re.sub(r"[^\w\s-]", "", name.lower()).strip()
    clean = re.sub(r"[-\s]+", "_", clean)
    return clean or "zone"


class ZoneService:
    """
    Operator Zone Management Service coordinating JSON configuration persistence
    and real-time ZoneEngine state updates.
    """

    def __init__(
        self,
        config_path: Optional[Union[str, Path]] = None,
        zone_engine: Optional[ZoneEngine] = None,
    ):
        settings = get_settings()
        self._lock = threading.RLock()

        if config_path is None:
            self.config_path = PROJECT_ROOT / getattr(settings, "ZONE_CONFIG_PATH", "configs/zones.json")
        else:
            p = Path(config_path)
            self.config_path = p if p.is_absolute() else PROJECT_ROOT / p

        # Ensure parent directory exists
        self.config_path.parent.mkdir(parents=True, exist_ok=True)

        # Initialize or attach existing ZoneEngine
        if zone_engine is not None:
            self._engine = zone_engine
        else:
            self._engine = ZoneEngine(
                config_path=self.config_path,
                confirmation_frames=getattr(settings, "ZONE_ENTRY_CONFIRMATION_FRAMES", 3),
                default_max_dwell_seconds=getattr(settings, "ZONE_MAX_DWELL_SECONDS", 10.0),
            )

        logger.info(
            f"ZoneService initialized with config: {self.config_path} "
            f"({len(self._engine.zones)} zones active)"
        )

    @property
    def engine(self) -> ZoneEngine:
        """Returns the active ZoneEngine instance."""
        return self._engine

    def list_zones(self, enabled_only: bool = False) -> List[RestrictedZone]:
        """
        Returns all registered safety zones.
        """
        with self._lock:
            zones = list(self._engine.zones.values())
            if enabled_only:
                return [z for z in zones if z.enabled]
            return zones

    def get_zone(self, zone_id: str) -> Optional[RestrictedZone]:
        """
        Retrieves a single zone by ID.
        """
        with self._lock:
            return self._engine.zones.get(zone_id)

    def create_zone(self, request: ZoneCreateRequest) -> RestrictedZone:
        """
        Validates, registers, and persists a newly created safety zone.
        """
        with self._lock:
            # 1. Determine or generate unique zone ID
            if request.zone_id and request.zone_id.strip():
                zid = request.zone_id.strip()
            else:
                base_slug = slugify_zone_name(request.name)
                candidate = base_slug
                counter = 1
                while candidate in self._engine.zones:
                    candidate = f"{base_slug}_{counter}"
                    counter += 1
                zid = candidate

            # Check for conflict if explicit ID was supplied
            if request.zone_id and zid in self._engine.zones:
                raise ValueError(f"Zone with ID '{zid}' already exists.")

            # 2. Normalize and validate polygon
            normalized_poly = normalize_polygon_points(request.polygon)

            # Optional resolution bounds check
            if request.source_resolution:
                w, h = request.source_resolution
                if w > 0 and h > 0:
                    for pt in normalized_poly:
                        if pt[0] < -50 or pt[0] > w + 50 or pt[1] < -50 or pt[1] > h + 50:
                            raise ValueError(
                                f"Vertex ({pt[0]}, {pt[1]}) is significantly outside source resolution ({w}x{h})."
                            )

            # 3. Create RestrictedZone model
            new_zone = RestrictedZone(
                zone_id=zid,
                name=request.name,
                zone_type=request.zone_type,
                polygon=normalized_poly,
                enabled=request.enabled,
                max_dwell_seconds=request.max_dwell_seconds,
                source_resolution=request.source_resolution,
            )

            # 4. Register in ZoneEngine
            success = self._engine.add_zone(new_zone)
            if not success:
                raise ValueError(f"Failed to register zone '{zid}' in ZoneEngine contour compiler.")

            # 5. Persist to disk
            self.save_to_disk()
            logger.info(f"Created safety zone '{zid}' ({new_zone.name}) with {len(normalized_poly)} vertices.")
            return new_zone

    def update_zone(self, zone_id: str, request: ZoneUpdateRequest) -> Optional[RestrictedZone]:
        """
        Updates fields of an existing zone and re-compiles its geometric contour.
        Returns None if zone does not exist.
        """
        with self._lock:
            existing = self._engine.zones.get(zone_id)
            if existing is None:
                return None

            # Prepare updated values
            updated_name = request.name if request.name is not None else existing.name
            updated_type = request.zone_type if request.zone_type is not None else existing.zone_type
            updated_enabled = request.enabled if request.enabled is not None else existing.enabled
            updated_dwell = request.max_dwell_seconds if request.max_dwell_seconds is not None else existing.max_dwell_seconds
            updated_res = request.source_resolution if request.source_resolution is not None else existing.source_resolution

            if request.polygon is not None:
                updated_poly = normalize_polygon_points(request.polygon)
                if updated_res:
                    w, h = updated_res
                    if w > 0 and h > 0:
                        for pt in updated_poly:
                            if pt[0] < -50 or pt[0] > w + 50 or pt[1] < -50 or pt[1] > h + 50:
                                raise ValueError(
                                    f"Vertex ({pt[0]}, {pt[1]}) is significantly outside source resolution ({w}x{h})."
                                )
            else:
                updated_poly = existing.polygon

            updated_zone = RestrictedZone(
                zone_id=zone_id,
                name=updated_name,
                zone_type=updated_type,
                polygon=updated_poly,
                enabled=updated_enabled,
                max_dwell_seconds=updated_dwell,
                source_resolution=updated_res,
            )

            # Update in engine
            self._engine.add_zone(updated_zone)

            # Persist to disk
            self.save_to_disk()
            logger.info(f"Updated safety zone '{zone_id}' ({updated_zone.name}).")
            return updated_zone

    def delete_zone(self, zone_id: str) -> bool:
        """
        Removes a zone from active evaluation and disk persistence.
        Returns True if removed, False if not found.
        """
        with self._lock:
            if zone_id not in self._engine.zones:
                return False

            self._engine.remove_zone(zone_id)
            self.save_to_disk()
            logger.info(f"Deleted safety zone '{zone_id}'.")
            return True

    def save_to_disk(self) -> None:
        """
        Serializes all registered zones into JSON configuration file safely and atomically.
        """
        with self._lock:
            zone_list = [z.model_dump(mode="json") for z in self._engine.zones.values()]
            tmp_path = self.config_path.with_suffix(".tmp")
            try:
                with open(tmp_path, "w", encoding="utf-8") as f:
                    json.dump(zone_list, f, indent=2)
                # Atomic file replacement
                tmp_path.replace(self.config_path)
            except Exception as e:
                logger.error(f"Failed to persist zones to {self.config_path}: {e}")
                if tmp_path.exists():
                    tmp_path.unlink(missing_ok=True)
                raise IOError(f"Could not persist safety zones configuration: {e}")

    def reload_from_disk(self) -> int:
        """
        Reloads zone configurations from disk.
        """
        with self._lock:
            count = self._engine.load_zones_from_file(self.config_path)
            logger.info(f"Reloaded {count} zones from {self.config_path}")
            return count


# Singleton instance
_zone_service_instance: Optional[ZoneService] = None
_zone_service_lock = threading.Lock()


def get_zone_service() -> ZoneService:
    """Returns the module-level singleton ZoneService."""
    global _zone_service_instance
    if _zone_service_instance is None:
        with _zone_service_lock:
            if _zone_service_instance is None:
                _zone_service_instance = ZoneService()
    return _zone_service_instance


def get_zone_engine() -> ZoneEngine:
    """Returns the active ZoneEngine singleton driven by ZoneService."""
    return get_zone_service().engine
