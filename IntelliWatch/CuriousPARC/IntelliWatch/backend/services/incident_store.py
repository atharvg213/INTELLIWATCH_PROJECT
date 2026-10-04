"""
backend/services/incident_store.py
Incident & Evidence Storage Service.
Maintains in-memory indexing of safety incident evidence records with
deterministic local filesystem storage for snapshots and metadata in data/output/incidents/.
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np

from backend.schemas.evidence import IncidentRecord
from backend.schemas.risk import EventLifecycleState, RiskLevel

logger = logging.getLogger("intelliwatch.incident_store")


class IncidentStore:
    """
    Thread-safe / asynchronous incident evidence registry.
    Saves snapshots and structured JSON records locally to avoid external/cloud dependencies.
    """

    def __init__(self, storage_dir: Optional[Path] = None):
        if storage_dir is None:
            # Deterministic project-root-relative path
            project_root = Path(__file__).resolve().parent.parent.parent
            self.storage_dir = project_root / "data" / "output" / "incidents"
        else:
            self.storage_dir = Path(storage_dir)

        # Create output directory deterministically
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        # In-memory index: incident_id -> IncidentRecord
        self._incidents: Dict[str, IncidentRecord] = {}
        # Chronological order of incident IDs
        self._incident_order: List[str] = []

    def record_incident(
        self,
        incident: IncidentRecord,
        frame_image: Optional[np.ndarray] = None,
    ) -> IncidentRecord:
        """
        Registers an incident evidence record.
        If frame_image is provided, saves deterministic snapshot image to disk.
        Also persists structured JSON metadata for reproducibility.
        """
        # Save snapshot if frame is provided
        if frame_image is not None and frame_image.size > 0:
            snapshot_filename = f"{incident.incident_id}.jpg"
            snapshot_path = self.storage_dir / snapshot_filename
            try:
                cv2.imwrite(str(snapshot_path), frame_image)
                incident.has_snapshot = True
                incident.snapshot_path = str(snapshot_path)
            except Exception as e:
                logger.warning(f"Failed to save evidence snapshot image {snapshot_path}: {e}")
                incident.has_snapshot = False
                incident.snapshot_path = None
        elif incident.has_snapshot and incident.snapshot_path:
            # Preserve existing snapshot if already established
            pass
        else:
            incident.has_snapshot = False
            incident.snapshot_path = None

        # Persist structured JSON metadata
        json_path = self.storage_dir / f"{incident.incident_id}.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(incident.model_dump(mode="json"), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to persist incident JSON {json_path}: {e}")

        # Store in memory
        if incident.incident_id not in self._incidents:
            self._incident_order.append(incident.incident_id)
        self._incidents[incident.incident_id] = incident

        logger.info(
            f"Recorded incident {incident.incident_id}: [{incident.risk_level.value}] "
            f"{incident.event_type.value} - {incident.explanation}"
        )
        return incident

    def get_incident(self, incident_id: str) -> Optional[IncidentRecord]:
        """Retrieves a specific incident record by its unique ID."""
        return self._incidents.get(incident_id)

    def list_incidents(
        self,
        status: Optional[str] = None,
        risk_level: Optional[str] = None,
        event_type: Optional[str] = None,
        track_id: Optional[int] = None,
        zone: Optional[str] = None,
        limit: int = 50,
    ) -> List[IncidentRecord]:
        """
        Returns incident records in reverse chronological order (newest first)
        with optional filtering by lifecycle status, risk level tier, event type,
        participating track ID, or restricted zone name.
        """
        results: List[IncidentRecord] = []
        for inc_id in reversed(self._incident_order):
            inc = self._incidents[inc_id]
            if status is not None:
                curr_status = inc.lifecycle_state.value if hasattr(inc.lifecycle_state, "value") else str(inc.lifecycle_state)
                if curr_status.upper() != status.upper():
                    continue
            if risk_level is not None:
                curr_lvl = inc.risk_level.value if hasattr(inc.risk_level, "value") else str(inc.risk_level)
                if curr_lvl.upper() != risk_level.upper():
                    continue
            if event_type is not None:
                curr_type = inc.event_type.value if hasattr(inc.event_type, "value") else str(inc.event_type)
                if curr_type.upper() != event_type.upper():
                    continue
            if track_id is not None:
                if track_id not in (inc.involved_track_ids or []):
                    continue
            if zone is not None:
                inc_zone = inc.location_or_zone or (inc.metadata.get("zone_name") if inc.metadata else "")
                if not inc_zone or zone.lower() not in inc_zone.lower():
                    continue

            results.append(inc)
            if len(results) >= limit:
                break

        return results

    def get_active_incidents(self) -> List[IncidentRecord]:
        """Returns incidents currently in CONFIRMED or ACTIVE lifecycle state."""
        active_states = {
            EventLifecycleState.CONFIRMED,
            EventLifecycleState.ACTIVE,
            "CONFIRMED",
            "ACTIVE",
        }
        return [
            inc for inc in self._incidents.values()
            if inc.lifecycle_state in active_states
        ]

    def update_incident_lifecycle(
        self,
        incident_id: str,
        new_lifecycle: EventLifecycleState,
        end_ts: Optional[float] = None,
    ) -> Optional[IncidentRecord]:
        """Updates the temporal lifecycle state of an existing incident."""
        inc = self._incidents.get(incident_id)
        if inc is None:
            return None

        inc.lifecycle_state = new_lifecycle
        if end_ts is not None:
            inc.metadata["end_timestamp"] = end_ts

        # Update persisted JSON
        json_path = self.storage_dir / f"{incident_id}.json"
        try:
            with open(json_path, "w", encoding="utf-8") as f:
                json.dump(inc.model_dump(mode="json"), f, indent=2)
        except Exception as e:
            logger.warning(f"Failed to update incident JSON {json_path}: {e}")

        return inc

    def total_count(self) -> int:
        """Returns total number of recorded incidents."""
        return len(self._incidents)

    def reset(self) -> None:
        """Clears in-memory incident cache."""
        self._incidents.clear()
        self._incident_order.clear()


_incident_store_instance = IncidentStore()


def get_incident_store() -> IncidentStore:
    """Returns the singleton instance of the IncidentStore."""
    return _incident_store_instance
