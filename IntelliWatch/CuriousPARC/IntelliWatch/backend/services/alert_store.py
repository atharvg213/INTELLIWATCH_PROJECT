"""
backend/services/alert_store.py
Step 22 — Persistent Alert Storage & Lifecycle Management.

Provides thread-safe persistent SQLite storage for safety alerts,
lifecycle transitions, audit logging, filtering, pagination, and statistical aggregation.
"""
import json
import logging
import math
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend.schemas.alert import (
    AlertHistoryEntry,
    AlertListResponse,
    AlertRecord,
    AlertSeverity,
    AlertStatsResponse,
    AlertStatus,
)

logger = logging.getLogger("intelliwatch.alert_store")


class AlertStore:
    """
    Thread-safe SQLite persistent repository for safety alerts and their lifecycle history.
    Stores tickets in data/alerts.db by default.
    """

    def __init__(self, db_path: Optional[Path] = None):
        if db_path is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.db_path = project_root / "data" / "alerts.db"
        else:
            self.db_path = Path(db_path)

        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_database()

    def _get_connection(self) -> sqlite3.Connection:
        """Returns a SQLite connection configured for WAL mode and row factory."""
        conn = sqlite3.connect(str(self.db_path), timeout=15.0)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode = WAL;")
        conn.execute("PRAGMA synchronous = NORMAL;")
        return conn

    def _init_database(self) -> None:
        """Initializes tables and indexes safely without dropping existing data."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alerts (
                        alert_id TEXT PRIMARY KEY,
                        camera_id TEXT NOT NULL,
                        camera_name TEXT NOT NULL,
                        track_id INTEGER,
                        violation_type TEXT NOT NULL,
                        severity TEXT NOT NULL,
                        confidence REAL NOT NULL,
                        status TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        first_detected_at REAL NOT NULL,
                        last_detected_at REAL NOT NULL,
                        occurrence_count INTEGER NOT NULL,
                        acknowledged_at REAL,
                        acknowledged_by TEXT,
                        resolved_at REAL,
                        resolved_by TEXT,
                        resolution_notes TEXT,
                        dismissed_at REAL,
                        dismissed_by TEXT,
                        dismiss_reason TEXT,
                        has_evidence INTEGER NOT NULL,
                        evidence_image_path TEXT,
                        metadata_json TEXT
                    )
                    """
                )
                cursor.execute(
                    """
                    CREATE TABLE IF NOT EXISTS alert_history (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        alert_id TEXT NOT NULL,
                        from_status TEXT,
                        to_status TEXT NOT NULL,
                        timestamp REAL NOT NULL,
                        user TEXT NOT NULL,
                        notes TEXT,
                        FOREIGN KEY (alert_id) REFERENCES alerts(alert_id) ON DELETE CASCADE
                    )
                    """
                )
                # Performance indexes
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_cam ON alerts(camera_id);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_status ON alerts(status);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_sev ON alerts(severity);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_violation ON alerts(violation_type);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_alerts_ts ON alerts(timestamp DESC);")
                cursor.execute("CREATE INDEX IF NOT EXISTS idx_history_alert ON alert_history(alert_id);")
                conn.commit()

    def create_or_update_alert(self, alert: AlertRecord) -> AlertRecord:
        """
        Inserts a new alert or updates an existing alert.
        If it is a new alert, an initial lifecycle history entry is logged.
        """
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT alert_id, status FROM alerts WHERE alert_id = ?", (alert.alert_id,))
                row = cursor.fetchone()

                metadata_str = json.dumps(alert.metadata or {})
                if row is None:
                    # Insert new alert
                    cursor.execute(
                        """
                        INSERT INTO alerts (
                            alert_id, camera_id, camera_name, track_id, violation_type,
                            severity, confidence, status, timestamp, first_detected_at,
                            last_detected_at, occurrence_count, acknowledged_at, acknowledged_by,
                            resolved_at, resolved_by, resolution_notes, dismissed_at,
                            dismissed_by, dismiss_reason, has_evidence, evidence_image_path,
                            metadata_json
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """,
                        (
                            alert.alert_id,
                            alert.camera_id,
                            alert.camera_name,
                            alert.track_id,
                            alert.violation_type,
                            alert.severity.value if hasattr(alert.severity, "value") else str(alert.severity),
                            float(alert.confidence),
                            alert.status.value if hasattr(alert.status, "value") else str(alert.status),
                            float(alert.timestamp),
                            float(alert.first_detected_at),
                            float(alert.last_detected_at),
                            int(alert.occurrence_count),
                            alert.acknowledged_at,
                            alert.acknowledged_by,
                            alert.resolved_at,
                            alert.resolved_by,
                            alert.resolution_notes,
                            alert.dismissed_at,
                            alert.dismissed_by,
                            alert.dismiss_reason,
                            1 if alert.has_evidence else 0,
                            alert.evidence_image_path,
                            metadata_str,
                        ),
                    )
                    # Log initial history
                    cursor.execute(
                        """
                        INSERT INTO alert_history (alert_id, from_status, to_status, timestamp, user, notes)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        (
                            alert.alert_id,
                            None,
                            AlertStatus.NEW.value,
                            float(alert.timestamp),
                            "system",
                            "Alert triggered by perception engine",
                        ),
                    )
                else:
                    # Update existing alert (e.g. occurrence aggregation, timestamp update, evidence)
                    cursor.execute(
                        """
                        UPDATE alerts SET
                            camera_name = ?,
                            track_id = ?,
                            severity = ?,
                            confidence = ?,
                            status = ?,
                            last_detected_at = ?,
                            occurrence_count = ?,
                            acknowledged_at = ?,
                            acknowledged_by = ?,
                            resolved_at = ?,
                            resolved_by = ?,
                            resolution_notes = ?,
                            dismissed_at = ?,
                            dismissed_by = ?,
                            dismiss_reason = ?,
                            has_evidence = ?,
                            evidence_image_path = ?,
                            metadata_json = ?
                        WHERE alert_id = ?
                        """,
                        (
                            alert.camera_name,
                            alert.track_id,
                            alert.severity.value if hasattr(alert.severity, "value") else str(alert.severity),
                            float(alert.confidence),
                            alert.status.value if hasattr(alert.status, "value") else str(alert.status),
                            float(alert.last_detected_at),
                            int(alert.occurrence_count),
                            alert.acknowledged_at,
                            alert.acknowledged_by,
                            alert.resolved_at,
                            alert.resolved_by,
                            alert.resolution_notes,
                            alert.dismissed_at,
                            alert.dismissed_by,
                            alert.dismiss_reason,
                            1 if alert.has_evidence else 0,
                            alert.evidence_image_path,
                            metadata_str,
                            alert.alert_id,
                        ),
                    )
                conn.commit()

        return self.get_alert(alert.alert_id) or alert

    def get_alert(self, alert_id: str) -> Optional[AlertRecord]:
        """Retrieves a single alert and its full transition history."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                cursor.execute("SELECT * FROM alerts WHERE alert_id = ?", (alert_id,))
                row = cursor.fetchone()
                if not row:
                    return None

                cursor.execute(
                    "SELECT from_status, to_status, timestamp, user, notes FROM alert_history WHERE alert_id = ? ORDER BY id ASC",
                    (alert_id,),
                )
                history_rows = cursor.fetchall()
                history = [
                    AlertHistoryEntry(
                        from_status=AlertStatus(h["from_status"]) if h["from_status"] else None,
                        to_status=AlertStatus(h["to_status"]),
                        timestamp=h["timestamp"],
                        user=h["user"],
                        notes=h["notes"],
                    )
                    for h in history_rows
                ]

                return self._row_to_alert(row, history)

    def list_alerts(
        self,
        camera_id: Optional[str] = None,
        severity: Optional[str] = None,
        status: Optional[str] = None,
        violation_type: Optional[str] = None,
        track_id: Optional[int] = None,
        start_time: Optional[float] = None,
        end_time: Optional[float] = None,
        search: Optional[str] = None,
        page: int = 1,
        page_size: int = 20,
    ) -> AlertListResponse:
        """
        Lists alerts matching filters with pagination and sorting (newest first).
        """
        page = max(1, page)
        page_size = max(1, min(100, page_size))
        offset = (page - 1) * page_size

        where_clauses: List[str] = []
        params: List[Any] = []

        if camera_id:
            where_clauses.append("camera_id = ?")
            params.append(camera_id)
        if severity:
            where_clauses.append("UPPER(severity) = ?")
            params.append(severity.upper())
        if status:
            where_clauses.append("UPPER(status) = ?")
            params.append(status.upper())
        if violation_type:
            where_clauses.append("UPPER(violation_type) LIKE ?")
            params.append(f"%{violation_type.upper()}%")
        if track_id is not None:
            where_clauses.append("track_id = ?")
            params.append(track_id)
        if start_time is not None:
            where_clauses.append("timestamp >= ?")
            params.append(float(start_time))
        if end_time is not None:
            where_clauses.append("timestamp <= ?")
            params.append(float(end_time))
        if search:
            where_clauses.append("(camera_name LIKE ? OR violation_type LIKE ? OR alert_id LIKE ?)")
            term = f"%{search}%"
            params.extend([term, term, term])

        where_str = f"WHERE {' AND '.join(where_clauses)}" if where_clauses else ""

        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()
                # Count total matching
                cursor.execute(f"SELECT COUNT(*) as count FROM alerts {where_str}", params)
                total = cursor.fetchone()["count"]

                # Fetch page
                query = f"""
                    SELECT * FROM alerts {where_str}
                    ORDER BY timestamp DESC, last_detected_at DESC
                    LIMIT ? OFFSET ?
                """
                cursor.execute(query, params + [page_size, offset])
                rows = cursor.fetchall()

                items: List[AlertRecord] = []
                for row in rows:
                    items.append(self._row_to_alert(row, history=[]))

                total_pages = max(1, math.ceil(total / page_size)) if total > 0 else 1

                return AlertListResponse(
                    items=items,
                    total=total,
                    page=page,
                    page_size=page_size,
                    total_pages=total_pages,
                )

    def acknowledge_alert(
        self,
        alert_id: str,
        user: str = "operator",
        notes: Optional[str] = None,
    ) -> Optional[AlertRecord]:
        """Transitions an alert from NEW to ACKNOWLEDGED."""
        with self._lock:
            alert = self.get_alert(alert_id)
            if not alert:
                return None

            if alert.status in (AlertStatus.RESOLVED, AlertStatus.DISMISSED):
                raise ValueError(f"Cannot acknowledge alert '{alert_id}' with status '{alert.status.value}'")

            now = time.time()
            prev_status = alert.status
            alert.status = AlertStatus.ACKNOWLEDGED
            alert.acknowledged_at = now
            alert.acknowledged_by = user

            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE alerts SET
                        status = ?,
                        acknowledged_at = ?,
                        acknowledged_by = ?
                    WHERE alert_id = ?
                    """,
                    (alert.status.value, now, user, alert_id),
                )
                conn.execute(
                    """
                    INSERT INTO alert_history (alert_id, from_status, to_status, timestamp, user, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (alert_id, prev_status.value, AlertStatus.ACKNOWLEDGED.value, now, user, notes or "Acknowledged by operator"),
                )
                conn.commit()

            return self.get_alert(alert_id)

    def resolve_alert(
        self,
        alert_id: str,
        user: str = "operator",
        notes: Optional[str] = None,
    ) -> Optional[AlertRecord]:
        """Transitions an alert to RESOLVED (from NEW or ACKNOWLEDGED)."""
        with self._lock:
            alert = self.get_alert(alert_id)
            if not alert:
                return None

            if alert.status == AlertStatus.DISMISSED:
                raise ValueError(f"Cannot resolve an alert that has already been dismissed.")

            now = time.time()
            prev_status = alert.status
            alert.status = AlertStatus.RESOLVED
            alert.resolved_at = now
            alert.resolved_by = user
            alert.resolution_notes = notes

            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE alerts SET
                        status = ?,
                        resolved_at = ?,
                        resolved_by = ?,
                        resolution_notes = ?
                    WHERE alert_id = ?
                    """,
                    (alert.status.value, now, user, notes, alert_id),
                )
                conn.execute(
                    """
                    INSERT INTO alert_history (alert_id, from_status, to_status, timestamp, user, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (alert_id, prev_status.value, AlertStatus.RESOLVED.value, now, user, notes or "Resolved by operator"),
                )
                conn.commit()

            return self.get_alert(alert_id)

    def dismiss_alert(
        self,
        alert_id: str,
        user: str = "operator",
        reason: str = "",
    ) -> Optional[AlertRecord]:
        """Transitions an alert to DISMISSED with required reason."""
        if not reason or not reason.strip():
            raise ValueError("Dismissal requires a valid reason justification.")

        with self._lock:
            alert = self.get_alert(alert_id)
            if not alert:
                return None

            now = time.time()
            prev_status = alert.status
            alert.status = AlertStatus.DISMISSED
            alert.dismissed_at = now
            alert.dismissed_by = user
            alert.dismiss_reason = reason.strip()

            with self._get_connection() as conn:
                conn.execute(
                    """
                    UPDATE alerts SET
                        status = ?,
                        dismissed_at = ?,
                        dismissed_by = ?,
                        dismiss_reason = ?
                    WHERE alert_id = ?
                    """,
                    (alert.status.value, now, user, alert.dismiss_reason, alert_id),
                )
                conn.execute(
                    """
                    INSERT INTO alert_history (alert_id, from_status, to_status, timestamp, user, notes)
                    VALUES (?, ?, ?, ?, ?, ?)
                    """,
                    (alert_id, prev_status.value, AlertStatus.DISMISSED.value, now, user, f"Dismissed: {alert.dismiss_reason}"),
                )
                conn.commit()

            return self.get_alert(alert_id)

    def get_statistics(self) -> AlertStatsResponse:
        """Computes aggregated alert counts grouped by status, severity, and violation type."""
        with self._lock:
            with self._get_connection() as conn:
                cursor = conn.cursor()

                # Total alerts
                cursor.execute("SELECT COUNT(*) as cnt FROM alerts")
                total = cursor.fetchone()["cnt"]

                # By status
                cursor.execute("SELECT status, COUNT(*) as cnt FROM alerts GROUP BY status")
                by_status = {row["status"]: row["cnt"] for row in cursor.fetchall()}

                # By severity
                cursor.execute("SELECT severity, COUNT(*) as cnt FROM alerts GROUP BY severity")
                by_severity = {row["severity"]: row["cnt"] for row in cursor.fetchall()}

                # By violation type
                cursor.execute("SELECT violation_type, COUNT(*) as cnt FROM alerts GROUP BY violation_type")
                by_violation = {row["violation_type"]: row["cnt"] for row in cursor.fetchall()}

                active_unresolved = by_status.get(AlertStatus.NEW.value, 0) + by_status.get(AlertStatus.ACKNOWLEDGED.value, 0)

                return AlertStatsResponse(
                    total_alerts=total,
                    by_status=by_status,
                    by_severity=by_severity,
                    by_violation=by_violation,
                    active_unresolved_count=active_unresolved,
                )

    def reset(self) -> None:
        """Clears all alert and history records (used for test isolation)."""
        with self._lock:
            with self._get_connection() as conn:
                conn.execute("DELETE FROM alert_history")
                conn.execute("DELETE FROM alerts")
                conn.commit()

    @staticmethod
    def _row_to_alert(row: sqlite3.Row, history: List[AlertHistoryEntry]) -> AlertRecord:
        """Deserializes a SQLite row to an AlertRecord instance."""
        meta = {}
        if row["metadata_json"]:
            try:
                meta = json.loads(row["metadata_json"])
            except Exception:
                meta = {}

        return AlertRecord(
            alert_id=row["alert_id"],
            camera_id=row["camera_id"],
            camera_name=row["camera_name"],
            track_id=row["track_id"],
            violation_type=row["violation_type"],
            severity=AlertSeverity(row["severity"]),
            confidence=row["confidence"],
            status=AlertStatus(row["status"]),
            timestamp=row["timestamp"],
            first_detected_at=row["first_detected_at"],
            last_detected_at=row["last_detected_at"],
            occurrence_count=row["occurrence_count"],
            acknowledged_at=row["acknowledged_at"],
            acknowledged_by=row["acknowledged_by"],
            resolved_at=row["resolved_at"],
            resolved_by=row["resolved_by"],
            resolution_notes=row["resolution_notes"],
            dismissed_at=row["dismissed_at"],
            dismissed_by=row["dismissed_by"],
            dismiss_reason=row["dismiss_reason"],
            has_evidence=bool(row["has_evidence"]),
            evidence_image_path=row["evidence_image_path"],
            metadata=meta,
            transition_history=history,
        )


_alert_store_instance: Optional[AlertStore] = None
_store_lock = threading.Lock()


def get_alert_store() -> AlertStore:
    """Returns the singleton instance of AlertStore."""
    global _alert_store_instance
    if _alert_store_instance is None:
        with _store_lock:
            if _alert_store_instance is None:
                _alert_store_instance = AlertStore()
    return _alert_store_instance
