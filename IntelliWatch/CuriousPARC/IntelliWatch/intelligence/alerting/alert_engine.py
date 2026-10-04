"""
intelligence/alerting/alert_engine.py
Step 22 — Centralized Safety Alert Engine.

Transforms raw perception events (PPE violations, dangerous worker behaviors,
zone intrusions, and falls) into actionable, traceable alerts with severity grading,
confidence gating, deduplication, cooldown correlation, and visual evidence capture.
"""
import logging
import math
import os
import re
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from backend.schemas.alert import (
    AlertRecord,
    AlertSeverity,
    AlertStatus,
)
from backend.schemas.assessment import FrameAssessment
from backend.schemas.behavior import PrimaryBehavior
from backend.schemas.ppe import ComplianceStatus, PPEItemState
from backend.services.alert_store import AlertStore, get_alert_store
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.alert_engine")


class AlertEngine:
    """
    Centralized industrial safety alert engine.
    Ingests assessments from live streams or video jobs, enforces severity rules,
    performs deduplication across temporal cooldown windows, captures visual evidence,
    and publishes tickets to the persistent AlertStore.
    """

    DEFAULT_SEVERITY_MAP: Dict[str, AlertSeverity] = {
        # Falls & Urgent Physical Hazards (Immediate Alert Path)
        "FALL_DETECTED": AlertSeverity.CRITICAL,
        "POSSIBLE_FALL": AlertSeverity.CRITICAL,
        "FALL_LIKE_EVENT": AlertSeverity.CRITICAL,

        # High Risk Spatial & Compound Hazards
        "RESTRICTED_ZONE_INTRUSION": AlertSeverity.HIGH,
        "COMPOUND_SAFETY_EVENT": AlertSeverity.HIGH,
        "WORKER_MACHINE_RISK": AlertSeverity.HIGH,
        "APPROACHING_VEHICLE": AlertSeverity.HIGH,
        "PERSON_VEHICLE_PROXIMITY": AlertSeverity.HIGH,

        # Mandatory Critical PPE Non-Compliance
        "NO_HARDHAT": AlertSeverity.HIGH,
        "NO_SAFETY_VEST": AlertSeverity.HIGH,
        "PPE_VIOLATION": AlertSeverity.HIGH,

        # Secondary PPE & Moderate Movement Hazards
        "NO_MASK": AlertSeverity.MEDIUM,
        "NO_GLOVES": AlertSeverity.MEDIUM,
        "RAPID_MOVEMENT_EVENT": AlertSeverity.MEDIUM,

        # Operational & Dwell Disclosures
        "RESTRICTED_ZONE_DWELL": AlertSeverity.LOW,
        "PROLONGED_STATIONARY_EVENT": AlertSeverity.LOW,
    }

    def __init__(
        self,
        alert_store: Optional[AlertStore] = None,
        evidence_dir: Optional[Path] = None,
        cooldown_seconds: Optional[float] = None,
        confidence_threshold: Optional[float] = None,
        immediate_critical: Optional[bool] = None,
        severity_map: Optional[Dict[str, AlertSeverity]] = None,
    ):
        settings = get_settings()
        self.store = alert_store or get_alert_store()

        if evidence_dir is None:
            project_root = Path(__file__).resolve().parent.parent.parent
            self.evidence_dir = project_root / settings.ALERT_EVIDENCE_DIR
        else:
            self.evidence_dir = Path(evidence_dir)

        self.evidence_dir.mkdir(parents=True, exist_ok=True)

        self.cooldown_seconds = cooldown_seconds if cooldown_seconds is not None else settings.ALERT_COOLDOWN_SECONDS
        self.confidence_threshold = confidence_threshold if confidence_threshold is not None else settings.ALERT_CONFIDENCE_THRESHOLD
        self.immediate_critical = immediate_critical if immediate_critical is not None else settings.ALERT_CRITICAL_IMMEDIATE

        self.severity_map = dict(self.DEFAULT_SEVERITY_MAP)
        if severity_map:
            self.severity_map.update(severity_map)

        # Thread-safe in-memory deduplication tracker
        # key: (camera_id, worker_id_or_unassigned, violation_type)
        # value: dict with alert_id, first_detected_at, last_detected_at, occurrence_count
        self._active_correlations: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
        self._lock = threading.RLock()

    def get_severity_for_violation(self, violation_type: str) -> AlertSeverity:
        """Determines configured severity tier for a given violation identifier."""
        norm_type = violation_type.upper().replace("-", "_").replace(" ", "_")
        for key, sev in self.severity_map.items():
            if key in norm_type or norm_type in key:
                return sev
        return AlertSeverity.MEDIUM

    def is_critical_immediate(self, violation_type: str, severity: AlertSeverity) -> bool:
        """Checks if event qualifies for immediate alert bypass."""
        if not self.immediate_critical:
            return False
        if severity == AlertSeverity.CRITICAL:
            return True
        norm = violation_type.upper()
        return "FALL" in norm

    def process_assessment(
        self,
        assessment: FrameAssessment,
        frame_image: Optional[np.ndarray] = None,
        camera_name: Optional[str] = None,
    ) -> List[AlertRecord]:
        """
        Processes a FrameAssessment from live stream or video analysis, extracting
        all PPE violations, hazardous behaviors, and zone intrusions.
        Returns any new or updated alert tickets.
        """
        generated_alerts: List[AlertRecord] = []
        camera_id = assessment.camera_id or "cam_01"
        cam_name = camera_name or f"Camera {camera_id}"
        ts = assessment.timestamp if assessment.timestamp > 0 else time.time()

        # 1. Process Worker PPE Non-Compliance
        for worker in (assessment.worker_inventories or []):
            track_id = worker.track_id
            bbox_coords = [int(worker.bbox.x1), int(worker.bbox.y1), int(worker.bbox.x2), int(worker.bbox.y2)] if worker.bbox else None

            # Explicit negative items (e.g. NO-Hardhat, NO-Safety Vest)
            for item in worker.items:
                if item.is_negative or "NO-" in item.class_name.upper():
                    violation_key = item.class_name.upper().replace("-", "_").replace(" ", "_")
                    conf = item.confidence
                    alert = self.process_violation_event(
                        camera_id=camera_id,
                        violation_type=violation_key,
                        confidence=conf,
                        track_id=track_id,
                        bbox=bbox_coords,
                        timestamp=ts,
                        frame_image=frame_image,
                        camera_name=cam_name,
                        metadata={
                            "item_name": item.class_name,
                            "body_region": item.body_region,
                            "association_score": item.association_score,
                            "frame_id": assessment.frame_id,
                        },
                    )
                    if alert:
                        generated_alerts.append(alert)

            # Missing mandatory items from ppe_status
            if worker.compliance_status == ComplianceStatus.NON_COMPLIANT:
                for ppe_name, state in worker.ppe_status.items():
                    if state == PPEItemState.MISSING:
                        violation_key = f"NO_{ppe_name.upper().replace('-', '_').replace(' ', '_')}"
                        alert = self.process_violation_event(
                            camera_id=camera_id,
                            violation_type=violation_key,
                            confidence=0.88,
                            track_id=track_id,
                            bbox=bbox_coords,
                            timestamp=ts,
                            frame_image=frame_image,
                            camera_name=cam_name,
                            metadata={"mandatory_missing": ppe_name, "frame_id": assessment.frame_id},
                        )
                        if alert:
                            generated_alerts.append(alert)

        # 2. Process Hazardous Behaviors (Fall, Rapid Movement, Prolonged Stationary)
        for beh in (assessment.behavior_states or []):
            track_id = beh.track_id
            beh_type = beh.primary_behavior.value if hasattr(beh.primary_behavior, "value") else str(beh.primary_behavior)
            norm_beh = beh_type.upper()

            if "FALL" in norm_beh or getattr(beh, "fall_risk_score", 0.0) >= 0.70:
                conf = getattr(beh, "fall_risk_score", 0.85) or 0.85
                alert = self.process_violation_event(
                    camera_id=camera_id,
                    violation_type="FALL_DETECTED",
                    confidence=float(conf),
                    track_id=track_id,
                    timestamp=ts,
                    frame_image=frame_image,
                    camera_name=cam_name,
                    metadata={"behavior": norm_beh, "fall_risk_score": float(conf), "frame_id": assessment.frame_id},
                )
                if alert:
                    generated_alerts.append(alert)

        # 3. Process Active Confirmed Safety Events from Risk Assessment
        if assessment.risk_assessment and assessment.risk_assessment.active_events:
            for ev in assessment.risk_assessment.active_events:
                ev_type = ev.event_type.value if hasattr(ev.event_type, "value") else str(ev.event_type)
                # Map involved track ID if present
                first_track: Optional[int] = None
                for ent in ev.involved_entities:
                    if ent.startswith("person_") or ent.isdigit():
                        digits = re.findall(r"\d+", ent)
                        if digits:
                            first_track = int(digits[0])
                            break

                conf = min(1.0, max(0.40, ev.risk_score / 100.0))
                alert = self.process_violation_event(
                    camera_id=camera_id,
                    violation_type=ev_type,
                    confidence=float(conf),
                    track_id=first_track,
                    timestamp=ts,
                    frame_image=frame_image,
                    camera_name=cam_name,
                    metadata={
                        "risk_score": ev.risk_score,
                        "risk_level": ev.risk_level.value if hasattr(ev.risk_level, "value") else str(ev.risk_level),
                        "explanation": ev.explanation,
                        "frame_id": assessment.frame_id,
                    },
                )
                if alert:
                    generated_alerts.append(alert)

        return generated_alerts

    def process_violation_event(
        self,
        camera_id: str,
        violation_type: str,
        confidence: float,
        track_id: Optional[int] = None,
        bbox: Optional[List[int]] = None,
        timestamp: Optional[float] = None,
        frame_image: Optional[np.ndarray] = None,
        camera_name: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Optional[AlertRecord]:
        """
        Core alerting engine entrypoint for a single detected violation event.
        Handles confidence gating, severity determination, deduplication within cooldown,
        evidence snapshot capture, and persistent ticket synchronization.
        """
        now = timestamp if timestamp is not None and timestamp > 0 else time.time()
        cam_id = camera_id or "cam_01"
        cam_name = camera_name or f"Camera {cam_id}"
        norm_violation = violation_type.upper().replace("-", "_").replace(" ", "_")
        meta = dict(metadata or {})
        if bbox:
            meta["bbox"] = bbox

        # 1. Determine Severity
        severity = self.get_severity_for_violation(norm_violation)
        is_immediate = self.is_critical_immediate(norm_violation, severity)

        # 2. Confidence Gating
        # Immediate critical alerts may pass with lower threshold (e.g. 0.20), others require standard threshold
        required_conf = 0.20 if is_immediate else self.confidence_threshold
        if confidence < required_conf:
            logger.debug(
                f"Ignored low-confidence violation '{norm_violation}' (conf: {confidence:.2f} < {required_conf:.2f})"
            )
            return None

        # 3. Deduplication Key & Cooldown Check
        worker_key = str(track_id) if track_id is not None else "unassigned"
        corr_key = (cam_id, worker_key, norm_violation)

        with self._lock:
            self._prune_stale_correlations(now)

            if corr_key in self._active_correlations:
                entry = self._active_correlations[corr_key]
                time_since_last = now - entry["last_detected_at"]

                # Still within cooldown window
                if time_since_last < self.cooldown_seconds:
                    existing_id = entry["alert_id"]
                    existing_alert = self.store.get_alert(existing_id)

                    if existing_alert and existing_alert.status in (AlertStatus.NEW, AlertStatus.ACKNOWLEDGED):
                        # Aggregate continuing violation
                        existing_alert.last_detected_at = now
                        existing_alert.occurrence_count += 1
                        existing_alert.confidence = max(existing_alert.confidence, float(confidence))
                        if bbox:
                            existing_alert.metadata["bbox"] = bbox
                        if metadata:
                            existing_alert.metadata.update(metadata)

                        self.store.create_or_update_alert(existing_alert)
                        entry["last_detected_at"] = now
                        entry["occurrence_count"] = existing_alert.occurrence_count
                        return existing_alert

            # 4. Create New Alert Ticket
            alert_id = f"alert_{uuid.uuid4().hex[:10]}"

            # 5. Capture Visual Evidence Snapshot
            evidence_rel_path, has_evidence = self._capture_evidence(
                alert_id=alert_id,
                camera_id=cam_id,
                violation_type=norm_violation,
                track_id=track_id,
                confidence=confidence,
                bbox=bbox,
                frame_image=frame_image,
                timestamp=now,
            )

            if "explanation" not in meta or not meta["explanation"]:
                entity_label = f"Person #{track_id}" if track_id is not None else "Worker"
                if "NO_HARDHAT" in norm_violation:
                    meta["explanation"] = f"{entity_label} detected without required Hardhat."
                elif "NO_SAFETY_VEST" in norm_violation:
                    meta["explanation"] = f"{entity_label} detected without required Safety Vest."
                elif "NO_GLOVES" in norm_violation:
                    meta["explanation"] = f"{entity_label} detected without required Protective Gloves."
                elif "NO_MASK" in norm_violation:
                    meta["explanation"] = f"{entity_label} detected without required Safety Mask."
                elif "RESTRICTED_ZONE" in norm_violation or "ZONE" in norm_violation:
                    zname = meta.get("zone_name", "Restricted Area")
                    meta["explanation"] = f"{entity_label} entered {zname} without clearance."
                elif "FALL" in norm_violation:
                    meta["explanation"] = f"Fall event confirmed involving {entity_label}."
                elif "APPROACHING" in norm_violation or "VEHICLE" in norm_violation or "PROXIMITY" in norm_violation:
                    meta["explanation"] = f"{entity_label} in close proximity to moving industrial vehicle."
                else:
                    v_title = norm_violation.replace("_", " ").title()
                    meta["explanation"] = f"{v_title} event confirmed involving {entity_label}."

            new_alert = AlertRecord(
                alert_id=alert_id,
                camera_id=cam_id,
                camera_name=cam_name,
                track_id=track_id,
                violation_type=norm_violation,
                severity=severity,
                confidence=float(round(confidence, 3)),
                status=AlertStatus.NEW,
                timestamp=now,
                first_detected_at=now,
                last_detected_at=now,
                occurrence_count=1,
                has_evidence=has_evidence,
                evidence_image_path=evidence_rel_path,
                metadata=meta,
            )

            # Persist to AlertStore
            persisted = self.store.create_or_update_alert(new_alert)

            # Track in deduplication correlation index
            self._active_correlations[corr_key] = {
                "alert_id": alert_id,
                "first_detected_at": now,
                "last_detected_at": now,
                "occurrence_count": 1,
            }

            logger.info(
                f"Generated [{severity.value}] Alert {alert_id} on {cam_id} for '{norm_violation}' "
                f"(Worker: {worker_key}, Conf: {confidence:.2f}, Evidence: {has_evidence})"
            )
            return persisted

    def _capture_evidence(
        self,
        alert_id: str,
        camera_id: str,
        violation_type: str,
        track_id: Optional[int],
        confidence: float,
        bbox: Optional[List[int]],
        frame_image: Optional[np.ndarray],
        timestamp: float,
    ) -> Tuple[Optional[str], bool]:
        """
        Annotates and saves an evidence image snapshot safely.
        Returns (relative_filepath, success_bool).
        """
        if frame_image is None or not isinstance(frame_image, np.ndarray) or frame_image.size == 0:
            return None, False

        try:
            evidence_img = frame_image.copy()
            h, w = evidence_img.shape[:2]

            # Choose annotation color based on severity
            sev = self.get_severity_for_violation(violation_type)
            if sev == AlertSeverity.CRITICAL:
                box_color = (0, 0, 255)  # BGR Red
            elif sev == AlertSeverity.HIGH:
                box_color = (0, 69, 255)  # Orange-Red
            elif sev == AlertSeverity.MEDIUM:
                box_color = (0, 165, 255)  # Orange
            else:
                box_color = (0, 255, 255)  # Yellow

            # Draw bounding box if provided
            if bbox and len(bbox) == 4:
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(w - 1, int(x1)))
                y1 = max(0, min(h - 1, int(y1)))
                x2 = max(0, min(w - 1, int(x2)))
                y2 = max(0, min(h - 1, int(y2)))

                cv2.rectangle(evidence_img, (x1, y1), (x2, y2), box_color, 2)
                lbl = f"{violation_type} ({confidence:.2f})"
                if track_id is not None:
                    lbl = f"ID:{track_id} | " + lbl

                # Label background
                (tw, th), _ = cv2.getTextSize(lbl, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                lbl_y = max(th + 4, y1 - 4)
                cv2.rectangle(evidence_img, (x1, lbl_y - th - 4), (x1 + tw + 6, lbl_y + 2), box_color, -1)
                cv2.putText(
                    evidence_img,
                    lbl,
                    (x1 + 3, lbl_y - 2),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Top banner timestamp & watermark
            dt_str = datetime.fromtimestamp(timestamp, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
            header_text = f"INTELLIWATCH EVIDENCE | {alert_id} | CAM: {camera_id} | {dt_str}"
            cv2.rectangle(evidence_img, (0, 0), (w, 28), (15, 23, 42), -1)
            cv2.putText(
                evidence_img,
                header_text,
                (10, 19),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (56, 189, 248),
                1,
                cv2.LINE_AA,
            )

            # Safe filename
            safe_cam = re.sub(r"[^a-zA-Z0-9_-]", "_", camera_id)
            safe_id = re.sub(r"[^a-zA-Z0-9_-]", "_", alert_id)
            filename = f"evidence_{safe_cam}_{safe_id}_{int(timestamp)}.jpg"
            dest_path = self.evidence_dir / filename

            # Save JPEG with standard compression
            cv2.imwrite(str(dest_path), evidence_img, [int(cv2.IMWRITE_JPEG_QUALITY), 85])

            # Store path relative to project or absolute
            return str(dest_path), True

        except Exception as e:
            logger.warning(f"Failed to capture visual evidence for alert '{alert_id}': {e}", exc_info=True)
            return None, False

    def _prune_stale_correlations(self, current_time: float) -> None:
        """Evicts expired correlations to keep memory usage strictly bounded."""
        stale_threshold = current_time - (self.cooldown_seconds * 3.0)
        to_delete = [
            k for k, v in self._active_correlations.items()
            if v["last_detected_at"] < stale_threshold
        ]
        for k in to_delete:
            del self._active_correlations[k]

    def reset(self) -> None:
        """Resets active correlations for testing isolation."""
        with self._lock:
            self._active_correlations.clear()


_alert_engine_instance: Optional[AlertEngine] = None
_engine_lock = threading.Lock()


def get_alert_engine() -> AlertEngine:
    """Returns the singleton instance of AlertEngine."""
    global _alert_engine_instance
    if _alert_engine_instance is None:
        with _engine_lock:
            if _alert_engine_instance is None:
                _alert_engine_instance = AlertEngine()
    return _alert_engine_instance
