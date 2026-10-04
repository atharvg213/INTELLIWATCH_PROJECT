"""
backend/services/temporal_log.py
Step 15 — Temporal Event Log Service.

A stateful singleton service that records meaningful state/relationship
transitions over the lifecycle of a video analysis session.

Design principles:
- Only records SIGNIFICANT state changes (not every frame).
- Bounded memory: keeps at most MAX_ENTRIES entries total.
- Per-entity timeline capped at TIMELINE_PER_ENTITY entries.
- Thread-safe via simple Python GIL (no async complexity).
- No ML inference; purely event-driven bookkeeping.
- Technical honesty: never claims metric distances, physical speeds, or
  guaranteed accident prediction.
"""
import threading
from collections import defaultdict, deque
from typing import Dict, List, Optional, Tuple

from backend.schemas.assessment import FrameAssessment
from backend.schemas.behavior import BehaviorState, PrimaryBehavior
from backend.schemas.risk import RiskEvent, RiskLevel
from backend.schemas.scene_graph import FrameScene, SceneRelationType
from backend.schemas.temporal import (
    TemporalEntityProfile,
    TemporalEventCategory,
    TemporalLogEntry,
)

_RISK_LEVEL_ORDER = {
    RiskLevel.INFO:     0,
    RiskLevel.LOW:      1,
    RiskLevel.MEDIUM:   2,
    RiskLevel.HIGH:     3,
    RiskLevel.CRITICAL: 4,
}

MAX_ENTRIES        = 500   # Session-wide cap
TIMELINE_PER_ENTITY = 20   # Per-entity timeline cap
RISK_TREND_WINDOW  = 6     # How many recent risk levels to summarize


class TemporalLogService:
    """
    Incremental temporal event logger.
    Call `update(assessment)` once per frame in the pipeline.
    """

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._seq: int = 0
        self._entries: deque = deque(maxlen=MAX_ENTRIES)

        # Per-entity state snapshots (keyed by entity_id, e.g. 'person_1')
        self._entity_behaviors:     Dict[str, str]   = {}   # entity_id -> last behavior
        self._entity_spatial:       Dict[str, Tuple[Optional[str], Optional[str]]] = {}
        # (proximity_tier, target_id)
        self._entity_risk_levels:   Dict[str, deque]  = defaultdict(lambda: deque(maxlen=RISK_TREND_WINDOW))
        self._entity_first_seen:    Dict[str, float]  = {}
        self._entity_last_seen:     Dict[str, float]  = {}
        self._entity_class:         Dict[str, str]    = {}
        self._entity_track_id:      Dict[str, int]    = {}

        # Risk level of each active event (event_id -> risk_level)
        self._active_event_risk:    Dict[str, RiskLevel] = {}

        # Scene relations: relation_id -> (rel_type, source, target)
        self._active_rels:          Dict[str, Tuple[str, str, str]] = {}

        # Per-entity timeline deques
        self._entity_timelines:     Dict[str, deque] = defaultdict(lambda: deque(maxlen=TIMELINE_PER_ENTITY))

        # Per-entity early-warning cache (indicator_id set)
        self._entity_warnings:      Dict[str, set]   = defaultdict(set)

    # ------------------------------------------------------------------
    # Main public API
    # ------------------------------------------------------------------

    def update(self, assessment: FrameAssessment) -> List[TemporalLogEntry]:
        """
        Processes one FrameAssessment and returns newly generated log entries.
        Called once per frame by the orchestrator or video pipeline.
        """
        with self._lock:
            new_entries: List[TemporalLogEntry] = []
            ts  = assessment.timestamp
            fid = assessment.frame_id

            # 1. Behavior transitions
            for bs in assessment.behavior_states:
                new_entries.extend(self._check_behavior_transition(bs, ts, fid))

            # 2. Spatial relationship transitions (from scene graph)
            if assessment.scene:
                new_entries.extend(self._check_spatial_transitions(assessment.scene, ts, fid))

            # 3. Risk escalation / decay
            new_entries.extend(self._check_risk_transitions(
                assessment.active_events, assessment.tracks, ts, fid))

            # 4. Early-warning indicators
            new_entries.extend(self._check_early_warnings(
                assessment.active_early_warnings, ts, fid))

            # 5. Incident confirmations
            for inc_id in assessment.new_incident_ids:
                entry = self._make_entry(
                    ts=ts, fid=fid,
                    cat=TemporalEventCategory.INCIDENT_CONFIRMED,
                    entity_id="system",
                    label=f"Incident confirmed: {inc_id}",
                    description=f"Safety incident evidence recorded with ID '{inc_id}' at frame {fid}.",
                    new_state=inc_id,
                )
                new_entries.append(entry)

            # 6. Update per-entity first/last seen + class lookup from tracks
            for t in assessment.tracks:
                eid = self._track_to_entity_id(t.class_name, t.track_id)
                if eid not in self._entity_first_seen:
                    self._entity_first_seen[eid] = ts
                    entry = self._make_entry(
                        ts=ts, fid=fid,
                        cat=TemporalEventCategory.ENTITY_DETECTED,
                        entity_id=eid,
                        label=f"{t.class_name.title()} #{t.track_id} detected",
                        description=(
                            f"New entity '{eid}' ({t.class_name}) appeared in the scene at"
                            f" frame {fid} (t={ts:.2f}s)."
                        ),
                        new_state="DETECTED",
                    )
                    new_entries.append(entry)
                self._entity_last_seen[eid]  = ts
                self._entity_class[eid]      = t.class_name
                self._entity_track_id[eid]   = t.track_id

            # Persist new entries
            for e in new_entries:
                self._entries.append(e)
                self._entity_timelines[e.entity_id].append(e)

            return new_entries

    def get_all_entries(self) -> List[TemporalLogEntry]:
        """Returns a snapshot of all log entries (newest last)."""
        with self._lock:
            return list(self._entries)

    def get_entity_profile(self, entity_id: str,
                           assessment: Optional[FrameAssessment] = None) -> Optional[TemporalEntityProfile]:
        """Builds a TemporalEntityProfile from accumulated state."""
        with self._lock:
            if entity_id not in self._entity_first_seen:
                return None

            # Build risk trend label
            risk_hist = list(self._entity_risk_levels.get(entity_id, []))
            risk_trend = " → ".join(l.value for l in risk_hist) if risk_hist else None

            # Current behavior / spatial
            behavior    = self._entity_behaviors.get(entity_id, "UNKNOWN")
            spat        = self._entity_spatial.get(entity_id, (None, None))
            prox_tier, tgt_id = spat

            # Behavior duration from assessment if available
            beh_dur = 0.0
            img_speed = None
            heading   = None
            depth_trend = None
            if assessment:
                for bs in assessment.behavior_states:
                    eid = self._track_to_entity_id(bs.track_id)
                    if eid == entity_id or bs.track_id == self._entity_track_id.get(entity_id):
                        beh_dur     = bs.behavior_duration_s
                        img_speed   = bs.image_speed_px_per_s
                        heading     = bs.heading_degrees
                        depth_trend = bs.depth_trend
                        break

            active_warnings = [w for w in self._entity_warnings.get(entity_id, set())]

            return TemporalEntityProfile(
                entity_id=entity_id,
                track_id=self._entity_track_id.get(entity_id, -1),
                class_name=self._entity_class.get(entity_id, "unknown"),
                first_seen_ts=self._entity_first_seen.get(entity_id, 0.0),
                last_seen_ts=self._entity_last_seen.get(entity_id, 0.0),
                current_behavior=behavior,
                behavior_duration_s=beh_dur,
                current_spatial_state=prox_tier,
                spatial_target_id=tgt_id,
                image_speed_px_per_s=img_speed,
                heading_degrees=heading,
                depth_trend=depth_trend,
                risk_trend_label=risk_trend,
                active_early_warnings=active_warnings,
                timeline=list(self._entity_timelines.get(entity_id, [])),
            )

    def get_all_entity_profiles(self, assessment: Optional[FrameAssessment] = None) -> List[TemporalEntityProfile]:
        """Returns profiles for all known entities."""
        with self._lock:
            return [
                self.get_entity_profile(eid, assessment)
                for eid in self._entity_first_seen
                if self.get_entity_profile(eid, assessment) is not None
            ]

    def recent_timeline(self, n: int = 20) -> List[TemporalLogEntry]:
        """Returns the N most recent entries across all entities."""
        with self._lock:
            all_e = list(self._entries)
            return all_e[-n:]

    def reset(self) -> None:
        """Resets all state (call on new video/session start)."""
        with self._lock:
            self._seq = 0
            self._entries.clear()
            self._entity_behaviors.clear()
            self._entity_spatial.clear()
            self._entity_risk_levels.clear()
            self._entity_first_seen.clear()
            self._entity_last_seen.clear()
            self._entity_class.clear()
            self._entity_track_id.clear()
            self._active_event_risk.clear()
            self._active_rels.clear()
            self._entity_timelines.clear()
            self._entity_warnings.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _next_seq(self) -> int:
        self._seq += 1
        return self._seq

    @staticmethod
    def _track_to_entity_id(class_name_or_id, track_id: Optional[int] = None) -> str:
        """Derives a canonical entity_id matching the scene graph node_id format."""
        if track_id is None:
            # Called with just track_id as first arg (int)
            return f"track_{class_name_or_id}"
        name = str(class_name_or_id).lower().replace(" ", "_")
        return f"{name}_{track_id}"

    def _make_entry(
        self,
        ts: float, fid: int,
        cat: TemporalEventCategory,
        entity_id: str,
        label: str,
        description: str,
        new_state: str,
        prev_state: Optional[str] = None,
        target_id: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> TemporalLogEntry:
        return TemporalLogEntry(
            seq=self._next_seq(),
            timestamp=ts,
            frame_id=fid,
            category=cat,
            entity_id=entity_id,
            target_id=target_id,
            label=label,
            description=description,
            prev_state=prev_state,
            new_state=new_state,
            metadata=metadata or {},
        )

    def _check_behavior_transition(
        self, bs: BehaviorState, ts: float, fid: int
    ) -> List[TemporalLogEntry]:
        entries = []
        eid = self._track_to_entity_id(bs.track_id)
        new_beh = bs.primary_behavior.value
        prev_beh = self._entity_behaviors.get(eid)

        if prev_beh != new_beh:
            self._entity_behaviors[eid] = new_beh
            if prev_beh is None:
                # First classification — only log interesting states
                if new_beh not in ("UNKNOWN",):
                    entries.append(self._make_entry(
                        ts=ts, fid=fid,
                        cat=TemporalEventCategory.BEHAVIOR_TRANSITION,
                        entity_id=eid,
                        label=f"Track #{bs.track_id}: {new_beh}",
                        description=f"Track #{bs.track_id} initial behavior confirmed: {new_beh}.",
                        new_state=new_beh,
                        metadata={"speed_px_s": bs.image_speed_px_per_s},
                    ))
            else:
                speed_info = ""
                if bs.image_speed_px_per_s is not None:
                    speed_info = f" Image-space speed: {bs.image_speed_px_per_s:.1f} px/s."
                entries.append(self._make_entry(
                    ts=ts, fid=fid,
                    cat=TemporalEventCategory.BEHAVIOR_TRANSITION,
                    entity_id=eid,
                    label=f"Track #{bs.track_id}: {prev_beh} → {new_beh}",
                    description=(
                        f"Track #{bs.track_id} transitioned from {prev_beh} to {new_beh} at t={ts:.2f}s.{speed_info}"
                    ),
                    prev_state=prev_beh,
                    new_state=new_beh,
                    metadata={"speed_px_s": bs.image_speed_px_per_s, "stationary_s": bs.stationary_duration_s},
                ))
        return entries

    def _check_spatial_transitions(
        self, scene: FrameScene, ts: float, fid: int
    ) -> List[TemporalLogEntry]:
        entries = []
        SPATIAL_RELS = {
            SceneRelationType.NEAR, SceneRelationType.FAR,
            SceneRelationType.APPROACHING,
        }

        for rel in scene.relationships:
            if rel.relation_type not in SPATIAL_RELS:
                continue
            src = rel.source_node_id
            tgt = rel.target_node_id
            prox_tier = (
                rel.evidence.get("proximity_state")
                or rel.metadata.get("spatial_state")
                or rel.relation_type.value
            )

            prev_spat = self._entity_spatial.get(src)
            prev_tier = prev_spat[0] if prev_spat else None

            if prev_tier != prox_tier or (prev_spat and prev_spat[1] != tgt):
                self._entity_spatial[src] = (prox_tier, tgt)
                cat = (TemporalEventCategory.RELATIONSHIP_START
                       if prev_tier is None
                       else TemporalEventCategory.SPATIAL_TRANSITION)
                dist_px = rel.evidence.get("distance_px")
                dist_info = f" (image-space separation: {dist_px:.0f} px)" if dist_px else ""
                entries.append(self._make_entry(
                    ts=ts, fid=fid,
                    cat=cat,
                    entity_id=src,
                    target_id=tgt,
                    label=f"{src} → {rel.relation_type.value} → {tgt}",
                    description=(
                        f"Relative proximity from {src} to {tgt} is now {prox_tier}"
                        f" (relation: {rel.relation_type.value}){dist_info}."
                    ),
                    prev_state=prev_tier,
                    new_state=prox_tier,
                    metadata={"distance_px": dist_px, "relation_type": rel.relation_type.value},
                ))
        return entries

    def _check_risk_transitions(
        self,
        active_events: list,
        tracks: list,
        ts: float,
        fid: int,
    ) -> List[TemporalLogEntry]:
        entries = []
        # Build a mapping from entity_id -> highest risk level this frame
        entity_highest: Dict[str, RiskLevel] = {}
        for ev in active_events:
            for ent in ev.involved_entities:
                lvl = ev.risk_level
                cur = entity_highest.get(ent)
                if cur is None or _RISK_LEVEL_ORDER[lvl] > _RISK_LEVEL_ORDER[cur]:
                    entity_highest[ent] = lvl

        for ent, lvl in entity_highest.items():
            hist = self._entity_risk_levels[ent]
            prev_lvl = hist[-1] if hist else None
            hist.append(lvl)

            if prev_lvl is None:
                continue  # first observation, no transition to log

            prev_ord = _RISK_LEVEL_ORDER[prev_lvl]
            new_ord  = _RISK_LEVEL_ORDER[lvl]

            if new_ord > prev_ord:
                entries.append(self._make_entry(
                    ts=ts, fid=fid,
                    cat=TemporalEventCategory.RISK_ESCALATION,
                    entity_id=ent,
                    label=f"Risk escalated: {prev_lvl.value} → {lvl.value}",
                    description=(
                        f"Risk level for entity '{ent}' escalated from {prev_lvl.value} to"
                        f" {lvl.value} at frame {fid} (t={ts:.2f}s)."
                    ),
                    prev_state=prev_lvl.value,
                    new_state=lvl.value,
                    metadata={"frame_id": fid},
                ))
            elif new_ord < prev_ord:
                entries.append(self._make_entry(
                    ts=ts, fid=fid,
                    cat=TemporalEventCategory.RISK_DECAY,
                    entity_id=ent,
                    label=f"Risk reduced: {prev_lvl.value} → {lvl.value}",
                    description=(
                        f"Risk level for entity '{ent}' decreased from {prev_lvl.value} to"
                        f" {lvl.value} at frame {fid}."
                    ),
                    prev_state=prev_lvl.value,
                    new_state=lvl.value,
                ))
        return entries

    def _check_early_warnings(self, indicators: list, ts: float, fid: int) -> List[TemporalLogEntry]:
        entries = []
        for ind in indicators:
            ind_id = getattr(ind, "indicator_id", None)
            if ind_id is None:
                continue
            # Determine associated entity
            entity_ids = getattr(ind, "entity_ids", []) or []
            ent = entity_ids[0] if entity_ids else "system"
            if ind_id not in self._entity_warnings[ent]:
                self._entity_warnings[ent].add(ind_id)
                ind_type = (
                    ind.indicator_type.value
                    if hasattr(ind.indicator_type, "value")
                    else str(ind.indicator_type)
                )
                entries.append(self._make_entry(
                    ts=ts, fid=fid,
                    cat=TemporalEventCategory.EARLY_WARNING,
                    entity_id=ent,
                    label=f"Early warning: {ind_type}",
                    description=getattr(ind, "explanation", f"Early-warning indicator '{ind_type}' detected."),
                    new_state=ind_type,
                    metadata={"indicator_id": ind_id},
                ))
        return entries


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------
_temporal_log_instance: Optional[TemporalLogService] = None
_temporal_log_lock = threading.Lock()


def get_temporal_log() -> TemporalLogService:
    """Returns the module-level singleton TemporalLogService."""
    global _temporal_log_instance
    if _temporal_log_instance is None:
        with _temporal_log_lock:
            if _temporal_log_instance is None:
                _temporal_log_instance = TemporalLogService()
    return _temporal_log_instance
