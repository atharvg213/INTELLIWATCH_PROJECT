"""
intelligence/risk/engine.py
Step 10 - Deterministic Safety Risk & Event Reasoning Engine.
Main orchestrator that consumes multimodal scene-understanding signals (SceneGraph,
PPE compliance, Zone occupancy, Behavior analysis, Depth estimation),
extracts atomic risk factors, evaluates candidate events, performs multi-frame temporal
confirmation, tracks explicit event lifecycles (CANDIDATE -> CONFIRMED -> ACTIVE -> ENDED),
and computes explainable compound risk assessments.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import uuid

from backend.schemas.risk import (
    EventLifecycleState,
    FrameRiskAssessment,
    RiskEvent,
    RiskEventType,
    RiskFactor,
    RiskLevel,
    RiskSummary,
)
from backend.schemas.scene_graph import FrameScene, SceneNodeType
from backend.schemas.ppe import FramePPEAssociation
from backend.schemas.zones import FrameZoneOccupancy
from backend.schemas.behavior import BehaviorState
from backend.schemas.depth import DepthResult
from intelligence.risk.factors import RiskFactorExtractor
from intelligence.risk.rules import CandidateEvent, RiskRuleEvaluator
from intelligence.risk.scorer import RiskScorer
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.risk.engine")


class _TrackedEventState:
    """Internal temporal state tracker for a risk event identity."""

    def __init__(
        self,
        event_id: str,
        event_key: str,
        event_type: RiskEventType,
        involved_entities: List[str],
        first_seen_ts: float,
        initial_factors: List[RiskFactor],
        initial_score: float,
        initial_level: RiskLevel,
        initial_explanation: str,
        initial_evidence: Dict[str, Any],
        is_compound: bool = False,
    ):
        self.event_id = event_id
        self.event_key = event_key
        self.event_type = event_type
        self.involved_entities = involved_entities
        self.first_seen_ts = first_seen_ts
        self.last_seen_ts = first_seen_ts
        self.end_ts: Optional[float] = None

        self.lifecycle_state = EventLifecycleState.CANDIDATE
        self.consecutive_hits = 1
        self.consecutive_misses = 0
        self.cooldown_remaining = 0

        self.factors = initial_factors
        self.score = initial_score
        self.level = initial_level
        self.explanation = initial_explanation
        self.evidence = initial_evidence
        self.is_compound = is_compound


class RiskEngine:
    """
    Deterministic Safety Risk & Event Reasoning Engine.
    Executes entirely on CPU without LLMs or heavy ML re-inference.
    Maintains temporal event confirmation and lifecycle tracking across frames.
    """

    def __init__(
        self,
        confirmation_frames: Optional[int] = None,
        end_confirmation_frames: Optional[int] = None,
        cooldown_frames: Optional[int] = None,
        compound_enabled: Optional[bool] = None,
        settings_override: Optional[Dict[str, Any]] = None,
        camera_id: str = "cam_01",
    ):
        s = get_settings()
        self._overrides = settings_override or {}
        self.camera_id = camera_id

        self.confirmation_frames = (
            confirmation_frames
            if confirmation_frames is not None
            else self._overrides.get("RISK_EVENT_CONFIRMATION_FRAMES", s.RISK_EVENT_CONFIRMATION_FRAMES)
        )
        self.end_confirmation_frames = (
            end_confirmation_frames
            if end_confirmation_frames is not None
            else self._overrides.get("RISK_EVENT_END_CONFIRMATION_FRAMES", s.RISK_EVENT_END_CONFIRMATION_FRAMES)
        )
        self.cooldown_frames = (
            cooldown_frames
            if cooldown_frames is not None
            else self._overrides.get("RISK_EVENT_COOLDOWN_FRAMES", s.RISK_EVENT_COOLDOWN_FRAMES)
        )
        self.compound_enabled = (
            compound_enabled
            if compound_enabled is not None
            else self._overrides.get("RISK_COMPOUND_EVENT_ENABLED", s.RISK_COMPOUND_EVENT_ENABLED)
        )

        self._factor_extractor = RiskFactorExtractor(settings_override=self._overrides)
        self._rule_evaluator = RiskRuleEvaluator(settings_override=self._overrides)
        self._scorer = RiskScorer(settings_override=self._overrides)

        # Internal active tracked event identities: event_key -> _TrackedEventState
        self._tracked_events: Dict[str, _TrackedEventState] = {}
        self._prev_active_entities: Set[str] = set()

    def evaluate_frame(
        self,
        scene: Optional[FrameScene] = None,
        ppe_association: Optional[FramePPEAssociation] = None,
        zone_occupancy: Optional[FrameZoneOccupancy] = None,
        behavior_states: Optional[List[BehaviorState]] = None,
        depth_result: Optional[DepthResult] = None,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> FrameRiskAssessment:
        """
        Processes a single frame's scene understanding outputs to generate
        a complete, explainable FrameRiskAssessment.
        """
        cur_ts = timestamp if timestamp is not None else (scene.timestamp if scene else 0.0)
        cur_fid = frame_id if frame_id is not None else (scene.frame_id if scene else None)

        # 1. Identify currently active entities in the scene to detect departures
        current_entity_ids: Set[str] = set()
        if scene:
            for node in scene.nodes:
                current_entity_ids.add(node.node_id)
        if ppe_association:
            inventories = getattr(ppe_association, "worker_inventories", []) or getattr(ppe_association, "inventories", [])
            for inv in inventories:
                current_entity_ids.add(f"person_{inv.track_id}")
        if behavior_states:
            for b in behavior_states:
                current_entity_ids.add(f"person_{b.track_id}")

        # 2. Extract atomic risk factors
        factors = self._factor_extractor.extract_factors(
            scene=scene,
            ppe_association=ppe_association,
            zone_occupancy=zone_occupancy,
            behavior_states=behavior_states,
            depth_result=depth_result,
            timestamp=cur_ts,
        )

        # 3. Evaluate candidate events from rules
        candidates = self._rule_evaluator.evaluate_candidate_events(
            factors=factors,
            timestamp=cur_ts,
        )
        candidate_map: Dict[str, CandidateEvent] = {c.event_key: c for c in candidates}

        # 4. Temporal Lifecycle Update
        recent_events: List[RiskEvent] = []
        active_events: List[RiskEvent] = []
        keys_to_purge: List[str] = []

        # A. Handle all candidates observed in this frame
        for ekey, cand in candidate_map.items():
            score, level, breakdown = self._scorer.compute_risk(cand.factors, is_compound=cand.is_compound)
            enhanced_evidence = dict(cand.evidence)
            enhanced_evidence["score_breakdown"] = breakdown

            if ekey not in self._tracked_events:
                # Check if in cooldown
                event_id = f"evt_{cand.event_type.value.lower()}_{uuid.uuid4().hex[:8]}"
                state = _TrackedEventState(
                    event_id=event_id,
                    event_key=ekey,
                    event_type=cand.event_type,
                    involved_entities=cand.involved_entities,
                    first_seen_ts=cur_ts,
                    initial_factors=cand.factors,
                    initial_score=score,
                    initial_level=level,
                    initial_explanation=cand.explanation,
                    initial_evidence=enhanced_evidence,
                    is_compound=cand.is_compound,
                )

                if self.confirmation_frames <= 1:
                    state.lifecycle_state = EventLifecycleState.CONFIRMED
                else:
                    state.lifecycle_state = EventLifecycleState.CANDIDATE

                self._tracked_events[ekey] = state

            else:
                state = self._tracked_events[ekey]
                # If currently in cooldown, ignore new hits until cooldown expires
                if state.lifecycle_state == EventLifecycleState.COOLDOWN:
                    continue

                state.last_seen_ts = cur_ts
                state.consecutive_hits += 1
                state.consecutive_misses = 0
                state.factors = cand.factors
                state.score = score
                state.level = level
                state.explanation = cand.explanation
                state.evidence = enhanced_evidence

                if state.lifecycle_state == EventLifecycleState.CANDIDATE:
                    if state.consecutive_hits >= self.confirmation_frames:
                        state.lifecycle_state = EventLifecycleState.CONFIRMED
                elif state.lifecycle_state == EventLifecycleState.CONFIRMED:
                    state.lifecycle_state = EventLifecycleState.ACTIVE
                elif state.lifecycle_state == EventLifecycleState.ENDED:
                    # Re-triggered post-end if cooldown had expired
                    state.lifecycle_state = (
                        EventLifecycleState.CONFIRMED
                        if self.confirmation_frames <= 1
                        else EventLifecycleState.CANDIDATE
                    )
                    state.consecutive_hits = 1

        # B. Handle missing events (unobserved in this frame) or departed tracks
        for ekey, state in list(self._tracked_events.items()):
            if state.lifecycle_state == EventLifecycleState.COOLDOWN:
                state.cooldown_remaining -= 1
                if state.cooldown_remaining <= 0:
                    keys_to_purge.append(ekey)
                continue

            if ekey not in candidate_map:
                # Check if involved entities departed scene
                has_frame_input = scene is not None or ppe_association is not None or behavior_states is not None
                departed = has_frame_input and any(
                    ent.startswith("person_") and ent not in current_entity_ids
                    for ent in state.involved_entities
                )

                if departed:
                    # Immediate end on track departure to avoid ghost alerts
                    if state.lifecycle_state in (EventLifecycleState.CONFIRMED, EventLifecycleState.ACTIVE):
                        state.lifecycle_state = EventLifecycleState.ENDED
                        state.end_ts = cur_ts
                        state.cooldown_remaining = self.cooldown_frames
                    elif state.lifecycle_state == EventLifecycleState.CANDIDATE:
                        keys_to_purge.append(ekey)
                else:
                    state.consecutive_misses += 1
                    if state.lifecycle_state == EventLifecycleState.CANDIDATE:
                        # Candidate condition vanished before confirmation -> drop
                        keys_to_purge.append(ekey)
                    elif state.lifecycle_state in (EventLifecycleState.CONFIRMED, EventLifecycleState.ACTIVE):
                        if state.consecutive_misses >= self.end_confirmation_frames:
                            state.lifecycle_state = EventLifecycleState.ENDED
                            state.end_ts = cur_ts
                            state.cooldown_remaining = self.cooldown_frames


        # C. Materialize RiskEvent schemas
        for ekey, state in self._tracked_events.items():
            if ekey in keys_to_purge:
                continue

            # Skip candidate events from public active/recent output (they are awaiting confirmation)
            if state.lifecycle_state == EventLifecycleState.CANDIDATE:
                continue

            rev = RiskEvent(
                event_id=state.event_id,
                event_type=state.event_type,
                lifecycle_state=state.lifecycle_state,
                timestamp=cur_ts,
                start_timestamp=state.first_seen_ts,
                end_timestamp=state.end_ts,
                involved_entities=state.involved_entities,
                risk_level=state.level,
                risk_score=state.score,
                risk_factors=state.factors,
                evidence=state.evidence,
                explanation=state.explanation,
                metadata={"event_key": state.event_key, "is_compound": state.is_compound},
            )

            if state.lifecycle_state in (EventLifecycleState.CONFIRMED, EventLifecycleState.ACTIVE):
                active_events.append(rev)

            if state.lifecycle_state in (EventLifecycleState.CONFIRMED, EventLifecycleState.ENDED):
                recent_events.append(rev)

        # Clean up purged events and transition ended to cooldown
        for k in keys_to_purge:
            del self._tracked_events[k]

        for ekey, state in self._tracked_events.items():
            if state.lifecycle_state == EventLifecycleState.ENDED and state.cooldown_remaining > 0:
                state.lifecycle_state = EventLifecycleState.COOLDOWN

        # 5. Compute Risk Summary across active events
        summary_counts = {
            "info": 0,
            "low": 0,
            "medium": 0,
            "high": 0,
            "critical": 0,
        }
        max_level = RiskLevel.INFO
        max_score = 0.0

        level_order = [RiskLevel.INFO, RiskLevel.LOW, RiskLevel.MEDIUM, RiskLevel.HIGH, RiskLevel.CRITICAL]

        for ev in active_events:
            lvl_key = ev.risk_level.value.lower()
            if lvl_key in summary_counts:
                summary_counts[lvl_key] += 1
            if ev.risk_score > max_score:
                max_score = ev.risk_score
            if level_order.index(ev.risk_level) > level_order.index(max_level):
                max_level = ev.risk_level

        risk_summary = RiskSummary(
            info=summary_counts["info"],
            low=summary_counts["low"],
            medium=summary_counts["medium"],
            high=summary_counts["high"],
            critical=summary_counts["critical"],
            total_active_events=len(active_events),
            max_risk_level=max_level,
            max_risk_score=max_score,
        )

        assessment = FrameRiskAssessment(
            frame_id=cur_fid,
            timestamp=cur_ts,
            camera_id=self.camera_id,
            active_events=active_events,
            recent_events=recent_events,
            risk_factors=factors,
            risk_summary=risk_summary,
        )

        # Update singleton store
        from backend.services.risk_store import get_risk_store
        get_risk_store().set_current_assessment(assessment)

        self._prev_active_entities = current_entity_ids
        return assessment

    def reset(self) -> None:
        """Resets all internal temporal tracking states."""
        self._tracked_events.clear()
        self._prev_active_entities.clear()
