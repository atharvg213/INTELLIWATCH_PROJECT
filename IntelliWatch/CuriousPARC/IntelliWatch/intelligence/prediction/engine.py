"""
intelligence/prediction/engine.py
Step 11 - Predictive & Advanced Anomaly Intelligence Engine.
Orchestrates short-horizon trajectory projection, hazard-direction analysis,
risk escalation modeling, and explainable behavioral anomaly detection.
Maintains temporal confirmation lifecycles (CANDIDATE -> CONFIRMED -> ACTIVE -> ENDED)
to prevent flickering false alarms.
"""
from typing import Any, Dict, List, Optional, Set, Tuple
import logging
import uuid

from backend.schemas.prediction import (
    EarlyWarningIndicator,
    EarlyWarningIndicatorType,
    FramePredictionAssessment,
    IndicatorLifecycleState,
    IndicatorSeverity,
    ProjectedTrajectory,
)
from backend.schemas.tracking import FrameTracks, TrackedObject
from backend.schemas.scene_graph import FrameScene, SceneNodeType, SceneRelationType
from backend.schemas.risk import FrameRiskAssessment, RiskEventType
from backend.schemas.behavior import BehaviorState
from backend.schemas.zones import RestrictedZone
from intelligence.prediction.trajectory import TrajectoryAnalyzer
from intelligence.prediction.risk_trend import RiskTrendAnalyzer
from intelligence.prediction.anomaly import TemporalAnomalyDetector
from intelligence.prediction.indicators import CandidateIndicator
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.prediction.engine")


class _TrackedIndicatorState:
    """Internal temporal state for an early-warning indicator instance."""

    def __init__(
        self,
        indicator_id: str,
        indicator_key: str,
        indicator_type: EarlyWarningIndicatorType,
        track_ids: List[int],
        entity_ids: List[str],
        severity: IndicatorSeverity,
        explanation: str,
        evidence: Dict[str, Any],
        first_seen_ts: float,
        target_zone_id: Optional[str] = None,
        projected_trajectory: Optional[ProjectedTrajectory] = None,
        trend: Optional[str] = None,
        supporting_events: Optional[List[str]] = None,
        supporting_risk_factors: Optional[List[str]] = None,
    ):
        self.indicator_id = indicator_id
        self.indicator_key = indicator_key
        self.indicator_type = indicator_type
        self.track_ids = track_ids
        self.entity_ids = entity_ids
        self.severity = severity
        self.explanation = explanation
        self.evidence = evidence
        self.first_seen_ts = first_seen_ts
        self.last_seen_ts = first_seen_ts
        self.end_ts: Optional[float] = None

        self.target_zone_id = target_zone_id
        self.projected_trajectory = projected_trajectory
        self.trend = trend
        self.supporting_events = supporting_events or []
        self.supporting_risk_factors = supporting_risk_factors or []

        self.lifecycle_state = IndicatorLifecycleState.CANDIDATE
        self.consecutive_hits = 1
        self.consecutive_misses = 0


class PredictionEngine:
    """
    Early-Warning Predictive & Anomaly Intelligence Engine.
    Executes lightweight deterministic reasoning over existing tracking, scene graph,
    and risk history without running new ML inference models.
    """

    def __init__(
        self,
        confirmation_frames: Optional[int] = None,
        end_confirmation_frames: Optional[int] = None,
        horizon_frames: Optional[int] = None,
        settings_override: Optional[Dict[str, Any]] = None,
        camera_id: str = "cam_01",
    ):
        s = get_settings()
        self._overrides = settings_override or {}
        self.camera_id = camera_id

        self.confirmation_frames = (
            confirmation_frames
            if confirmation_frames is not None
            else self._overrides.get("PREDICTION_CONFIRMATION_FRAMES", s.PREDICTION_CONFIRMATION_FRAMES)
        )
        self.end_confirmation_frames = (
            end_confirmation_frames
            if end_confirmation_frames is not None
            else self._overrides.get("PREDICTION_END_CONFIRMATION_FRAMES", s.PREDICTION_END_CONFIRMATION_FRAMES)
        )
        self.horizon_frames = (
            horizon_frames
            if horizon_frames is not None
            else self._overrides.get("PREDICTION_HORIZON_FRAMES", s.PREDICTION_HORIZON_FRAMES)
        )

        self._trajectory_analyzer = TrajectoryAnalyzer(settings_override=self._overrides)
        self._risk_trend_analyzer = RiskTrendAnalyzer(settings_override=self._overrides)
        self._anomaly_detector = TemporalAnomalyDetector(settings_override=self._overrides)

        # indicator_key -> _TrackedIndicatorState
        self._tracked_indicators: Dict[str, _TrackedIndicatorState] = {}
        self._prev_active_track_ids: Set[int] = set()

    def evaluate_frame(
        self,
        frame_tracks: Optional[FrameTracks] = None,
        scene: Optional[FrameScene] = None,
        risk_assessment: Optional[FrameRiskAssessment] = None,
        behavior_states: Optional[List[BehaviorState]] = None,
        zones: Optional[List[RestrictedZone]] = None,
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
        fps: float = 30.0,
    ) -> FramePredictionAssessment:
        """
        Executes Step 11 early-warning reasoning across perception and risk states.
        """
        cur_ts = (
            timestamp
            if timestamp is not None
            else (
                frame_tracks.timestamp
                if frame_tracks
                else (scene.timestamp if scene else (risk_assessment.timestamp if risk_assessment else 0.0))
            )
        )
        cur_fid = (
            frame_id
            if frame_id is not None
            else (
                frame_tracks.frame_id
                if frame_tracks
                else (scene.frame_id if scene else (risk_assessment.frame_id if risk_assessment else None))
            )
        )

        current_track_ids: Set[int] = set()
        active_tracks: List[TrackedObject] = []
        if frame_tracks:
            active_tracks = frame_tracks.active_tracks
            current_track_ids = {t.track_id for t in active_tracks}
        elif scene:
            for node in scene.nodes:
                if node.track_id is not None:
                    current_track_ids.add(node.track_id)

        # ---------------------------------------------------------------------
        # 1. Trajectory Projection & Hazard-Direction Analysis
        # ---------------------------------------------------------------------
        projected_trajectories: List[ProjectedTrajectory] = []
        projection_by_track: Dict[int, ProjectedTrajectory] = {}
        candidates: List[CandidateIndicator] = []

        for track in active_tracks:
            proj = self._trajectory_analyzer.project_trajectory(
                track=track,
                horizon_frames=self.horizon_frames,
                fps=fps,
            )
            if proj is not None:
                projected_trajectories.append(proj)
                projection_by_track[track.track_id] = proj

        # A. Trajectory toward Restricted Zones
        if zones and projected_trajectories:
            for proj in projected_trajectories:
                for z in zones:
                    if not z.enabled:
                        continue
                    toward_zone, geom_evidence = self._trajectory_analyzer.check_trajectory_toward_zone(
                        projection=proj,
                        zone=z,
                    )
                    if toward_zone:
                        ikey = f"TRAJECTORY_TOWARD_RESTRICTED_ZONE:{proj.track_id}:{z.zone_id}"
                        tth_str = f" Estimated time to boundary: {geom_evidence['time_to_hazard_seconds']}s." if geom_evidence.get("time_to_hazard_seconds") is not None else ""
                        expl = (
                            f"Worker #{proj.track_id} projected trajectory is converging toward restricted zone "
                            f"'{z.name}' (distance: {geom_evidence.get('current_distance_px')}px -> "
                            f"{geom_evidence.get('projected_distance_px')}px).{tth_str}"
                        )
                        candidates.append(
                            CandidateIndicator(
                                indicator_key=ikey,
                                indicator_type=EarlyWarningIndicatorType.TRAJECTORY_TOWARD_RESTRICTED_ZONE,
                                track_ids=[proj.track_id],
                                entity_ids=[f"person_{proj.track_id}", f"zone_{z.zone_id}"],
                                target_zone_id=z.zone_id,
                                severity=IndicatorSeverity.HIGH if geom_evidence.get("will_intersect_boundary") else IndicatorSeverity.MEDIUM,
                                explanation=expl,
                                evidence=geom_evidence,
                                projected_trajectory=proj,
                                trend="converging",
                            )
                        )

        # B. Trajectory toward Vehicles & Machines (using SceneGraph nodes)
        if scene and projected_trajectories:
            vehicle_nodes = [n for n in scene.nodes if n.node_type == SceneNodeType.VEHICLE and n.centroid]
            machine_nodes = [n for n in scene.nodes if n.node_type == SceneNodeType.MACHINE and n.centroid]

            for proj in projected_trajectories:
                # Toward Vehicle
                for v in vehicle_nodes:
                    if v.track_id == proj.track_id:
                        continue
                    toward_veh, veh_ev = self._trajectory_analyzer.check_trajectory_toward_point(
                        projection=proj,
                        target_point=v.centroid,
                    )
                    if toward_veh:
                        ikey = f"TRAJECTORY_TOWARD_VEHICLE:{proj.track_id}:{v.node_id}"
                        tth_str = f" Estimated time to contact: {veh_ev['time_to_hazard_seconds']}s." if veh_ev.get("time_to_hazard_seconds") is not None else ""
                        expl = f"Worker #{proj.track_id} trajectory is moving toward vehicle ({v.node_id}).{tth_str}"
                        candidates.append(
                            CandidateIndicator(
                                indicator_key=ikey,
                                indicator_type=EarlyWarningIndicatorType.TRAJECTORY_TOWARD_VEHICLE,
                                track_ids=[proj.track_id] + ([v.track_id] if v.track_id else []),
                                entity_ids=[f"person_{proj.track_id}", v.node_id],
                                severity=IndicatorSeverity.MEDIUM,
                                explanation=expl,
                                evidence=veh_ev,
                                projected_trajectory=proj,
                                trend="approaching",
                            )
                        )

                # Toward Machine
                for m in machine_nodes:
                    toward_mach, mach_ev = self._trajectory_analyzer.check_trajectory_toward_point(
                        projection=proj,
                        target_point=m.centroid,
                    )
                    if toward_mach:
                        ikey = f"TRAJECTORY_TOWARD_MACHINE:{proj.track_id}:{m.node_id}"
                        expl = f"Worker #{proj.track_id} trajectory is moving toward machinery ({m.node_id})."
                        candidates.append(
                            CandidateIndicator(
                                indicator_key=ikey,
                                indicator_type=EarlyWarningIndicatorType.TRAJECTORY_TOWARD_MACHINE,
                                track_ids=[proj.track_id],
                                entity_ids=[f"person_{proj.track_id}", m.node_id],
                                severity=IndicatorSeverity.MEDIUM,
                                explanation=expl,
                                evidence=mach_ev,
                                projected_trajectory=proj,
                                trend="approaching",
                            )
                        )

        # ---------------------------------------------------------------------
        # 2. Risk Trend & Escalation Analysis
        # ---------------------------------------------------------------------
        if risk_assessment:
            trends = self._risk_trend_analyzer.update_and_evaluate(risk_assessment)

            for ind_type_str, ev_list in trends.items():
                ind_type = EarlyWarningIndicatorType[ind_type_str]
                for item in ev_list:
                    ent = item["entity_id"]
                    tid = int(ent.split("_")[1]) if ent.startswith("person_") and ent.split("_")[1].isdigit() else None
                    tids = [tid] if tid is not None else []
                    ikey = f"{ind_type_str}:{ent}"

                    # Determine severity and explanation
                    if ind_type == EarlyWarningIndicatorType.RISK_ESCALATING:
                        sev = IndicatorSeverity.HIGH if item.get("current_score", 0.0) >= 60.0 else IndicatorSeverity.MEDIUM
                        expl = f"Sustained risk escalation detected for {ent}: score increased by +{item.get('score_delta')} pts."
                    elif ind_type == EarlyWarningIndicatorType.PERSISTENT_UNSAFE_PATTERN:
                        sev = IndicatorSeverity.HIGH
                        expl = f"Persistent unsafe pattern: {ent} maintained elevated risk across {item.get('unsafe_observations')} observations."
                    elif ind_type == EarlyWarningIndicatorType.REPEATED_PPE_VIOLATION:
                        sev = IndicatorSeverity.MEDIUM
                        expl = f"Repeated PPE non-compliance: {ent} observed missing mandatory safety gear across {item.get('violation_count')} frames."
                    elif ind_type == EarlyWarningIndicatorType.REPEATED_ZONE_VIOLATION:
                        sev = IndicatorSeverity.HIGH
                        expl = f"Repeated restricted zone intrusion: {ent} entered hazard areas across {item.get('violation_count')} frames."
                    elif ind_type == EarlyWarningIndicatorType.PERSISTENT_VEHICLE_PROXIMITY:
                        sev = IndicatorSeverity.HIGH
                        expl = f"Persistent vehicle proximity: {ent} remained in close vicinity of moving machinery."
                    elif ind_type == EarlyWarningIndicatorType.PERSISTENT_MACHINE_PROXIMITY:
                        sev = IndicatorSeverity.MEDIUM
                        expl = f"Persistent machine proximity: {ent} remained near industrial machinery."
                    elif ind_type == EarlyWarningIndicatorType.REPEATED_COMPOUND_RISK:
                        sev = IndicatorSeverity.CRITICAL
                        expl = f"Repeated multi-factor compound risk: {ent} generated multiple compound hazard events."
                    else:
                        sev = IndicatorSeverity.LOW
                        expl = f"Early warning trend {ind_type_str} detected for {ent}."

                    candidates.append(
                        CandidateIndicator(
                            indicator_key=ikey,
                            indicator_type=ind_type,
                            track_ids=tids,
                            entity_ids=[ent],
                            severity=sev,
                            explanation=expl,
                            evidence=item,
                            trend="escalating" if "ESCALATING" in ind_type_str else "persistent",
                        )
                    )

        # ---------------------------------------------------------------------
        # 3. Behavioral Temporal Anomaly Detection
        # ---------------------------------------------------------------------
        if behavior_states:
            anomalies = self._anomaly_detector.evaluate_behavior_anomalies(
                behavior_states=behavior_states,
                timestamp=cur_ts,
            )
            for anom in anomalies:
                tid = anom["track_id"]
                ikey = f"TEMPORAL_BEHAVIOR_ANOMALY:person_{tid}:{anom['anomaly_type']}"
                candidates.append(
                    CandidateIndicator(
                        indicator_key=ikey,
                        indicator_type=EarlyWarningIndicatorType.TEMPORAL_BEHAVIOR_ANOMALY,
                        track_ids=[tid],
                        entity_ids=[anom["entity_id"]],
                        severity=IndicatorSeverity.HIGH if anom["anomaly_type"] == "FALL_AND_COLLAPSE_SEQUENCE" else IndicatorSeverity.MEDIUM,
                        explanation=anom["explanation"],
                        evidence=anom,
                        trend="anomalous",
                    )
                )

        candidate_map: Dict[str, CandidateIndicator] = {c.indicator_key: c for c in candidates}

        # ---------------------------------------------------------------------
        # 4. Temporal Confirmation & Indicator Lifecycle
        # ---------------------------------------------------------------------
        active_indicators: List[EarlyWarningIndicator] = []
        recent_indicators: List[EarlyWarningIndicator] = []
        keys_to_purge: List[str] = []

        # A. Update seen candidates
        for ikey, cand in candidate_map.items():
            if ikey not in self._tracked_indicators:
                ind_id = f"ind_{cand.indicator_type.value.lower()}_{uuid.uuid4().hex[:8]}"
                state = _TrackedIndicatorState(
                    indicator_id=ind_id,
                    indicator_key=ikey,
                    indicator_type=cand.indicator_type,
                    track_ids=cand.track_ids,
                    entity_ids=cand.entity_ids,
                    severity=cand.severity,
                    explanation=cand.explanation,
                    evidence=cand.evidence,
                    first_seen_ts=cur_ts,
                    target_zone_id=cand.target_zone_id,
                    projected_trajectory=cand.projected_trajectory,
                    trend=cand.trend,
                    supporting_events=cand.supporting_events,
                    supporting_risk_factors=cand.supporting_risk_factors,
                )
                if self.confirmation_frames <= 1:
                    state.lifecycle_state = IndicatorLifecycleState.CONFIRMED
                else:
                    state.lifecycle_state = IndicatorLifecycleState.CANDIDATE

                self._tracked_indicators[ikey] = state
            else:
                state = self._tracked_indicators[ikey]
                state.last_seen_ts = cur_ts
                state.consecutive_hits += 1
                state.consecutive_misses = 0
                state.severity = cand.severity
                state.explanation = cand.explanation
                state.evidence = cand.evidence
                state.projected_trajectory = cand.projected_trajectory

                if state.lifecycle_state == IndicatorLifecycleState.CANDIDATE:
                    if state.consecutive_hits >= self.confirmation_frames:
                        state.lifecycle_state = IndicatorLifecycleState.CONFIRMED
                elif state.lifecycle_state == IndicatorLifecycleState.CONFIRMED:
                    state.lifecycle_state = IndicatorLifecycleState.ACTIVE
                elif state.lifecycle_state == IndicatorLifecycleState.ENDED:
                    state.lifecycle_state = (
                        IndicatorLifecycleState.CONFIRMED
                        if self.confirmation_frames <= 1
                        else IndicatorLifecycleState.CANDIDATE
                    )
                    state.consecutive_hits = 1

        # B. Handle missing candidates & departed tracks
        for ikey, state in list(self._tracked_indicators.items()):
            if ikey not in candidate_map:
                has_active_inputs = frame_tracks is not None or scene is not None
                departed = has_active_inputs and any(
                    tid not in current_track_ids for tid in state.track_ids
                )

                if departed:
                    if state.lifecycle_state in (IndicatorLifecycleState.CONFIRMED, IndicatorLifecycleState.ACTIVE):
                        state.lifecycle_state = IndicatorLifecycleState.ENDED
                        state.end_ts = cur_ts
                    elif state.lifecycle_state == IndicatorLifecycleState.CANDIDATE:
                        keys_to_purge.append(ikey)
                else:
                    state.consecutive_misses += 1
                    if state.lifecycle_state == IndicatorLifecycleState.CANDIDATE:
                        keys_to_purge.append(ikey)
                    elif state.lifecycle_state in (IndicatorLifecycleState.CONFIRMED, IndicatorLifecycleState.ACTIVE):
                        if state.consecutive_misses >= self.end_confirmation_frames:
                            state.lifecycle_state = IndicatorLifecycleState.ENDED
                            state.end_ts = cur_ts

        # C. Materialize EarlyWarningIndicator schemas
        for ikey, state in list(self._tracked_indicators.items()):
            if ikey in keys_to_purge:
                del self._tracked_indicators[ikey]
                continue

            # Candidate indicators awaiting confirmation are not published
            if state.lifecycle_state == IndicatorLifecycleState.CANDIDATE:
                continue

            ind = EarlyWarningIndicator(
                indicator_id=state.indicator_id,
                indicator_type=state.indicator_type,
                lifecycle_state=state.lifecycle_state,
                timestamp=cur_ts,
                first_observed_timestamp=state.first_seen_ts,
                end_timestamp=state.end_ts,
                track_ids=state.track_ids,
                entity_ids=state.entity_ids,
                target_zone_id=state.target_zone_id,
                severity=state.severity,
                horizon_frames=state.projected_trajectory.horizon_frames if state.projected_trajectory else None,
                projected_trajectory=state.projected_trajectory,
                trend=state.trend,
                evidence=state.evidence,
                supporting_events=state.supporting_events,
                supporting_risk_factors=state.supporting_risk_factors,
                explanation=state.explanation,
                metadata={"indicator_key": state.indicator_key},
            )

            if state.lifecycle_state in (IndicatorLifecycleState.CONFIRMED, IndicatorLifecycleState.ACTIVE):
                active_indicators.append(ind)

            if state.lifecycle_state in (IndicatorLifecycleState.CONFIRMED, IndicatorLifecycleState.ENDED):
                recent_indicators.append(ind)

            # Purge ended indicators from memory in subsequent frame
            if state.lifecycle_state == IndicatorLifecycleState.ENDED:
                keys_to_purge.append(ikey)

        for k in keys_to_purge:
            self._tracked_indicators.pop(k, None)

        # 5. Determine maximum indicator severity
        sev_rank = [IndicatorSeverity.LOW, IndicatorSeverity.MEDIUM, IndicatorSeverity.HIGH, IndicatorSeverity.CRITICAL]
        max_sev = IndicatorSeverity.LOW
        for ind in active_indicators:
            if sev_rank.index(ind.severity) > sev_rank.index(max_sev):
                max_sev = ind.severity

        assessment = FramePredictionAssessment(
            frame_id=cur_fid,
            timestamp=cur_ts,
            camera_id=self.camera_id,
            projected_trajectories=projected_trajectories,
            active_indicators=active_indicators,
            recent_indicators=recent_indicators,
            total_active_indicators=len(active_indicators),
            max_indicator_severity=max_sev,
        )

        # Update singleton store
        from backend.services.prediction_store import get_prediction_store
        get_prediction_store().set_current_assessment(assessment)

        self._prev_active_track_ids = current_track_ids
        return assessment

    def reset(self) -> None:
        """Resets all internal prediction, trend, and anomaly histories."""
        self._tracked_indicators.clear()
        self._prev_active_track_ids.clear()
        self._risk_trend_analyzer.reset()
        self._anomaly_detector.reset()
