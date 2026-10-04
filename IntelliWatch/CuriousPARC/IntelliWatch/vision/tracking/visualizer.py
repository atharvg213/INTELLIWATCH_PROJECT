import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np

from backend.schemas.tracking import FrameTracks, TrackPoint, TrackedObject

logger = logging.getLogger("intelliwatch.tracking_visualizer")


class TrackingVisualizer:
    """
    Visualization utility to overlay multi-object tracking data onto video frames.
    Renders:
      - Bounding boxes with consistent, unique track-specific colors
      - Compact identity badges (Track ID, class name, confidence, motion)
      - Centroid markers
      - Historical motion trajectory trails with fading lines/points
      - Real-time frame tracking telemetry banner
    """

    # Hand-curated vibrant palette for prominent initial track IDs (BGR format)
    PALETTE: List[Tuple[int, int, int]] = [
        (0, 230, 118),    # Neon Green
        (0, 176, 255),    # Electric Blue
        (255, 171, 0),    # Amber / Gold
        (213, 0, 249),    # Vivid Violet
        (255, 61, 0),     # Bright Coral Red
        (0, 229, 255),    # Cyan
        (255, 234, 0),    # Yellow
        (245, 0, 87),     # Hot Pink
        (118, 255, 3),    # Lime
        (101, 31, 255),   # Deep Purple
    ]

    def __init__(
        self,
        box_thickness: int = 2,
        font_scale: float = 0.5,
        font_thickness: int = 1,
        trajectory_thickness: int = 2,
    ):
        self.box_thickness = box_thickness
        self.font_scale = font_scale
        self.font_thickness = font_thickness
        self.trajectory_thickness = trajectory_thickness
        self._color_cache: Dict[int, Tuple[int, int, int]] = {}

    def get_track_color(self, track_id: int) -> Tuple[int, int, int]:
        """
        Returns a deterministic, high-visibility BGR color for a given track ID.
        Uses predefined vibrant colors for the first IDs, then computes golden-ratio HSV hues.
        """
        if track_id in self._color_cache:
            return self._color_cache[track_id]

        if track_id <= len(self.PALETTE):
            color = self.PALETTE[(track_id - 1) % len(self.PALETTE)]
        else:
            # Golden ratio hue distribution ensures adjacent IDs have maximally distinct colors
            golden_ratio_conjugate = 0.618033988749895
            h = ((track_id * golden_ratio_conjugate) % 1.0) * 180.0
            hsv = np.uint8([[[int(h), 220, 240]]])
            bgr = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
            color = (int(bgr[0]), int(bgr[1]), int(bgr[2]))

        self._color_cache[track_id] = color
        return color

    def draw_tracks(
        self,
        image: np.ndarray,
        tracks: Union[FrameTracks, List[TrackedObject]],
        show_trajectories: bool = True,
        show_labels: bool = True,
        show_conf: bool = True,
        show_speed: bool = True,
        show_banner: bool = True,
        worker_inventories: Optional[Union[Dict[int, Any], List[Any], Any]] = None,
        show_ppe: bool = True,
        zones: Optional[Union[List[Any], Dict[str, Any]]] = None,
        zone_memberships: Optional[Union[List[Any], Dict[int, Any]]] = None,
        show_zones: bool = True,
    ) -> np.ndarray:
        """
        Renders tracking bounding boxes, IDs, and trajectories on a copy of the input frame.
        Optionally renders worker PPE compliance badges, associated gear, and restricted safety zones.

        Args:
            image: Frame image in BGR format (numpy ndarray).
            tracks: FrameTracks schema or list of TrackedObject instances.
            show_trajectories: Whether to draw historical trajectory lines.
            show_labels: Whether to render track ID and class labels.
            show_conf: Whether to include detection confidence score.
            show_speed: Whether to include image-space speed (px/s) if available.
            show_banner: Whether to overlay top informational metadata banner.
            worker_inventories: Optional FramePPEAssociation, dict, or list of WorkerPPEInventory.
            show_ppe: Whether to overlay worker PPE compliance status and items.
            zones: Optional list or dict of RestrictedZone schemas to render.
            zone_memberships: Optional list of ZoneMembership evaluations for the active tracks.
            show_zones: Whether to render restricted zone boundaries and intrusion badges.

        Returns:
            Annotated frame image (BGR ndarray).
        """
        if image is None or image.size == 0:
            raise ValueError("Input image cannot be empty.")

        annotated = image.copy()
        h, w = annotated.shape[:2]

        if isinstance(tracks, FrameTracks):
            track_list = tracks.active_tracks
            frame_id = tracks.frame_id
            timestamp = tracks.timestamp
            total_tracks = tracks.total_track_count
        else:
            track_list = tracks
            frame_id = None
            timestamp = None
            total_tracks = len(track_list)

        # 1. Draw trajectories behind all objects first so bounding boxes stay on top
        if show_trajectories:
            for track in track_list:
                color = self.get_track_color(track.track_id)
                traj = track.trajectory
                n = len(traj)

                if n >= 2:
                    for i in range(1, n):
                        pt_prev = (int(round(traj[i - 1].cx)), int(round(traj[i - 1].cy)))
                        pt_curr = (int(round(traj[i].cx)), int(round(traj[i].cy)))

                        # Fade thickness towards older points
                        ratio = i / n
                        thickness = max(1, int(round(self.trajectory_thickness * ratio)))

                        cv2.line(annotated, pt_prev, pt_curr, color, thickness, cv2.LINE_AA)
                        cv2.circle(annotated, pt_curr, max(2, int(round(3 * ratio))), color, -1)

        # Prepare PPE inventory lookup if provided
        inv_map: Dict[int, Any] = {}
        unassoc_list: List[Any] = []
        if worker_inventories is not None:
            if hasattr(worker_inventories, "worker_inventories"):
                for inv in worker_inventories.worker_inventories:
                    inv_map[inv.track_id] = inv
                if hasattr(worker_inventories, "unassociated_ppe"):
                    unassoc_list = worker_inventories.unassociated_ppe
            elif isinstance(worker_inventories, dict):
                inv_map = worker_inventories
            elif isinstance(worker_inventories, (list, tuple)):
                for inv in worker_inventories:
                    if hasattr(inv, "track_id"):
                        inv_map[inv.track_id] = inv

        # Prepare Zone lookup and active occupancy mapping
        zone_list: List[Any] = []
        if zones is not None:
            if isinstance(zones, dict):
                zone_list = list(zones.values())
            elif isinstance(zones, (list, tuple)):
                zone_list = list(zones)

        track_to_inside_zones: Dict[int, List[Any]] = {}
        zone_to_inside_tracks: Dict[str, List[int]] = {}
        if zone_memberships is not None:
            mem_iter = (
                zone_memberships if isinstance(zone_memberships, (list, tuple))
                else [m for sublist in zone_memberships.values() for m in (sublist if isinstance(sublist, list) else [sublist])]
            )
            for m in mem_iter:
                if getattr(m, "is_inside", False):
                    tid = getattr(m, "track_id", None)
                    zid = getattr(m, "zone_id", None)
                    if tid is not None:
                        track_to_inside_zones.setdefault(tid, []).append(m)
                    if zid is not None and tid is not None:
                        zone_to_inside_tracks.setdefault(zid, []).append(tid)

        # Draw restricted zones behind actor bounding boxes
        if show_zones and zone_list:
            zone_overlay = annotated.copy()
            for zone in zone_list:
                if not getattr(zone, "enabled", True):
                    continue
                poly = getattr(zone, "polygon", [])
                if len(poly) < 3:
                    continue

                pts = np.array(poly, dtype=np.int32).reshape((-1, 1, 2))
                zid = getattr(zone, "zone_id", "")
                zname = getattr(zone, "name", zid)
                inside_workers = zone_to_inside_tracks.get(zid, [])

                # Visual distinction: Red if active intrusion, Amber/Orange if clear
                if len(inside_workers) > 0:
                    border_color = (0, 0, 235)    # High-vis Red
                    fill_color = (0, 0, 200)
                    fill_alpha = 0.22
                    status_badge = f"{zname} [INTRUSION: {len(inside_workers)} WORKER(S)]"
                else:
                    border_color = (0, 165, 255)  # Amber
                    fill_color = (0, 140, 220)
                    fill_alpha = 0.10
                    status_badge = f"{zname} [RESTRICTED]"

                # Semi-transparent polygon fill
                cv2.fillPoly(zone_overlay, [pts], fill_color)

                # Crisp polygon border
                cv2.polylines(annotated, [pts], isClosed=True, color=border_color, thickness=2, lineType=cv2.LINE_AA)

                # Zone identifier badge at min-y vertex
                min_idx = int(np.argmin(pts[:, 0, 1]))
                header_x = int(pts[min_idx, 0, 0])
                header_y = int(pts[min_idx, 0, 1])

                (bw, bh), _ = cv2.getTextSize(status_badge, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.85, 1)
                bx1 = max(0, min(w - bw - 10, header_x))
                by1 = max(0, min(h - 1, header_y - bh - 6))
                bx2 = bx1 + bw + 8
                by2 = by1 + bh + 6

                cv2.rectangle(annotated, (bx1, by1), (bx2, by2), border_color, -1)
                cv2.putText(
                    annotated,
                    status_badge,
                    (bx1 + 4, by2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale * 0.85,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

            # Blend semi-transparent zone fill
            cv2.addWeighted(zone_overlay, 0.25, annotated, 0.75, 0, annotated)

        # 2. Draw bounding boxes, centroids, and labels for each active track
        for track in track_list:

            color = self.get_track_color(track.track_id)
            bbox = track.bbox

            x1 = max(0, min(w - 1, int(round(bbox.x1))))
            y1 = max(0, min(h - 1, int(round(bbox.y1))))
            x2 = max(0, min(w - 1, int(round(bbox.x2))))
            y2 = max(0, min(h - 1, int(round(bbox.y2))))

            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, self.box_thickness)

            # Draw current centroid dot
            cx = int(round(track.centroid_x))
            cy = int(round(track.centroid_y))
            if 0 <= cx < w and 0 <= cy < h:
                cv2.circle(annotated, (cx, cy), 4, color, -1)
                cv2.circle(annotated, (cx, cy), 6, (255, 255, 255), 1)

            # Draw badge
            if show_labels:
                label_parts = [f"ID {track.track_id}", track.class_name]
                if show_conf:
                    label_parts.append(f"{track.confidence:.2f}")
                if show_speed and track.speed_pixels_per_second and track.speed_pixels_per_second > 0:
                    label_parts.append(f"{track.speed_pixels_per_second:.0f}px/s")

                label = " | ".join(label_parts)

                (tw, th), baseline = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale, self.font_thickness
                )
                badge_y1 = max(0, y1 - th - 8)
                badge_y2 = y1
                badge_x2 = min(w, x1 + tw + 8)

                # Solid background badge
                cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), color, -1)

                # Text contrast calculation
                brightness = (color[0] * 0.114 + color[1] * 0.587 + color[2] * 0.299)
                text_color = (0, 0, 0) if brightness > 150 else (255, 255, 255)

                cv2.putText(
                    annotated,
                    label,
                    (x1 + 4, badge_y2 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    self.font_scale,
                    text_color,
                    self.font_thickness,
                    cv2.LINE_AA,
                )

            # Draw PPE Compliance badge if available
            if show_ppe and track.track_id in inv_map:
                inv = inv_map[track.track_id]
                status_val = str(getattr(inv, "compliance_status", "")).upper()
                if "NON_COMPLIANT" in status_val or "NON-COMPLIANT" in status_val:
                    status_col = (0, 0, 220)  # Bright Red
                    missing_items = getattr(inv, "missing_ppe", [])
                    missing_str = f"Missing: {', '.join(missing_items)}" if missing_items else "Violation"
                    ppe_text = f"PPE: NON-COMPLIANT ({missing_str})"
                elif "COMPLIANT" in status_val:
                    status_col = (0, 180, 50)  # Green
                    ppe_text = "PPE: OK"
                else:
                    status_col = (0, 160, 240)  # Amber
                    unk_items = getattr(inv, "unknown_ppe", [])
                    unk_str = f"Unknown: {', '.join(unk_items)}" if unk_items else "Insufficient Evidence"
                    ppe_text = f"PPE: UNKNOWN ({unk_str})"

                (pw, ph_text), _ = cv2.getTextSize(
                    ppe_text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.85, self.font_thickness
                )
                p_badge_y1 = min(h - 1, y2 + 2)
                p_badge_y2 = min(h - 1, y2 + ph_text + 7)
                p_badge_x2 = min(w - 1, x1 + pw + 8)

                if p_badge_y2 > p_badge_y1 and p_badge_x2 > x1:
                    cv2.rectangle(annotated, (x1, p_badge_y1), (p_badge_x2, p_badge_y2), status_col, -1)
                    cv2.putText(
                        annotated,
                        ppe_text,
                        (x1 + 4, p_badge_y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale * 0.85,
                        (255, 255, 255),
                        self.font_thickness,
                        cv2.LINE_AA,
                    )

                # Draw associated PPE item boxes
                for item in getattr(inv, "items", []):
                    ibox = item.bbox
                    ix1 = max(0, min(w - 1, int(round(ibox.x1))))
                    iy1 = max(0, min(h - 1, int(round(ibox.y1))))
                    ix2 = max(0, min(w - 1, int(round(ibox.x2))))
                    iy2 = max(0, min(h - 1, int(round(ibox.y2))))
                    icol = (0, 0, 240) if getattr(item, "is_negative", False) else (255, 215, 0)
                    cv2.rectangle(annotated, (ix1, iy1), (ix2, iy2), icol, 1)

            # Draw Zone Intrusion badge and foot contact point if inside restricted zone(s)
            if track.track_id in track_to_inside_zones:
                inside_zones = track_to_inside_zones[track.track_id]
                z_strs = [f"{m.zone_name} ({m.dwell_seconds:.1f}s)" for m in inside_zones]
                zone_label = f"ZONE: {', '.join(z_strs)}"

                (zw, zh_text), _ = cv2.getTextSize(
                    zone_label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.85, self.font_thickness
                )

                # Offset below PPE badge if present, else below bounding box
                y_offset = p_badge_y2 + 2 if (show_ppe and track.track_id in inv_map) else y2 + 2
                z_badge_y1 = min(h - 1, y_offset)
                z_badge_y2 = min(h - 1, y_offset + zh_text + 7)
                z_badge_x2 = min(w - 1, x1 + zw + 8)

                if z_badge_y2 > z_badge_y1 and z_badge_x2 > x1:
                    cv2.rectangle(annotated, (x1, z_badge_y1), (z_badge_x2, z_badge_y2), (0, 0, 230), -1)
                    cv2.putText(
                        annotated,
                        zone_label,
                        (x1 + 4, z_badge_y2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        self.font_scale * 0.85,
                        (255, 255, 255),
                        self.font_thickness,
                        cv2.LINE_AA,
                    )

                # Highlight ground contact point
                fx = max(0, min(w - 1, int(round((bbox.x1 + bbox.x2) / 2.0))))
                fy = max(0, min(h - 1, int(round(bbox.y2))))
                cv2.circle(annotated, (fx, fy), 5, (0, 0, 255), -1)
                cv2.circle(annotated, (fx, fy), 7, (255, 255, 255), 1)

        # Draw unassociated PPE detections if any
        if show_ppe and unassoc_list:
            for unp in unassoc_list:
                ubox = unp.bbox
                ux1 = max(0, min(w - 1, int(round(ubox.x1))))
                uy1 = max(0, min(h - 1, int(round(ubox.y1))))
                ux2 = max(0, min(w - 1, int(round(ubox.x2))))
                uy2 = max(0, min(h - 1, int(round(ubox.y2))))
                cv2.rectangle(annotated, (ux1, uy1), (ux2, uy2), (160, 160, 160), 1)
                cv2.putText(
                    annotated,
                    f"Unassigned: {unp.class_name}",
                    (ux1, max(12, uy1 - 3)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (160, 160, 160),
                    1,
                    cv2.LINE_AA,
                )

        # 3. Draw top telemetry banner

        if show_banner and frame_id is not None:
            zone_info = ""
            if show_zones and zone_list:
                intrusion_cnt = sum(1 for zid in zone_to_inside_tracks if zone_to_inside_tracks[zid])
                zone_info = f" | Zones: {len(zone_list)} ({intrusion_cnt} alert)"

            banner_text = (
                f"Frame: {frame_id:04d} | "
                f"Time: {timestamp:.2f}s | "
                f"Active Tracks: {len(track_list)} | "
                f"Total Unique: {total_tracks} | "
                f"MOT: ByteTrack"
                f"{zone_info}"
            )
            (tw, th), _ = cv2.getTextSize(banner_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            cv2.rectangle(annotated, (0, 0), (min(w, tw + 20), th + 14), (20, 20, 20), -1)
            cv2.putText(
                annotated,
                banner_text,
                (10, th + 7),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (230, 230, 230),
                1,
                cv2.LINE_AA,
            )

        return annotated

    def save_annotated_frame(self, image: np.ndarray, output_path: Union[str, Path]) -> Path:
        """Saves annotated frame image to disk, creating destination folders if needed."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out), image)
        logger.info(f"Saved annotated tracking frame to: {out}")
        return out

    def draw_behavior_overlays(
        self,
        image: np.ndarray,
        tracks: Union["FrameTracks", List["TrackedObject"]],
        behavior_states: Optional[List[Any]] = None,
    ) -> np.ndarray:
        """
        Renders compact behavior state badges beneath each tracked worker's bounding box.
        Call AFTER draw_tracks() so behavior badges appear on top of the existing rendering.

        Args:
            image: Annotated frame (result of draw_tracks).
            tracks: FrameTracks or list of TrackedObjects (same as passed to draw_tracks).
            behavior_states: List of BehaviorState objects from BehaviorEngine.process().
                             If None or empty, this method is a no-op.

        Returns:
            image with behavior badges overlaid (in-place copy).
        """
        if behavior_states is None or len(behavior_states) == 0:
            return image

        annotated = image  # modify in place (already a copy from draw_tracks)
        h, w = annotated.shape[:2]

        # Build track_id -> BehaviorState lookup
        bstate_map: Dict[int, Any] = {}
        for bs in behavior_states:
            tid = getattr(bs, "track_id", None)
            if tid is not None:
                bstate_map[tid] = bs

        if isinstance(tracks, FrameTracks):
            track_list = tracks.active_tracks
        else:
            track_list = tracks

        for track in track_list:
            tid = track.track_id
            if tid not in bstate_map:
                continue
            bs = bstate_map[tid]
            behavior = getattr(bs, "primary_behavior", None)
            if behavior is None:
                continue

            behavior_val = str(behavior.value) if hasattr(behavior, "value") else str(behavior)

            bbox = track.bbox
            x1 = max(0, min(w - 1, int(round(bbox.x1))))
            y2 = max(0, min(h - 1, int(round(bbox.y2))))

            # Choose badge color per behavior
            color_map = {
                "MOVING":               (0, 200, 80),    # Green
                "STATIONARY":           (0, 165, 255),   # Amber
                "RAPID_MOVEMENT":       (0, 0, 230),     # Red
                "SUDDEN_MOVEMENT":      (0, 80, 220),    # Orange-red
                "PROLONGED_STATIONARY": (0, 100, 200),   # Deep orange
                "POSSIBLE_FALL":        (20, 20, 220),   # Bright red
                "UNKNOWN":              (120, 120, 120), # Grey
            }
            badge_col = color_map.get(behavior_val, (100, 100, 100))

            # Compose badge text
            speed = getattr(bs, "image_speed_px_per_s", None)
            stat_dur = getattr(bs, "stationary_duration_s", 0.0)

            if behavior_val == "POSSIBLE_FALL":
                text = f"ID {tid}: POSSIBLE FALL"
            elif behavior_val == "PROLONGED_STATIONARY":
                text = f"ID {tid}: PROLONGED STAT {stat_dur:.1f}s"
            elif behavior_val == "STATIONARY":
                text = f"ID {tid}: STATIONARY {stat_dur:.1f}s"
            elif behavior_val in ("RAPID_MOVEMENT", "SUDDEN_MOVEMENT"):
                spd_str = f" {speed:.0f}px/s" if speed else ""
                label = "RAPID" if behavior_val == "RAPID_MOVEMENT" else "SUDDEN"
                text = f"ID {tid}: {label}{spd_str}"
            elif behavior_val == "MOVING":
                spd_str = f" {speed:.0f}px/s" if speed else ""
                text = f"ID {tid}: MOVING{spd_str}"
            else:
                text = f"ID {tid}: {behavior_val}"

            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.80, 1)

            # Compute vertical offset: stack below zone badge if present
            badge_y1 = min(h - 1, y2 + 26)
            badge_y2 = min(h - 1, badge_y1 + th + 6)
            badge_x2 = min(w - 1, x1 + tw + 8)

            if badge_y2 > badge_y1 and badge_x2 > x1:
                cv2.rectangle(annotated, (x1, badge_y1), (badge_x2, badge_y2), badge_col, -1)
                brightness = (badge_col[0] * 0.114 + badge_col[1] * 0.587 + badge_col[2] * 0.299)
                txt_col = (0, 0, 0) if brightness > 150 else (255, 255, 255)
                cv2.putText(
                    annotated, text, (x1 + 4, badge_y2 - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.80,
                    txt_col, 1, cv2.LINE_AA,
                )

        return annotated

    def draw_scene_graph(
        self,
        frame: np.ndarray,
        scene: Any,
        draw_hud: bool = True,
        draw_edges: bool = True,
    ) -> np.ndarray:
        """
        Overlays scene graph relational links and situational awareness HUD onto the frame.

        Args:
            frame: Input image array (BGR uint8).
            scene: FrameScene or SceneGraph object containing nodes and relationships.
            draw_hud: Whether to render high-level situational metrics card.
            draw_edges: Whether to draw proximity / kinematic links between entities.

        Returns:
            Annotated frame (BGR uint8).
        """
        annotated = frame.copy()
        h, w = annotated.shape[:2]

        # Extract nodes and relationships from FrameScene or SceneGraph
        nodes = getattr(scene, "nodes", [])
        if callable(nodes):
            nodes = nodes()
        relationships = getattr(scene, "relationships", [])
        if callable(relationships):
            relationships = relationships()
        summary = getattr(scene, "summary", None)

        node_map = {n.node_id: n for n in nodes if hasattr(n, "node_id")}

        # 1. Draw relational edges (NEAR / APPROACHING)
        if draw_edges and relationships:
            for rel in relationships:
                # Only draw active/created proximity & kinematic edges
                lifecycle = getattr(rel, "lifecycle", None)
                lval = lifecycle.value if hasattr(lifecycle, "value") else str(lifecycle)
                if lval not in ("ACTIVE", "CREATED"):
                    continue

                rtype = getattr(rel, "relation_type", None)
                rval = rtype.value if hasattr(rtype, "value") else str(rtype)
                if rval not in ("NEAR", "APPROACHING"):
                    continue

                src_node = node_map.get(getattr(rel, "source_node_id", None))
                tgt_node = node_map.get(getattr(rel, "target_node_id", None))
                if not src_node or not tgt_node:
                    continue

                pt1 = getattr(src_node, "contact_point", None) or getattr(src_node, "centroid", None)
                pt2 = getattr(tgt_node, "contact_point", None) or getattr(tgt_node, "centroid", None)
                if not pt1 or not pt2:
                    continue

                x1, y1 = int(round(pt1[0])), int(round(pt1[1]))
                x2, y2 = int(round(pt2[0])), int(round(pt2[1]))

                # Clamp coordinates to frame boundary
                x1, y1 = max(0, min(w - 1, x1)), max(0, min(h - 1, y1))
                x2, y2 = max(0, min(w - 1, x2)), max(0, min(h - 1, y2))

                # Edge styling
                if rval == "APPROACHING":
                    line_color = (0, 69, 255)  # Orange-red
                    thickness = 2
                    label = "APPROACHING"
                else:
                    line_color = (0, 215, 255)  # Gold
                    thickness = 1
                    dist = rel.evidence.get("distance_px") if hasattr(rel, "evidence") and isinstance(rel.evidence, dict) else None
                    label = f"NEAR ({dist:.0f}px)" if dist is not None else "NEAR"

                # Draw connecting line
                cv2.line(annotated, (x1, y1), (x2, y2), line_color, thickness, cv2.LINE_AA)

                # Midpoint badge
                mx, my = (x1 + x2) // 2, (y1 + y2) // 2
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.70, 1)
                bx1 = max(0, min(w - 1, mx - tw // 2 - 4))
                by1 = max(0, min(h - 1, my - th // 2 - 4))
                bx2 = max(0, min(w - 1, bx1 + tw + 8))
                by2 = max(0, min(h - 1, by1 + th + 6))

                if bx2 > bx1 and by2 > by1:
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), (20, 20, 20), -1)
                    cv2.rectangle(annotated, (bx1, by1), (bx2, by2), line_color, 1)
                    cv2.putText(
                        annotated, label, (bx1 + 4, by2 - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.70,
                        (255, 255, 255), 1, cv2.LINE_AA,
                    )

        # 2. Draw Situational Summary HUD
        if draw_hud and summary:
            workers = getattr(summary, "workers_count", 0)
            vehicles = getattr(summary, "vehicles_count", 0)
            machines = getattr(summary, "machines_count", 0)
            zones_occ = getattr(summary, "occupied_zones_count", 0)
            prox_rels = getattr(summary, "active_proximity_relationships", 0)
            non_ppe = getattr(summary, "workers_non_compliant_ppe", 0)

            hud_lines = [
                f"SCENE: W:{workers} V:{vehicles} M:{machines} | ZONES OCC: {zones_occ}",
                f"PROXIMITY RELS: {prox_rels}",
            ]
            if non_ppe > 0:
                hud_lines.append(f"PPE VIOLATION: {non_ppe} WORKER(S)")

            box_w = 260
            box_h = 16 + len(hud_lines) * 18
            hx1 = max(0, w - box_w - 12)
            hy1 = 12
            hx2 = min(w - 1, hx1 + box_w)
            hy2 = min(h - 1, hy1 + box_h)

            overlay = annotated.copy()
            cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (15, 15, 15), -1)
            cv2.addWeighted(overlay, 0.75, annotated, 0.25, 0, annotated)
            cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), (70, 70, 70), 1)

            for idx, line in enumerate(hud_lines):
                ly = hy1 + 16 + idx * 18
                color = (0, 0, 255) if "VIOLATION" in line else (230, 230, 230)
                cv2.putText(
                    annotated, line, (hx1 + 8, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.75,
                    color, 1, cv2.LINE_AA,
                )

        return annotated

    def draw_risk_assessment(
        self,
        frame: np.ndarray,
        assessment: Any,
        draw_hud: bool = True,
    ) -> np.ndarray:
        """
        Overlays safety risk assessments, active confirmed events, and risk tier badges onto the frame.

        Args:
            frame: Input video frame (BGR uint8).
            assessment: FrameRiskAssessment object.
            draw_hud: Whether to render the risk summary HUD card.

        Returns:
            Annotated frame (BGR uint8).
        """
        if not draw_hud or assessment is None:
            return frame

        annotated = frame.copy()
        h, w = annotated.shape[:2]

        active_events = getattr(assessment, "active_events", [])
        risk_summary = getattr(assessment, "risk_summary", None)
        max_level = getattr(risk_summary, "max_risk_level", None)
        max_level_str = max_level.value if hasattr(max_level, "value") else str(max_level or "INFO")
        max_score = getattr(risk_summary, "max_risk_score", 0.0)

        # Color mapping for risk levels (BGR)
        level_colors = {
            "CRITICAL": (0, 0, 255),       # Bright Red
            "HIGH": (0, 69, 255),          # Red-Orange
            "MEDIUM": (0, 215, 255),       # Gold / Amber
            "LOW": (0, 230, 118),          # Neon Green
            "INFO": (200, 200, 200),       # Light Gray
        }
        badge_color = level_colors.get(max_level_str, (200, 200, 200))

        hud_lines = []
        if not active_events:
            hud_lines.append("SAFETY STATUS: NORMAL | 0 ACTIVE EVENTS")
        else:
            hud_lines.append(f"SAFETY RISK: {max_level_str} (SCORE: {max_score:.1f})")
            hud_lines.append(f"ACTIVE EVENTS: {len(active_events)}")
            # Show up to top 3 active events
            for ev in active_events[:3]:
                ev_type = getattr(ev, "event_type", "")
                ev_type_str = ev_type.value if hasattr(ev_type, "value") else str(ev_type)
                ev_lvl = getattr(ev, "risk_level", "")
                ev_lvl_str = ev_lvl.value if hasattr(ev_lvl, "value") else str(ev_lvl)
                ents = getattr(ev, "involved_entities", [])
                ents_str = ", ".join(ents[:2])
                hud_lines.append(f" * [{ev_lvl_str}] {ev_type_str}: {ents_str}")

        box_w = 320
        box_h = 16 + len(hud_lines) * 18
        hx1 = 12
        hy1 = max(0, h - box_h - 12)
        hx2 = min(w - 1, hx1 + box_w)
        hy2 = min(h - 1, hy1 + box_h)

        overlay = annotated.copy()
        cv2.rectangle(overlay, (hx1, hy1), (hx2, hy2), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.78, annotated, 0.22, 0, annotated)
        cv2.rectangle(annotated, (hx1, hy1), (hx2, hy2), badge_color, 1)

        for idx, line in enumerate(hud_lines):
            ly = hy1 + 16 + idx * 18
            color = badge_color if idx == 0 and active_events else (235, 235, 235)
            cv2.putText(
                annotated, line, (hx1 + 8, ly),
                cv2.FONT_HERSHEY_SIMPLEX, self.font_scale * 0.70,
                color, 1, cv2.LINE_AA,
            )

        return annotated


