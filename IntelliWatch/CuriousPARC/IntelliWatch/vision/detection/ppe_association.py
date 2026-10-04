import logging
from typing import Dict, List, Optional, Set, Tuple, Union

from backend.schemas.detection import BoundingBox, DetectionResult, FrameDetections
from backend.schemas.ppe import AssociatedPPEItem, FramePPEAssociation, WorkerPPEInventory
from backend.schemas.tracking import FrameTracks, TrackedObject
from configs.settings import get_settings

logger = logging.getLogger("intelliwatch.ppe_association")
LOW_CONFIDENCE_MIN_ASSOCIATION_SCORE = 0.80


# Anatomical definition: (y_min_relative, y_max_relative, y_ideal_relative, body_region)
ANATOMICAL_REGIONS: Dict[str, Tuple[float, float, float, str]] = {
    "hardhat": (-0.25, 0.40, 0.08, "head"),
    "goggles": (-0.05, 0.35, 0.18, "face"),
    "mask": (0.02, 0.40, 0.22, "face"),
    "safety vest": (0.10, 0.75, 0.40, "torso"),
    "gloves": (0.48, 1.12, 0.82, "hands"),
    "harness": (0.10, 0.85, 0.45, "torso"),
    "fall-detected": (-0.20, 1.20, 0.50, "body"),
}

# Synonyms and negative violation mapping
CLASS_NAME_NORMALIZATION: Dict[str, Tuple[str, bool]] = {
    # Positive gear
    "hardhat": ("Hardhat", False),
    "helmet": ("Hardhat", False),
    "hard hat": ("Hardhat", False),
    "safety vest": ("Safety Vest", False),
    "vest": ("Safety Vest", False),
    "safety-vest": ("Safety Vest", False),
    "gloves": ("Gloves", False),
    "glove": ("Gloves", False),
    "goggles": ("Goggles", False),
    "goggle": ("Goggles", False),
    "safety glasses": ("Goggles", False),
    "glasses": ("Goggles", False),
    "mask": ("Mask", False),
    "face mask": ("Mask", False),
    "harness": ("Harness", False),
    "safety harness": ("Harness", False),
    # Negative violation indicators
    "no-hardhat": ("Hardhat", True),
    "no-helmet": ("Hardhat", True),
    "no-hard hat": ("Hardhat", True),
    "no_hardhat": ("Hardhat", True),
    "no-safety vest": ("Safety Vest", True),
    "no-vest": ("Safety Vest", True),
    "no_safety vest": ("Safety Vest", True),
    "no-gloves": ("Gloves", True),
    "no-glove": ("Gloves", True),
    "no_gloves": ("Gloves", True),
    "no-goggles": ("Goggles", True),
    "no-goggle": ("Goggles", True),
    "no_goggles": ("Goggles", True),
    "no-mask": ("Mask", True),
    "no_mask": ("Mask", True),
    "no-harness": ("Harness", True),
    "no_harness": ("Harness", True),
    # Model behavioral indicator
    "fall-detected": ("Fall-Detected", False),
    "fall": ("Fall-Detected", False),
}


class PPEAssociationEngine:
    """
    Transparent geometric spatial reasoning engine that associates detected PPE items
    with persistent tracked workers (ByteTrack output).

    Spatial Evidence Model:
      1. Bounding-Box Containment: Measures what fraction of the PPE bounding box is contained
         within the candidate worker's bounding box.
      2. Relative Anatomical Sub-regions: Ensures gear items are located where expected on a human body
         (e.g., Hardhat on upper head, Vest on torso, Goggles on face, Gloves near hands/hips).
      3. Horizontal Span Verification: Rejects detections horizontally outside the worker's body bounds.
      4. Competitive Disambiguation: If a PPE item lies near multiple workers, it assigns it to the worker
         with the strongest spatial evidence.
      5. Strict False-Association Guard: Discards detections without sufficiently strong spatial fit,
         designating them as 'unassociated PPE detections'.
    """

    def __init__(
        self,
        association_threshold: Optional[float] = None,
        containment_threshold: Optional[float] = None,
        horizontal_margin: float = 0.20,
        person_classes: Optional[Set[str]] = None,
    ):
        settings = get_settings()
        self.association_threshold = (
            association_threshold
            if association_threshold is not None
            else getattr(settings, "PPE_ASSOCIATION_IOU_THRESHOLD", 0.10)
        )
        self.containment_threshold = (
            containment_threshold
            if containment_threshold is not None
            else getattr(settings, "PPE_CENTER_CONTAINMENT_THRESHOLD", 0.50)
        )
        self.low_confidence_association_score = LOW_CONFIDENCE_MIN_ASSOCIATION_SCORE
        self.horizontal_margin = horizontal_margin
        self.person_classes = person_classes or {"person", "worker"}

    def normalize_class(self, raw_class_name: str) -> Optional[Tuple[str, bool]]:
        """
        Normalizes raw detector class name to (canonical_name, is_negative).
        Returns None for general person detections or unrecognized non-PPE classes.
        """
        cleaned = raw_class_name.lower().strip()
        if cleaned in ("person", "worker"):
            return None
        return CLASS_NAME_NORMALIZATION.get(cleaned, (raw_class_name.title(), False))

    @staticmethod
    def compute_box_intersection_area(b1: BoundingBox, b2: BoundingBox) -> float:
        """Calculates area of intersection between two bounding boxes in pixels."""
        ix1 = max(b1.x1, b2.x1)
        iy1 = max(b1.y1, b2.y1)
        ix2 = min(b1.x2, b2.x2)
        iy2 = min(b1.y2, b2.y2)

        iw = max(0.0, ix2 - ix1)
        ih = max(0.0, iy2 - iy1)
        return iw * ih

    def compute_spatial_score(
        self,
        person_box: BoundingBox,
        ppe_box: BoundingBox,
        canonical_class: str,
    ) -> float:
        """
        Computes an explainable spatial association score between a candidate worker
        and a PPE bounding box in the range [0.0, 1.0].
        """
        pw = person_box.width
        ph = person_box.height
        if pw <= 0.0 or ph <= 0.0:
            return 0.0

        ppe_area = ppe_box.width * ppe_box.height
        if ppe_area <= 0.0:
            return 0.0

        cx_ppe, cy_ppe = ppe_box.center
        norm_class = canonical_class.lower().strip()

        # 1. Horizontal Span Verification
        horizontal_margin = self.horizontal_margin
        if norm_class == "gloves":
            # Hands may extend beyond the torso/person box, so allow a wider
            # span while still scoring candidates against the two wrist areas.
            horizontal_margin = max(horizontal_margin, 0.35)
        x_min_allowed = person_box.x1 - (horizontal_margin * pw)
        x_max_allowed = person_box.x2 + (horizontal_margin * pw)
        if cx_ppe < x_min_allowed or cx_ppe > x_max_allowed:
            return 0.0

        # 2. Anatomical Vertical Region Verification
        y_min_rel, y_max_rel, y_ideal_rel, _ = ANATOMICAL_REGIONS.get(
            norm_class, (-0.20, 1.15, 0.50, "general")
        )

        # Adapt bounds for upper-body / seated workers (pw / ph >= 0.70)
        if (pw / ph) >= 0.70:
            if norm_class in ("safety vest", "vest", "safety-vest"):
                y_max_rel = max(y_max_rel, 0.95)
                y_ideal_rel = 0.55
            elif norm_class in ("hardhat", "helmet"):
                y_max_rel = max(y_max_rel, 0.45)

        y_rel = (cy_ppe - person_box.y1) / ph
        if y_rel < y_min_rel or y_rel > y_max_rel:
            return 0.0

        # 3. Containment Ratio (Fraction of PPE box inside person box)
        inter_area = self.compute_box_intersection_area(person_box, ppe_box)
        containment = min(1.0, inter_area / ppe_area)

        # 4. Center Containment bonus
        center_inside = (
            person_box.x1 <= cx_ppe <= person_box.x2 and
            person_box.y1 <= cy_ppe <= person_box.y2
        )

        # 5. Anatomical Vertical Alignment Score
        vert_diff = abs(y_rel - y_ideal_rel)
        v_span = max(0.1, y_max_rel - y_min_rel)
        v_score = max(0.0, 1.0 - (vert_diff / v_span))

        # 6. Horizontal Alignment Score (closeness to worker's vertical centerline)
        if norm_class == "gloves":
            # Approximate left/right wrist anchors from the tracked body box.
            # This avoids penalizing naturally lateral hand positions.
            x_rel = (cx_ppe - person_box.x1) / pw
            wrist_distance = min(abs(x_rel - 0.12), abs(x_rel - 0.88))
            h_score = max(0.0, 1.0 - (wrist_distance / 0.55))
        else:
            cx_person, _ = person_box.center
            h_score = max(0.0, 1.0 - (abs(cx_ppe - cx_person) / (pw / 2.0 + 1e-6)))

        # Weighted combination of geometric spatial factors
        score = (0.45 * containment) + (0.35 * v_score) + (0.20 * h_score)

        if center_inside:
            score = min(1.0, score + 0.10)

        return round(float(score), 4)

    def associate(
        self,
        tracks: Union[FrameTracks, List[TrackedObject]],
        ppe_detections: Union[FrameDetections, List[DetectionResult]],
        frame_id: Optional[int] = None,
        timestamp: Optional[float] = None,
    ) -> FramePPEAssociation:
        """
        Associates PPE detections with tracked persons for a given video frame.
        Uses competitive capacity-constrained bipartite matching to prevent
        duplicate gear assignment to one person when multiple workers are nearby.
        """
        # Unpack tracks
        if isinstance(tracks, FrameTracks):
            fid = tracks.frame_id
            ts = tracks.timestamp
            tracked_list = tracks.active_tracks
        else:
            fid = 0 if frame_id is None else frame_id
            ts = 0.0 if timestamp is None else timestamp
            tracked_list = tracks

        # Filter for tracked persons
        person_tracks: List[TrackedObject] = [
            t for t in tracked_list if t.class_name.lower().strip() in self.person_classes
        ]

        # Unpack PPE detections
        if isinstance(ppe_detections, FrameDetections):
            det_list = ppe_detections.detections
            if frame_id is None:
                fid = ppe_detections.frame_id
            if timestamp is None:
                ts = ppe_detections.timestamp
        else:
            det_list = ppe_detections

        # Prepare worker inventories
        worker_inventories: Dict[int, WorkerPPEInventory] = {}
        for p in person_tracks:
            worker_inventories[p.track_id] = WorkerPPEInventory(
                track_id=p.track_id,
                timestamp=ts,
                bbox=p.bbox,
                items=[],
                ppe_status={},
                missing_ppe=[],
                present_ppe=[],
                unknown_ppe=[],
            )

        unassociated_ppe: List[DetectionResult] = []

        if not person_tracks or not det_list:
            if not person_tracks and det_list:
                for det in det_list:
                    norm = self.normalize_class(det.class_name)
                    if norm is not None:
                        unassociated_ppe.append(det)
                        bbox = det.bbox
                        logger.debug(
                            "[PPE_ASSOCIATION] frame=%s class=%s confidence=%.3f "
                            "bbox=(%.1f,%.1f,%.1f,%.1f) roi_owner=%s "
                            "associated_track=None result=unassociated (no person track)",
                            fid,
                            det.class_name,
                            det.confidence,
                            bbox.x1,
                            bbox.y1,
                            bbox.x2,
                            bbox.y2,
                            getattr(det, "metadata", {}).get("crop_parent_track"),
                        )

            return FramePPEAssociation(
                frame_id=fid,
                timestamp=ts,
                worker_inventories=list(worker_inventories.values()),
                unassociated_ppe=unassociated_ppe,
            )

        # 1. Build all candidate (worker, ppe_item) matches above threshold
        candidate_matches: List[dict] = []
        for ppe_idx, ppe_det in enumerate(det_list):
            norm_res = self.normalize_class(ppe_det.class_name)
            if norm_res is None:
                continue

            canonical_class, is_neg = norm_res
            metadata = getattr(ppe_det, "metadata", {}) or {}
            crop_parent_track = metadata.get("crop_parent_track")
            low_confidence_candidate = metadata.get("low_confidence_candidate") is True
            # Low-score candidates must have a strong anatomical/spatial match.
            # Compliance applies temporal confirmation to low-score negatives.
            required_score = (
                self.low_confidence_association_score
                if low_confidence_candidate
                else self.association_threshold
            )
            _, _, _, body_reg = ANATOMICAL_REGIONS.get(
                canonical_class.lower(), (0, 0, 0, "general")
            )

            for p in person_tracks:
                # PPE found inside a worker-specific ROI has an authoritative
                # owner. Do not let geometry transfer it to a nearby track.
                if crop_parent_track is not None and p.track_id != crop_parent_track:
                    continue
                score = self.compute_spatial_score(
                    person_box=p.bbox,
                    ppe_box=ppe_det.bbox,
                    canonical_class=canonical_class,
                )
                if score >= required_score:
                    candidate_matches.append({
                        "score": score,
                        "ppe_idx": ppe_idx,
                        "ppe_det": ppe_det,
                        "track_id": p.track_id,
                        "canonical_class": canonical_class,
                        "is_negative": is_neg,
                        "low_confidence_candidate": low_confidence_candidate,
                        "body_region": body_reg,
                    })

        # Match positive evidence first so a conflicting NO-* box cannot consume
        # the worker/category slot when actual gear was also detected.
        candidate_matches.sort(
            key=lambda m: (
                not m["is_negative"],
                m["score"],
                m["ppe_det"].confidence,
            ),
            reverse=True,
        )

        # Competitive bipartite matching with per-category worker capacity
        assigned_ppe_indices: Set[int] = set()
        assigned_tracks: Dict[int, int] = {}
        worker_gear_counts: Dict[int, Dict[str, int]] = {
            p.track_id: {} for p in person_tracks
        }

        # Category capacity: at most 2 for gloves, 1 for all other PPE gear
        def get_max_capacity(cat: str) -> int:
            return 2 if cat.lower() == "gloves" else 1

        # Pass 1: Assign each gear item to the best worker who doesn't yet have it
        for match in candidate_matches:
            p_idx = match["ppe_idx"]
            t_id = match["track_id"]
            c_class = match["canonical_class"]

            if p_idx in assigned_ppe_indices:
                continue

            current_count = worker_gear_counts[t_id].get(c_class, 0)
            if current_count < get_max_capacity(c_class):
                associated_item = AssociatedPPEItem(
                    class_id=match["ppe_det"].class_id,
                    class_name=c_class,
                    confidence=match["ppe_det"].confidence,
                    bbox=match["ppe_det"].bbox,
                    association_score=match["score"],
                    is_negative=match["is_negative"],
                    low_confidence_candidate=match["low_confidence_candidate"],
                    body_region=match["body_region"],
                )
                worker_inventories[t_id].items.append(associated_item)
                worker_gear_counts[t_id][c_class] = current_count + 1
                assigned_ppe_indices.add(p_idx)
                assigned_tracks[p_idx] = t_id
                logger.debug(
                    f"Frame {fid}: Associated {c_class} (score={match['score']:.2f}) "
                    f"with Track #{t_id}"
                )

        # Collect unassociated PPE items
        for ppe_idx, ppe_det in enumerate(det_list):
            bbox = ppe_det.bbox
            logger.debug(
                "[PPE_ASSOCIATION] frame=%s class=%s confidence=%.3f bbox=(%.1f,%.1f,%.1f,%.1f) "
                "roi_owner=%s associated_track=%s result=%s",
                fid,
                ppe_det.class_name,
                ppe_det.confidence,
                bbox.x1,
                bbox.y1,
                bbox.x2,
                bbox.y2,
                getattr(ppe_det, "metadata", {}).get("crop_parent_track"),
                assigned_tracks.get(ppe_idx),
                "associated" if ppe_idx in assigned_ppe_indices else "unassociated",
            )
            if ppe_idx not in assigned_ppe_indices:
                norm = self.normalize_class(ppe_det.class_name)
                if norm is not None:
                    unassociated_ppe.append(ppe_det)
                    logger.debug(
                        f"Frame {fid}: PPE detection {ppe_det.class_name} unassociated "
                        f"(no suitable worker candidate)"
                    )

        return FramePPEAssociation(
            frame_id=fid,
            timestamp=ts,
            worker_inventories=list(worker_inventories.values()),
            unassociated_ppe=unassociated_ppe,
        )
