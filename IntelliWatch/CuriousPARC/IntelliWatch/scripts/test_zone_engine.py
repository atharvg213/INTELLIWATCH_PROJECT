"""
IntelliWatch - Restricted Zone Spatial Reasoning Verification Script (Step 6)

Demonstrates the Spatial Intelligence Layer:
  1. Polygon-based Restricted Zone Definition
  2. Tracked Worker Foot-Point Contact Localization
  3. Geometric Point-in-Polygon Reasoning
  4. Multi-frame Temporal Entry Confirmation
  5. Real-time Dwell Accumulation and Threshold Exceeded Alerts
  6. Structured, Explainable Zone Events (ENTRY, EXIT, DWELL_EXCEEDED)
  7. Visual Verification Overlay
"""
import argparse
import sys
import time
from pathlib import Path
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.schemas.detection import BoundingBox
from backend.schemas.events import EventType, IndustrialEvent
from backend.schemas.tracking import FrameTracks, TrackPoint, TrackedObject
from backend.schemas.zones import (
    RestrictedZone,
    ZoneMembership,
    ZoneMembershipStatus,
    ZoneType,
)
from configs.logging_config import setup_logging
from configs.settings import get_settings
from intelligence.zones.zone_engine import ZoneEngine, compute_contact_point
from vision.tracking.visualizer import TrackingVisualizer


def _create_worker(
    track_id: int,
    bbox: BoundingBox,
    frame_id: int = 1,
    timestamp: float = 0.0,
) -> TrackedObject:
    """Helper to instantiate a synthetic TrackedObject for worker."""
    return TrackedObject(
        track_id=track_id,
        class_id=0,
        class_name="person",
        confidence=0.92,
        bbox=bbox,
        trajectory=[
            TrackPoint(
                frame_id=frame_id,
                x=bbox.center[0],
                y=bbox.center[1],
                timestamp=timestamp,
            )
        ],
    )


def run_synthetic_zone_scenario(
    confirmation_frames: int = 3,
    output_image: Path = Path("data/output/test_zone_engine_output.jpg"),
):
    """
    Runs a deterministic 6-frame scenario:
      Frame 1 (t=10.0s): Worker #1 outside High Voltage Area
      Frame 2 (t=11.0s): Worker #1 steps inside (1 of 3 inside)
      Frame 3 (t=12.0s): Worker #1 remains inside (2 of 3 inside)
      Frame 4 (t=13.0s): Worker #1 remains inside (3 of 3 inside) -> ENTRY CONFIRMED!
      Frame 5 (t=15.0s): Worker #1 remains inside (dwell = 5.0s)
      Frame 6 (t=17.5s): Worker #1 exits -> EXIT CONFIRMED! (dwell = 7.5s)
    """
    print("=" * 65)
    print("IntelliWatch Zone Test (Step 6)")
    print("Mode: Deterministic Synthetic Geometry Scenario")
    print(f"Confirmation Frames: {confirmation_frames}")
    print("=" * 65)

    # 1. Define restricted safety zones
    zone_high_voltage = RestrictedZone(
        zone_id="high_voltage",
        name="High Voltage Area",
        zone_type=ZoneType.RESTRICTED,
        polygon=[
            [150.0, 150.0],
            [500.0, 150.0],
            [550.0, 420.0],
            [450.0, 520.0],
            [150.0, 500.0],
        ],
        enabled=True,
        max_dwell_seconds=6.0,
    )

    zone_machine = RestrictedZone(
        zone_id="machine_hazard",
        name="Robotic Arm Cell",
        zone_type=ZoneType.HAZARD,
        polygon=[
            [600.0, 150.0],
            [850.0, 150.0],
            [850.0, 450.0],
            [600.0, 450.0],
        ],
        enabled=True,
        max_dwell_seconds=4.0,
    )

    engine = ZoneEngine(
        zones=[zone_high_voltage, zone_machine],
        confirmation_frames=confirmation_frames,
    )

    # Synthetic bounding boxes:
    # Outside High Voltage: x1=40, y1=100, x2=100, y2=250 -> Foot = (70, 250) (outside)
    # Inside High Voltage:  x1=250, y1=200, x2=350, y2=400 -> Foot = (300, 400) (inside)
    bbox_outside = BoundingBox(x1=40.0, y1=100.0, x2=100.0, y2=250.0)
    bbox_inside = BoundingBox(x1=250.0, y1=200.0, x2=350.0, y2=400.0)

    frames_data = [
        (1, 10.0, bbox_outside, "Worker #1 outside zone"),
        (2, 11.0, bbox_inside, "Worker #1 enters polygon (Frame 1 inside)"),
        (3, 12.0, bbox_inside, "Worker #1 inside (Frame 2 inside)"),
        (4, 13.0, bbox_inside, "Worker #1 inside (Frame 3 inside) -> Threshold reached"),
        (5, 15.0, bbox_inside, "Worker #1 dwelling (dwell = 5.0s)"),
        (6, 17.5, bbox_outside, "Worker #1 leaves zone -> Exit detected"),
    ]

    last_memberships = []
    last_frame_tracks = None
    all_events: list[IndustrialEvent] = []

    for frame_id, t_sec, bbox, desc in frames_data:
        worker = _create_worker(track_id=1, bbox=bbox, frame_id=frame_id, timestamp=t_sec)
        ftracks = FrameTracks(frame_id=frame_id, timestamp=t_sec, active_tracks=[worker])
        last_frame_tracks = ftracks

        memberships, events = engine.process_frame(ftracks, timestamp=t_sec, frame_id=frame_id)
        last_memberships = memberships
        all_events.extend(events)

        print(f"\n--- Frame {frame_id:02d} (t={t_sec:.1f}s) : {desc} ---")
        for m in memberships:
            status_str = "INSIDE" if m.is_inside else "OUTSIDE"
            foot_x, foot_y = m.contact_point
            print(f"  Worker #{m.track_id} -> {m.zone_name} [{status_str}] (Foot: ({foot_x:.0f}, {foot_y:.0f})) Dwell: {m.dwell_seconds:.1f}s")

        for ev in events:
            print(f"  >>> EVENT GENERATED: [{ev.event_type.value.upper()}] Severity: {ev.severity.value}")
            print(f"      Explanation: {ev.explanation}")

    # Summarize Results in the exact format required
    print("\n" + "=" * 65)
    print("IntelliWatch Zone Test Summary")
    print("=" * 65)
    print("Worker #1")
    print("Zone: High Voltage Area\n")

    entry_evts = [e for e in all_events if e.event_type == EventType.ZONE_ENTRY]
    dwell_evts = [e for e in all_events if e.event_type == EventType.ZONE_DWELL_EXCEEDED]
    exit_evts = [e for e in all_events if e.event_type == EventType.ZONE_EXIT]

    if entry_evts:
        print("ENTRY CONFIRMED")
        print(f"Timestamp: {entry_evts[0].timestamp:.2f}s")
        print(f"Initial entry detected: {entry_evts[0].metadata.get('entry_timestamp', 0.0):.2f}s")
    else:
        print("ENTRY: Not confirmed")

    if dwell_evts:
        print(f"\nDWELL LIMIT EXCEEDED:")
        print(f"Timestamp: {dwell_evts[0].timestamp:.2f}s")
        print(f"Dwell time: {dwell_evts[0].metadata.get('dwell_seconds', 0.0):.1f} seconds")

    if exit_evts:
        print(f"\nEXIT CONFIRMED")
        print(f"Timestamp: {exit_evts[0].timestamp:.2f}s")
        print(f"Dwell duration: {exit_evts[0].metadata.get('dwell_seconds', 0.0):.1f} seconds")
    else:
        print("\nEXIT: None")

    # Render Visual Verification Canvas
    canvas_w, canvas_h = 960, 600
    canvas = np.full((canvas_h, canvas_w, 3), 35, dtype=np.uint8)

    # Draw grid
    for gx in range(0, canvas_w, 80):
        cv2.line(canvas, (gx, 0), (gx, canvas_h), (48, 48, 48), 1)
    for gy in range(0, canvas_h, 80):
        cv2.line(canvas, (0, gy), (canvas_w, gy), (48, 48, 48), 1)

    vis = TrackingVisualizer()
    active_worker = _create_worker(track_id=1, bbox=bbox_inside, frame_id=4, timestamp=13.0)
    tracks_for_vis = FrameTracks(frame_id=4, timestamp=13.0, active_tracks=[active_worker])

    vis_mem = [
        ZoneMembership(
            track_id=1,
            zone_id="high_voltage",
            zone_name="High Voltage Area",
            is_inside=True,
            status=ZoneMembershipStatus.INSIDE,
            contact_point=(300.0, 400.0),
            dwell_seconds=5.0,
            entry_timestamp=11.0,
        )
    ]

    annotated = vis.draw_tracks(
        image=canvas,
        tracks=tracks_for_vis,
        zones=[zone_high_voltage, zone_machine],
        zone_memberships=vis_mem,
        show_zones=True,
    )

    output_image.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_image), annotated)
    print(f"\nSaved synthetic zone visualization snapshot to: {output_image}")
    print("=" * 65)


def main():
    parser = argparse.ArgumentParser(description="IntelliWatch Restricted Zone Test")
    parser.add_argument(
        "--confirmation-frames",
        type=int,
        default=3,
        help="Number of consecutive frames required to confirm zone entry (default: 3)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/output/test_zone_engine_output.jpg",
        help="Path to save annotated visual verification snapshot",
    )
    args = parser.parse_args()

    setup_logging()
    run_synthetic_zone_scenario(
        confirmation_frames=args.confirmation_frames,
        output_image=Path(args.output),
    )


if __name__ == "__main__":
    main()
