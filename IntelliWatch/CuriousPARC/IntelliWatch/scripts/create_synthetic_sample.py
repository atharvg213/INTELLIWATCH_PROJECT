"""
Creates a lightweight, self-contained synthetic video for testing
and pipeline verification without downloading external datasets.
"""
import sys
from pathlib import Path
import cv2
import numpy as np

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def generate_synthetic_video(
    output_path: Path,
    width: int = 640,
    height: int = 360,
    fps: int = 30,
    num_frames: int = 60,
) -> Path:
    """
    Generates a 2-second synthetic video with moving geometric objects,
    an industrial floor grid, and timestamp text overlays.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    if not out.isOpened():
        raise RuntimeError(f"OpenCV could not open VideoWriter for {output_path}")

    print(f"Generating synthetic video ({width}x{height} @ {fps}fps, {num_frames} frames)...")

    for i in range(num_frames):
        # Industrial dark gray floor background
        frame = np.full((height, width, 3), 35, dtype=np.uint8)

        # Floor grid lines
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (50, 50, 50), 1)
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (50, 50, 50), 1)

        # Simulated moving worker (cyan rectangle)
        worker_x = int(80 + (i * 6) % (width - 160))
        worker_y = int(120 + 20 * np.sin(i * 0.2))
        cv2.rectangle(frame, (worker_x, worker_y), (worker_x + 40, worker_y + 100), (220, 180, 0), -1)
        cv2.putText(frame, "Worker", (worker_x - 5, worker_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 180, 0), 1)

        # Simulated moving forklift (orange box)
        forklift_x = int((width - 120) - (i * 4) % (width - 160))
        forklift_y = int(height - 120)
        cv2.rectangle(frame, (forklift_x, forklift_y), (forklift_x + 80, forklift_y + 50), (0, 140, 255), -1)
        cv2.putText(frame, "Forklift", (forklift_x, forklift_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 140, 255), 1)

        # Restricted yellow hazard zone border
        cv2.rectangle(frame, (width // 2 - 80, height // 2 - 40), (width // 2 + 80, height // 2 + 60), (0, 255, 255), 2)
        cv2.putText(frame, "KEEP CLEAR", (width // 2 - 60, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

        # Frame counter & timestamp overlay
        timestamp_sec = i / fps
        cv2.putText(
            frame,
            f"Frame: {i:03d} | Time: {timestamp_sec:.2f}s | IntelliWatch CCTV Sim",
            (15, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )

        out.write(frame)

    out.release()
    print(f"Synthetic video successfully saved to: {output_path}")
    return output_path


if __name__ == "__main__":
    target = PROJECT_ROOT / "data" / "samples" / "synthetic_test.mp4"
    generate_synthetic_video(target)
