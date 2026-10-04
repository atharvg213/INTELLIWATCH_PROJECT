import os
import sys

sys.path.insert(0, os.path.abspath("."))

from backend.services.job_manager import get_job_manager

def main():
    jm = get_job_manager()
    img_path = r"data\input\uploads\job_img_00a2c78ed0_input.jpg"
    if not os.path.exists(img_path):
        print(f"Image not found at {img_path}")
        return
        
    with open(img_path, "rb") as f:
        img_bytes = f.read()

    print("Running process_image...")
    res = jm.process_image(img_bytes, original_filename="test_input.jpg", camera_id="cam_01")
    print("Image job completed:")
    print(f"  detections_count: {res.detections_count}")
    print(f"  workers_count: {res.workers_count}")
    print(f"  non_compliant_count: {res.non_compliant_count}")
    print(f"  has inference_breakdown: {res.inference_breakdown is not None}")
    if res.inference_breakdown:
        for p in res.inference_breakdown.people:
            print(f"    {p.label} (ID {p.id}): status={p.compliance_status}, missing={p.missing_ppe}, detected={p.detected_ppe}")

if __name__ == "__main__":
    main()
