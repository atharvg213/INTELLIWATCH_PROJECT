import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from backend.services.job_manager import get_job_manager
from backend.schemas.analysis import JobStatus
import time

jm = get_job_manager()
video_file = project_root / "data" / "samples" / "cctv_worker_moving.mp4"
assert video_file.exists(), f"Video file not found at {video_file}"

print(f"Submitting test video: {video_file.name}")
submit_resp = jm.start_video_job(
    file_bytes=video_file.read_bytes(),
    original_filename=video_file.name,
    camera_id="cam_test_fix",
)
job_id = submit_resp.job_id
print(f"Submitted job: {job_id}")

# Wait for completion
for _ in range(60):
    status = jm.get_job_status(job_id)
    if status.status in (JobStatus.COMPLETED, JobStatus.FAILED):
        break
    time.sleep(1)

print(f"Job status: {status.status}, progress: {status.progress_pct}%, incidents: {status.incidents_count}")
result = jm.get_video_result(job_id)
assert result is not None
print(f"\nFinal Result:")
print(f"  total_incidents: {result.total_incidents}")
print(f"  highest_risk_tier: {result.highest_risk_tier}")
print(f"  has assessment: {result.assessment is not None}")
print(f"  has inference_breakdown: {result.inference_breakdown is not None}")
if result.inference_breakdown:
    ib = result.inference_breakdown
    print(f"  total_people: {ib.total_people}")
    print(f"  compliant_people: {ib.compliant_people}")
    print(f"  non_compliant_people: {ib.non_compliant_people}")
    for p in ib.people:
        print(f"    {p.label} (ID {p.id}): status={p.compliance_status}, missing={p.missing_ppe}, detected={p.detected_ppe}")
print(f"--- Verification Test Done ---")
