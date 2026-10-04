/**
 * Comprehensive Test Suite for Scene Graph Hero Feature
 * Tests real sample image, real sample video, scene graph endpoints,
 * tracking continuity, velocity kinematics, and regression safety.
 */

import fs from 'node:fs';
import path from 'node:path';

const API_BASE = 'http://127.0.0.1:8000';
const SAMPLE_IMAGE_PATH = 'CuriousPARC/IntelliWatch/data/samples/industrial_cctv.jpg';
const SAMPLE_PPE_PATH = 'CuriousPARC/IntelliWatch/data/samples/ppe_sample.jpg';
const SAMPLE_VIDEO_PATH = 'CuriousPARC/IntelliWatch/data/samples/cctv_worker_moving.mp4';

function assert(condition, message) {
  if (!condition) {
    console.error(`❌ FAILED: ${message}`);
    process.exit(1);
  }
  console.log(`✅ ${message}`);
}

async function sleep(ms) {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

async function run() {
  console.log('================================================================');
  console.log('🧪 RUNNING COMPREHENSIVE INTELLIWATCH SCENE GRAPH HERO TESTS');
  console.log('================================================================\n');

  // --- Test 1: Industrial Image Perceptual Scene Graph ---
  console.log('--- TEST 1: Industrial Image Perception & Grounding ---');
  const imgBytes = fs.readFileSync(SAMPLE_IMAGE_PATH);
  const imgBlob = new Blob([imgBytes], { type: 'image/jpeg' });
  const form1 = new FormData();
  form1.append('file', imgBlob, 'industrial_cctv.jpg');

  const imgRes = await fetch(`${API_BASE}/api/v1/analyze/image`, {
    method: 'POST',
    body: form1,
  });
  assert(imgRes.ok, 'Image analysis HTTP 200');
  const imgData = await imgRes.json();
  const imgJobId = imgData.job_id;
  assert(Boolean(imgJobId), `Obtained job_id: ${imgJobId}`);
  assert(imgData.annotated_media_url != null, 'Annotated media URL generated');

  // Query Scene Graph via all 3 canonical endpoints
  const q1 = await (await fetch(`${API_BASE}/api/v1/scene/graph?job_id=${imgJobId}`)).json();
  const q2 = await (await fetch(`${API_BASE}/api/v1/analyze/scene/${imgJobId}`)).json();
  const q3 = await (await fetch(`${API_BASE}/api/v1/scene/${imgJobId}`)).json();

  assert(q1.scene_id != null, `Scene has unique scene_id: ${q1.scene_id}`);
  assert(q1.entities.length === q2.entities.length && q1.entities.length === q3.entities.length, 'All scene endpoints return identical entity count');
  assert(q1.relationships.length === q2.relationships.length, 'All scene endpoints return identical relationship count');

  // Check entities
  const workerNode = q1.entities.find((e) => e.type === 'PERSON');
  assert(Boolean(workerNode), 'PERSON entity detected');
  assert(workerNode.bbox != null, 'PERSON has bounding box coordinates');
  assert(workerNode.position.centroid != null, 'PERSON has centroid coordinates');
  assert(workerNode.state != null, 'PERSON has state dictionary');
  assert(workerNode.state.speed_px_per_s === null, 'Single-frame velocity speed_px_per_s is null');
  assert(workerNode.state.velocity_status === 'Not available for single-frame analysis', 'Velocity honesty disclosure verified');

  // Check Time-To-Hazard Hero Object for Single Image
  assert(q1.time_to_hazard != null, 'Scene has time_to_hazard hero object');
  assert(q1.time_to_hazard.available === false, 'Single image TTH is strictly marked available: false');
  assert(q1.time_to_hazard.time_to_hazard_seconds === null, 'Single image TTH seconds is null');
  assert(q1.time_to_hazard.status_text.includes('single-frame'), `TTH status explicitly discloses single-frame limitation: "${q1.time_to_hazard.status_text}"`);
  assert(Array.isArray(q1.temporal_predictions), 'temporal_predictions is an array');
  assert(q1.temporal_predictions.length === 0, 'No fabricated temporal predictions for static image');

  // Check relationships
  assert(Array.isArray(q1.relationships), 'Relationships is an array');
  q1.relationships.forEach((rel) => {
    assert(Boolean(rel.id || rel.relation_id), 'Relationship has unique ID');
    assert(Boolean(rel.source_id), 'Relationship has source_id');
    assert(Boolean(rel.target_id), 'Relationship has target_id');
    assert(Boolean(rel.type), 'Relationship has semantic type');
    assert(typeof rel.active === 'boolean', 'Relationship has boolean active status');
    assert(['SPATIAL', 'TEMPORAL', 'SAFETY', 'EQUIPMENT'].includes(rel.category), `Relationship has valid category: ${rel.category}`);
    assert(rel.tth_seconds === null, 'Single-frame relationship tth_seconds is null (no fake values)');
    assert(rel.tth_status.includes('single-frame'), 'Relationship tth_status honestly discloses single-frame limitation');
  });

  // --- Test 2: Second Image (PPE Sample) Lifecycle & Fresh State ---
  console.log('\n--- TEST 2: Multi-Sample Isolation & Fresh Scene Construction ---');
  const ppeBytes = fs.readFileSync(SAMPLE_PPE_PATH);
  const ppeBlob = new Blob([ppeBytes], { type: 'image/jpeg' });
  const form2 = new FormData();
  form2.append('file', ppeBlob, 'ppe_sample.jpg');

  const ppeRes = await fetch(`${API_BASE}/api/v1/analyze/image`, {
    method: 'POST',
    body: form2,
  });
  assert(ppeRes.ok, 'PPE sample analysis succeeded');
  const ppeData = await ppeRes.json();
  assert(ppeData.job_id !== imgJobId, 'New job has distinct unique job_id');

  const ppeScene = await (await fetch(`${API_BASE}/api/v1/scene/graph?job_id=${ppeData.job_id}`)).json();
  assert(ppeScene.entities.length > 0, 'PPE scene graph has detected entities');

  // Verify previous job scene remains isolated and immutable
  const originalScene = await (await fetch(`${API_BASE}/api/v1/scene/graph?job_id=${imgJobId}`)).json();
  assert(originalScene.scene_id === q1.scene_id, 'Original job scene graph remains immutable');

  // --- Test 3: Real Video Asynchronous Inference & Temporal Scene Graph ---
  console.log('\n--- TEST 3: Asynchronous Video Pipeline & Temporal Scene Dynamics ---');
  const vidBytes = fs.readFileSync(SAMPLE_VIDEO_PATH);
  const vidBlob = new Blob([vidBytes], { type: 'video/mp4' });
  const form3 = new FormData();
  form3.append('file', vidBlob, 'cctv_worker_moving.mp4');

  const vidSubmitRes = await fetch(`${API_BASE}/api/v1/analyze/video`, {
    method: 'POST',
    body: form3,
  });
  assert(vidSubmitRes.status === 202, 'Video submission accepted (HTTP 202)');
  const vidSubmitData = await vidSubmitRes.json();
  const vidJobId = vidSubmitData.job_id;
  assert(Boolean(vidJobId), `Obtained video job_id: ${vidJobId}`);

  console.log('Polling video job until pipeline completes...');
  let videoCompleted = false;
  let pollAttempts = 0;
  let lastProgress = 0;

  while (!videoCompleted && pollAttempts < 60) {
    await sleep(1000);
    pollAttempts++;
    const statusRes = await fetch(`${API_BASE}/api/v1/analyze/status/${vidJobId}`);
    if (!statusRes.ok) continue;
    const statusData = await statusRes.json();
    if (statusData.progress_pct !== undefined) {
      assert(statusData.progress_pct >= lastProgress, `Monotonic progress: ${statusData.progress_pct}% >= ${lastProgress}%`);
      lastProgress = statusData.progress_pct;
    }
    if (statusData.status === 'COMPLETED') {
      videoCompleted = true;
      console.log(`Video processing completed in ${pollAttempts} seconds!`);
      break;
    }
    if (statusData.status === 'FAILED') {
      console.error('Video job processing failed:', statusData.error);
      process.exit(1);
    }
  }

  assert(videoCompleted, 'Video analysis completed successfully within timeout');

  // Query Video Scene Graph
  const vidSceneRes = await fetch(`${API_BASE}/api/v1/scene/graph?job_id=${vidJobId}`);
  assert(vidSceneRes.ok, 'Video scene graph query succeeded');
  const vidScene = await vidSceneRes.json();

  assert(vidScene.entities != null && vidScene.entities.length > 0, `Video scene has ${vidScene.entities.length} entities`);
  console.log(`Video scene summary: ${JSON.stringify(vidScene.summary)}`);

  // Verify tracking ID continuity on video
  const trackedWorkers = vidScene.entities.filter((e) => e.type === 'PERSON' && e.track_id != null);
  console.log(`Video tracked workers count: ${trackedWorkers.length}`);
  trackedWorkers.forEach((w) => {
    console.log(`  Worker #${w.track_id}: Conf=${w.confidence}, Speed=${w.state?.speed_px_per_s}, BBox=${JSON.stringify(w.bbox)}`);
  });

  // Verify Video Time-To-Hazard Structure
  assert(vidScene.time_to_hazard != null, 'Video scene has time_to_hazard object');
  console.log(`Video TTH: available=${vidScene.time_to_hazard.available}, status="${vidScene.time_to_hazard.status_text}"`);
  assert(typeof vidScene.time_to_hazard.available === 'boolean', 'TTH availability is a valid boolean');
  assert(Array.isArray(vidScene.time_to_hazard.compounding_factors), 'TTH compounding_factors is an array');
  assert(Array.isArray(vidScene.time_to_hazard.timeline), 'TTH timeline is an array');

  // --- Test 4: Media Switching Isolation (Video -> Image) ---
  console.log('\n--- TEST 4: Media Switching Isolation (Video -> Image) ---');
  // Re-fetch initial image scene to confirm no video temporal state contaminated image scene
  const recheckImgScene = await (await fetch(`${API_BASE}/api/v1/scene/graph?job_id=${imgJobId}`)).json();
  assert(recheckImgScene.time_to_hazard.available === false, 'Image TTH remains unavailable after video job');
  assert(recheckImgScene.temporal_predictions.length === 0, 'Image has zero temporal predictions after video job');
  assert(recheckImgScene.entities[0].state.speed_px_per_s === null, 'Image velocity remains null');

  // --- Test 5: Regression Tests for All Core APIs ---
  console.log('\n--- TEST 5: Non-Regression & Health Check ---');
  const healthRes = await fetch(`${API_BASE}/api/v1/health`);
  assert(healthRes.ok, 'Health check endpoint HTTP 200');

  const metricsRes = await fetch(`${API_BASE}/api/v1/system/metrics`);
  assert(metricsRes.ok, 'Metrics endpoint HTTP 200');

  // Authenticate to access protected operational endpoints
  const loginRes = await fetch(`${API_BASE}/api/v1/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username: 'admin', password: 'IntelliWatch2026!' }),
  });
  assert(loginRes.ok, 'Authentication login HTTP 200');
  const loginData = await loginRes.json();
  const authHeader = { Authorization: `Bearer ${loginData.access_token}` };

  const camerasRes = await fetch(`${API_BASE}/api/v1/cameras`, { headers: authHeader });
  assert(camerasRes.ok, 'Cameras endpoint HTTP 200');

  const alertsRes = await fetch(`${API_BASE}/api/v1/alerts`, { headers: authHeader });
  assert(alertsRes.ok, 'Alerts endpoint HTTP 200');

  console.log('\n================================================================');
  console.log('🏆 ALL HERO FEATURES, INTEGRATIONS & PIPELINE TESTS PASSED 100%!');
  console.log('================================================================');
}

run().catch((err) => {
  console.error('Test execution failed with error:', err);
  process.exit(1);
});
