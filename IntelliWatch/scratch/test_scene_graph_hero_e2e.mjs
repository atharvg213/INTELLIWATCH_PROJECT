/**
 * E2E Verification Script for Scene Graph Hero Feature
 */

import fs from 'node:fs';
import path from 'node:path';

const API_BASE = 'http://127.0.0.1:8000';
const SAMPLE_IMAGE_PATH = 'CuriousPARC/IntelliWatch/data/samples/industrial_cctv.jpg';

function assert(condition, message) {
  if (!condition) {
    console.error(`❌ ASSERTION FAILED: ${message}`);
    process.exit(1);
  }
  console.log(`✅ ${message}`);
}

async function run() {
  console.log('--- Step 1: Verify Static UI and Endpoints Served ---');
  const indexRes = await fetch(`${API_BASE}/`);
  assert(indexRes.ok, 'Root index.html is served successfully (HTTP 200)');
  const indexHtml = await indexRes.text();
  assert(indexHtml.includes('app.js'), 'index.html includes app.js script');

  const appJsRes = await fetch(`${API_BASE}/app.js`);
  assert(appJsRes.ok, 'app.js is served successfully');
  const appJs = await appJsRes.text();
  assert(appJs.includes('/app/scene-graph'), 'app.js has /app/scene-graph route');
  assert(appJs.includes('sceneGraphPage'), 'app.js defines sceneGraphPage');
  assert(appJs.includes('renderSceneGraphSvg'), 'app.js defines renderSceneGraphSvg');
  assert(appJs.includes('renderEntityInspector'), 'app.js defines renderEntityInspector');
  assert(appJs.includes('inspect-person-in-graph'), 'app.js has bi-directional focus action inspect-person-in-graph');
  assert(appJs.includes('focus-in-inference-breakdown'), 'app.js has bi-directional focus action focus-in-inference-breakdown');

  console.log('\n--- Step 2: Upload Real Image & Run Neural Inference ---');
  const imageBytes = fs.readFileSync(SAMPLE_IMAGE_PATH);
  const blob = new Blob([imageBytes], { type: 'image/jpeg' });
  const form = new FormData();
  form.append('file', blob, 'industrial_cctv.jpg');

  const uploadRes = await fetch(`${API_BASE}/api/v1/analyze/image`, {
    method: 'POST',
    body: form,
  });
  assert(uploadRes.ok, 'Image analysis succeeded (HTTP 200)');
  const uploadData = await uploadRes.json();
  const jobId = uploadData.job_id;
  assert(Boolean(jobId), `Received valid job_id: ${jobId}`);
  assert(uploadData.detections_count > 0, `Detected ${uploadData.detections_count} objects in image`);

  console.log('\n--- Step 3: Query Scene Graph via API Endpoints ---');
  // Test /api/v1/scene/current
  const currentSceneRes = await fetch(`${API_BASE}/api/v1/scene/current`);
  assert(currentSceneRes.ok, 'GET /api/v1/scene/current succeeded');
  const currentScene = await currentSceneRes.json();

  // Test /api/v1/scene/graph
  const graphRes = await fetch(`${API_BASE}/api/v1/scene/graph?job_id=${jobId}`);
  assert(graphRes.ok, 'GET /api/v1/scene/graph?job_id=... succeeded');
  const graphData = await graphRes.json();

  console.log('\n--- Step 4: Validate Scene Graph Hero Schema & Canonical Aliases ---');
  assert(Array.isArray(graphData.entities), 'graphData contains entities array');
  assert(Array.isArray(graphData.relationships), 'graphData contains relationships array');
  assert(graphData.entities.length > 0, `Graph contains ${graphData.entities.length} real detected entities`);
  assert(graphData.relationships.length > 0, `Graph contains ${graphData.relationships.length} real relationships`);

  console.log('Sample entities detected:');
  graphData.entities.forEach((e) => {
    console.log(`  - [${e.type}] ID: ${e.id}, Label: "${e.label}", Conf: ${e.confidence}, Risk: ${e.risk?.level}`);
  });

  console.log('Sample relationships detected:');
  graphData.relationships.forEach((r) => {
    console.log(`  - ${r.source_id} --[${r.type}]--> ${r.target_id} (State: ${r.state})`);
  });

  // Verify entity structure
  const personEntity = graphData.entities.find((e) => e.type === 'PERSON');
  assert(Boolean(personEntity), 'Found PERSON entity in scene graph');
  assert(personEntity.state != null, 'PERSON entity has state dictionary');
  assert(personEntity.risk != null, 'PERSON entity has risk assessment');
  assert(personEntity.position != null, 'PERSON entity has position data');

  // Verify technical honesty: Single frame has no velocity or temporal trajectory
  assert(
    personEntity.state.speed_px_per_s === null,
    'Honesty check: speed_px_per_s is null for single frame'
  );
  assert(
    personEntity.state.velocity_status === 'Not available for single-frame analysis',
    'Honesty check: velocity_status explicitly discloses "Not available for single-frame analysis"'
  );

  // Verify relationship aliases
  const firstRel = graphData.relationships[0];
  assert(Boolean(firstRel.source_id), `Relationship has canonical source_id: ${firstRel.source_id}`);
  assert(Boolean(firstRel.target_id), `Relationship has canonical target_id: ${firstRel.target_id}`);
  assert(Boolean(firstRel.type), `Relationship has canonical type: ${firstRel.type}`);
  assert(['ACTIVE', 'CREATED', 'UNKNOWN'].includes(firstRel.state), `Relationship has valid lifecycle state: ${firstRel.state}`);

  console.log('\n--- Step 5: Test Non-Existent Job ID Handling ---');
  const notFoundRes = await fetch(`${API_BASE}/api/v1/scene/graph?job_id=non_existent_job_12345`);
  assert(notFoundRes.status === 404, 'Non-existent job returns HTTP 404 cleanly');

  console.log('\n--- Step 6: Test Upload New Image to Verify Scene Refresh ---');
  const secondUploadRes = await fetch(`${API_BASE}/api/v1/analyze/image`, {
    method: 'POST',
    body: form,
  });
  assert(secondUploadRes.ok, 'Second image analysis succeeded');
  const secondJobData = await secondUploadRes.json();
  assert(secondJobData.job_id !== jobId, 'Second job has unique new job_id');

  const secondGraphRes = await fetch(`${API_BASE}/api/v1/scene/graph?job_id=${secondJobData.job_id}`);
  assert(secondGraphRes.ok, 'Second graph query succeeded');
  const secondGraph = await secondGraphRes.json();
  assert(secondGraph.entities.length > 0, 'New scene graph is constructed for second job');

  console.log('\n======================================================');
  console.log('🎉 ALL SCENE GRAPH HERO E2E VERIFICATIONS PASSED 100%!');
  console.log('======================================================');
}

run().catch((err) => {
  console.error('Fatal error in E2E runner:', err);
  process.exit(1);
});
