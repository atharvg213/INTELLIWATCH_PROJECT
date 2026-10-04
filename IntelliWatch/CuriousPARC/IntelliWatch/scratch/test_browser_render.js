const fs = require('fs');

// Read dashboard.js
const code = fs.readFileSync('frontend/js/dashboard.js', 'utf-8');

// Create mock DOM environment
const mockElement = () => ({
  textContent: '',
  innerHTML: '',
  style: {},
  classList: { add: () => {}, remove: () => {} },
  getContext: () => ({
    clearRect: () => {},
    closePath: () => {},
    fillRect: () => {},
    strokeRect: () => {},
    beginPath: () => {},
    moveTo: () => {},
    lineTo: () => {},
    stroke: () => {},
    fill: () => {},
    arc: () => {},
    fillText: () => {},
    measureText: () => ({ width: 50 }),
    save: () => {},
    restore: () => {},
    setLineDash: () => {},
    roundRect: () => {},
  }),
  getBoundingClientRect: () => ({ left: 0, top: 0, width: 800, height: 600 }),
});

global.document = {
  getElementById: (id) => mockElement(),
  querySelectorAll: () => [],
  addEventListener: () => {},
};
global.window = {
  addEventListener: () => {},
  location: { href: 'http://localhost:8000' },
};
global.localStorage = {
  getItem: () => null,
  setItem: () => {},
  removeItem: () => {},
};

const vm = require('vm');

// Evaluate dashboard.js in this context
try {
  vm.runInThisContext(code);
  console.log("dashboard.js loaded successfully into runtime.");
} catch (e) {
  console.error("Failed to load dashboard.js:", e);
  process.exit(1);
}

// Test sample assessment from backend
const sampleAssessment = {
  frame_width: 1264,
  frame_height: 848,
  workers_count: 10,
  detections: Array(10).fill({
    class_id: 0,
    class_name: "person",
    confidence: 0.9,
    bbox: { x1: 100, y1: 100, x2: 200, y2: 300 }
  }),
  tracks: Array(10).fill(null).map((_, i) => ({
    track_id: i + 1,
    class_id: 0,
    class_name: "person",
    confidence: 0.9,
    bbox: { x1: 100, y1: 100, x2: 200, y2: 300 },
    state: "ACTIVE"
  })),
  scene: { relationships: [] },
  situational_summary: { total_entities: 10, workers_count: 10 },
  active_events: [],
  risk_assessment: { risk_level: "INFO", risk_score: 0.1 },
  worker_inventories: [],
  behavior_states: [],
  depth_statistics: null,
  object_depths: [],
  entity_profiles: [],
  temporal_events: Array(10).fill(null).map((_, i) => ({
    seq: i + 1,
    timestamp: 0.0,
    frame_id: 1,
    category: "ENTITY_DETECTED",
    entity_id: `person_${i + 1}`,
    label: `Person #${i + 1} detected`,
    description: `New entity appeared`,
    new_state: "DETECTED"
  })),
};

console.log("Invoking renderAssessment(sampleAssessment)...");
try {
  // Simulate single-frame upload state
  state.isSingleFrame = true;
  renderAssessment(sampleAssessment);
  console.log("SUCCESS: renderAssessment executed without errors!");
  
  // Verify temporal rendering
  const profContainer = elements.temporalEntityProfiles;
  console.log("Temporal profiles container content length:", profContainer.innerHTML.length);
  console.log("Contains single-frame notice:", profContainer.innerHTML.includes("single-frame-notice"));
  
  const timeContainer = elements.temporalTimeline;
  console.log("Temporal timeline container content length:", timeContainer.innerHTML.length);
  console.log("Contains timeline entries:", timeContainer.innerHTML.includes("temporal-entity-timeline-entry"));

  // Test multi-frame video case
  state.isSingleFrame = false;
  sampleAssessment.entity_profiles = [{
    entity_id: "person_1",
    track_id: 1,
    class_name: "worker",
    current_behavior: "MOVING",
    behavior_duration_s: 2.5,
    current_spatial_state: "NEAR",
    image_speed_px_per_s: 45.0,
  }];
  renderAssessment(sampleAssessment);
  console.log("SUCCESS (multi-frame): renderAssessment executed without errors!");
  console.log("Multi-frame profiles contains card:", profContainer.innerHTML.includes("temporal-entity-card"));
} catch (err) {
  console.error("FAIL: renderAssessment threw error:", err);
  process.exit(1);
}
