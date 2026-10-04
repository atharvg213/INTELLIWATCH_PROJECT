# IntelliWatch: AI-Powered Scene Understanding for Industrial Safety & Monitoring

IntelliWatch is a modular, competition-level computer vision platform engineered to transform ordinary industrial CCTV and video streams into an intelligent, proactive monitoring system for factory floors and high-risk operational environments.

---

## 1. Project Objective

Industrial facilities frequently experience preventable accidents, compliance violations, and operational bottlenecks. Traditional CCTV systems are passive, relying on manual observation and post-incident review.

**IntelliWatch** bridges this gap by applying cutting-edge computer vision, multi-object tracking, spatial reasoning, and temporal event analysis to:
- Detect safety hazards in real-time.
- Enforce strict Personal Protective Equipment (PPE) compliance.
- Safeguard restricted and hazardous zones.
- Analyze worker-machine proximity and predict potential collisions.
- Generate explainable, contextual alerts with verifiable visual evidence.

---

## 2. Current Project Stage

**Current Stage: Step 25 — Production Hardening & Release Readiness (448 tests passing, 0 failures, 0 regressions)**

> **Completed Pipeline Progression:**
> - **Step 1:** Project foundation, architecture & schemas
> - **Step 2:** Video ingestion, frame preprocessing & letterbox transformation
> - **Step 3:** General object detection using YOLO11n
> - **Step 4:** Multi-object tracking (ByteTrack) with persistent identities
> - **Step 5A:** Specialized industrial PPE detection using SafetyVision YOLOv8n
> - **Step 5B:** Worker-PPE spatial association, inventory evaluation & temporal compliance
> - **Step 6:** Polygon-based restricted zones, foot-point ground localization, point-in-polygon reasoning, temporal entry confirmation, dwell duration analytics & explainable zone events
> - **Step 7:** Monocular depth estimation (Depth-Anything-V2-Small), native resolution coordinate alignment, object-level depth distribution sampling, contact-point depth estimation & spatial visualization
> - **Step 8:** Behavioral state machine, kinematic temporal tracking, velocity/acceleration/direction profiling, loitering/prolonged stationarity detection, heuristic geometric fall alerting & explainable behavior events
> - **Step 9:** Relational scene graph representation, factual entity-relationship modeling (PPE wearing, restricted-zone occupancy, behavior profiling, spatial proximity & approaching kinematics), temporal relationship lifecycle tracking (CREATED/ACTIVE/ENDED), and machine-readable situational summaries
> - **Step 10:** Safety Risk & Event Reasoning engine, deterministic multi-factor risk scoring, controlled event taxonomy, temporal confirmation & event lifecycles (CANDIDATE/CONFIRMED/ACTIVE/ENDED), compound multi-hazard events, anti-flapping & cooldown suppression, and situational risk telemetry
> - **Step 11:** Predictive & Advanced Anomaly Intelligence: short-horizon image-space trajectory projection, hazard-direction analysis (toward restricted zones, vehicles, machinery), risk escalation trend modeling, repeated violation pattern detection, explainable temporal behavioral anomaly detection (fall-and-collapse sequences, erratic locomotion oscillation), early-warning indicator lifecycle (CANDIDATE/CONFIRMED/ACTIVE/ENDED), anti-flicker temporal confirmation, and prediction API endpoints (`/api/v1/prediction/current`, `/api/v1/prediction/indicators`)
> - **Step 12:** End-to-End Orchestrator, Incident Evidence System (`data/output/incidents/`), Professional Industrial Operator Console (dark-mode glassmorphic UI served at `/` and `/dashboard`), Demo Video Processing Runner (`scripts/run_demo.py`), System Telemetry (`/api/v1/status`), and Comprehensive Test Suite
> - **Step 15:** Temporal Intelligence, Multi-Frame Event Progression, Real-Video Validation, and Evidence Persistence
> - **Step 16:** Interactive Safety Zones & Scene Configuration: In-browser polygon drawing, coordinate system transformation with letterbox/scaling inversion, ZoneService CRUD API, disk persistence (`configs/zones.json`), and dynamic ZoneEngine / Risk integration
> - **Step 17:** Explainability, Evidence & Safety Analytics: Deterministic Five-W (Who, What, Where, When, Why) structured incident audit, full arithmetic risk score breakdown with multi-hazard escalation transparency, before/during/after event timelines from temporal logs, risk trend trajectory visualization, auditable JSON evidence packages with verification disclaimers, multi-parameter incident filtering, and non-punitive descriptive safety analytics
> - **Step 18:** Final Evaluation, Benchmarking, UI Polish & Competition Preparation: Comprehensive real-image/real-video validation, CPU performance profiling, edge/failure-case hardening, verified feature matrix, complete technology stack documentation, architecture diagrams, and repeatable competition demonstration workflow
> - **Step 19:** GPU-Accelerated Accuracy Improvement: Hardware probe, RTX 5050 diagnostics, automated fallback and device reporting
> - **Step 20:** GPU Compatibility and Real-World Accuracy Improvement: Blackwell architecture diagnostics, CUDA 12.6 verification, device transparency
> - **Step 21:** Real-Time RTSP Ingestion and Multi-Camera Stream Processing: Background stream capture with OpenCV, reconnect resilience, frame sampling, max-FPS rate limiting, dynamic CameraManager, multi-camera live dashboard grid, MJPEG streaming, and per-camera snapshot/telemetry
> - **Step 22:** Intelligent Safety Alerting, Evidence Capture & Alert Lifecycle: Centralized AlertEngine converting PPE non-compliance, falls, and zone hazards into actionable tickets; severity grading (CRITICAL, HIGH, MEDIUM, LOW) with immediate critical bypass; temporal deduplication and cooldown windows; timestamped visual evidence snapshot capture with bounding boxes and watermark; SQLite-backed persistent AlertStore (`data/alerts.db`) with full lifecycle state machine (`NEW` -> `ACKNOWLEDGED` -> `RESOLVED` / `DISMISSED`) and audit history; REST API (`/api/v1/alerts*`); and interactive control room dashboard feed with operator action modals (435 passing tests)
> - **Step 23:** Authentication, Role-Based Access Control, Audit Logs & Administration: PBKDF2-HMAC-SHA256 password hashing (100k iterations, per-user salt), cryptographic session token lifecycle with instant server-side revocation and configurable expiration; brute-force protection (5 attempts / 300s lockout); 4-tier RBAC (`ADMIN`, `SAFETY_MANAGER`, `OPERATOR`, `VIEWER`) with 10 granular backend permissions and camera-level access scoping (`permitted_cameras`); administrative user management CRUD with last-admin deletion guard; append-only structured SQLite audit trail (`data/auth.db`) with automated secret/PII redaction; control room dashboard integration with role badges, permission-aware navigation, User Admin panel, and filtered Audit Log viewer (447 passing tests)
> - **Step 24:** Frontend & Landing Page Integration: Seamless integration of the modern industrial visual intelligence design system into `CuriousPARC/IntelliWatch/frontend/`. Features an editorial landing page served at `/` and `/landing` with live facility metrics (`CAMERAS ONLINE`, `EVENTS TODAY`, `ACTIVE INCIDENTS`), interactive industrial environment preview (Manufacturing, Warehousing, Construction, Energy), and clear "Launch Dashboard" navigation.
> - **Step 25:** Production Hardening & Release Readiness: Security audit & hardening (configurable non-wildcard CORS, global unhandled exception mask preventing internal path/traceback leakage, strict path traversal boundary verification for evidence endpoints, payload size limits for image and video analysis, SYSTEM_CONFIG permission enforcement and audit logging on all safety zone mutations, production credential warnings, placeholder-only `.env.example`, and clean Git exclusion verification with zero secrets or runtime databases tracked). (448 passing tests)

---

## 3. What Object Detection Means in IntelliWatch

Object detection answers the fundamental perception question:
> **"What objects are present in this frame, where are they located, and how confident is the model?"**

Each detection outputs:
```text
- Class Name   : Human-readable object label (e.g., 'person', 'truck')
- Class ID     : Numeric identifier corresponding to model class index
- Confidence   : Detection confidence score between 0.00 and 1.00 (e.g., 0.82)
- Bounding Box : Pixel coordinates in the ORIGINAL frame resolution (x1, y1, x2, y2)
- Frame Index  : Monotonically increasing sequential frame counter
- Timestamp    : Stream elapsed time in seconds (e.g., 1.40s)
```

### Coordinate Convention
Bounding boxes use the standard top-left / bottom-right pixel convention:
- `(x1, y1)`: Top-left pixel coordinate
- `(x2, y2)`: Bottom-right pixel coordinate
- Coordinate origin `(0, 0)` is at the top-left corner of the video frame.

---

## 4. Model Weights & Technical Honesty

### Pretrained General-Purpose Model
IntelliWatch Step 3 utilizes **YOLO11 Nano (`yolo11n.pt`)** from Ultralytics (~5.4 MB, 80 COCO classes).
- Weights are stored locally under `weights/yolo11n.pt` and are excluded from Git via `.gitignore`.
- Primary classes relevant for industrial monitoring:
  - `person` (Class 0)
  - `bicycle` (Class 1), `car` (Class 2), `motorcycle` (Class 3), `bus` (Class 5), `truck` (Class 7)
  - `backpack` (Class 24), `handbag` (Class 26), `suitcase` (Class 28)

### Important Clarification on PPE Detection
> **Technical Honesty Notice:**
> Standard pretrained COCO weights do **NOT** detect specialized industrial PPE classes such as hard hats (`helmet`), high-visibility vests (`vest`), safety goggles, or steel-toed boots. In Step 3, the detector architecture and coordinate mapping pipeline are fully operational. Custom PPE detection will be enabled in a future step via fine-tuned domain-specific weights without changing the detector interface.

---

## 5. Coordinate Transformation (Letterbox Inversion)

When frames of arbitrary aspect ratios (e.g. 1376×768 or 1920×1080) are ingested, [`FramePreprocessor`](vision/preprocessing/frame_processor.py) letterboxes them into uniform 640×640 dimensions, tracking:
- `scale_factor`: The resizing ratio `r = min(target_w / orig_w, target_h / orig_h)`
- `pad_offset`: Padding offsets `(pad_top, pad_left)`

The [`unpad_and_rescale_bbox`](vision/detection/coordinates.py) function reverses this transformation:
```text
Preprocessed 640x640 Box
           ↓
Undo Padding  : x - pad_left, y - pad_top
           ↓
Undo Scaling  : x / scale_factor, y / scale_factor
           ↓
Boundary Clamp: max(0, min(orig_w, x)), max(0, min(orig_h, y))
           ↓
Original CCTV Coordinate Bounding Box
```

---

## 6. Current Pipeline Flow (Step 3)

```text
       Video File / CCTV Stream
                 ↓
       VideoReader (vision/preprocessing/video_reader.py)
                 ↓
       Frame Sampling (frame_skip = N)
                 ↓
       FramePreprocessor (Letterbox 640x640, Preserve Aspect Ratio)
                 ↓
       FrameData (In-memory typed representation)
                 ↓
       YOLODetector (vision/detection/yolo_detector.py)
       - YOLO11n Tensor Inference
       - Confidence & IoU Filtering
       - Coordinate Inversion (unpad_and_rescale_bbox)
                 ↓
       FrameDetections (Pydantic schema in original coordinates)
                 ↓
       DetectionVisualizer (Annotation & Evidence Crop Storage)
```

---

## 7. Directory Structure

```text
IntelliWatch/
│
├── backend/                  # FastAPI Application & Core Services
│   ├── api/                  # REST API Endpoints (/health, /api/v1/config, /api/v1/video/metadata)
│   ├── services/             # Pipeline & Service Orchestration
│   ├── models/               # Database ORM Models (Planned for DB step)
│   ├── schemas/              # Pydantic Schemas (detection, tracking, events, video)
│   ├── database/             # Database Connection Management
│   └── main.py               # Backend Application Entrypoint
│
├── vision/                   # Perception & Computer Vision Modules
│   ├── detection/            # Object Detection (Step 3)
│   │   ├── base.py           # BaseDetector Abstract Interface
│   │   ├── yolo_detector.py  # YOLODetector (YOLO11 / YOLOv8 Integration)
│   │   ├── coordinates.py    # Letterbox Inversion & Boundary Clamping
│   │   ├── visualizer.py     # Bounding Box & Label Rendering Utility
│   │   └── pipeline.py       # End-to-End Detection Pipeline
│   ├── tracking/             # Multi-Object Tracking Interfaces (ByteTrack)
│   ├── depth/                # Depth Estimation Interfaces (Monocular Depth)
│   ├── segmentation/         # Instance/Semantic Segmentation Interfaces
│   ├── pose/                 # Human Pose Estimation Interfaces
│   └── preprocessing/        # Video Ingestion & Frame Processing (Step 2)
│       ├── frame.py          # FrameData In-memory Representation
│       ├── video_reader.py   # VideoReader & Metadata Extraction
│       ├── frame_processor.py# Letterbox Resizing & Preprocessing
│       └── pipeline.py       # Reusable Video Processing Pipeline
│
├── intelligence/             # Reasoning & Decision Engines
│   ├── zones/                # Polygonal Geofencing & Intrusion Logic
│   ├── behavior/             # Temporal Behavior & Motion Analysis
│   ├── scene_graph/          # Entity-Relationship Modeling
│   ├── events/               # Industrial Safety Event Detectors
│   ├── risk/                 # Composite Risk Scoring Engine
│   └── prediction/           # Trajectory & Collision Predictors
│
├── data/                     # Media & Stream Data (Git Ignored)
│   ├── input/                # Incoming Video Feeds
│   ├── output/               # Processed Frames & Evidence Crops
│   └── samples/              # Test Media (synthetic_test.mp4, sample_test.jpg)
│
├── weights/                  # Downloaded Pretrained Checkpoints (Git Ignored)
│   └── yolo11n.pt            # YOLO11 Nano Pretrained Model
│
├── configs/                  # Central Configuration & Logging
│   ├── settings.py           # Pydantic Settings & Environment Loading
│   └── logging_config.py     # Unified Logging Formatter & Handlers
│
├── tests/                    # Pytest Suite (34 Tests Passing)
│   ├── test_imports.py       # Import & Interface Verification
│   ├── test_config.py        # Configuration Loading Tests
│   ├── test_schemas.py       # Pydantic Schema Validation Tests
│   ├── test_health_api.py    # Health Check API Endpoint Tests
│   ├── test_video_pipeline.py# Step 2 Video & Preprocessing Tests
│   ├── test_coordinates.py   # Coordinate Inversion & Clamping Tests
│   └── test_detection.py     # Detector, Mock Inference, & Visualizer Tests
│
├── scripts/                  # Utility & Execution Scripts
│   ├── run_server.py         # Launch Script for FastAPI Server
│   ├── verify_setup.py       # Setup Verification
│   ├── create_synthetic_sample.py # Offline Test Video Generator
│   ├── test_video_pipeline.py# Step 2 Video Pipeline Runner
│   ├── test_detection_image.py   # Step 3 Single Image Detection Tool
│   └── test_detection_pipeline.py# Step 3 Video Detection Pipeline Runner
│
├── pytest.ini                # Pytest Test Discovery Configuration
├── requirements.txt          # Python Dependencies
├── .env.example              # Environment Variable Template
├── .gitignore                # Git Exclusions
└── README.md                 # Project Overview & Architecture Guide
```

---

## 8. Getting Started & Verification

### 1. Single Image Object Detection Verification
Verify YOLO detection on a sample image:
```bash
python scripts/test_detection_image.py data/samples/sample_test.jpg --conf 0.25
```
Parameters:
- `--conf N`: Confidence threshold (e.g. `--conf 0.35`).
- `--classes person truck`: Filter for specific classes.
- `--output-dir data/output`: Saves annotated image with mapped original coordinates.

### 2. Video Stream Object Detection Verification
Execute detection on a video stream with frame sampling:
```bash
python scripts/test_detection_pipeline.py data/samples/synthetic_test.mp4 --frame-skip 5 --save-frames 3
```
Parameters:
- `--frame-skip N`: Sample every Nth frame (default: 5).
- `--conf N`: Confidence threshold (default: 0.25).
- `--save-frames N`: Saves first N annotated frames into `data/output/` for inspection.

### 3. Placing Your Own Local Test Media
You can place any local test video or photo into:
```text
data/samples/your_cctv_feed.mp4
data/samples/your_cctv_photo.jpg
```
Run detection directly:
```bash
python scripts/test_detection_pipeline.py data/samples/your_cctv_feed.mp4 --frame-skip 2
```

### 4. Running the Complete Automated Test Suite
```bash
pytest
```
All **72 unit and integration tests** will execute cleanly and deterministically on CPU.


### 5. Health Check
```bash
curl http://127.0.0.1:8000/health
```
Response:
```json
{
    "status": "ok",
    "project": "IntelliWatch"
}
```

---

## 9. Step 4 — Multi-Object Tracking (ByteTrack)

### Purpose & Architecture
Multi-Object Tracking (MOT) converts independent frame-level detections into persistent spatio-temporal entities.

```text
Video / CCTV Stream
       ↓
Frame Preprocessing (Letterbox 640x640)
       ↓
YOLO11n Object Detection
       ↓
Coordinate Re-projection (Original Video Pixel Space)
       ↓
ByteTrack Multi-Object Tracker (Kalman Filter + IoU Association)
       ↓
FrameTracks (Persistent Track IDs, Centroids, Trajectories, Image-Space Motion)
       ↓
TrackingVisualizer (Trajectory Trails & Unique ID Badges)
```

* **Detection answers:** *"What objects are visible in this single frame?"*
* **Tracking answers:** *"Which detected object in this frame corresponds to the same entity seen in previous frames?"*

### Key Features
* **ByteTrack Association:** Two-stage Kalman filter state propagation and Hungarian matching that retains identity across temporary detection loss without GPU requirement.
* **Trajectory History:** Historical centroid points stored with configurable limit (`TRACK_HISTORY_LENGTH = 30`).
* **Image-Space Motion:** Exposes image-space displacement (`velocity_x`, `velocity_y`) and magnitude (`speed_pixels_per_second`). Note: this is strictly image-space motion in pixels, not real-world metric speed.
* **Coordinate Fidelity:** All bounding boxes, centroids, and trajectories remain anchored in the original video resolution.

### Verification Command
```bash
python scripts/test_tracking_pipeline.py data/samples/synthetic_test.mp4 --frame-skip 1
```

---

## 10. Step 5A — Personal Protective Equipment (PPE) Detection

### Why a Separate PPE Model?
The general-purpose YOLO11n model is trained on the 80 COCO categories (`person`, `car`, `truck`, etc.) and does not recognize industrial safety gear such as hard hats, high-visibility safety vests, protective gloves, or safety goggles. Rather than replacing or compromising the general scene detector, IntelliWatch uses a modular two-model architecture:

```text
                   Video Frame
                        │
         ┌──────────────┴──────────────┐
         ▼                             ▼
General Scene Perception        PPE Perception Layer
  (YOLO11n COCO Detector)         (SafetyVision YOLOv8n)
         │                             │
  Workers, Machinery,           Hardhat, Safety Vest,
  Vehicles, Equipment           Gloves, Goggles, Masks
         │                             │
         ▼                             ▼
    ByteTrack MOT              PPE Detections (Original Coords)
         │                             │
         └──────────────┬──────────────┘
                        ▼
           [Step 5B: Future Association]
```

### Model Details
* **Candidate Model:** SafetyVision YOLOv8n (`ayushgupta7777/safetyvision-yolov8`)
* **Framework:** PyTorch / Ultralytics YOLO
* **Weights:** `weights/ppe_yolov8n.pt` (~6.2 MB, local file)
* **Execution Mode:** Explicit CPU execution (`PPE_DEVICE="cpu"`)
* **License:** AGPL-3.0 (Open-Source)
* **Recognized Categories:** `Hardhat`, `Safety Vest`, `Gloves`, `Goggles`, `Mask`, `Person`, `Fall-Detected`, `No_Harness`, and negative/absence indicators (`NO-Hardhat`, `NO-Safety Vest`, `NO-Gloves`, `NO-Goggles`, `NO-Mask`).

### Critical Architectural Distinction (Step 5A vs. Step 5B)
* **Step 5A:** Detects PPE equipment instances in image space.
  * *"What PPE objects are visible?"*
  * *"A hard hat is located at [x1, y1, x2, y2]."*
* **Step 5B:** Associates detected PPE items with specific tracked workers (`TrackedObject`) and evaluates deterministic compliance rules with temporal stability.
  * *"Which tracked worker is associated with each PPE item, and what is that worker's current PPE compliance state?"*
  * *"Worker #17 is NON_COMPLIANT: missing Gloves and Goggles."*

### PPE Verification Command
```bash
python scripts/test_ppe_detector.py data/samples/ppe_sample.jpg --conf 0.25
```

---

## 11. Step 5B — Person-PPE Spatial Association & Compliance Engine

### Purpose & Architecture
Step 5B establishes the first intelligence/scene-understanding layer of IntelliWatch. It bridges perception outputs from ByteTrack (tracked persons) and the SafetyVision PPE detector into worker-level safety inventories and explainable safety violation events.

```text
General Detection (YOLO11n)
      ↓
ByteTrack MOT
      ↓
Tracked Persons ───────────────┐
                               ▼
PPE Detections ─────→ Spatial Person-PPE Association
(SafetyVision YOLO)            ↓
                     Worker PPE Inventory
                               ↓
                     Deterministic Compliance Engine
                               ↓
                     Temporal Confirmation (N Frames)
                               ↓
                     Explainable IndustrialEvent (PPE_VIOLATION)
```

### 1. Spatial Association Method
Spatial association maps PPE detections onto candidate tracked workers using transparent, explainable geometric evidence rather than unexplainable black-box models:
1. **Bounding-Box Containment:** Evaluates the fraction of the PPE bounding box contained within the worker's bounding box.
2. **Relative Anatomical Sub-regions:** Enforces anatomical feasibility:
   - **Hardhat:** Upper head region ($y_{\text{rel}} \in [-0.25, 0.40]$, ideal $0.08$).
   - **Goggles:** Face region ($y_{\text{rel}} \in [-0.05, 0.35]$, ideal $0.18$).
   - **Safety Vest:** Torso region ($y_{\text{rel}} \in [0.10, 0.75]$, ideal $0.40$).
   - **Gloves:** Hands/arms region ($y_{\text{rel}} \in [0.30, 1.10]$, ideal $0.70$).
3. **Horizontal Span Bounds:** Rejects detections located outside the worker's horizontal bounds ($\pm 20\%$ margin).
4. **Competitive Disambiguation:** When an item is near multiple workers, it assigns it to the worker with the highest spatial score.
5. **False-Association Guard:** If no candidate person satisfies the spatial threshold (`PPE_ASSOCIATION_IOU_THRESHOLD = 0.10`), the item is strictly marked as an `unassociated PPE detection`.

### 2. Worker PPE Inventory Schema
Each tracked worker receives a structured, extensible inventory:
- `track_id`: Persistent track ID from ByteTrack.
- `timestamp`: Current video timestamp.
- `items`: List of `AssociatedPPEItem` records with bounding boxes, scores, and target body regions.
- `ppe_status`: Dict mapping canonical PPE categories to `PRESENT`, `MISSING`, or `UNKNOWN`.
- `compliance_status`: Overall status (`COMPLIANT`, `NON_COMPLIANT`, `UNKNOWN`).
- `missing_ppe` / `present_ppe` / `unknown_ppe`: Explicit lists for downstream explainability.

### 3. Configurable Requirements
Requirements are configuration-driven via `configs/settings.py`:
```python
PPE_REQUIRED_CLASSES = ["Hardhat", "Safety Vest", "Gloves", "Goggles"]
PPE_VIOLATION_CONFIRMATION_FRAMES = 3
PPE_ASSOCIATION_IOU_THRESHOLD = 0.10
PPE_CENTER_CONTAINMENT_THRESHOLD = 0.50
```

### 4. UNKNOWN vs. NON_COMPLIANT Distinction
IntelliWatch avoids false violations by distinguishing between verified absence and unobservable conditions:
- **PRESENT:** Associated positive detection confirmed.
- **MISSING:** Positive gear not observed, or explicit negative detection (e.g. `NO-Hardhat`) confirmed.
- **UNKNOWN:** Visual evidence is insufficient:
  - **Boundary Truncation:** Worker head clipped at top frame edge $\rightarrow$ Hardhat/Goggles `UNKNOWN`.
  - **Lower Body Truncation:** Hands clipped at bottom or side edges $\rightarrow$ Gloves `UNKNOWN`.
  - **Low Resolution / Distance:** Person height $< 120\text{px}$ $\rightarrow$ Goggles `UNKNOWN`; $< 100\text{px}$ $\rightarrow$ Gloves `UNKNOWN`.

**Deterministic Compliance Logic:**
- If any required PPE is `MISSING` $\rightarrow$ `NON_COMPLIANT`
- Else if any required PPE is `UNKNOWN` $\rightarrow$ `UNKNOWN`
- Else (all required PPE are `PRESENT`) $\rightarrow$ `COMPLIANT`

### 5. Temporal Stability & Violation Confirmation
To prevent alert fatigue from single-frame detector dropouts, a violation must persist for `PPE_VIOLATION_CONFIRMATION_FRAMES` (default: 3 frames).
- Frame 101 (Gloves missing) $\rightarrow$ No alert.
- Frame 102 (Gloves detected) $\rightarrow$ Sequence reset.
- Frame 103 (Gloves missing) $\rightarrow$ No alert.
- Consecutive Frame 101, 102, 103 missing $\rightarrow$ Confirmed `IndustrialEvent`.
- Memory is automatically purged when track IDs leave the scene.

### 6. Explainable Industrial Safety Event
Confirmed violations generate a structured `IndustrialEvent`:
```json
{
  "event_id": "evt_ppe_2_3099",
  "event_type": "ppe_violation",
  "timestamp": 3.099,
  "camera_id": "cam_facility_01",
  "tracked_object_ids": [2],
  "severity": "high",
  "explanation": "Worker #2 PPE violation confirmed: missing required gear [Gloves, Goggles] persisting for 3 consecutive frames.",
  "metadata": {
    "track_id": 2,
    "status": "NON_COMPLIANT",
    "missing_ppe": ["Gloves", "Goggles"],
    "detected_ppe": ["Hardhat", "Safety Vest"],
    "unknown_ppe": [],
    "confirmation_frames": 3,
    "consecutive_violation_count": 3
  }
}
```

### Verification & Demonstration Commands
Run synthetic verification:
```bash
python scripts/test_ppe_compliance.py --synthetic
```
Run the complete automated test suite (**72 tests passing**):
```bash
pytest
```

### Known Limitations
- Heavy multi-person occlusions (workers standing directly in front of each other in 2D perspective) can cause ambiguous association.
- Extremely dark CCTV footage may trigger `UNKNOWN` states due to low detector confidence.

---

## 12. Step 6: Restricted Zone Spatial Understanding (Geofencing & Dwell Analytics)

### Conceptual Role & Distinction
While **Step 5B** answers:
> *"Which PPE items are associated with this worker, and are they compliant with required safety gear?"*

**Step 6** answers the fundamental spatial intelligence question:
> *"Where is this tracked worker relative to defined restricted hazard areas, and how long have they remained there?"*

```text
Video Frame
     ↓
Frame Preprocessing
     ↓
Object Detection (YOLO11n)
     ↓
ByteTrack Multi-Object Tracking
     ↓
Tracked Objects / Workers
     ├─────────────────────────────────────────┐
     ↓                                         ↓
Step 5B: PPE Engine                       Step 6: Zone Engine
- Spatial Association                     - Foot-point contact localization
- Inventory Compliance                    - Point-in-polygon reasoning
- Temporal Violation Confirmation         - Multi-zone membership
     ↓                                         ↓
PPE Violation Events                      Temporal Confirmation (e.g. 3 frames)
                                               ↓
                                          Zone Entry / Exit Detection
                                               ↓
                                          Real-time Dwell Time Tracking
                                               ↓
                                          Dwell Threshold Exceeded Alert
```

### 1. What Restricted Zones Are
Restricted zones are user-defined hazardous or sensitive 2D geofences established within the camera field of view to protect workers and maintain operational security (e.g., High Voltage switchgear cells, automated robotic arm operating envelopes, crane swing radii, active forklift transit aisles).

### 2. Why Arbitrary Polygons Are Used
Industrial facilities rarely align neatly with horizontal axis-aligned bounding rectangles. Perspectives, angled walkways, safety boundaries, and machine enclosures necessitate **arbitrary 2D polygons** (convex and concave). Restricting geofences to rectangles creates severe blind spots and rampant false alarms. IntelliWatch uses OpenCV's deterministic `cv2.pointPolygonTest` for exact geometric testing.

### 3. Foot-Point Ground Contact Reasoning
A person's bounding-box center `((x1 + x2) / 2, (y1 + y2) / 2)` remains elevated around their waist or chest. If a worker steps into a ground-level hazard or crosses a floor boundary line, their center might still be outside while their body and feet have already entered the danger zone.

IntelliWatch computes a representative ground contact point:
- **For tracked persons/workers:**
  $$\text{foot\_point} = \left(\frac{x_1 + x_2}{2},\; y_2\right)$$
  This accurately reflects where the worker's feet meet the physical floor plane.
- **For non-person entities (machinery, vehicles, general objects):**
  $$\text{centroid} = \left(\frac{x_1 + x_2}{2},\; \frac{y_1 + y_2}{2}\right)$$
  This modular dispatcher (`compute_contact_point`) cleanly supports future object classes without person-specific assumptions.

### 4. Zone Membership & Boundary Handling
For every tracked object and every enabled zone, spatial membership is evaluated:
- **`INSIDE`:** Contact point is strictly within the polygon boundary.
- **`BOUNDARY`:** Contact point lies directly on the boundary edge. Under industrial safety-first principles, boundary contact is treated as an active geofence entry (`is_inside = True`).
- **`OUTSIDE`:** Contact point is outside the polygon.

### 5. Multi-Zone Evaluation
All defined and enabled zones are evaluated independently. A worker may simultaneously be outside all zones, inside a single zone (e.g., `high_voltage`), or inside overlapping zones (e.g., `high_voltage` + `forklift_aisle`). Disabled zones (`enabled: false`) are ignored with zero computational overhead.

### 6. Temporal Confirmation
To prevent false alarms caused by single-frame detector jitter or tracking noise near boundary edges, zone entry requires persistence for a configurable number of consecutive frames (`ZONE_ENTRY_CONFIRMATION_FRAMES`, default: `3`).
- **Frame 1 (Inside):** Counter = 1 $\rightarrow$ No alert.
- **Frame 2 (Inside):** Counter = 2 $\rightarrow$ No alert.
- **Frame 3 (Inside):** Counter = 3 $\rightarrow$ **CONFIRMED ENTRY** (`ZONE_ENTRY`).
- If an actor enters for only 1 or 2 frames and leaves, no false intrusion alert is emitted.

### 7. Dwell Time Analytics & Long-Dwell Alerting
Once an actor's entry is confirmed, IntelliWatch tracks cumulative dwell duration:
$$\text{dwell\_seconds} = \text{current\_timestamp} - \text{entry\_timestamp}$$

- **Threshold Monitoring (`ZONE_MAX_DWELL_SECONDS`):** If a worker remains inside longer than the allowable limit (configured globally or per-zone, e.g. 10.0s), a `ZONE_DWELL_EXCEEDED` incident is generated.
- **Single-Trigger Policy:** The dwell exceeded alert is emitted **once per zone-entry episode** to prevent alert spam across subsequent frames.
- **Exit Event (`ZONE_EXIT`):** When the actor departs the zone, an exit event is emitted detailing the total dwell duration. The dwell timer then completely resets.
- **Departed Track Cleanup:** If a track ID disappears from the scene while inside a zone, the engine logs a departure exit event and purges internal tracking memory to prevent resource leaks.

### 8. Zone Configuration (`configs/zones.json`)
Zones are configuration-driven and loaded at runtime without modifying Python code:
```json
[
  {
    "zone_id": "high_voltage_01",
    "name": "High Voltage Switchgear Area",
    "zone_type": "restricted",
    "polygon": [
      [100.0, 100.0],
      [450.0, 100.0],
      [500.0, 300.0],
      [420.0, 450.0],
      [120.0, 420.0]
    ],
    "enabled": true,
    "max_dwell_seconds": 10.0
  }
]
```

### 9. Explainable Structured Events
Zone events strictly adhere to the project's Pydantic `IndustrialEvent` schema:

**Zone Entry Incident:**
```json
{
  "event_id": "evt_zone_entry_17_high_voltage_12430",
  "event_type": "zone_entry",
  "timestamp": 12.43,
  "camera_id": "cam_01",
  "tracked_object_ids": [17],
  "severity": "high",
  "explanation": "Worker #17 confirmed entry into restricted zone 'High Voltage Area' (high_voltage) after 3 consecutive frames.",
  "metadata": {
    "track_id": 17,
    "zone_id": "high_voltage",
    "zone_name": "High Voltage Area",
    "zone_type": "restricted",
    "entry_timestamp": 12.43,
    "confirmation_frames": 3,
    "contact_point": [320.0, 415.0]
  }
}
```

**Zone Dwell Exceeded Incident:**
```json
{
  "event_id": "evt_zone_dwell_17_high_voltage_22510",
  "event_type": "zone_dwell_exceeded",
  "timestamp": 22.51,
  "camera_id": "cam_01",
  "tracked_object_ids": [17],
  "severity": "critical",
  "explanation": "Worker #17 exceeded maximum dwell limit in restricted zone 'High Voltage Area' (high_voltage): stayed 10.1s (threshold: 10.0s).",
  "metadata": {
    "track_id": 17,
    "zone_id": "high_voltage",
    "zone_name": "High Voltage Area",
    "zone_type": "restricted",
    "entry_timestamp": 12.43,
    "dwell_seconds": 10.08,
    "max_dwell_seconds": 10.0
  }
}
```

**Zone Exit Incident:**
```json
{
  "event_id": "evt_zone_exit_17_high_voltage_25000",
  "event_type": "zone_exit",
  "timestamp": 25.0,
  "camera_id": "cam_01",
  "tracked_object_ids": [17],
  "severity": "low",
  "explanation": "Worker #17 exited restricted zone 'High Voltage Area' (high_voltage) after total dwell time of 12.6s.",
  "metadata": {
    "track_id": 17,
    "zone_id": "high_voltage",
    "zone_name": "High Voltage Area",
    "zone_type": "restricted",
    "entry_timestamp": 12.43,
    "exit_timestamp": 25.0,
    "dwell_seconds": 12.57
  }
}
```

### 10. Visualization Overlay
The [`TrackingVisualizer`](vision/tracking/visualizer.py) renders:
- Semi-transparent zone overlay polygons with crisp perimeter lines.
- Color distinction: **Red** when active intrusions are present, **Amber** when clear.
- Header badges identifying zone names and live intrusion counts.
- Worker badge annotations indicating current restricted zone and active dwell duration (e.g. `ZONE: High Voltage Area (4.5s)`).
- Distinct ground contact point dot at the worker's foot contact location.

### Verification & Demonstration Commands
Run synthetic zone scenario integration script:
```bash
python scripts/test_zone_engine.py
```
Run the complete automated test suite (**93 tests passing**):
```bash
pytest
```

### Known Limitations
- Pure 2D perspective projection: ground contact localization assumes camera pitch enables floor visibility; extreme camera angles or occlusion of worker feet can impact contact point precision.
- Stationary objects obstructing foot contact: handled gracefully, but ground contact calculation currently assumes standard standing orientation.

---

## 13. Step 7: Monocular Depth & Relative 3D Spatial Understanding

### Conceptual Role & Distinction
While **Step 6** evaluates 2D ground-plane polygon geofences:
> *"Is this worker inside this floor boundary?"*

**Step 7** introduces relative camera-to-subject proximity reasoning:
> *"How far or close is this worker or machine relative to the camera and to other scene entities?"*

```text
Video Frame
     ↓
Frame Preprocessing
     ↓
Object Detection (YOLO11n)
     ↓
ByteTrack Multi-Object Tracking
     ↓
Tracked Objects / Workers
     ├──────────────────────────┬──────────────────────────┐
     ↓                          ↓                          ↓
Step 5B: PPE Engine        Step 6: Zone Engine        Step 7: Depth Pipeline
- Association              - Contact point            - Monocular depth map
- Compliance               - Polygon testing          - Native coordinate alignment
- PPE Events               - Dwell analytics          - Foot-point depth sampling
                                                      - Bounding-box depth stats
                                                           ↓
                                                      Relative Proximity & Ordering
```

### 1. Monocular Relative Depth Estimation
Industrial CCTV deployments rarely feature expensive stereo rigs, calibrated multicamera arrays, or LiDAR sensors. Step 7 provides relative depth estimation from monocular 2D video feeds using Depth-Anything-V2-Small.

- **Relative Values:** Model outputs normalized relative disparity/depth values in $[0.0, 1.0]$.
- **Mock/Fallback Mode:** For CI/testing environments without external model weight downloads, `MockDepthEstimator` provides deterministic planar or gradient depth maps with zero GPU requirements.

### 2. Native Resolution Coordinate Alignment
Depth estimation models often process inputs at specific square resolutions (e.g. $518 \times 518$). To accurately map depth values back to high-resolution CCTV coordinates:
- The depth map is bilinearly sampled or resized to match the original video resolution.
- Object coordinates from YOLO11n/ByteTrack map directly into the depth buffer without aspect distortion or misregistration.

### 3. Contact-Point & Object Depth Aggregation
For each tracked actor and entity:
- **Contact-Point Depth:** Depth is sampled directly at the ground contact point (`foot_point` for persons, centroid for machinery). This represents the entity's physical ground position rather than floating upper-body depth.
- **Bounding-Box Distribution:** Depth distribution statistics (`min`, `max`, `mean`, `median`) are aggregated over the bounding box mask to evaluate object volume and extent.

### 4. Relative Depth Ordering & Trends
- **Proximity Ranking:** Ranks all active tracks from nearest to farthest relative to the camera lens.
- **Temporal Depth Trend:** Compares multi-frame depth observations to categorize motion as `approaching`, `receding`, or `stable`.

### Verification Commands
Run synthetic depth integration script:
```bash
python scripts/test_depth_estimator.py
```
Run depth unit tests:
```bash
pytest tests/test_depth.py
```

### Known Limitations
- **Uncalibrated Relative Depth:** Outputs represent relative proximity rather than metric meters ($m$). Distance conversions require camera calibration intrinsics and extrinsics.
- **Reflective & Low-Texture Surfaces:** Monocular depth models may exhibit noise on high-gloss factory floors or stark featureless walls.

---

## 14. Step 8: Behavior & Temporal Analysis

### Conceptual Role & Distinction
While **Steps 3–7** establish spatial perception at single frame snapshots:
> *"Where is each worker, what PPE do they have, which zone are they in, and how close are they?"*

**Step 8** introduces continuous temporal reasoning:
> *"What is this worker doing over time? Are they walking, running, loitering, making sudden erratic movements, or exhibiting fall-like motion patterns?"*

```text
Tracked Objects (Step 4) + Zone Occupancy (Step 6) + Relative Depth (Step 7)
                                ↓
                 TrackStateRegistry & History Buffer
                                ↓
               Kinematic Motion Analysis (motion.py)
        - Image velocity (vx, vy, speed px/s)
        - Image acceleration (ax, ay, accel px/s²)
        - Direction change & trajectory heading
        - Aspect-ratio geometry change
                                ↓
             Per-Track Behavioral State Machine (FSM)
        - UNKNOWN → STATIONARY / MOVING / RAPID_MOVEMENT
        - Loitering detection → PROLONGED_STATIONARY
        - Fall heuristic → POSSIBLE_FALL
        - Acceleration spike → SUDDEN_MOVEMENT
                                ↓
         Temporal Confirmation & Anti-Flapping Hysteresis
         (Requires N consecutive frames, e.g. 3 frames)
                                ↓
                   Structured Behavior Events
         (Transitions, Prolonged Stationarity, Sudden Accel, Fall Heuristic)
```

### 1. Primary Behavioral States
The `BehaviorEngine` classifies each active track into an explainable primary behavior:
- **`UNKNOWN`:** Initial state during the first few frames before sufficient motion history is established.
- **`STATIONARY`:** Track displacement is under the stationary threshold (`BEHAVIOR_STATIONARY_DISTANCE_THRESHOLD`, default $15.0\text{ px}$) for at least `BEHAVIOR_STATIONARY_SECONDS` ($0.3\text{s}$).
- **`MOVING`:** Track displays consistent translation across frames within normal walking speeds.
- **`RAPID_MOVEMENT`:** Rolling average image velocity meets or exceeds `BEHAVIOR_RUNNING_VELOCITY_THRESHOLD` ($250.0\text{ px/s}$). Useful for detecting sprinting or equipment runaway.
- **`PROLONGED_STATIONARY`:** Track remains stationary for longer than `BEHAVIOR_LOITERING_SECONDS` ($10.0\text{s}$ default). Immediately escalates to alert operators of potential collapse, entrapment, or unauthorized loitering.
- **`POSSIBLE_FALL`:** Geometric heuristic triggered when a track undergoes a sudden vertical aspect-ratio collapse ($>0.3$ reduction) accompanied by downward/lateral centroid displacement.
- **`SUDDEN_MOVEMENT`:** Single-frame acceleration exceeding `BEHAVIOR_SUDDEN_ACCELERATION_THRESHOLD` ($600.0\text{ px/s}^2$). Identifies sudden jolts, near-miss collisions, or abrupt starts/stops.

### 2. Multi-Frame Confirmation & Hysteresis
To prevent noisy detector bounding-box jitter from triggering false state changes:
- Candidate behaviors require confirmation across a configurable window (`BEHAVIOR_CONFIRMATION_FRAMES = 3`).
- **Anti-Flapping Logic:** Once escalated to `PROLONGED_STATIONARY`, the state machine does not oscillate or demote back to `STATIONARY` while stationarity persists. Demotion only occurs upon confirmed physical movement.

### 3. Contextual Fusion (Zones & Depth)
The behavior engine decorates behavioral states with operational context:
- **Zone Fusion:** Attaches active zone IDs, names, and dwell times from Step 6 (`zone_id`, `zone_name`, `zone_dwell_s`).
- **Depth Trend Fusion:** Integrates relative depth changes from Step 7 (`approaching`, `receding`, `stable`).

### 4. Explainable Structured Events
Behavior events follow the Pydantic `BehaviorEvent` schema:

```json
{
  "event_id": "9d8e7f6a-5b4c-3d2e-1f0a-9b8c7d6e5f4a",
  "event_type": "prolonged_stationary",
  "timestamp": 24.5,
  "frame_id": 245,
  "track_id": 3,
  "camera_id": "cam_01",
  "primary_behavior": "PROLONGED_STATIONARY",
  "severity": "medium",
  "explanation": "Track 3 has remained nearly stationary for 10.2 seconds, exceeding the configured loitering threshold of 10.0s.",
  "zone_id": "machinery_operating_envelope",
  "image_speed_px_per_s": 0.0,
  "stationary_duration_s": 10.2
}
```

### 5. Visualization Overlay
The [`TrackingVisualizer`](vision/tracking/visualizer.py) renders real-time behavioral annotations:
- Status chip above worker bounding boxes displaying current primary behavior.
- Instantaneous image-space speed (e.g. `45.2 px/s`).
- Dwell time counter for stationary workers (`Stat: 4.8s`).
- Warning badges for active flags (`[RAPID]`, `[SUDDEN]`, `[DIR_CHG]`, `[FALL]`).

### Verification & Demonstration Commands
Run synthetic behavior scenario integration script:
```bash
python scripts/test_behavior_engine.py
```
Run complete automated test suite (**160 tests passing**):
```bash
pytest
```

### Technical Honesty & Known Limitations
- **Image-Space Kinematics:** Velocities and accelerations are measured in pixel coordinates ($\text{px/s}$, $\text{px/s}^2$). Without camera intrinsics and ground-plane calibration, perspective foreshortening means workers farther from the camera exhibit lower pixel speeds for identical physical walking speeds.
- **Geometric Fall Heuristic:** `POSSIBLE_FALL` is a preliminary geometric indicator (bounding-box aspect-ratio collapse) and does NOT constitute medically or semantically confirmed fall detection. Skeletal pose estimation (planned for future steps) is required for high-confidence fall classification.
- **Occlusions:** Severe track occlusions can introduce instantaneous centroid jumps; temporal confirmation and acceleration limits mitigate false alerts.

---

## 15. Step 10: Safety Risk & Event Reasoning

### Conceptual Role & Architecture
Step 10 represents the deterministic reasoning and decision layer of IntelliWatch. While previous perception layers answer:
> *"What objects exist, where are they moving, what PPE are they wearing, which zone are they in, and how do they relate to each other?"*

**Step 10** answers the operational safety questions:
1. *"What potentially hazardous event is occurring right now?"*
2. *"Which entities are involved?"*
3. *"What verifiable risk factors triggered this alert?"*
4. *"What is the current composite risk level, and why?"*
5. *"Has the event been sufficiently confirmed across time, or is it a transient false alarm?"*
6. *"Is the event active, resolved, or cooling down?"*

```text
Detection (YOLO11n)
       ↓
Multi-Object Tracking (ByteTrack)
       ↓
PPE Perception (SafetyVision YOLOv8n)
       ↓
Restricted Zones (Polygonal Geofencing)
       ↓
Monocular Depth (Depth-Anything-V2)
       ↓
Behavior & Kinematics (Temporal Motion FSM)
       ↓
Scene Graph (Relational Entity Topology)
       ↓
Atomic Risk Factors (factors.py)
       ↓
Transparent Risk Scoring (scorer.py)
       ↓
Rule & Compound Event Reasoning (rules.py)
       ↓
Temporal Confirmation & Lifecycle State Machine (engine.py)
       ↓
Confirmed Safety Events & FrameRiskAssessment
```

### 1. Risk Factor Taxonomy
Atomic, verifiable risk factors extracted directly from perception and relational outputs:
- **`PPE_NON_COMPLIANCE`:** Tracked worker missing mandatory safety gear (e.g. `Hardhat`, `Safety Vest`).
- **`RESTRICTED_ZONE_INTRUSION`:** Entity foot contact point located inside a restricted or hazardous zone polygon.
- **`RESTRICTED_ZONE_DWELL`:** Entity remaining inside a restricted zone longer than the allowable dwell threshold (`ZONE_MAX_DWELL_SECONDS`).
- **`RAPID_MOVEMENT`:** Image-space velocity exceeding `BEHAVIOR_RUNNING_VELOCITY_THRESHOLD`.
- **`SUDDEN_MOVEMENT`:** Single-observation acceleration exceeding `BEHAVIOR_SUDDEN_ACCELERATION_THRESHOLD`.
- **`PROLONGED_STATIONARY`:** Continuous stationarity exceeding `BEHAVIOR_LOITERING_SECONDS`.
- **`FALL_LIKE_BEHAVIOR`:** Rapid vertical aspect-ratio collapse heuristic (`POSSIBLE_FALL`).
- **`PERSON_VEHICLE_PROXIMITY`:** `NEAR` relationship between a person and a vehicle/machinery node in the scene graph.
- **`APPROACHING_VEHICLE`:** Kinematic `APPROACHING` relationship between a vehicle and a worker.
- **`WORKER_NEAR_MACHINE`:** Spatial proximity between a worker and industrial machinery.
- **`MULTIPLE_RISK_FACTORS`:** Incremental compounding contribution when multiple hazards act simultaneously on the same actor.

### 2. Controlled Event Taxonomy
- `PPE_VIOLATION`: Confirmed worker PPE non-compliance.
- `RESTRICTED_ZONE_INTRUSION`: Confirmed entry into an authorized hazard area.
- `RESTRICTED_ZONE_DWELL`: Prolonged dwell exceeding authorized limit.
- `RAPID_MOVEMENT_EVENT`: Confirmed sprinting or runaway motion in industrial environment.
- `FALL_LIKE_EVENT`: Geometric heuristic flagging possible worker collapse or slip.
- `PERSON_VEHICLE_PROXIMITY`: Worker and machinery in close spatial proximity.
- `APPROACHING_VEHICLE`: Vehicle closing distance toward worker.
- `WORKER_MACHINE_RISK`: Operational proximity to operating machinery.
- `COMPOUND_SAFETY_EVENT`: Escalated multi-factor incident combining two or more distinct concurrent hazards on the same entity.

### 3. Risk Levels & Transparent Scoring
Risk severity is categorized into five controlled tiers based on configurable numeric thresholds:
- `INFO` ($\ge 0.0\text{ pts}$)
- `LOW` ($\ge 10.0\text{ pts}$)
- `MEDIUM` ($\ge 30.0\text{ pts}$)
- `HIGH` ($\ge 60.0\text{ pts}$)
- `CRITICAL` ($\ge 85.0\text{ pts}$)

**Scoring Formula:**
$$\text{Score} = \sum_{f \in \text{factors}} \text{SeverityContribution}(f) + (\max(0, N_{\text{factors}} - 1) \times \text{RISK\_SCORE\_MULTI\_FACTOR\_ESCALATION})$$

Every point addition is explicitly accounted for in `score_breakdown` without opaque magic numbers or unexplainable black-box models.

### 4. Temporal Confirmation & Event Lifecycle
To prevent alert fatigue and eliminate single-frame detector jitter:
- **`CANDIDATE`:** Condition observed, awaiting persistence across `RISK_EVENT_CONFIRMATION_FRAMES` (default: 3 frames).
- **`CONFIRMED`:** Condition sustained for the required window; newly emitted to alert operators.
- **`ACTIVE`:** Confirmed event persisting across continuous subsequent observations.
- **Anti-Flapping:** When condition disappears, it requires `RISK_EVENT_END_CONFIRMATION_FRAMES` (default: 3 frames) of continuous absence before marking as `ENDED`.
- **`ENDED`:** Event concluded with verified `end_timestamp`.
- **`COOLDOWN`:** Post-end hysteresis window (`RISK_EVENT_COOLDOWN_FRAMES`, default: 5 frames) preventing rapid fluttering.
- **Track Departure:** When a tracked actor disappears from the CCTV field of view, their active events are immediately finalized to `ENDED` without ghost persistence.

### 5. Compound Hazard Reasoning
When a worker exhibits multiple concurrent independent risks (e.g. worker inside high voltage zone + missing hardhat + forklift approaching), IntelliWatch generates an escalated `COMPOUND_SAFETY_EVENT` linking all participating entities. Each contributing factor and piece of evidence is preserved separately without double-counting.

### 6. REST API Endpoints
- `GET /api/v1/risk/current`: Returns the latest structured `FrameRiskAssessment`, active events, atomic risk factors, and `RiskSummary`.
- `GET /api/v1/events/current`: Returns the list of active confirmed safety events currently in progress.
- `GET /api/v1/config`: Exposes active risk thresholds and confirmation configurations.

### 7. Performance Benchmark & Verification
- **CPU Latency:** Evaluated in `scripts/test_risk_engine.py` across multi-frame compound scenarios; average execution time is **~1.4 ms per frame** on CPU.
- **Unit & Integration Suite:** **212 tests passing** deterministically with 0 failures (`pytest`).

### 8. Technical Honesty & Known Limitations
---

## 14. Step 12: Final Dashboard, Evidence System & End-to-End Integration

### Architecture Overview

IntelliWatch integrates all 11 previous engineering phases into a unified, demonstrable, and explainable industrial safety monitoring platform:

```text
                   ┌─────────────────┐
                   │  CCTV / VIDEO   │
                   └────────┬────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Frame Preprocessing │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Object Detection    │
                 └──────────┬──────────┘
                            ↓
                 ┌─────────────────────┐
                 │ Multi-Object Track  │
                 └──────────┬──────────┘
                            ↓
          ┌─────────────────┼─────────────────┐
          ↓                 ↓                 ↓
       PPE                 Zones             Depth
          └─────────────────┼─────────────────┘
                            ↓
                  Behavior / Temporal
                            ↓
                      Scene Graph
                            ↓
                   Risk & Events
                            ↓
                Prediction / Anomaly
                            ↓
                   Early Warnings
                            ↓
                  Evidence / Incident
                            ↓
                       Dashboard
```

### 1. End-to-End Orchestrator (`intelligence/pipeline/orchestrator.py`)
Connects all perception and intelligence modules in sequential order:
1. **Frame Ingestion:** Ingests raw video frame, letterboxing to uniform coordinates.
2. **Object Detection:** Detects workers, vehicles, and machinery using YOLO11n.
3. **Multi-Object Tracking:** Assigns and maintains persistent track IDs via ByteTrack (`FrameTracks`).
4. **PPE Compliance:** Runs domain-specific PPE detection, anatomical spatial association, and compliance evaluation (`PPEComplianceEngine`).
5. **Restricted Zone Spatial Reasoning:** Evaluates foot-ground contact points against polygon safety zones using point-in-polygon tests (`ZoneEngine`).
6. **Relative Monocular Depth:** Conservative relative disparity depth estimation (`DepthAnythingEstimator`).
7. **Temporal Behavior Analysis:** Tracks motion kinematics (velocity, acceleration, direction changes, loitering, fall heuristics) (`BehaviorEngine`).
8. **Relational Scene Graph:** Builds dynamic spatial and kinematic graph edges (`INSIDE`, `WEARING`, `NEAR`, `APPROACHING`, `CLOSER_THAN`) with confirmation lifecycles (`SceneGraphBuilder`).
9. **Risk & Event Reasoning:** Deterministic multi-factor safety scoring, compound hazard escalation, and anti-flapping event lifecycles (`RiskEngine`).
10. **Predictive & Early-Warning Intelligence:** Short-horizon trajectory projection, hazard heading intersection, risk escalation trends, and anomaly indicators (`PredictionEngine`).
11. **Incident Evidence Logger:** Automatically captures annotated visual snapshots and structured JSON records in `data/output/incidents/`.
12. **Telemetry & Assessment Assembly:** Emits comprehensive `FrameAssessment` schema updating in-memory caches for real-time operator endpoints.

### 2. Incident Evidence System (`data/output/incidents/`)
When high-risk or confirmed safety events occur, IntelliWatch creates a deterministic, tamper-evident record:
- **`incident_id`:** Unique identifier (e.g. `inc_01a87af1`).
- **`timestamp` & `frame_id`:** Precise stream temporal coordinate.
- **`event_type` & `risk_level`:** Classified event tier (`CRITICAL`, `HIGH`, `MEDIUM`, `LOW`, `INFO`).
- **`involved_track_ids` & `involved_entity_ids`:** Active ByteTrack IDs participating in the event.
- **`risk_factors`:** Verifiable factual factors without black-box scores.
- **`explanation`:** Deterministic, human-readable justification of WHY the alert fired.
- **Visual Evidence Snapshot:** Associated native frame saved deterministically to disk (`{incident_id}.jpg`).

### 3. Professional Operator Dashboard (`frontend/`)
A state-of-the-art dark-mode glassmorphic control room console built with zero external Node/npm dependencies (pure semantic HTML5, Vanilla CSS3, and JavaScript):
- **Live Viewport:** Canvas rendering bounding boxes, track IDs, motion trajectories, PPE compliance badges, restricted zones, and connecting relational links.
- **Scene Understanding Panel:** Factual scene graph relationships (`Worker #1 ➔ inside ➔ High Voltage Zone`, `Forklift #2 ➔ approaching ➔ Worker #1`) with entity metrics.
- **Risk Reasoning & Explainability:** Real-time risk level badge, transparent score, and contributing risk factors.
- **Worker PPE Monitoring:** Worker-by-worker gear checklist (`Hardhat`, `Safety Vest`, `Gloves`, `Goggles`) distinguishing `COMPLIANT`, `NON_COMPLIANT`, and `UNKNOWN`.
- **Temporal Behavior & Relative Depth:** Kinematic velocity, primary behavior classifications, and explicit disclosures regarding non-metric relative depth.
- **Predictive Early Warnings:** Visual indicators for projected trajectory intersections and escalating risk trends.
- **Incident Evidence Timeline:** Chronological incident log with an interactive Evidence Inspector modal showing the snapshot image and structured evidence factors.

### 4. Demo Runner (`scripts/run_demo.py`)
Deterministic end-to-end command-line runner supporting local video files, configurable sampling, and CPU-compatible execution:
```bash
# Process default CCTV sample video
python scripts/run_demo.py

# Process custom video with 60 max frames and CPU device
python scripts/run_demo.py --input data/samples/cctv_worker_moving.mp4 --max-frames 60 --device cpu

# Process synthetic benchmark video with custom output locations
python scripts/run_demo.py --input data/samples/synthetic_test.mp4 --output-video data/output/demo_annotated.mp4 --output-json data/output/demo_results.json
```

python scripts/run_demo.py --input data/samples/synthetic_test.mp4 --output-video data/output/demo_annotated.mp4 --output-json data/output/demo_results.json
```

### 5. Step 12 API Endpoints
- `GET /` & `GET /dashboard`: Serves the operator dashboard console.
- `GET /api/v1/assessment/current`: Returns the comprehensive `FrameAssessment` for the latest processed frame.
- `GET /api/v1/incidents`: Returns incident evidence history with filtering (`status`, `risk_level`, `limit`).
- `GET /api/v1/incidents/{incident_id}`: Returns specific incident details.
- `GET /api/v1/incidents/{incident_id}/evidence`: Returns the JPEG image snapshot for the confirmed incident.
- `GET /api/v1/status`: Returns operational telemetry (CPU execution mode, measured FPS, latency, model status).
- `POST /api/v1/analyze/image`: Directly analyzes uploaded image (JPG/PNG), stores annotated result, updates assessment.
- `POST /api/v1/analyze/video`: Enqueues asynchronous video processing job (MP4/AVI/MOV/MKV), returns `job_id`.
- `GET /api/v1/analyze/status/{job_id}`: Returns live video progress telemetry (progress %, frames, FPS, current risk, tracks, incidents).
- `GET /api/v1/analyze/result/{job_id}`: Returns final execution summary metrics upon job completion.
- `GET /api/v1/media/{media_id}`: Safely serves uploaded and annotated media assets with strict path traversal prevention.

### 6. CPU Performance Benchmark
Measured on an Intel CPU (without CUDA/GPU acceleration) running the full end-to-end pipeline:
- **Average Frame Latency:** **~100 - 147 ms** (with full YOLO11n detection, ByteTrack tracking, PPE association, zone testing, behavior analysis, scene graph construction, risk reasoning, prediction evaluation, and video annotation).
- **Processing Throughput:** **~6.8 - 10.0 FPS** on CPU.
- **Intelligence Layer Overhead:** **~6.2 ms** (pure scene graph + behavior + risk + prediction reasoning without neural inference).

### 7. Technical Honesty & System Disclosures
- **Perception vs Reasoning:** Object detection (YOLO11n) and PPE detection (SafetyVision YOLOv8n) are trained neural networks. Depth perception is uncalibrated monocular relative disparity. Behavior analysis, zone intrusion, scene graph construction, risk scoring, and trajectory projections are **deterministic geometric and temporal reasoning algorithms**, NOT black-box heuristics or unvalidated neural networks.
- **Single-Frame Image Analysis Limitation:** Single images contain only one static frame. Therefore, motion history, velocity, acceleration, trajectory trails, and predictive early-warning escalations are **NOT** available and are **NEVER fabricated**. The UI and schemas explicitly label these fields with *"Not available for single-frame analysis"*.
- **Non-Metric Depth:** Relative depth is explicitly disclosed in the UI and documentation as inverse relative disparity; metric distances in meters ($m$) are NOT claimed without physical calibration.
- **Trajectory Projection:** Projected trajectories are short-horizon image-space linear kinematic approximations ($15\text{ frames} \approx 0.5\text{s}$); they do NOT claim confirmed accident prediction or long-range physical dynamics modeling.
- **Incident Explainability:** Every incident explanation is generated from verifiable geometric and compliance facts; no confidence values or probabilities are fabricated.

---

## 15. How to Run the Complete System

### 1. Start the API Server & Operator Dashboard
```bash
python -m scripts.run_server
```
- **Operator Dashboard:** Open [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) in any modern browser.
- **Interactive OpenAPI Documentation:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Health Check:** [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

### 2. Execute the End-to-End Demo Runner (Optional CLI)
```bash
python scripts/run_demo.py --max-frames 60
```

### 3. Run the Comprehensive Test Suite
```bash
python -m pytest
```
**Current Test Baseline:** **385 passing tests, 0 failures, 0 errors**.

---

## 16. Dashboard Media Analysis

The IntelliWatch dashboard is fully interactive. Operators can directly upload media and inspect scene understanding outputs without using the command line:

```text
Dashboard
    ↓
Upload Media (Browse or Drag & Drop)
    ↓
Select Image or Video
    ↓
Analyze
    ↓
Live Processing Telemetry (FPS, Frame Count, Risk, Tracks)
    ↓
Results & Interactive Operator Dashboard
```

### Workflow Instructions
1. **Start the server:**
   ```bash
   python -m scripts.run_server
   ```
2. **Open the Operator Dashboard:**
   Navigate to [http://127.0.0.1:8000/dashboard](http://127.0.0.1:8000/dashboard) in your browser.
3. **Upload Media:**
   - Choose **Upload Image** or **Upload Video** mode.
   - Drag and drop a file or click **Choose File**.
   - Review selected file details (filename, file type, file size).
4. **Click "ANALYZE MEDIA":**
   - **For Images:** Processed immediately through the complete pipeline. The annotated image is displayed in the viewport with detections, PPE status, zones, depth, scene relationships, and risk factors. Single-frame notices are honestly displayed for temporal panels.
   - **For Videos:** Enqueued as a background task. The dashboard polls live progress (~1s interval) showing current frame count, actual measured CPU FPS, active tracks, and current risk tier. The latest processed frame updates in real time. Upon completion, the annotated video is playable directly in the viewport alongside comprehensive processing metrics.
5. **Inspect Incidents & Evidence:**
   - Click **View Incidents ↓** or scroll to the **Incident Evidence Timeline**.
   - Click **Evidence** on any incident to inspect the metadata and visual JPEG evidence snapshot stored deterministically in `data/output/incidents/`.
6. **New Analysis:**
   - Click **↺ New Analysis** to reset the viewport and upload controls for a new media file while preserving historical incident evidence.

### Supported Formats & File Limits
- **Images:** `.jpg`, `.jpeg`, `.png` (Max size: 25 MB).
- **Videos:** `.mp4`, `.avi`, `.mov`, `.mkv` (Max size: 100 MB).

### Storage Architecture
- `data/input/uploads/`: Stores sanitized uploaded raw images and videos (`<job_id>_input.<ext>`).
- `data/output/uploads/`: Stores annotated output videos (`<job_id>_annotated.mp4`), annotated images (`<job_id>_annotated.jpg`), and live frames (`<job_id>_latest.jpg`).
- `data/output/incidents/`: Permanent local repository for confirmed safety incident JSON metadata and JPEG evidence snapshots.

---

## 17. Interactive Safety Zones & Scene Configuration (Step 16)

Step 16 empowers control room operators to interactively draw, name, categorize, save, edit, and delete polygonal safety zones directly in the dashboard over live telemetry, uploaded images, or video frames without manual JSON editing.

```text
Operator Clicks on Viewport
            ↓
Screen Coordinate (clientX, clientY)
            ↓
Invert Viewport Offsets & Letterbox Padding (padX, padY)
            ↓
Invert Aspect Ratio Scaling (scale = min(dispW/origW, dispH/origH))
            ↓
Original Media Resolution Coordinates [x, y]
            ↓
REST API (POST /api/v1/zones)
            ↓
Atomic Persistence (configs/zones.json)
            ↓
ZoneEngine Contour Compilation & Risk Engine Evaluation
```

### Supported Zone Types
- **Restricted Zone (`restricted`):** Critical exclusion area (red). Zero-tolerance or low-dwell threshold for unauthorized personnel.
- **Hazard Zone (`hazard`):** General hazard area (amber), e.g., elevated machinery or high-temperature processes.
- **Machine Zone (`machine`):** Heavy equipment working perimeter (purple), e.g., robotic cells, CNC mills, conveyors.
- **Safety Zone (`safety`):** Designated safe pedestrian walkway or egress route (emerald).
- **Monitored Zone (`monitored`):** Monitored operational sector (sky blue), e.g., active forklift lanes or loading bays.
- **Safe Zone (`safe`):** Assembly or staging area (green).

### Bidirectional Coordinate Transformation System
The system decouples browser rendering from analytical coordinate spaces via [`intelligence/zones/coordinates.py`](intelligence/zones/coordinates.py):
- **Display to Original:** `display_to_original()` removes CSS viewport padding and letterbox offsets, inverting the uniform downscaling to map clicks back to the source frame resolution.
- **Original to Display:** `original_to_display()` projects stored polygon coordinates onto the rendered canvas overlay for real-time visual feedback and rubberband guide lines during drawing.
- **Cross-Resolution Scaling:** `scale_polygon_between_resolutions()` adapts polygons when switching media streams of differing resolutions.

### REST API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/zones` | List all configured safety zones (with optional `?enabled_only=true`) |
| `GET` | `/api/v1/zones/{zone_id}` | Retrieve details and polygon vertices of a specific zone |
| `POST` | `/api/v1/zones` | Create a new safety zone (status 201 Created) |
| `PUT` | `/api/v1/zones/{zone_id}` | Update existing zone name, type, polygon vertices, or dwell limits |
| `DELETE` | `/api/v1/zones/{zone_id}` | Delete a safety zone and purge active tracking state |
| `POST` | `/api/v1/zones/reload` | Force reload configurations from `configs/zones.json` |

### Technical Limitations & Honest Disclosures
- **2D Image-Space Polygons:** Zones are defined in 2D image coordinates and evaluate worker foot-points via Ray-Casting point-in-polygon algorithms. They do not represent metric 3D volumes unless cameras are calibrated with intrinsic/extrinsic homography matrices.
- **Camera Movement:** Static camera perspectives are assumed. PTZ (pan-tilt-zoom) or vibrating cameras require homography re-alignment or dynamic zone stabilization.

---

## 18. Explainability, Evidence & Safety Analytics (Step 17)

Step 17 elevates IntelliWatch from an alerting system to an auditable industrial intelligence platform. Control room operators and EHS investigators can answer the fundamental investigative questions: *What happened? Who was involved? Where? When? Why? What evidence supports it? How did risk evolve over time? What happened before and after?*

```text
Confirmed Incident Record
           ↓
Deterministic Explanation Service
           ├─→ 5W Structured Audit (Who, What, Where, When, Why)
           ├─→ Transparent Risk Score Breakdown (Base Severity + Multi-Hazard Escalation)
           ├─→ Temporal Timeline Reconstruction (Pre-Incident, Peak Event, Post-Incident)
           └─→ Risk Trajectory Coordinates (T-minus to T-plus Trend Profile)
           ↓
Interactive Operator Console & Auditable JSON Evidence Package
```

### Deterministic Five-W (5W) Causation Engine
- **No Generative AI / LLM Hallucinations:** All explanations are deterministically compiled directly from structured vision and risk states (`RiskFactor`, `SceneRelationship`, `EarlyWarningIndicator`, `BehaviorState`).
- **Who:** Specific actor identification by persistent tracking ID and classification (e.g., `"Track #4 (worker)"`), with strict fallback to `"Unassigned Monitored Actor"` if no track is associated.
- **What:** Controlled human-readable event taxonomy (e.g., `"Restricted Zone Breach & Moving Vehicle Proximity"`).
- **Where:** Monitored zone name (e.g., `"Zone 'Heavy Machinery Cell' (restricted)"`) or relative image frame quadrant.
- **When:** Stream elapsed timestamp (e.g., `"14.20s (Frame 426)"`).
- **Why:** Structured bullet points detailing every contributing perception finding:
  - Exact missing PPE items (e.g., `Missing required hard_hat, safety_vest`).
  - Zone intrusion and dwell duration (e.g., `Occupied restricted zone 'Cell A' for 5.2s`).
  - Hazardous spatial relationships (e.g., `Rapid approach toward Forklift #2 at 180 px/s`).
  - Preceding early-warning indicators (e.g., `Trajectory projection showed zone interception in 1.8s`).

### Transparent Risk Score Breakdown
The risk engine deconstructs every incident score into an auditable arithmetic audit table matching the core risk calculation:
- **Base Severity Contributions:** Atomic weights mapped to each distinct risk factor (e.g., Zone Breach: +70.0, Missing Hard Hat: +35.0).
- **Concurrent Escalation Penalties:** Multi-factor compound hazards incur an explicit compounding penalty (+15.0 per concurrent hazard).
- **Classification Tiers:** Final score normalized to `[0, 100]` and classified into `CRITICAL (≥80)`, `HIGH (≥60)`, `MEDIUM (≥35)`, `LOW (≥15)`, or `INFO (<15)`.

### Temporal Incident Progression Timeline
Leveraging the rolling temporal log service, the system reconstructs a chronological window (preceding frames `BEFORE`, incident trigger frame `EVENT`, and subsequent frames `AFTER`):
- Tracks behavior state transitions (e.g., `STATIONARY → MOVING → RAPID_APPROACH`).
- Tracks spatial distance convergence (e.g., `120px (FAR) → 45px (NEAR)`).
- Captures risk escalation trajectory over time.

### Risk Trend Trajectory Visualization
The dashboard renders a 2D canvas sparkline trajectory showing risk score dynamics before, during, and after each incident.
- **Technical Honesty:** If fewer than 2 temporal frames are recorded near the event window, the chart explicitly indicates `"Insufficient temporal data for trend trajectory (< 2 data points)"` rather than fabricating synthetic points.

### Incident Evidence Package & Disclaimer
Operators can export an auditable JSON package (`GET /api/v1/incidents/{incident_id}/package`) containing:
- Complete incident metadata, tracking IDs, bounding boxes, and timestamp.
- Five-W causation statements and full risk score breakdown.
- Temporal timeline and trend points.
- Visual evidence status (`has_snapshot`, `snapshot_path`).
- **Investigative Disclaimer:** Explicit notice affirming that vision telemetry is algorithmic assistance and requires human safety officer validation prior to formal disciplinary or regulatory action.

### Descriptive Operational Safety Analytics
Accessible via the Analytics tab and `/api/v1/analytics/summary`:
- **KPI Metrics:** Total recorded incidents, critical incidents, zone intrusions, PPE violations, average incident duration, active incidents, proximity alerts.
- **Event Distribution:** Proportional breakdown across safety event types.
- **Active Safety Zones Summary:** Incident counts and risk severity per configured zone.
- **Strict Anti-Profiling Policy:** Purely facility- and equipment-level descriptive statistics. No employee ranking, subjective grading, or individual worker productivity metrics.
### Step 17 REST API Endpoints
| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/v1/incidents` | List recorded incidents with query filters (`status`, `risk_level`, `event_type`, `track_id`, `zone`) |
| `GET` | `/api/v1/incidents/{incident_id}/explanation` | Comprehensive audit package: 5W breakdown, risk score math, timeline, trend points |
| `GET` | `/api/v1/incidents/{incident_id}/timeline` | Chronological event window timeline entries |
| `GET` | `/api/v1/incidents/{incident_id}/package` | Complete downloadable JSON evidence package with verification disclaimer |
| `GET` | `/api/v1/analytics/summary` | Descriptive facility-level safety analytics summary |

---

## 19. Final Evaluation, Benchmarking & Competition Readiness (Step 18)

Step 18 represents the final stabilization, empirical benchmarking, edge-case hardening, and presentation readiness of IntelliWatch.

### 19.1 System Architecture Diagram

```text
CCTV Stream / Image / Video Upload
                  │
                  ▼
         [ Preprocessing ]
         - Resolution normalization (640x640 letterbox)
         - Aspect ratio & padding tracking (scale_factor, pad_offset)
                  │
                  ▼
          [ Perception ]
    ┌─────────────┼───────────────┬────────────────┬──────────────┐
    ▼             ▼               ▼                ▼              ▼
[YOLO11n]  [YOLOv8s-World]    [SafetyVision]   [ByteTrack]   [Depth-Anything]
COCO Base  Industrial Domain    Custom PPE      Kalman MOT    Relative Disparity
    └─────────────┬───────────────┘                │              │
                  ▼                                │              │
        [ Multi-Detector Merge ]                   │              │
        - Class prioritization                     │              │
        - Non-Maximum Suppression (IoU 0.45)       │              │
                  │                                │              │
                  ▼                                │              │
         [ Bounding Boxes ] ───────────────────────┘              │
                  │                                               │
                  ▼                                               │
       [ Persistent Tracking ]                                    │
       - Trajectory histories & motion vectors                    │
       - Contact foot-points (x_mid, y_bottom)                    │
                  │                                               │
                  ▼                                               │
      [ Scene Understanding ] ◄───────────────────────────────────┘
    ┌─────────────┼────────────────┬────────────────┐
    ▼             ▼                ▼                ▼
[Zones]      [Behavior]      [Compliance]     [Scene Graph]
Ray-Casting  Kinematic State Spatial IoU &    Entity relations,
In-Polygon   Machine (FALL/  Containment      spatial proximity,
Dwell timer  LOITER/RAPID)   Temporal checks  approach vectors
    └─────────────┬────────────────┴────────────────┘
                  │
                  ▼
       [ Safety Intelligence ]
    ┌─────────────┴────────────────┐
    ▼                              ▼
[Risk Engine]            [Prediction Engine]
Multi-factor arithmetic  Linear trajectory projection,
concurrency escalation   hazard interception warnings,
Lifecycle management     escalation trend estimation
    └─────────────┬────────────────┘
                  │
                  ▼
      [ Incident Intelligence ]
    ┌─────────────┼────────────────┬────────────────┐
    ▼             ▼                ▼                ▼
[Evidence]   [Explanation]   [Timeline]       [Analytics]
Snapshot     5W Structured   Before / Event / Facility KPIs,
storage      Causation       After window     event distributions,
JPEG/JSON    Audit package   reconstruction   no-profiling policy
                  │
                  ▼
     [ Industrial Operator Console ]
     - Glassmorphism dark-mode UI (http://localhost:8000/)
     - Live perception telemetry & interactive safety zones
     - Auditable incident investigation & JSON evidence download
```

### 19.2 Verified Technology Stack
Only verified, actively executed components are utilized. **Zero paid APIs, zero cloud AI, zero external LLMs, and zero unverified database dependencies.**

| Domain | Technology / Library | Version / Weights | Operational Role |
|---|---|---|---|
| **Runtime & Backend** | Python | 3.13.7 | Core language runtime |
| | FastAPI | >= 0.110.0 | High-performance asynchronous REST API framework |
| | Pydantic | >= 2.6.0 | Strict schema validation and data integrity |
| | Uvicorn | >= 0.28.0 | ASGI web server for local/production serving |
| **Perception** | Ultralytics / PyTorch | PyTorch 2.x / CPU | Deep learning tensor execution |
| | YOLO11 Nano | `weights/yolo11n.pt` (~5.6 MB) | Primary person and standard entity detection |
| | YOLOv8s-World-v2 | `weights/yolov8s-worldv2.pt` (~25.9 MB) | Industrial open-vocabulary detector (vehicles, machinery) |
| | SafetyVision YOLOv8n | `weights/ppe_yolov8n.pt` | Specialized PPE class detector |
| | ByteTrack | Python / NumPy | Multi-object tracking (MOT) with Kalman filter |
| **Spatial & Reasoning** | OpenCV | >= 4.8.0 | Frame manipulation, geometry, ray-casting point-in-polygon |
| | Monocular Relative Depth | Depth-Anything-V2-Small / Heuristic | Relative inverse disparity estimation |
| | Behavior State Machine | Python / Heuristic kinematics | Temporal velocity, acceleration, loitering, fall heuristics |
| | Relational Scene Graph | Directed Graph / Python | Subject-predicate-object spatial & kinematic facts |
| | Deterministic Risk Engine | Python arithmetic | Multi-factor severity scoring + compound escalation penalties |
| | Predictive Trajectory Engine| Linear kinematic projection | Short-horizon collision & zone interception warning |
| **Frontend** | Vanilla HTML5 / CSS3 / ES6 | No external frontend frameworks | Responsive dark-mode industrial control room console |
| | HTML5 Canvas 2D | Native browser API | Real-time zone polygon drawing & risk trend sparkline |
| **Storage & Persistence**| Local Disk JSON | Atomic file writes | Zone configurations (`configs/zones.json`) |
| | Local File Repository | JPEG / JSON | Incident evidence snapshots (`data/output/incidents/`) |
| | In-Memory Ring Buffer | Deque (maxlen=1000) | Temporal rolling logs & entity trajectory history |

### 19.3 Final Feature Verification Matrix

| Feature | Implementation Status | Tested? | Real Video Evidence? | Synthetic Evidence? | Known Limitation |
|---|---|---|---|---|---|
| **General Object Detection** | Complete | Yes (Pytest) | Yes (`cctv_worker_moving.mp4`, 48 detections) | Yes | Standard COCO classes; downsizes to 640x640 |
| **Industrial Detection** | Complete | Yes (Pytest) | Yes (Identified industrial structures) | Yes | Open-vocabulary confidences vary with lighting |
| **Multi-Object Tracking (ByteTrack)**| Complete | Yes (Pytest) | Yes (Track #1, Track #17 persistent) | Yes | Occlusion exceeding buffer (30 frames) drops ID |
| **PPE Compliance & Association** | Complete | Yes (Pytest) | Yes (Evaluated worker head/torso) | Yes | Dependent on detector bbox overlap heuristics |
| **Interactive Safety Zones** | Complete | Yes (Pytest) | Yes (Evaluated against CCTV frames) | Yes | 2D image coordinates; non-metric depth |
| **Monocular Relative Depth** | Complete | Yes (Pytest) | Yes (Disparity percentiles computed) | Yes | Non-metric; physical meters not calibrated |
| **Kinematic Behavior Analysis** | Complete | Yes (Pytest) | Yes (Observed STATIONARY -> RAPID_MOVEMENT) | Yes | 2D pixel velocity proxy; perspective distortion |
| **Relational Scene Graph** | Complete | Yes (Pytest) | Yes (Active entity-zone spatial edges) | Yes | Graph edges evaluated within visible frame only |
| **Worker-Vehicle Proximity** | Complete | Yes (Pytest) | No* (No moving vehicles in real clip) | Yes (Verified in Step 15 synthetic suite) | Real clip lacks moving vehicles; verified synthetically |
| **Deterministic Risk Engine** | Complete | Yes (Pytest) | Yes (Active multi-factor scoring) | Yes | Arithmetic formula; requires tuned thresholds |
| **Predictive Early Warnings** | Complete | Yes (Pytest) | Yes (Trajectory projection warnings) | Yes | Short-horizon linear projection (0.5s - 2.0s) |
| **Incident Recording & Evidence** | Complete | Yes (Pytest) | Yes (Recorded incident packages) | Yes | Limited to configured cooldown intervals |
| **Deterministic 5W Explanation** | Complete | Yes (Pytest) | Yes (Evaluated on confirmed events) | Yes | Purely rule-based; zero generative hallucination |
| **Risk Score Breakdown** | Complete | Yes (Pytest) | Yes (Transparent factor weights) | Yes | Formula-bound arithmetic sum |
| **Chronological Timeline** | Complete | Yes (Pytest) | Yes (Reconstructed BEFORE/EVENT/AFTER) | Yes | Dependent on rolling buffer depth (1000 frames) |
| **Risk Trend Sparkline** | Complete | Yes (Pytest) | Yes (Visualized on UI canvas) | Yes | Shows explicit fallback if < 2 data points |
| **Descriptive Safety Analytics** | Complete | Yes (Pytest) | Yes (Aggregated KPI metrics) | Yes | Facility-level only; no worker profiling |
| **Evidence Package Download** | Complete | Yes (Pytest) | Yes (JSON download with disclaimer) | Yes | Operator verification disclaimer required |

*\*Technical Honesty Notice: The real CCTV sample `cctv_worker_moving.mp4` contains walking workers but no moving forklifts. Multi-entity Worker → Vehicle proximity interaction was validated through deterministic synthetic benchmark tests in Step 15.*

### 19.4 Empirical Performance Benchmarks (Measured on CPU)
*Benchmark Environment: Windows 11, Intel Core CPU, Python 3.13.7, CPU Execution Mode (`torch.device("cpu")`)*

| Pipeline Component | Measured Load Time | Average Latency / Frame | Measured Throughput (FPS) | Hardware Profile |
|---|---|---|---|---|
| **YOLO11n General Detector** | 67.5 ms | **33.7 ± 1.6 ms** | **29.7 FPS** | CPU |
| **YOLOv8s-World-v2 Industrial Detector** | 1973.2 ms | **79.0 ± 1.7 ms** | **12.7 FPS** | CPU |
| **ByteTrack Multi-Object Tracker** | < 1 ms | **0.60 ms** | **1661.9 FPS** | CPU |
| **Reasoning, Scene Graph & Intelligence** | N/A | **~42.6 ms** | **23.5 FPS** | CPU |
| **Complete End-to-End Orchestrator Pipeline** | 0.1 ms | **164.1 ± 8.0 ms** | **6.09 FPS** | CPU (All layers active) |

### 19.5 Real Image & Real Video Validation Results

#### Real Image (`industrial_cctv.jpg` - 1376x768)
- **Detections:** 2 objects (1 worker, 1 industrial vehicle structure).
- **Tracks:** 2 active tracks.
- **PPE Compliance:** 1 worker evaluated (verified missing hard hat / high-vis vest).
- **Zone Occupancy:** 4 configured zones evaluated via Ray-Casting.
- **Latency:** Cold start 3327.6 ms.

#### Real Video (`cctv_worker_moving.mp4` - 1376x768, 30 FPS, 45 frames)
- **Processed:** 45 / 45 frames (100%).
- **Processing FPS on CPU:** 4.29 FPS.
- **Unique Track Identities:** 2 persistent tracks (Track #1, Track #17).
- **Behavior Transitions Observed:**
  - `Track #1`: `UNKNOWN` $\rightarrow$ `RAPID_MOVEMENT` at $0.13\text{s}$
  - `Track #17`: `UNKNOWN` $\rightarrow$ `STATIONARY` at $1.00\text{s}$
  - `Track #17`: `STATIONARY` $\rightarrow$ `RAPID_MOVEMENT` at $1.10\text{s}$
- **Recorded Incidents:** 50 incidents across timeline with visual snapshots and audit packages.

### 19.6 Edge & Failure-Case Hardening
All tested failure modes passed without uncaught exceptions or crashes:
- **Empty / 0-dim frame:** Handled gracefully with explicit `ValueError: Input frame cannot be empty or None.`
- **Blank uniform frame:** 0 detections, 0 risk score, zero pipeline errors.
- **Invalid video filepath:** Raises clean `VideoFileNotFoundError`.
- **Missing incident explanation query:** Returns HTTP 404 response.
- **Degenerate zone polygon (< 3 vertices):** Rejected by Pydantic validation.
- **Missing visual evidence snapshot:** Handled honestly (`has_snapshot=False`, metadata-only notice displayed, zero placeholder generation).

### 19.7 Honest Technical Limitations & Disclosures
1. **CPU Throughput:** In CPU mode, the compound pipeline executes at ~4.3 – 6.1 FPS. Real-time 30 FPS processing requires GPU acceleration or selective frame-skipping (`FRAME_SAMPLING_INTERVAL = 3`).
2. **Relative Monocular Depth:** Depth values represent inverse relative disparity. Metric real-world distances (meters) are not claimed without physical camera calibration.
3. **2D Image-Space Zones:** Safety zones are defined in 2D image coordinates and evaluate worker foot-points. Perspective foreshortening can affect apparent boundary proximity.
4. **Short-Horizon Trajectory:** Trajectory prediction is a short-horizon (0.5s – 2.0s) linear extrapolation for early warning; it does not guarantee future events.
5. **No Generative Hallucination:** Explanations are strictly deterministic facts compiled from vision state; they contain no speculative or conversational text.
6. **No Worker Profiling:** Safety analytics are strictly facility- and equipment-focused; individual worker scoring or performance profiling is intentionally not supported.

### 19.8 Competition Demonstration Workflow

```text
Step 1: Start Backend Server
        python scripts/run_server.py
        (Available at http://localhost:8000 and http://localhost:8000/dashboard)

Step 2: Open Operator Console in Browser
        Navigate to http://localhost:8000/

Step 3: Demonstrate Live Scene Understanding
        - View real-time perception viewport: Bounding boxes, Track IDs, Contact foot-points.
        - Observe Scene Graph tab: Real-time directional relationships (Track #1 NEAR Track #2).

Step 4: Demonstrate Interactive Safety Zones (Step 16)
        - Switch to "Safety Zones" tab.
        - Click "Start Drawing Polygon".
        - Click vertices on the viewport to outline a restricted hazard perimeter.
        - Name the zone "High Voltage Cell", select type "Restricted", set dwell limit 5.0s, and click "Save Safety Zone".
        - Observe the zone persist in the active list and immediately engage the risk engine.

Step 5: Demonstrate PPE Compliance & Behavior
        - Click "PPE Compliance" tab: Show itemized inventory (Hard Hat, High-Vis Vest) and compliance status.
        - Click "Behavior & Depth" tab: Show kinematic speed (px/s) and behavioral state (MOVING / STATIONARY).

Step 6: Demonstrate Confirmed Incident & Explainability (Step 17)
        - Scroll down to the Recorded Safety Incidents table.
        - Click "Evidence" button on any confirmed incident.
        - The Incident Audit Modal opens:
          • Five-W Box: Review deterministic WHO, WHAT, WHERE, WHEN, WHY statements.
          • Risk Score Breakdown: Review base factor contributions + compounding escalation penalties.
          • Chronological Timeline: Review BEFORE, EVENT, and AFTER frame progressions.
          • Risk Trajectory Sparkline: Observe risk curve before, during, and after event.
          • Evidence Snapshot: Verify captured visual snapshot or explicit metadata-only notice.

Step 7: Download Auditable Evidence Package
        - Click "📥 Download Package (.json)" in the modal.
        - Inspect the exported audit file with structured causation data and legal verification disclaimer.

Step 8: Demonstrate Descriptive Safety Analytics
        - Click "Safety Analytics" tab.
        - Review facility KPI cards (Total Incidents, Critical Incidents, Zone Intrusions, PPE Violations).
        - Review event type distribution bars and active zone risk rankings.
        - Highlight the Anti-Worker Profiling disclosure.
```

### 19.9 Startup Commands & Verification
```bash
# 1. Run complete pytest regression suite (448 passing tests)
pytest

# 2. Run comprehensive E2E validation script
python scripts/validate_step3_e2e.py

# 3. Start IntelliWatch API & Operator Console
python scripts/run_server.py
# Server running at: http://localhost:8000/
# Dashboard accessible at: http://localhost:8000/dashboard
```

---

## 20. Production Hardening, Setup & Deployment Guide

### 20.1 Prerequisites & System Requirements
- **Operating System:** Windows 10/11 64-bit or Linux (Ubuntu 20.04+)
- **Python:** Python 3.10 – 3.14 (Python 3.14 tested with `.venv`)
- **Memory & Storage:** Minimum 8 GB RAM, 2 GB free disk space (excluding video media)
- **GPU (Optional):** NVIDIA GPU supported by PyTorch CUDA build.
  > **Hardware Compatibility Notice:** If running on newest generation GPUs (e.g. NVIDIA GeForce RTX 5050 Laptop with Compute Capability 12.0 / `sm_120`), PyTorch `2.14.0+cu126` automatically falls back to CPU execution cleanly. CUDA tensor acceleration for CC 12.0 requires PyTorch built with CUDA 13.x+.

### 20.2 Initial Setup & Environment Configuration (Windows PowerShell)

```powershell
# 1. Navigate to the project root
cd CuriousPARC\IntelliWatch

# 2. Create the Python virtual environment
python -m venv .venv

# 3. Activate the virtual environment
.\.venv\Scripts\Activate.ps1

# 4. Install dependencies
pip install --upgrade pip
pip install -r requirements.txt

# 5. Configure environment variables from template
cp .env.example .env
# Edit .env to set a unique AUTH_SECRET_KEY and strong admin password
```

### 20.3 Starting the Server
```powershell
# Launch the FastAPI server with live reload enabled
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```
- **Landing Page:** [http://localhost:8000/](http://localhost:8000/)
- **Operations Console:** [http://localhost:8000/dashboard](http://localhost:8000/dashboard)
- **Interactive OpenAPI Docs:** [http://localhost:8000/docs](http://localhost:8000/docs)

### 20.4 First-Run Administrator Account & Security
When launched against a fresh `data/auth.db`, IntelliWatch automatically bootstraps the initial Administrator account using credentials defined in `.env`:
- Default username: `admin`
- Set `AUTH_DEFAULT_ADMIN_PASSWORD` in `.env` to your desired strong password before first launch.
- If using the default password, immediately update it via the **🔑 Change Password** button in the dashboard or via `POST /api/v1/auth/change-password`.

### 20.5 GitHub Upload Safety & Exclusions
The repository `.gitignore` ensures that zero sensitive data or operational files are committed:
- **Databases:** `data/alerts.db`, `data/auth.db`, `*.sqlite`, `*.db` are completely excluded.
- **Secrets:** `.env` is ignored; only `.env.example` with non-sensitive placeholders is tracked.
- **Weights:** Heavy model weight files (`*.pt`, `*.onnx`, `weights/`) are excluded.
- **Media & Caches:** `.venv/`, `__pycache__/`, `logs/`, test outputs, and upload buffers are ignored.

