from functools import lru_cache
from typing import Any, List, Optional
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    """
    Central configuration for IntelliWatch.
    Loads values from environment variables or a .env file with sensible defaults.
    """
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application Information
    PROJECT_NAME: str = "IntelliWatch"
    VERSION: str = "0.1.0"
    ENVIRONMENT: str = "development"
    DEBUG: bool = False

    # API Server Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8000

    # Security & CORS Configuration
    CORS_ORIGINS: List[str] = [
        "http://localhost:8000",
        "http://127.0.0.1:8000",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ]

    @field_validator("CORS_ORIGINS", mode="before")
    @classmethod
    def parse_cors_origins(cls, v: Any) -> List[str]:
        if isinstance(v, str):
            return [origin.strip() for origin in v.split(",") if origin.strip()]
        return v

    # Logging Configuration
    LOG_LEVEL: str = "INFO"
    LOG_FILE: str = "logs/intelliwatch.log"

    # Computer Vision Pipeline Configuration (Step 1 Baseline)
    MODEL_PATH: str = "weights/yolov8n.pt"
    CONFIDENCE_THRESHOLD: float = 0.50
    IOU_THRESHOLD: float = 0.45
    VIDEO_SOURCE: str = "data/samples/synthetic_test.mp4"
    OUTPUT_DIR: str = "data/output"
    FPS_SETTINGS: int = 30

    # Video Preprocessing & Sampling (Step 2)
    TARGET_WIDTH: int = 640
    TARGET_HEIGHT: int = 640
    RESIZE_ENABLED: bool = True
    PRESERVE_ASPECT_RATIO: bool = True
    FRAME_SAMPLING_INTERVAL: int = 1  # 1 = process every frame, 5 = every 5th frame
    NORMALIZE_FRAME: bool = False

    # Object Detection Configuration (Step 3 & Accuracy Hardening)
    DETECTION_MODEL_PATH: str = "weights/yolo11n.pt"
    DETECTION_CONFIDENCE_THRESHOLD: float = 0.25
    DETECTION_PERSON_CONFIDENCE_THRESHOLD: float = 0.20  # Lower threshold for person class to recover occluded/seated workers
    DETECTION_VEHICLE_CONFIDENCE_THRESHOLD: float = 0.38  # Higher threshold for vehicles to reject background clutter false positives
    DETECTION_IOU_THRESHOLD: float = 0.45
    DETECTION_DEVICE: str = "auto"
    DETECTION_IMGSZ: int = 640
    DETECTION_DEBUG: bool = False  # Lightweight diagnostic logging mode
    DEBUG_PERSON_DETECTION: bool = False  # Explicit flag for person detection diagnostic telemetry

    # Vehicle Deduplication & Occupant Perception
    VEHICLE_DEDUPLICATION_ENABLED: bool = True
    VEHICLE_DEDUPLICATION_IOU: float = 0.40    # Overlapping vehicle suppression threshold
    VEHICLE_DEDUPLICATION_IOMIN: float = 0.65  # Sub-box containment threshold (IoMin) to eliminate nested vehicle boxes
    OCCUPANT_DETECTION_ENABLED: bool = True    # High-resolution ROI perception for in-vehicle workers
    OCCUPANT_CROP_PADDING: float = 0.15       # 15% bounding padding for vehicle cabs and roll-cages
    OCCUPANT_CONFIDENCE_THRESHOLD: float = 0.15  # High recall threshold for localized ROI person search
    TRACK_HIGH_THRESH: float = 0.20
    TRACK_NEW_THRESH: float = 0.18
    PPE_LOCALIZED_INFERENCE_ENABLED: bool = True

    # Person Deduplication (Sub-box containment suppression for duplicate upper-body / torso boxes)
    PERSON_DEDUPLICATION_ENABLED: bool = True
    PERSON_DEDUPLICATION_IOMIN: float = 0.75  # Containment threshold (IoMin) to eliminate nested person sub-boxes

    # Industrial Perception & Domain Configuration
    INDUSTRIAL_DETECTION_ENABLED: bool = True
    INDUSTRIAL_MODEL_PATH: str = "weights/yolov8s-worldv2.pt"
    INDUSTRIAL_CONFIDENCE_THRESHOLD: float = 0.15
    INDUSTRIAL_IOU_THRESHOLD: float = 0.45
    INDUSTRIAL_DEVICE: str = "auto"
    INDUSTRIAL_IMGSZ: int = 640
    INDUSTRIAL_VOCABULARY: List[str] = [
        "forklift",
        "industrial vehicle",
        "machinery",
        "robotic arm",
        "conveyor",
        "pallet",
        "safety barrier",
        "electrical cabinet",
    ]
    # False Positive Filtering for Industrial Scenes
    FILTER_IRRELEVANT_COCO_CLASSES: bool = True
    IRRELEVANT_COCO_CLASSES: List[str] = [
        "boat", "bench", "sports ball", "frisbee", "skis", "snowboard", "baseball bat",
        "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
        "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut",
        "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
        "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
        "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush", "tie", "suitcase",
        "fire hydrant", "traffic light", "stop sign", "parking meter", "bird", "cat",
        "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"
    ]

    # Multi-Object Tracking Configuration (Step 4)
    TRACKING_ENABLED: bool = True
    TRACKER_TYPE: str = "bytetrack"          # Identifier for active tracker ("bytetrack")
    TRACK_HISTORY_LENGTH: int = 30           # Max centroid points per trajectory (frames)
    TRACK_HIGH_THRESH: float = 0.25          # ByteTrack: high-confidence detection threshold
    TRACK_LOW_THRESH: float = 0.05           # ByteTrack: low-confidence detection threshold (2nd pass)
    TRACK_NEW_THRESH: float = 0.18           # ByteTrack: threshold to initialize new track (aligned with person recall)
    TRACK_MATCH_THRESH: float = 0.8          # ByteTrack: IoU match threshold for association
    TRACK_BUFFER: int = 30                   # ByteTrack: frames to keep a lost track alive
    TRACK_MAX_TIME_LOST: int = 30            # Max frames before a LOST track is REMOVED

    # PPE Detection Configuration (Step 5A & 5B)
    PPE_MODEL_PATH: str = "weights/ppe_yolov8n.pt"
    PPE_CONFIDENCE_THRESHOLD: float = 0.25
    PPE_IOU_THRESHOLD: float = 0.45
    PPE_DEVICE: str = "auto"
    PPE_LOCALIZED_INFERENCE_ENABLED: bool = True  # High-resolution worker-ROI PPE inference
    PPE_IMGSZ: int = 640
    PPE_ENABLED: bool = True
    PPE_CLASSES: Optional[List[str]] = None

    # PPE Compliance & Spatial Association (Step 5B)
    PPE_COMPLIANCE_ENABLED: bool = True
    PPE_REQUIRED_CLASSES: List[str] = ["Hardhat", "Safety Vest", "Gloves", "Goggles"]
    PPE_ASSOCIATION_IOU_THRESHOLD: float = 0.10
    PPE_CENTER_CONTAINMENT_THRESHOLD: float = 0.50
    PPE_VIOLATION_CONFIRMATION_FRAMES: int = 3
    PPE_MIN_PERSON_HEIGHT_GOGGLES: float = 120.0
    PPE_MIN_PERSON_HEIGHT_GLOVES: float = 100.0
    PPE_MIN_PERSON_HEIGHT_GENERAL: float = 50.0

    # Restricted Zone Spatial Understanding Configuration (Step 6)
    ZONE_DETECTION_ENABLED: bool = True
    ZONE_ENTRY_CONFIRMATION_FRAMES: int = 3
    ZONE_MAX_DWELL_SECONDS: float = 10.0
    ZONE_CONFIG_PATH: str = "configs/zones.json"

    # Monocular Depth Estimation & 2D-to-3D Spatial Understanding (Step 7)
    DEPTH_ENABLED: bool = True
    DEPTH_MODEL_NAME: str = "depth-anything/Depth-Anything-V2-Small-hf"
    DEPTH_DEVICE: str = "cpu"
    DEPTH_INPUT_SIZE: int = 518
    DEPTH_OUTPUT_MODE: str = "relative"

    # Behavior & Temporal Analysis Configuration (Step 8)
    # All motion thresholds operate in image-space pixels — NOT real-world physical units.
    BEHAVIOR_ENABLED: bool = True

    # History window: how many seconds of centroid history to retain per track
    BEHAVIOR_HISTORY_SECONDS: float = 5.0

    # Temporal confirmation: consecutive frames required before confirming a new behavior state
    BEHAVIOR_CONFIRMATION_FRAMES: int = 3

    # Stationary detection: movement below this pixel displacement per observation → candidate stationary
    BEHAVIOR_STATIONARY_DISTANCE_THRESHOLD: float = 5.0  # pixels per observation

    # How long (seconds) a track must remain below the stationary threshold before entering STATIONARY state
    BEHAVIOR_STATIONARY_SECONDS: float = 1.5

    # Rapid-movement detection: image-space speed threshold (pixels/second)
    BEHAVIOR_RUNNING_VELOCITY_THRESHOLD: float = 80.0   # px/s in image space

    # Sudden-movement detection: acceleration magnitude threshold (pixels/second²)
    BEHAVIOR_SUDDEN_ACCELERATION_THRESHOLD: float = 150.0  # px/s²

    # Direction-change detection: minimum bearing change (degrees) to count as a direction change
    BEHAVIOR_DIRECTION_CHANGE_DEGREES: float = 60.0

    # Prolonged stationary / loitering: seconds stationary before escalating to PROLONGED_STATIONARY
    BEHAVIOR_LOITERING_SECONDS: float = 10.0

    # Possible-fall heuristic: aspect-ratio change threshold (bbox h/w ratio drop) and
    # accompanying centroid displacement threshold (pixels) within a single observation
    BEHAVIOR_FALL_ASPECT_RATIO_CHANGE: float = 0.5    # sudden drop in height/width ratio
    BEHAVIOR_FALL_DISPLACEMENT_THRESHOLD: float = 30.0  # px sudden centroid jump

    # -------------------------------------------------------------------------
    # Step 9: Scene Graph & Situational Awareness Configuration
    # -------------------------------------------------------------------------
    SCENE_GRAPH_ENABLED: bool = True
    SCENE_VERY_NEAR_DISTANCE_THRESHOLD: float = 75.0   # pixels: max distance to classify entity pair as VERY_NEAR
    SCENE_NEAR_DISTANCE_THRESHOLD: float = 150.0       # pixels: max distance to classify entity pair as NEAR
    SCENE_MODERATE_DISTANCE_THRESHOLD: float = 300.0   # pixels: max distance to classify entity pair as MODERATE
    SCENE_FAR_DISTANCE_THRESHOLD: float = 400.0        # pixels: min distance beyond which entity pair is FAR
    SCENE_MIN_APPROACHING_RATE_PX_S: float = 15.0      # pixels/s: minimum net closing velocity to classify as APPROACHING
    SCENE_RELATION_CONFIRMATION_FRAMES: int = 3        # consecutive observations needed to confirm NEAR/APPROACHING
    SCENE_RELATION_END_CONFIRMATION_FRAMES: int = 3    # consecutive unobserved frames before ending relation
    SCENE_APPROACHING_ENABLED: bool = True             # whether temporal approaching/moving-away is evaluated
    SCENE_DEPTH_RELATION_ENABLED: bool = True          # whether relative depth modulates spatial proximity
    SCENE_DEPTH_NEAR_THRESHOLD: float = 0.20           # max relative depth difference for physical proximity
    INDUSTRIAL_DETECTION_INTERVAL: int = 1             # frames between full industrial open-vocabulary detections (1=every frame)

    # -------------------------------------------------------------------------
    # Step 10: Risk & Event Reasoning Configuration
    # -------------------------------------------------------------------------
    RISK_ENGINE_ENABLED: bool = True
    RISK_EVENT_CONFIRMATION_FRAMES: int = 3            # Consecutive frames before confirming an event
    RISK_EVENT_END_CONFIRMATION_FRAMES: int = 3        # Consecutive frames without condition before ending event
    RISK_EVENT_COOLDOWN_FRAMES: int = 5                # Cooldown frames post-end to suppress duplicate alarms
    RISK_COMPOUND_EVENT_ENABLED: bool = True           # Whether multi-factor compound events are generated

    # Risk score contributions per factor
    RISK_SCORE_PPE_VIOLATION: float = 25.0
    RISK_SCORE_ZONE_INTRUSION: float = 35.0
    RISK_SCORE_ZONE_DWELL: float = 40.0
    RISK_SCORE_RAPID_MOVEMENT: float = 20.0
    RISK_SCORE_SUDDEN_MOVEMENT: float = 20.0
    RISK_SCORE_FALL_LIKE: float = 50.0
    RISK_SCORE_PERSON_VEHICLE_PROXIMITY: float = 30.0
    RISK_SCORE_APPROACHING_VEHICLE: float = 45.0
    RISK_SCORE_WORKER_MACHINE: float = 30.0
    RISK_SCORE_PROLONGED_STATIONARY: float = 25.0
    RISK_SCORE_MULTI_FACTOR_ESCALATION: float = 15.0   # Incremental score added per concurrent factor beyond 1

    # Risk level classification thresholds
    RISK_LEVEL_INFO_THRESHOLD: float = 0.0
    RISK_LEVEL_LOW_THRESHOLD: float = 10.0
    RISK_LEVEL_MEDIUM_THRESHOLD: float = 30.0
    RISK_LEVEL_HIGH_THRESHOLD: float = 60.0
    RISK_LEVEL_CRITICAL_THRESHOLD: float = 85.0

    # -------------------------------------------------------------------------
    # Step 11: Predictive & Advanced Anomaly Intelligence Configuration
    # -------------------------------------------------------------------------
    PREDICTION_ENABLED: bool = True
    PREDICTION_HISTORY_FRAMES: int = 5                 # History points used for trajectory velocity fitting
    PREDICTION_HORIZON_FRAMES: int = 15                # Number of frames forward to project kinematics (~0.5s - 1.0s)
    PREDICTION_MIN_TRACK_HISTORY: int = 3              # Minimum consecutive tracking points required before projection
    PREDICTION_CONFIRMATION_FRAMES: int = 3            # Consecutive observations needed to confirm an early-warning indicator
    PREDICTION_END_CONFIRMATION_FRAMES: int = 3        # Consecutive frames without condition before ending indicator
    PREDICTION_RISK_HISTORY_FRAMES: int = 6            # Sliding window of risk scores for trend detection
    PREDICTION_RISK_ESCALATION_THRESHOLD: float = 20.0 # Delta increase in risk score across window to flag escalation
    PREDICTION_ZONE_MARGIN: float = 30.0               # Pixel margin around zone to flag trajectory-toward-zone
    PREDICTION_ANOMALY_WINDOW: int = 5                 # Window of behavior states to evaluate for temporal anomalies
    PREDICTION_REPEAT_EVENT_WINDOW: int = 8            # Window of frames to check for repeated safety incidents
    PREDICTION_PERSISTENCE_THRESHOLD: int = 4          # Minimum incident/proximity frames to flag persistent unsafe pattern

    # -------------------------------------------------------------------------
    # Step 22: Intelligent Safety Alerting, Evidence Capture & Alert Lifecycle
    # -------------------------------------------------------------------------
    ALERTING_ENABLED: bool = True
    ALERT_DATABASE_PATH: str = "data/alerts.db"
    ALERT_EVIDENCE_DIR: str = "data/output/alerts/evidence"
    ALERT_COOLDOWN_SECONDS: float = 30.0
    ALERT_CONFIDENCE_THRESHOLD: float = 0.30
    ALERT_CRITICAL_IMMEDIATE: bool = True
    ALERT_PERSISTENCE_FRAMES: int = 2

    # -------------------------------------------------------------------------
    # Step 23: Authentication, Role-Based Access Control, Audit Logs & Admin
    # -------------------------------------------------------------------------
    AUTH_ENABLED: bool = True
    AUTH_SECRET_KEY: str = "intelliwatch-control-room-super-secret-key-2026-cctv"
    AUTH_TOKEN_EXPIRE_MINUTES: int = 120
    AUTH_DB_PATH: str = "data/auth.db"
    AUTH_RATE_LIMIT_ATTEMPTS: int = 5
    AUTH_RATE_LIMIT_LOCKOUT_SECONDS: int = 300
    AUTH_DEFAULT_ADMIN_USERNAME: str = "admin"
    AUTH_DEFAULT_ADMIN_PASSWORD: str = "IntelliWatch2026!"
    AUTH_DEFAULT_ADMIN_EMAIL: str = "admin@intelliwatch.local"

    # Database Configuration
    DATABASE_URL: Optional[str] = None






@lru_cache()
def get_settings() -> AppSettings:
    """Returns a cached singleton instance of application settings."""
    return AppSettings()


# Module-level convenience constants
SCENE_VERY_NEAR_DISTANCE_THRESHOLD: float = 75.0
SCENE_NEAR_DISTANCE_THRESHOLD: float = 150.0
SCENE_MODERATE_DISTANCE_THRESHOLD: float = 300.0
SCENE_FAR_DISTANCE_THRESHOLD: float = 400.0
SCENE_MIN_APPROACHING_RATE_PX_S: float = 15.0

