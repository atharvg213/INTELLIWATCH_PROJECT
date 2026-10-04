"""
backend/schemas/camera.py
Schemas for RTSP camera ingestion, multi-camera management, and stream telemetry.
"""
from enum import Enum
from typing import Any, Dict, List, Optional
import uuid
from pydantic import BaseModel, Field, model_validator


class CameraStatus(str, Enum):
    """Operational lifecycle state of a camera stream."""
    DISCONNECTED = "DISCONNECTED"
    CONNECTING = "CONNECTING"
    CONNECTED = "CONNECTED"
    RECONNECTING = "RECONNECTING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


class CameraRegisterRequest(BaseModel):
    """Payload to register a new camera stream."""
    camera_id: str = Field(default="", description="Unique alphanumeric identifier for camera (e.g. 'cam_01')")
    name: str = Field(..., description="Descriptive label for the camera location (e.g. 'Loading Dock East')")
    source: str = Field(default="", description="RTSP/RTMP/HTTP stream URL, local video file path, or webcam index")
    sampling_interval: int = Field(default=1, ge=1, description="Process every Nth frame (1 = all frames)")
    max_processing_fps: Optional[float] = Field(default=10.0, ge=0.5, le=60.0, description="Rate limit processing FPS to match hardware")
    auto_start: bool = Field(default=True, description="Automatically start ingestion and processing upon registration")
    loop_file: bool = Field(default=True, description="If source is a local video file, loop continuously to simulate live CCTV")
    reconnect_interval_sec: float = Field(default=3.0, ge=1.0, le=60.0, description="Seconds between reconnect attempts")

    @model_validator(mode="before")
    @classmethod
    def populate_defaults_and_aliases(cls, data: Any) -> Any:
        if isinstance(data, dict):
            if not data.get("camera_id"):
                data["camera_id"] = f"cam_{uuid.uuid4().hex[:8]}"
            if not data.get("source") and (data.get("rtsp_url") or data.get("url")):
                data["source"] = data.get("rtsp_url") or data.get("url")
            if "fps_limit" in data and "max_processing_fps" not in data:
                data["max_processing_fps"] = data["fps_limit"]
        return data


class CameraUpdateRequest(BaseModel):
    """Payload to update an existing camera configuration."""
    name: Optional[str] = Field(default=None, description="Updated descriptive label")
    source: Optional[str] = Field(default=None, description="Updated stream URL or file path")
    sampling_interval: Optional[int] = Field(default=None, ge=1, description="Updated frame sampling interval")
    max_processing_fps: Optional[float] = Field(default=None, ge=0.5, le=60.0, description="Updated max processing FPS")


class CameraTelemetrySchema(BaseModel):
    """Comprehensive operational telemetry and statistics for a single camera."""
    camera_id: str = Field(..., description="Camera ID")
    name: str = Field(..., description="Camera descriptive name")
    source_sanitized: str = Field(..., description="Sanitized stream URL with credentials masked")
    status: CameraStatus = Field(..., description="Current connection status")
    is_active: bool = Field(..., description="True if stream processing thread is actively running")
    width: int = Field(default=0, description="Native video width in pixels")
    height: int = Field(default=0, description="Native video height in pixels")
    stream_fps: float = Field(default=0.0, description="Source stream frame rate")
    ingest_fps: float = Field(default=0.0, description="Measured frame ingest rate")
    processing_fps: float = Field(default=0.0, description="Measured AI pipeline processing rate")
    avg_latency_ms: float = Field(default=0.0, description="Average AI pipeline latency in milliseconds")
    total_ingested_frames: int = Field(default=0, description="Total frames read from stream")
    total_processed_frames: int = Field(default=0, description="Total frames analyzed by AI pipeline")
    dropped_frames_count: int = Field(default=0, description="Total frames dropped to prevent latency backlog")
    active_tracks_count: int = Field(default=0, description="Currently tracked objects in camera scene")
    active_incidents_count: int = Field(default=0, description="Active safety violations/incidents on this camera")
    non_compliant_workers_count: int = Field(default=0, description="Workers currently failing PPE compliance")
    reconnect_attempts: int = Field(default=0, description="Number of reconnect attempts performed")
    last_error: Optional[str] = Field(default=None, description="Last recorded connection or processing error")
    uptime_seconds: float = Field(default=0.0, description="Continuous operational uptime in seconds")


class CameraListResponse(BaseModel):
    """Response containing all registered cameras."""
    total_cameras: int = Field(..., description="Total registered cameras")
    connected_count: int = Field(..., description="Number of currently connected cameras")
    cameras: List[CameraTelemetrySchema] = Field(default_factory=list, description="List of camera telemetry records")


class CameraDetailResponse(BaseModel):
    """Detailed response for a single camera, including its latest scene assessment."""
    camera: CameraTelemetrySchema = Field(..., description="Camera telemetry")
    latest_assessment: Optional[Dict[str, Any]] = Field(default=None, description="Latest structured frame assessment")
