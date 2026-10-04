from pydantic import BaseModel, Field


class VideoMetadataSchema(BaseModel):
    """Schema for public video stream/file metadata inspection."""
    source_path: str = Field(..., description="File path or URL of video source")
    fps: float = Field(..., description="Frames per second")
    width: int = Field(..., description="Original frame width in pixels")
    height: int = Field(..., description="Original frame height in pixels")
    total_frames: int = Field(..., description="Total number of frames")
    duration_sec: float = Field(..., description="Total video duration in seconds")


class VideoInspectRequest(BaseModel):
    """Request payload to inspect a local video source."""
    video_path: str = Field(..., description="Local path to video file")
