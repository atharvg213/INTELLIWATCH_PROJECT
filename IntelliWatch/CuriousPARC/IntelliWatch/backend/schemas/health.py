from pydantic import BaseModel


class HealthResponse(BaseModel):
    """
    Standard health check response schema.
    """
    status: str = "ok"
    project: str = "IntelliWatch"
