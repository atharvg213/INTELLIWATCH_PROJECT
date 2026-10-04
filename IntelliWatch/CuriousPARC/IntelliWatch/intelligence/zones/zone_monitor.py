from abc import ABC, abstractmethod
from backend.schemas.tracking import FrameTracks
from backend.schemas.events import IndustrialEvent


class BaseZoneMonitor(ABC):
    """
    Abstract Base Class for polygonal restricted zone & geofence monitoring.
    """

    @abstractmethod
    def check_intrusion(self, tracks: FrameTracks) -> list[IndustrialEvent]:
        """Evaluate whether tracked actors violate polygonal keep-out zones."""
        pass
