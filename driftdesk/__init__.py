"""DriftDesk — API schema drift RL environment."""
from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("openenv-driftdesk")
except PackageNotFoundError:
    __version__ = "0.1.0-dev"

# Public API
from driftdesk.models import DriftDeskAction, DriftDeskObservation, DriftDeskState
from driftdesk.schemas import REGISTRY
from driftdesk.client import DriftDeskClient

__all__ = [
    "DriftDeskAction",
    "DriftDeskObservation",
    "DriftDeskState",
    "REGISTRY",
    "DriftDeskClient",
    "__version__",
]
