"""Optional joint log–memo API. The original memo-only API is independent."""

from .sources import Snapshot, SourceError, prepare
from .model import JointEncoder, batch
from .inference import load_model, predict

__all__ = [
    "Snapshot",
    "SourceError",
    "prepare",
    "JointEncoder",
    "batch",
    "load_model",
    "predict",
]
