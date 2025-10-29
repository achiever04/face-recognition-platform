# backend/app/models/__init__.py
# This file marks the 'models' directory as a Python package.
# It can be used to easily import all models.

from .person import (
    FaceModel,
    TrackingRecordModel,
    AlertLogModel,
    DeepfakeLogModel,
    ConfigModel,
    PyObjectId
)

__all__ = [
    "FaceModel",
    "TrackingRecordModel",
    "AlertLogModel",
    "DeepfakeLogModel",
    "ConfigModel",
    "PyObjectId"
]