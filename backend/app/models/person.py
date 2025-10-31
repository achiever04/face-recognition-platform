# backend/app/models/person.py

from __future__ import annotations
from typing import List, Tuple, Optional, Any, Dict
from datetime import datetime
from pydantic import BaseModel, Field, field_validator
from pydantic import core_schema, GetCoreSchemaHandler
from bson import ObjectId

# ---------------------------------------------------------
# Helper: Pydantic-compatible ObjectId type
# ---------------------------------------------------------
class PyObjectId(ObjectId):
    """Wrapper to allow Pydantic models to accept/encode MongoDB ObjectId."""

    @classmethod
    def __get_pydantic_core_schema__(cls, source_type, handler: GetCoreSchemaHandler):
        def validate_objectid(v, info):
            if isinstance(v, ObjectId):
                return v
            if isinstance(v, str) and ObjectId.is_valid(v):
                return ObjectId(v)
            raise ValueError("Invalid ObjectId")
        return core_schema.no_info_plain_validator(validate_objectid)

    @classmethod
    def __modify_schema__(cls, field_schema: Dict[str, Any]) -> None:
        field_schema.update(type="string")


# ---------------------------------------------------------
# Face model stored in faces_collection
# ---------------------------------------------------------
class FaceModel(BaseModel):
    """
    Representation of a stored face/embedding.

    Fields:
      - _id (ObjectId) stored in Mongo as _id
      - target: unique name / identifier for the face (e.g., person name or filename)
      - embedding: base64 (or otherwise serialized) embedding string
      - updated_at: timestamp (datetime)
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    target: str = Field(..., description="Unique name/identifier for the face (e.g. filename or person name)")
    embedding: str = Field(..., description="Serialized (e.g., base64) embedding for the face")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update timestamp (UTC)")

    @field_validator("target")
    def non_empty_target(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("target must be a non-empty string")
        return v.strip()

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str, datetime: lambda v: v.isoformat()},
        # keep schema examples in a backward compatible place
        "json_schema_extra": {
            "example": {
                "_id": "654f9f6b1a2b3c4d5e6f7890",
                "target": "person_image.jpg",
                "embedding": "gAAAAAB... (base64)",
                "updated_at": "2025-10-26T12:00:00.000Z"
            }
        }
    }


# ---------------------------------------------------------
# Tracking record stored in tracking_collection
# ---------------------------------------------------------
class TrackingRecordModel(BaseModel):
    """
    Movement/tracking record for a detected person.

    Fields:
      - person: target identifier/name
      - camera_id: numeric camera id
      - camera_name: human readable name of camera
      - geo: (lat, lon) tuple
      - distance: matching distance (float)
      - confidence: 'high'|'medium'|'low'
      - timestamp: datetime of detection (UTC)
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    person: str = Field(..., description="Person identifier / target name")
    camera_id: int = Field(..., description="Camera numeric ID")
    camera_name: str = Field(..., description="Human readable camera name")
    geo: Tuple[float, float] = Field(..., description="(latitude, longitude)")
    distance: float = Field(..., description="Matching distance (lower = better match)")
    confidence: str = Field(..., description="Confidence level (high/medium/low)")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Detection timestamp (UTC)")

    @field_validator("person", "camera_name")
    def non_empty_str(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("must be a non-empty string")
        return s

    @field_validator("geo")
    def geo_shape_and_range(cls, v: Tuple[float, float]) -> Tuple[float, float]:
        if not (isinstance(v, (list, tuple)) and len(v) == 2):
            raise ValueError("geo must be a tuple/list of (lat, lon)")
        lat, lon = float(v[0]), float(v[1])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError("geo coordinates out of range")
        return lat, lon

    @field_validator("confidence")
    def validate_confidence(cls, v: str) -> str:
        allowed = {"high", "medium", "low"}
        val = v.strip().lower()
        if val not in allowed:
            raise ValueError(f"confidence must be one of {allowed}")
        return val

    @field_validator("timestamp", mode="before")
    def parse_timestamp(cls, v: Any) -> datetime:
        """
        Accept either:
          - datetime
          - ISO-formatted string
        """
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except Exception:
                # Fallback: try common formats, or raise
                raise ValueError("timestamp must be a datetime or ISO formatted string")
        raise ValueError("timestamp must be a datetime or ISO formatted string")

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str, datetime: lambda v: v.isoformat()},
        "json_schema_extra": {
            "example": {
                "_id": "654f9f6b1a2b3c4d5e6f7891",
                "person": "person_image.jpg",
                "camera_id": 1,
                "camera_name": "Pune Station",
                "geo": [18.528, 73.847],
                "distance": 0.35,
                "confidence": "high",
                "timestamp": "2025-10-26T12:05:30.123Z"
            }
        }
    }


# ---------------------------------------------------------
# Alert log stored in logs_collection (legacy schema kept)
# ---------------------------------------------------------
class AlertLogModel(BaseModel):
    """
    Alert log entry.
    Note: previous implementation stored geo as string; we accept both string and tuple but
    prefer storing geo as a string for backward compatibility with existing log_alert code.
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    target: str = Field(..., description="Target / person name")
    camera_id: int = Field(..., description="Camera numeric ID")
    camera_name: str = Field(..., description="Camera name")
    geo: str = Field(..., description="Geo coordinates stored as a string (legacy)")
    distance: float = Field(..., description="Matching distance")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp (UTC)")

    @field_validator("geo", mode="before")
    def geo_to_string(cls, v: Any) -> str:
        """
        Convert (lat, lon) tuple to string if necessary, otherwise accept strings.
        Keeps legacy behavior where geo is stored as a string.
        """
        if isinstance(v, (list, tuple)) and len(v) == 2:
            try:
                lat, lon = float(v[0]), float(v[1])
                return f"{lat:.6f},{lon:.6f}"
            except Exception:
                raise ValueError("Invalid geo coordinate values")
        if isinstance(v, str):
            return v
        raise ValueError("geo must be a string or a (lat, lon) tuple/list")

    @field_validator("timestamp", mode="before")
    def parse_ts(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except Exception:
                raise ValueError("timestamp must be ISO formatted string or datetime")
        raise ValueError("timestamp must be ISO formatted string or datetime")

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str, datetime: lambda v: v.isoformat()}
    }


# ---------------------------------------------------------
# Deepfake detection log stored in deepfake_collection
# ---------------------------------------------------------
class DeepfakeLogModel(BaseModel):
    """
    Deepfake detection event log schema.
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    camera_id: int = Field(..., description="Camera numeric ID")
    camera_name: str = Field(..., description="Camera name")
    geo: str = Field(..., description="Geo coordinates as string (legacy)")
    frame_id: int = Field(..., description="Frame number within source")
    is_fake: bool = Field(..., description="Whether detection judged the face as fake")
    confidence: float = Field(..., description="Classifier confidence (0.0 - 1.0)")
    bbox: List[int] = Field(..., description="[x1, y1, x2, y2]")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Event timestamp (UTC)")

    @field_validator("bbox")
    def bbox_shape(cls, v: List[int]) -> List[int]:
        if not (isinstance(v, (list, tuple)) and len(v) == 4):
            raise ValueError("bbox must be a list/tuple of 4 integers: [x1,y1,x2,y2]")
        return [int(x) for x in v]

    @field_validator("timestamp", mode="before")
    def parse_ts2(cls, v: Any) -> datetime:
        if isinstance(v, datetime):
            return v
        if isinstance(v, str):
            try:
                return datetime.fromisoformat(v)
            except Exception:
                raise ValueError("timestamp must be ISO formatted string or datetime")
        raise ValueError("timestamp must be ISO formatted string or datetime")

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str, datetime: lambda v: v.isoformat()}
    }


# ---------------------------------------------------------
# Generic configuration model (watchlist, geofences, etc.)
# ---------------------------------------------------------
class ConfigModel(BaseModel):
    """
    Generic configuration document used to store watchlist, geofences, and other site-wide configs.

    Fields:
      - name: e.g., 'watchlist' or 'geofences'
      - data: arbitrary JSON-compatible dict/list for the configuration
      - updated_at: timestamp
    """
    id: Optional[PyObjectId] = Field(alias="_id", default=None)
    name: str = Field(..., description="Configuration name (e.g., 'watchlist', 'geofences')")
    data: dict = Field(..., description="Configuration payload (list/dict)")
    updated_at: datetime = Field(default_factory=datetime.utcnow, description="Last update time (UTC)")

    @field_validator("name")
    def name_not_empty(cls, v: str) -> str:
        s = v.strip()
        if not s:
            raise ValueError("name must be a non-empty string")
        return s

    model_config = {
        "populate_by_name": True,
        "arbitrary_types_allowed": True,
        "json_encoders": {ObjectId: str, datetime: lambda v: v.isoformat()},
        "json_schema_extra": {
            "example": {
                "_id": "654f9f6b1a2b3c4d5e6f7892",
                "name": "watchlist",
                "data": ["person1", "person2"],
                "updated_at": "2025-10-26T12:00:00.000Z"
            }
        }
    }
