# backend/app/models/person.py

from pydantic import BaseModel, Field
from typing import List, Tuple, Optional
from datetime import datetime
from bson import ObjectId # Import ObjectId for Pydantic compatibility

# Helper for ObjectId validation in Pydantic
class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid ObjectId")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")

# --- Face Model (faces_collection) ---
class FaceModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias='_id', default=None)
    target: str = Field(...) # Name or identifier (e.g., filename)
    embedding: str = Field(...) # Base64 encoded encrypted embedding
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True # Needed for ObjectId
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "target": "person_image.jpg",
                "embedding": "gAAAAAB...",
                "updated_at": "2025-10-26T12:00:00.000Z"
            }
        }

# --- Tracking Record Model (tracking_collection) ---
class TrackingRecordModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias='_id', default=None)
    person: str = Field(...)
    camera_id: int = Field(...)
    camera_name: str = Field(...)
    geo: Tuple[float, float] = Field(...)
    distance: float = Field(...)
    confidence: str = Field(...) # "high", "medium", "low"
    timestamp: str = Field(...) # ISO format string

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "person": "person_image.jpg",
                "camera_id": 1,
                "camera_name": "Pune Station",
                "geo": [18.528, 73.847],
                "distance": 0.35,
                "confidence": "high",
                "timestamp": "2025-10-26T12:05:30.123Z"
            }
        }

# --- Alert Log Model (logs_collection) ---
# Note: This is similar to TrackingRecordModel but might evolve differently
class AlertLogModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias='_id', default=None)
    target: str = Field(...)
    camera_id: int = Field(...)
    camera_name: str = Field(...)
    geo: str = Field(...) # Stored as string in the current log_alert
    distance: float = Field(...)
    timestamp: str = Field(...) # ISO format string

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# --- Deepfake Log Model (deepfake_collection) ---
class DeepfakeLogModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias='_id', default=None)
    camera_id: int = Field(...)
    camera_name: str = Field(...)
    geo: str = Field(...)
    frame_id: int = Field(...)
    is_fake: bool = Field(...)
    confidence: float = Field(...)
    bbox: List[int] = Field(...)
    timestamp: str = Field(...) # ISO format string

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}

# --- Config Model (config_collection) ---
# Used for storing watchlist and geofences
class ConfigModel(BaseModel):
    id: Optional[PyObjectId] = Field(alias='_id', default=None)
    name: str = Field(...) # e.g., "watchlist" or "geofences"
    data: dict = Field(...) # The actual config data (list for watchlist, dict for geofences)
    updated_at: datetime = Field(default_factory=datetime.now)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}