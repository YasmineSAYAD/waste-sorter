"""
Pydantic schemas — request/response validation and OpenAPI documentation.
Each schema maps to a route's expected input or output shape.
"""

import uuid
from datetime import datetime

from pydantic import BaseModel, Field


# ── Users ─────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    first_name: str = Field(..., min_length=1, max_length=100)
    last_name: str = Field(..., min_length=1, max_length=100)
    email: str = Field(..., pattern=r"^[\w\.-]+@[\w\.-]+\.\w+$")
    password: str = Field(..., min_length=8)
    role: str = Field(default="user")
    class Config:
        json_schema_extra = {
            "example": {
                "first_name": "Marie",
                "last_name": "Dupont",
                "email": "marie.dupont@email.com",
                "password": "motdepasse123",
                "role": "user"
            }
        }


class UserOut(BaseModel):
    id: uuid.UUID
    first_name: str
    last_name: str
    email: str
    role: str
    created_at: datetime

    model_config = {"from_attributes": True}

class UserLogin(BaseModel):
    email: str
    password: str


class TokenOut(BaseModel):
    access_token: str
    token_type: str
    user: dict

class UserUpdate(BaseModel):
    first_name: str | None = None
    last_name: str | None = None
    email: str | None = None
    password: str | None = None

# ── Predictions ───────────────────────────────────────────────────

class PredictionOut(BaseModel):
    id: uuid.UUID
    confidence_score: float
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionResult(BaseModel):
    """Full inference result returned after image upload."""
    predicted_class: str
    confidence: float
    recyclable: bool
    bac: str
    alt: str
    waste_type: str
    advice: str
    model_version: str
    inference_ms: float
    image_id: uuid.UUID
    prediction_id: uuid.UUID


# ── Images ────────────────────────────────────────────────────────

class ImageOut(BaseModel):
    id: uuid.UUID
    image_path: str
    uploaded_at: datetime
    user_id: uuid.UUID | None
    prediction_id: uuid.UUID | None

    model_config = {"from_attributes": True}


# ── Waste ─────────────────────────────────────────────────────────

class WasteTypeOut(BaseModel):
    id: uuid.UUID
    label_key: str

    model_config = {"from_attributes": True}


class WasteInfoOut(BaseModel):
    id: uuid.UUID
    type_name: str
    recyclable: bool
    bac: str | None
    alt: str | None
    advice: str | None
    waste_type: WasteTypeOut

    model_config = {"from_attributes": True}


# ── Health ────────────────────────────────────────────────────────

class HealthOut(BaseModel):
    status: str
    version: str
