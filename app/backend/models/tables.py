"""
SQLAlchemy ORM models — maps Python classes to PostgreSQL tables.
Matches the agreed schema: users, images, predictions, waste_infos, waste_types.
"""

import uuid
from datetime import datetime

from sqlalchemy import Boolean, DateTime, Float, ForeignKey, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from app.backend.db.session import Base


class User(Base):
    __tablename__ = "users"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    first_name: Mapped[str] = mapped_column(String(100), nullable=False)
    last_name: Mapped[str] = mapped_column(String(100), nullable=False)
    email: Mapped[str] = mapped_column(String(255), unique=True, nullable=False)
    password: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(50), default="user")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    images: Mapped[list["Image"]] = relationship("Image", back_populates="user")


class WasteType(Base):
    __tablename__ = "waste_types"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    label_key: Mapped[str] = mapped_column(String(100), unique=True, nullable=False)

    waste_infos: Mapped[list["WasteInfo"]] = relationship(
        "WasteInfo", back_populates="waste_type"
    )


class WasteInfo(Base):
    __tablename__ = "waste_infos"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    type_name: Mapped[str] = mapped_column(String(100), nullable=False)
    recyclable: Mapped[bool] = mapped_column(Boolean, nullable=False)
    bac: Mapped[str | None] = mapped_column(String(100))  # which bin to use
    alt: Mapped[str | None] = mapped_column(String(255))  # alternative disposal
    advice: Mapped[str | None] = mapped_column(Text)  # user-facing advice
    waste_type_id: Mapped[uuid.UUID] = mapped_column(
        ForeignKey("waste_types.id"), nullable=False
    )

    waste_type: Mapped["WasteType"] = relationship(
        "WasteType", back_populates="waste_infos"
    )
    images: Mapped[list["Image"]] = relationship(
        "Image", back_populates="waste_info"
    )


class Prediction(Base):
    __tablename__ = "predictions"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    confidence_score: Mapped[float] = mapped_column(Float, nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    image: Mapped["Image"] = relationship(
        "Image", back_populates="prediction", uselist=False
    )


class Image(Base):
    __tablename__ = "images"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    image_path: Mapped[str] = mapped_column(String(500), nullable=False)
    uploaded_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now()
    )

    user_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("users.id")
    )
    waste_info_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("waste_infos.id")
    )
    prediction_id: Mapped[uuid.UUID | None] = mapped_column(
        ForeignKey("predictions.id")
    )

    user: Mapped["User | None"] = relationship("User", back_populates="images")
    waste_info: Mapped["WasteInfo | None"] = relationship(
        "WasteInfo", back_populates="images"
    )
    prediction: Mapped["Prediction | None"] = relationship(
        "Prediction", back_populates="image"
    )
