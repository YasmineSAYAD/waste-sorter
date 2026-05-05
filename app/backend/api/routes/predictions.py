"""Predictions router — read prediction records."""

import uuid

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession

from app.backend.api.schemas.schemas import PredictionOut
from app.backend.db.session import get_db
from app.backend.models.tables import Prediction

router = APIRouter()


@router.get("/{prediction_id}", response_model=PredictionOut, summary="Get prediction by ID")
async def get_prediction(prediction_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Return a single prediction record by UUID."""
    prediction = await db.get(Prediction, prediction_id)
    if not prediction:
        raise HTTPException(status_code=404, detail="Prediction not found")
    return prediction
