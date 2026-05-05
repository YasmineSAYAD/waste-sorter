"""
Images router — handles image upload and triggers ML inference.
POST /api/v1/images/upload  →  saves image, runs prediction, stores result.
GET  /api/v1/images/{id}    →  returns image metadata.
"""

import shutil
import uuid
import os
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile
from starlette.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select

from app.backend.api.schemas.schemas import ImageOut, PredictionResult
from app.backend.core.config import settings
from app.backend.core.model import run_inference
from app.backend.db.session import get_db
from app.backend.models.tables import Image, Prediction, WasteInfo

router = APIRouter()

UPLOAD_DIR = Path(settings.UPLOAD_DIR)
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


@router.post("/upload", response_model=PredictionResult, summary="Upload image and classify waste")
async def upload_image(
    user_id: str,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload a waste image and get an AI classification result.

    - Saves the image to disk
    - Runs YOLOv8 inference
    - Stores prediction and image record in PostgreSQL
    - Returns class, confidence, recyclability info, and latency
    """
    # Validate file type
    if file.content_type not in ALLOWED_TYPES:
        raise HTTPException(status_code=400, detail=f"File type not allowed: {file.content_type}")

    # Validate file size
    contents = await file.read()
    if len(contents) > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large")

    # Save image to disk
    image_id = uuid.uuid4()
    suffix = Path(file.filename or "image.jpg").suffix
    image_path = UPLOAD_DIR / f"{image_id}{suffix}"
    with open(image_path, "wb") as f:
        f.write(contents)

    # Run ML inference
    result = run_inference(str(image_path))

    # Find matching WasteInfo from DB
    stmt = select(WasteInfo).join(WasteInfo.waste_type).where(
        WasteInfo.type_name == result["waste_type"]
    )
    waste_info = (await db.execute(stmt)).scalar_one_or_none()

    # Store prediction
    prediction = Prediction(confidence_score=result["confidence"])
    db.add(prediction)
    await db.flush()

    # Store image record
    image = Image(
        id=image_id,
        image_path=str(image_path),
        user_id = user_id,
        prediction_id=prediction.id,
        waste_info_id=waste_info.id if waste_info else None,
    )
    db.add(image)
    await db.commit()

    return PredictionResult(
        **result,
        image_id=image_id,
        prediction_id=prediction.id,
    )

@router.get("/", response_model=list[ImageOut], summary="Get all images")
async def get_all_images(db: AsyncSession = Depends(get_db)):
    """Return metadata for all uploaded images."""
    result = await db.execute(select(Image))
    return result.scalars().all()

@router.get("/{image_id}", response_model=ImageOut, summary="Get image metadata")
async def get_image(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Return metadata for a previously uploaded image."""
    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")
    return image

@router.delete("/{image_id}", status_code=204, summary="Delete image")
async def delete_image(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):

    image = await db.get(Image, image_id)
    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    # Delete associated predictions
    if image.prediction_id:
        prediction = await db.get(Prediction, image.prediction_id)
        if prediction:
            await db.delete(prediction)

    await db.delete(image)
    await db.commit()

@router.get("/{image_id}/file", summary="Get image file by ID")
async def get_image_file(image_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Return the image file based on its ID."""
    
    result = await db.execute(select(Image).where(Image.id == image_id))
    image = result.scalar_one_or_none()

    if not image:
        raise HTTPException(status_code=404, detail="Image not found")

    full_image_path = f"/app/{image.image_path}"

    if not os.path.exists(full_image_path):
        raise HTTPException(status_code=404, detail="Image file not found")

    return FileResponse(full_image_path)
