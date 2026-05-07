import uuid
from fastapi import APIRouter, Depends, HTTPException
from passlib.context import CryptContext
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.backend.api.schemas.schemas import (
    TokenOut, UserCreate, UserLogin, UserOut, UserUpdate
)
from app.backend.db.session import get_db
from app.backend.models.tables import Image, Prediction, User, WasteInfo

router = APIRouter()
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

@router.post(
    "/register",
    response_model=UserOut,
    status_code=201,
    summary="Register a new user",
)
async def register(payload: UserCreate, db: AsyncSession = Depends(get_db)):
    """Create a new user account with hashed password."""
    # Check email not already taken
    existing = await db.execute(select(User).where(User.email == payload.email))
    if existing.scalar_one_or_none():
        raise HTTPException(status_code=409, detail="Email already registered")

    user = User(
        first_name=payload.first_name,
        last_name=payload.last_name,
        email=payload.email,
        password=pwd_context.hash(payload.password[:72]),
        role=payload.role,
    )
    db.add(user)
    await db.commit()
    await db.refresh(user)
    return user

@router.post(
    "/login",
    response_model=TokenOut,
    summary="Login and get access token",
)
async def login(payload: UserLogin, db: AsyncSession = Depends(get_db)):
    """Authenticate user — returns user info and a simple token."""
    result = await db.execute(select(User).where(User.email == payload.email))
    user = result.scalar_one_or_none()

    if not user or not pwd_context.verify(payload.password, user.password):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    return {
        "access_token": str(user.id),  # simple token — replace with JWT in prod
        "token_type": "bearer",
        "user": {
            "id": str(user.id),
            "first_name": user.first_name,
            "last_name": user.last_name,
            "email": user.email,
            "role": user.role,
        },
    }

@router.post("/logout", summary="Logout")
async def logout():
    """
    Logout endpoint — token invalidation is handled client-side.
    In production, use a token blacklist or JWT expiry.
    """
    return {"message": "Logged out successfully"}

@router.get("/", response_model=list[UserOut], summary="List all users (admin)")
async def list_users(db: AsyncSession = Depends(get_db)):
    """Return all registered users."""
    result = await db.execute(select(User))
    return result.scalars().all()

@router.get("/{user_id}", response_model=UserOut, summary="Get user by ID")
async def get_user(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Return a single user by UUID."""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    return user

@router.put(
    "/{user_id}",
    response_model=UserOut,
    summary="Update user profile",
)
async def update_user(
    user_id: uuid.UUID,
    payload: UserUpdate,
    db: AsyncSession = Depends(get_db),
):
    """Update first name, last name, email or password."""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    if payload.first_name is not None:
        user.first_name = payload.first_name
    if payload.last_name is not None:
        user.last_name = payload.last_name
    if payload.email is not None:
        # Check new email not taken by another user
        existing = await db.execute(
            select(User).where(User.email == payload.email, User.id != user_id)
        )
        if existing.scalar_one_or_none():
            raise HTTPException(status_code=409, detail="Email already in use")
        user.email = payload.email
    if payload.password is not None:
        if len(payload.password) < 8:
            raise HTTPException(status_code=400, detail="Password too short")
        user.password = pwd_context.hash(payload.password)

    await db.commit()
    await db.refresh(user)
    return user

@router.delete("/{user_id}", status_code=204, summary="Delete user account")
async def delete_user(user_id: uuid.UUID, db: AsyncSession = Depends(get_db)):
    """Permanently delete a user and all their associated data (RGPD)."""
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Delete associated images and predictions
    images_result = await db.execute(
        select(Image).where(Image.user_id == user_id)
    )
    for image in images_result.scalars().all():
        if image.prediction_id:
            pred = await db.get(Prediction, image.prediction_id)
            if pred:
                await db.delete(pred)
        await db.delete(image)

    await db.delete(user)
    await db.commit()

@router.get(
    "/{user_id}/history",
    summary="Get user prediction history",
)
async def get_user_history(
    user_id: uuid.UUID, db: AsyncSession = Depends(get_db)
):
    """
    Return all waste classifications performed by a user.
    Joins images → predictions → waste_infos for full context.
    """
    user = await db.get(User, user_id)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    result = await db.execute(
        select(Image)
        .where(Image.user_id == user_id)
        .options(
            selectinload(Image.prediction),
            selectinload(Image.waste_info).selectinload(WasteInfo.waste_type),
        )
        .order_by(Image.uploaded_at.desc())
    )
    images = result.scalars().all()

    history = []
    for img in images:
        history.append({
            "image_id": str(img.id),
            "image_path": img.image_path,
            "uploaded_at": img.uploaded_at.isoformat() if img.uploaded_at else None,
            "predicted_class": img.waste_info.waste_type.label_key
            if img.waste_info and img.waste_info.waste_type
            else None,
            "waste_type": img.waste_info.type_name if img.waste_info else None,
            "recyclable": img.waste_info.recyclable if img.waste_info else None,
            "bac": img.waste_info.bac if img.waste_info else None,
            "alt": img.waste_info.alt if img.waste_info else None,
            "confidence": img.prediction.confidence_score if img.prediction else None,
            "prediction_id": str(img.prediction_id) if img.prediction_id else None,
        })

    return history
