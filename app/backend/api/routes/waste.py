"""Waste router — expose waste types and waste info records."""

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.future import select
from sqlalchemy.orm import selectinload

from app.backend.api.schemas.schemas import WasteInfoOut, WasteTypeOut
from app.backend.db.session import get_db
from app.backend.models.tables import WasteInfo, WasteType

router = APIRouter()


@router.get("/types", response_model=list[WasteTypeOut], summary="List all waste types")
async def list_waste_types(db: AsyncSession = Depends(get_db)):
    """Return all waste type label keys."""
    result = await db.execute(select(WasteType))
    return result.scalars().all()


@router.get("/infos", response_model=list[WasteInfoOut], summary="List all waste infos")
async def list_waste_infos(db: AsyncSession = Depends(get_db)):
    """Return all waste info records with recyclability and bin advice."""
    result = await db.execute(
        select(WasteInfo).options(selectinload(WasteInfo.waste_type))
    )
    return result.scalars().all()
