"""
waste-sorter — FastAPI entry point.
Registers all routers, middleware, Prometheus metrics, and OpenAPI config.
"""
import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_fastapi_instrumentator import Instrumentator
from prometheus_fastapi_instrumentator import routing as _pfi_routing

from app.backend.api.routes import images, predictions, users, waste
from app.backend.core.config import settings
from app.backend.core.model import load_model
from app.backend.db.session import init_db

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Wait for DB to be ready, then initialize."""
    # Retry connection up to 10 times with 3s delay
    for attempt in range(10):
        try:
            await init_db()
            break
        except Exception as e:
            if attempt == 9:
                raise
            print(f"DB not ready (attempt {attempt + 1}/10) — retrying in 3s... {e}")
            await asyncio.sleep(3)

    load_model()
    yield


app = FastAPI(
    title="waste-sorter API",
    description=(
        "AI-powered waste classification — "
        "classifies images into 11 waste categories."
    ),
    version="1.0.0",
    docs_url="/docs",        # Swagger UI
    redoc_url="/redoc",      # ReDoc UI
    lifespan=lifespan,
)

# ── CORS ─────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Handler ─────────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    import traceback
    print(traceback.format_exc())  # visible in Docker logs
    return JSONResponse(
        status_code=500,
        content={
            "detail": str(exc),
            "trace": traceback.format_exc()
        },
    )

# ── Patch Prometheus routing bug (_IncludedRouter has no .path) ──
_original_get_route_name = _pfi_routing._get_route_name

def _patched_get_route_name(scope, routes):
    filtered = [r for r in routes if hasattr(r, "path")]
    return _original_get_route_name(scope, filtered)

_pfi_routing._get_route_name = _patched_get_route_name

# ── Prometheus metrics ────────────────────────────────────────────
Instrumentator().instrument(app).expose(app, endpoint="/metrics")

# ── Routers ───────────────────────────────────────────────────────
app.include_router(users.router,       prefix="/api/v1/users",       tags=["users"])
app.include_router(images.router,      prefix="/api/v1/images",      tags=["images"])
app.include_router(
    predictions.router,
    prefix="/api/v1/predictions",
    tags=["predictions"]
)
app.include_router(waste.router,       prefix="/api/v1/waste",       tags=["waste"])


@app.get("/health", tags=["health"])
async def health():
    """Health check endpoint — used by Docker and CI."""
    return {"status": "ok", "version": "1.0.0"}
