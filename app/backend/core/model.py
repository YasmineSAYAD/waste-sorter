"""
ML model loader — loads YOLOv8 classification model once at startup.
All prediction logic lives here to keep routes thin.
"""

import time
import json
from pathlib import Path

from ultralytics import YOLO

from app.backend.core.config import settings

# ── Constants ─────────────────────────────────────────────────────
CLASSES = [
    "battery", "cardboard", "electronic", "glass", "medical",
    "metal", "organic", "paper", "plastic", "textile", "trash",
]
BASE_DIR = Path("/app")
_label_map_path =  BASE_DIR / "model/data/splits/label_map.json"
if _label_map_path.exists():
    with open(_label_map_path, encoding="utf-8") as f:
        LABEL_MAP: dict[str, dict] = json.load(f)

# ── Singleton ─────────────────────────────────────────────────────
_model: YOLO | None = None


def load_model() -> None:
    """Load YOLO model into memory — called once at app startup."""
    global _model
    path = Path(settings.MODEL_PATH)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path}")
    _model = YOLO(str(path))


def get_model() -> YOLO:
    """Return the loaded model — raises if not initialized."""
    if _model is None:
        raise RuntimeError("Model not loaded. Call load_model() first.")
    return _model

def run_inference(image_path: str) -> dict:
    """
    Run YOLO inference on a single image.
    Returns predicted class, confidence, recyclability info, and latency.
    """
    model = get_model()

    start = time.perf_counter()
    results = model(image_path, verbose=False)[0]
    inference_ms = (time.perf_counter() - start) * 1000

    top1_idx = results.probs.top1
    predicted_class = CLASSES[top1_idx]
    confidence = float(results.probs.top1conf)

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "recyclable": LABEL_MAP[predicted_class]["recyclable"],
        "bac": LABEL_MAP[predicted_class]["bac"],
        "alt": LABEL_MAP[predicted_class]["alt"] or "",
        "waste_type": LABEL_MAP[predicted_class]["type"],
        "advice": LABEL_MAP[predicted_class]["advice"],
        "model_version": settings.MODEL_VERSION,
        "inference_ms": round(inference_ms, 2),
    }
