"""
Application settings loaded from environment variables.
All values can be overridden via .env file.
"""
from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # ── Database ──────────────────────────────────────────────────
    POSTGRES_USER: str = "waste_sorter"
    POSTGRES_PASSWORD: str = "G7vkP9Lm2Qx5ZrT8nBs3"
    POSTGRES_DB: str = "waste_sorter_db"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432

    @property
    def DATABASE_URL(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    # ── ML Model ──────────────────────────────────────────────────
    MODEL_PATH: str = str(Path("model/saved/yolo_final/train/weights/best.pt"))
    MODEL_VERSION: str = "yolov8n-cls-v1"
    CONFIDENCE_THRESHOLD: float = 0.5

    # ── App ───────────────────────────────────────────────────────
    SECRET_KEY: str = "pR9xT7vK2LmQ8zF4wY6cD1eA5uB"
    CORS_ORIGINS: list[str] = ["http://localhost:3000"]
    UPLOAD_DIR: str = "uploads"
    MAX_IMAGE_SIZE_MB: int = 10

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
