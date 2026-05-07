.PHONY: up down build logs lint test mlflow db-migrate db-seed clean

# ── Docker ────────────────────────────────────────────────────────

up:
	docker compose up -d

down:
	docker compose down

build:
	docker compose build --no-cache

logs:
	docker compose logs -f backend

# ── Dev ───────────────────────────────────────────────────────────

install:
	pip install -e ".[dev]"

lint:
	ruff check app/backend/ --select E,F,W
	mypy app/backend/ --ignore-missing-imports

test:
	pytest app/backend/tests/ --cov=app/backend --cov-fail-under=70 -v

security:
	bandit -r app/backend/ -ll -x app/backend/tests/
	safety check

# ── Database ──────────────────────────────────────────────────────

db-migrate:
	docker compose exec backend alembic upgrade head

db-seed:
	docker compose exec backend python db/seed.py

# ── MLflow ───────────────────────────────────────────────────────

mlflow:
	mlflow ui --backend-store-uri model/mlruns --port 5000

# ── Cleanup ──────────────────────────────────────────────────────

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type d -name .pytest_cache -exec rm -rf {} +
	find . -name "*.pyc" -delete
