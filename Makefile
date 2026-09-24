.PHONY: install dev-install services services-native api worker test test-unit test-integration lint fmt typecheck clean

install:
	python -m pip install -e .

dev-install:
	python -m pip install -e ".[dev]"

services:
	docker compose up -d

# For machines without a Docker daemon (or behind a registry rate limit).
# See docs/running.md for the full recipe.
services-native:
	@echo "See docs/running.md -- installs postgres/redis from your package"
	@echo "manager and runs the Qdrant release binary."

api:
	uvicorn geolytics.api.app:app --reload --app-dir backend

worker:
	arq geolytics.tasks.queue.WorkerSettings

test:
	pytest -q

test-unit:
	pytest tests -q --ignore=tests/integration

# Service-backed tests. Each one skips itself when its service is not running,
# so this is safe to run with nothing up -- it just does less.
test-integration:
	pytest tests/integration -q -rs

lint:
	ruff check backend tests

fmt:
	ruff format backend tests
	ruff check --fix backend tests

typecheck:
	mypy

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
