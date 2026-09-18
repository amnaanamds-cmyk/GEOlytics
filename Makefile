.PHONY: install dev-install services api worker test lint fmt typecheck clean

install:
	python -m pip install -e .

dev-install:
	python -m pip install -e ".[dev]"

services:
	docker compose up -d

api:
	uvicorn geolytics.api.app:app --reload --app-dir backend

worker:
	arq geolytics.tasks.queue.WorkerSettings

test:
	pytest -q

lint:
	ruff check backend tests

fmt:
	ruff format backend tests
	ruff check --fix backend tests

typecheck:
	mypy

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache **/__pycache__
