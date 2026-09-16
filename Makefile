.PHONY: install lint typecheck test validate split-data train evaluate predict-batch pipeline clean serve-api test-api

install:
	poetry install

lint:
	poetry run ruff check .

format:
	poetry run ruff format .

typecheck:
	poetry run mypy src

test:
	poetry run pytest

validate:
	poetry run bnpl-risk validate

split-data:
	poetry run python scripts/split_data.py

train:
	poetry run bnpl-risk train

evaluate:
	poetry run bnpl-risk evaluate

predict-batch:
	poetry run bnpl-risk predict-batch --input data/processed/applications_to_score.csv --output data/predictions/predictions.csv

pipeline: validate train evaluate

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .mypy_cache htmlcov .coverage

# FastAPI : la clé BNPL_API_API_KEY doit être configurée avant le démarrage.
serve-api:
	poetry run bnpl-api

test-api:
	poetry run python -m pytest tests/integration/test_realtime_api.py --no-cov
