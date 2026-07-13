.PHONY: install lint typecheck test validate train evaluate predict-batch pipeline clean

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
