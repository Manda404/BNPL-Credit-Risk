#!/usr/bin/env python3
"""Thin entry point: `python scripts/predict_batch.py --input IN.csv --output OUT.csv`.
Equivalent to `poetry run bnpl-risk predict-batch`. This is the script a
nightly cron job / scheduled GitHub Action / Airflow task should call."""

from __future__ import annotations

import typer

from bnpl_credit_risk.cli import predict_batch

if __name__ == "__main__":
    typer.run(predict_batch)
