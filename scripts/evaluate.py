#!/usr/bin/env python3
"""Thin entry point: `python scripts/evaluate.py [--model-version latest]`.
Equivalent to `poetry run bnpl-risk evaluate`."""

from __future__ import annotations

import typer

from bnpl_credit_risk.cli import evaluate

if __name__ == "__main__":
    typer.run(evaluate)
