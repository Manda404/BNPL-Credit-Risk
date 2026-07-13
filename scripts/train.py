#!/usr/bin/env python3
"""Thin entry point: `python scripts/train.py [--model-config PATH]`.
Equivalent to `poetry run bnpl-risk train`."""

from __future__ import annotations

import typer

from bnpl_credit_risk.cli import train

if __name__ == "__main__":
    typer.run(train)
