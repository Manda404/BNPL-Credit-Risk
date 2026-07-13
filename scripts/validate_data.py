#!/usr/bin/env python3
"""Thin entry point for cron/CI: `python scripts/validate_data.py [--input PATH]`.

Equivalent to `poetry run bnpl-risk validate` — kept as a plain script for
environments that invoke Python scripts directly rather than through Poetry.
No logic lives here: it just hands CLI args to the same Typer command.
"""

from __future__ import annotations

import typer

from bnpl_credit_risk.cli import validate

if __name__ == "__main__":
    typer.run(validate)
