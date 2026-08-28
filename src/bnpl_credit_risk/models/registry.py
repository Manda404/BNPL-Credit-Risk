"""Resolves a model_version string ("latest" or an explicit run id) to a
concrete artifact directory under artifacts/models/.
"""

from __future__ import annotations

import json
from pathlib import Path

from bnpl_credit_risk.constants import LATEST_POINTER_FILENAME
from bnpl_credit_risk.exceptions import ModelNotFoundError


def resolve_model_dir(models_dir: Path, model_version: str) -> Path:
    if model_version == "latest":
        pointer_path = models_dir / LATEST_POINTER_FILENAME
        if not pointer_path.exists():
            raise ModelNotFoundError(
                f"No trained model found: {pointer_path} does not exist. Run `bnpl-risk train` first."
            )
        pointer = json.loads(pointer_path.read_text())
        version = pointer.get("version")
        if not version:
            raise ModelNotFoundError(
                f"Invalid latest model pointer: missing 'version' in {pointer_path}"
            )
        version_dir = models_dir / version
        if not version_dir.exists():
            raise ModelNotFoundError(
                f"Latest model version '{version}' not found under {models_dir}"
            )
        return version_dir

    version_dir = models_dir / model_version
    if not version_dir.exists():
        raise ModelNotFoundError(f"Model version '{model_version}' not found under {models_dir}")
    return version_dir


def list_versions(models_dir: Path) -> list[str]:
    if not models_dir.exists():
        return []
    return sorted(
        p.name for p in models_dir.iterdir() if p.is_dir() and (p / "pipeline.joblib").exists()
    )
