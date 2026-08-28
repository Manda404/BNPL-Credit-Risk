"""ArtifactBundle — saves/loads a complete inference-ready artifact.

A single `pipeline.joblib` holds feature engineering + encoding + model (and
the calibrator, when enabled) as one fitted `sklearn` object, so training and
inference can never drift apart. It's accompanied by metadata, metrics, the
selected decision threshold, and the ordered feature schema — everything
`inference/predictor.py` needs, and everything a reviewer needs to audit a
model version without retraining it.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
from loguru import logger
from sklearn.pipeline import Pipeline

from bnpl_credit_risk.constants import (
    FEATURE_SCHEMA_FILENAME,
    LATEST_POINTER_FILENAME,
    METADATA_FILENAME,
    METRICS_FILENAME,
    MODEL_ARTIFACT_FILENAME,
    MODEL_CARD_FILENAME,
    THRESHOLD_FILENAME,
)
from bnpl_credit_risk.exceptions import ArtifactNotFoundError


@dataclass
class ArtifactBundle:
    pipeline: Any
    metadata: dict[str, Any]
    metrics: dict[str, Any]
    threshold: dict[str, Any]
    feature_schema: dict[str, Any]
    model_card: str = ""
    version: str = field(default="")

    def save(self, models_dir: Path, *, set_as_latest: bool = True) -> Path:
        version = self.version or datetime.now().strftime("%Y-%m-%d_%H%M%S")
        version_dir = models_dir / version
        version_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.pipeline, version_dir / MODEL_ARTIFACT_FILENAME)
        (version_dir / METADATA_FILENAME).write_text(json.dumps(self.metadata, indent=2, default=str))
        (version_dir / METRICS_FILENAME).write_text(json.dumps(self.metrics, indent=2, default=str))
        (version_dir / THRESHOLD_FILENAME).write_text(json.dumps(self.threshold, indent=2, default=str))
        (version_dir / FEATURE_SCHEMA_FILENAME).write_text(
            json.dumps(self.feature_schema, indent=2, default=str)
        )
        if self.model_card:
            (version_dir / MODEL_CARD_FILENAME).write_text(self.model_card)

        logger.bind(pipeline="models.persistence").info(
            "Saved model artifact version={} dir={}", version, version_dir
        )

        if set_as_latest:
            promote_model_version(models_dir, version)

        return version_dir

    @staticmethod
    def load(version_dir: Path) -> ArtifactBundle:
        if not version_dir.exists():
            raise ArtifactNotFoundError(f"Model artifact directory not found: {version_dir}")
        pipeline_path = version_dir / MODEL_ARTIFACT_FILENAME
        if not pipeline_path.exists():
            raise ArtifactNotFoundError(f"Missing {MODEL_ARTIFACT_FILENAME} in {version_dir}")

        pipeline: Pipeline = joblib.load(pipeline_path)
        metadata = _read_json(version_dir / METADATA_FILENAME)
        metrics = _read_json(version_dir / METRICS_FILENAME)
        threshold = _read_json(version_dir / THRESHOLD_FILENAME)
        feature_schema = _read_json(version_dir / FEATURE_SCHEMA_FILENAME)
        model_card_path = version_dir / MODEL_CARD_FILENAME
        model_card = model_card_path.read_text() if model_card_path.exists() else ""

        return ArtifactBundle(
            pipeline=pipeline,
            metadata=metadata,
            metrics=metrics,
            threshold=threshold,
            feature_schema=feature_schema,
            model_card=model_card,
            version=version_dir.name,
        )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise ArtifactNotFoundError(f"Missing artifact file: {path}")
    return json.loads(path.read_text())


def promote_model_version(models_dir: Path, version: str) -> Path:
    """Atomically move the relocatable ``latest`` pointer to one saved version."""
    version_dir = models_dir / version
    if not version_dir.exists():
        raise ArtifactNotFoundError(f"Cannot promote missing model version: {version_dir}")
    pointer_path = models_dir / LATEST_POINTER_FILENAME
    temporary_path = models_dir / f".{LATEST_POINTER_FILENAME}.tmp"
    temporary_path.write_text(json.dumps({"version": version}, indent=2))
    temporary_path.replace(pointer_path)
    return pointer_path


def current_git_commit() -> str | None:
    """Best-effort git commit hash for traceability. Returns None outside a git repo."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=False
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return None
