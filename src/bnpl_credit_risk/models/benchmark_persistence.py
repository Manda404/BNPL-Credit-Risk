"""Persistence for the development benchmark shared by notebooks 04 and 05."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import joblib

from bnpl_credit_risk.exceptions import ArtifactNotFoundError
from bnpl_credit_risk.models.boosting import BoostingBenchmarkResult

BENCHMARK_FILENAME = "benchmark.joblib"
BENCHMARK_METADATA_FILENAME = "benchmark_metadata.json"
BENCHMARK_LATEST_FILENAME = "latest.json"


class BoostingBenchmarkStore:
    """Save one fitted benchmark and resolve its latest version."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def save(self, result: BoostingBenchmarkResult) -> Path:
        version = datetime.now().strftime("%Y-%m-%d_%H%M%S")
        version_dir = self._root / version
        version_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(result, version_dir / BENCHMARK_FILENAME)
        metadata = {
            "version": version,
            "best_model": result.best_model_name,
            "selection_ranking": result.comparison.index.tolist(),
            "feature_names": result.feature_names,
        }
        (version_dir / BENCHMARK_METADATA_FILENAME).write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        self._root.mkdir(parents=True, exist_ok=True)
        (self._root / BENCHMARK_LATEST_FILENAME).write_text(
            json.dumps({"version": version}, indent=2),
            encoding="utf-8",
        )
        return version_dir

    def load(self, version: str = "latest") -> BoostingBenchmarkResult:
        version_dir = self.resolve(version)
        artifact_path = version_dir / BENCHMARK_FILENAME
        if not artifact_path.exists():
            raise ArtifactNotFoundError(f"Missing benchmark artifact: {artifact_path}")
        return joblib.load(artifact_path)

    def resolve(self, version: str = "latest") -> Path:
        if version != "latest":
            version_dir = self._root / version
        else:
            pointer_path = self._root / BENCHMARK_LATEST_FILENAME
            if not pointer_path.exists():
                raise ArtifactNotFoundError(
                    "No boosting benchmark found. Run notebook 04 and save its result first."
                )
            pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
            version_dir = self._root / pointer["version"]
        if not version_dir.exists():
            raise ArtifactNotFoundError(f"Benchmark directory not found: {version_dir}")
        return version_dir
