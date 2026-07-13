"""Thin pipeline entry point wrapping `inference.batch.BatchPredictor`, kept
alongside the other pipelines for a consistent CLI/scripts surface."""

from __future__ import annotations

from pathlib import Path

from bnpl_credit_risk.inference.batch import BatchInferenceResult, BatchPredictor
from bnpl_credit_risk.settings import ProjectConfig, Settings


def run_batch_inference_pipeline(
    config: ProjectConfig,
    settings: Settings,
    *,
    input_path: str | Path,
    output_path: str | Path,
    model_version: str = "latest",
) -> BatchInferenceResult:
    return BatchPredictor(config, settings).run(input_path, output_path, model_version=model_version)
