"""Typer CLI — `bnpl-risk <command>`.

Every command: resolves config, configures Loguru, runs one pipeline, and
maps outcomes to process exit codes (0 success, 1 data validation failure,
2 technical/config error) so it composes cleanly with cron / CI / Airflow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from loguru import logger
from rich.console import Console

from bnpl_credit_risk.exceptions import BNPLError, ConfigError, DataValidationError
from bnpl_credit_risk.logging import configure_logging
from bnpl_credit_risk.pipelines.batch_inference_pipeline import run_batch_inference_pipeline
from bnpl_credit_risk.pipelines.evaluation_pipeline import run_evaluation_pipeline
from bnpl_credit_risk.pipelines.training_pipeline import run_training_pipeline
from bnpl_credit_risk.pipelines.validation_pipeline import run_validation_pipeline
from bnpl_credit_risk.settings import ProjectConfig, Settings, get_settings, load_config

app = typer.Typer(add_completion=False, help="BNPL credit risk scoring pipelines")
console = Console()


def _bootstrap(
    config_dir: Path | None, model_config: Path | None = None, inference_config: Path | None = None
) -> tuple[Settings, ProjectConfig]:
    settings = get_settings()
    config = load_config(
        config_dir, model_config_path=model_config, inference_config_path=inference_config
    )
    configure_logging(settings, config.logging)
    return settings, config


@app.command()
def validate(
    input: Annotated[Path | None, typer.Option(help="CSV to validate. Defaults to configs/data.yaml raw path.")] = None,
    config_dir: Annotated[Path | None, typer.Option(help="Override configs/ directory")] = None,
) -> None:
    """Run the standalone data validation pipeline."""
    try:
        settings, config = _bootstrap(config_dir)
        result = run_validation_pipeline(config, settings, input_path=input)
    except BNPLError as exc:
        logger.bind(pipeline="cli").error("validate failed: {}", exc)
        console.print(f"[red]Validation error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    if not result.success:
        console.print(f"[red]Data contract violated:[/red] {result.error}")
        raise typer.Exit(code=1)

    console.print(f"[green]OK[/green] valid_rows={len(result.validation.valid)} invalid_rows={len(result.validation.invalid)}")
    raise typer.Exit(code=0)


@app.command()
def train(
    model_config: Annotated[Path | None, typer.Option(help="Override configs/model.yaml (e.g. configs/model_behavioral_risk.yaml)")] = None,
    config_dir: Annotated[Path | None, typer.Option(help="Override configs/ directory")] = None,
) -> None:
    """Train a model end to end and persist a versioned artifact bundle."""
    try:
        settings, config = _bootstrap(config_dir, model_config=model_config)
        result = run_training_pipeline(config, settings)
    except DataValidationError as exc:
        console.print(f"[red]Training aborted — data contract violation:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except (ConfigError, BNPLError) as exc:
        logger.bind(pipeline="cli").exception("train failed")
        console.print(f"[red]Training error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    console.print(
        f"[green]Model trained[/green] version={result.version} "
        f"roc_auc={result.test_metrics['roc_auc']:.4f} brier={result.test_metrics['brier_score']:.4f} "
        f"threshold={result.threshold['threshold']:.3f} -> {result.version_dir}"
    )
    raise typer.Exit(code=0)


@app.command()
def evaluate(
    model_version: Annotated[str, typer.Option(help="Model version to evaluate, or 'latest'")] = "latest",
    config_dir: Annotated[Path | None, typer.Option(help="Override configs/ directory")] = None,
) -> None:
    """Re-evaluate a saved model artifact on the (deterministically rebuilt) test split."""
    try:
        settings, config = _bootstrap(config_dir)
        report = run_evaluation_pipeline(config, settings, model_version=model_version)
    except BNPLError as exc:
        logger.bind(pipeline="cli").error("evaluate failed: {}", exc)
        console.print(f"[red]Evaluation error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    console.print(f"[green]Evaluation complete[/green] roc_auc={report['roc_auc']:.4f} pr_auc={report['pr_auc']:.4f} brier={report['brier_score']:.4f}")
    raise typer.Exit(code=0)


@app.command(name="predict-batch")
def predict_batch(
    input: Annotated[Path, typer.Option(help="CSV of applications to score")],
    output: Annotated[Path, typer.Option(help="Where to write predictions")],
    model_version: Annotated[str, typer.Option(help="Model version to use, or 'latest'")] = "latest",
    config_dir: Annotated[Path | None, typer.Option(help="Override configs/ directory")] = None,
) -> None:
    """Score a CSV of new applications in batch and write predictions atomically."""
    try:
        settings, config = _bootstrap(config_dir)
        result = run_batch_inference_pipeline(
            config, settings, input_path=input, output_path=output, model_version=model_version
        )
    except DataValidationError as exc:
        console.print(f"[red]Batch inference aborted — data contract violation:[/red] {exc}")
        raise typer.Exit(code=1) from exc
    except BNPLError as exc:
        logger.bind(pipeline="cli").exception("predict-batch failed")
        console.print(f"[red]Batch inference error:[/red] {exc}")
        raise typer.Exit(code=2) from exc

    console.print(
        f"[green]Scored {result.n_scored} row(s)[/green] rejected={result.n_rejected} "
        f"output={result.output_path} risk_bands={result.risk_band_distribution}"
    )
    raise typer.Exit(code=0)


if __name__ == "__main__":
    app()
