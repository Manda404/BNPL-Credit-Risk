"""Start the shared MLflow UI from a notebook and expose a clickable link."""

from __future__ import annotations

import html
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

from bnpl_credit_risk.settings import ProjectConfig
from bnpl_credit_risk.tracking.mlflow_tracker import MLflowTracker


@dataclass
class MLflowUIHandle:
    """Notebook-friendly handle for an existing or newly started MLflow UI."""

    url: str
    experiment_url: str
    experiment_name: str
    mlflow_version: str
    process: subprocess.Popen[bytes] | None = None

    @property
    def started_by_notebook(self) -> bool:
        return self.process is not None

    def stop(self) -> None:
        """Stop the server only when this handle started it."""
        if self.process is not None and self.process.poll() is None:
            self.process.terminate()
            self.process = None

    def _repr_html_(self) -> str:
        status = "started by this notebook" if self.started_by_notebook else "already running"
        return (
            '<div style="padding:10px 12px;border-left:4px solid #0194E2;">'
            f"<strong>MLflow central — {html.escape(self.experiment_name)}</strong><br>"
            f'<a href="{html.escape(self.experiment_url)}" target="_blank" '
            'rel="noopener noreferrer">Open the experiment in MLflow ↗</a>'
            f"<br><small>MLflow {html.escape(self.mlflow_version)} · "
            f"Server {html.escape(status)} · {html.escape(self.url)}</small>"
            "</div>"
        )


def launch_mlflow_ui(config: ProjectConfig) -> MLflowUIHandle:
    """Start the shared local MLflow server once and return its experiment link."""
    mlflow_config = config.training.mlflow
    ui_config = mlflow_config.ui
    base_url = f"http://{ui_config.host}:{ui_config.port}"
    process: subprocess.Popen[bytes] | None = None

    if not _server_is_ready(base_url):
        command = [
            sys.executable,
            "-m",
            "mlflow",
            "server",
            "--backend-store-uri",
            mlflow_config.tracking_uri,
            "--default-artifact-root",
            mlflow_config.artifact_uri,
            "--host",
            ui_config.host,
            "--port",
            str(ui_config.port),
            "--workers",
            str(ui_config.workers),
        ]
        process = subprocess.Popen(
            command,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        _wait_until_ready(
            base_url,
            process,
            timeout_seconds=ui_config.startup_timeout_seconds,
        )

    tracker = MLflowTracker(config)
    mlflow = tracker._mlflow()
    tracker._configure(mlflow)
    experiment = mlflow.get_experiment_by_name(mlflow_config.experiment_name)
    if experiment is None:
        raise RuntimeError(f"MLflow experiment was not created: {mlflow_config.experiment_name}")
    experiment_url = f"{base_url}/#/experiments/{experiment.experiment_id}"
    return MLflowUIHandle(
        url=base_url,
        experiment_url=experiment_url,
        experiment_name=experiment.name,
        mlflow_version=mlflow.__version__,
        process=process,
    )


def _wait_until_ready(
    base_url: str,
    process: subprocess.Popen[Any],
    *,
    timeout_seconds: int,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(
                "MLflow server stopped during startup. Verify the central URI and port."
            )
        if _server_is_ready(base_url):
            return
        time.sleep(0.25)
    process.terminate()
    raise TimeoutError(
        f"MLflow server did not become ready within {timeout_seconds} seconds: {base_url}"
    )


def _server_is_ready(base_url: str) -> bool:
    try:
        with urlopen(f"{base_url}/health", timeout=0.5) as response:  # noqa: S310
            return response.status == 200
    except (OSError, URLError):
        return False
