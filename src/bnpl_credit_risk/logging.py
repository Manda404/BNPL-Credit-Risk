"""Centralized Loguru configuration.

Call `configure_logging()` once, near the start of every entry point (CLI
commands, scripts, pipelines). Components elsewhere simply
`from loguru import logger` and log — they never configure sinks themselves,
and they never use `print()`.
"""

from __future__ import annotations

import sys

from loguru import logger

from bnpl_credit_risk.settings import LoggingConfig, Settings, project_root

_configured = False


def configure_logging(settings: Settings, logging_config: LoggingConfig) -> None:
    """Reset Loguru sinks according to configs/logging.yaml + environment overrides."""
    global _configured

    logger.remove()
    level = settings.log_level or logging_config.level

    if logging_config.console.enabled:
        logger.add(
            sys.stderr,
            level=level,
            colorize=logging_config.console.colorize,
            format=(
                "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
                "<level>{level: <8}</level> | "
                "<cyan>{extra[pipeline]}</cyan> | {message}"
            ),
        )

    if logging_config.file.enabled:
        file_path = settings.resolve(logging_config.file.path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        logger.add(
            file_path,
            level=level,
            rotation=logging_config.file.rotation,
            retention=logging_config.file.retention,
            compression=logging_config.file.compression,
            format=(
                "{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | "
                "{extra[pipeline]} | {name}:{function}:{line} | {message}"
            ),
        )

    logger.configure(extra={"pipeline": "-"})
    _configured = True


def get_pipeline_logger(pipeline_name: str):
    """Return a logger bound with a pipeline name for consistent log prefixes."""
    if not _configured:
        configure_logging(Settings(), _default_logging_config())
    return logger.bind(pipeline=pipeline_name)


def _default_logging_config() -> LoggingConfig:
    from bnpl_credit_risk.settings import _load_yaml  # local import to avoid cycle at module load

    raw = _load_yaml(project_root() / "configs" / "logging.yaml")
    return LoggingConfig.model_validate(raw)
