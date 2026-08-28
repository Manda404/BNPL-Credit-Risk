"""Stratified dataset partitioning and atomic CSV persistence."""

from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from loguru import logger

from bnpl_credit_risk.data.splitting import StratifiedRandomSplitter
from bnpl_credit_risk.exceptions import ConfigError, DataValidationError


@dataclass(frozen=True)
class DatasetSplit:
    """In-memory train/test partition and the parameters that produced it."""

    train: pd.DataFrame
    test: pd.DataFrame
    target_column: str
    test_size: float
    random_seed: int


@dataclass(frozen=True)
class SavedSplitPaths:
    """Locations of the two persisted CSV partitions."""

    train_path: Path
    test_path: Path


def stratified_train_test_split(
    df: pd.DataFrame,
    *,
    target_column: str,
    test_size: float = 0.10,
    random_seed: int = 42,
) -> DatasetSplit:
    """Create a reproducible split preserving the target distribution.

    The default reserves 10% of rows for testing and keeps 90% for training.
    The source dataframe is never modified.
    """
    if target_column not in df.columns:
        raise DataValidationError(f"Target column '{target_column}' is missing")
    if df.empty:
        raise DataValidationError("Cannot split an empty dataframe")
    if not 0 < test_size < 1:
        raise ConfigError("test_size must be strictly between 0 and 1")
    if df[target_column].isna().any():
        raise DataValidationError(f"Target column '{target_column}' contains missing values")

    class_counts = df[target_column].value_counts()
    if len(class_counts) < 2:
        raise DataValidationError(
            f"Target column '{target_column}' must contain at least two classes"
        )
    if int(class_counts.min()) < 2:
        raise DataValidationError(
            "Every target class must contain at least two rows for stratification"
        )

    splitter = StratifiedRandomSplitter(target_column, test_size, random_seed)
    try:
        train_df, test_df = splitter.split(df)
    except ValueError as exc:
        raise DataValidationError(f"Stratified split failed: {exc}") from exc

    train_df = train_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    log = logger.bind(pipeline="data.partitioning")
    log.info(
        "Stratified split complete train_rows={} ({:.1%}) test_rows={} ({:.1%}) seed={}",
        len(train_df),
        len(train_df) / len(df),
        len(test_df),
        len(test_df) / len(df),
        random_seed,
    )
    log.info(
        "Target distribution train={} test={}",
        _target_rates(train_df, target_column),
        _target_rates(test_df, target_column),
    )
    return DatasetSplit(
        train=train_df,
        test=test_df,
        target_column=target_column,
        test_size=test_size,
        random_seed=random_seed,
    )


def save_train_test_csv(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    output_dir: str | Path,
    train_filename: str = "train.csv",
    test_filename: str = "test.csv",
) -> SavedSplitPaths:
    """Persist train and test CSV files atomically in ``output_dir``."""
    _validate_filename(train_filename)
    _validate_filename(test_filename)
    if train_filename == test_filename:
        raise ConfigError("train_filename and test_filename must be different")

    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    train_path = destination / train_filename
    test_path = destination / test_filename
    _write_csv_atomic(train_df, train_path)
    _write_csv_atomic(test_df, test_path)

    logger.bind(pipeline="data.partitioning").info(
        "Saved dataset partitions train={} test={}", train_path, test_path
    )
    return SavedSplitPaths(train_path=train_path, test_path=test_path)


def _target_rates(df: pd.DataFrame, target_column: str) -> dict[str, float]:
    rates = df[target_column].value_counts(normalize=True).sort_index()
    return {str(label): round(float(rate), 4) for label, rate in rates.items()}


def _validate_filename(filename: str) -> None:
    if not filename or Path(filename).name != filename:
        raise ConfigError(f"Expected a filename without directories, received: {filename!r}")
    if Path(filename).suffix.lower() != ".csv":
        raise ConfigError(f"Split output must use a .csv extension: {filename!r}")


def _write_csv_atomic(df: pd.DataFrame, output_path: Path) -> None:
    fd, temporary_path = tempfile.mkstemp(
        dir=output_path.parent, prefix=f".{output_path.stem}_", suffix=".tmp"
    )
    os.close(fd)
    try:
        df.to_csv(temporary_path, index=False)
        os.replace(temporary_path, output_path)
        output_path.chmod(0o644)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)
