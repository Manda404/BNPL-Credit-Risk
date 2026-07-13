from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from bnpl_credit_risk.settings import Settings, load_config, project_root

FIXTURES_DIR = Path(__file__).parent / "fixtures"


@pytest.fixture
def sample_csv_path() -> Path:
    return FIXTURES_DIR / "sample_bnpl.csv"


@pytest.fixture
def sample_df(sample_csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(sample_csv_path)
    df["transaction_date"] = pd.to_datetime(df["transaction_date"])
    return df


@pytest.fixture
def project_config():
    """Real project config (configs/*.yaml) — the fixture CSV is built to satisfy
    the same schema, so validation/feature rules apply unchanged."""
    return load_config(project_root() / "configs")


@pytest.fixture
def test_settings(tmp_path: Path, sample_csv_path: Path) -> Settings:
    """Settings pointed at an isolated tmp_path so tests never write into the
    real artifacts/ or data/ directories."""
    return Settings(
        data_raw_path=str(sample_csv_path),
        artifacts_dir=str(tmp_path / "artifacts"),
        logs_dir=str(tmp_path / "logs"),
        random_seed=42,
    )


@pytest.fixture
def test_config(project_config, tmp_path: Path):
    """project_config with pipeline-output paths redirected into tmp_path and a
    small cross_validation fold count (fixture only has 200 rows). The raw
    dataset path and artifacts dir are environment concerns, overridden via
    `test_settings` instead — see docs/architecture.md#configuration-flow."""
    base = project_config.base.model_copy(
        update={
            "paths": project_config.base.paths.model_copy(
                update={
                    "data_reports": str(tmp_path / "reports"),
                    "data_processed": str(tmp_path / "processed"),
                }
            )
        }
    )
    model = project_config.model.model_copy(
        update={"cross_validation": project_config.model.cross_validation.model_copy(update={"n_splits": 3})}
    )
    return project_config.model_copy(update={"base": base, "model": model})
