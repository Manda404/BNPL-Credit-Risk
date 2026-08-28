"""Parity check against the original notebook
(Dev/bnpl-credit-risk-eda-feature-engineering-xgboost.ipynb).

The notebook trained on ALL 17 raw columns (including the post-origination
behavioral signals we now flag as leaky — see docs/data_contract.md), with a
stratified random split. That corresponds to risk_scope=behavioral_risk +
split.strategy=stratified_random in this package. This test reproduces that
exact configuration on the real raw dataset and checks the package's
ROC-AUC lands within tolerance of a reference value obtained by re-running
the notebook's own logic (LabelEncoder + train_test_split + XGBClassifier)
line for line — see the reference computation in the PR description /
docs/notebook_migration.md.

This is intentionally NOT a bitwise-identical check: the package replaces
LabelEncoder with a persisted OneHotEncoder, which changes the exact
numeric input to XGBoost. The tolerance below is what a genuine
architecture-preserving migration allows for; a match against the *default*
application_risk config would fail this test on purpose, because that
config deliberately drops the leaky features the notebook used.
"""

from __future__ import annotations

import pandas as pd
import pytest

from bnpl_credit_risk.data.loaders import BNPLDataLoader
from bnpl_credit_risk.settings import Settings, load_config, project_root

NOTEBOOK_REFERENCE_ROC_AUC = 0.7769
ROC_AUC_TOLERANCE = 0.03
RAW_DATASET_PATH = project_root() / "data" / "raw" / "BNPL_CreditRisk_Dataset.csv"

pytestmark = pytest.mark.skipif(
    not RAW_DATASET_PATH.exists(), reason="Full raw dataset not available in this environment"
)


@pytest.fixture
def real_config():
    config = load_config(project_root() / "configs", model_config_path=project_root() / "configs" / "model_behavioral_risk.yaml")
    split = config.training.split.model_copy(update={"strategy": "stratified_random"})
    training = config.training.model_copy(update={"split": split})
    return config.model_copy(update={"training": training})


def test_raw_dataset_shape_matches_notebook():
    settings = Settings(data_raw_path=str(RAW_DATASET_PATH))
    config = load_config(project_root() / "configs")
    df = BNPLDataLoader(settings, config.data).load_raw()
    assert df.shape == (10345, 17)


def test_raw_dataset_default_rate_matches_notebook():
    df = pd.read_csv(RAW_DATASET_PATH)
    default_rate = df["default_flag"].mean()
    assert abs(default_rate - 0.3905) < 0.005


def test_engineered_features_match_notebook_set(real_config):
    from bnpl_credit_risk.features.builder import ENGINEERED_FEATURE_COLUMNS, BNPLFeatureBuilder

    settings = Settings(data_raw_path=str(RAW_DATASET_PATH))
    df = BNPLDataLoader(settings, real_config.data).load_raw()
    out = BNPLFeatureBuilder(
        real_config.features,
        risk_scope="behavioral_risk",
    ).transform(df)
    for col in ENGINEERED_FEATURE_COLUMNS:
        assert col in out.columns
    assert set(ENGINEERED_FEATURE_COLUMNS) == {
        "payment_stress",
        "installment_amount",
        "installment_burden_ratio",
        "affordability_band",
        "income_after_installment",
        "credit_score_band",
        "age_group",
        "installment_term",
        "transaction_month_sin",
        "transaction_month_cos",
        "is_high_risk",
    }


def test_behavioral_risk_stratified_split_roc_auc_within_tolerance_of_notebook(real_config, tmp_path):
    settings = Settings(
        data_raw_path=str(RAW_DATASET_PATH),
        artifacts_dir=str(tmp_path / "artifacts"),
        random_seed=42,
    )
    from bnpl_credit_risk.pipelines.training_pipeline import run_training_pipeline

    result = run_training_pipeline(real_config, settings)
    assert abs(result.test_metrics["roc_auc"] - NOTEBOOK_REFERENCE_ROC_AUC) < ROC_AUC_TOLERANCE


def test_application_risk_scope_scores_meaningfully_lower_than_behavioral(real_config, tmp_path):
    """Confirms the leakage audit empirically: dropping post-origination
    signals should measurably reduce ROC-AUC versus the notebook's full
    feature set. If this ever stops being true, the leakage audit in
    docs/data_contract.md needs to be revisited."""
    from bnpl_credit_risk.pipelines.training_pipeline import run_training_pipeline

    behavioral_settings = Settings(
        data_raw_path=str(RAW_DATASET_PATH), artifacts_dir=str(tmp_path / "behavioral"), random_seed=42
    )
    behavioral_result = run_training_pipeline(real_config, behavioral_settings)

    application_model = real_config.model.model_copy(update={"risk_scope": "application_risk"})
    application_config = real_config.model_copy(update={"model": application_model})
    application_settings = Settings(
        data_raw_path=str(RAW_DATASET_PATH), artifacts_dir=str(tmp_path / "application"), random_seed=42
    )
    application_result = run_training_pipeline(application_config, application_settings)

    assert application_result.test_metrics["roc_auc"] < behavioral_result.test_metrics["roc_auc"]
