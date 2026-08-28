from __future__ import annotations

import pandas as pd
import pytest

from bnpl_credit_risk.constants import VALIDATION_QUARANTINE, VALIDATION_STRICT, VALIDATION_WARN
from bnpl_credit_risk.data.cleaning import BNPLDataCleaner
from bnpl_credit_risk.data.loaders import BNPLDataLoader
from bnpl_credit_risk.data.schemas import inference_input_schema, training_input_schema
from bnpl_credit_risk.data.validation import BNPLDataValidator
from bnpl_credit_risk.exceptions import ConfigError, DataValidationError
from bnpl_credit_risk.settings import Settings


def test_loader_parses_transaction_date(sample_csv_path, project_config):
    settings = Settings(data_raw_path=str(sample_csv_path))
    df = BNPLDataLoader(settings, project_config.data).load_raw()
    assert pd.api.types.is_datetime64_any_dtype(df["transaction_date"])
    assert len(df) == 200


def test_loader_raises_on_missing_file(project_config):
    settings = Settings(data_raw_path="data/raw/does_not_exist.csv")
    with pytest.raises(ConfigError):
        BNPLDataLoader(settings, project_config.data).load_raw()


def test_cleaner_strips_categorical_whitespace(project_config):
    df = pd.DataFrame({"employment_type": [" Salaried ", "Student"], "product_category": ["Electronics", "Fashion"]})
    cleaned = BNPLDataCleaner(project_config.data).clean(df)
    assert cleaned["employment_type"].tolist() == ["Salaried", "Student"]


def test_cleaner_drops_exact_duplicate_rows(project_config):
    df = pd.DataFrame({"age": [30, 30], "employment_type": ["Salaried", "Salaried"]})
    cleaned = BNPLDataCleaner(project_config.data).clean(df)
    assert len(cleaned) == 1


def test_inference_schema_excludes_target(project_config):
    schema = inference_input_schema(
        project_config.data,
        project_config.features,
        project_config.model.risk_scope,
    )
    assert schema.target_column is None
    assert not set(project_config.features.leaky_raw_columns).intersection(
        schema.required_columns
    )
    assert project_config.data.target_column not in schema.required_columns


def test_training_schema_requires_target(project_config):
    schema = training_input_schema(project_config.data)
    assert schema.target_column == project_config.data.target_column
    assert project_config.data.target_column in schema.required_columns


def test_validator_raises_on_missing_required_column(project_config):
    schema = training_input_schema(project_config.data)
    df = pd.DataFrame({"age": [30]})  # almost everything missing
    validator = BNPLDataValidator(schema, VALIDATION_STRICT)
    with pytest.raises(DataValidationError):
        validator.validate(df)


def test_validator_strict_raises_on_out_of_bounds_row(sample_df, project_config):
    schema = training_input_schema(project_config.data)
    bad = sample_df.copy()
    bad.loc[0, "age"] = 5  # below configured min of 18
    validator = BNPLDataValidator(schema, VALIDATION_STRICT)
    with pytest.raises(DataValidationError):
        validator.validate(bad)


def test_validator_warn_keeps_all_rows(sample_df, project_config):
    schema = training_input_schema(project_config.data)
    bad = sample_df.copy()
    bad.loc[0, "age"] = 5
    validator = BNPLDataValidator(schema, VALIDATION_WARN)
    result = validator.validate(bad)
    assert len(result.valid) == len(bad)
    assert len(result.invalid) == 0


def test_validator_quarantine_splits_rows(sample_df, project_config):
    schema = training_input_schema(project_config.data)
    bad = sample_df.copy()
    bad.loc[0, "age"] = 5
    validator = BNPLDataValidator(schema, VALIDATION_QUARANTINE)
    result = validator.validate(bad)
    assert len(result.valid) == len(bad) - 1
    assert len(result.invalid) == 1


def test_validator_flags_duplicate_id(sample_df, project_config):
    schema = training_input_schema(project_config.data)
    dup = pd.concat([sample_df.iloc[[0]], sample_df], ignore_index=True)
    validator = BNPLDataValidator(schema, VALIDATION_QUARANTINE)
    result = validator.validate(dup)
    assert len(result.invalid) >= 1


def test_validator_flags_non_binary_target(sample_df, project_config):
    schema = training_input_schema(project_config.data)
    bad = sample_df.copy()
    bad.loc[0, "default_flag"] = 2
    validator = BNPLDataValidator(schema, VALIDATION_QUARANTINE)
    result = validator.validate(bad)
    assert len(result.invalid) >= 1
