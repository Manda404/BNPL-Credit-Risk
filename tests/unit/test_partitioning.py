from __future__ import annotations

import pandas as pd
import pytest

from bnpl_credit_risk.data.partitioning import (
    save_train_test_csv,
    stratified_train_test_split,
)
from bnpl_credit_risk.exceptions import ConfigError, DataValidationError


def test_stratified_split_uses_ten_percent_for_test_by_default(sample_df):
    original = sample_df.copy(deep=True)

    split = stratified_train_test_split(sample_df, target_column="default_flag")

    assert len(split.train) == 180
    assert len(split.test) == 20
    assert len(split.train) + len(split.test) == len(sample_df)
    assert abs(split.train["default_flag"].mean() - sample_df["default_flag"].mean()) < 0.01
    assert abs(split.test["default_flag"].mean() - sample_df["default_flag"].mean()) < 0.01
    pd.testing.assert_frame_equal(sample_df, original)


def test_stratified_split_is_reproducible(sample_df):
    first = stratified_train_test_split(
        sample_df, target_column="default_flag", random_seed=17
    )
    second = stratified_train_test_split(
        sample_df, target_column="default_flag", random_seed=17
    )

    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)


@pytest.mark.parametrize("test_size", [0.0, 1.0, -0.1, 1.1])
def test_stratified_split_rejects_invalid_test_size(sample_df, test_size):
    with pytest.raises(ConfigError, match="test_size"):
        stratified_train_test_split(
            sample_df,
            target_column="default_flag",
            test_size=test_size,
        )


def test_stratified_split_rejects_missing_target(sample_df):
    with pytest.raises(DataValidationError, match="missing"):
        stratified_train_test_split(
            sample_df.drop(columns=["default_flag"]), target_column="default_flag"
        )


def test_save_train_test_csv_writes_without_index(sample_df, tmp_path):
    split = stratified_train_test_split(sample_df, target_column="default_flag")

    paths = save_train_test_csv(split.train, split.test, output_dir=tmp_path)

    saved_train = pd.read_csv(paths.train_path)
    saved_test = pd.read_csv(paths.test_path)
    assert paths.train_path == tmp_path / "train.csv"
    assert paths.test_path == tmp_path / "test.csv"
    assert "Unnamed: 0" not in saved_train.columns
    assert "Unnamed: 0" not in saved_test.columns
    assert len(saved_train) == 180
    assert len(saved_test) == 20


def test_save_train_test_csv_rejects_nested_filename(sample_df, tmp_path):
    split = stratified_train_test_split(sample_df, target_column="default_flag")

    with pytest.raises(ConfigError, match="filename"):
        save_train_test_csv(
            split.train,
            split.test,
            output_dir=tmp_path,
            train_filename="nested/train.csv",
        )
