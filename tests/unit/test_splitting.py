from __future__ import annotations

import pandas as pd

from bnpl_credit_risk.data.splitting import get_splitter
from bnpl_credit_risk.settings import SplitConfig


def test_stratified_random_preserves_class_ratio(sample_df, project_config):
    config = SplitConfig(strategy="stratified_random", test_size=0.2, random_seed=42)
    splitter = get_splitter(
        "stratified_random", config, id_column="user_id", date_column="transaction_date", target_column="default_flag"
    )
    train_df, test_df = splitter.split(sample_df)
    overall_rate = sample_df["default_flag"].mean()
    assert abs(train_df["default_flag"].mean() - overall_rate) < 0.1
    assert abs(test_df["default_flag"].mean() - overall_rate) < 0.15


def test_group_by_user_never_splits_a_user_across_sets(sample_df, project_config):
    # duplicate every user_id twice so the grouping constraint is actually exercised
    duplicated = pd.concat([sample_df, sample_df], ignore_index=True)
    config = SplitConfig(strategy="group_by_user", test_size=0.2, random_seed=42)
    splitter = get_splitter(
        "group_by_user", config, id_column="user_id", date_column="transaction_date", target_column="default_flag"
    )
    train_df, test_df = splitter.split(duplicated)
    train_ids = set(train_df["user_id"])
    test_ids = set(test_df["user_id"])
    assert not (train_ids & test_ids)


def test_time_based_split_orders_by_date(sample_df, project_config):
    config = SplitConfig(strategy="time_based", test_size=0.2, random_seed=42)
    splitter = get_splitter(
        "time_based", config, id_column="user_id", date_column="transaction_date", target_column="default_flag"
    )
    train_df, test_df = splitter.split(sample_df)
    assert train_df["transaction_date"].max() <= test_df["transaction_date"].min()
    assert len(train_df) + len(test_df) == len(sample_df)
