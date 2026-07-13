"""Configurable train/test splitting strategies.

- stratified_random : sklearn train_test_split(stratify=target). What the
  original notebook does. Ignores time — can overstate performance when the
  data has a time axis, since the model may see rows temporally "after" test
  rows during training.
- group_by_user      : GroupShuffleSplit on the id column, so no group
  (user) straddles the train/test boundary. On the current dataset every
  user_id appears exactly once, so this is numerically identical to
  stratified_random today; kept for when the data grows multiple rows per
  user (repeat purchases).
- time_based (default): sort by the date column, train on the earliest rows,
  test on the most recent — the only strategy that mirrors how the model is
  actually used in production (score future applications with a model fit on
  the past).
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd
from loguru import logger
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from bnpl_credit_risk.constants import (
    SPLIT_GROUP_BY_USER,
    SPLIT_STRATIFIED_RANDOM,
    SPLIT_TIME_BASED,
)
from bnpl_credit_risk.exceptions import ConfigError
from bnpl_credit_risk.settings import SplitConfig


class BaseSplitter(ABC):
    name: str

    @abstractmethod
    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return (train_df, test_df)."""


class StratifiedRandomSplitter(BaseSplitter):
    name = SPLIT_STRATIFIED_RANDOM

    def __init__(self, target_column: str, test_size: float, random_seed: int) -> None:
        self._target_column = target_column
        self._test_size = test_size
        self._random_seed = random_seed

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        train_df, test_df = train_test_split(
            df,
            test_size=self._test_size,
            random_state=self._random_seed,
            stratify=df[self._target_column],
        )
        return train_df, test_df


class GroupByUserSplitter(BaseSplitter):
    name = SPLIT_GROUP_BY_USER

    def __init__(self, id_column: str, test_size: float, random_seed: int) -> None:
        self._id_column = id_column
        self._test_size = test_size
        self._random_seed = random_seed

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        splitter = GroupShuffleSplit(
            n_splits=1, test_size=self._test_size, random_state=self._random_seed
        )
        train_idx, test_idx = next(splitter.split(df, groups=df[self._id_column]))
        return df.iloc[train_idx], df.iloc[test_idx]


class TimeBasedSplitter(BaseSplitter):
    name = SPLIT_TIME_BASED

    def __init__(self, date_column: str, test_size: float) -> None:
        self._date_column = date_column
        self._test_size = test_size

    def split(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        ordered = df.sort_values(self._date_column, kind="mergesort")
        cutoff = int(len(ordered) * (1 - self._test_size))
        train_df = ordered.iloc[:cutoff]
        test_df = ordered.iloc[cutoff:]
        if len(train_df) and len(test_df):
            logger.bind(pipeline="data.splitting").info(
                "Time-based split: train up to {}, test from {} onward",
                train_df[self._date_column].max(),
                test_df[self._date_column].min(),
            )
        return train_df, test_df


def get_splitter(
    strategy: str, config: SplitConfig, *, id_column: str, date_column: str, target_column: str
) -> BaseSplitter:
    if strategy == SPLIT_STRATIFIED_RANDOM:
        return StratifiedRandomSplitter(target_column, config.test_size, config.random_seed)
    if strategy == SPLIT_GROUP_BY_USER:
        return GroupByUserSplitter(id_column, config.test_size, config.random_seed)
    if strategy == SPLIT_TIME_BASED:
        return TimeBasedSplitter(date_column, config.test_size)
    raise ConfigError(f"Unknown split strategy: {strategy}")
