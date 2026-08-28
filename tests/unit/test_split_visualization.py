from __future__ import annotations

import matplotlib.pyplot as plt
import pytest

from bnpl_credit_risk.data.partitioning import stratified_train_test_split
from bnpl_credit_risk.exceptions import DataValidationError
from bnpl_credit_risk.visualization.splitting import plot_split_target_distribution


def test_plot_split_target_distribution_compares_both_splits(sample_df, tmp_path):
    split = stratified_train_test_split(sample_df, target_column="default_flag")
    output_path = tmp_path / "split_distribution.png"

    figure = plot_split_target_distribution(
        split.train,
        split.test,
        target_column="default_flag",
        size=(15, 6),
        save_path=output_path,
    )

    assert output_path.exists()
    assert len(figure.axes) == 1
    assert tuple(figure.get_size_inches()) == (15.0, 6.0)
    assert len(figure.axes[0].patches) == 4
    assert [label.get_text().splitlines()[0] for label in figure.axes[0].get_xticklabels()] == [
        "Train",
        "Test",
    ]
    assert [text.get_text() for text in figure.axes[0].get_legend().get_texts()] == [
        "Class 0",
        "Class 1",
    ]
    assert figure.axes[0].get_title().startswith("Target Distribution — default_flag")
    plt.close(figure)


def test_plot_split_target_distribution_rejects_missing_target(sample_df):
    split = stratified_train_test_split(sample_df, target_column="default_flag")

    with pytest.raises(DataValidationError, match="missing"):
        plot_split_target_distribution(
            split.train.drop(columns=["default_flag"]),
            split.test,
            target_column="default_flag",
        )
