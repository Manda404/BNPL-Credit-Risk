# Modeling

## Split strategy

Configurable in `configs/training.yaml` (`split.strategy`): `stratified_random`,
`group_by_user`, `time_based`.

**Default: `time_based`.** The dataset spans 2023-01-01 to 2024-12-31.
Training on the earliest ~80% of transactions and testing on the most recent
~20% is the only strategy that mirrors actual deployment — the model is
always used to score applications that come *after* the data it was trained
on. `stratified_random` (what the original notebook used) ignores time
entirely and can let the model implicitly learn from "future" transactions
relative to a given test row, overstating real-world performance.

`group_by_user` guarantees no `user_id` straddles the train/test boundary.
On the current dataset every `user_id` appears in exactly one row, so it is
numerically identical to `stratified_random` today — it's kept configurable
for when the data grows multiple rows per user (repeat purchases), at which
point it stops being optional.

## scale_pos_weight

Computed from the **train split only** (`ModelTrainer.compute_scale_pos_weight`).
The original notebook computed it from the full dataset before splitting,
which lets the test set's class balance leak into a training-time
hyperparameter.

## Threshold selection

Configurable in `configs/training.yaml` (`threshold.policy`): `fixed`,
`best_f1`, `min_recall`, `min_precision`, `cost_matrix`. **0.5 is never
assumed.**

The threshold is selected on **out-of-fold predictions on the train split**
(`sklearn.model_selection.cross_val_predict`, `configs/model.yaml`'s
`cross_validation.n_splits` folds) — never on the held-out test set. Picking
a threshold on the same data used to report final metrics would make those
metrics optimistic.

`cost_matrix` turns the decision into a business trade-off:
`false_negative_cost` (granting credit to someone who defaults) vs.
`false_positive_cost` (rejecting someone who would have repaid) — set these
per your actual unit economics before relying on that policy.

## Risk bands

`configs/training.yaml`'s `risk_bands` maps a probability to one of Low /
Medium / High / Very High Risk. Bounds are configurable business
parameters, not statistical artifacts — persisted into `threshold.json`
alongside the model so a given artifact version always bands the same way,
even if the config file changes later.

## Calibration

Off by default (`configs/model.yaml`'s `calibration.enabled: false`). XGBoost
probabilities are reasonably well-ranked but not necessarily well-calibrated
out of the box, and calibration triples training cost (fits `k` extra
models via `CalibratedClassifierCV`). Enable it before using predicted
probabilities directly in a cost calculation (rather than just for
ranking/banding) — `sigmoid` (Platt scaling) or `isotonic`, both
configurable. `evaluation/metrics.py`'s `brier_score` is the metric to watch
before/after enabling it; `visualization/calibration.py` plots the
before/after calibration curve when calibration is on.

## Metrics

Beyond ROC-AUC, every training and evaluation run reports: PR-AUC (Average
Precision — more informative than ROC-AUC under class imbalance), precision,
recall, F1, specificity, balanced accuracy, Brier score, log loss, KS
statistic, the full confusion matrix, the predicted-probability distribution,
metrics at a grid of thresholds, and (when a cost matrix is configured) total
business cost. See `evaluation/metrics.py` and `evaluation/business_metrics.py`.

## Cross-validation

`configs/model.yaml`'s `cross_validation` (5-fold `StratifiedKFold` by
default) runs on the **train split only**, both to report a stability
estimate (mean/std ROC-AUC, persisted to `metrics.json`) and to produce the
out-of-fold predictions used for threshold selection. The original
notebook's final CV cell ran on the full `X, y` (train + test combined) —
folding in rows already used for the reported test-set evaluation.

## Extending to another algorithm

`bnpl_credit_risk.models.factory.ModelFactory.register("catboost", build_fn)`
is the intended extension point. `ModelTrainer` and everything downstream
(persistence, evaluation, inference) is algorithm-agnostic as long as the
constructed estimator implements `fit`/`predict_proba`.
