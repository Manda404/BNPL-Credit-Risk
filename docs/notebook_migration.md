# Notebook Migration

Source: [Dev/bnpl-credit-risk-eda-feature-engineering-xgboost.ipynb](../Dev/bnpl-credit-risk-eda-feature-engineering-xgboost.ipynb)
(kept unmodified for reference — never re-run as part of the pipeline).

## Correspondence table

| Notebook cell / step | New module | Function / Class |
|---|---|---|
| Cell 2 — `pd.read_csv(kaggle path)` + `pd.to_datetime` | `data/loaders.py` | `BNPLDataLoader.load_raw` |
| Cell 4 — shape / duplicates / nulls / dtypes health check | `data/quality.py` | `DataQualityReport` |
| Cell 5 — target distribution pie/bar | `visualization/eda.py` | `plot_target_distribution` |
| Cell 7 — age histogram, income boxplot by target | `visualization/eda.py` | `plot_numeric_by_target` |
| Cell 8 — default rate by employment / product category | `visualization/eda.py` | `plot_default_rate_by_category` |
| Cell 9 — credit score histogram, risk score by segment | `visualization/eda.py` | `plot_numeric_by_target`; the risk-score visual was retired after the leakage audit |
| Cell 10 — default rate by country / installments | `visualization/eda.py` | `plot_default_rate_by_category`, `plot_default_rate_by_installments` |
| Cell 11 — correlation heatmap | `visualization/eda.py` | `plot_correlation_heatmap` |
| Cell 12 — default rate by missed payments / delay bins | `visualization/eda.py` | Retired: post-origination timing, not a chart, establishes leakage |
| Cell 15 — original engineered features | `features/builder.py` | `BNPLFeatureBuilder`, extended with checkout-time affordability, term and cyclical seasonality features |
| Cell 15 — `LabelEncoder` loop over categoricals | `features/preprocessing.py` | `build_preprocessing_pipeline` (`ColumnTransformer` + `OneHotEncoder`) |
| Cell 17 — feature/target split, `scale_pos_weight`, `train_test_split` | `data/splitting.py`, `models/trainer.py` | `get_splitter`, `ModelTrainer.compute_scale_pos_weight` |
| Cell 17 — `XGBClassifier(...)` + `.fit(...)` | `models/factory.py`, `models/trainer.py` | `ModelFactory`, `ModelTrainer.train` |
| Cell 19 — `predict`, `predict_proba`, ROC-AUC, classification report | `evaluation/metrics.py`, `evaluation/evaluator.py` | `ClassificationEvaluator` |
| Cell 20 — confusion matrix, ROC curve plots | `visualization/evaluation.py` | `plot_confusion_matrix`, `plot_roc_curve` |
| Cell 21 — feature importance plot | `visualization/evaluation.py` | `plot_feature_importance` |
| Cell 22 — `StratifiedKFold` cross-validation | `pipelines/training_pipeline.py` | inlined, corrected to run on the train split only |
| Notebook 04 — three native boosting candidates | `models/boosting.py`, `pipelines/boosting_benchmark_pipeline.py` | `BoostingBenchmark`, `BoostingBenchmarkPipeline` |
| Notebook 04 — repeated comparison figures | `visualization/boosting.py` | `BoostingBenchmarkVisualizer` |
| Notebook 05 — calibration, threshold, lift and gains | `evaluation/diagnostics.py`, `pipelines/boosting_evaluation_pipeline.py` | `BoostingEvaluationPipeline`, `ThresholdTradeoffAnalyzer` |
| Notebook 05 — feature importance and SHAP views | `visualization/model_diagnostics.py` | `ModelDiagnosticsVisualizer` |
| Notebook 05 — publish the approved winner | `pipelines/boosting_evaluation_pipeline.py`, `models/persistence.py` | `BoostingEvaluationPipeline.publish`, `ArtifactBundle` |
| Notebook 06 — target-free batch scoring | `pipelines/batch_inference_pipeline.py`, `inference/batch.py` | `BatchInferencePipeline`, `BatchPredictor` |
| Notebook 06 — portfolio inference overview | `visualization/inference.py` | `BatchInferenceVisualizer` |
| *(absent from notebook)* | `evaluation/thresholding.py`, `evaluation/business_metrics.py` | `ThresholdSelector`, `business_cost` |
| *(absent from notebook)* | `models/calibration.py`, `visualization/calibration.py` | `ProbabilityCalibrator`, calibration curve plots |
| *(absent from notebook)* | `models/persistence.py`, `models/registry.py` | `ArtifactBundle`, `resolve_model_dir` |
| *(absent from notebook)* | `inference/` | `Predictor`, `BatchPredictor` |
| *(absent from notebook)* | `data/validation.py`, `data/schemas.py` | `BNPLDataValidator`, `TrainingInputSchema` / `InferenceInputSchema` |

## Behavior changes from the notebook, and why

| Change | Reason | How to verify |
|---|---|---|
| Kaggle-hardcoded CSV path removed | Broken outside Kaggle; violates "no hardcoded paths" | `configs/base.yaml` / `BNPL_DATA_RAW_PATH` env var |
| `LabelEncoder` (fit on full df, unpersisted) → `OneHotEncoder` (fit on train only, persisted) | Encoder must be reusable at inference and must not see test data during fit | `tests/unit/test_preprocessing.py::test_preprocessing_handles_unknown_category_at_transform_time` |
| `scale_pos_weight` computed on full `y` before split → computed on train `y` only | Avoids test-set class balance leaking into a training hyperparameter | `models/trainer.py::ModelTrainer.compute_scale_pos_weight`, called with `y_train` only in `training_pipeline.py` |
| Cross-validation over full `X, y` → over train split only, used for threshold selection via out-of-fold predictions | Prevents CV from reusing rows already held out for final evaluation | `pipelines/training_pipeline.py` |
| Fixed 0.5 decision threshold → configurable policy, selected on out-of-fold train predictions | 0.5 has no statistical justification for an imbalanced target; see `docs/modeling.md` | `evaluation/thresholding.py`, `configs/training.yaml` |
| `stratified_random` split → `time_based` split (default) | The data has a time axis (`transaction_date`); a random split lets the model implicitly see "future" rows during training | `data/splitting.py::TimeBasedSplitter`, `docs/modeling.md#split-strategy` |
| Single feature set (all 17 raw columns) → `application_risk` (default) / `behavioral_risk` (explicit) scopes | Leakage audit: `repayment_delay_days`, `missed_payments`, `risk_score`, `customer_segment` are post-origination signals unavailable at the accept/reject decision | `docs/data_contract.md#leakage-audit` |
| ROC-AUC + classification report only → full metric suite (PR-AUC, Brier, log loss, KS, calibration, business cost) | A model whose probabilities decide who gets credit needs its probabilities evaluated, not just its ranking | `evaluation/metrics.py`, `docs/modeling.md#metrics` |

## Parity verification

`tests/regression/test_notebook_parity.py` reproduces the notebook's exact
configuration (`risk_scope=behavioral_risk`, `split.strategy=stratified_random`,
same XGBoost hyperparameters, `random_state=42`) against the real raw
dataset and asserts:

- Raw shape matches (10,345 rows × 17 columns).
- Default rate matches (~39.05%).
- The configured engineered feature columns are produced.
- ROC-AUC on the held-out test split lands within 0.03 of a reference value
  (0.7769) obtained by re-running the notebook's own `LabelEncoder` +
  `train_test_split` + `XGBClassifier` logic line-for-line. The package's own
  run under the same configuration measured 0.7759 — a 0.001 gap, well
  inside tolerance; the residual difference is expected, since the package
  replaces `LabelEncoder` with a persisted `OneHotEncoder`, which changes the
  exact numeric input XGBoost sees.
- `application_risk` scores measurably lower than `behavioral_risk` on the
  same data and split — the leakage audit holds up empirically (~0.70 vs. ~0.78).
