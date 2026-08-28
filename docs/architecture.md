# Architecture

## Layout

```
configs/       YAML configuration (paths, data contract, features, model, training, inference, logging)
data/          raw / interim / processed / predictions / reports
artifacts/     models / benchmarks / metrics / figures / schemas (versioned outputs)
notebooks/     thin consumers of the package (00-06)
scripts/       plain-Python entry points for cron/CI, wrapping the CLI commands
src/bnpl_credit_risk/
├── cli.py               Typer CLI — the only place that maps pipeline outcomes to exit codes
├── settings.py           env Settings (pydantic-settings) + typed YAML config loader
├── constants.py          fixed vocabularies (risk scopes, split strategies, ...)
├── exceptions.py         BNPLError and subclasses
├── logging.py             centralized Loguru sink configuration
├── data/                  loading, schema, validation, quality, cleaning, splitting
├── features/               feature engineering + preprocessing pipeline
├── models/                 model factory, training, calibration, persistence, registry
├── evaluation/              metrics, thresholding, business cost, reports
├── visualization/           EDA / evaluation / calibration plots
├── pipelines/               orchestration: validation / training / evaluation / batch inference
├── tracking/                MLflow experiment tracking for comparison and approval
└── inference/                Predictor (core), batch, realtime
tests/
├── unit/          one module, one behavior at a time
├── integration/    pipelines run end-to-end against a small fixture dataset
└── regression/      parity against the original notebook, on the real dataset
```

## Module responsibilities

- **`data`** answers "is this dataframe safe to use, and in what shape?" —
  including reusable profiling and stratified train/test export. Nothing here
  trains anything or computes a prediction.
- **`features`** answers "given a safe dataframe, what does the model see?" —
  `BNPLFeatureBuilder` is pure pandas (deterministic, no fitted state);
  `preprocessing.build_preprocessing_pipeline` wraps it plus a
  `ColumnTransformer` into one fittable `sklearn.Pipeline`.
- **`models`** answers "how do we fit, compare and persist estimators?" —
  `ModelTrainer` owns the production sklearn pipeline, while
  `BoostingBenchmark` compares XGBoost, LightGBM and CatBoost with native
  categorical support. `BoostingBenchmarkStore` persists the development
  hand-off between notebooks `04` and `05`.
- **`evaluation`** answers "how good is this pipeline, and at what threshold?"
  — pure functions over `(y_true, y_prob)`, including calibration bins,
  threshold trade-offs, lift and cumulative gains; no I/O or side effects.
- **`visualization`** turns evaluation/EDA data into `Figure` objects, including
  consistent boosting dashboards and native TreeSHAP views. Never
  calls `plt.show()`; always closable via `save_and_close`, so the same
  functions work interactively in a notebook and unattended in a pipeline.
- **`pipelines`** is where the modules above get wired into complete,
  runnable procedures (validate / leakage control / feature materialization /
  train / evaluate / batch-infer). This is the only layer that knows the
  *order* of operations. `BoostingBenchmarkPipeline` owns model comparison;
  `BoostingEvaluationPipeline` owns post-selection diagnostics. The progressive notebook flow writes leakage-safe
  partitions to `data/interim` and their deterministic feature-enriched
  counterparts to `data/processed`.
- **`tracking`** records the model lifecycle in MLflow. Notebook `04` creates
  one comparison parent run with nested XGBoost, LightGBM and CatBoost runs;
  notebook `05` records the approved winner and its final diagnostics. A local
  SQLite backend and its artifacts live in the shared user-level platform
  `/Users/surelmanda/.mlflow`, outside every Git repository.
- **`inference`** is IO-agnostic scoring (`Predictor`) plus a file-based
  wrapper (`BatchPredictor`) — the core scoring path a future realtime API
  would reuse without duplicating logic.
- **`cli.py`** is the only layer that talks to the terminal (argument
  parsing, exit codes, colored output). Business logic never lives here.

## The one-pipeline rule

`bnpl_credit_risk.features.preprocessing.build_preprocessing_pipeline` +
`bnpl_credit_risk.models.trainer.ModelTrainer` produce a **single**
`sklearn.Pipeline`: `[feature engineering → encoding → XGBoost]` (optionally
wrapped again by `CalibratedClassifierCV`). That whole object — not just the
XGBoost model — is what gets serialized to `pipeline.joblib` and reloaded at
inference time. There is no second code path that re-implements encoding or
feature engineering for scoring; `Predictor.predict_proba` is a single call
to `pipeline.predict_proba`.

The native boosting benchmark is a development experiment, not a second
production inference contract. Its versioned `benchmark.joblib` preserves OOF
predictions, the winning estimator and its training-only categorical vocabulary
between notebooks `04` and `05`. Once the experiment is accepted, the production
artifact still has to be published through `ArtifactBundle` before notebook `06`.
That explicit publication cell in notebook `05` records the approval flag,
selected native categorical adapter, frozen threshold, risk bands, metrics and
feature schema. `BatchInferencePipeline` can then refuse unapproved artifacts.

## Configuration flow

1. `bnpl_credit_risk.settings.Settings` (pydantic-settings) reads environment
   variables / `.env` — machine-specific overrides (paths and log level).
2. `bnpl_credit_risk.settings.load_config` reads `configs/*.yaml` into typed
   Pydantic models (`ProjectConfig`), validated against the enums in
   `constants.py`.
3. Every path in config is resolved relative to the project root
   (`Settings.resolve`), located by walking up from `settings.py` to the
   nearest `pyproject.toml` — no path in source code is ever absolute.

## Why not more layers

DDD-style ports/adapters, a repository pattern, or a plugin system were
deliberately not introduced — this is a single-model batch-scoring pipeline
with one data source, not a multi-tenant platform. `ModelFactory` is the one
extension point that's actually justified (swapping XGBoost for another
algorithm is a named, expected future need); everything else stays direct.
