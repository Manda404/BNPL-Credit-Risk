# Architecture

## Layout

```
configs/       YAML configuration (paths, data contract, features, model, training, inference, logging)
data/          raw / interim / processed / predictions / reports
artifacts/     models / preprocessors / metrics / figures / schemas (versioned outputs of training)
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
└── inference/                Predictor (core), batch, realtime
tests/
├── unit/          one module, one behavior at a time
├── integration/    pipelines run end-to-end against a small fixture dataset
└── regression/      parity against the original notebook, on the real dataset
```

## Module responsibilities

- **`data`** answers "is this dataframe safe to use, and in what shape?" —
  nothing here trains anything or computes a prediction.
- **`features`** answers "given a safe dataframe, what does the model see?" —
  `BNPLFeatureBuilder` is pure pandas (deterministic, no fitted state);
  `preprocessing.build_preprocessing_pipeline` wraps it plus a
  `ColumnTransformer` into one fittable `sklearn.Pipeline`.
- **`models`** answers "how do we produce and persist a fitted pipeline?" —
  `ModelTrainer` composes the preprocessing pipeline with an XGBoost step and
  fits it on train data only; `ArtifactBundle` is the sole persistence format.
- **`evaluation`** answers "how good is this pipeline, and at what threshold?"
  — pure functions over `(y_true, y_prob)`, no I/O, no side effects.
- **`visualization`** turns evaluation/EDA data into `Figure` objects. Never
  calls `plt.show()`; always closable via `save_and_close`, so the same
  functions work interactively in a notebook and unattended in a pipeline.
- **`pipelines`** is where the modules above get wired into complete,
  runnable procedures (validate / train / evaluate / batch-infer). This is
  the only layer that knows the *order* of operations.
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

## Configuration flow

1. `bnpl_credit_risk.settings.Settings` (pydantic-settings) reads environment
   variables / `.env` — machine-specific overrides (paths, log level, MLflow
   on/off).
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
