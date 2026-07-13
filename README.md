# BNPL Credit Risk

Predicts whether a Buy-Now-Pay-Later (BNPL) customer will default, from a raw
transaction dataset to a batch-scored CSV of probabilities. Started as an
exploratory notebook (`Dev/bnpl-credit-risk-eda-feature-engineering-xgboost.ipynb`,
kept for reference — see [docs/notebook_migration.md](docs/notebook_migration.md))
and has been rebuilt as an installable, tested, configurable Python package.

## 1. The business problem

A merchant offers "pay in installments" at checkout. Before granting credit,
they need to estimate the probability that the customer will default. Two
distinct questions live in this repo, and the codebase keeps them explicitly
separate (see [docs/data_contract.md](docs/data_contract.md#leakage-audit)):

- **`application_risk`** (production default): should we grant this BNPL
  loan, using only information known *before* any repayment happens?
- **`behavioral_risk`**: given repayment events observed so far, how risky
  does this customer look now? Useful for collections prioritization, not
  for the initial accept/reject decision.

## 2. The target

`default_flag` (0/1) — whether the customer defaulted on their BNPL
installments. Roughly 39% of the raw dataset defaulted (mild imbalance,
handled via `scale_pos_weight` computed on the training split only).

## 3. Install

```bash
# Requires Python 3.11 or 3.12 and Poetry >= 1.9
poetry install
```

Copy `.env.example` to `.env` if you want to override any path, log level,
or MLflow setting (no secrets required for local use).

## 4. Place the dataset

The raw CSV already lives at `data/raw/BNPL_CreditRisk_Dataset.csv`. To use a
different file, either replace it in place or point `BNPL_DATA_RAW_PATH`
(env var) / `--input` (CLI flag) at another path — never hardcode a path in
source (the original notebook's `/kaggle/input/...` path has been removed).

## 5. Validate the data

```bash
poetry run bnpl-risk validate
```

Checks the data contract in [configs/data.yaml](configs/data.yaml) (required
columns, dtypes, bounds, allowed categories, duplicate ids, binary target).
Policy is configurable: `strict` (stop on any violation), `warn` (log and
continue), `quarantine` (split valid/invalid rows to a report file).

## 6. Train

```bash
poetry run bnpl-risk train                                            # application_risk (default)
poetry run bnpl-risk train --model-config configs/model_behavioral_risk.yaml
```

Runs validation → split (`time_based` by default, see
[docs/modeling.md](docs/modeling.md#split-strategy)) → threshold selection
on train-only out-of-fold predictions → fit → optional calibration →
evaluation on the held-out test split → persists a versioned artifact under
`artifacts/models/<timestamp>/` and updates `artifacts/models/latest.json`.

## 7. Evaluate

```bash
poetry run bnpl-risk evaluate --model-version latest
```

Re-scores the persisted pipeline against a deterministically rebuilt test
split and writes a fresh report to `data/reports/`.

## 8. Run batch inference

```bash
poetry run bnpl-risk predict-batch \
  --input data/processed/applications_to_score.csv \
  --output data/predictions/predictions_2026_07_13.csv \
  --model-version latest
```

Never retrains. Loads the persisted pipeline, validates the input (target
column not required — dropped if present), scores, applies the saved
threshold and risk bands, and writes the output file atomically. See
[docs/inference.md](docs/inference.md) for the output schema, exit codes,
and a nightly-cron / GitHub Actions example.

## 9. Notebooks

```bash
poetry run jupyter lab notebooks/
```

`notebooks/00`–`06` are thin consumers of the package (imports, config,
plots) — they don't redefine any logic that lives in `src/bnpl_credit_risk/`.
The original exploratory notebook is kept, unmodified, under `Dev/`.

## 10. Tests

```bash
poetry run pytest          # unit + integration + regression
poetry run ruff check .
poetry run mypy src
```

## 11. Artifacts produced by a training run

```
artifacts/models/2026-07-13_101500/
├── pipeline.joblib       # feature engineering + encoding + model (+ calibrator), one fitted object
├── metadata.json         # risk_scope, params, split strategy, git commit, data quality snapshot
├── metrics.json          # test-set metrics + train cross-validation summary
├── threshold.json        # selected threshold, policy, rationale, risk bands
└── feature_schema.json   # ordered input/output feature names
artifacts/models/latest.json   # pointer to the current version
```

## Documentation

- [docs/architecture.md](docs/architecture.md) — module responsibilities
- [docs/data_contract.md](docs/data_contract.md) — schema + **leakage audit**
- [docs/modeling.md](docs/modeling.md) — split, threshold, calibration, metrics
- [docs/inference.md](docs/inference.md) — batch inference contract + orchestration
- [docs/notebook_migration.md](docs/notebook_migration.md) — notebook → package correspondence table
- [docs/model_card.md](docs/model_card.md) — intended use, limitations, risks
