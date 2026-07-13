# Inference

Batch is the priority use case: score a CSV of applications once a night.
Realtime scoring reuses the same core (`Predictor`) but no API server is
built here — see [inference/realtime.py](../src/bnpl_credit_risk/inference/realtime.py).

## Batch contract

```bash
poetry run bnpl-risk predict-batch \
  --input data/processed/applications_to_score.csv \
  --output data/predictions/predictions_2026_07_13.csv \
  --model-version latest
```

Steps (`inference/batch.py::BatchPredictor.run`):

1. Resolve `model-version` ("latest" or an explicit run id) via
   `models/registry.py` and load the full `ArtifactBundle`.
2. Load the input CSV.
3. If `default_flag` is present in the input, drop it and log a warning — it
   is **never** required and **never** used.
4. Clean (dtype coercion, whitespace) and validate against
   `InferenceInputSchema` under the configured policy (`configs/inference.yaml`:
   `strict` / `warn` / `quarantine`).
5. Score valid rows: `pipeline.predict_proba` (the exact same fitted
   [feature engineering → encoding → model] object used at training time —
   no re-fitting, ever).
6. Apply the threshold and risk bands persisted in `threshold.json` at
   training time (not read from the current `configs/training.yaml`, so a
   config edit after training can't silently change a live model's
   decisions).
7. Write the output atomically (write to a temp file in the same directory,
   then `os.replace`) so a crash mid-write never leaves a partial file at
   the target path.

### Output schema

| Column | Meaning |
|---|---|
| `user_id` | from `configs/data.yaml`'s `id_column` |
| `default_probability` | `pipeline.predict_proba(...)[:, 1]` |
| `predicted_default` | `1` if `default_probability >= decision_threshold` |
| `decision_threshold` | the threshold persisted with this model version |
| `risk_band` | Low / Medium / High / Very High Risk, per `configs/training.yaml`'s `risk_bands` at train time |
| `model_version` | the artifact version that produced this row |
| `scoring_timestamp` | ISO timestamp of the scoring run |

### Invalid rows

Under `quarantine` (the recommended policy for production batch runs),
rejected rows are written to `data/reports/rejected_rows_<output-stem>.csv`
and a warning is logged with the count. Under `strict`, any invalid row
aborts the whole run (`DataValidationError`, exit code 1) — appropriate for
a first integration pass, less appropriate for a fully unattended nightly
job where one malformed row shouldn't block a few thousand valid scores.

### Exit codes

| Code | Meaning |
|---|---|
| 0 | success |
| 1 | data contract violation (`strict` policy, or no valid rows to score) |
| 2 | technical/config error (missing model, misconfigured YAML, ...) |

## Orchestration

The pipeline never talks to a scheduler — it's a plain CLI command with exit
codes, so any of the following can drive it without touching
`bnpl_credit_risk` source:

**Cron** (on a host with the repo + Poetry installed):

```cron
0 2 * * * cd /path/to/BNPL-Credit-Risk && poetry run bnpl-risk predict-batch \
  --input data/processed/applications_to_score.csv \
  --output "data/predictions/predictions_$(date +\%Y_\%m_\%d).csv" \
  >> logs/cron_batch.log 2>&1
```

**GitHub Actions** — see [.github/workflows/nightly_batch.yml](../.github/workflows/nightly_batch.yml),
scheduled at 02:00 UTC, also runnable on demand via `workflow_dispatch`.

**Airflow** (future) — wrap `poetry run bnpl-risk predict-batch ...` in a
`BashOperator` / `PythonOperator` calling `scripts/predict_batch.py`; the
script has no Airflow-specific code, so this is a config change, not a
rewrite.

## Realtime (minimal, not built out)

`inference/predictor.py::Predictor` and `inference/realtime.py::score_single`
score one record synchronously with no batch-specific I/O. A future FastAPI
route would call `score_single(predictor, record, id_column)` directly — no
new scoring logic needed, only request/response plumbing.
