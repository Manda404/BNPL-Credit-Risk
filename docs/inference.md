# Inference

Batch is the priority use case: score a CSV of applications once a night.
Realtime scoring reuses the same core (`Predictor`) but no API server is
built here — see [inference/realtime.py](../src/bnpl_credit_risk/inference/realtime.py).

## Publication prerequisite

Artifacts under `artifacts/benchmarks/` are development results and cannot be
used for scoring. After notebook `04` selects the best ranking model, notebook
`05` verifies calibration, freezes the decision threshold, evaluates the
reserved test once and explicitly publishes an `ArtifactBundle` under
`artifacts/models/` with `approved_for_inference: true`.

When `require_approved_model` is enabled in `configs/inference.yaml`, notebook
`06` and the batch pipeline reject legacy or unapproved artifacts. This keeps
`latest.json` from silently routing scoring to an old experiment.

## Batch contract

```bash
poetry run bnpl-risk predict-batch \
  --input /path/to/applications.csv \
  --output data/predictions/predictions_2026_07_13.csv \
  --model-version latest
```

Steps (`inference/batch.py::BatchPredictor.run`):

1. Resolve `model-version` ("latest" or an explicit run id) via
   `models/registry.py` and load the full `ArtifactBundle`.
2. Verify that the artifact is approved and inspect its feature contract.
3. Load the input CSV.
4. If `default_flag` is present in the input, drop it and log a warning — it
   is **never** required and **never** used.
5. Clean (dtype coercion, whitespace) and validate against
   `InferenceInputSchema` under the configured policy (`configs/inference.yaml`:
   `strict` / `warn` / `quarantine`).
6. Verify the features required by the published artifact and score valid rows
   with `pipeline.predict_proba` (the exact same fitted
   [feature engineering → encoding → model] object used at training time —
   no re-fitting, ever).
7. Apply the threshold and risk bands persisted in `threshold.json` at
   training time (not read from the current `configs/training.yaml`, so a
   config edit after training can't silently change a live model's
   decisions).
8. Enforce the configured output-column order and write atomically (write to a temp file in the same directory,
   then `os.replace`) so a crash mid-write never leaves a partial file at
   the target path.

### Notebook test submission

Notebook `06` uses `data/processed/test_features.csv` as a labeled evaluation
partition. `BatchInferencePipeline.run_test_submission` preserves the target in
a separate Series, delegates scoring to `BatchPredictor` (which drops the target
before model input), then reconciles rows by `user_id`. It writes exactly two
columns to `data/output/submission.csv`: `predicted_label` and `true_label`.
Detailed probabilities and risk bands remain in memory for diagnostics and are
not added to the requested submission file.

`artifacts/models/latest.json` stores only a version identifier. The registry
always resolves that identifier relative to the current project artifact
directory, so moving or cloning the repository cannot leave a stale absolute
path in the deployment pointer.

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
  --input /path/to/applications.csv \
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
