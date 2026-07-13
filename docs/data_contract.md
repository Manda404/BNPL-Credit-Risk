# Data Contract

Source of truth: [configs/data.yaml](../configs/data.yaml). This document explains
*why* the contract looks the way it does — read it alongside the YAML, not
instead of it.

## Schema

| Column | Role | Type | Notes |
|---|---|---|---|
| `user_id` | id | int | unique per row in the current dataset (verified: 0 duplicates on 10,345 rows) |
| `age` | feature | int | bounds [18, 100] |
| `employment_type` | feature | category | Salaried, Student, Self-Employed, Unemployed |
| `monthly_income` | feature | float | bounds [0, 1,000,000] |
| `credit_score` | feature | int | bounds [300, 850] (FICO-like range) |
| `purchase_amount` | feature | float | bounds [0, 100,000] |
| `product_category` | feature | category | Electronics, Fashion, Sports, Home, Beauty |
| `bnpl_installments` | feature | int | observed values: 3, 6, 9, 12 |
| `repayment_delay_days` | **leaky** | int | see audit below |
| `missed_payments` | **leaky** | int | see audit below |
| `default_flag` | **target** | binary (0/1) | required for training, forbidden for inference |
| `app_usage_frequency` | feature | float | |
| `location` | feature | category | Australia, USA, Germany, India, Canada, UK |
| `transaction_date` | feature / split key | date | range observed: 2023-01-01 → 2024-12-31 |
| `debt_to_income_ratio` | feature | float | bounds [0, 5] |
| `risk_score` | **leaky** | float | see audit below |
| `customer_segment` | **leaky** | category | Low Risk, Medium Risk, High Risk — see audit below |

Two schemas are derived from this table (`bnpl_credit_risk.data.schemas`):

- **`TrainingInputSchema`** — requires `default_flag`, validates it's binary.
- **`InferenceInputSchema`** — does **not** require `default_flag`. If a batch
  input file happens to include it, `BatchPredictor` drops it before scoring
  and logs a warning — it is never read.

## Validation policy

Configurable per `configs/data.yaml` / `configs/inference.yaml` (`validation.strategy`):

- `strict` — any row-level violation aborts the pipeline (`DataValidationError`).
- `warn` — violations are logged, all rows proceed.
- `quarantine` — invalid rows are split out and written to a CSV report; only
  valid rows continue through the pipeline.

Structural violations (a required column entirely missing, or the date column
entirely unparseable) always abort, regardless of policy — there's no
sensible way to quarantine a missing column.

## Leakage audit

This is the most consequential decision in this migration. The question that
matters is: **what information is actually available at the moment a BNPL
application must be accepted or rejected?**

Four columns were audited and found to depend on events that only exist
*after* the loan is granted:

| Column | Evidence | Verdict |
|---|---|---|
| `repayment_delay_days` | rows with `delay > 15` days default 73% of the time vs. 31% otherwise | **Leaky** — a repayment delay cannot exist before repayment has started |
| `missed_payments` | `missed_payments ≥ 3` → 83% default rate vs. 28% at `missed_payments = 0` | **Leaky** — same reasoning |
| `risk_score` | corr. with `default_flag` = 0.399, the single strongest numeric correlate | **Leaky** — no documentation ties this to a pre-approval bureau score, and its distribution overlaps heavily with `customer_segment` (see below); treated as a derived, near-target signal until proven otherwise |
| `customer_segment` | `High Risk` segment (risk_score 120–398) defaults 49.6% of the time; `Low Risk` (0–186) defaults 6.1% of the time — segments are bucketed directly from `risk_score` | **Leaky** — inherits `risk_score`'s problem |

Two engineered features inherit the leakage transitively:

- `payment_stress` = `repayment_delay_days × missed_payments` — leaky by construction.
- `is_high_risk` = `risk_score > 250` — leaky by construction.

**Everything else** (`age`, `employment_type`, `monthly_income`, `credit_score`,
`purchase_amount`, `product_category`, `bnpl_installments`, `app_usage_frequency`,
`location`, `debt_to_income_ratio`, and the engineered `income_to_purchase_ratio`,
`age_group`, `txn_month`) is information a merchant already has at checkout,
before any repayment behavior exists.

### Two model configurations, not one

`configs/features.yaml` encodes this split via `risk_scope`:

- **`application_risk`** (default, `configs/model.yaml`) — excludes every
  leaky column above. This is the model that answers the real business
  question: *should we grant this BNPL loan?*
- **`behavioral_risk`** (`configs/model_behavioral_risk.yaml`, run explicitly)
  — includes everything, reproducing the original notebook's feature set.
  Useful for **in-life risk monitoring / collections prioritization** on
  loans that have already been granted — never for the accept/reject
  decision.

Measured impact on held-out ROC-AUC (same data, same split, same XGBoost
params — see `tests/regression/test_notebook_parity.py`):

| Scope | ROC-AUC (test) |
|---|---|
| `application_risk` | ~0.70 |
| `behavioral_risk` | ~0.78 |

The ~0.08 AUC gap **is** the leakage — it quantifies how much of the
notebook's reported performance came from information that doesn't exist at
decision time. `configs/model.yaml` defaults to `application_risk` precisely
so this gap can never be silently baked into "the" production model.
