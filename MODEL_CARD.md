# Model Card

## What the model predicts

For each policy-month snapshot, the model estimates the probability that the
policy **lapses within the next month** (`lapse_probability_1m`). Outputs are
used to rank policies by lapse risk and, alongside the segmentation engine, to
support retention decisions.

## Target construction

The training data includes a convenience label `lapse_next_month`, which is
already a forward-looking, next-month indicator. We use it directly as the
binary target. We verified its meaning during EDA: a row with
`lapse_next_month = 1` is immediately followed by a row whose `status` becomes
`lapsed` in the next monthly snapshot.

Because a lapsed policy cannot lapse again, rows whose current `status` is
already `lapsed` (post-event rows) are **dropped** from training — they are not
part of the at-risk population. This removes 24 rows, leaving 5,018 at-risk
snapshots with an ~0.36% positive rate (18 lapse events).

## Features used

- **Numeric**: `current_age`, `income_estimate`, `face_amount`,
  `annualized_premium`, `premium_paid_last_3m`, `missed_payment_count_6m`,
  `agent_touch_count_12m`, `cash_value`, `loan_balance`, `autopay_flag`,
  `address_change_12m`, `beneficiary_change_12m`.
- **Engineered**: `tenure_months` = months between `issue_date` and
  `snapshot_date` (negative values clipped to 0).
- **Categorical** (one-hot): `product_type`, `state`, `urbanicity`,
  `distribution_channel`, `payment_mode`.

Missing numeric values are median-imputed; missing categoricals are
most-frequent-imputed then one-hot encoded with `handle_unknown="ignore"` so
unseen categories do not break inference. All preprocessing lives inside the
sklearn `Pipeline`, so training and inference transforms are identical.

## Algorithm and configuration

The classifier is a **`sklearn.ensemble.RandomForestClassifier`** wrapped in a
single `sklearn.pipeline.Pipeline` together with all preprocessing
(`src/model.py`, class `LapseModel`). The pipeline has two stages:

1. **Preprocessing** (`ColumnTransformer`):
   - numeric branch → `SimpleImputer(strategy="median")`
   - categorical branch → `SimpleImputer(strategy="most_frequent")` →
     `OneHotEncoder(handle_unknown="ignore")`
2. **Estimator** (`RandomForestClassifier`) with:

   | Hyperparameter | Value | Why |
   |---|---|---|
   | `n_estimators` | `300` | enough trees for stable probabilities on a small dataset |
   | `class_weight` | `"balanced"` | counteracts the ~0.36% positive rate so the rare lapse class is not ignored |
   | `min_samples_leaf` | `5` | regularizes leaves to limit overfitting given few positives |
   | `random_state` | `42` | full reproducibility of the trained model |
   | `n_jobs` | `-1` | use all cores for training |

**Why a random forest?** The brief favours a *simple, correct, reproducible*
solution over a powerful one. A random forest needs no feature scaling, handles
mixed numeric/categorical data and non-linear interactions out of the box, gives
free global feature importances (used for `top_lapse_drivers`), and is
deterministic given a fixed `random_state`. Gradient-boosted alternatives
(XGBoost/LightGBM) were deliberately **not** used to keep the dependency
footprint small — see `AI_USAGE.md`.

The probability of the positive class (`predict_proba[:, 1]`) is the reported
`lapse_probability_1m`.

## Leakage risks considered

- **`status`** directly reveals the outcome (active vs lapsed) → **excluded**
  from features.
- **Post-lapse rows** would let the model "see" an already-lapsed state →
  **dropped** from training.
- **Identifiers / raw dates** (`policy_id`, `snapshot_date`, `issue_date`) are
  excluded as features; dates are used only to compute tenure (a past-looking
  quantity).
- **Temporal leakage**: validation uses a time-based split (train on the past,
  validate on the future) rather than random folds.

## Validation method

A **time-based holdout**: snapshots before the 80th-percentile snapshot date
(`2024-08-01`) train the model; later snapshots are the holdout. We report
ROC-AUC, PR-AUC (average precision) and Brier score — accuracy is meaningless
at a ~0.4% base rate. The final deployed model is then refit on all data.

**Important caveat**: with only 18 lapse events in total, the holdout window
contains a single positive, so the reported holdout metrics are statistically
unreliable and should be read as a sanity check, not a performance guarantee.

## Segmentation engine

A deterministic, rule-based engine (`src/segmentation.py`). The four segment
definitions were transcribed **manually** from the PDFs in
`data/segment_docs/` into a mandatory **gate** (the defining "Age"/"Duration"
signal) plus supporting signals:

| Segment | Gate (mandatory) | Supporting signals |
|---|---|---|
| `high_income_urban_builder` | age 35–55 | income > 120k; face ≥ 500k; urban |
| `post_retirement_high_coverage` | age 65+ | face ≥ 500k or cash ≥ 50k; non-term product |
| `early_tenure_payment_friction` | tenure < 18m | monthly/no-autopay/missed payments; direct or low agent touch |
| `legacy_stable_value` | tenure ≥ 84m | no missed payments; autopay on; agent touch in 12m |

A policy is **eligible** for a segment only if its gate matches — this prevents
nonsensical assignments such as a 2-month-old policy being labelled
`legacy_stable_value` just because it matched the other signals. Among eligible
segments, confidence is the **fraction of all conditions (gate + signals) that
match**; the highest-confidence segment wins (ties break by definition order),
and the explanation lists exactly which conditions fired. Policies eligible for
no segment are labelled `unsegmented`. The assignment is fully deterministic and
explainable.

`top_lapse_drivers` is produced by combining the model's global feature
importance with how far each numeric feature deviates from the training mean in
the risk-increasing direction, then listing the top contributors per policy.
Policies whose predicted risk is at or below the training base rate are reported
as low-risk instead, so the driver string never contradicts a near-zero
probability.

## Known limitations

- **Very few positive events (18)** → the model and its evaluation are weak;
  metrics are indicative only.
- The time-based holdout has a single positive, so AUC there is noisy.
- `top_lapse_drivers` uses numeric features only (categoricals omitted for
  readability) and is a heuristic, not exact SHAP attribution.
- Segment rules use thresholds taken literally from the PDFs; borderline
  policies may sit just outside a segment.
- Synthetic data: relationships may not reflect real lapse behaviour.

## What I would improve with more time

- Gather more lapse events (or downsample/SMOTE carefully) and use grouped,
  rolling time-series cross-validation by policy to get stable metrics.
- Add SHAP-based per-policy explanations and probability calibration.
- Engineer trend features (e.g. change in missed payments / premium over recent
  months) from the panel structure.
- Make segment thresholds configurable and add overlap/priority handling plus
  unit tests for both the feature and segmentation logic.
