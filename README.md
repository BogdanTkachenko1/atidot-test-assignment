# Atidot — Lapse Prediction & Policy Segmentation

A small, reproducible Python project that, for each life-insurance policy
snapshot:

1. predicts the probability the policy **lapses within the next month**, and
2. assigns it to a **business segment** with a confidence score and an
   explanation.

## Setup

**Requirements:** Python **3.9+** (developed and tested on 3.11). The only
third-party dependencies are `pandas`, `numpy`, `scikit-learn` and `joblib`
(pinned in `requirements.txt`).

### Option A — venv + pip (standard library, recommended)

```bash
# from the repo root
python3 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Option B — conda

```bash
conda create -n atidot python=3.11 -y
conda activate atidot
pip install -r requirements.txt
```

### Verify the install

```bash
python -c "import pandas, numpy, sklearn, joblib; print('environment OK')"
```

To leave the environment afterwards: `deactivate` (venv) or
`conda deactivate` (conda).

## Model at a glance

- **Algorithm:** `RandomForestClassifier` (scikit-learn) inside a single
  `Pipeline` with median/most-frequent imputation and one-hot encoding — so
  training and inference apply identical transforms.
- **Features:** 12 raw numeric columns + 1 engineered (`tenure_months`) + 5
  one-hot categoricals (listed under *Expected input format* below).
- **Target:** the forward-looking `lapse_next_month` label; post-lapse rows are
  dropped to keep an at-risk population.

See `MODEL_CARD.md` for the full algorithm configuration, hyperparameters, the
rationale for each choice, leakage handling and validation.

## Train

```bash
python train.py --input data/train_policies.csv --output artifacts/
```

This builds features and the target, runs a **time-based holdout evaluation**,
refits on all data, and writes to `artifacts/`:

- `model.joblib` — the feature builder + trained model bundle (everything
  `predict.py` needs).
- `metrics.json` — holdout evaluation metrics.
- `feature_importances.json` — top global lapse drivers.

## Predict

```bash
python predict.py --input data/new_policies.csv --model artifacts/ --output predictions.csv
```

Loads the saved bundle and scores a new CSV **without retraining**.

## Project layout

```
src/
  utils.py          # IO, date parsing, tenure, schema reconciliation
  features.py       # FeatureBuilder: target construction + feature matrix
  model.py          # LapseModel: sklearn pipeline + driver explanations
  segmentation.py   # SegmentationEngine: rule-based segment assignment
train.py            # CLI: train + evaluate + persist artifacts
predict.py          # CLI: load artifacts + score + segment -> CSV
data/               # provided training/inference CSVs + segment PDFs
artifacts/          # created by train.py
```

## Expected input format

A CSV of monthly policy snapshots (one row per policy-month). See
`DATA_DICTIONARY.md` for column meanings. The key columns used are:

- dates: `snapshot_date`, `issue_date`
- numeric: `current_age`, `income_estimate`, `face_amount`,
  `annualized_premium`, `premium_paid_last_3m`, `missed_payment_count_6m`,
  `agent_touch_count_12m`, `cash_value`, `loan_balance`, `autopay_flag`,
  `address_change_12m`, `beneficiary_change_12m`
- categorical: `product_type`, `state`, `urbanicity`,
  `distribution_channel`, `payment_mode`
- target (training only): `lapse_next_month`

The inference CSV does **not** need the target. The pipeline tolerates
realistic schema drift (see Assumptions).

## Expected output format

`predict.py` writes a CSV with exactly these columns:

| column | meaning |
|---|---|
| `policy_id` | policy identifier |
| `snapshot_date` | snapshot date scored |
| `lapse_probability_1m` | predicted P(lapse within one month) |
| `lapse_rank` | risk rank; **1 = highest risk** |
| `segment_name` | assigned business segment (gate must match; else `unsegmented`) |
| `segment_confidence` | fraction of segment conditions (gate + signals) matched (0–1) |
| `segment_reason` | which conditions matched |
| `top_lapse_drivers` | main factors raising risk (or a low-risk note if ≤ base rate) |

## Assumptions

- **Target**: the provided `lapse_next_month` label is used directly — it is
  already a forward-looking next-month flag, so no future information leaks.
- **At-risk population**: rows already in `status == lapsed` are post-event and
  are dropped from training (a lapsed policy cannot lapse again).
- **Leakage**: `status` is excluded from features (it reveals the outcome); so
  are identifiers and raw dates (dates are used only to derive tenure).
- **Schema robustness**, handled gracefully at inference:
  - missing optional columns → added as NaN and imputed,
  - extra/irrelevant columns → ignored,
  - null values → imputed (median / most-frequent),
  - unseen categorical values → encoded as all-zeros,
  - a different date range → fine (only relative tenure is used).
- Negative tenure (snapshot before issue, a data quirk) is clipped to 0.
- `lapse_rank` uses rank 1 for the highest-risk policy.

See `MODEL_CARD.md` for modeling details and limitations, and `AI_USAGE.md`
for how AI tools were used.
