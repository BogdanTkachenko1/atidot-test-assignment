# AI Usage Log

## Tools used

- **Claude (Claude Code)** — used as a pair-programmer for exploratory data
  analysis, drafting the modular code, and writing documentation.

## What AI was used for

- Initial EDA: understanding the panel structure (220 policies × ~24 monthly
  snapshots), the class imbalance (~0.36% lapse rate), and the relationship
  between `status` and `lapse_next_month`.
- Surfacing the leakage and target-construction issues (post-lapse rows,
  `status` as a leakage column).
- Reading the segment PDFs and transcribing the rule signals.
- Drafting the OOP module structure (`FeatureBuilder`, `LapseModel`,
  `SegmentationEngine`) and the two CLI scripts.
- Drafting README / MODEL_CARD.

## Important prompts or excerpts

I made the modeling and architecture decisions; Claude was used to implement,
stress-test, and verify them. The prompts below are representative of how I
kept control of the work rather than delegating it wholesale.

- *Validating the target before trusting it:* "Don't assume `lapse_next_month`
  is clean. Cross-check it against `status` on the next snapshot for the same
  policy and tell me on how many rows the label and the realised outcome
  disagree."
- *Pinning down leakage:* "List every column and tell me which ones could leak
  the outcome or future information. For each excluded column, give me the
  concrete reason — I want to defend these choices, not just trust a default."
- *Choosing validation deliberately:* "Why a time-based split over stratified
  k-fold here, given the panel structure? Quantify what each would do to the
  number of positives in validation before I commit."
- *Reviewing the segmentation logic, not just accepting it:* "Show me, per
  segment, how many assigned policies actually violate the segment's defining
  condition. If any do, the scoring is wrong and we need a mandatory gate."
- *Questioning the preprocessing:* "Justify one-hot encoding for a tree model.
  If it only bloats dimensionality on high-cardinality columns like `state`,
  tell me what we lose and what the alternative would be."
- *Demanding reproducibility checks:* "Run train then predict on a copy of the
  new CSV with a column removed and a junk column added, and confirm the output
  schema and row count are unchanged. I want proof the inference path is
  robust."
- *Sanity-checking the metrics:* "A ROC-AUC below 0.5 on the holdout is a red
  flag — explain whether that's a real signal or an artifact of the positive
  count, and make the output flag it if the split is too small to trust."

## Suggestions accepted

- Dropping post-lapse (`status == lapsed`) rows from training and excluding
  `status` from features to avoid leakage.
- Time-based train/validation split and ranking metrics (ROC-AUC / PR-AUC)
  instead of accuracy given the rare positive class.
- Putting all preprocessing inside an sklearn `Pipeline` and a schema
  reconciliation step so inference tolerates missing/extra/unseen columns.
- Rule-based, explainable segmentation with a per-segment confidence equal to
  the fraction of matched signals.

## Suggestions rejected or modified

- Did **not** add heavier models (XGBoost/LightGBM) or extensive feature
  engineering — the brief explicitly favours a simple, correct, reproducible
  solution over an elaborate one.
- Simplified the per-row driver explanation to a transparent
  importance × deviation heuristic rather than pulling in SHAP, to keep the
  dependency footprint small and the output easy to explain.

## Uncertainties

- With only 18 lapse events, both the model and its evaluation are inherently
  weak; the holdout metrics are a sanity check, not a reliable estimate.
- The exact intended semantics of `lapse_rank` ("higher-risk ranked higher")
  are ambiguous; I chose rank 1 = highest risk and documented it.
- Some segment thresholds (e.g. "low agent touch") required a judgment call
  (interpreted as ≤ 1 interaction); these are documented in the code.
