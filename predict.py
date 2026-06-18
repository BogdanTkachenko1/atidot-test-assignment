"""Score new policies: lapse probability + risk rank + business segment.

    python predict.py --input data/new_policies.csv --model artifacts/ --output predictions.csv

Loads the trained bundle, scores the input CSV without retraining, assigns
segments via the rule engine, and writes the required output columns. The
inference path tolerates missing optional columns, extra columns, nulls and
unseen categorical values (handled inside the feature builder and pipeline).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.segmentation import SegmentationEngine
from src.utils import get_logger, read_policies_csv

logger = get_logger()

OUTPUT_COLUMNS = [
    "policy_id",
    "snapshot_date",
    "lapse_probability_1m",
    "lapse_rank",
    "segment_name",
    "segment_confidence",
    "segment_reason",
    "top_lapse_drivers",
]


def load_bundle(model_dir: str):
    """Load the trained model bundle from a directory (or direct file path)."""
    path = Path(model_dir)
    if path.is_dir():
        path = path / "model.joblib"
    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path}")
    bundle = joblib.load(path)
    return bundle["feature_builder"], bundle["lapse_model"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Score new policies for lapse risk and segment.")
    parser.add_argument("--input", required=True, help="Path to new policy CSV")
    parser.add_argument("--model", required=True, help="Directory containing saved model artifacts")
    parser.add_argument("--output", required=True, help="Path to output predictions CSV")
    args = parser.parse_args()

    logger.info("Loading model artifacts from %s", args.model)
    builder, model = load_bundle(args.model)

    logger.info("Loading new policies from %s", args.input)
    raw = read_policies_csv(args.input).reset_index(drop=True)

    # --- Lapse scoring ---
    X, meta = builder.build_inference_frame(raw)
    probabilities = model.predict_proba(X)
    drivers = model.explain_drivers(X, probabilities)

    out = meta.copy()
    out["lapse_probability_1m"] = probabilities.round(6)
    # Rank 1 = highest-risk policy (probabilities tied broken deterministically).
    out["lapse_rank"] = out["lapse_probability_1m"].rank(ascending=False, method="first").astype(int)
    out["top_lapse_drivers"] = drivers

    # --- Segmentation (rule-based, rebuilt from code) ---
    segments = SegmentationEngine().assign(raw)
    out = out.join(segments)

    out = out[OUTPUT_COLUMNS]
    out.to_csv(args.output, index=False)
    logger.info("Wrote %d predictions to %s", len(out), args.output)
    logger.info("Segment distribution:\n%s", out["segment_name"].value_counts().to_string())


if __name__ == "__main__":
    main()
