"""Train the lapse classifier and persist all inference artifacts.

    python train.py --input data/train_policies.csv --output artifacts/

Steps: load data -> build features + target -> time-based holdout evaluation
-> refit on all data -> save model bundle, metrics and feature importances.
The segmentation engine is rule-based (no training) and is rebuilt from code at
prediction time, so nothing about it needs to be persisted here.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib

from src.features import FeatureBuilder
from src.model import LapseModel
from src.utils import ensure_dir, get_logger, read_policies_csv, save_json

logger = get_logger()

# Fraction of the timeline used for training; the most recent slice is holdout.
TIME_SPLIT_QUANTILE = 0.8


def time_based_split(snapshot_date, quantile: float):
    """Boolean (train_mask, val_mask) splitting on a snapshot-date cutoff.

    A time-based split mimics production: we train on the past and validate on
    the future, which prevents leaking later-period information into training.
    """
    cutoff = snapshot_date.quantile(quantile)
    train_mask = snapshot_date < cutoff
    val_mask = snapshot_date >= cutoff
    return train_mask, val_mask, cutoff


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the lapse model.")
    parser.add_argument("--input", required=True, help="Path to training CSV")
    parser.add_argument("--output", required=True, help="Directory for saved model artifacts")
    args = parser.parse_args()

    output_dir = ensure_dir(args.output)

    logger.info("Loading training data from %s", args.input)
    df = read_policies_csv(args.input)

    builder = FeatureBuilder()
    X, y, snapshot_date = builder.build_training_frame(df)
    logger.info("Built %d feature rows; positive rate = %.4f", len(X), y.mean())

    # --- Time-based holdout evaluation ---
    train_mask, val_mask, cutoff = time_based_split(snapshot_date, TIME_SPLIT_QUANTILE)
    logger.info(
        "Time split at %s -> train=%d, val=%d (val positives=%d)",
        getattr(cutoff, "date", lambda: cutoff)(),
        int(train_mask.sum()),
        int(val_mask.sum()),
        int(y[val_mask].sum()),
    )

    eval_model = LapseModel(builder.numeric_features, builder.categorical)
    eval_model.fit(X[train_mask], y[train_mask])
    metrics = LapseModel.evaluate(y[val_mask], eval_model.predict_proba(X[val_mask]))
    metrics["time_split_cutoff"] = str(cutoff)
    logger.info("Holdout metrics: %s", metrics)

    # --- Refit on all available data for the deployed artifact ---
    logger.info("Refitting on the full dataset for deployment.")
    model = LapseModel(builder.numeric_features, builder.categorical)
    model.fit(X, y)

    # --- Persist artifacts ---
    bundle_path = Path(output_dir) / "model.joblib"
    joblib.dump({"feature_builder": builder, "lapse_model": model}, bundle_path)
    save_json(metrics, Path(output_dir) / "metrics.json")
    save_json(
        model.feature_importances().head(15).round(4).to_dict(),
        Path(output_dir) / "feature_importances.json",
    )

    logger.info("Saved model bundle to %s", bundle_path)
    logger.info("Top lapse drivers (global): %s", list(model.feature_importances().head(5).index))


if __name__ == "__main__":
    main()
