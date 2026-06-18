"""Feature engineering and target construction for the lapse model.

The :class:`FeatureBuilder` is the single source of truth for *which* columns
are features, how the target is built, and how leakage is avoided. Both
``train.py`` and ``predict.py`` go through it, so training and inference can
never drift apart. The builder is configuration only (column lists), so it is
cheap to serialize alongside the model.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import pandas as pd

from .utils import compute_tenure_months, get_logger, reconcile_columns

logger = get_logger()


@dataclass
class FeatureBuilder:
    """Builds model-ready feature frames and the training target.

    Column groups are class-level defaults but kept as instance fields so the
    exact configuration used at training time is persisted with the model.
    """

    # Raw numeric columns taken directly from the CSV.
    numeric_raw: list[str] = field(
        default_factory=lambda: [
            "current_age",
            "income_estimate",
            "face_amount",
            "annualized_premium",
            "premium_paid_last_3m",
            "missed_payment_count_6m",
            "agent_touch_count_12m",
            "cash_value",
            "loan_balance",
            "autopay_flag",
            "address_change_12m",
            "beneficiary_change_12m",
        ]
    )
    # Numeric features we engineer (not present in the raw CSV).
    numeric_engineered: list[str] = field(default_factory=lambda: ["tenure_months"])
    # Categorical columns (one-hot encoded downstream).
    categorical: list[str] = field(
        default_factory=lambda: [
            "product_type",
            "state",
            "urbanicity",
            "distribution_channel",
            "payment_mode",
        ]
    )

    # --- Columns deliberately excluded from features (documented for the reviewer) ---
    # status              -> leakage: directly reveals lapsed/active outcome.
    # lapse_next_month    -> the target itself.
    # policy_id           -> identifier, no predictive meaning.
    # snapshot_date/issue_date -> raw dates; used only to derive tenure_months.
    target_col: str = "lapse_next_month"
    status_col: str = "status"

    # ------------------------------------------------------------------ #
    # Feature columns
    # ------------------------------------------------------------------ #
    @property
    def numeric_features(self) -> list[str]:
        return self.numeric_raw + self.numeric_engineered

    @property
    def feature_columns(self) -> list[str]:
        return self.numeric_features + self.categorical

    @property
    def required_raw_columns(self) -> list[str]:
        """Raw input columns needed to build features (excludes engineered)."""
        return self.numeric_raw + self.categorical + ["snapshot_date", "issue_date"]

    # ------------------------------------------------------------------ #
    # Transformation
    # ------------------------------------------------------------------ #
    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """Turn a raw (already date-parsed) frame into the model feature matrix.

        Handles schema drift: missing optional columns are added as NaN and
        extra/irrelevant columns are dropped. Categorical columns are cast to
        string so unseen values are handled consistently by the encoder.
        """
        df = reconcile_columns(df, self.required_raw_columns)
        df["tenure_months"] = compute_tenure_months(df)
        for col in self.categorical:
            # Preserve NaN, stringify everything else for stable one-hot behaviour.
            df[col] = df[col].where(df[col].isna(), df[col].astype(str))
        return df[self.feature_columns]

    # ------------------------------------------------------------------ #
    # Training frame + target construction
    # ------------------------------------------------------------------ #
    def build_training_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Return ``(X, y, snapshot_date)`` for model training.

        Target construction
        --------------------
        We use the provided ``lapse_next_month`` label directly: it is already
        a forward-looking, next-month target, so no future information leaks
        into the features for a given snapshot.

        At-risk filtering
        ------------------
        Rows whose current ``status`` is already ``lapsed`` are *post-event*:
        the policy has lapsed and cannot lapse again, so they are not part of
        the at-risk population and are dropped. This also removes the only
        rows where ``status`` would otherwise leak the outcome.
        """
        if self.target_col not in df.columns:
            raise ValueError(f"Training data must contain the target column '{self.target_col}'.")

        n_start = len(df)
        if self.status_col in df.columns:
            at_risk = df[self.status_col].astype(str).str.lower() != "lapsed"
            df = df[at_risk].copy()
            logger.info("Dropped %d post-lapse rows; %d at-risk rows remain.", n_start - len(df), len(df))

        # Drop rows with an unusable target.
        df = df[df[self.target_col].notna()].copy()
        y = df[self.target_col].astype(int)

        X = self.transform(df)
        snapshot_date = df["snapshot_date"].reset_index(drop=True)
        return X.reset_index(drop=True), y.reset_index(drop=True), snapshot_date

    def build_inference_frame(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Return ``(X, meta)`` for scoring.

        ``meta`` carries ``policy_id`` and ``snapshot_date`` for the output
        file. Both are reconstructed defensively so prediction never crashes on
        a missing identifier column.
        """
        meta = pd.DataFrame(
            {
                "policy_id": df["policy_id"] if "policy_id" in df.columns else range(len(df)),
                "snapshot_date": df["snapshot_date"] if "snapshot_date" in df.columns else pd.NaT,
            }
        ).reset_index(drop=True)
        X = self.transform(df).reset_index(drop=True)
        return X, meta
