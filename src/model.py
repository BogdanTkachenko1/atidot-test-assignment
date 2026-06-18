"""Lapse classifier: an sklearn pipeline wrapped in an explainable OOP class.

:class:`LapseModel` owns the full preprocessing + RandomForest pipeline and a
light, dependency-free driver explainer that turns a prediction into a short
"why" string. Keeping preprocessing inside the pipeline guarantees that
inference applies exactly the transforms fitted at training time.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from .utils import get_logger

logger = get_logger()


class LapseModel:
    """Random-forest lapse classifier with built-in preprocessing + explanations."""

    def __init__(
        self,
        numeric_features: list[str],
        categorical_features: list[str],
        random_state: int = 42,
        n_estimators: int = 300,
    ) -> None:
        self.numeric_features = list(numeric_features)
        self.categorical_features = list(categorical_features)
        self.random_state = random_state
        self.n_estimators = n_estimators
        self.pipeline: Pipeline = self._build_pipeline()
        self._driver_stats: dict[str, dict[str, float]] = {}
        self._base_rate: float = 0.0

    # ------------------------------------------------------------------ #
    # Pipeline definition
    # ------------------------------------------------------------------ #
    def _build_pipeline(self) -> Pipeline:
        """Median-impute numerics; most-frequent-impute + one-hot categoricals.

        ``handle_unknown="ignore"`` lets unseen categorical values at inference
        encode as all-zeros instead of raising.
        """
        numeric = Pipeline(steps=[("impute", SimpleImputer(strategy="median"))])
        categorical = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        pre = ColumnTransformer(
            transformers=[
                ("num", numeric, self.numeric_features),
                ("cat", categorical, self.categorical_features),
            ]
        )
        clf = RandomForestClassifier(
            n_estimators=self.n_estimators,
            class_weight="balanced",
            min_samples_leaf=5,
            random_state=self.random_state,
            n_jobs=-1,
        )
        return Pipeline(steps=[("pre", pre), ("clf", clf)])

    # ------------------------------------------------------------------ #
    # Fit / predict
    # ------------------------------------------------------------------ #
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "LapseModel":
        self.pipeline.fit(X, y)
        self._base_rate = float(np.asarray(y).mean())
        self._fit_driver_stats(X, y)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.pipeline.predict_proba(X)[:, 1]

    # ------------------------------------------------------------------ #
    # Evaluation
    # ------------------------------------------------------------------ #

    MIN_RELIABLE_POSITIVES = 10

    @staticmethod
    def evaluate(y_true: pd.Series, y_prob: np.ndarray) -> dict:
        """Compute ranking + calibration metrics suited to a rare positive class.

        Accuracy is meaningless at a ~0.4% base rate, so we report ROC-AUC and
        PR-AUC (average precision) for ranking quality plus the Brier score for
        calibration. Returns NaN-safe placeholders if a split has one class.

        A ``reliable`` flag marks whether the split has enough positives for the
        ranking metrics to be meaningful, so downstream consumers do not read a
        single-positive AUC as a real performance number.
        """
        y_true = np.asarray(y_true)
        n_pos = int(y_true.sum())
        reliable = LapseModel.MIN_RELIABLE_POSITIVES <= n_pos < len(y_true)
        metrics = {
            "n_samples": int(len(y_true)),
            "n_positives": n_pos,
            "base_rate": float(y_true.mean()) if len(y_true) else float("nan"),
            "reliable": reliable,
        }
        if not reliable:
            logger.warning(
                "Holdout has %d positive(s) (< %d); ranking metrics are noisy and flagged unreliable.",
                n_pos,
                LapseModel.MIN_RELIABLE_POSITIVES,
            )
        if n_pos == 0 or n_pos == len(y_true):
            logger.warning("Evaluation split has a single class; ranking metrics are undefined.")
            metrics.update({"roc_auc": None, "pr_auc": None, "brier": None})
        else:
            metrics.update(
                {
                    "roc_auc": float(roc_auc_score(y_true, y_prob)),
                    "pr_auc": float(average_precision_score(y_true, y_prob)),
                    "brier": float(brier_score_loss(y_true, y_prob)),
                }
            )
        return metrics

    # ------------------------------------------------------------------ #
    # Feature importance / driver explanations
    # ------------------------------------------------------------------ #
    def feature_importances(self) -> pd.Series:
        names = self.pipeline.named_steps["pre"].get_feature_names_out()
        importances = self.pipeline.named_steps["clf"].feature_importances_
        clean = [n.split("__", 1)[-1] for n in names]
        return pd.Series(importances, index=clean).sort_values(ascending=False)

    def _fit_driver_stats(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Store per-numeric-feature stats for per-row driver explanations.

        For each numeric feature we keep its mean, std, RF importance and the
        sign of its correlation with the target. At inference we score a row's
        feature as ``importance * signed_zscore`` so a positive value means the
        feature pushes this policy *toward* higher lapse risk.
        """
        importances = self.feature_importances()
        y = y.astype(float)
        for col in self.numeric_features:
            series = pd.to_numeric(X[col], errors="coerce")
            std = float(series.std(ddof=0))
            corr = series.corr(y) if std > 0 else 0.0
            self._driver_stats[col] = {
                "mean": float(series.mean()),
                "std": std if std > 0 else 1.0,
                "importance": float(importances.get(col, 0.0)),
                "direction": 1.0 if (corr or 0.0) >= 0 else -1.0,
            }

    def explain_drivers(
        self, X: pd.DataFrame, probabilities: np.ndarray | None = None, top_k: int = 3
    ) -> list[str]:
        """Return a short human-readable lapse-driver string for each row.

        Combines global importance with how far each numeric feature deviates
        from the training mean *in the risk-increasing direction*. Only factors
        that raise risk are listed; rows with none get a neutral message.

        When ``probabilities`` are supplied, a row whose predicted risk is not
        above the training base rate is reported as low-risk rather than being
        given a list of "drivers" that would contradict its own near-zero
        probability.
        """
        contributions = pd.DataFrame(index=X.index)
        for col, stats in self._driver_stats.items():
            values = pd.to_numeric(X[col], errors="coerce")
            z = (values - stats["mean"]) / stats["std"]
            contributions[col] = stats["importance"] * stats["direction"] * z.fillna(0.0)

        prob = np.asarray(probabilities) if probabilities is not None else None
        drivers: list[str] = []
        for pos, (_, row) in enumerate(contributions.iterrows()):
            if prob is not None and prob[pos] <= self._base_rate:
                drivers.append("Risk at or below base rate; no elevated factors")
                continue
            positive = row[row > 0].sort_values(ascending=False).head(top_k)
            if positive.empty:
                drivers.append("No strongly elevated lapse-risk factors")
            else:
                parts = [f"{name}={self._fmt(X.loc[row.name, name])}" for name in positive.index]
                drivers.append("; ".join(parts))
        return drivers

    @staticmethod
    def _fmt(value) -> str:
        """Compactly format a feature value for the driver string."""
        if isinstance(value, float):
            return f"{value:.2f}"
        return str(value)

    # ------------------------------------------------------------------ #
    # Persistence
    # ------------------------------------------------------------------ #
    def save(self, path: str | Path) -> None:
        joblib.dump(self, Path(path))

    @staticmethod
    def load(path: str | Path) -> "LapseModel":
        return joblib.load(Path(path))
