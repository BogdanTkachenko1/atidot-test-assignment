"""Shared utilities for the Atidot lapse + segmentation pipeline.

This module deliberately holds only generic, reusable helpers (logging, IO,
date parsing, schema reconciliation). Domain logic lives in ``features.py``,
``model.py`` and ``segmentation.py`` so the pieces stay decoupled and testable.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

# Average number of days per month, used to convert a date delta into a
# fractional month count for tenure. Kept as a constant so the choice is
# explicit and consistent everywhere it is used.
DAYS_PER_MONTH = 30.4375


def get_logger(name: str = "atidot") -> logging.Logger:
    """Return a module logger configured for simple, single-line console output.

    Idempotent: repeated calls do not stack duplicate handlers.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)-7s | %(message)s"))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger


def read_policies_csv(path: str | Path, date_columns: Iterable[str] = ("snapshot_date", "issue_date")) -> pd.DataFrame:
    """Load a policy CSV and parse known date columns.

    Date columns are parsed only if present so the same loader works for the
    training file and for new inference files that may omit optional columns.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Input CSV not found: {path}")
    df = pd.read_csv(path)
    for col in date_columns:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")
    return df


def compute_tenure_months(df: pd.DataFrame) -> pd.Series:
    """Months between ``issue_date`` and ``snapshot_date`` (policy age at snapshot).

    Negative values (snapshot before issue, a known data-quality quirk) are
    clipped to 0. Missing dates yield NaN, which downstream imputation handles.
    """
    if "snapshot_date" not in df.columns or "issue_date" not in df.columns:
        return pd.Series(np.nan, index=df.index, name="tenure_months")
    delta_days = (df["snapshot_date"] - df["issue_date"]).dt.days
    tenure = (delta_days / DAYS_PER_MONTH).clip(lower=0)
    return tenure.rename("tenure_months")


def reconcile_columns(df: pd.DataFrame, expected: Iterable[str]) -> pd.DataFrame:
    """Return a copy of ``df`` containing exactly the ``expected`` columns.

    Missing columns are added as NaN (so imputation can fill them) and extra
    columns are dropped. This is the single place that makes inference robust
    to "missing optional column / extra irrelevant column" schema drift.
    """
    expected = list(expected)
    out = df.copy()
    for col in expected:
        if col not in out.columns:
            out[col] = np.nan
    return out[expected]


def ensure_dir(path: str | Path) -> Path:
    """Create ``path`` as a directory (including parents) and return it."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(obj: dict, path: str | Path) -> None:
    """Write ``obj`` to ``path`` as indented JSON (numpy-safe)."""
    path = Path(path)
    with path.open("w") as fh:
        json.dump(obj, fh, indent=2, default=_json_default)


def load_json(path: str | Path) -> dict:
    """Load a JSON file into a dict."""
    with Path(path).open() as fh:
        return json.load(fh)


def _json_default(value):
    """Fallback serializer so numpy scalars/arrays survive ``json.dump``."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value)} is not JSON serializable")
