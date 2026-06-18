"""Rule-based segmentation engine.

The four segment definitions are transcribed *manually* from the PDFs in
``data/segment_docs`` into deterministic, inspectable rules. Each segment has:

* a **gate** — the single defining condition from the PDF (the "Age" / "Duration"
  headline signal). The gate is *mandatory*: a policy that fails it is not
  eligible for the segment at all. This prevents nonsensical assignments such
  as a 2-month-old policy being labelled ``legacy_stable_value``.
* a list of **signals** — the remaining named conditions. Confidence is the
  fraction of (gate + signals) that match, so a gate-only match still yields a
  small positive confidence, and a full match yields 1.0.

The explanation lists exactly which conditions fired, keeping assignment fully
explainable and reproducible.

PDF source (gate | signals per segment):
  segment_1 high_income_urban_builder    : age 35-55 | income>120k, face>=500k, urban
  segment_2 post_retirement_high_coverage: age 65+   | face>=500k or cash>=50k, non-term product
  segment_3 early_tenure_payment_friction: tenure<18m| payment friction, direct/low-touch
  segment_4 legacy_stable_value          : tenure>=84m| no missed pmts, autopay, agent touch
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd

from .utils import compute_tenure_months

# A predicate takes a row (with raw fields + ``tenure_months``) -> bool.
Predicate = Callable[[pd.Series], bool]

# Products that are not term life (used by the post-retirement segment).
NON_TERM_PRODUCTS = {"Whole Life", "Universal Life", "Indexed Universal Life"}


@dataclass(frozen=True)
class Rule:
    """A single named, human-readable segment signal."""

    description: str
    predicate: Predicate

    def matches(self, row: pd.Series) -> bool:
        # NaN-safe: any error or NaN comparison is treated as "not matched".
        try:
            return bool(self.predicate(row))
        except (TypeError, ValueError):
            return False


@dataclass(frozen=True)
class Segment:
    """A business segment: a label, a mandatory ``gate`` and supporting signals."""

    name: str
    gate: Rule
    signals: list[Rule]

    @property
    def n_total(self) -> int:
        """Total conditions defining the segment (gate counts as one)."""
        return 1 + len(self.signals)

    def score(self, row: pd.Series) -> tuple[float, list[str]]:
        """Return ``(confidence, matched_descriptions)`` for one policy row.

        The gate is mandatory: if it does not match, the segment is not eligible
        (confidence 0). Otherwise confidence is the fraction of all conditions
        (gate + signals) that match, so the matched gate alone already yields a
        positive score and a full match yields 1.0.
        """
        if not self.gate.matches(row):
            return 0.0, []
        matched = [self.gate.description] + [s.description for s in self.signals if s.matches(row)]
        confidence = len(matched) / self.n_total
        return confidence, matched


def _num(row: pd.Series, key: str) -> float:
    """Safely read a numeric field as float (NaN if absent/unparseable)."""
    try:
        return float(row.get(key, np.nan))
    except (TypeError, ValueError):
        return np.nan


def _str(row: pd.Series, key: str) -> str:
    """Safely read a categorical field as a string (empty if absent/NaN)."""
    value = row.get(key, None)
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return ""
    return str(value)


class SegmentationEngine:
    """Assigns each policy snapshot to its best-matching business segment."""

    UNSEGMENTED = "unsegmented"

    def __init__(self) -> None:
        self.segments: list[Segment] = self._build_segments()

    # ------------------------------------------------------------------ #
    # Segment definitions (transcribed from the PDFs)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_segments() -> list[Segment]:
        return [
            Segment(
                "high_income_urban_builder",
                gate=Rule("age 35-55", lambda r: 35 <= _num(r, "current_age") <= 55),
                signals=[
                    Rule("income > 120k", lambda r: _num(r, "income_estimate") > 120_000),
                    Rule("face amount >= 500k", lambda r: _num(r, "face_amount") >= 500_000),
                    Rule("urbanicity = urban", lambda r: _str(r, "urbanicity") == "urban"),
                ],
            ),
            Segment(
                "post_retirement_high_coverage",
                gate=Rule("age 65+", lambda r: _num(r, "current_age") >= 65),
                signals=[
                    Rule(
                        "face >= 500k or cash value >= 50k",
                        lambda r: _num(r, "face_amount") >= 500_000 or _num(r, "cash_value") >= 50_000,
                    ),
                    Rule("non-term product", lambda r: _str(r, "product_type") in NON_TERM_PRODUCTS),
                ],
            ),
            Segment(
                "early_tenure_payment_friction",
                gate=Rule("tenure < 18 months", lambda r: _num(r, "tenure_months") < 18),
                signals=[
                    Rule(
                        "payment friction (monthly / no autopay / missed payments)",
                        lambda r: (
                            _str(r, "payment_mode") == "monthly"
                            or _num(r, "autopay_flag") == 0
                            or _num(r, "missed_payment_count_6m") > 0
                        ),
                    ),
                    Rule(
                        "direct channel or low agent touch",
                        lambda r: _str(r, "distribution_channel") == "Direct" or _num(r, "agent_touch_count_12m") <= 1,
                    ),
                ],
            ),
            Segment(
                "legacy_stable_value",
                gate=Rule("tenure >= 84 months", lambda r: _num(r, "tenure_months") >= 84),
                signals=[
                    Rule("no missed payments", lambda r: _num(r, "missed_payment_count_6m") == 0),
                    Rule("autopay enabled", lambda r: _num(r, "autopay_flag") == 1),
                    Rule("agent touch in last 12m", lambda r: _num(r, "agent_touch_count_12m") > 0),
                ],
            ),
        ]

    # ------------------------------------------------------------------ #
    # Assignment
    # ------------------------------------------------------------------ #
    def assign_row(self, row: pd.Series) -> tuple[str, float, str]:
        """Return ``(segment_name, confidence, reason)`` for a single row.

        Only segments whose mandatory gate matches are eligible. Among those,
        the highest-confidence segment wins; ties break by definition order
        (most specific segments are listed first). A policy that is eligible for
        no segment is labelled ``unsegmented``.
        """
        best: Segment | None = None
        best_conf, best_matched = 0.0, []
        for segment in self.segments:
            confidence, matched = segment.score(row)
            if confidence > best_conf:
                best, best_conf, best_matched = segment, confidence, matched

        if best is None:
            return self.UNSEGMENTED, 0.0, "No segment gate matched"

        reason = f"{'; '.join(best_matched)} ({len(best_matched)}/{best.n_total} signals matched)"
        return best.name, round(best_conf, 3), reason

    def assign(self, df: pd.DataFrame) -> pd.DataFrame:
        """Assign every row in ``df``; returns name/confidence/reason columns."""
        work = df.copy()
        work["tenure_months"] = compute_tenure_months(work)
        records = [self.assign_row(row) for _, row in work.iterrows()]
        return pd.DataFrame(
            records, columns=["segment_name", "segment_confidence", "segment_reason"], index=df.index
        )
