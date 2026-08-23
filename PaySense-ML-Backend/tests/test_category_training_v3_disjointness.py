"""
================================================================================
  tests/test_category_training_v3_disjointness.py
  ────────────────────────────────────────────────────────────────────────────
  v2 (generate_category_training_v2.py) was invalidated because its
  templates turned out to be the gold eval set's own sentence skeletons
  with only the merchant name swapped -- the existing digit-collapsing
  disjointness check didn't catch it because it doesn't mask merchant
  names. v3 (generate_category_training_v3.py) was built without ever
  reading the eval set's content, and passed a STRONGER check (merchant-
  name masking, not just digits) at generation time. This test guards that
  stronger check as a permanent regression guard, mirroring
  tests/test_category_generalization.py's weaker digit-only check.
================================================================================
"""
import pathlib
import re

import pandas as pd
import pytest

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
V3_CSV = BASE_DIR / "category_training_v3_synthetic.csv"
EVAL_CSV = BASE_DIR / "category_generalization_test_set.csv"


def _strong_skeleton(text: str) -> str:
    """Same masking used by generate_category_training_v3.py's own gate:
    collapse digits, then collapse runs of capitalized words (likely
    merchant/proper nouns) and account-number-like XX# tokens."""
    t = re.sub(r"\d+", "#", str(text))
    t = re.sub(r"\b([A-Z][A-Za-z]*(?:[\s.]+[A-Z][A-Za-z.]*){1,5})\b", "MERCHANT", t)
    t = re.sub(r"XX#+", "ACCT", t)
    return t.lower().strip()


@pytest.mark.skipif(
    not (V3_CSV.exists() and EVAL_CSV.exists()),
    reason="v3 training set and/or gold eval set not present in this environment.",
)
def test_v3_training_data_has_no_exact_overlap_with_eval_set():
    v3 = pd.read_csv(V3_CSV)
    ev = pd.read_csv(EVAL_CSV)
    overlap = set(v3["text"]) & set(ev["text"])
    assert not overlap, (
        f"{len(overlap)} exact-text rows appear in both the v3 training set "
        f"and the gold eval set -- this is the exact v2 contamination "
        f"pattern, must be zero."
    )


@pytest.mark.skipif(
    not (V3_CSV.exists() and EVAL_CSV.exists()),
    reason="v3 training set and/or gold eval set not present in this environment.",
)
def test_v3_training_data_has_no_merchant_masked_skeleton_overlap_with_eval_set():
    """The check v2's own disjointness test would have MISSED (it only
    masks digits) -- this masks merchant-name-like tokens too, which is
    exactly what let v2's "same skeleton, different merchant" contamination
    slip through undetected."""
    v3 = pd.read_csv(V3_CSV)
    ev = pd.read_csv(EVAL_CSV)
    v3_skel = set(v3["text"].apply(_strong_skeleton))
    ev_skel = set(ev["text"].apply(_strong_skeleton))
    overlap = v3_skel & ev_skel
    assert not overlap, (
        f"{len(overlap)} merchant-masked skeleton(s) appear in both the v3 "
        f"training set and the gold eval set: {list(overlap)[:5]} -- this is "
        f"the specific contamination pattern that invalidated v2."
    )
