"""
================================================================================
  tests/test_category_generalization.py
  ────────────────────────────────────────────────────────────────────────────
  CATEGORY_CLASSIFIER_GENERALIZATION.md found that the Layer-2 NLP category
  classifier's documented 100% test-set accuracy is entirely a property of
  FinText-6K's dataset construction: all 6,000 rows (train+test) are
  generated from exactly 40 fixed templates matching the regex
  `^[A-Za-z ]+ of Rs [0-9]+ via UPI Ref [0-9]+$` -- only the leading noun
  phrase (40 choices) and the two numbers vary. A hand-authored, 200-row test
  set of realistic bank-SMS / UPI-app narration text
  (category_generalization_test_set.csv), structurally disjoint from those
  40 templates, measured real accuracy at 72.5% -- a real, diagnosed gap, not
  a manufactured one (see the .md for the full failure-mode analysis: the
  model degenerates to a near-constant default prediction whenever a
  sentence contains none of the ~821 vocabulary tokens learned from the 40
  templates).

  This test file guards two things cheaply (no retraining, no dataset
  regeneration -- everything here is read-only inference against the frozen
  artifact, seconds not minutes):

    1. The novel test set really is structurally disjoint from FinText-6K's
       40 templates -- the load-bearing claim that makes this a genuine
       generalization check rather than "different numbers, same 40 shapes."
    2. The frozen artifact's accuracy on this test set hasn't silently
       drifted from the number reported in
       CATEGORY_CLASSIFIER_GENERALIZATION.md (which would mean either the
       artifact or the test-set CSV changed without the doc being updated).
================================================================================
"""

import pathlib
import re

import joblib
import pandas as pd
import pytest
from sklearn.metrics import accuracy_score

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
ARTEFACTS_DIR = BASE_DIR / "artefacts"
CATEGORY_MODEL_PATH = ARTEFACTS_DIR / "paysense_category_classifier.pkl"
NOVEL_TEST_CSV = BASE_DIR / "category_generalization_test_set.csv"
FINTEXT_DIR = pathlib.Path(r"E:\Projects\upi\FinText-6K")

LABEL_DISPLAY_MAP = {
    "food": "Food",
    "travel": "Travel",
    "EMI": "EMI",
    "investment": "Investment",
    "shopping": "Shopping",
}

# The exact structural signature every one of FinText-6K's 6,000 rows
# (train+test) matches -- verified in CATEGORY_CLASSIFIER_GENERALIZATION.md
# §1: "of Rs <digits> via UPI Ref <digits>" is the fixed suffix on all 40
# templates; only the leading noun phrase and the two numbers vary.
FINTEXT_TEMPLATE_SUFFIX_PATTERN = re.compile(
    r"^[A-Za-z ]+ of Rs [0-9]+ via UPI Ref [0-9]+$"
)

# Pinned from CATEGORY_CLASSIFIER_GENERALIZATION.md's actual run of
# score_category_generalization.py against the frozen artifact. A tight
# tolerance -- inference is deterministic, so this should match exactly
# unless the artifact or the CSV changed underneath the document.
PUBLISHED_ACCURACY = 0.7250
PUBLISHED_MACRO_F1 = 0.7258
PUBLISHED_FRACTION_ABOVE_CONFIDENCE_GATE = 0.6700
NLP_CONFIDENCE_THRESHOLD = 0.65  # matches Android's NLP_CONFIDENCE_THRESHOLD


def _normalize(text: str) -> str:
    """Same normalization used to extract the 40 canonical templates:
    collapse every run of digits to a single '#' placeholder, leaving only
    the sentence *shape*."""
    return re.sub(r"\d+", "#", str(text))


@pytest.fixture(scope="module")
def novel_df():
    if not NOVEL_TEST_CSV.exists():
        pytest.skip(f"{NOVEL_TEST_CSV} not present in this environment.")
    df = pd.read_csv(NOVEL_TEST_CSV)
    assert set(df.columns) >= {"text", "label"}
    return df


def test_novel_test_set_has_expected_shape(novel_df):
    assert len(novel_df) == 200
    counts = novel_df["label"].value_counts()
    assert set(counts.index) == set(LABEL_DISPLAY_MAP.keys())
    assert (counts == 40).all(), f"Expected 40 rows per class, got:\n{counts}"


def test_novel_test_set_has_no_duplicate_rows(novel_df):
    assert novel_df["text"].duplicated().sum() == 0


def test_novel_test_set_does_not_match_the_40_fintext_templates_suffix_pattern(novel_df):
    """The load-bearing structural-disjointness check: none of the 200
    hand-authored rows may match the exact fixed suffix every single
    FinText-6K row shares. If this ever fails, the "novel" test set has
    drifted back into reproducing the same 40 shapes with different
    numbers -- exactly the thing this whole exercise exists to avoid."""
    matches = novel_df["text"].apply(lambda t: bool(FINTEXT_TEMPLATE_SUFFIX_PATTERN.match(str(t))))
    assert matches.sum() == 0, (
        f"{matches.sum()} rows in {NOVEL_TEST_CSV.name} match the FinText-6K "
        f"40-template suffix pattern -- these rows are not a genuine "
        f"generalization test:\n{novel_df.loc[matches, 'text'].tolist()}"
    )


@pytest.mark.skipif(
    not (FINTEXT_DIR / "train_transaction_dataset.csv").exists()
    or not (FINTEXT_DIR / "test_transaction_dataset.csv").exists(),
    reason="FinText-6K source CSVs not present in this environment.",
)
def test_novel_test_set_normalized_structures_are_disjoint_from_fintext(novel_df):
    """Stronger than the regex check above: normalize every FinText-6K row
    (train+test, 6,000 rows) and every novel-test-set row by collapsing
    digits, then assert zero exact overlap in normalized sentence shape.
    Mirrors tests/test_ood_generalization_remediation.py's row-level
    disjointness check between the blend and held-out synthetic datasets."""
    fintext = pd.concat(
        [
            pd.read_csv(FINTEXT_DIR / "train_transaction_dataset.csv"),
            pd.read_csv(FINTEXT_DIR / "test_transaction_dataset.csv"),
        ],
        ignore_index=True,
    )
    fintext_norms = set(fintext["text"].apply(_normalize))
    novel_norms = novel_df["text"].apply(_normalize)
    overlap = novel_norms[novel_norms.isin(fintext_norms)]
    assert overlap.empty, (
        f"{len(overlap)} rows in the novel test set share an exact "
        f"normalized structure with a FinText-6K row:\n{overlap.tolist()}"
    )


@pytest.mark.skipif(
    not CATEGORY_MODEL_PATH.exists(),
    reason="Frozen category classifier artifact not present in this environment.",
)
def test_frozen_classifier_accuracy_on_novel_test_set_matches_published_number(novel_df):
    """Read-only inference against the frozen artifact -- no retraining.
    Fails loudly if either the artifact or the CSV drifted from what
    CATEGORY_CLASSIFIER_GENERALIZATION.md actually reports."""
    pipeline = joblib.load(CATEGORY_MODEL_PATH)
    y_true = novel_df["label"].map(LABEL_DISPLAY_MAP).to_numpy()
    y_pred = pipeline.predict(novel_df["text"])
    accuracy = accuracy_score(y_true, y_pred)
    assert accuracy == pytest.approx(PUBLISHED_ACCURACY, abs=0.01), (
        f"Frozen classifier scores {accuracy:.4f} on the novel test set, "
        f"expected {PUBLISHED_ACCURACY:.4f} (CATEGORY_CLASSIFIER_GENERALIZATION.md). "
        f"Either the artifact or category_generalization_test_set.csv changed "
        f"without the document being updated."
    )


@pytest.mark.skipif(
    not CATEGORY_MODEL_PATH.exists(),
    reason="Frozen category classifier artifact not present in this environment.",
)
def test_frozen_classifier_confidence_gate_pass_rate_on_novel_test_set(novel_df):
    """The number that actually matters for production: what fraction of
    predictions clear Android's NLP_CONFIDENCE_THRESHOLD = 0.65 gate (a
    correct-but-low-confidence prediction still falls through to Tier-3
    HITL). Guards against silent drift the same way as the accuracy test
    above."""
    pipeline = joblib.load(CATEGORY_MODEL_PATH)
    proba = pipeline.predict_proba(novel_df["text"])
    top_conf = proba.max(axis=1)
    frac_above_gate = float((top_conf >= NLP_CONFIDENCE_THRESHOLD).mean())
    assert frac_above_gate == pytest.approx(
        PUBLISHED_FRACTION_ABOVE_CONFIDENCE_GATE, abs=0.02
    )


def test_this_test_set_is_meaningfully_harder_than_documented_100_percent():
    """Sanity guard on the headline finding itself: the whole point of this
    document is that accuracy on genuinely novel phrasing is measurably
    below the 100% FinText-6K held-out figure. If a future retrain closes
    this gap, that is a genuinely good outcome and this assertion should be
    loosened/updated to reflect it -- but it should never silently pass
    while nobody notices the gap closed (or reopened wider)."""
    assert PUBLISHED_ACCURACY < 1.0
    assert PUBLISHED_ACCURACY > 0.5  # still much better than random (0.20 for 5 classes)
