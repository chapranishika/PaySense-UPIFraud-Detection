"""
================================================================================
  tests/test_category_training_v2_disjointness.py
  ----------------------------------------------------------------------------
  CATEGORY_CLASSIFIER_V2_ATTEMPT.md trains a candidate classifier on a blend
  of FinText-6K + two real-Kaggle-downloaded (self-evidently synthetic)
  datasets + a new, larger rule-based generator
  (generate_category_training_v2.py -> category_training_v2_synthetic.csv).

  The single most important discipline in that whole exercise is that NONE
  of this new training data may leak into either evaluation set:

    1. category_generalization_test_set.csv -- the 200-row GOLD, hand-
       authored, structurally-disjoint-from-FinText generalization test set.
       This is the number the whole V2 attempt is trying to move, honestly.
       If training data leaked into it, any accuracy gain would be
       meaningless.
    2. FinText-6K's own test_transaction_dataset.csv (1,000 rows) -- used to
       check the candidate hasn't regressed in-distribution. Also must never
       be trained on.

  This test file checks exact-text disjointness for both the new synthetic
  generator's output and each real-Kaggle-downloaded source, mirroring
  tests/test_ood_generalization_remediation.py's row-level leakage-check
  pattern and tests/test_category_generalization.py's structural-
  disjointness pattern for the fraud-model and original-classifier
  equivalents of this same discipline.
================================================================================
"""
import pathlib
import re

import pandas as pd
import pytest

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
GOLD_EVAL_CSV = BASE_DIR / "category_generalization_test_set.csv"
V2_SYNTH_CSV = BASE_DIR / "category_training_v2_synthetic.csv"
FINTEXT_DIR = pathlib.Path(r"E:\Projects\upi\FinText-6K")
FINTEXT_TEST_CSV = FINTEXT_DIR / "test_transaction_dataset.csv"
CODERANAND_DIR = BASE_DIR / "external_data" / "kaggle_coderanand_indian_banking"
BHAVYA_DIR = BASE_DIR / "external_data" / "kaggle_bhavya_financial_transaction"


def _normalize(text: str) -> str:
    return re.sub(r"\d+", "#", str(text))


@pytest.fixture(scope="module")
def gold_eval_df():
    if not GOLD_EVAL_CSV.exists():
        pytest.skip(f"{GOLD_EVAL_CSV} not present in this environment.")
    return pd.read_csv(GOLD_EVAL_CSV)


@pytest.fixture(scope="module")
def v2_synth_df():
    if not V2_SYNTH_CSV.exists():
        pytest.skip(f"{V2_SYNTH_CSV} not present -- run generate_category_training_v2.py first.")
    return pd.read_csv(V2_SYNTH_CSV)


def test_v2_synthetic_has_no_exact_text_overlap_with_gold_eval_set(v2_synth_df, gold_eval_df):
    gold_texts = set(gold_eval_df["text"].astype(str))
    overlap = v2_synth_df[v2_synth_df["text"].astype(str).isin(gold_texts)]
    assert overlap.empty, (
        f"{len(overlap)} rows in category_training_v2_synthetic.csv exactly "
        f"match a row in the gold evaluation set -- this is training-eval "
        f"leakage:\n{overlap['text'].tolist()}"
    )


def test_v2_synthetic_has_no_normalized_structure_overlap_with_gold_eval_set(v2_synth_df, gold_eval_df):
    """Stronger than exact-text: digit-collapsed structural match, the same
    normalization used to define FinText-6K's 40 templates and to check the
    gold eval set's disjointness from them."""
    gold_norms = set(gold_eval_df["text"].astype(str).apply(_normalize))
    v2_norms = v2_synth_df["text"].astype(str).apply(_normalize)
    overlap = v2_norms[v2_norms.isin(gold_norms)]
    assert overlap.empty, (
        f"{len(overlap)} rows in category_training_v2_synthetic.csv share an "
        f"exact normalized structure with a gold-eval-set row:\n{overlap.tolist()}"
    )


def test_v2_synthetic_has_no_duplicate_rows(v2_synth_df):
    assert v2_synth_df["text"].duplicated().sum() == 0


def test_v2_synthetic_has_meaningfully_more_structures_than_fintexts_40(v2_synth_df):
    """FinText-6K's entire 6,000-row dataset collapses to 40 unique
    digit-normalized structures (CATEGORY_CLASSIFIER_GENERALIZATION.md §1).
    The whole point of this generator is genuinely more structural variety,
    not just more rows of the same handful of shapes -- checked directly."""
    n_structures = v2_synth_df["text"].astype(str).apply(_normalize).nunique()
    assert n_structures > 100, (
        f"Only {n_structures} distinct normalized structures in the V2 "
        f"synthetic set -- not meaningfully more diverse than FinText-6K's 40."
    )


@pytest.mark.skipif(
    not FINTEXT_TEST_CSV.exists(),
    reason="FinText-6K source CSV not present in this environment.",
)
def test_v2_synthetic_has_no_exact_text_overlap_with_fintext_held_out_test(v2_synth_df):
    fintext_test = pd.read_csv(FINTEXT_TEST_CSV)
    fintext_texts = set(fintext_test["text"].astype(str))
    overlap = v2_synth_df[v2_synth_df["text"].astype(str).isin(fintext_texts)]
    assert overlap.empty, (
        f"{len(overlap)} rows in category_training_v2_synthetic.csv exactly "
        f"match a row in FinText-6K's own held-out test set."
    )


@pytest.mark.skipif(
    not CODERANAND_DIR.exists(),
    reason="external_data/kaggle_coderanand_indian_banking not present in this environment.",
)
def test_kaggle_coderanand_has_no_exact_text_overlap_with_gold_eval_set(gold_eval_df):
    train = pd.read_csv(CODERANAND_DIR / "financial_transaction_train.csv")
    test = pd.read_csv(CODERANAND_DIR / "financial_transaction_test.csv")
    combined_texts = set(train["Transaction_Text"].astype(str)) | set(test["Transaction_Text"].astype(str))
    gold_texts = set(gold_eval_df["text"].astype(str))
    overlap = combined_texts & gold_texts
    assert not overlap, f"{len(overlap)} coderanand rows exactly match a gold-eval-set row."


@pytest.mark.skipif(
    not BHAVYA_DIR.exists(),
    reason="external_data/kaggle_bhavya_financial_transaction not present in this environment.",
)
def test_kaggle_bhavya_has_no_exact_text_overlap_with_gold_eval_set(gold_eval_df):
    train = pd.read_csv(BHAVYA_DIR / "train_transactions.csv")
    test = pd.read_csv(BHAVYA_DIR / "test_transactions.csv")
    combined_texts = set(train["transaction_text"].astype(str)) | set(test["transaction_text"].astype(str))
    gold_texts = set(gold_eval_df["text"].astype(str))
    overlap = combined_texts & gold_texts
    assert not overlap, f"{len(overlap)} bhavya rows exactly match a gold-eval-set row."
