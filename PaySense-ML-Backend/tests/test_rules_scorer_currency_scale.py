"""
================================================================================
  tests/test_rules_scorer_currency_scale.py
  ----------------------------------------------------------------------------
  REAL_DATA_AND_RESEARCH_GROUNDING.md found Dataset 5 (a real, well-vetted
  fraud dataset with a genuinely strong usr_account_age_days signal) scoring
  at chance through the full ensemble, despite having more honest feature
  overlap than any real dataset used before. This tests the hypothesis for
  why: src.fraud_model._score_rules()'s cold-start bonus is gated on
  `amount > 5000`, a threshold calibrated to PaySense's own INR-scale
  training data. Dataset 5's purchase_value column (USD-denominated
  e-commerce data) ranges 9-154 across all 151,112 rows -- the gate can
  never fire.

  Read-only: no retraining, no artifact changes, does not modify
  src/fraud_model.py (production only ever sees real UPI/INR traffic, so
  there is no real bug to fix there -- this is a confirmed evaluation
  artifact, not a production defect). Isolates the rules scorer's own
  contribution (not the full ensemble) to cleanly measure the effect
  without the other two scorers diluting the signal.
================================================================================
"""
import pathlib
import sys

import numpy as np
import pytest
from sklearn.metrics import roc_auc_score

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))

DATASET5_CSV = BASE_DIR / "external_data" / "kaggle_vbinh002_fraud_ecommerce" / "Fraud_Data.csv"


def _rules_score_row(row: dict, amount_threshold: float) -> float:
    """Reproduces src.fraud_model._score_rules() exactly, parameterized by
    the cold-start amount threshold (5000 = the real production value)."""
    score = 0.02
    if row.get("new_device_flag"):
        score += 0.35
    if row.get("ip_location_mismatch"):
        score += 0.20
    if not row.get("kyc_verified_flag"):
        score += 0.15
    if row.get("usr_is_high_risk"):
        score += 0.12
    amt_dev = row.get("amount_deviation_score", 0.0) or 0.0
    if amt_dev > 4.0:
        score += 0.20
    elif amt_dev > 2.0:
        score += 0.10
    elif amt_dev > 1.0:
        score += 0.03
    if row.get("is_night_transaction"):
        score += 0.05
    if (row.get("failed_attempts_last_24h", 0) or 0) > 2:
        score += 0.08
    acc_age = row.get("usr_account_age_days", 999) or 999
    if acc_age < 30 and row.get("amount", 0) > amount_threshold:
        score += 0.08
    return score


@pytest.mark.skipif(
    not DATASET5_CSV.exists(),
    reason="Dataset 5 (external_data/kaggle_vbinh002_fraud_ecommerce/Fraud_Data.csv) not present in this environment.",
)
def test_currency_scale_mismatch_confirmed_and_fixable():
    from real_data_and_research_grounding import load_dataset_5

    mapped, y, _meta = load_dataset_5()
    records = mapped.to_dict(orient="records")

    max_amount = mapped["amount"].max()
    assert max_amount < 5000, (
        f"Expected Dataset 5's amounts (max {max_amount:.2f}) to stay far "
        f"below the production amount>5000 gate -- if this no longer holds, "
        f"the currency-scale finding in REAL_DATA_AND_RESEARCH_GROUNDING.md "
        f"needs re-checking, not assuming."
    )

    scores_original = np.array([_rules_score_row(r, 5000) for r in records])
    n_fired_original = sum(
        1 for r in records
        if (r.get("usr_account_age_days", 999) or 999) < 30 and r.get("amount", 0) > 5000
    )
    assert n_fired_original == 0, (
        "Expected the original threshold to never fire on Dataset 5 -- if "
        "it now fires, the currency-scale finding no longer holds as stated."
    )
    auc_original = roc_auc_score(y, scores_original)
    assert auc_original < 0.55, (
        f"Expected the original (unfixed) rules scorer to perform near "
        f"chance on Dataset 5 (got ROC-AUC={auc_original:.4f})."
    )

    scores_fixed = np.array([_rules_score_row(r, 0) for r in records])
    auc_fixed = roc_auc_score(y, scores_fixed)
    assert auc_fixed > auc_original + 0.15, (
        f"Expected removing the currency-scale-mismatched gate to "
        f"meaningfully improve the rules scorer's own ranking power on "
        f"Dataset 5 (original={auc_original:.4f}, fixed={auc_fixed:.4f})."
    )
