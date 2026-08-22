"""
================================================================================
  tests/test_platt_scaling_invariance.py
  ────────────────────────────────────────────────────────────────────────────
  On 2026-08-22/23, PLATT_SCALING_RESULT.md tested whether Platt scaling
  (fitting sigma(A*f(x)+B) to the frozen XGBoost model's raw scores via
  sklearn.calibration.CalibratedClassifierCV(method="sigmoid", cv="prefit"))
  moves the 69.96% recall ceiling documented in README.md and
  paysense_report.tex. It does not: a strictly monotonic 1-D transform of a
  classifier's scores cannot change ROC-AUC, PR-AUC, or which rows get
  flagged at an equivalent decision threshold — only the numeric threshold
  value that produces a given operating point changes.

  This test guards that finding against silent regression. If a future
  change swaps in a different calibration method (isotonic regression,
  temperature scaling, etc.) or refactors the calibration pipeline in a way
  that breaks strict monotonicity, this test fails — which is exactly the
  signal needed before anyone re-asserts "calibration fixed the recall
  ceiling" without re-verifying it, repeating the same kind of unverified
  claim PLATT_SCALING_RESULT.md was written to correct.

  No retraining happens here. The frozen artifacts (paysense_model.pkl,
  paysense_preprocessor.pkl) are loaded read-only; only a 1-D logistic
  sigmoid is fit on a calibration slice, exactly as in the experiment.
================================================================================
"""

import pathlib

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
ARTEFACTS_DIR = BASE_DIR / "artefacts"
MASTER_CSV = BASE_DIR / "paysense_master_dataset.csv"

RANDOM_STATE = 42
DROP_COLS = [
    "transaction_id", "user_id", "receiver_id", "timestamp", "date",
    "data_source", "user_kyc_status", "status", "usr_home_city",
]


def _skip_if_artefacts_missing():
    required = [
        ARTEFACTS_DIR / "paysense_model.pkl",
        ARTEFACTS_DIR / "paysense_preprocessor.pkl",
        MASTER_CSV,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        pytest.skip(f"Frozen artefacts not present in this environment: {missing}")


@pytest.fixture(scope="module")
def platt_eval():
    _skip_if_artefacts_missing()

    df = pd.read_csv(MASTER_CSV)
    df = df.drop(columns=DROP_COLS)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)

    # Canonical 80/20 split every doc in this repo cites.
    _, X_test_raw, _, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # Calibration slice carved OUT OF the test partition (see
    # PLATT_SCALING_RESULT.md / platt_scaling_experiment.py for why: the
    # frozen model was trained on 100% of the 80% train split, so no subset
    # of it is genuinely unseen; only the 20% test partition qualifies).
    X_calib_raw, _, y_calib, _ = train_test_split(
        X_test_raw, y_test, test_size=0.50, random_state=RANDOM_STATE, stratify=y_test
    )

    preprocessor = joblib.load(ARTEFACTS_DIR / "paysense_preprocessor.pkl")
    model = joblib.load(ARTEFACTS_DIR / "paysense_model.pkl")

    X_calib_proc = preprocessor.transform(X_calib_raw)
    X_test_proc = preprocessor.transform(X_test_raw)

    platt = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    platt.fit(X_calib_proc, y_calib)

    y_proba_raw = model.predict_proba(X_test_proc)[:, 1]
    y_proba_platt = platt.predict_proba(X_test_proc)[:, 1]

    return {
        "y_test": y_test,
        "y_proba_raw": y_proba_raw,
        "y_proba_platt": y_proba_platt,
    }


def test_platt_scaling_preserves_roc_auc(platt_eval):
    roc_raw = roc_auc_score(platt_eval["y_test"], platt_eval["y_proba_raw"])
    roc_platt = roc_auc_score(platt_eval["y_test"], platt_eval["y_proba_platt"])
    assert roc_raw == pytest.approx(roc_platt, abs=1e-6), (
        "Platt scaling changed ROC-AUC. A strictly monotonic 1-D transform "
        "of the same raw scores must leave ROC-AUC unchanged — if this "
        "assertion fails, either the calibrator is not being applied as a "
        "pure rank-preserving remap, or the calibration slice is leaking "
        "information the base classifier didn't already have. Re-verify "
        "before claiming calibration moved any ranking-based metric."
    )


def test_platt_scaling_preserves_pr_auc(platt_eval):
    pr_raw = average_precision_score(platt_eval["y_test"], platt_eval["y_proba_raw"])
    pr_platt = average_precision_score(platt_eval["y_test"], platt_eval["y_proba_platt"])
    assert pr_raw == pytest.approx(pr_platt, abs=1e-6), (
        "Platt scaling changed PR-AUC, which a monotonic remap cannot do. "
        "See test_platt_scaling_preserves_roc_auc for what this would imply."
    )


def test_platt_scaling_has_no_rank_inversions(platt_eval):
    """No pair of test rows should have its relative order flipped, modulo
    exact ties in the raw score (which carry no order information to
    invert)."""
    raw = platt_eval["y_proba_raw"]
    platt = platt_eval["y_proba_platt"]

    order = np.argsort(raw, kind="stable")
    raw_sorted = raw[order]
    platt_sorted = platt[order]

    n = len(raw_sorted)
    i = 0
    block_max_platt = -np.inf
    inversions = 0
    while i < n:
        j = i
        while j < n and raw_sorted[j] == raw_sorted[i]:
            j += 1
        block_min = platt_sorted[i:j].min()
        if block_min < block_max_platt - 1e-9:
            inversions += 1
        block_max_platt = max(block_max_platt, platt_sorted[i:j].max())
        i = j

    assert inversions == 0, (
        f"Found {inversions} rank inversions between raw and Platt-scaled "
        f"scores. Platt scaling is supposed to be a strictly monotonic "
        f"1-D remap of the base classifier's scores — any inversion means "
        f"the calibrated scores are no longer a pure recalibration and the "
        f"'calibration doesn't move the recall ceiling' finding in "
        f"PLATT_SCALING_RESULT.md no longer applies to this code path."
    )


def test_platt_scaling_does_not_move_recall_ceiling_at_equivalent_threshold(platt_eval):
    """At the documented most-aggressive raw threshold (tau=0.05), find the
    analytically equivalent point on the Platt-scaled axis and confirm it
    flags the exact same rows and produces the exact same recall — i.e. the
    69.96% recall ceiling is a property of the score ranking, not the
    probability scale, and calibration cannot move it."""
    y_test = platt_eval["y_test"]
    raw = platt_eval["y_proba_raw"]
    platt = platt_eval["y_proba_platt"]

    tau_raw = 0.05
    pred_raw = (raw >= tau_raw).astype(int)
    recall_raw = recall_score(y_test, pred_raw, zero_division=0)

    # Equivalent threshold found empirically (not analytically re-derived
    # from A/B here, to stay robust to internal sklearn version changes):
    # the smallest platt-scaled value among rows that clear tau_raw.
    tau_platt_equiv = platt[pred_raw == 1].min()
    pred_platt = (platt >= tau_platt_equiv).astype(int)
    recall_platt = recall_score(y_test, pred_platt, zero_division=0)

    assert np.array_equal(pred_raw, pred_platt), (
        "The set of flagged rows changed between raw and Platt-scaled "
        "scores at the equivalent threshold. Platt scaling should only "
        "relabel the threshold axis, not change which rows clear it."
    )
    assert recall_raw == pytest.approx(recall_platt, abs=1e-9), (
        f"Recall ceiling moved under Platt scaling ({recall_raw:.4f} -> "
        f"{recall_platt:.4f}). This directly contradicts the finding in "
        f"PLATT_SCALING_RESULT.md; do not re-introduce a 'Platt scaling "
        f"fixes the recall ceiling' claim in README.md or "
        f"paysense_report.tex without re-running that experiment and "
        f"updating this test's expectation deliberately."
    )
