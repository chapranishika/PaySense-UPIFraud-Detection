"""
================================================================================
  tests/test_ood_threshold_sweep_variant_a.py
  ────────────────────────────────────────────────────────────────────────────
  OOD_GENERALIZATION_REMEDIATION.md left one question open on purpose:
  Variant A (blended-training candidate model) ranks real out-of-distribution
  fraud better than the deployed model on every real check, but still scored
  0/701 and 0/64 true positives at the deployed threshold (0.30) -- because
  its scores never get close to it. Nobody had tested whether a much lower,
  OOD-specific threshold on Variant A's own improved ranking would actually
  recover any of those missed real fraud rows.

  ood_threshold_sweep_variant_a.py answered it: on Dataset 1 (the primary
  real-world check), a real, usable, non-zero operating point exists --
  something no check this project has ever run found before, at any
  threshold, on this dataset. This is read-only inference against the
  already-saved paysense_model_blended_training.pkl (no retraining), so
  this test just guards the pinned numbers from silent drift the same way
  every other frozen-artifact test in this suite does.

  What this test does NOT claim: that this threshold is validated or
  production-ready. It was found by sweeping directly against Dataset 1's
  own labels, not a separate calibration split -- the honest read is "this
  specific dataset, in retrospect," not "this generalizes to the next
  external dataset." See OOD_GENERALIZATION_REMEDIATION.md §7 for the full
  caveat and why Variant A remains undeployed.
================================================================================
"""
import json
import pathlib

import pytest

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
RESULTS_JSON = BASE_DIR / "ood_threshold_sweep_variant_a_results.json"


def _skip_if_missing():
    if not RESULTS_JSON.exists():
        pytest.skip(f"{RESULTS_JSON} not present -- run ood_threshold_sweep_variant_a.py first.")


def _row_at(rows, threshold):
    for r in rows:
        if r["threshold"] == pytest.approx(threshold, abs=1e-6):
            return r
    raise AssertionError(f"No row at threshold={threshold} in results (available: "
                          f"{[r['threshold'] for r in rows]})")


def test_dataset1_still_zero_recall_at_the_actual_deployed_threshold():
    """Confirms this experiment does not silently contradict every other
    check tonight: at the real deployed threshold (0.30), Variant A recovers
    exactly as little real fraud on Dataset 1 as the baseline model does --
    0/701. The improvement this file documents only exists at a much lower,
    non-default threshold nothing in production actually uses."""
    _skip_if_missing()
    results = json.loads(RESULTS_JSON.read_text())
    row = _row_at(results["dataset1"], 0.30)
    assert row["tp"] == 0
    assert row["recall"] == 0.0


def test_dataset1_first_usable_operating_point_is_real_not_manufactured():
    """The headline finding: at threshold=0.06, Variant A catches 37/701
    (5.28%) of Dataset 1's real fraud rows with ZERO false positives out of
    74,216 legitimate rows (100% precision). This is the first time any
    check in this project has found ANY real fraud caught on this dataset,
    at any threshold -- a small, genuinely real capability, not a rounding
    artifact (checked: 0 FP, not close to 0)."""
    _skip_if_missing()
    results = json.loads(RESULTS_JSON.read_text())
    row = _row_at(results["dataset1"], 0.06)
    assert row["tp"] == 37
    assert row["fp"] == 0
    assert row["precision"] == pytest.approx(1.0, abs=1e-6)
    assert row["recall"] == pytest.approx(37 / 701, abs=1e-4)


def test_dataset1_higher_recall_option_has_a_real_quantified_cost():
    """A more aggressive threshold (0.055) recovers far more real fraud
    (45.5%) but the cost has to be reported alongside it, not hidden: 283
    legitimate transactions out of 74,634 would also be flagged at that
    setting. Both numbers are pinned so neither can silently drift out of
    sync with the other."""
    _skip_if_missing()
    results = json.loads(RESULTS_JSON.read_text())
    row = _row_at(results["dataset1"], 0.055)
    assert row["tp"] == 319
    assert row["fp"] == 283
    assert row["recall"] == pytest.approx(319 / 701, abs=1e-4)


def test_dataset1_scores_collapse_below_threshold_0_035():
    """Below 0.035, Variant A's score distribution on Dataset 1 stops
    discriminating anything -- literally every row (fraud and legitimate
    alike) scores above the threshold, so "recall" hits 100% only by
    flagging the entire dataset (0.94% precision, useless). This guards
    against someone reading the sweep table in isolation and picking a
    threshold below the real usable range."""
    _skip_if_missing()
    results = json.loads(RESULTS_JSON.read_text())
    row = _row_at(results["dataset1"], 0.03)
    assert row["tp"] == 701  # every fraud row...
    assert row["fp"] == 74216  # ...and every legitimate row too.
    assert row["false_positive_rate"] == pytest.approx(1.0, abs=1e-6)


def test_dataset3_has_no_usable_middle_ground_unlike_dataset1():
    """Dataset 3 (only 5% of features honestly mappable, vs. Dataset 1's
    15%) shows a fundamentally different, worse shape: no gradual recall
    ramp exists at all. The jump is a cliff straight from 0 recall to 100%
    recall / 93.6% false-positive-rate between threshold 0.04 and 0.035 --
    confirming the remediation found for Dataset 1 does not generalize to
    lower-feature-coverage real data."""
    _skip_if_missing()
    results = json.loads(RESULTS_JSON.read_text())
    just_above = _row_at(results["dataset3"], 0.04)
    just_below = _row_at(results["dataset3"], 0.035)
    assert just_above["tp"] == 0
    assert just_below["tp"] == 64  # all 64, nothing in between
    assert just_below["fp"] == 936  # every one of the 936 legitimate rows too
    assert just_below["false_positive_rate"] == pytest.approx(1.0, abs=1e-6)
