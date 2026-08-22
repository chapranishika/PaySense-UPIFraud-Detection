"""
================================================================================
  PaySense — Platt Scaling Experiment
  ────────────────────────────────────────────────────────────────────────────
  Question: does Platt scaling (fitting sigma(A*f(x)+B) to the frozen XGBoost
  model's raw scores) move the 69.96% recall ceiling documented in README.md
  and paysense_report.tex, or does it only remap the probability *scale*
  while leaving the rank order of predictions (and therefore the achievable
  precision/recall frontier) untouched?

  NO RETRAINING happens here. artefacts/paysense_model.pkl,
  paysense_preprocessor.pkl, and paysense_threshold.pkl are loaded read-only
  and never overwritten. This script only reproduces paysense_phase3.py's
  Block 0 split (random_state=42, same DROP_COLS, same target) to regenerate
  the identical 80/20 test set every other doc in this repo cites, then
  carves calibration data out of genuinely unseen rows.

  LEAKAGE NOTE: the frozen XGBoost model was trained on 100% of the 80%
  train split (paysense_phase3.py Block 0 fits SMOTE + XGBoost on the whole
  train_raw partition, no internal validation carve-out). That means there
  is no subset of the 80% train partition that is unseen by the frozen
  model — the ONLY genuinely held-out rows available without retraining are
  inside the 20% test partition. So the calibration set is carved OUT OF
  the standard 20% test partition (stratified, random_state=42), split further
  into:
    - calib slice   (50%) — used ONLY to fit the Platt sigmoid (A, B)
    - final_test slice (50%) — used ONLY to report calibration-quality
      metrics (Brier score, ECE, reliability table). Never touched during
      fitting.
  This final_test slice, not the calib slice and not the training data, is
  what answers "is Platt scaling honestly better calibrated."

  For the ranking/AUC/recall-ceiling question, applying (not fitting) the
  already-fit monotonic transform to the FULL standard 253-fraud test
  partition is not a leakage risk — a fixed strictly-increasing function
  preserves rank order and AUC identically regardless of what data it was
  fit on. So that check is run against the full canonical test set, directly
  comparable to the numbers in README.md / paysense_report.tex /
  test_frozen_model_metrics.py.

  Outputs:
    - artefacts/platt_calibrator.pkl   (new, separate artifact — experiment only)
    - stdout report used to write PLATT_SCALING_RESULT.md
================================================================================
"""

import pathlib
import warnings

warnings.filterwarnings("ignore")

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

BASE_DIR = pathlib.Path(__file__).resolve().parent
ARTEFACTS_DIR = BASE_DIR / "artefacts"
MASTER_CSV = BASE_DIR / "paysense_master_dataset.csv"

RANDOM_STATE = 42
DROP_COLS = [
    "transaction_id", "user_id", "receiver_id", "timestamp", "date",
    "data_source", "user_kyc_status", "status", "usr_home_city",
]
TARGET = "is_fraud"

THRESHOLDS = np.arange(0.05, 0.51, 0.05)


def expected_calibration_error(y_true, y_prob, n_bins=10):
    """Standard equal-width-bin ECE: sum over bins of (n_bin/N) * |acc - conf|."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    n = len(y_true)
    rows = []
    for i in range(n_bins):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        if i == n_bins - 1:
            mask = (y_prob >= lo) & (y_prob <= hi)
        else:
            mask = (y_prob >= lo) & (y_prob < hi)
        count = mask.sum()
        if count == 0:
            rows.append((lo, hi, 0, np.nan, np.nan))
            continue
        conf = y_prob[mask].mean()
        acc = y_true[mask].mean()
        ece += (count / n) * abs(acc - conf)
        rows.append((lo, hi, int(count), conf, acc))
    return ece, rows


def main():
    print("=" * 78)
    print("  PaySense — Platt Scaling Experiment")
    print("=" * 78)

    # ── Block 0 replay: identical split to paysense_phase3.py ────────────────
    df = pd.read_csv(MASTER_CSV)
    df = df.drop(columns=DROP_COLS)
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\n  Reproduced canonical split: train={len(X_train_raw)} rows, "
          f"test={len(X_test_raw)} rows, test fraud={int(y_test.sum())}")
    assert int(y_test.sum()) == 253, (
        "Test-set fraud count no longer matches the canonical 253 every doc "
        "cites — master dataset or split logic changed upstream."
    )

    # ── Load frozen artifacts (read-only, never modified) ────────────────────
    preprocessor = joblib.load(ARTEFACTS_DIR / "paysense_preprocessor.pkl")
    model = joblib.load(ARTEFACTS_DIR / "paysense_model.pkl")
    frozen_threshold = joblib.load(ARTEFACTS_DIR / "paysense_threshold.pkl")
    print(f"  Loaded frozen artifacts (model, preprocessor, threshold="
          f"{frozen_threshold}) — read-only, not modified.")

    # ── Carve calib / final_test out of the 20% test partition ──────────────
    # This is the ONLY genuinely-unseen-by-the-frozen-model data available
    # without retraining (see module docstring). Further stratified 50/50
    # split, random_state=42.
    X_calib_raw, X_finaltest_raw, y_calib, y_finaltest = train_test_split(
        X_test_raw, y_test, test_size=0.50, random_state=RANDOM_STATE, stratify=y_test
    )
    print(f"  Calibration slice: {len(X_calib_raw)} rows, "
          f"{int(y_calib.sum())} fraud (fit A,B here ONLY)")
    print(f"  Final-test slice : {len(X_finaltest_raw)} rows, "
          f"{int(y_finaltest.sum())} fraud (report Brier/ECE here ONLY, "
          f"never used to fit)")

    X_calib_proc = preprocessor.transform(X_calib_raw)
    X_finaltest_proc = preprocessor.transform(X_finaltest_raw)
    X_fulltest_proc = preprocessor.transform(X_test_raw)  # canonical 253-fraud set

    # ── Fit Platt scaling: CalibratedClassifierCV(cv="prefit", method="sigmoid") ──
    # This is textbook Platt scaling: fits sigma(A*f(x)+B) via sklearn's
    # internal _SigmoidCalibration, minimising log-loss against the
    # calibration slice's true labels. The base XGBoost estimator is frozen
    # (cv="prefit") — it is NOT refit here.
    platt = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
    platt.fit(X_calib_proc, y_calib)
    joblib.dump(platt, ARTEFACTS_DIR / "platt_calibrator.pkl")
    print(f"\n  Fitted Platt calibrator on the {len(X_calib_raw)}-row calib "
          f"slice only. Saved to artefacts/platt_calibrator.pkl (NEW artifact, "
          f"experiment-only, not wired into main.py/src/fraud_model.py).")

    # ══════════════════════════════════════════════════════════════════════
    #  PART 1 — Does Platt scaling change ROC-AUC / PR-AUC / the recall
    #  ceiling? Evaluated on the FULL canonical 253-fraud test set (applying,
    #  not fitting, the transform — see docstring on why this isn't leakage).
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 78)
    print("  PART 1 — Ranking invariance (ROC-AUC / PR-AUC / recall ceiling)")
    print("  Evaluated on the full canonical 253-fraud test set")
    print("─" * 78)

    y_proba_raw = model.predict_proba(X_fulltest_proc)[:, 1]
    y_proba_platt = platt.predict_proba(X_fulltest_proc)[:, 1]

    roc_raw = roc_auc_score(y_test, y_proba_raw)
    roc_platt = roc_auc_score(y_test, y_proba_platt)
    pr_raw = average_precision_score(y_test, y_proba_raw)
    pr_platt = average_precision_score(y_test, y_proba_platt)

    print(f"\n  ROC-AUC   raw={roc_raw:.6f}   platt={roc_platt:.6f}   "
          f"delta={roc_platt - roc_raw:+.8f}")
    print(f"  PR-AUC    raw={pr_raw:.6f}   platt={pr_platt:.6f}   "
          f"delta={pr_platt - pr_raw:+.8f}")

    # Rank-order identity check: for every pair of test rows, does raw_i < raw_j
    # imply platt_i <= platt_j (allowing exact ties to resolve either way,
    # since a tie carries no order information to invert)? Checked via
    # dense-rank equality rather than positional "first" tie-breaking, which
    # is sensitive to arbitrary tie order and would produce false negatives.
    order = np.argsort(y_proba_raw, kind="stable")
    raw_sorted = y_proba_raw[order]
    platt_sorted = y_proba_platt[order]
    # Group by exact-tie blocks in raw score; within a block platt scores may
    # be in any order (ties carry no rank information), but the platt values
    # of one block must not exceed the minimum of the next block.
    inversions = 0
    n = len(raw_sorted)
    i = 0
    block_max_platt = -np.inf
    while i < n:
        j = i
        while j < n and raw_sorted[j] == raw_sorted[i]:
            j += 1
        this_block_min = platt_sorted[i:j].min()
        this_block_max = platt_sorted[i:j].max()
        if this_block_min < block_max_platt - 1e-9:
            inversions += 1
        block_max_platt = max(block_max_platt, this_block_max)
        i = j
    rank_identical = inversions == 0
    print(f"  Rank order preserved (no raw-tie-block inversions across all "
          f"{len(y_proba_raw)} test rows): {rank_identical} "
          f"({inversions} inversions found)")

    # Recall ceiling: find raw threshold 0.05's equivalent point on the
    # Platt-scaled axis and confirm the SAME set of rows is flagged.
    tau_raw = 0.05
    y_pred_raw_005 = (y_proba_raw >= tau_raw).astype(int)
    recall_raw_005 = recall_score(y_test, y_pred_raw_005, zero_division=0)

    # Analytic equivalent point: sklearn's _SigmoidCalibration stores A, B as
    # a_, b_ on the internal calibrator object, but internally
    # CalibratedClassifierCV computes the CLASS-1 probability as
    # sigma(-(A*p_raw + B)) = 1 / (1 + exp(A*p_raw + B)) — i.e. the fitted
    # sigmoid represents the class-0 probability, and class 1 is its
    # complement. Verified empirically below (max abs diff vs
    # platt.predict_proba against this formula is ~6e-8, i.e. float32 noise).
    sig = platt.calibrated_classifiers_[0].calibrators[0]
    A, B = sig.a_, sig.b_
    _sanity = 1.0 / (1.0 + np.exp(A * y_proba_raw + B))
    _max_diff = float(np.max(np.abs(_sanity - y_proba_platt)))
    assert _max_diff < 1e-5, (
        f"A,B sign-convention sanity check failed (max abs diff {_max_diff}) "
        f"— the analytic tau' formula below would silently be wrong."
    )
    tau_platt_equiv = 1.0 / (1.0 + np.exp(A * tau_raw + B))
    y_pred_platt_equiv = (y_proba_platt >= tau_platt_equiv).astype(int)
    recall_platt_equiv = recall_score(y_test, y_pred_platt_equiv, zero_division=0)
    same_rows_flagged = np.array_equal(y_pred_raw_005, y_pred_platt_equiv)

    n_below_raw = int(((y_test == 1) & (y_proba_raw < tau_raw)).sum())
    n_below_platt = int(((y_test == 1) & (y_proba_platt < tau_platt_equiv)).sum())

    print(f"\n  Fitted Platt sigmoid: A={A:.6f}, B={B:.6f}")
    print(f"  Raw threshold tau=0.05 -> recall={recall_raw_005:.4%} "
          f"({int(y_pred_raw_005.sum())} flagged, {n_below_raw}/253 fraud below tau)")
    print(f"  Equivalent Platt threshold tau'={tau_platt_equiv:.6f} -> "
          f"recall={recall_platt_equiv:.4%} ({int(y_pred_platt_equiv.sum())} flagged, "
          f"{n_below_platt}/253 fraud below tau')")
    print(f"  Identical set of flagged rows at the equivalent threshold: "
          f"{same_rows_flagged}")

    # Full sweep comparison: same threshold *index* (not same raw numeric
    # value) should produce identical (precision, recall) pairs when the
    # Platt-side threshold is the corresponding equivalent point.
    print("\n  Full precision/recall frontier comparison "
          "(raw tau vs Platt tau'=sigma(A*tau+B), same underlying row set):")
    print(f"  {'tau':>6}  {'raw P':>8}  {'raw R':>8}  {'tau_prime':>10}  "
          f"{'platt P':>8}  {'platt R':>8}  {'same rows':>10}")
    sweep_rows = []
    for t in THRESHOLDS:
        t = round(float(t), 2)
        yp_raw = (y_proba_raw >= t).astype(int)
        p_raw = precision_score(y_test, yp_raw, zero_division=0)
        r_raw = recall_score(y_test, yp_raw, zero_division=0)

        t_prime = 1.0 / (1.0 + np.exp(A * t + B))
        yp_platt = (y_proba_platt >= t_prime).astype(int)
        p_platt = precision_score(y_test, yp_platt, zero_division=0)
        r_platt = recall_score(y_test, yp_platt, zero_division=0)
        same = np.array_equal(yp_raw, yp_platt)
        sweep_rows.append((t, p_raw, r_raw, t_prime, p_platt, r_platt, same))
        print(f"  {t:>6.2f}  {p_raw:>8.4f}  {r_raw:>8.4f}  {t_prime:>10.6f}  "
              f"{p_platt:>8.4f}  {r_platt:>8.4f}  {str(same):>10}")

    all_sweep_identical = all(row[6] for row in sweep_rows)
    print(f"\n  All 10 threshold points produce identical flagged-row sets: "
          f"{all_sweep_identical}")

    # ══════════════════════════════════════════════════════════════════════
    #  PART 2 — Calibration quality: Brier score + ECE, raw vs Platt-scaled,
    #  measured ONLY on the never-touched final_test slice.
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 78)
    print("  PART 2 — Calibration quality (Brier score, ECE)")
    print(f"  Evaluated on the held-out final_test slice only "
          f"({len(X_finaltest_raw)} rows, {int(y_finaltest.sum())} fraud) — "
          f"never used to fit the Platt sigmoid")
    print("─" * 78)

    y_proba_raw_ft = model.predict_proba(X_finaltest_proc)[:, 1]
    y_proba_platt_ft = platt.predict_proba(X_finaltest_proc)[:, 1]

    brier_raw = brier_score_loss(y_finaltest, y_proba_raw_ft)
    brier_platt = brier_score_loss(y_finaltest, y_proba_platt_ft)

    ece_raw, bins_raw = expected_calibration_error(y_finaltest, y_proba_raw_ft, n_bins=10)
    ece_platt, bins_platt = expected_calibration_error(y_finaltest, y_proba_platt_ft, n_bins=10)

    print(f"\n  Brier score   raw={brier_raw:.6f}   platt={brier_platt:.6f}   "
          f"(lower is better)")
    print(f"  ECE (10 bins) raw={ece_raw:.6f}   platt={ece_platt:.6f}   "
          f"(lower is better)")

    print(f"\n  Reliability table — RAW scores:")
    print(f"  {'bin':>14}  {'n':>6}  {'mean pred':>10}  {'empirical rate':>15}")
    for lo, hi, count, conf, acc in bins_raw:
        conf_s = f"{conf:.4f}" if not np.isnan(conf) else "  n/a"
        acc_s = f"{acc:.4f}" if not np.isnan(acc) else "  n/a"
        print(f"  [{lo:.2f},{hi:.2f})  {count:>6}  {conf_s:>10}  {acc_s:>15}")

    print(f"\n  Reliability table — PLATT-SCALED scores:")
    print(f"  {'bin':>14}  {'n':>6}  {'mean pred':>10}  {'empirical rate':>15}")
    for lo, hi, count, conf, acc in bins_platt:
        conf_s = f"{conf:.4f}" if not np.isnan(conf) else "  n/a"
        acc_s = f"{acc:.4f}" if not np.isnan(acc) else "  n/a"
        print(f"  [{lo:.2f},{hi:.2f})  {count:>6}  {conf_s:>10}  {acc_s:>15}")

    # ══════════════════════════════════════════════════════════════════════
    #  PART 3 — Robustness: is the Part-2 verdict specific to one 50/50 draw
    #  of the calib/final_test split (126 fraud rows is not a lot), or does
    #  it hold across resamples? Re-splits X_test_raw/y_test with 5 different
    #  seeds, refits Platt each time on the calib half, evaluates Brier/ECE
    #  on the other half. Still zero contact with the 80% train partition.
    # ══════════════════════════════════════════════════════════════════════
    print("\n" + "─" * 78)
    print("  PART 3 — Robustness across 5 calib/final_test split seeds")
    print("─" * 78)
    print(f"\n  {'seed':>6}  {'brier_raw':>10}  {'brier_platt':>12}  "
          f"{'ece_raw':>9}  {'ece_platt':>10}  {'platt_better(brier)':>20}  "
          f"{'platt_better(ece)':>18}")
    resample_rows = []
    for seed in range(5):
        Xc_raw, Xf_raw, yc, yf = train_test_split(
            X_test_raw, y_test, test_size=0.50, random_state=seed, stratify=y_test
        )
        Xc_proc = preprocessor.transform(Xc_raw)
        Xf_proc = preprocessor.transform(Xf_raw)
        p = CalibratedClassifierCV(estimator=model, method="sigmoid", cv="prefit")
        p.fit(Xc_proc, yc)

        praw = model.predict_proba(Xf_proc)[:, 1]
        pplatt = p.predict_proba(Xf_proc)[:, 1]
        b_raw = brier_score_loss(yf, praw)
        b_platt = brier_score_loss(yf, pplatt)
        e_raw, _ = expected_calibration_error(yf, praw, n_bins=10)
        e_platt, _ = expected_calibration_error(yf, pplatt, n_bins=10)
        resample_rows.append((seed, b_raw, b_platt, e_raw, e_platt))
        print(f"  {seed:>6}  {b_raw:>10.6f}  {b_platt:>12.6f}  "
              f"{e_raw:>9.6f}  {e_platt:>10.6f}  "
              f"{str(b_platt < b_raw):>20}  {str(e_platt < e_raw):>18}")

    brier_wins = sum(1 for r in resample_rows if r[2] < r[1])
    ece_wins = sum(1 for r in resample_rows if r[4] < r[3])
    print(f"\n  Platt beats raw on Brier in {brier_wins}/5 seeds "
          f"(plus the original seed=42 50/50 draw reported in Part 2: "
          f"{brier_platt < brier_raw})")
    print(f"  Platt beats raw on ECE   in {ece_wins}/5 seeds "
          f"(plus the original seed=42 50/50 draw reported in Part 2: "
          f"{ece_platt < ece_raw})")

    print("\n" + "=" * 78)
    print("  SUMMARY")
    print("=" * 78)
    print(f"  ROC-AUC delta (raw vs platt): {roc_platt - roc_raw:+.8f}")
    print(f"  PR-AUC delta  (raw vs platt): {pr_platt - pr_raw:+.8f}")
    print(f"  Rank order identical (full test): {rank_identical}")
    print(f"  All threshold-sweep row sets identical: {all_sweep_identical}")
    print(f"  Recall ceiling unchanged (same rows below equivalent tau): "
          f"{n_below_raw} == {n_below_platt} -> {n_below_raw == n_below_platt}")
    print(f"  Brier improvement (raw - platt, positive = platt better): "
          f"{brier_raw - brier_platt:+.6f}")
    print(f"  ECE improvement   (raw - platt, positive = platt better): "
          f"{ece_raw - ece_platt:+.6f}")


if __name__ == "__main__":
    main()
