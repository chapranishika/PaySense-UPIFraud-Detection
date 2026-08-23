"""
================================================================================
  PaySense — Out-of-Distribution Generalization Check (FULL ENSEMBLE)
  ────────────────────────────────────────────────────────────────────────────
  generalization_check.py scored the raw frozen XGBoost artifact directly
  (model.predict_proba()), which never exercised the rules scorer or the
  now-trained LightLR scorer — even though LightLR's 5 features were
  specifically chosen (per src/fraud_model.py's module docstring) as the
  ones "consistently available at inference time," which is exactly the
  scenario this OOD check simulates (a foreign dataset where only a small
  slice of the 40-feature vector is honestly mappable).

  This script re-runs the EXACT SAME accepted dataset and EXACT SAME honest
  column mapping already documented in GENERALIZATION_CHECK.md (reused
  verbatim from generalization_check.py — not re-derived), but scores every
  row through the real production code path: src/fraud_model.py's score(),
  the same function main.py's /predict endpoint calls. That means every row
  goes through all three scorers (XGBoost, rules, LightLR) exactly as
  production would run them, using the honestly-mapped columns and letting
  everything else default the same way it would for a real caller who simply
  didn't send those fields.

  No retraining, no fine-tuning, no threshold recalibration — same rule as
  the original check. The frozen model, the frozen preprocessor, the frozen
  threshold (0.40), and now the newly-trained (but NOT retrained on this OOD
  data) light_lr.pkl are all used exactly as loaded by load_artefacts().

  Run
  ───
      venv\\Scripts\\python.exe generalization_check_ensemble.py
================================================================================
"""

import io
import os
import sys
import time
import warnings

# LightLR's scorer passes a plain list (not a DataFrame) into predict_proba(),
# same as production's _score_light_lr() does — sklearn emits a benign
# "X does not have valid feature names" UserWarning on every single call.
# It doesn't affect correctness (same behavior main.py's /predict has always
# had), but at 74,917 rows it floods stdout; suppress it here for a readable
# log, exactly as production silently tolerates it every request.
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")

# Force UTF-8 stdout regardless of the Windows console codepage, so the
# box-drawing characters in the summary tables below don't crash on a
# cp1252 console (this only affects how this script prints output; it
# changes nothing about how fraud_model.score() computes results).
if __name__ == "__main__":
    # Guarded: unconditional reassignment at import time breaks pytest's
    # capture mechanism for anything that imports this module. See
    # ood_generalization_remediation.py's identical fix for the confirmed
    # failure mode.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

from src import fraud_model  # noqa: E402

# Reuse the EXACT dataset-1 loader + mapping tables from the original check —
# not re-derived, imported verbatim.
from generalization_check import load_dataset_1  # noqa: E402


def main():
    print("Loading production ensemble artefacts via load_artefacts() "
          "(same call main.py makes at startup) ...")
    state = fraud_model.load_artefacts()
    print(f"Active scorers: {state.active_scorers}")
    assert "light_lr" in state.active_scorers, (
        "light_lr is not active — artefacts/light_lr.pkl must exist and be "
        "loaded from disk for this check to actually test the full ensemble."
    )
    threshold = state.ps_threshold
    print(f"Frozen threshold: {threshold}")

    mapped, y, stats = load_dataset_1()
    print(f"\nDataset 1: {stats['dataset']} — {stats['n_rows']:,} rows, "
          f"{stats['fraud_rate']*100:.2f}% fraud")
    print(f"Honestly-mapped columns feeding the ensemble: {stats['features_mapped_names']}")
    print("(All other fields — including all 5 LightLR features and the "
          "rules scorer's new_device_flag/ip_location_mismatch/kyc_verified_flag/"
          "usr_is_high_risk — are simply ABSENT from txn_dict for every row, "
          "exactly as they would be for a real caller who never sent them; "
          "each scorer's own .get(key, default) governs what happens next.)")

    # Build one dict per row (same shape /predict would receive from a client
    # that only ever sends the 6 honestly-mappable fields) and score it
    # through the real production ensemble path.
    records = mapped.to_dict(orient="records")

    print(f"\nScoring {len(records):,} rows through fraud_model.score() "
          f"(full ensemble: XGBoost + rules + LightLR) ...")
    t0 = time.time()

    ensemble_scores = np.empty(len(records), dtype=float)
    paysense_scores = np.empty(len(records), dtype=float)
    lr_scores       = np.full(len(records), np.nan, dtype=float)
    rules_scores    = np.empty(len(records), dtype=float)
    weights_seen    = set()

    for i, rec in enumerate(records):
        # NaN mrc_category/device_type (unmapped rows, e.g. merchant_category
        # values with no entry in MRC_CATEGORY_MAP) become float('nan') via
        # pandas' to_dict — leave them as-is, exactly like a client that sent
        # nothing for that field would.
        clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        result = fraud_model.score(clean)
        ensemble_scores[i] = result.ensemble_score
        paysense_scores[i] = result.paysense_score if result.paysense_score is not None else np.nan
        if result.light_lr_score is not None:
            lr_scores[i] = result.light_lr_score
        rules_scores[i] = result.rules_score
        weights_seen.add(tuple(sorted(result.weights_used.items())))
        if i and i % 10000 == 0:
            print(f"  ... {i:,} / {len(records):,} rows scored "
                  f"({time.time()-t0:.1f}s elapsed)")

    elapsed = time.time() - t0
    print(f"Scored {len(records):,} rows in {elapsed:.1f}s "
          f"({len(records)/max(elapsed,1e-9):.0f} rows/sec)")
    print(f"Distinct weights_used maps seen across all rows: {weights_seen}")

    preds = (ensemble_scores >= threshold).astype(int)

    roc_auc = roc_auc_score(y, ensemble_scores)
    pr_auc  = average_precision_score(y, ensemble_scores)
    cm      = confusion_matrix(y, preds)
    report  = classification_report(y, preds, digits=4, zero_division=0)

    print("\n" + "═" * 78)
    print("  FULL ENSEMBLE — Dataset 1: upi_fraud_dataset.csv")
    print("═" * 78)
    print(f"  Rows                       : {stats['n_rows']:,}")
    print(f"  Fraud rate                 : {stats['fraud_rate']*100:.2f}%")
    print(f"  ROC-AUC                    : {roc_auc:.4f}")
    print(f"  PR-AUC (avg precision)     : {pr_auc:.4f}")
    print(f"  Confusion matrix [[TN FP][FN TP]] @ threshold {threshold}:")
    print(f"      {cm.tolist()}")
    print(f"  Max ensemble score (any row): {ensemble_scores.max():.4f}")
    print(f"  Min/mean/max paysense_score : {np.nanmin(paysense_scores):.4f} / "
          f"{np.nanmean(paysense_scores):.4f} / {np.nanmax(paysense_scores):.4f}")
    print(f"  Min/mean/max light_lr_score : {np.nanmin(lr_scores):.4f} / "
          f"{np.nanmean(lr_scores):.4f} / {np.nanmax(lr_scores):.4f}")
    print(f"  Min/mean/max rules_score    : {rules_scores.min():.4f} / "
          f"{rules_scores.mean():.4f} / {rules_scores.max():.4f}")
    print(f"  Unique rules_score values   : {sorted(set(np.round(rules_scores, 4).tolist()))}")
    print(f"  Unique light_lr_score values: {sorted(set(np.round(lr_scores[~np.isnan(lr_scores)], 6).tolist()))}")
    print(f"\n{report}")

    # ── Side-by-side vs. the original raw-XGBoost-only numbers (from
    #    GENERALIZATION_CHECK.md §4.1, produced by generalization_check.py) ──
    print("═" * 78)
    print("  SIDE-BY-SIDE: raw XGBoost-only (original check) vs. full ensemble")
    print("═" * 78)
    print(f"  {'Metric':<35}{'Raw XGBoost':>18}{'Full ensemble':>18}")
    print(f"  {'-'*35}{'-'*18}{'-'*18}")
    print(f"  {'ROC-AUC':<35}{0.8064:>18.4f}{roc_auc:>18.4f}")
    print(f"  {'PR-AUC':<35}{0.4052:>18.4f}{pr_auc:>18.4f}")
    print(f"  {'TP @ frozen threshold (of 701)':<35}{0:>18}{int(cm[1,1]) if cm.shape==(2,2) else 'n/a':>18}")
    print(f"  {'FP @ frozen threshold':<35}{0:>18}{int(cm[0,1]) if cm.shape==(2,2) else 'n/a':>18}")
    print(f"  {'Max predicted probability':<35}{0.0095:>18.4f}{ensemble_scores.max():>18.4f}")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "n": len(y),
        "n_fraud": int(y.sum()),
        "max_ensemble_score": float(ensemble_scores.max()),
        "rules_score_unique": sorted(set(np.round(rules_scores, 4).tolist())),
        "light_lr_score_unique": sorted(set(np.round(lr_scores[~np.isnan(lr_scores)], 6).tolist())),
    }


if __name__ == "__main__":
    main()
