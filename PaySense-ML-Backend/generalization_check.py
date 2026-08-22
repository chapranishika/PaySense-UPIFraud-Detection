"""
================================================================================
  PaySense — Out-of-Distribution Generalization Check
  ────────────────────────────────────────────────────────────────────────────
  Purpose
  ───────
  The frozen PaySense XGBoost model (artefacts/paysense_model.pkl) was
  trained and tested EXCLUSIVELY on paysense_master_dataset.csv — a
  30,000-row dataset built from one 20,000-row anchor + one 10,000-row
  synthetic supplement. Every number in the project report (ROC-AUC 0.8851,
  PR-AUC 0.5303) describes performance on a held-out split of THAT SAME
  pipeline's output, not on data the model has never seen the likes of.

  This script is a genuine generalization test: it scores the frozen model
  (NO retraining, NO fine-tuning, NO threshold recalibration) against real,
  previously-unused third-party UPI/financial fraud datasets, honestly
  mapped onto the model's frozen 40-feature schema. Columns that cannot be
  honestly mapped are left as NaN and handled by the frozen preprocessor's
  own imputers (SimpleImputer, median for numerics / most-frequent for
  categoricals) and by OrdinalEncoder's unknown_value=-1 for any category
  string outside what the encoder saw during training.

  This is NOT a training run. It exists to answer one question honestly:
  does the frozen model's performance hold up outside its own training
  distribution — and if not, say so plainly.

  Datasets evaluated (see GENERALIZATION_CHECK.md for the full trust
  analysis that decided inclusion/rejection):
    1. UPI Fraud detection dataset / upi_fraud_dataset.csv          — USED
    2. UPI Transactions Dataset for Fraud Detection / ...100000.csv — REJECTED
    3. Synthetic Financial Fraud Detection Dataset / fraud_dataset.csv — USED (secondary, low power)

  Run
  ───
      venv\\Scripts\\python.exe generalization_check.py
================================================================================
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.abspath(__file__))
ARTEFACTS_DIR = os.path.join(BASE_DIR, "artefacts")

MODEL_PATH   = os.path.join(ARTEFACTS_DIR, "paysense_model.pkl")
PREP_PATH    = os.path.join(ARTEFACTS_DIR, "paysense_preprocessor.pkl")
THRESH_PATH  = os.path.join(ARTEFACTS_DIR, "paysense_threshold.pkl")
FEATS_PATH   = os.path.join(ARTEFACTS_DIR, "paysense_feature_names.pkl")

# External datasets live one level above the PS repo root, under E:\Projects\upi
UPI_ROOT = r"E:\Projects\upi"
DATASET_1_PATH = os.path.join(UPI_ROOT, "UPI Fraud detection dataset", "upi_fraud_dataset.csv")
DATASET_3_PATH = os.path.join(UPI_ROOT, "Synthetic Financial Fraud Detection Dataset", "fraud_dataset.csv")


# ════════════════════════════════════════════════════════════════════════════
#  LOAD FROZEN ARTEFACTS
# ════════════════════════════════════════════════════════════════════════════
def load_frozen_artefacts():
    model     = joblib.load(MODEL_PATH)
    prep      = joblib.load(PREP_PATH)
    threshold = float(joblib.load(THRESH_PATH))
    features  = joblib.load(FEATS_PATH)
    assert len(features) == 40, f"Expected 40 frozen features, found {len(features)}"
    return model, prep, threshold, features


# ════════════════════════════════════════════════════════════════════════════
#  DATASET 1 — upi_fraud_dataset.csv  (74,917 rows, 0.94% fraud)
# ════════════════════════════════════════════════════════════════════════════
#  Honest mapping — 6 of 40 features carry real signal, the remaining 34 are
#  left as NaN and imputed by the frozen preprocessor.
#
#    amount               <- Amount                       (direct, same unit: INR)
#    hour_of_day          <- hour                          (direct, 0-23)
#    mrc_category          <- merchant_category            (case + synonym mapped,
#                                                            see MRC_CATEGORY_MAP)
#    device_type           <- device                       (case-normalised only:
#                                                            android/ios/web ->
#                                                            Android/iOS/Web)
#    txn_success_flag      <- transaction_status            (SUCCESS->1, FAILED->0)
#    is_night_transaction  <- derived from hour, using the SAME rule the
#                              training pipeline used for its own synthetic
#                              supplement (paysense_pipeline.py, line ~352):
#                              1 if hour < 6 or hour >= 22, else 0.
#
#  Explicitly NOT mapped (and why):
#    is_new_payee -> new_device_flag       : different concepts (payee novelty
#                                             vs. device novelty). Conflating
#                                             them would be a dishonest shortcut.
#    payee_type -> receiver_type           : payee_type has 4 categories
#                                             (bank/p2m/p2p/upi_number) split
#                                             roughly evenly; only p2m/p2p map
#                                             cleanly to Merchant/User, and
#                                             bank/upi_number are genuinely
#                                             ambiguous as to whether the
#                                             receiver is a business or a
#                                             person. Left unmapped rather than
#                                             guess on half the rows.
#    usual_hour/usual_location/usual_device/avg_amount : this dataset's own
#                                             personalisation features. They
#                                             measure the same idea PaySense's
#                                             z-score features do, but are not
#                                             the same statistic (no z-score,
#                                             no 90-day window, different
#                                             baseline construction) — cannot
#                                             be substituted for
#                                             amount_deviation_score without
#                                             fabricating a transformation.
#    user_id, transaction_id, location, payment_retries, payer_type : no
#                                             corresponding frozen feature.

MRC_CATEGORY_MAP = {
    "education":     "Education",
    "shopping":       "Shopping",
    "food":           "Food",
    "entertainment":  "Entertainment",
    "transport":      "Travel",       # closest semantic match in the frozen schema
    "medical":        "Healthcare",
}

DEVICE_MAP = {
    "android": "Android",
    "ios":     "iOS",
    "web":     "Web",
}


def load_dataset_1() -> tuple[pd.DataFrame, pd.Series, dict]:
    df = pd.read_csv(DATASET_1_PATH)
    y = df["isFraud"].astype(int)

    n = len(df)
    mapped = pd.DataFrame(index=df.index)
    mapped["amount"]      = df["Amount"].astype(float)
    mapped["hour_of_day"] = df["hour"].astype(int)
    mapped["mrc_category"] = df["merchant_category"].map(MRC_CATEGORY_MAP)
    mapped["device_type"]  = df["device"].map(DEVICE_MAP)
    mapped["txn_success_flag"] = (df["transaction_status"] == "SUCCESS").astype(int)
    mapped["is_night_transaction"] = (
        (df["hour"] < 6) | (df["hour"] >= 22)
    ).astype(int)

    mapping_stats = {
        "dataset": "upi_fraud_dataset.csv",
        "n_rows": n,
        "fraud_rate": float(y.mean()),
        "features_mapped": 6,
        "features_total": 40,
        "features_mapped_names": list(mapped.columns),
    }
    return mapped, y, mapping_stats


# ════════════════════════════════════════════════════════════════════════════
#  DATASET 3 — Synthetic Financial Fraud Detection Dataset (1,000 rows, 6.4% fraud)
# ════════════════════════════════════════════════════════════════════════════
#  Honest mapping — only 2 of 40 features carry real signal. This is a very
#  thin test (2/40 = 5% feature coverage, and only 64 positive examples) —
#  reported as a low-power secondary sanity check only, per the task brief.
#
#    amount                <- amount                       (direct, same concept)
#    usr_account_age_days  <- account_age_days              (direct, same unit: days)
#
#  Explicitly NOT mapped (and why):
#    device_type            <- device_type   : NAME COLLISION, DIFFERENT CONCEPT.
#                                               This dataset's device_type is a
#                                               FORM FACTOR (desktop/mobile/tablet).
#                                               PaySense's device_type is an OS
#                                               (Android/iOS/Web). "mobile" does
#                                               not tell us Android vs iOS —
#                                               mapping it would be a guess
#                                               dressed up as a match.
#    transaction_type       <- transaction_type : this dataset's values
#                                               (cash_out/debit/transfer/payment)
#                                               are generic ledger transaction
#                                               types (looks like a mobile-money
#                                               / PaySim-style schema), not
#                                               UPI-specific channels
#                                               (P2M/P2P/Bill Payment/EMI/
#                                               Recharge/Subscription/ATM).
#                                               "transfer" could arguably become
#                                               P2P and "cash_out" could arguably
#                                               become ATM, but this is a
#                                               judgment call without ground
#                                               truth, so it is left unmapped.
#    location_risk           -> no frozen feature   : it is a coarse risk
#                                               bucket (low/medium/high), not
#                                               the binary ip_location_mismatch
#                                               flag PaySense uses — different
#                                               semantics, not substitutable.
#    num_prev_transactions   -> no frozen feature   : a raw lifetime count,
#                                               not directly convertible to
#                                               user_avg_monthly_txn or
#                                               transaction_frequency_score
#                                               without inventing a rate.

def load_dataset_3() -> tuple[pd.DataFrame, pd.Series, dict]:
    df = pd.read_csv(DATASET_3_PATH)
    y = df["is_fraud"].astype(int)

    mapped = pd.DataFrame(index=df.index)
    mapped["amount"]               = df["amount"].astype(float)
    mapped["usr_account_age_days"] = df["account_age_days"].astype(float)

    mapping_stats = {
        "dataset": "Synthetic Financial Fraud Detection Dataset/fraud_dataset.csv",
        "n_rows": len(df),
        "fraud_rate": float(y.mean()),
        "features_mapped": 2,
        "features_total": 40,
        "features_mapped_names": list(mapped.columns),
    }
    return mapped, y, mapping_stats


# ════════════════════════════════════════════════════════════════════════════
#  SCORING
# ════════════════════════════════════════════════════════════════════════════
def score_dataset(mapped: pd.DataFrame, y: pd.Series, model, prep, features, threshold):
    """
    Reindexes the honestly-mapped columns onto the frozen 40-feature schema
    (missing columns become all-NaN), runs the frozen preprocessor, then the
    frozen XGBoost model. No fitting happens anywhere in this function.
    """
    X = mapped.reindex(columns=features)  # unmapped features -> all-NaN columns
    X_proc = prep.transform(X)
    proba = model.predict_proba(X_proc)[:, 1]
    preds = (proba >= threshold).astype(int)

    roc_auc = roc_auc_score(y, proba)
    pr_auc  = average_precision_score(y, proba)
    cm = confusion_matrix(y, preds)
    report = classification_report(y, preds, digits=4, zero_division=0)

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "n": len(y),
        "n_fraud": int(y.sum()),
    }


def print_result(name: str, mapping_stats: dict, result: dict):
    print("\n" + "═" * 78)
    print(f"  {name}")
    print("═" * 78)
    print(f"  Rows                  : {mapping_stats['n_rows']:,}")
    print(f"  Fraud rate            : {mapping_stats['fraud_rate']*100:.2f}%")
    print(f"  Features with real signal : {mapping_stats['features_mapped']} / "
          f"{mapping_stats['features_total']} "
          f"({mapping_stats['features_mapped']/mapping_stats['features_total']*100:.1f}%)")
    print(f"  Mapped feature names  : {mapping_stats['features_mapped_names']}")
    print(f"\n  ROC-AUC               : {result['roc_auc']:.4f}")
    print(f"  PR-AUC (avg precision): {result['pr_auc']:.4f}")
    print(f"  Confusion matrix [[TN FP][FN TP]] @ frozen threshold:")
    print(f"      {result['confusion_matrix']}")
    print(f"\n{result['classification_report']}")


# ════════════════════════════════════════════════════════════════════════════
#  MAIN
# ════════════════════════════════════════════════════════════════════════════
def main():
    model, prep, threshold, features = load_frozen_artefacts()
    print(f"Loaded frozen artefacts — {len(features)} features, threshold={threshold}")

    results = {}

    # Dataset 1 — leading candidate
    mapped1, y1, stats1 = load_dataset_1()
    res1 = score_dataset(mapped1, y1, model, prep, features, threshold)
    print_result("Dataset 1: upi_fraud_dataset.csv", stats1, res1)
    results["dataset_1"] = {"mapping": stats1, "metrics": res1}

    # Dataset 3 — secondary, low-power sanity check
    mapped3, y3, stats3 = load_dataset_3()
    res3 = score_dataset(mapped3, y3, model, prep, features, threshold)
    print_result("Dataset 3: Synthetic Financial Fraud Detection Dataset", stats3, res3)
    results["dataset_3"] = {"mapping": stats3, "metrics": res3}

    print("\n" + "═" * 78)
    print("  Dataset 2 (fraud_detection_data_100000.csv) was NOT scored.")
    print("  It failed the trust check (see GENERALIZATION_CHECK.md) and was")
    print("  rejected before scoring, the same way the pre-balanced 50.01%")
    print("  Trojan Family file was rejected without being trained on.")
    print("═" * 78)

    return results


if __name__ == "__main__":
    main()
