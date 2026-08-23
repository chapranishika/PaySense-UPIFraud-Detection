"""
================================================================================
  PaySense — EDA-Driven Feature Engineering for OOD Generalization
  ────────────────────────────────────────────────────────────────────────────
  GENERALIZATION_CHECK.md / SYNTHETIC_GROUNDING.md diagnosed that the frozen
  model ranks WORSE on an independently-generated full-40-feature synthetic
  dataset (ROC-AUC ~0.70) than on a real dataset carrying only 15% of its
  features (ROC-AUC ~0.79-0.84 depending on model variant).
  OOD_GENERALIZATION_REMEDIATION.md tried blending in more synthetic data
  from a different generator seed -- it improved ranking everywhere but never
  moved the deployed-threshold recall.

  This script is the first attempt at REAL EDA-driven feature engineering.
  The EDA (see EDA_FEATURE_ENGINEERING.md for the full writeup) found a
  concrete, previously-undocumented root cause: paysense_master_dataset.csv's
  "supplement" partition (10,000 of 30,000 rows, drawn from
  Financial Fraud Dataset/synthetic_fraud_dataset.csv) has new_device_flag
  and ip_location_mismatch PERFECTLY separating is_fraud (9500/9500 legit,
  500/500 fraud -- zero overlap), while the "anchor" partition (20,000 rows)
  has only a weak, organic correlation (r=0.06-0.12) between those same
  flags and is_fraud. The model was trained on a blend where 1/3 of its
  rows teach "these flags ARE the label" -- a training-pipeline artifact,
  not a real-world pattern -- which plausibly explains both the recall
  ceiling (RECALL_CEILING_REMEDIATION.md) and the OOD ranking gap: any
  differently-generated dataset (including this project's OWN grounded
  synthetic set, which deliberately built only a realistic 4-5x flag
  elevation, not a deterministic one) will be ranked worse wherever the
  model over-trusts that artifact.

  Engineered features tested (both added, one variant, not swept
  individually -- time budget: an ensemble evaluation of ANY variant against
  Dataset 1 alone takes ~18-20 minutes):

    1. stealth_fraud_score = (1-new_device_flag)*(1-ip_location_mismatch)
                              * mean(min-max-normalized amount_deviation_score,
                                     transaction_velocity,
                                     failed_attempts_last_24h)
       Directly targets the population RECALL_CEILING_REMEDIATION.md already
       diagnosed (76 fraud rows invisible at tau=0.05: low hard-signal means
       0.066/0.053, HIGH behavioral means 0.649-0.697) but with a GATED
       formulation (only activates when both hard flags read clean) rather
       than RECALL_CEILING's flat, ungated "composite_feature" (which it
       found to be the weakest of its three variants, 8/76 recovered).
       Given monotone_constraints=+1 (same treatment as the three underlying
       behavioral features it's built from -- consistent design intent, not
       an arbitrary extra choice).

    2. hour_sin, hour_cos = cyclical encoding of hour_of_day
       EDA found is_night_transaction correlates with fraud far more weakly
       in training data (r=0.090, 2.3x day/night fraud-rate ratio) than in
       real Dataset 1 (r=0.262 on the honestly-derived night flag, 34.5x
       ratio) -- and a KS test confirms master's hour_of_day distribution
       (near-uniform, std=6.52) differs materially from Dataset 1's real
       hour distribution (peaked, std=3.81, KS stat=0.276, p<1e-300). Raw
       integer hour_of_day forces XGBoost to learn the midnight wrap-around
       (23 -> 0) as two unrelated regions; a cyclical encoding removes that
       artificial discontinuity. No principled monotonic direction, so
       monotone_constraints=0 for both.

  Both features are computed with a single shared function so the EXACT
  SAME transformation logic is applied to the training data, the canonical
  test split, the held-out synthetic set, and the honestly-mapped external
  datasets -- when an underlying input is absent (e.g. Dataset 1 doesn't
  carry new_device_flag/ip_location_mismatch/behavioral features at all),
  the engineered feature legitimately becomes NaN and is median-imputed by
  the same frozen-style preprocessor logic used everywhere else in this
  project, not silently zero-filled or fabricated.

  Same Block-0 hyperparameters and monotone_constraints treatment as the
  currently-frozen model (paysense_phase3.py) -- this experiment does not
  regress the already-deployed recall-ceiling fix.

  NO CHANGES to artefacts/paysense_model.pkl, paysense_preprocessor.pkl, or
  paysense_threshold.pkl anywhere in this script. New artifacts only:
    - artefacts/paysense_model_feature_engineered.pkl
    - artefacts/paysense_preprocessor_feature_engineered.pkl
    - artefacts/paysense_feature_names_feature_engineered.pkl
    - eda_feature_engineering_results.json

  Baseline and blended-training (Variant A) numbers on the OOD checks are
  CITED VERBATIM from ood_generalization_remediation_results.json /
  GENERALIZATION_CHECK.md / SYNTHETIC_GROUNDING.md (already computed against
  the exact same frozen/blended artifacts and unseen evaluation sets) rather
  than re-run -- re-running them would burn ~40 more minutes of ensemble
  scoring to reproduce numbers that don't change, since neither of those
  models nor the evaluation CSVs are touched by this script.

  Run
  ───
      venv\\Scripts\\python.exe eda_feature_engineering.py
================================================================================
"""

import io
import json
import logging
import os
import sys
import time
import warnings

warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
# NOTE: stdout re-wrapping (utf-8, errors="replace") is done by
# ood_generalization_remediation.py at import time below -- doing it again
# here would create a second TextIOWrapper over the same underlying buffer
# and the first wrapper's GC-triggered close() would then invalidate it.
logging.getLogger("paysense.ensemble").setLevel(logging.ERROR)

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

# Reuse the already-reviewed, already-working helpers from the prior
# remediation experiment rather than re-implementing them.
from ood_generalization_remediation import (  # noqa: E402
    build_preprocessor, monotone_tuple, sweep_thresholds,
    score_ensemble_dataframe, swap_ps_state, metrics_from_scores,
    BASE_XGB_KWARGS, BEHAVIORAL_FEATURES, DEPLOYED_THRESHOLD,
    DROP_COLS, TARGET, RANDOM_STATE, ARTEFACTS_DIR, MASTER_CSV,
    HELD_OUT_SYNTH_CSV,
)
from generalization_check import load_dataset_1, load_dataset_3  # noqa: E402
from src import fraud_model  # noqa: E402

NEW_FEATURES = ["stealth_fraud_score", "hour_sin", "hour_cos"]


def line(c="=", n=78):
    print(c * n)


def compute_engineered_features(df: pd.DataFrame, train_stats: dict) -> pd.DataFrame:
    """
    Adds stealth_fraud_score, hour_sin, hour_cos to a copy of df. Uses
    TRAIN-ONLY min/max stats (passed in via train_stats, computed once on
    X_train_raw) for the behavioral normalization -- no leakage. Any row
    missing an underlying input (new_device_flag, ip_location_mismatch,
    amount_deviation_score, transaction_velocity, failed_attempts_last_24h,
    hour_of_day) legitimately produces NaN for the derived feature via
    normal pandas arithmetic NaN-propagation -- this is intentional, not a
    bug: an external dataset that never supplied the underlying signal has
    no honest way to compute the derived one either, and the frozen-style
    preprocessor's median imputation handles it the same way it already
    handles every other missing feature in this project's OOD checks.
    """
    out = df.copy()

    def norm(col):
        if col not in out.columns:
            return pd.Series(np.nan, index=out.index)
        lo, hi = train_stats[col]["min"], train_stats[col]["max"]
        rng = hi - lo if hi > lo else 1.0
        return ((out[col] - lo) / rng).clip(0, 1)

    behavioral_mean = pd.concat(
        [norm(c) for c in ["amount_deviation_score", "transaction_velocity",
                            "failed_attempts_last_24h"]],
        axis=1,
    ).mean(axis=1, skipna=False)  # skipna=False: ANY missing input -> NaN, not silently averaged over fewer terms

    ndf = out["new_device_flag"] if "new_device_flag" in out.columns else np.nan
    ipm = out["ip_location_mismatch"] if "ip_location_mismatch" in out.columns else np.nan
    out["stealth_fraud_score"] = (1 - ndf) * (1 - ipm) * behavioral_mean

    if "hour_of_day" in out.columns:
        radians = 2 * np.pi * out["hour_of_day"].astype(float) / 24.0
        out["hour_sin"] = np.sin(radians)
        out["hour_cos"] = np.cos(radians)
    else:
        out["hour_sin"] = np.nan
        out["hour_cos"] = np.nan

    return out


def main():
    results = {}

    line("=")
    print("STEP 0 | Reproducing paysense_phase3.py Block 0 split, adding engineered features")
    line("=")

    master_df = pd.read_csv(MASTER_CSV)
    master_df = master_df.drop(columns=DROP_COLS)
    X_master = master_df.drop(columns=[TARGET])
    y_master = master_df[TARGET].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X_master, y_master, test_size=0.20, random_state=RANDOM_STATE, stratify=y_master
    )
    assert int(y_test.sum()) == 253, f"Canonical test fraud count drifted: {int(y_test.sum())}"
    print(f"Canonical split confirmed: train={X_train_raw.shape}, test={X_test_raw.shape}, "
          f"fraud in test={int(y_test.sum())}/253")

    # Train-only stats for the behavioral min-max normalization (no leakage).
    train_stats = {
        c: {"min": float(X_train_raw[c].min()), "max": float(X_train_raw[c].max())}
        for c in ["amount_deviation_score", "transaction_velocity", "failed_attempts_last_24h"]
    }
    print(f"Train-only min/max stats for normalization: {train_stats}")

    X_train_fe = compute_engineered_features(X_train_raw, train_stats)
    X_test_fe = compute_engineered_features(X_test_raw, train_stats)
    print(f"stealth_fraud_score on TRAIN: mean={X_train_fe['stealth_fraud_score'].mean():.4f} "
          f"non-null={X_train_fe['stealth_fraud_score'].notna().sum()}/{len(X_train_fe)}")
    print(f"stealth_fraud_score by class (TRAIN): fraud={X_train_fe.loc[y_train==1,'stealth_fraud_score'].mean():.4f} "
          f"legit={X_train_fe.loc[y_train==0,'stealth_fraud_score'].mean():.4f}")

    CAT_COLS = X_train_fe.select_dtypes(include=["object"]).columns.tolist()
    NUM_COLS = X_train_fe.select_dtypes(include=[np.number]).columns.tolist()
    FEATURE_NAMES = CAT_COLS + NUM_COLS
    print(f"Feature count: {len(FEATURE_NAMES)} (40 original + {len(NEW_FEATURES)} engineered)")
    assert set(NEW_FEATURES).issubset(FEATURE_NAMES)

    prep_fe = build_preprocessor(CAT_COLS, NUM_COLS)
    prep_fe.fit(X_train_fe)
    X_train_proc = prep_fe.transform(X_train_fe)
    X_test_proc = prep_fe.transform(X_test_fe)

    smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=RANDOM_STATE)
    X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train.values)
    print(f"After SMOTE: {X_train_bal.shape}")

    # monotone_constraints: +1 for the 3 original behavioral features (same
    # treatment as the frozen model -- not regressing that fix) AND for
    # stealth_fraud_score (consistent design intent: it's built to increase
    # with fraud risk). hour_sin/hour_cos get 0 (no principled direction).
    mono = []
    for f in FEATURE_NAMES:
        if f in BEHAVIORAL_FEATURES or f == "stealth_fraud_score":
            mono.append(1)
        else:
            mono.append(0)
    mono = tuple(mono)
    print(f"monotone_constraints: {dict(zip(FEATURE_NAMES, mono))}")

    kwargs = dict(BASE_XGB_KWARGS)
    kwargs["monotone_constraints"] = mono
    model_fe = XGBClassifier(**kwargs)
    print("Training feature-engineered variant ... ", end="", flush=True)
    model_fe.fit(X_train_bal, y_train_bal, eval_set=[(X_test_proc, y_test)], verbose=False)
    print("done.")

    joblib.dump(model_fe, os.path.join(ARTEFACTS_DIR, "paysense_model_feature_engineered.pkl"))
    joblib.dump(prep_fe, os.path.join(ARTEFACTS_DIR, "paysense_preprocessor_feature_engineered.pkl"))
    joblib.dump(FEATURE_NAMES, os.path.join(ARTEFACTS_DIR, "paysense_feature_names_feature_engineered.pkl"))
    print("Saved -> paysense_model_feature_engineered.pkl / "
          "paysense_preprocessor_feature_engineered.pkl")

    # Feature importance sanity check for the new features.
    importances = dict(zip(FEATURE_NAMES, model_fe.feature_importances_))
    for f in NEW_FEATURES:
        rank = sorted(importances.values(), reverse=True).index(importances[f]) + 1
        print(f"  {f}: importance={importances[f]:.5f}  rank={rank}/{len(FEATURE_NAMES)}")
    results["feature_importances_new_features"] = {f: float(importances[f]) for f in NEW_FEATURES}

    # ════════════════════════════════════════════════════════════════════
    #  CHECK 1 — canonical held-out test (raw XGBoost, matching the format
    #  baseline_canonical_test/variant_A_canonical_test already use, PLUS
    #  the real ensemble path since the task asked for score() everywhere).
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("CHECK 1 | Canonical held-out test (in-distribution)")
    line("=")

    y_proba_raw = model_fe.predict_proba(X_test_proc)[:, 1]
    raw_metrics = metrics_from_scores(y_test.values, y_proba_raw, DEPLOYED_THRESHOLD)
    raw_metrics["roc_auc_full"] = float(roc_auc_score(y_test, y_proba_raw))
    print(f"Raw XGBoost @ tau={DEPLOYED_THRESHOLD}: ROC-AUC={raw_metrics['roc_auc']:.4f} "
          f"recall={raw_metrics['recall']:.4f} precision={raw_metrics['precision']:.4f}")
    results["fe_canonical_test_raw"] = raw_metrics
    results["fe_canonical_test_sweep"] = sweep_thresholds(y_test, y_proba_raw)

    fraud_model.load_artefacts()
    assert "light_lr" in fraud_model.get_state().active_scorers, "light_lr must be active."

    records_test = X_test_fe.assign(**{TARGET: y_test.values}).to_dict(orient="records")
    with swap_ps_state(model_fe, prep_fe, FEATURE_NAMES, threshold=DEPLOYED_THRESHOLD):
        scored_test = score_ensemble_dataframe(records_test)
    ens_metrics_test = metrics_from_scores(y_test.values, scored_test["ensemble_scores"], DEPLOYED_THRESHOLD)
    print(f"Ensemble @ tau={DEPLOYED_THRESHOLD}: ROC-AUC={ens_metrics_test['roc_auc']:.4f} "
          f"recall={ens_metrics_test['recall']:.4f} precision={ens_metrics_test['precision']:.4f}")
    results["fe_canonical_test_ensemble"] = ens_metrics_test

    # ════════════════════════════════════════════════════════════════════
    #  CHECK 2 — held-out synthetic set (seed 918273, full 40 features,
    #  never trained on).
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("CHECK 2 | Held-out synthetic set (seed 918273)")
    line("=")

    df_held_out = pd.read_csv(HELD_OUT_SYNTH_CSV)
    y_held_out = df_held_out["is_fraud"].astype(int).to_numpy()
    df_held_out_fe = compute_engineered_features(df_held_out, train_stats)
    print(f"stealth_fraud_score on held-out synthetic: mean={df_held_out_fe['stealth_fraud_score'].mean():.4f} "
          f"non-null={df_held_out_fe['stealth_fraud_score'].notna().sum()}/{len(df_held_out_fe)}")
    records_held_out = df_held_out_fe.to_dict(orient="records")

    t0 = time.time()
    with swap_ps_state(model_fe, prep_fe, FEATURE_NAMES, threshold=DEPLOYED_THRESHOLD):
        scored_synth = score_ensemble_dataframe(records_held_out)
    metrics_synth = metrics_from_scores(y_held_out, scored_synth["ensemble_scores"], DEPLOYED_THRESHOLD)
    print(f"  Feature-engineered: ROC-AUC={metrics_synth['roc_auc']:.4f} PR-AUC={metrics_synth['pr_auc']:.4f} "
          f"recall={metrics_synth['recall']:.4f} TP={metrics_synth['tp']}/{metrics_synth['n_fraud']} "
          f"({time.time()-t0:.0f}s)")
    results["fe_held_out_synthetic_ensemble"] = metrics_synth

    # ════════════════════════════════════════════════════════════════════
    #  CHECK 3 — real Dataset 1 (74,917 rows, 15% features honestly mappable)
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("CHECK 3 | Real Dataset 1 (upi_fraud_dataset.csv, 74,917 rows)")
    line("=")

    mapped1, y1, stats1 = load_dataset_1()
    mapped1_fe = compute_engineered_features(mapped1, train_stats)
    print(f"stealth_fraud_score on Dataset 1: non-null={mapped1_fe['stealth_fraud_score'].notna().sum()}/{len(mapped1_fe)} "
          f"(expected 0 -- Dataset 1 supplies no hard-flag/behavioral columns)")
    print(f"hour_sin/hour_cos on Dataset 1: non-null={mapped1_fe['hour_sin'].notna().sum()}/{len(mapped1_fe)} "
          f"(expected {len(mapped1_fe)} -- Dataset 1 maps hour_of_day directly)")
    records1 = mapped1_fe.to_dict(orient="records")

    t0 = time.time()
    with swap_ps_state(model_fe, prep_fe, FEATURE_NAMES, threshold=DEPLOYED_THRESHOLD):
        scored_d1 = score_ensemble_dataframe(records1)
    metrics_d1 = metrics_from_scores(y1.to_numpy(), scored_d1["ensemble_scores"], DEPLOYED_THRESHOLD)
    print(f"  Feature-engineered: ROC-AUC={metrics_d1['roc_auc']:.4f} PR-AUC={metrics_d1['pr_auc']:.4f} "
          f"TP={metrics_d1['tp']}/{metrics_d1['n_fraud']} max_score={metrics_d1['max_score']:.4f} "
          f"({time.time()-t0:.0f}s)")
    results["fe_dataset1_ensemble"] = metrics_d1

    # ════════════════════════════════════════════════════════════════════
    #  CHECK 4 — real Dataset 3 (1,000 rows, low power)
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("CHECK 4 | Real Dataset 3 (low power, 64 fraud)")
    line("=")

    mapped3, y3, stats3 = load_dataset_3()
    mapped3_fe = compute_engineered_features(mapped3, train_stats)
    records3 = mapped3_fe.to_dict(orient="records")

    with swap_ps_state(model_fe, prep_fe, FEATURE_NAMES, threshold=DEPLOYED_THRESHOLD):
        scored_d3 = score_ensemble_dataframe(records3)
    metrics_d3 = metrics_from_scores(y3.to_numpy(), scored_d3["ensemble_scores"], DEPLOYED_THRESHOLD)
    print(f"  Feature-engineered: ROC-AUC={metrics_d3['roc_auc']:.4f} PR-AUC={metrics_d3['pr_auc']:.4f} "
          f"TP={metrics_d3['tp']}/{metrics_d3['n_fraud']}")
    results["fe_dataset3_ensemble"] = metrics_d3

    out_path = os.path.join(BASE_DIR, "eda_feature_engineering_results.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved -> {out_path}")

    line("=")
    print("DONE")
    line("=")
    return results


if __name__ == "__main__":
    main()
