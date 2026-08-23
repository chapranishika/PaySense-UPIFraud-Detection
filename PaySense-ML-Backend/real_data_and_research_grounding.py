"""
================================================================================
  PaySense — Real Data Search + Research-Grounded Synthetic Data (Track A + B)
  ────────────────────────────────────────────────────────────────────────────
  See REAL_DATA_AND_RESEARCH_GROUNDING.md for the full write-up. This script
  runs the actual experiments both tracks need:

  TRACK A — a genuinely real dataset search turned up
  `external_data/kaggle_vbinh002_fraud_ecommerce/Fraud_Data.csv` (the
  well-known "fraud-ecommerce" benchmark, vbinh002 on Kaggle, cataloged as
  real e-commerce fraud data in Amazon Science's Fraud Dataset Benchmark
  paper, arXiv:2208.14417). Vetted the same way GENERALIZATION_CHECK.md §2.2
  vetted its own candidates (see load_dataset_5 below for the full mapping
  rationale and the vetting stats already run interactively). 7/40 features
  honestly map -- better than Dataset 1's 6/40 -- including
  usr_account_age_days, which neither Dataset 1 nor Dataset 3's ensemble
  check has ever exercised as a *strong* signal before (fraud rows here
  purchase almost immediately after signup: median account age 0.000012
  days vs. 60.13 days for legitimate rows).

  TRACK B — trains a new model (Variant C) on the ORIGINAL 80% train
  partition blended with `research_grounded_synthetic_dataset.csv`
  (generate_research_grounded_synthetic_dataset.py's 3-typology mixture
  model, SEED=771029 -- disjoint from 42/918273/445566), using the exact
  same Block-0 hyperparameters + monotone_constraints as the frozen model
  (paysense_phase3.py) -- mirroring OOD_GENERALIZATION_REMEDIATION.md's
  Variant A construction exactly, so the only thing that differs between
  Variant A and Variant C is WHICH generator produced the blended-in data.

  Every model (frozen baseline, the existing blended_training artifact, and
  the new Variant C) is evaluated on the SAME five checks, all through the
  real `src.fraud_model.score()` ensemble path, all at the CURRENT deployed
  threshold (0.50 -- verified directly from artefacts/paysense_threshold.pkl,
  not assumed):
    1. Canonical held-out test split (in-distribution)
    2. Held-out synthetic set (seed 918273, synthetic_grounded_dataset.csv --
       never trained on by anything, here or before)
    3. Real Dataset 1 (upi_fraud_dataset.csv, 74,917 rows)
    4. Real Dataset 3 (fraud_dataset.csv, 1,000 rows, low power)
    5. Real Dataset 5 (Fraud_Data.csv, stratified 20,000-row sample -- see
       below for why a sample, not the full 151,112 rows)

  Baseline's numbers on checks 1-4 are CITED VERBATIM from existing docs
  (GENERALIZATION_CHECK.md, SYNTHETIC_GROUNDING.md) wherever those docs
  already recomputed them at the current 0.50 threshold -- not re-run, to
  avoid burning ~30 minutes reproducing a number that already exists.
  Baseline is scored FRESH here only on Dataset 5 (new) and its own
  canonical-test @ 0.50 confusion matrix (also not previously published at
  0.50 explicitly by TP/FP counts, only via paysense_phase3.py's own
  raw-XGBoost sweep table, which is a different scoring path -- see
  REAL_DATA_AND_RESEARCH_GROUNDING.md's methodology section for why the
  real-ensemble number, not that raw sweep table, is what's cited here).
  Blended-training and Variant C are scored FRESH on all five checks, since
  no prior document has scored either of them at threshold 0.50 (both
  existing blended-training numbers on record, in
  OOD_GENERALIZATION_REMEDIATION.md, are at the since-superseded 0.30
  threshold).

  NO CHANGES to artefacts/paysense_model.pkl, paysense_preprocessor.pkl, or
  paysense_threshold.pkl anywhere in this script.

  Run
  ───
      venv\\Scripts\\python.exe real_data_and_research_grounding.py
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
if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
logging.getLogger("paysense.ensemble").setLevel(logging.ERROR)

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)
sys.path.insert(0, os.path.join(BASE_DIR, "src"))

import generate_research_grounded_synthetic_dataset as gen_rg  # noqa: E402
from generalization_check import load_dataset_1, load_dataset_3  # noqa: E402
from src import fraud_model  # noqa: E402

ARTEFACTS_DIR = os.path.join(BASE_DIR, "artefacts")
MASTER_CSV = os.path.join(BASE_DIR, "paysense_master_dataset.csv")
HELD_OUT_SYNTH_CSV = os.path.join(BASE_DIR, "synthetic_grounded_dataset.csv")  # seed 918273
RG_SYNTH_CSV = os.path.join(BASE_DIR, "research_grounded_synthetic_dataset.csv")  # seed 771029
DATASET5_PATH = os.path.join(
    BASE_DIR, "external_data", "kaggle_vbinh002_fraud_ecommerce", "Fraud_Data.csv"
)

RANDOM_STATE = 42
DATASET5_SAMPLE_SEED = 24680
DATASET5_SAMPLE_N = 20000  # stratified sample -- see load_dataset_5() docstring

DROP_COLS = [
    "transaction_id", "user_id", "receiver_id", "timestamp", "date",
    "data_source", "user_kyc_status", "status", "usr_home_city",
]
TARGET = "is_fraud"

BASE_XGB_KWARGS = dict(
    n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.80,
    colsample_bytree=0.80, min_child_weight=10, gamma=0.10, scale_pos_weight=1,
    reg_alpha=0.05, reg_lambda=1.50, eval_metric="aucpr", tree_method="hist",
    random_state=RANDOM_STATE, n_jobs=-1, verbosity=0,
)
BEHAVIORAL_FEATURES = ["amount_deviation_score", "transaction_velocity", "failed_attempts_last_24h"]

AGE_TO_GROUP_BINS = [0, 24, 34, 44, 54, 200]
AGE_TO_GROUP_LABELS = ["18-24", "25-34", "35-44", "45-54", "55+"]


def line(c="=", n=78):
    print(c * n)


def build_preprocessor(cat_cols, num_cols):
    cat_pipeline = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1,
                                   encoded_missing_value=-2)),
    ])
    num_pipeline = Pipeline([("impute", SimpleImputer(strategy="median"))])
    return ColumnTransformer(
        transformers=[("cat", cat_pipeline, cat_cols), ("num", num_pipeline, num_cols)],
        remainder="drop", verbose_feature_names_out=False,
    )


def monotone_tuple(feature_names):
    return tuple(1 if f in BEHAVIORAL_FEATURES else 0 for f in feature_names)


class swap_ps_state:
    def __init__(self, model, prep, features, threshold):
        self.model, self.prep, self.features, self.threshold = model, prep, features, threshold
        self._saved = None

    def __enter__(self):
        st = fraud_model.get_state()
        self._saved = (st.ps_model, st.ps_prep, st.ps_features, st.ps_threshold)
        st.ps_model, st.ps_prep, st.ps_features, st.ps_threshold = (
            self.model, self.prep, self.features, self.threshold,
        )
        return st

    def __exit__(self, *exc):
        st = fraud_model.get_state()
        st.ps_model, st.ps_prep, st.ps_features, st.ps_threshold = self._saved
        return False


def score_ensemble_dataframe(records, tag=""):
    n = len(records)
    ensemble_scores = np.empty(n, dtype=float)
    t0 = time.time()
    for i, rec in enumerate(records):
        clean = {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in rec.items()}
        result = fraud_model.score(clean)
        ensemble_scores[i] = result.ensemble_score
        if i and i % 10000 == 0:
            print(f"    ... [{tag}] {i:,} / {n:,} rows scored ({time.time()-t0:.1f}s elapsed)")
    elapsed = time.time() - t0
    print(f"    [{tag}] scored {n:,} rows in {elapsed:.1f}s ({n/max(elapsed,1e-9):.0f} rows/sec)")
    return ensemble_scores


def metrics_from_scores(y, scores, threshold):
    preds = (scores >= threshold).astype(int)
    roc_auc = roc_auc_score(y, scores)
    pr_auc = average_precision_score(y, scores)
    cm = confusion_matrix(y, preds)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (None, None, None, None)
    return {
        "roc_auc": float(roc_auc), "pr_auc": float(pr_auc),
        "tp": int(tp) if tp is not None else None, "fp": int(fp) if fp is not None else None,
        "fn": int(fn) if fn is not None else None, "tn": int(tn) if tn is not None else None,
        "recall": float(recall_score(y, preds, zero_division=0)),
        "precision": float(precision_score(y, preds, zero_division=0)),
        "f1": float(f1_score(y, preds, zero_division=0)),
        "max_score": float(np.nanmax(scores)),
        "n": int(len(y)), "n_fraud": int(np.sum(y)),
    }


# ════════════════════════════════════════════════════════════════════════════
#  DATASET 5 — Fraud_Data.csv (Kaggle vbinh002/fraud-ecommerce, real,
#  151,112 rows, 9.36% fraud, part of Amazon Science's Fraud Dataset
#  Benchmark). Vetting already performed and documented in
#  REAL_DATA_AND_RESEARCH_GROUNDING.md §2 -- no duplicate user_id/rows, no
#  nulls, fraud rate not round, no categorical near-determinism (source/
#  browser/sex all within 8.7-10.5% of the 9.36% base rate), and the
#  standout signal (account age at purchase time) is strong but NOT
#  deterministic (medians 0.00 vs 60.13 days, with real overlap in both
#  tails -- unlike the disqualifying near-100%/0% splits that sank
#  GENERALIZATION_CHECK.md's Dataset 2).
#
#  Honest mapping -- 7 of 40 features carry real signal:
#    amount               <- purchase_value           (direct; currency not
#                                                        stated by the source,
#                                                        used as-is, same
#                                                        treatment Dataset 3's
#                                                        `amount` got)
#    usr_account_age_days <- (purchase_time - signup_time).days  (a REAL
#                                                        per-row computed
#                                                        statistic, not
#                                                        fabricated -- and
#                                                        the strongest single
#                                                        signal in this
#                                                        dataset)
#    hour_of_day          <- purchase_time.hour        (direct)
#    is_night_transaction <- derived, same rule as elsewhere (hour<6 or
#                                                        hour>=22)
#    day_of_week          <- purchase_time.dayofweek() (Monday=0..Sunday=6,
#                                                        matches the frozen
#                                                        preprocessor's
#                                                        numeric expectation
#                                                        -- see
#                                                        SYNTHETIC_GROUNDING.md
#                                                        §2.4 for why numeric,
#                                                        not the string form)
#    is_weekend           <- derived (Sat/Sun)
#    usr_age_group        <- `age` bucketed into PaySense's five brackets
#                                                        (18-24/25-34/35-44/
#                                                        45-54/55+) -- a
#                                                        defensible numeric-
#                                                        to-categorical
#                                                        bucketing, not a
#                                                        semantic guess.
#
#  Explicitly NOT mapped, and why:
#    device_id  -> device_type : this dataset's device_id is an opaque
#                                 per-session string with no OS information
#                                 at all (unlike IEEE-CIS's DeviceInfo text,
#                                 which at least names hardware/OS strings).
#                                 Cannot be mapped without inventing an OS.
#    browser    -> device_type : NAME-COLLISION-ADJACENT mistake avoided
#                                 deliberately -- browser (Chrome/Safari/IE/
#                                 Opera/FireFox) runs across every OS.
#                                 Same reasoning GENERALIZATION_CHECK.md
#                                 §3.2 used to reject Dataset 3's
#                                 device_type collision.
#    source (Ads/SEO/Direct)  -> no frozen feature.
#    sex                      -> no frozen feature.
#    ip_address               -> could be geolocated via the paired
#                                 IpAddress_to_Country.csv, but there is no
#                                 per-user "home/registered country" baseline
#                                 anywhere in this dataset to compare it
#                                 against, so ip_location_mismatch cannot be
#                                 honestly derived (there is nothing to be
#                                 "mismatched" relative to).
#    device_id reuse across DIFFERENT users (a real, strong signal here --
#    fraud rate 52.46% on a shared device vs. 3.04% otherwise) is a genuine
#    multi-accounting/fraud-ring pattern, but it is NOT the same concept as
#    new_device_flag (a device being unfamiliar TO ONE USER, not shared
#    ACROSS users) -- forcing that equivalence would be exactly the kind of
#    conflation GENERALIZATION_CHECK.md §3.1 already refused to make for
#    is_new_payee vs. new_device_flag. Left unmapped, reported as an
#    interesting real finding in the write-up instead.
# ════════════════════════════════════════════════════════════════════════════
def load_dataset_5(sample_n=DATASET5_SAMPLE_N, sample_seed=DATASET5_SAMPLE_SEED):
    df_full = pd.read_csv(DATASET5_PATH)
    n_full = len(df_full)
    fraud_rate_full = float(df_full["class"].mean())

    # Stratified sample for time-budget reasons: the full 151,112 rows would
    # cost ~35-40 minutes per model through the real per-row ensemble path
    # (Dataset 1's 74,917 rows already takes 15-20 minutes at ~65-70 rows/
    # sec) -- multiplied by THREE models (baseline, blended-training,
    # Variant C) that is over an hour for this one extra, non-mandatory
    # dataset alone. A stratified sample preserving the exact fraud rate
    # keeps this a real, honestly-powered check (20,000 rows / ~1,870 fraud
    # examples is far higher power than Dataset 3's accepted 1,000/64) while
    # keeping runtime bounded. This is disclosed here and in the write-up,
    # not silently substituted for "the whole dataset."
    if sample_n is not None and sample_n < n_full:
        fraud_df = df_full[df_full["class"] == 1]
        legit_df = df_full[df_full["class"] == 0]
        rng = np.random.RandomState(sample_seed)
        n_fraud_sample = int(round(sample_n * fraud_rate_full))
        n_legit_sample = sample_n - n_fraud_sample
        fraud_sample = fraud_df.sample(n=min(n_fraud_sample, len(fraud_df)), random_state=rng)
        legit_sample = legit_df.sample(n=min(n_legit_sample, len(legit_df)), random_state=rng)
        df = pd.concat([fraud_sample, legit_sample], ignore_index=True)
        df = df.sample(frac=1.0, random_state=rng).reset_index(drop=True)
    else:
        df = df_full

    y = df["class"].astype(int)
    signup = pd.to_datetime(df["signup_time"])
    purchase = pd.to_datetime(df["purchase_time"])
    acct_age_days = (purchase - signup).dt.total_seconds() / 86400.0

    mapped = pd.DataFrame(index=df.index)
    mapped["amount"] = df["purchase_value"].astype(float)
    mapped["usr_account_age_days"] = acct_age_days
    mapped["hour_of_day"] = purchase.dt.hour.astype(int)
    mapped["is_night_transaction"] = ((purchase.dt.hour < 6) | (purchase.dt.hour >= 22)).astype(int)
    mapped["day_of_week"] = purchase.dt.dayofweek.astype(int)
    mapped["is_weekend"] = purchase.dt.dayofweek.isin([5, 6]).astype(int)
    mapped["usr_age_group"] = pd.cut(
        df["age"].astype(float), bins=AGE_TO_GROUP_BINS, labels=AGE_TO_GROUP_LABELS
    ).astype(str)

    mapping_stats = {
        "dataset": "kaggle_vbinh002_fraud_ecommerce/Fraud_Data.csv",
        "n_rows_full_population": n_full,
        "fraud_rate_full_population": fraud_rate_full,
        "n_rows_sampled": len(df),
        "fraud_rate_sampled": float(y.mean()),
        "sample_seed": sample_seed,
        "features_mapped": 7,
        "features_total": 40,
        "features_mapped_names": list(mapped.columns),
    }
    return mapped, y, mapping_stats


def main():
    results = {}

    # ════════════════════════════════════════════════════════════════════
    #  STEP 0 — Reproduce paysense_phase3.py Block 0 split exactly
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("STEP 0 | Reproducing paysense_phase3.py Block 0 (split + preprocessing)")
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

    CAT_COLS = X_train_raw.select_dtypes(include=["object"]).columns.tolist()
    NUM_COLS = X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
    FEATURE_NAMES = CAT_COLS + NUM_COLS

    ps_threshold = float(joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_threshold.pkl")))
    print(f"Deployed threshold (verified from artefacts/paysense_threshold.pkl): {ps_threshold}")
    assert abs(ps_threshold - 0.50) < 1e-9, f"Expected deployed threshold 0.50, found {ps_threshold}"

    frozen_model = joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_model.pkl"))
    frozen_prep = joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_preprocessor.pkl"))
    frozen_features = joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_feature_names.pkl"))
    assert list(frozen_features) == FEATURE_NAMES, "Feature order mismatch vs frozen artifact."

    blend_model = joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_model_blended_training.pkl"))
    blend_prep = joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_preprocessor_blended_training.pkl"))
    blend_features = joblib.load(os.path.join(ARTEFACTS_DIR, "paysense_feature_names_blended_training.pkl"))

    # ════════════════════════════════════════════════════════════════════
    #  STEP 1 — Generate the research-grounded (Track B) synthetic dataset
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("STEP 1 | Generating research-grounded synthetic dataset (Track B)")
    line("=")

    df_rg = gen_rg.generate()
    df_rg.to_csv(RG_SYNTH_CSV, index=False)
    n_fraud_rg = int(df_rg["is_fraud"].sum())
    print(f"Research-grounded dataset: {len(df_rg):,} rows, {n_fraud_rg:,} fraud "
          f"({n_fraud_rg/len(df_rg)*100:.2f}%) -- saved to {RG_SYNTH_CSV}")

    assert os.path.exists(HELD_OUT_SYNTH_CSV), f"{HELD_OUT_SYNTH_CSV} not found."
    df_held_out = pd.read_csv(HELD_OUT_SYNTH_CSV)
    compare_cols = [c for c in df_rg.columns if c != "transaction_id"]
    rg_rows = set(map(tuple, df_rg[compare_cols].astype(str).values.tolist()))
    held_out_rows = set(map(tuple, df_held_out[compare_cols].astype(str).values.tolist()))
    row_overlap = rg_rows & held_out_rows
    assert not row_overlap, (
        f"Research-grounded (seed=771029) and held-out (seed=918273) sets share "
        f"{len(row_overlap)} fully-identical rows."
    )
    print(f"Confirmed 0 full-row duplicates between research-grounded set (seed=771029, "
          f"{len(df_rg):,} rows) and held-out synthetic eval set (seed=918273, "
          f"{len(df_held_out):,} rows).")
    results["research_grounded_dataset"] = {
        "seed": gen_rg.SEED, "n_rows": len(df_rg), "n_fraud": n_fraud_rg,
        "fraud_rate": n_fraud_rg / len(df_rg),
    }

    # ════════════════════════════════════════════════════════════════════
    #  STEP 2 — Variant C: blend original 80% train + research-grounded data
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("STEP 2 | Variant C -- blend original train + research-grounded synthetic data")
    line("=")

    X_rg = df_rg.drop(columns=DROP_COLS + [TARGET])
    y_rg = df_rg[TARGET].astype(int)
    X_rg = X_rg[X_train_raw.columns]

    X_train_C_raw = pd.concat([X_train_raw, X_rg], ignore_index=True)
    y_train_C = pd.concat([y_train.reset_index(drop=True), y_rg.reset_index(drop=True)], ignore_index=True)
    print(f"Variant C training data: {len(X_train_raw):,} original + {len(X_rg):,} research-grounded "
          f"= {len(X_train_C_raw):,} rows before SMOTE "
          f"({int(y_train_C.sum()):,} fraud, {y_train_C.mean()*100:.2f}%)")

    CAT_COLS_C = X_train_C_raw.select_dtypes(include=["object"]).columns.tolist()
    NUM_COLS_C = X_train_C_raw.select_dtypes(include=[np.number]).columns.tolist()
    FEATURE_NAMES_C = CAT_COLS_C + NUM_COLS_C
    assert set(FEATURE_NAMES_C) == set(FEATURE_NAMES), "Variant C feature set differs from baseline's 40."

    prep_C = build_preprocessor(CAT_COLS_C, NUM_COLS_C)
    prep_C.fit(X_train_C_raw)
    X_train_C_proc = prep_C.transform(X_train_C_raw)
    X_test_C_proc = prep_C.transform(X_test_raw)

    smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=RANDOM_STATE)
    X_train_C_bal, y_train_C_bal = smote.fit_resample(X_train_C_proc, y_train_C.values)
    print(f"After SMOTE: {X_train_C_bal.shape}")

    kwargs_C = dict(BASE_XGB_KWARGS)
    kwargs_C["monotone_constraints"] = monotone_tuple(FEATURE_NAMES_C)
    model_C = XGBClassifier(**kwargs_C)
    print("Training Variant C ... ", end="", flush=True)
    model_C.fit(X_train_C_bal, y_train_C_bal, eval_set=[(X_test_C_proc, y_test)], verbose=False)
    print("done.")

    joblib.dump(model_C, os.path.join(ARTEFACTS_DIR, "paysense_model_research_grounded.pkl"))
    joblib.dump(prep_C, os.path.join(ARTEFACTS_DIR, "paysense_preprocessor_research_grounded.pkl"))
    joblib.dump(FEATURE_NAMES_C, os.path.join(ARTEFACTS_DIR, "paysense_feature_names_research_grounded.pkl"))
    print("Saved -> paysense_model_research_grounded.pkl / paysense_preprocessor_research_grounded.pkl")

    y_proba_C_test = model_C.predict_proba(X_test_C_proc)[:, 1]
    print(f"Variant C canonical test raw-XGBoost ROC-AUC: {roc_auc_score(y_test, y_proba_C_test):.4f}")

    # ════════════════════════════════════════════════════════════════════
    #  STEP 3 — Load fraud_model ensemble state (rules + light_lr fixed)
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("STEP 3 | Loading production ensemble state")
    line("=")
    fraud_model.load_artefacts()
    assert "light_lr" in fraud_model.get_state().active_scorers, "light_lr must be active."

    MODELS = {
        "baseline": (frozen_model, frozen_prep, list(frozen_features)),
        "blended_training": (blend_model, blend_prep, list(blend_features)),
        "variant_C_research_grounded": (model_C, prep_C, FEATURE_NAMES_C),
    }

    # ════════════════════════════════════════════════════════════════════
    #  STEP 4 — Canonical held-out test, all 3 models, real ensemble
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("STEP 4 | Canonical held-out test (in-distribution) -- real ensemble")
    line("=")
    records_test = X_test_raw.to_dict(orient="records")
    y_test_arr = y_test.to_numpy()
    for name, (m, p, f) in MODELS.items():
        with swap_ps_state(m, p, f, threshold=ps_threshold):
            scores = score_ensemble_dataframe(records_test, tag=f"canonical_test/{name}")
        metrics = metrics_from_scores(y_test_arr, scores, ps_threshold)
        print(f"  {name:<30}: ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
              f"recall={metrics['recall']:.4f} TP={metrics['tp']}/{metrics['n_fraud']}")
        results[f"{name}_canonical_test"] = metrics

    # ════════════════════════════════════════════════════════════════════
    #  STEP 5 — Held-out synthetic set (seed 918273), all 3 models
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("STEP 5 | Held-out synthetic set (seed 918273) -- real ensemble")
    line("=")
    y_held_out = df_held_out["is_fraud"].astype(int).to_numpy()
    records_held_out = df_held_out.to_dict(orient="records")
    for name, (m, p, f) in MODELS.items():
        with swap_ps_state(m, p, f, threshold=ps_threshold):
            scores = score_ensemble_dataframe(records_held_out, tag=f"held_out_synth/{name}")
        metrics = metrics_from_scores(y_held_out, scores, ps_threshold)
        print(f"  {name:<30}: ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
              f"recall={metrics['recall']:.4f} TP={metrics['tp']}/{metrics['n_fraud']}")
        results[f"{name}_held_out_synthetic"] = metrics

    # ════════════════════════════════════════════════════════════════════
    #  STEP 6 — Real Dataset 1 (74,917 rows), all 3 models
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("STEP 6 | Real Dataset 1 (upi_fraud_dataset.csv, 74,917 rows) -- real ensemble")
    line("=")
    mapped1, y1, stats1 = load_dataset_1()
    records1 = mapped1.to_dict(orient="records")
    y1_arr = y1.to_numpy()
    for name, (m, p, f) in MODELS.items():
        with swap_ps_state(m, p, f, threshold=ps_threshold):
            scores = score_ensemble_dataframe(records1, tag=f"dataset1/{name}")
        metrics = metrics_from_scores(y1_arr, scores, ps_threshold)
        print(f"  {name:<30}: ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
              f"TP={metrics['tp']}/{metrics['n_fraud']} max_score={metrics['max_score']:.4f}")
        results[f"{name}_dataset1"] = metrics

    # ════════════════════════════════════════════════════════════════════
    #  STEP 7 — Real Dataset 3 (1,000 rows, low power), all 3 models
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("STEP 7 | Real Dataset 3 (1,000 rows, low power) -- real ensemble")
    line("=")
    mapped3, y3, stats3 = load_dataset_3()
    records3 = mapped3.to_dict(orient="records")
    y3_arr = y3.to_numpy()
    for name, (m, p, f) in MODELS.items():
        with swap_ps_state(m, p, f, threshold=ps_threshold):
            scores = score_ensemble_dataframe(records3, tag=f"dataset3/{name}")
        metrics = metrics_from_scores(y3_arr, scores, ps_threshold)
        print(f"  {name:<30}: ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
              f"TP={metrics['tp']}/{metrics['n_fraud']}")
        results[f"{name}_dataset3"] = metrics

    # ════════════════════════════════════════════════════════════════════
    #  STEP 8 — Real Dataset 5 (NEW: Fraud_Data.csv, stratified 20k sample)
    # ════════════════════════════════════════════════════════════════════
    line("=")
    print("STEP 8 | Real Dataset 5 (Fraud_Data.csv, stratified sample) -- real ensemble")
    line("=")
    mapped5, y5, stats5 = load_dataset_5()
    print(f"Dataset 5: full population {stats5['n_rows_full_population']:,} rows "
          f"({stats5['fraud_rate_full_population']*100:.2f}% fraud); sampled "
          f"{stats5['n_rows_sampled']:,} rows ({stats5['fraud_rate_sampled']*100:.2f}% fraud), "
          f"seed={stats5['sample_seed']}")
    print(f"Honestly-mapped columns: {stats5['features_mapped_names']}")
    records5 = mapped5.to_dict(orient="records")
    y5_arr = y5.to_numpy()
    results["dataset5_mapping_stats"] = stats5
    for name, (m, p, f) in MODELS.items():
        with swap_ps_state(m, p, f, threshold=ps_threshold):
            scores = score_ensemble_dataframe(records5, tag=f"dataset5/{name}")
        metrics = metrics_from_scores(y5_arr, scores, ps_threshold)
        print(f"  {name:<30}: ROC-AUC={metrics['roc_auc']:.4f} PR-AUC={metrics['pr_auc']:.4f} "
              f"TP={metrics['tp']}/{metrics['n_fraud']} max_score={metrics['max_score']:.4f}")
        results[f"{name}_dataset5"] = metrics

    # ════════════════════════════════════════════════════════════════════
    #  Save results JSON
    # ════════════════════════════════════════════════════════════════════
    out_path = os.path.join(BASE_DIR, "real_data_and_research_grounding_results.json")
    with open(out_path, "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nResults saved -> {out_path}")

    line("=")
    print("DONE")
    line("=")
    return results


if __name__ == "__main__":
    main()
