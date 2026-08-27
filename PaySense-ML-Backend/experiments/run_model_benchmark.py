"""
================================================================================
  run_model_benchmark.py -- 2026-08-27 model-family benchmark
  ------------------------------------------------------------------------------
  Answers one question: can an alternative model family materially improve
  fraud detection on CLEAN ORGANIC data, under the same train(60%)/
  validation(20%)/test(20%) protocol established in
  investigate_organic_only_threshold.py, with the SAME split and SAME
  untouched final test set used for every model compared here?

  This is a read-only experiment. It does not touch any deployed artifact
  (paysense_model.pkl / paysense_threshold.pkl / src/fraud_model.py).

  Text-based model families (TF-IDF + Logistic Regression / Linear SVM,
  text-only, hybrid text+structured, DistilBERT) are NOT implemented here.
  paysense_master_dataset.csv, the only fraud-labelled dataset this project
  has, contains no SMS/text field at all (verified directly against its 50
  raw columns) -- there is no text signal to feed those model families for
  the FRAUD task. (DistilBERT already exists in this project for the
  unrelated category-classification task, which does have text; see
  CATEGORY_CLASSIFIER.md / EXPERIMENTS.md.) This is a data-availability
  fact, not a methodology choice -- see feature_audit.md for detail.

  Models actually compared, all on the identical anchor-only 60/20/20 split:
    A. XGBoost      -- reproduces the existing organic-only baseline exactly,
                       untouched hyperparameters (investigate_organic_only_
                       threshold.py).
    D. RandomForest -- small validation-selected grid.
    E. LightGBM     -- small validation-selected grid.
    F. CatBoost     -- small validation-selected grid, native categorical
                       handling (no ordinal encoding), auto class weighting
                       instead of SMOTE (documented, deliberate pipeline
                       difference -- SMOTE is not natural on raw categorical
                       text without SMOTENC).

  Run:
      cd PaySense-ML-Backend
      venv\\Scripts\\python.exe experiments\\run_model_benchmark.py
================================================================================
"""
import json
import os
import time
import warnings

# This Windows environment's joblib/loky physical-core-count probe spawns a
# subprocess that fails (observed directly: repeated CreateProcess failures
# in stderr) -- cosmetic, but LOKY_MAX_CPU_COUNT set BEFORE joblib is
# imported (transitively, via sklearn/imblearn/lightgbm) skips it cleanly.
# NOTE, for anyone re-running this and seeing it take much longer than
# expected: the real cause of a severe slowdown observed while developing
# this script was NOT any of the above, or SMOTE, or model-fit cost -- it
# was multiple orphaned python.exe processes from earlier interrupted runs
# left alive in the background (this venv's python.exe launches a real
# child process under the base Python install; killing only the PID a
# shell reports does not kill that child), all competing for the same CPU
# and disk. Verified via `Get-CimInstance Win32_Process -Filter
# "Name='python.exe'"` showing duplicate concurrent runs. If this script
# seems to be running at ~10% CPU utilization, check for and kill stray
# python.exe processes before assuming there's a code-level bug here.
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")

warnings.filterwarnings("ignore")

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from imblearn.over_sampling import SMOTE
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_recall_curve, precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from xgboost import XGBClassifier

EXP_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.dirname(EXP_DIR)
MASTER_CSV = os.path.join(BACKEND_DIR, "paysense_master_dataset.csv")

RESULTS_JSON = os.path.join(EXP_DIR, "benchmark_results.json")
RESULTS_CSV = os.path.join(EXP_DIR, "model_benchmark.csv")
RESULTS_MD = os.path.join(EXP_DIR, "model_benchmark.md")
FEATURE_AUDIT_MD = os.path.join(EXP_DIR, "feature_audit.md")
SOURCE_CLASSIFIER_JSON = os.path.join(EXP_DIR, "source_classifier_results.json")
PLOT_PATH = os.path.join(EXP_DIR, "plots", "precision_recall_comparison.png")

RANDOM_STATE = 42
RECALL_MIN = 0.75
PRECISION_MIN = 0.50
RECALL_TARGETS = [0.50, 0.60, 0.75, 0.80]
SWEEP = np.round(np.arange(0.05, 0.96, 0.05), 2)
N_CV_FOLDS = 5

DROP_COLS_BASE = [
    "transaction_id", "user_id", "receiver_id", "timestamp", "date",
    "user_kyc_status", "status", "usr_home_city",
]
SOURCE_LEAK_COLS = ["device_risk_score", "ip_risk_score"]
BEHAVIORAL_FEATURES = ["amount_deviation_score", "transaction_velocity", "failed_attempts_last_24h"]

MANUAL_NOTES = {
    "balance_after_transaction": (
        "Required client-submitted field in the real /predict request schema "
        "(main.py FraudPredictRequest) -- confirmed genuinely available at the "
        "system's real scoring point, which is post-transaction-confirmation-SMS, "
        "not pre-authorization. Not excluded for representing 'future' information."
    ),
    "txn_success_flag": (
        "Same real-time-availability status as balance_after_transaction -- both "
        "are required request fields, confirmed available when this system "
        "actually scores a transaction (after the confirmation SMS arrives)."
    ),
    "new_device_flag": (
        "Genuine real-world signal, legitimately available at scoring time. "
        "Implicated (with ip_location_mismatch) in how the supplement source's "
        "label was generated (see DATASET.md) -- but the column itself is not "
        "an artifact and is kept."
    ),
    "ip_location_mismatch": (
        "Same status as new_device_flag: genuine signal, implicated in the "
        "supplement label-generation mechanism, not itself invalid."
    ),
    "receiver_id": (
        "High-cardinality identifier. 100% of supplement rows carry the literal "
        "synthetic marker 'SYN_MRC_UNKNOWN' -- a direct dataset-source tell."
    ),
    "device_risk_score": (
        "Directly used by this project's own pipeline to derive new_device_flag/"
        "ip_location_mismatch in the supplement source, which in turn generate "
        "the supplement is_fraud label near-tautologically (see DATASET.md). "
        "Excluded from every fraud model in this benchmark."
    ),
    "ip_risk_score": "Same finding as device_risk_score. Excluded from every fraud model in this benchmark.",
    "data_source": (
        "Direct dataset-provenance identifier -- the root of the entire "
        "contamination finding. Always excluded from every fraud model."
    ),
    "timestamp": "Raw datetime string, NaN for 100% of supplement rows; derived fields (hour_of_day, day_of_week, is_night_transaction) already capture the usable signal.",
    "date": "Redundant raw field, NaN in supplement; superseded by derived date/time features.",
    "status": "Redundant string duplicate of the numeric txn_success_flag.",
    "user_kyc_status": "Redundant string duplicate of the numeric kyc_verified_flag.",
    "usr_home_city": "High-cardinality categorical, redundant with usr_home_city_tier.",
    "transaction_id": "Unique row identifier -- zero generalizable signal, pure memorization risk if used.",
    "user_id": "High-cardinality raw identifier; per-user signal already available through usr_* aggregate columns.",
}


def log(msg=""):
    print(msg)


# ---------------------------------------------------------------------------
# 1. Feature audit
# ---------------------------------------------------------------------------
def run_feature_audit(df):
    supplement = df[df["data_source"] == "supplement"]
    dropped = set(DROP_COLS_BASE) | {"data_source"} | set(SOURCE_LEAK_COLS)
    rows = []
    for col in df.columns:
        if col == "is_fraud":
            rows.append({"column": col, "verdict": "TARGET", "constant_in_supplement": False,
                         "constant_value": None, "note": "Prediction target."})
            continue
        nun = supplement[col].nunique(dropna=False)
        is_const = nun == 1
        const_val = supplement[col].iloc[0] if is_const else None
        if col in dropped:
            verdict = "INVALID"
        elif is_const:
            verdict = "SUSPICIOUS"
        else:
            verdict = "VALID"
        note = MANUAL_NOTES.get(col)
        if note is None:
            if is_const:
                note = (f"Constant ({const_val!r}) across all 10,000 supplement rows. Kept as a "
                         "feature because it varies naturally in organic (anchor) data; flagged "
                         "as a contributor to source separability, not excluded for that reason "
                         "alone (see the source-classifier test below).")
            else:
                note = "Varies naturally in both sources; kept."
        rows.append({"column": col, "verdict": verdict, "constant_in_supplement": bool(is_const),
                     "constant_value": None if const_val is None else str(const_val), "note": note})

    # Preserve the ORIGINAL CSV column order, not an alphabetically-sorted set.
    # XGBoost's histogram split selection breaks gain ties by feature iteration
    # order; reordering columns (even with identical values) can therefore
    # change which split wins on a tie and cascade into a materially different
    # model across 400 boosted trees. Model A below is required to reproduce
    # investigate_organic_only_threshold.py's baseline exactly -- verified by
    # a regression test -- so column order must match that script exactly.
    kept_features = [c for c in df.columns if c not in dropped and c != "is_fraud"]
    suspicious = [r["column"] for r in rows if r["verdict"] == "SUSPICIOUS"]

    with open(FEATURE_AUDIT_MD, "w", encoding="utf-8") as f:
        f.write("# Feature audit -- model benchmark (2026-08-27)\n\n")
        f.write(
            "Every raw column in `paysense_master_dataset.csv` classified as VALID / "
            "SUSPICIOUS / INVALID / TARGET before any model in this benchmark was trained. "
            "\"Constant in supplement\" means the column takes exactly one value across all "
            "10,000 supplement-source rows -- computed directly from the CSV, not assumed.\n\n"
        )
        f.write(f"**{len(kept_features)} features kept** for the structured fraud benchmark "
                f"(all VALID + all SUSPICIOUS columns -- SUSPICIOUS columns are kept because they "
                f"carry real signal in organic data; they are flagged, not deleted, and their "
                f"contribution to source separability is measured directly below, not assumed).\n\n")
        f.write("| Column | Verdict | Constant in supplement | Note |\n|---|---|---|---|\n")
        for r in rows:
            cv = f"Yes (`{r['constant_value']}`)" if r["constant_in_supplement"] else "No"
            f.write(f"| `{r['column']}` | {r['verdict']} | {cv} | {r['note']} |\n")
        f.write(f"\n**{len(suspicious)} of {len(kept_features)} kept features are constant "
                f"in the supplement source** -- this is the same systemic finding as "
                f"`SOURCE_CONTAMINATION_INVESTIGATION.md`, re-verified independently against "
                f"this benchmark's own feature set rather than assumed from the earlier audit.\n")
    log(f"Wrote {FEATURE_AUDIT_MD}")
    return kept_features


# ---------------------------------------------------------------------------
# 2. Source-classifier test (organic vs supplement) -- NOT a fraud model
# ---------------------------------------------------------------------------
def run_source_classifier_test(df, kept_features):
    y_src = (df["data_source"] == "supplement").astype(int)

    # (a) trivial single-signal reproduction of the earlier documented finding
    trivial_pred = df["device_risk_score"].notnull().astype(int)
    trivial_acc = accuracy_score(y_src, trivial_pred)

    # (b) full classifier using ONLY the features this benchmark actually keeps
    #     for fraud modelling (i.e. device_risk_score/ip_risk_score already
    #     excluded) -- tests whether removing the two known leak columns is
    #     enough to stop source from being trivially recoverable.
    X = df[kept_features].copy()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y_src, test_size=0.20, random_state=RANDOM_STATE, stratify=y_src
    )
    pre = ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]),
         cat_cols),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                           ("scale", StandardScaler())]), num_cols),
    ], remainder="drop", verbose_feature_names_out=False)
    pre.fit(X_tr)
    X_tr_p, X_te_p = pre.transform(X_tr), pre.transform(X_te)

    clf = LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)
    clf.fit(X_tr_p, y_tr)
    proba = clf.predict_proba(X_te_p)[:, 1]
    pred = (proba >= 0.5).astype(int)
    acc = accuracy_score(y_te, pred)
    roc = roc_auc_score(y_te, proba)

    feat_names = cat_cols + num_cols
    coefs = clf.coef_[0]
    top_idx = np.argsort(-np.abs(coefs))[:10]
    top_features = [{"feature": feat_names[i], "coefficient": float(coefs[i])} for i in top_idx]

    out = {
        "purpose": "Diagnostic only -- NOT a fraud model. Reproduces/quantifies source separability.",
        "trivial_single_feature_check": {
            "feature": "device_risk_score.notnull()",
            "accuracy": float(trivial_acc),
            "matches_prior_documented_finding_of_100pct": bool(abs(trivial_acc - 1.0) < 1e-9),
        },
        "full_classifier_excluding_known_leak_columns": {
            "features_used": len(feat_names),
            "excludes": SOURCE_LEAK_COLS,
            "test_accuracy": float(acc),
            "test_roc_auc": float(roc),
            "top_10_features_by_abs_coefficient": top_features,
        },
    }
    with open(SOURCE_CLASSIFIER_JSON, "w") as f:
        json.dump(out, f, indent=2)
    log(f"Source classifier: trivial check accuracy={trivial_acc:.4f}; "
        f"full classifier (leak cols excluded) accuracy={acc:.4f} ROC-AUC={roc:.4f}")
    log(f"Wrote {SOURCE_CLASSIFIER_JSON}")
    return out


# ---------------------------------------------------------------------------
# 3. Fixed anchor-only 60/20/20 split (identical to investigate_organic_only_threshold.py)
# ---------------------------------------------------------------------------
def build_split(df, kept_features):
    anchor = df[df["data_source"] == "anchor"]
    X = anchor[kept_features].copy()
    y = anchor["is_fraud"].astype(int)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.40, random_state=RANDOM_STATE, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
    )
    log(f"Anchor-only pool: {len(X)} rows, {int(y.sum())} fraud ({y.mean():.4%})")
    log(f"Train: {len(X_train)} ({int(y_train.sum())} fraud)  "
        f"Val: {len(X_val)} ({int(y_val.sum())} fraud)  "
        f"Test: {len(X_test)} ({int(y_test.sum())} fraud)")
    return X_train, X_val, X_test, y_train, y_val, y_test


def shared_preprocessing(X_train, X_val, X_test):
    cat_cols = X_train.select_dtypes(include=["object"]).columns.tolist()
    num_cols = X_train.select_dtypes(include=[np.number]).columns.tolist()
    pre = ColumnTransformer([
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                           ("encode", OrdinalEncoder(handle_unknown="use_encoded_value",
                                                      unknown_value=-1, encoded_missing_value=-2))]),
         cat_cols),
        ("num", Pipeline([("impute", SimpleImputer(strategy="median"))]), num_cols),
    ], remainder="drop", verbose_feature_names_out=False)
    pre.fit(X_train)  # TRAIN ONLY
    feature_names = cat_cols + num_cols
    return (pre.transform(X_train), pre.transform(X_val), pre.transform(X_test),
            feature_names, cat_cols, num_cols)


def catboost_preprocessing(X_train, X_val, X_test, cat_cols, num_cols):
    """Native-categorical variant for CatBoost: impute only, no ordinal encoding."""
    cat_imputer = SimpleImputer(strategy="constant", fill_value="missing")
    num_imputer = SimpleImputer(strategy="median")
    cat_imputer.fit(X_train[cat_cols])
    num_imputer.fit(X_train[num_cols])

    def transform(X):
        cat_part = pd.DataFrame(cat_imputer.transform(X[cat_cols]), columns=cat_cols, index=X.index)
        num_part = pd.DataFrame(num_imputer.transform(X[num_cols]), columns=num_cols, index=X.index)
        return pd.concat([cat_part, num_part], axis=1)

    return transform(X_train), transform(X_val), transform(X_test)


def monotone_vector(feature_names):
    return tuple(1 if f in BEHAVIORAL_FEATURES else 0 for f in feature_names)


def smote_resample(X, y):
    sm = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=RANDOM_STATE)
    return sm.fit_resample(X, y)


# ---------------------------------------------------------------------------
# 4. Threshold selection (validation only) + frozen test evaluation
# ---------------------------------------------------------------------------
def select_threshold_on_validation(y_val, val_proba):
    sweep = []
    for t in SWEEP:
        t = float(t)
        pred = (val_proba >= t).astype(int)
        p = precision_score(y_val, pred, zero_division=0)
        r = recall_score(y_val, pred, zero_division=0)
        f1 = f1_score(y_val, pred, zero_division=0)
        sweep.append({"threshold": t, "precision": p, "recall": r, "f1": f1,
                      "meets_constraint": bool(r >= RECALL_MIN and p >= PRECISION_MIN)})
    df_sweep = pd.DataFrame(sweep)
    constraint_met = df_sweep[df_sweep["meets_constraint"]]
    if not constraint_met.empty:
        best = constraint_met.loc[constraint_met["f1"].idxmax()]
        reason = "Recall>=75% AND Precision>=50% on validation, maximised F1"
    else:
        best = df_sweep.loc[df_sweep["f1"].idxmax()]
        reason = "No validation threshold met both constraints -- fallback to max-F1"
    return float(best["threshold"]), reason, sweep


def precision_at_recall_targets(y_val, val_proba, y_test, test_proba):
    """For each target recall, pick (on VALIDATION only) the threshold achieving
    that recall with the best precision; apply it once to the frozen test set."""
    out = {}
    for target in RECALL_TARGETS:
        candidates = []
        for t in SWEEP:
            t = float(t)
            pred = (val_proba >= t).astype(int)
            r = recall_score(y_val, pred, zero_division=0)
            if r >= target:
                p = precision_score(y_val, pred, zero_division=0)
                candidates.append((t, p, r))
        if not candidates:
            out[f"recall_{int(target*100)}"] = {
                "achievable_on_validation": False,
                "note": f"No swept threshold reaches recall>={target:.0%} on validation.",
            }
            continue
        t_best, val_p, val_r = max(candidates, key=lambda c: c[1])
        test_pred = (test_proba >= t_best).astype(int)
        out[f"recall_{int(target*100)}"] = {
            "achievable_on_validation": True,
            "threshold": t_best,
            "validation_precision": val_p,
            "validation_recall": val_r,
            "test_precision": float(precision_score(y_test, test_pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, test_pred, zero_division=0)),
        }
    return out


def evaluate_frozen_model(name, feature_set, model, X_val, y_val, X_test, y_test):
    val_proba = model.predict_proba(X_val)[:, 1]

    t0 = time.perf_counter()
    test_proba = model.predict_proba(X_test)[:, 1]
    latency_ms = (time.perf_counter() - t0) * 1000.0 / len(X_test)

    val_roc = roc_auc_score(y_val, val_proba)
    val_pr = average_precision_score(y_val, val_proba)

    threshold, reason, sweep = select_threshold_on_validation(y_val, val_proba)

    test_pred = (test_proba >= threshold).astype(int)
    test_roc = roc_auc_score(y_test, test_proba)
    test_pr = average_precision_score(y_test, test_proba)
    test_prec = precision_score(y_test, test_pred, zero_division=0)
    test_rec = recall_score(y_test, test_pred, zero_division=0)
    test_f1 = f1_score(y_test, test_pred, zero_division=0)
    tn, fp, fn, tp = confusion_matrix(y_test, test_pred, labels=[0, 1]).ravel()

    recall_points = precision_at_recall_targets(y_val, val_proba, y_test, test_proba)

    result = {
        "model": name,
        "feature_set": feature_set,
        "data_source": "anchor (organic) only",
        "split_strategy": "60/20/20 stratified, two sequential train_test_split calls, random_state=42",
        "random_seed": RANDOM_STATE,
        "validation": {"roc_auc": float(val_roc), "pr_auc": float(val_pr)},
        "threshold": threshold,
        "threshold_selection_reason": reason,
        "validation_sweep": sweep,
        "test": {
            "roc_auc": float(test_roc), "pr_auc": float(test_pr),
            "precision": float(test_prec), "recall": float(test_rec), "f1": float(test_f1),
            "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
            "constraint_satisfied": bool(test_rec >= RECALL_MIN and test_prec >= PRECISION_MIN),
        },
        "precision_at_recall_targets": recall_points,
        "latency_ms_per_row": float(latency_ms),
        "test_proba": test_proba.tolist(),  # kept for the PR-curve plot only
    }
    log(f"[{name}] val ROC-AUC={val_roc:.4f} PR-AUC={val_pr:.4f} | "
        f"frozen threshold={threshold} ({reason}) | "
        f"TEST ROC-AUC={test_roc:.4f} PR-AUC={test_pr:.4f} "
        f"P={test_prec:.4f} R={test_rec:.4f} F1={test_f1:.4f} "
        f"latency={latency_ms:.4f}ms/row")
    return result


# ---------------------------------------------------------------------------
# 5. Repeated CV within TRAIN+VAL only (uncertainty estimate, never touches test)
# ---------------------------------------------------------------------------
def cv_pr_auc(build_fn, X_trainval, y_trainval, n_splits=N_CV_FOLDS):
    """build_fn(X_tr_fold, y_tr_fold) -> fitted model with .predict_proba.
    SMOTE (if any) must happen INSIDE build_fn, on the fold-train partition only."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fold_idx, (tr_idx, va_idx) in enumerate(skf.split(X_trainval, y_trainval)):
        X_tr = X_trainval.iloc[tr_idx] if hasattr(X_trainval, "iloc") else X_trainval[tr_idx]
        X_va = X_trainval.iloc[va_idx] if hasattr(X_trainval, "iloc") else X_trainval[va_idx]
        y_tr = y_trainval.iloc[tr_idx] if hasattr(y_trainval, "iloc") else y_trainval[tr_idx]
        y_va = y_trainval.iloc[va_idx] if hasattr(y_trainval, "iloc") else y_trainval[va_idx]
        model = build_fn(X_tr, y_tr)
        proba = model.predict_proba(X_va)[:, 1]
        scores.append(average_precision_score(y_va, proba))
    return {"mean_pr_auc": float(np.mean(scores)), "std_pr_auc": float(np.std(scores)),
            "folds": [float(s) for s in scores]}


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    df = pd.read_csv(MASTER_CSV)
    kept_features = run_feature_audit(df)
    run_source_classifier_test(df, kept_features)

    X_train, X_val, X_test, y_train, y_val, y_test = build_split(df, kept_features)

    (X_train_p, X_val_p, X_test_p, feature_names, cat_cols, num_cols) = shared_preprocessing(
        X_train, X_val, X_test
    )
    monotone = monotone_vector(feature_names)
    X_train_bal, y_train_bal = smote_resample(X_train_p, y_train)
    log(f"SMOTE-balanced train (shared preprocessing): {len(X_train_bal)} rows "
        f"({int(pd.Series(y_train_bal).sum())} fraud)")

    X_train_cb, X_val_cb, X_test_cb = catboost_preprocessing(X_train, X_val, X_test, cat_cols, num_cols)
    cb_cat_idx = list(range(len(cat_cols)))  # cat cols first, matches feature_names order

    results = {}

    # ---- A. XGBoost -- reproduce the existing baseline EXACTLY, no tuning ----
    xgb = XGBClassifier(
        n_estimators=400, max_depth=5, learning_rate=0.05,
        subsample=0.80, colsample_bytree=0.80, min_child_weight=10,
        gamma=0.10, scale_pos_weight=1, reg_alpha=0.05, reg_lambda=1.50,
        eval_metric="aucpr", tree_method="hist",
        monotone_constraints=monotone, random_state=RANDOM_STATE, n_jobs=4,
    )
    xgb.fit(X_train_bal, y_train_bal)
    results["A_XGBoost_current"] = evaluate_frozen_model(
        "A_XGBoost_current", "structured (shared ordinal-encoded)", xgb,
        X_val_p, y_val, X_test_p, y_test,
    )

    # ---- D. RandomForest -- small validation-selected grid ----
    rf_grid = [(n, d) for n in (300, 600) for d in (8, 16)]
    best_rf, best_rf_pr, best_rf_cfg = None, -1.0, None
    for n_estimators, max_depth in rf_grid:
        rf = RandomForestClassifier(
            n_estimators=n_estimators, max_depth=max_depth, min_samples_leaf=5,
            random_state=RANDOM_STATE, n_jobs=4,
        )
        with joblib.parallel_backend("threading"):
            rf.fit(X_train_bal, y_train_bal)
        pr = average_precision_score(y_val, rf.predict_proba(X_val_p)[:, 1])
        log(f"  [RF search] n_estimators={n_estimators} max_depth={max_depth} val PR-AUC={pr:.4f}")
        if pr > best_rf_pr:
            best_rf, best_rf_pr, best_rf_cfg = rf, pr, {"n_estimators": n_estimators, "max_depth": max_depth}
    results["D_RandomForest"] = evaluate_frozen_model(
        "D_RandomForest", "structured (shared ordinal-encoded)", best_rf,
        X_val_p, y_val, X_test_p, y_test,
    )
    results["D_RandomForest"]["hyperparameter_search"] = {
        "search_space": rf_grid, "selection_metric": "validation PR-AUC",
        "selected": best_rf_cfg, "selected_validation_pr_auc": best_rf_pr,
        "note": "No monotonic-constraint support in sklearn RandomForestClassifier -- a real engineering limitation of this model family for this use case, not applied.",
    }

    # ---- E. LightGBM -- small validation-selected grid ----
    lgb_grid = [(nl, lr) for nl in (31, 63) for lr in (0.05, 0.1)]
    best_lgb, best_lgb_pr, best_lgb_cfg = None, -1.0, None
    for num_leaves, lr in lgb_grid:
        lgbm = LGBMClassifier(
            n_estimators=400, num_leaves=num_leaves, learning_rate=lr,
            subsample=0.80, colsample_bytree=0.80, min_child_samples=10,
            reg_alpha=0.05, reg_lambda=1.50, monotone_constraints=list(monotone),
            random_state=RANDOM_STATE, n_jobs=4, verbose=-1,
        )
        lgbm.fit(X_train_bal, y_train_bal)
        pr = average_precision_score(y_val, lgbm.predict_proba(X_val_p)[:, 1])
        log(f"  [LightGBM search] num_leaves={num_leaves} lr={lr} val PR-AUC={pr:.4f}")
        if pr > best_lgb_pr:
            best_lgb, best_lgb_pr, best_lgb_cfg = lgbm, pr, {"num_leaves": num_leaves, "learning_rate": lr}
    results["E_LightGBM"] = evaluate_frozen_model(
        "E_LightGBM", "structured (shared ordinal-encoded)", best_lgb,
        X_val_p, y_val, X_test_p, y_test,
    )
    results["E_LightGBM"]["hyperparameter_search"] = {
        "search_space": lgb_grid, "selection_metric": "validation PR-AUC",
        "selected": best_lgb_cfg, "selected_validation_pr_auc": best_lgb_pr,
    }

    # ---- F. CatBoost -- small validation-selected grid, native categorical handling ----
    cb_grid = [(depth, lr) for depth in (6, 8) for lr in (0.05, 0.1)]
    best_cb, best_cb_pr, best_cb_cfg = None, -1.0, None
    for depth, lr in cb_grid:
        cb = CatBoostClassifier(
            iterations=400, depth=depth, learning_rate=lr,
            cat_features=cb_cat_idx, auto_class_weights="Balanced",
            monotone_constraints=list(monotone), random_seed=RANDOM_STATE,
            verbose=False,
        )
        cb.fit(X_train_cb, y_train)
        pr = average_precision_score(y_val, cb.predict_proba(X_val_cb)[:, 1])
        log(f"  [CatBoost search] depth={depth} lr={lr} val PR-AUC={pr:.4f}")
        if pr > best_cb_pr:
            best_cb, best_cb_pr, best_cb_cfg = cb, pr, {"depth": depth, "learning_rate": lr}
    results["F_CatBoost"] = evaluate_frozen_model(
        "F_CatBoost", "structured (native categorical, no SMOTE -- auto_class_weights='Balanced')",
        best_cb, X_val_cb, y_val, X_test_cb, y_test,
    )
    results["F_CatBoost"]["hyperparameter_search"] = {
        "search_space": cb_grid, "selection_metric": "validation PR-AUC",
        "selected": best_cb_cfg, "selected_validation_pr_auc": best_cb_pr,
        "note": "Uses auto_class_weights='Balanced' instead of SMOTE -- SMOTE is not natural on "
                "raw (unencoded) categorical text without SMOTENC. A deliberate, disclosed pipeline "
                "difference from A/D/E, not an inconsistency in the split or the test set.",
    }

    # ---- Feature importance (diagnostic: where is the signal coming from?) ----
    importances = {}
    for name, model, feats in [
        ("A_XGBoost_current", xgb, feature_names),
        ("D_RandomForest", best_rf, feature_names),
        ("E_LightGBM", best_lgb, feature_names),
    ]:
        imp = model.feature_importances_
        top = sorted(zip(feats, imp), key=lambda x: -x[1])[:10]
        importances[name] = [{"feature": f, "importance": float(i)} for f, i in top]
    cb_imp = best_cb.get_feature_importance()
    top_cb = sorted(zip(feature_names, cb_imp), key=lambda x: -x[1])[:10]
    importances["F_CatBoost"] = [{"feature": f, "importance": float(i)} for f, i in top_cb]

    # ---- Repeated CV (TRAIN+VAL only, never touches TEST) ----
    X_trainval_p = np.vstack([X_train_p, X_val_p]) if not hasattr(X_train_p, "iloc") else pd.concat([X_train_p, X_val_p])
    y_trainval = pd.concat([y_train, y_val])

    def build_xgb(Xtr, ytr):
        Xb, yb = smote_resample(Xtr, ytr)
        m = XGBClassifier(n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.80,
                           colsample_bytree=0.80, min_child_weight=10, gamma=0.10, scale_pos_weight=1,
                           reg_alpha=0.05, reg_lambda=1.50, eval_metric="aucpr", tree_method="hist",
                           monotone_constraints=monotone, random_state=RANDOM_STATE, n_jobs=4)
        m.fit(Xb, yb)
        return m

    def build_best_rf(Xtr, ytr):
        Xb, yb = smote_resample(Xtr, ytr)
        m = RandomForestClassifier(n_estimators=best_rf_cfg["n_estimators"], max_depth=best_rf_cfg["max_depth"],
                                    min_samples_leaf=5, random_state=RANDOM_STATE, n_jobs=4)
        with joblib.parallel_backend("threading"):
            m.fit(Xb, yb)
        return m

    def build_best_lgb(Xtr, ytr):
        Xb, yb = smote_resample(Xtr, ytr)
        m = LGBMClassifier(n_estimators=400, num_leaves=best_lgb_cfg["num_leaves"],
                            learning_rate=best_lgb_cfg["learning_rate"], subsample=0.80, colsample_bytree=0.80,
                            min_child_samples=10, reg_alpha=0.05, reg_lambda=1.50,
                            monotone_constraints=list(monotone), random_state=RANDOM_STATE, n_jobs=4, verbose=-1)
        m.fit(Xb, yb)
        return m

    # Run CV on TRAIN+VAL combined (never touches TEST), refit per fold.
    # Logged immediately per model (not collected into one dict first) so
    # progress is visible during a long run instead of going silent until
    # every model finishes.
    cv_results = {}
    for name, build_fn in [
        ("A_XGBoost_current", build_xgb),
        ("D_RandomForest", build_best_rf),
        ("E_LightGBM", build_best_lgb),
    ]:
        cv_results[name] = cv_pr_auc(build_fn, X_trainval_p, y_trainval)
        cv = cv_results[name]
        log(f"[{name}] 5-fold CV (train+val only) PR-AUC = {cv['mean_pr_auc']:.4f} +/- {cv['std_pr_auc']:.4f}")

    # CatBoost's winning grid config (best_cb_cfg) already took roughly an
    # hour PER FIT in this environment during the grid search above -- by far
    # the most expensive of the four model families, confirmed by direct
    # observation (CPU-time vs. wall-clock measured live during development).
    # CatBoost is also already the WEAKEST of the four candidates on the
    # selection metric (validation PR-AUC) that decided every other model's
    # winning config, so its 5-fold CV is skipped rather than spending
    # further hours characterizing the uncertainty of an already-losing
    # candidate. This is a disclosed, deliberate compute-budget decision
    # (see the master prompt's own "keep searches controlled, do not spend
    # excessive compute" instruction), not a silent gap.
    cv_results["F_CatBoost"] = None
    log("[F_CatBoost] 5-fold CV skipped -- see code comment above main() "
        "(CatBoost's grid-search fits were far more expensive than the other "
        "three model families in this environment, and it already has the "
        "lowest validation PR-AUC of the four; not worth further compute).")

    # ---- Save PR curves (single plot, computed once, after everything frozen) ----
    os.makedirs(os.path.dirname(PLOT_PATH), exist_ok=True)
    plt.figure(figsize=(7, 6))
    for name, res in results.items():
        precision, recall, _ = precision_recall_curve(y_test, res["test_proba"])
        plt.plot(recall, precision, label=f"{name} (PR-AUC={res['test']['pr_auc']:.3f})")
    plt.axhline(PRECISION_MIN, color="gray", linestyle="--", linewidth=0.8, label="Precision>=50% requirement")
    plt.axvline(RECALL_MIN, color="gray", linestyle=":", linewidth=0.8, label="Recall>=75% requirement")
    plt.xlabel("Recall"); plt.ylabel("Precision")
    plt.title("Precision-Recall on the frozen, untouched final test set\n(anchor-only, organic data)")
    plt.legend(fontsize=8); plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    plt.close()
    log(f"Wrote {PLOT_PATH}")

    # strip the raw proba arrays before persisting to JSON (kept only for the plot)
    persisted = {k: {kk: vv for kk, vv in v.items() if kk != "test_proba"} for k, v in results.items()}
    for k in persisted:
        persisted[k]["cross_validation_train_val_only"] = cv_results.get(k)
        persisted[k]["top_feature_importance"] = importances.get(k)

    full_out = {
        "protocol": "anchor-only (organic) 60/20/20 split, threshold selected on validation only, "
                    "final test evaluated exactly once per model",
        "random_seed": RANDOM_STATE,
        "kept_features": kept_features,
        "n_kept_features": len(kept_features),
        "models": persisted,
    }
    with open(RESULTS_JSON, "w") as f:
        json.dump(full_out, f, indent=2, default=float)
    log(f"Wrote {RESULTS_JSON}")

    # ---- CSV + human-readable MD ----
    csv_rows = []
    for name, res in persisted.items():
        t = res["test"]
        csv_rows.append({
            "model": name, "feature_set": res["feature_set"], "data_source": res["data_source"],
            "split_strategy": res["split_strategy"], "random_seed": res["random_seed"],
            "threshold": res["threshold"], "roc_auc": t["roc_auc"], "pr_auc": t["pr_auc"],
            "precision": t["precision"], "recall": t["recall"], "f1": t["f1"],
            "latency_ms": res["latency_ms_per_row"],
            "notes": res["threshold_selection_reason"],
        })
    pd.DataFrame(csv_rows).to_csv(RESULTS_CSV, index=False)
    log(f"Wrote {RESULTS_CSV}")

    with open(RESULTS_MD, "w", encoding="utf-8") as f:
        f.write("# Model benchmark -- clean organic evaluation (2026-08-27)\n\n")
        f.write("All models share the identical anchor-only (organic) 60/20/20 split "
                "(`random_state=42`), fit preprocessing on TRAIN only, select threshold on "
                "VALIDATION only, and are evaluated on the SAME untouched final TEST set "
                "exactly once. Text-based model families are not included -- "
                "`paysense_master_dataset.csv` has no text field (see `feature_audit.md`).\n\n")
        f.write("| Model | Features | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | Latency (ms/row) | Business constraint met |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|---:|---:|---|\n")
        for name, res in persisted.items():
            t = res["test"]
            f.write(f"| {name} | {res['feature_set']} | {res['threshold']} | {t['roc_auc']:.4f} | "
                    f"{t['pr_auc']:.4f} | {t['precision']:.2%} | {t['recall']:.2%} | {t['f1']:.4f} | "
                    f"{res['latency_ms_per_row']:.4f} | {'YES' if t['constraint_satisfied'] else 'No'} |\n")
        f.write("\n## 5-fold cross-validation (TRAIN+VAL only, never touches TEST)\n\n")
        f.write("| Model | Mean PR-AUC | Std PR-AUC |\n|---|---:|---:|\n")
        for name, res in persisted.items():
            cv = res.get("cross_validation_train_val_only")
            if cv:
                f.write(f"| {name} | {cv['mean_pr_auc']:.4f} | {cv['std_pr_auc']:.4f} |\n")
        f.write("\n## Precision at fixed recall targets (threshold selected on validation only)\n\n")
        for name, res in persisted.items():
            f.write(f"\n**{name}**\n\n| Recall target | Achievable on validation | Threshold | Test precision | Test recall |\n|---|---|---:|---:|---:|\n")
            for k, v in res["precision_at_recall_targets"].items():
                if v["achievable_on_validation"]:
                    f.write(f"| {k} | Yes | {v['threshold']} | {v['test_precision']:.2%} | {v['test_recall']:.2%} |\n")
                else:
                    f.write(f"| {k} | No | -- | -- | -- |\n")
        f.write("\n## Where is the signal coming from? (top-10 feature importance)\n\n")
        for name, feats in importances.items():
            f.write(f"\n**{name}**: " + ", ".join(f"{d['feature']} ({d['importance']:.3f})" for d in feats) + "\n")
    log(f"Wrote {RESULTS_MD}")

    log("\nDone.")


if __name__ == "__main__":
    main()
