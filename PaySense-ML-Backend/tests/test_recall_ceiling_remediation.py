"""
================================================================================
  tests/test_recall_ceiling_remediation.py
  ────────────────────────────────────────────────────────────────────────────
  On 2026-08-23, RECALL_CEILING_REMEDIATION.md tested whether the frozen
  model's 76-of-253 invisible-fraud pattern at tau=0.05 is caused by XGBoost's
  trees gating hard on new_device_flag / ip_location_mismatch before letting
  the behavioral features (amount_deviation_score, transaction_velocity,
  failed_attempts_last_24h) matter. Retraining with interaction_constraints
  recovered 31 of the 76 rows (at a real cost to overall ROC-AUC and
  precision); monotone_constraints recovered 10 while slightly *improving*
  ROC-AUC/PR-AUC; an engineered composite feature recovered the fewest (8)
  and made PR-AUC slightly worse. None of the three candidate models were
  wired into production — this is exactly the kind of finding that needs a
  regression guard so it isn't silently invalidated by a future retrain of
  the frozen model or a future edit to the candidate artifacts.

  UPDATE (2026-08-23): the monotonic-constraints candidate was subsequently
  adopted as the new deployed model (see paysense_phase3.py and README.md's
  Key Results note) — artefacts/paysense_model.pkl is now the monotonic
  variant, not the original unconstrained baseline this experiment
  diagnosed. This test file therefore pins against
  paysense_model_vanilla_replica.pkl (the frozen, unchanging copy of the
  *original* baseline the experiment saved on disk) rather than the live
  paysense_model.pkl, so the historical diagnostic stays a fixed reference
  point independent of whatever gets deployed later. Anyone re-reading this
  file cold: "the frozen model" below means that historical baseline, not
  today's deployed artifact.

  This test guards two things:
    1. The diagnostic itself — the original baseline's 76/177 split at
       tau=0.05 and the feature-dominance pattern (invisible rows score
       LOWER on the two hard-signal flags and HIGHER on the three
       behavioral features than caught rows) — against silent drift in
       that historical artifact or the master dataset.
    2. The three saved candidate artifacts (if present) still reproduce the
       documented ROC-AUC and recovered-of-76 numbers within tolerance — a
       drift guard for the experiment's own outputs, mirroring how
       test_platt_scaling_invariance.py guards platt_scaling_experiment.py.

  NO RETRAINING happens in this test file. All artifacts (the historical
  baseline and the three candidates) are loaded read-only.
================================================================================
"""

import pathlib

import joblib
import numpy as np
import pandas as pd
import pytest
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
TAU = 0.05

HARD_SIGNAL_FEATURES = ["new_device_flag", "ip_location_mismatch"]
BEHAVIORAL_FEATURES = [
    "amount_deviation_score", "transaction_velocity", "failed_attempts_last_24h",
]

# Values reported in RECALL_CEILING_REMEDIATION.md, computed against the
# frozen artifacts as they exist today.
PUBLISHED_CAUGHT_AT_005 = 177
PUBLISHED_MISSED_AT_005 = 76
PUBLISHED_RECALL_AT_005 = 0.699605

CANDIDATE_MODELS = {
    "paysense_model_interaction_constrained.pkl": {
        "roc_auc": 0.8814, "recovered_of_76": 31,
    },
    "paysense_model_monotonic.pkl": {
        "roc_auc": 0.8889, "recovered_of_76": 10,
    },
    "paysense_model_composite_feature.pkl": {
        "roc_auc": 0.8868, "recovered_of_76": 8, "extra_column": True,
    },
}


BASELINE_MODEL_PATH = ARTEFACTS_DIR / "paysense_model_vanilla_replica.pkl"


def _skip_if_frozen_artefacts_missing():
    required = [
        BASELINE_MODEL_PATH,
        ARTEFACTS_DIR / "paysense_preprocessor.pkl",
        MASTER_CSV,
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        pytest.skip(f"Historical baseline artefacts not present in this environment: {missing}")


@pytest.fixture(scope="module")
def split_and_frozen_scores():
    _skip_if_frozen_artefacts_missing()

    df = pd.read_csv(MASTER_CSV)
    df = df.drop(columns=DROP_COLS)
    X = df.drop(columns=["is_fraud"])
    y = df["is_fraud"].astype(int)

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
    )

    # The preprocessor is unaffected by which XGBoost variant is deployed
    # (it was fit once, before any of the candidate models existed), so the
    # live artifact is still the right one to use here.
    preprocessor = joblib.load(ARTEFACTS_DIR / "paysense_preprocessor.pkl")
    model = joblib.load(BASELINE_MODEL_PATH)

    X_test_proc = preprocessor.transform(X_test_raw)
    y_proba = model.predict_proba(X_test_proc)[:, 1]
    y_pred_tau = (y_proba >= TAU).astype(int)

    fraud_mask = (y_test.values == 1)
    caught_mask = fraud_mask & (y_pred_tau == 1)
    missed_mask = fraud_mask & (y_pred_tau == 0)

    return {
        "X_test_raw": X_test_raw,
        "X_test_proc": X_test_proc,
        "y_test": y_test,
        "y_proba": y_proba,
        "caught_mask": caught_mask,
        "missed_mask": missed_mask,
    }


def test_diagnostic_split_matches_documented(split_and_frozen_scores):
    n_caught = int(split_and_frozen_scores["caught_mask"].sum())
    n_missed = int(split_and_frozen_scores["missed_mask"].sum())
    assert n_caught == PUBLISHED_CAUGHT_AT_005, (
        f"Frozen model now catches {n_caught} fraud rows at tau=0.05, not the "
        f"documented {PUBLISHED_CAUGHT_AT_005} — RECALL_CEILING_REMEDIATION.md's "
        f"diagnostic has drifted from the current frozen artifacts / master "
        f"dataset and needs to be recomputed."
    )
    assert n_missed == PUBLISHED_MISSED_AT_005, (
        f"Frozen model now misses {n_missed} fraud rows at tau=0.05, not the "
        f"documented {PUBLISHED_MISSED_AT_005} — see note above."
    )

    recall = recall_score(
        split_and_frozen_scores["y_test"],
        (split_and_frozen_scores["y_proba"] >= TAU).astype(int),
    )
    assert recall == pytest.approx(PUBLISHED_RECALL_AT_005, abs=1e-4)


def test_feature_dominance_pattern_holds(split_and_frozen_scores):
    """The core diagnostic finding: invisible fraud rows score LOWER on the
    two hard-signal flags but HIGHER on the three behavioral features than
    the fraud rows the model does catch. This is the pattern that motivated
    testing interaction_constraints / monotone_constraints in the first
    place; if this pattern disappears, the remediation experiment's premise
    no longer applies to the current frozen model and needs to be redone."""
    X_test_raw = split_and_frozen_scores["X_test_raw"]
    missed_mask = split_and_frozen_scores["missed_mask"]
    caught_mask = split_and_frozen_scores["caught_mask"]

    for feat in HARD_SIGNAL_FEATURES:
        missed_mean = X_test_raw[feat].values[missed_mask].mean()
        caught_mean = X_test_raw[feat].values[caught_mask].mean()
        assert missed_mean < caught_mean, (
            f"Expected invisible fraud to score LOWER on hard-signal feature "
            f"'{feat}' than caught fraud (missed={missed_mean:.3f}, "
            f"caught={caught_mean:.3f}) — the dominance pattern this "
            f"remediation experiment was built on no longer holds."
        )

    for feat in BEHAVIORAL_FEATURES:
        missed_mean = X_test_raw[feat].values[missed_mask].mean()
        caught_mean = X_test_raw[feat].values[caught_mask].mean()
        assert missed_mean > caught_mean, (
            f"Expected invisible fraud to score HIGHER on behavioral feature "
            f"'{feat}' than caught fraud (missed={missed_mean:.3f}, "
            f"caught={caught_mean:.3f}) — the dominance pattern this "
            f"remediation experiment was built on no longer holds."
        )


@pytest.mark.parametrize("filename,expected", CANDIDATE_MODELS.items())
def test_candidate_artifact_reproduces_documented_metrics(
    split_and_frozen_scores, filename, expected
):
    """If a candidate remediation artifact exists on disk, it must still
    reproduce the ROC-AUC and recovered-of-76 numbers RECALL_CEILING_REMEDIATION.md
    reports — a drift guard, not a re-run of the (expensive) training step."""
    path = ARTEFACTS_DIR / filename
    if not path.exists():
        pytest.skip(f"{filename} not present in this environment — experiment "
                     f"artifact, not a required build output.")

    model = joblib.load(path)
    X_test_proc = split_and_frozen_scores["X_test_proc"]

    if expected.get("extra_column"):
        # composite_feature model expects one extra column: the min-max
        # normalized mean of the 3 behavioral features, fit on TRAIN stats.
        # Recomputing train-fit min/max here would require retraining data
        # access already loaded by the fixture's own split; reuse it.
        preprocessor = joblib.load(ARTEFACTS_DIR / "paysense_preprocessor.pkl")
        df = pd.read_csv(MASTER_CSV).drop(columns=DROP_COLS)
        X = df.drop(columns=["is_fraud"])
        y = df["is_fraud"].astype(int)
        X_train_raw, _, _, _ = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
        )
        X_train_proc = preprocessor.transform(X_train_raw)
        feature_names = (
            X_train_raw.select_dtypes(include=["object"]).columns.tolist()
            + X_train_raw.select_dtypes(include=[np.number]).columns.tolist()
        )
        idxs = [feature_names.index(f) for f in BEHAVIORAL_FEATURES]
        train_sub = X_train_proc[:, idxs]
        mins, maxs = train_sub.min(axis=0), train_sub.max(axis=0)
        ranges = np.where((maxs - mins) == 0, 1.0, maxs - mins)
        test_sub = X_test_proc[:, idxs]
        composite = ((test_sub - mins) / ranges).mean(axis=1, keepdims=True)
        X_test_proc = np.hstack([X_test_proc, composite])

    y_proba = model.predict_proba(X_test_proc)[:, 1]
    y_test = split_and_frozen_scores["y_test"]

    roc_auc = roc_auc_score(y_test, y_proba)
    assert roc_auc == pytest.approx(expected["roc_auc"], abs=0.01), (
        f"{filename}: recomputed ROC-AUC ({roc_auc:.4f}) has drifted from the "
        f"documented {expected['roc_auc']} in RECALL_CEILING_REMEDIATION.md."
    )

    missed_mask = split_and_frozen_scores["missed_mask"]
    y_pred_tau = (y_proba >= TAU).astype(int)
    recovered = int(y_pred_tau[missed_mask].sum())
    assert recovered == expected["recovered_of_76"], (
        f"{filename}: recomputed recovered-of-76 count ({recovered}) has "
        f"drifted from the documented {expected['recovered_of_76']} in "
        f"RECALL_CEILING_REMEDIATION.md."
    )
