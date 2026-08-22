"""
================================================================================
  train_category_classifier.py — PaySense Layer 2 NLP Category Classifier
  ────────────────────────────────────────────────────────────────────────────
  Trains the classifier that backs the new POST /classify endpoint, which
  implements Tier 2 ("NLP Classifier") of the Android app's three-tier
  payee-category resolution pipeline (see
  PaySense-Android-Client-New/.../layer2/PayeeCacheRepository.kt).

  Data
  ────
  Source: FinText-6K (Kaggle, Apache 2.0 licensed, synthetic-but-realistic
  UPI/bank-SMS transaction narrations). Two files, already split by the
  dataset author:
      E:\\Projects\\upi\\FinText-6K\\train_transaction_dataset.csv  (5,000 rows)
      E:\\Projects\\upi\\FinText-6K\\test_transaction_dataset.csv   (1,000 rows)
  Columns: text, label. We train ONLY on the train split and evaluate ONLY
  on the untouched test split — no leakage, no re-shuffling across the two
  files.

  Labels (verified from the raw CSVs, case exactly as shipped):
      food, travel, EMI, investment, shopping
  Note the inconsistent casing (lowercase for four classes, upper for EMI) —
  this is preserved from the source data and is NOT a bug in this script.
  Model output is normalised to Title Case for the API response (Food,
  Travel, EMI, Investment, Shopping) — EMI is kept fully upper-cased since
  it's an acronym.

  Model
  ─────
  TF-IDF (word 1-2 grams, sublinear tf, min_df=2) + LinearSVC, wrapped in
  CalibratedClassifierCV (sigmoid/Platt scaling, 5-fold) so we get real
  predict_proba() confidences — LinearSVC alone only gives decision_function
  distances, and the Android client's NLP_CONFIDENCE_THRESHOLD = 0.65f gate
  needs an actual probability, not a raw margin. This is the right amount of
  model complexity for ~1000-1500 short, templated narration strings per
  class; a transformer would be overkill and cost real latency/memory on a
  free-tier Render deployment.

  Output
  ──────
  artefacts/paysense_category_classifier.pkl — a single sklearn Pipeline
  (TfidfVectorizer -> CalibratedClassifierCV(LinearSVC)) that exposes both
  .predict() and .predict_proba(), loadable with joblib.load() exactly like
  the existing fraud-model artefacts in this same folder.

  Run
  ───
      cd PaySense-ML-Backend
      venv\\Scripts\\python.exe train_category_classifier.py
================================================================================
"""

from __future__ import annotations

import json
import os
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

# ── Paths ──────────────────────────────────────────────────────────────────
_HERE        = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR  = os.environ.get("FINTEXT_DIR", r"E:\Projects\upi\FinText-6K")
TRAIN_CSV    = os.path.join(DATASET_DIR, "train_transaction_dataset.csv")
TEST_CSV     = os.path.join(DATASET_DIR, "test_transaction_dataset.csv")
ARTEFACT_DIR = os.path.join(_HERE, "artefacts")
ARTEFACT_PATH= os.path.join(ARTEFACT_DIR, "paysense_category_classifier.pkl")
METRICS_PATH = os.path.join(ARTEFACT_DIR, "paysense_category_classifier_metrics.json")

# Raw dataset label -> API-facing category label. Title-cased to match the
# app's existing category vocabulary conventions (Food, Travel, Shopping),
# EMI kept as an acronym, Investment title-cased.
LABEL_DISPLAY_MAP = {
    "food":       "Food",
    "travel":     "Travel",
    "EMI":        "EMI",
    "investment": "Investment",
    "shopping":   "Shopping",
}


def load_split(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    assert set(df.columns) >= {"text", "label"}, f"Unexpected columns in {path}: {df.columns.tolist()}"
    unknown = set(df["label"].unique()) - set(LABEL_DISPLAY_MAP.keys())
    if unknown:
        raise ValueError(f"{path} contains unexpected label(s): {unknown}")
    df["display_label"] = df["label"].map(LABEL_DISPLAY_MAP)
    return df


def build_pipeline() -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        ngram_range=(1, 2),
        min_df=2,
        sublinear_tf=True,
        strip_accents="unicode",
    )
    # LinearSVC is a strong, fast linear classifier for short bag-of-words
    # text; CalibratedClassifierCV wraps it so predict_proba() gives
    # genuine probability estimates (needed for the app's confidence gate).
    base_clf = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)
    return Pipeline([
        ("tfidf", vectorizer),
        ("clf", calibrated),
    ])


def main() -> None:
    if not os.path.exists(TRAIN_CSV) or not os.path.exists(TEST_CSV):
        print(f"ERROR: dataset not found. Expected:\n  {TRAIN_CSV}\n  {TEST_CSV}")
        sys.exit(1)

    print(f"Loading train split: {TRAIN_CSV}")
    train_df = load_split(TRAIN_CSV)
    print(f"Loading test split (held out, evaluation only): {TEST_CSV}")
    test_df = load_split(TEST_CSV)

    print(f"\nTrain rows: {len(train_df)}  |  class counts:\n{train_df['display_label'].value_counts()}")
    print(f"\nTest rows:  {len(test_df)}  |  class counts:\n{test_df['display_label'].value_counts()}")

    pipeline = build_pipeline()

    t0 = time.time()
    pipeline.fit(train_df["text"], train_df["display_label"])
    train_secs = time.time() - t0
    print(f"\nTrained in {train_secs:.2f}s")

    # ── Honest evaluation on the untouched 1,000-row test split ────────────
    y_true = test_df["display_label"].to_numpy()
    y_pred = pipeline.predict(test_df["text"])
    y_proba = pipeline.predict_proba(test_df["text"])
    classes = pipeline.named_steps["clf"].classes_

    accuracy   = accuracy_score(y_true, y_pred)
    macro_f1   = f1_score(y_true, y_pred, average="macro")
    weighted_f1= f1_score(y_true, y_pred, average="weighted")
    report_str = classification_report(y_true, y_pred, digits=4)
    report_dict= classification_report(y_true, y_pred, digits=4, output_dict=True)
    cm         = confusion_matrix(y_true, y_pred, labels=list(classes))

    # Mean confidence of the top predicted class (i.e. what the app's
    # NLP_CONFIDENCE_THRESHOLD = 0.65 gate will actually see).
    top_conf = y_proba.max(axis=1)
    mean_conf = float(top_conf.mean())
    frac_above_threshold = float((top_conf >= 0.65).mean())
    accuracy_above_threshold = float(
        accuracy_score(y_true[top_conf >= 0.65], y_pred[top_conf >= 0.65])
    ) if (top_conf >= 0.65).any() else None

    print("\n" + "=" * 78)
    print("HELD-OUT TEST SET METRICS (1,000 rows, never used in training)")
    print("=" * 78)
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}")
    print(f"\nClassification report:\n{report_str}")
    print(f"Confusion matrix (rows=true, cols=pred), class order {list(classes)}:")
    print(cm)
    print(f"\nMean top-class confidence         : {mean_conf:.4f}")
    print(f"Fraction of test rows >= 0.65 conf : {frac_above_threshold:.4f}")
    print(f"Accuracy among those >= 0.65 conf  : {accuracy_above_threshold}")

    # ── Save artefact ────────────────────────────────────────────────────
    os.makedirs(ARTEFACT_DIR, exist_ok=True)
    joblib.dump(pipeline, ARTEFACT_PATH)
    print(f"\nSaved classifier pipeline -> {ARTEFACT_PATH}")

    metrics = {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(classes),
        "mean_top_confidence": mean_conf,
        "fraction_above_0.65_confidence": frac_above_threshold,
        "accuracy_among_above_0.65_confidence": accuracy_above_threshold,
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "classes": list(classes),
        "trained_at_unix": time.time(),
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON        -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
