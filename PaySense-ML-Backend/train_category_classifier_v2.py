"""
================================================================================
  train_category_classifier_v2.py
  ----------------------------------------------------------------------------
  Trains a CANDIDATE category classifier on a blend of FinText-6K's train
  split plus genuinely more diverse data (two real-Kaggle-downloaded but
  self-evidently synthetic datasets from different generators, plus a new,
  much larger, structurally-varied rule-based generator grounded in
  documented Indian bank-SMS / UPI-app narration conventions), to test
  whether more/richer training data actually closes the 27.5-point
  generalization gap CATEGORY_CLASSIFIER_GENERALIZATION.md diagnosed.

  Does NOT touch artefacts/paysense_category_classifier.pkl (the frozen,
  currently-deployed model) anywhere. Writes a new artefact:
      artefacts/paysense_category_classifier_v2.pkl
      artefacts/paysense_category_classifier_v2_metrics.json

  Same model architecture as the original (TF-IDF 1-2gram + Calibrated
  LinearSVC) -- deliberately unchanged, so any accuracy difference measures
  the effect of the DATA change, not a confounded architecture change, the
  same isolation discipline OOD_GENERALIZATION_REMEDIATION.md used when
  testing the fraud model's blended-training hypothesis.

  Training data (blended, all four sources):
    1. FinText-6K train_transaction_dataset.csv  (5,000 rows) -- unchanged
       from the original CATEGORY_CLASSIFIER.md training data.
    2. external_data/kaggle_coderanand_indian_banking/*.csv (11,000 rows,
       both splits usable since this is not FinText's own held-out set) --
       real Kaggle download, self-evidently synthetic (see
       external_data/NOTICE.md), different generator/vocabulary.
    3. external_data/kaggle_bhavya_financial_transaction/*.csv, filtered to
       the 5 classes that exist in PaySense's taxonomy (~3,300 rows) --
       same treatment.
    4. category_training_v2_synthetic.csv (~23,700 rows) -- this project's
       own new, 40-skeleton, large-merchant-vocabulary generator (see
       generate_category_training_v2.py).

  Leakage discipline (checked and enforced, not assumed):
    - No training row (from ANY of the 4 sources) may exactly match a row in
      category_generalization_test_set.csv (the frozen gold evaluation set)
      or FinText-6K's OWN test_transaction_dataset.csv (used here purely to
      re-check in-distribution accuracy hasn't regressed). Any collision is
      dropped from training, counted, and reported.

  Evaluation (both held out, neither ever touched during fit()):
    - FinText-6K's own test_transaction_dataset.csv (1,000 rows) -- same
      check CATEGORY_CLASSIFIER.md ran on the original model, to catch
      in-distribution regression.
    - category_generalization_test_set.csv (200 rows) -- the gold,
      structurally-disjoint generalization check. This is the number that
      actually matters for this exercise.

  Run:
      cd PaySense-ML-Backend
      venv\\Scripts\\python.exe train_category_classifier_v2.py
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
from sklearn.utils import shuffle

_HERE = os.path.dirname(os.path.abspath(__file__))
FINTEXT_DIR = os.environ.get("FINTEXT_DIR", r"E:\Projects\upi\FinText-6K")
FINTEXT_TRAIN_CSV = os.path.join(FINTEXT_DIR, "train_transaction_dataset.csv")
FINTEXT_TEST_CSV = os.path.join(FINTEXT_DIR, "test_transaction_dataset.csv")

CODERANAND_DIR = os.path.join(_HERE, "external_data", "kaggle_coderanand_indian_banking")
BHAVYA_DIR = os.path.join(_HERE, "external_data", "kaggle_bhavya_financial_transaction")
V2_SYNTHETIC_CSV = os.path.join(_HERE, "category_training_v2_synthetic.csv")
GOLD_EVAL_CSV = os.path.join(_HERE, "category_generalization_test_set.csv")

ARTEFACT_DIR = os.path.join(_HERE, "artefacts")
ARTEFACT_PATH = os.path.join(ARTEFACT_DIR, "paysense_category_classifier_v2.pkl")
METRICS_PATH = os.path.join(ARTEFACT_DIR, "paysense_category_classifier_v2_metrics.json")

FINTEXT_LABEL_MAP = {"food": "Food", "travel": "Travel", "EMI": "EMI",
                      "investment": "Investment", "shopping": "Shopping"}
BHAVYA_LABEL_MAP = {"food": "Food", "travel": "Travel", "emi": "EMI",
                     "investment": "Investment", "shopping": "Shopping"}
# entertainment/healthcare/education/utilities deliberately excluded -- see
# external_data/NOTICE.md.

CONFIDENCE_THRESHOLD = 0.65
SEED = 20260823


def load_fintext_train() -> pd.DataFrame:
    df = pd.read_csv(FINTEXT_TRAIN_CSV)
    df["display_label"] = df["label"].map(FINTEXT_LABEL_MAP)
    assert df["display_label"].notna().all()
    return df[["text", "display_label"]].assign(source="fintext6k_train")


def load_coderanand() -> pd.DataFrame:
    train = pd.read_csv(os.path.join(CODERANAND_DIR, "financial_transaction_train.csv"))
    test = pd.read_csv(os.path.join(CODERANAND_DIR, "financial_transaction_test.csv"))
    df = pd.concat([train, test], ignore_index=True)
    df = df.rename(columns={"Transaction_Text": "text", "Label": "display_label"})
    assert set(df["display_label"].unique()) <= set(FINTEXT_LABEL_MAP.values())
    return df[["text", "display_label"]].assign(source="kaggle_coderanand")


def load_bhavya() -> pd.DataFrame:
    train = pd.read_csv(os.path.join(BHAVYA_DIR, "train_transactions.csv"))
    test = pd.read_csv(os.path.join(BHAVYA_DIR, "test_transactions.csv"))
    df = pd.concat([train, test], ignore_index=True)
    df = df.rename(columns={"transaction_text": "text", "category": "raw_label"})
    df["display_label"] = df["raw_label"].map(BHAVYA_LABEL_MAP)
    dropped = df["display_label"].isna().sum()
    df = df[df["display_label"].notna()].copy()
    print(f"  bhavya: dropped {dropped} rows outside the 5-class taxonomy "
          f"(entertainment/healthcare/education/utilities)")
    return df[["text", "display_label"]].assign(source="kaggle_bhavya")


def load_v2_synthetic() -> pd.DataFrame:
    df = pd.read_csv(V2_SYNTHETIC_CSV)
    df["display_label"] = df["label"].map(FINTEXT_LABEL_MAP)
    assert df["display_label"].notna().all()
    return df[["text", "display_label"]].assign(source="paysense_v2_synthetic")


def build_pipeline() -> Pipeline:
    # Deliberately identical architecture to train_category_classifier.py --
    # isolating the effect of the data change.
    vectorizer = TfidfVectorizer(
        lowercase=True, ngram_range=(1, 2), min_df=2, sublinear_tf=True,
        strip_accents="unicode",
    )
    base_clf = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)
    return Pipeline([("tfidf", vectorizer), ("clf", calibrated)])


def evaluate(pipeline, df: pd.DataFrame, label_col: str, name: str) -> dict:
    y_true = df[label_col].to_numpy()
    y_pred = pipeline.predict(df["text"])
    y_proba = pipeline.predict_proba(df["text"])
    classes = pipeline.named_steps["clf"].classes_

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro")
    weighted_f1 = f1_score(y_true, y_pred, average="weighted")
    report_str = classification_report(y_true, y_pred, digits=4, zero_division=0)
    report_dict = classification_report(y_true, y_pred, digits=4, output_dict=True, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(classes))

    top_conf = y_proba.max(axis=1)
    mean_conf = float(top_conf.mean())
    above_gate = top_conf >= CONFIDENCE_THRESHOLD
    frac_above_gate = float(above_gate.mean())
    accuracy_above_gate = (
        float(accuracy_score(y_true[above_gate], y_pred[above_gate])) if above_gate.any() else None
    )
    correct = y_true == y_pred
    correct_and_above_gate = float((correct & above_gate).mean())

    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"\n{report_str}")
    print(f"Confusion matrix (rows=true, cols=pred), class order {list(classes)}:")
    print(cm)
    print(f"Mean top-class confidence            : {mean_conf:.4f}")
    print(f"Fraction clearing 0.65 confidence gate: {frac_above_gate:.4f}")
    print(f"Accuracy among those >=0.65 conf      : {accuracy_above_gate}")
    print(f"Correct AND clears 0.65 gate           : {correct_and_above_gate:.4f}")

    return {
        "accuracy": accuracy, "macro_f1": macro_f1, "weighted_f1": weighted_f1,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(), "confusion_matrix_labels": list(classes),
        "mean_top_confidence": mean_conf,
        "fraction_above_0.65_confidence": frac_above_gate,
        "accuracy_among_above_0.65_confidence": accuracy_above_gate,
        "correct_and_above_0.65_gate_fraction": correct_and_above_gate,
        "n_rows": len(df),
    }


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")

    for p in [FINTEXT_TRAIN_CSV, FINTEXT_TEST_CSV, V2_SYNTHETIC_CSV, GOLD_EVAL_CSV]:
        if not os.path.exists(p):
            print(f"ERROR: required file missing: {p}")
            sys.exit(1)

    print("Loading data sources...")
    fintext_train = load_fintext_train()
    coderanand = load_coderanand()
    bhavya = load_bhavya()
    v2_synth = load_v2_synthetic()

    for name, d in [("fintext6k_train", fintext_train), ("kaggle_coderanand", coderanand),
                     ("kaggle_bhavya", bhavya), ("paysense_v2_synthetic", v2_synth)]:
        print(f"  {name}: {len(d)} rows")

    combined = pd.concat([fintext_train, coderanand, bhavya, v2_synth], ignore_index=True)
    print(f"\nCombined (pre-dedup, pre-leakage-filter): {len(combined)} rows")

    # ── Leakage discipline: drop any row that exactly matches gold eval or
    #    FinText-6K's own held-out test set. ──────────────────────────────
    gold_eval = pd.read_csv(GOLD_EVAL_CSV)
    fintext_test_raw = pd.read_csv(FINTEXT_TEST_CSV)
    protected_texts = set(gold_eval["text"].astype(str)) | set(fintext_test_raw["text"].astype(str))

    before = len(combined)
    combined = combined[~combined["text"].astype(str).isin(protected_texts)].copy()
    print(f"Dropped {before - len(combined)} rows colliding with a held-out evaluation set")

    before = len(combined)
    combined = combined.drop_duplicates(subset=["text"]).reset_index(drop=True)
    print(f"Dropped {before - len(combined)} exact-duplicate text rows across sources")

    combined = shuffle(combined, random_state=SEED).reset_index(drop=True)
    print(f"\nFinal training set: {len(combined)} rows")
    print(combined["display_label"].value_counts())
    print("\nBy source:")
    print(combined["source"].value_counts())

    pipeline = build_pipeline()
    t0 = time.time()
    pipeline.fit(combined["text"], combined["display_label"])
    train_secs = time.time() - t0
    print(f"\nTrained in {train_secs:.1f}s")

    vocab_size = len(pipeline.named_steps["tfidf"].vocabulary_)
    print(f"Fitted TF-IDF vocabulary size: {vocab_size} tokens "
          f"(baseline frozen model: 821 tokens)")

    # ── Evaluation 1: FinText-6K's own held-out test set (in-distribution
    #    regression check) ──────────────────────────────────────────────
    fintext_test_df = pd.read_csv(FINTEXT_TEST_CSV)
    fintext_test_df["display_label"] = fintext_test_df["label"].map(FINTEXT_LABEL_MAP)
    fintext_metrics = evaluate(
        pipeline, fintext_test_df, "display_label",
        "EVALUATION 1: FinText-6K held-out test set (1,000 rows, in-distribution regression check)"
    )

    # ── Evaluation 2: the gold, structurally-disjoint generalization set ──
    gold_eval["display_label"] = gold_eval["label"].map(FINTEXT_LABEL_MAP)
    gold_metrics = evaluate(
        pipeline, gold_eval, "display_label",
        "EVALUATION 2: category_generalization_test_set.csv (200 rows, GOLD, never trained on)"
    )

    # ── Re-check the exact cited failure examples from
    #    CATEGORY_CLASSIFIER_GENERALIZATION.md §4.2 ─────────────────────
    cited_examples = [
        ("Sent Rs.1250 to THE COFFEE HOUSE using Amazon Pay UPI", "Food"),
        ("IPO application blocked amount Rs 15000 - ZOMATO IPO ASBA, Ref 391827364511", "Investment"),
        ("EazyDiner reservation advance Rs 500 debited via UPI, ref 398271650912", "Food"),
    ]
    print("\n" + "=" * 78)
    print("RE-CHECK: exact cited failure examples from §4.2")
    print("=" * 78)
    cited_results = []
    for text, true_label in cited_examples:
        pred = pipeline.predict([text])[0]
        proba = pipeline.predict_proba([text])[0]
        classes = pipeline.named_steps["clf"].classes_
        conf = float(proba.max())
        correct = pred == true_label
        print(f"  TRUE={true_label:10s} PRED={pred:10s} conf={conf:.3f} "
              f"{'CORRECT' if correct else 'STILL WRONG'}  {text}")
        cited_results.append({"text": text, "true": true_label, "pred": pred,
                               "confidence": conf, "correct": bool(correct)})

    # ── Save candidate artefact (NOT the deployed path) ────────────────────
    os.makedirs(ARTEFACT_DIR, exist_ok=True)
    joblib.dump(pipeline, ARTEFACT_PATH)
    print(f"\nSaved candidate classifier -> {ARTEFACT_PATH}")

    metrics = {
        "trained_at_unix": time.time(),
        "train_rows_final": len(combined),
        "train_rows_by_source": combined["source"].value_counts().to_dict(),
        "train_rows_by_label": combined["display_label"].value_counts().to_dict(),
        "vocab_size": vocab_size,
        "baseline_vocab_size": 821,
        "fintext_holdout_test": fintext_metrics,
        "gold_generalization_test": gold_metrics,
        "cited_failure_examples_recheck": cited_results,
        "baseline_fintext_accuracy": 1.0000,
        "baseline_gold_accuracy": 0.7250,
        "baseline_gold_confidence_gate_pass_rate": 0.6200,
    }
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Saved metrics JSON -> {METRICS_PATH}")


if __name__ == "__main__":
    main()
