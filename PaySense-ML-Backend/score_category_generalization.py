"""
================================================================================
  score_category_generalization.py
  ────────────────────────────────────────────────────────────────────────────
  Scores the FROZEN category classifier (artefacts/paysense_category_classifier.pkl,
  loaded read-only -- no retraining, no fine-tuning, no threshold recalibration)
  against category_generalization_test_set.csv, the hand-authored, structurally
  disjoint-from-FinText-6K test set built by
  build_category_generalization_test_set.py.

  Loads the artefact and calls predict_proba() the same way
  src/fraud_model.py's classify_category() does (same .pkl, same
  named_steps["clf"].classes_ lookup for the class order), so the number
  reported here is what production /classify would actually return.

  Run:
      cd PaySense-ML-Backend
      venv\\Scripts\\python.exe score_category_generalization.py
================================================================================
"""
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score

_HERE = os.path.dirname(os.path.abspath(__file__))
ARTEFACT_PATH = os.path.join(_HERE, "artefacts", "paysense_category_classifier.pkl")
TEST_CSV = os.path.join(_HERE, "category_generalization_test_set.csv")
OUT_JSON = os.path.join(_HERE, "artefacts", "category_generalization_metrics.json")

LABEL_DISPLAY_MAP = {
    "food": "Food",
    "travel": "Travel",
    "EMI": "EMI",
    "investment": "Investment",
    "shopping": "Shopping",
}

CONFIDENCE_THRESHOLD = 0.65  # Android's NLP_CONFIDENCE_THRESHOLD


def main() -> None:
    # Windows console default codepage (cp1252) can't encode the rupee sign
    # some example narrations contain -- force UTF-8 stdout so this script
    # runs cleanly rather than crashing mid-report.
    sys.stdout.reconfigure(encoding="utf-8")

    pipeline = joblib.load(ARTEFACT_PATH)
    df = pd.read_csv(TEST_CSV)
    df["display_label"] = df["label"].map(LABEL_DISPLAY_MAP)
    assert df["display_label"].isna().sum() == 0, "Unmapped label in test set"

    y_true = df["display_label"].to_numpy()
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
    # The metric that actually matters in production: correct AND clears the
    # confidence gate. A correct-but-low-confidence prediction still falls
    # through to Tier-3 HITL, so it does not count as a Tier-2 success.
    correct = y_true == y_pred
    correct_and_above_gate = float((correct & above_gate).mean())

    print("=" * 78)
    print("NOVEL GENERALIZATION TEST SET METRICS (200 rows, hand-authored,")
    print("structurally disjoint from FinText-6K's 40 templates)")
    print("=" * 78)
    print(f"Accuracy      : {accuracy:.4f}")
    print(f"Macro F1      : {macro_f1:.4f}")
    print(f"Weighted F1   : {weighted_f1:.4f}")
    print(f"\nClassification report:\n{report_str}")
    print(f"Confusion matrix (rows=true, cols=pred), class order {list(classes)}:")
    print(cm)
    print(f"\nMean top-class confidence            : {mean_conf:.4f}")
    print(f"Fraction clearing 0.65 confidence gate: {frac_above_gate:.4f}")
    print(f"Accuracy among those >=0.65 conf      : {accuracy_above_gate}")
    print(f"Correct AND clears 0.65 gate (= real Tier-2 success rate): {correct_and_above_gate:.4f}")

    # Misclassified rows, for failure-pattern diagnosis.
    mis_idx = np.where(~correct)[0]
    print(f"\n{len(mis_idx)} misclassified rows:")
    failures = []
    for i in mis_idx:
        row = {
            "text": df["text"].iloc[i],
            "true": y_true[i],
            "pred": y_pred[i],
            "confidence": float(top_conf[i]),
        }
        failures.append(row)
        print(f"  TRUE={row['true']:10s} PRED={row['pred']:10s} conf={row['confidence']:.3f}  {row['text']}")

    metrics = {
        "test_set": "category_generalization_test_set.csv",
        "n_rows": len(df),
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "classification_report": report_dict,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_labels": list(classes),
        "mean_top_confidence": mean_conf,
        "fraction_above_0.65_confidence": frac_above_gate,
        "accuracy_among_above_0.65_confidence": accuracy_above_gate,
        "correct_and_above_0.65_gate_fraction": correct_and_above_gate,
        "misclassified_rows": failures,
    }
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics JSON -> {OUT_JSON}")


if __name__ == "__main__":
    main()
