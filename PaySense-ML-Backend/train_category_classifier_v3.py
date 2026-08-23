"""
train_category_classifier_v3.py -- trains on FinText-6K (train split) blended
with category_training_v3_synthetic.csv (built by generate_category_training_v3.py
WITHOUT ever reading category_generalization_test_set.csv's content), then
evaluates on: (a) FinText-6K's own held-out test split (in-distribution
regression check), and (b) category_generalization_test_set.csv (the real
generalization number, loaded here ONLY programmatically for scoring, never
read/quoted by a human or agent during template design).

Does NOT touch artefacts/paysense_category_classifier.pkl (the deployed
artifact) or paysense_category_classifier_v2.pkl (the invalidated attempt).
Saves to artefacts/paysense_category_classifier_v3.pkl.
"""
import json
import os

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

_HERE = os.path.dirname(os.path.abspath(__file__))
FINTEXT_DIR = r"E:\Projects\upi\FinText-6K"
TRAIN_CSV = os.path.join(FINTEXT_DIR, "train_transaction_dataset.csv")
TEST_CSV = os.path.join(FINTEXT_DIR, "test_transaction_dataset.csv")
V3_CSV = os.path.join(_HERE, "category_training_v3_synthetic.csv")
EVAL_CSV = os.path.join(_HERE, "category_generalization_test_set.csv")
OUT_MODEL = os.path.join(_HERE, "artefacts", "paysense_category_classifier_v3.pkl")
OUT_METRICS = os.path.join(_HERE, "artefacts", "category_classifier_v3_metrics.json")

LABEL_DISPLAY_MAP = {"food": "Food", "travel": "Travel", "EMI": "EMI",
                      "investment": "Investment", "shopping": "Shopping"}
CONF_THRESHOLD = 0.65


def build_pipeline():
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, strip_accents="unicode")
    base_clf = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)
    return Pipeline([("tfidf", vectorizer), ("clf", calibrated)])


def main():
    fintext_train = pd.read_csv(TRAIN_CSV)
    fintext_test = pd.read_csv(TEST_CSV)
    v3 = pd.read_csv(V3_CSV)
    eval_df = pd.read_csv(EVAL_CSV)

    for df in (fintext_train, fintext_test, v3, eval_df):
        df["display_label"] = df["label"].map(LABEL_DISPLAY_MAP)

    print(f"FinText-6K train: {len(fintext_train)}  FinText-6K test: {len(fintext_test)}  "
          f"v3 synthetic: {len(v3)}  eval (gold, held-out): {len(eval_df)}")

    blended_train = pd.concat([fintext_train, v3], ignore_index=True)
    print(f"Blended training set: {len(blended_train)} rows")
    print(blended_train["display_label"].value_counts())

    pipe = build_pipeline()
    print("\nTraining v3 pipeline ...")
    pipe.fit(blended_train["text"], blended_train["display_label"])
    print("done.")

    results = {}

    def evaluate(name, df):
        preds = pipe.predict(df["text"])
        probs = pipe.predict_proba(df["text"]).max(axis=1)
        acc = accuracy_score(df["display_label"], preds)
        gated_acc = ((preds == df["display_label"].values) & (probs >= CONF_THRESHOLD)).mean()
        gate_pass_rate = (probs >= CONF_THRESHOLD).mean()
        report = classification_report(df["display_label"], preds, output_dict=True)
        cm = confusion_matrix(df["display_label"], preds, labels=list(LABEL_DISPLAY_MAP.values()))
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Correct AND confidence>=0.65: {gated_acc:.4f}  (gate pass rate: {gate_pass_rate:.4f})")
        print(classification_report(df["display_label"], preds))
        print("Confusion matrix (order:", list(LABEL_DISPLAY_MAP.values()), "):")
        print(cm)
        results[name] = {
            "accuracy": float(acc),
            "correct_and_confident": float(gated_acc),
            "gate_pass_rate": float(gate_pass_rate),
            "per_class": report,
            "confusion_matrix": cm.tolist(),
            "n": len(df),
        }

    evaluate("fintext_own_test_set_in_distribution", fintext_test)
    evaluate("gold_novel_eval_set_generalization", eval_df)

    joblib.dump(pipe, OUT_MODEL)
    with open(OUT_METRICS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_MODEL}")
    print(f"Saved -> {OUT_METRICS}")


if __name__ == "__main__":
    main()
