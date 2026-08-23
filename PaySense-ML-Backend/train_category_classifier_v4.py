"""
train_category_classifier_v4.py -- v3 blended FinText-6K with a fresh,
non-eval-derived synthetic template generator. v4 adds two REAL Kaggle
datasets on top (found and vetted by an earlier, otherwise-invalidated
attempt -- see CATEGORY_CLASSIFIER_V3_ATTEMPT.md section 0 and
external_data/NOTICE.md for the full vetting record; only the DATASET
DISCOVERY from that attempt is reused here, none of its contaminated
templates):

  - coderanand/indian-banking-transaction-text-dataset (Apache 2.0,
    11,000 rows, itself synthetic/templated per its own description, but a
    DIFFERENT generator/vocabulary than both FinText-6K and this project's
    v3 templates -- Kaggle download, re-verified zero overlap with the
    gold eval set below).
  - bhavyasingh25/financial-transaction-description-dataset (MIT, 6,000
    rows, 9 classes -- only the 5 that map onto PaySense's taxonomy are
    kept; entertainment/healthcare/education/utilities rows are dropped,
    not force-mapped).

Evaluated the same way as v3: FinText-6K's own held-out test (regression
check) and category_generalization_test_set.csv (the gold, held-out
generalization number) -- loaded here ONLY programmatically.
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
CODERANAND_TRAIN = os.path.join(_HERE, "external_data", "kaggle_coderanand_indian_banking", "financial_transaction_train.csv")
CODERANAND_TEST = os.path.join(_HERE, "external_data", "kaggle_coderanand_indian_banking", "financial_transaction_test.csv")
BHAVYA_TRAIN = os.path.join(_HERE, "external_data", "kaggle_bhavya_financial_transaction", "train_transactions.csv")
BHAVYA_TEST = os.path.join(_HERE, "external_data", "kaggle_bhavya_financial_transaction", "test_transactions.csv")
OUT_MODEL = os.path.join(_HERE, "artefacts", "paysense_category_classifier_v4.pkl")
OUT_METRICS = os.path.join(_HERE, "artefacts", "category_classifier_v4_metrics.json")

LABEL_DISPLAY_MAP = {"food": "Food", "travel": "Travel", "EMI": "EMI",
                      "investment": "Investment", "shopping": "Shopping"}
CONF_THRESHOLD = 0.65


def build_pipeline():
    vectorizer = TfidfVectorizer(lowercase=True, ngram_range=(1, 2), min_df=2,
                                  sublinear_tf=True, strip_accents="unicode")
    base_clf = LinearSVC(C=1.0, class_weight="balanced", random_state=42)
    calibrated = CalibratedClassifierCV(base_clf, method="sigmoid", cv=5)
    return Pipeline([("tfidf", vectorizer), ("clf", calibrated)])


def load_coderanand():
    df = pd.concat([pd.read_csv(CODERANAND_TRAIN), pd.read_csv(CODERANAND_TEST)], ignore_index=True)
    df = df.rename(columns={"Transaction_Text": "text", "Label": "display_label"})
    return df[["text", "display_label"]]


def load_bhavya():
    df = pd.concat([pd.read_csv(BHAVYA_TRAIN), pd.read_csv(BHAVYA_TEST)], ignore_index=True)
    df = df.rename(columns={"transaction_text": "text", "category": "label"})
    label_map = {"food": "Food", "travel": "Travel", "emi": "EMI",
                 "investment": "Investment", "shopping": "Shopping"}
    df = df[df["label"].isin(label_map.keys())].copy()
    dropped = len(pd.concat([pd.read_csv(BHAVYA_TRAIN), pd.read_csv(BHAVYA_TEST)])) - len(df)
    print(f"bhavya: kept {len(df)} rows in PaySense's 5 classes, dropped {dropped} "
          f"rows in classes PaySense doesn't have (entertainment/healthcare/education/utilities)")
    df["display_label"] = df["label"].map(label_map)
    return df[["text", "display_label"]]


def main():
    fintext_train = pd.read_csv(TRAIN_CSV)
    fintext_test = pd.read_csv(TEST_CSV)
    v3 = pd.read_csv(V3_CSV)
    eval_df = pd.read_csv(EVAL_CSV)
    for df in (fintext_train, fintext_test, v3, eval_df):
        df["display_label"] = df["label"].map(LABEL_DISPLAY_MAP)

    coderanand = load_coderanand()
    bhavya = load_bhavya()

    # Leakage check against the gold eval set for the two new real sources
    eval_exact = set(eval_df["text"])
    for name, df in [("coderanand", coderanand), ("bhavya", bhavya)]:
        overlap = set(df["text"]) & eval_exact
        if overlap:
            raise SystemExit(f"REFUSING: {name} has {len(overlap)} exact-text rows overlapping the gold eval set.")
        print(f"{name}: {len(df)} usable rows, 0 overlap with gold eval set")

    blended_train = pd.concat(
        [fintext_train[["text", "display_label"]], v3[["text", "display_label"]],
         coderanand, bhavya],
        ignore_index=True,
    )
    print(f"\nBlended training set: {len(blended_train)} rows")
    print(blended_train["display_label"].value_counts())

    pipe = build_pipeline()
    print("\nTraining v4 pipeline ...")
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
            "accuracy": float(acc), "correct_and_confident": float(gated_acc),
            "gate_pass_rate": float(gate_pass_rate), "per_class": report,
            "confusion_matrix": cm.tolist(), "n": len(df),
        }

    evaluate("fintext_own_test_set_in_distribution", fintext_test.assign(display_label=fintext_test["label"].map(LABEL_DISPLAY_MAP)))
    evaluate("gold_novel_eval_set_generalization", eval_df)

    joblib.dump(pipe, OUT_MODEL)
    with open(OUT_METRICS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {OUT_MODEL}\nSaved -> {OUT_METRICS}")


if __name__ == "__main__":
    main()
