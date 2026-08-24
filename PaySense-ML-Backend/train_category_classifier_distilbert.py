"""
train_category_classifier_distilbert.py -- tests the actual open question
CATEGORY_CLASSIFIER_GENERALIZATION.md and CATEGORY_CLASSIFIER_V3_ATTEMPT.md
both left unanswered: is TF-IDF + linear SVM's ceiling a data problem (not
enough sentence diversity) or an architecture problem (a literal-token
lookup table can't understand language, no matter how much data you feed
it)? v3 already tested the data lever (broader templates: 72.5% -> 78.0%
accuracy on the gold eval set). This tests the architecture lever, holding
the data EXACTLY constant -- same FinText-6K train split, same v3 synthetic
blend, same eval sets -- so any accuracy difference is attributable to
DistilBERT vs. TF-IDF, not to a data change riding along with it.

Does NOT touch artefacts/paysense_category_classifier.pkl (the deployed v3
artifact) or any other existing classifier file. Saves a new fine-tuned
model directory + comparable metrics JSON.
"""
import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from torch.utils.data import Dataset
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizerFast,
    Trainer,
    TrainingArguments,
)

_HERE = os.path.dirname(os.path.abspath(__file__))
FINTEXT_DIR = r"E:\Projects\upi\FinText-6K"
TRAIN_CSV = os.path.join(FINTEXT_DIR, "train_transaction_dataset.csv")
TEST_CSV = os.path.join(FINTEXT_DIR, "test_transaction_dataset.csv")
V3_CSV = os.path.join(_HERE, "category_training_v3_synthetic.csv")
EVAL_CSV = os.path.join(_HERE, "category_generalization_test_set.csv")
OUT_MODEL_DIR = os.path.join(_HERE, "artefacts", "paysense_category_classifier_distilbert")
OUT_METRICS = os.path.join(_HERE, "artefacts", "category_classifier_distilbert_metrics.json")

LABEL_DISPLAY_MAP = {"food": "Food", "travel": "Travel", "EMI": "EMI",
                      "investment": "Investment", "shopping": "Shopping"}
LABELS = ["Food", "Travel", "EMI", "Investment", "Shopping"]  # fixed id order
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = {i: label for i, label in enumerate(LABELS)}
CONF_THRESHOLD = 0.65
MAX_LENGTH = 48
MODEL_NAME = "distilbert-base-uncased"


class TxnTextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.encodings = tokenizer(
            list(texts), truncation=True, padding="max_length", max_length=MAX_LENGTH,
        )
        self.labels = [LABEL2ID[l] for l in labels]

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: torch.tensor(v[idx]) for k, v in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item


def main():
    fintext_train = pd.read_csv(TRAIN_CSV)
    fintext_test = pd.read_csv(TEST_CSV)
    v3 = pd.read_csv(V3_CSV)
    eval_df = pd.read_csv(EVAL_CSV)

    for df in (fintext_train, fintext_test, v3, eval_df):
        df["display_label"] = df["label"].map(LABEL_DISPLAY_MAP)

    blended_train = pd.concat([fintext_train, v3], ignore_index=True)
    print(f"Blended training set: {len(blended_train)} rows (same as v3's -- "
          f"FinText-6K train {len(fintext_train)} + v3 synthetic {len(v3)})")
    print(blended_train["display_label"].value_counts())

    print(f"\nLoading tokenizer + model: {MODEL_NAME}")
    tokenizer = DistilBertTokenizerFast.from_pretrained(MODEL_NAME)
    model = DistilBertForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID,
    )

    train_ds = TxnTextDataset(blended_train["text"], blended_train["display_label"], tokenizer)
    fintext_test_ds = TxnTextDataset(fintext_test["text"], fintext_test["display_label"], tokenizer)

    training_args = TrainingArguments(
        output_dir=os.path.join(_HERE, "distilbert_train_tmp"),
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        logging_steps=50,
        eval_strategy="epoch",
        save_strategy="no",
        report_to=[],
        use_cpu=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=fintext_test_ds,
    )

    print("\nFine-tuning DistilBERT ...")
    trainer.train()
    print("done.")

    results = {}

    def evaluate(name, df):
        ds = TxnTextDataset(df["text"], df["display_label"], tokenizer)
        pred_output = trainer.predict(ds)
        logits = pred_output.predictions
        probs = torch.softmax(torch.tensor(logits), dim=1).numpy()
        pred_ids = probs.argmax(axis=1)
        preds = np.array([ID2LABEL[i] for i in pred_ids])
        confidences = probs.max(axis=1)

        true = df["display_label"].values
        acc = accuracy_score(true, preds)
        gated_acc = ((preds == true) & (confidences >= CONF_THRESHOLD)).mean()
        gate_pass_rate = (confidences >= CONF_THRESHOLD).mean()
        report = classification_report(true, preds, output_dict=True, labels=LABELS)
        cm = confusion_matrix(true, preds, labels=LABELS)
        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(f"Correct AND confidence>=0.65: {gated_acc:.4f}  (gate pass rate: {gate_pass_rate:.4f})")
        print(classification_report(true, preds, labels=LABELS))
        print("Confusion matrix (order:", LABELS, "):")
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

    os.makedirs(OUT_MODEL_DIR, exist_ok=True)
    trainer.save_model(OUT_MODEL_DIR)
    tokenizer.save_pretrained(OUT_MODEL_DIR)
    with open(OUT_METRICS, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved model -> {OUT_MODEL_DIR}")
    print(f"Saved metrics -> {OUT_METRICS}")


if __name__ == "__main__":
    main()
