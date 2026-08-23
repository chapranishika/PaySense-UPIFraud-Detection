"""Recompute CATEGORY_CLASSIFIER_GENERALIZATION.md §4.1's "vocabulary
collapse" bucketing (rows with ZERO content-word overlap vs. fitted TF-IDF
vocabulary) for both the baseline and V2 classifiers, against the same
200-row gold eval set, to check whether Failure Mode A is genuinely reduced
by vocabulary expansion, or whether the V2 accuracy gain is better explained
some other way. Read-only inference, no retraining."""
import re
import sys

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score

sys.stdout.reconfigure(encoding="utf-8")

BASELINE_PATH = "artefacts/paysense_category_classifier.pkl"
V2_PATH = "artefacts/paysense_category_classifier_v2.pkl"
GOLD_CSV = "category_generalization_test_set.csv"
LABEL_MAP = {"food": "Food", "travel": "Travel", "EMI": "EMI", "investment": "Investment", "shopping": "Shopping"}

TOKEN_RE = re.compile(r"[a-z0-9]+")

def content_tokens(vocab_keys):
    """Vocabulary keys minus the tiny set of connective tokens shared by
    every FinText-6K template (rs/ref/upi/via/of + their bigrams) -- same
    definition CATEGORY_CLASSIFIER_GENERALIZATION.md §4.2 used."""
    connectives = {"rs", "ref", "upi", "via", "of", "upi ref", "via upi", "of rs"}
    return {t for t in vocab_keys if t not in connectives}


def bucket(pipeline, df):
    vocab = set(pipeline.named_steps["tfidf"].vocabulary_.keys())
    content_vocab = content_tokens(vocab)

    def has_content_overlap(text):
        words = TOKEN_RE.findall(str(text).lower())
        unigrams = set(words)
        bigrams = {f"{a} {b}" for a, b in zip(words, words[1:])}
        return bool((unigrams | bigrams) & content_vocab)

    df = df.copy()
    df["has_overlap"] = df["text"].apply(has_content_overlap)
    y_true = df["label"].map(LABEL_MAP).to_numpy()
    y_pred = pipeline.predict(df["text"])
    df["correct"] = (y_true == y_pred)

    zero = df[~df["has_overlap"]]
    some = df[df["has_overlap"]]
    print(f"  Vocabulary size (incl. connectives): {len(vocab)}")
    print(f"  Rows with ZERO content-word overlap: {len(zero)}/{len(df)} "
          f"({len(zero)/len(df)*100:.1f}%) -> accuracy "
          f"{zero['correct'].mean() if len(zero) else float('nan'):.3f}")
    print(f"  Rows with >=1 content-word overlap : {len(some)}/{len(df)} "
          f"({len(some)/len(df)*100:.1f}%) -> accuracy "
          f"{some['correct'].mean() if len(some) else float('nan'):.3f}")
    print(f"  Overall accuracy: {df['correct'].mean():.4f}")
    return df


def main():
    gold = pd.read_csv(GOLD_CSV)

    print("=" * 78)
    print("BASELINE (frozen, deployed) classifier -- vocabulary collapse bucketing")
    print("=" * 78)
    baseline = joblib.load(BASELINE_PATH)
    base_df = bucket(baseline, gold)

    print()
    print("=" * 78)
    print("V2 (candidate) classifier -- vocabulary collapse bucketing")
    print("=" * 78)
    v2 = joblib.load(V2_PATH)
    v2_df = bucket(v2, gold)

    # Rows the baseline got wrong -- does V2 fix them?
    print()
    print("=" * 78)
    print("Of the rows BASELINE got wrong, how many does V2 get right?")
    print("=" * 78)
    baseline_wrong = base_df[~base_df["correct"]]
    v2_correct_on_those = v2_df.loc[baseline_wrong.index, "correct"]
    print(f"  Baseline wrong: {len(baseline_wrong)} rows")
    print(f"  Of those, V2 correct: {v2_correct_on_those.sum()} "
          f"({v2_correct_on_those.mean()*100:.1f}%)")
    print(f"  Of those, V2 still wrong: {(~v2_correct_on_those).sum()}")

    # Are there any rows where V2 collapses to a single dominant default the
    # way the baseline did (identical predicted-probability vector across
    # true-label-disjoint rows with zero vocab overlap)?
    print()
    print("=" * 78)
    print("V2: predicted-probability vectors for the zero-overlap bucket")
    print("(baseline collapsed ALL such rows to one fixed default vector)")
    print("=" * 78)
    v2_zero = v2_df[~v2_df["has_overlap"]]
    if len(v2_zero):
        proba = v2.predict_proba(v2_zero["text"])
        unique_rounded = {tuple(round(x, 2) for x in row) for row in proba}
        print(f"  {len(v2_zero)} zero-overlap rows -> {len(unique_rounded)} "
              f"distinct (rounded) probability vectors")
    else:
        print("  No zero-overlap rows in V2's vocabulary -- bucket is empty.")


if __name__ == "__main__":
    main()
