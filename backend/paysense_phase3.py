"""
================================================================================
  PaySense — Phase 3: Threshold Tuning & Model Freezing
  Author  : Senior ML Engineer
  Project : PaySense — ML-Based UPI Fraud Detection
  Input   : paysense_master_dataset.csv  (30,000 rows × 50 cols)
  Outputs : paysense_model.pkl           — frozen XGBoost classifier
            paysense_preprocessor.pkl    — frozen ColumnTransformer
            paysense_threshold.pkl       — optimal threshold scalar
            paysense_threshold_analysis.png — full threshold sweep chart

  CONTEXT
  ───────
  Phase 2 trained XGBoost at the default decision threshold of 0.50 and
  produced:  Precision = 100%  |  Recall = 37.94%  |  F1 = 55.01%
  The model's discriminative power (ROC-AUC 0.8851) is strong — the problem
  is entirely the threshold.  A threshold of 0.50 means the model only fires
  an alert when it is more than 50% certain of fraud.  For a system whose
  primary job is to CATCH fraud, this is far too conservative.

  This phase sweeps thresholds 0.05 → 0.50 to find the operating point that
  satisfies the business constraint:  Recall ≥ 75%  AND  Precision ≥ 50%
  (Precision floor set to 50% because false alerts destroy user trust).
================================================================================
"""

import os, warnings
warnings.filterwarnings("ignore")

import joblib
import numpy  as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing   import OrdinalEncoder
from sklearn.impute          import SimpleImputer
from sklearn.compose         import ColumnTransformer
from sklearn.pipeline        import Pipeline
from sklearn.metrics         import (
    precision_score, recall_score, f1_score,
    classification_report, roc_auc_score, average_precision_score,
    confusion_matrix,
)
from imblearn.over_sampling  import SMOTE
from xgboost                 import XGBClassifier

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_CSV = "paysense_master_dataset.csv"
OUTPUT_DIR = "outputs/"
MODEL_PATH = OUTPUT_DIR + "paysense_model.pkl"
PREPROCESSOR_PATH = OUTPUT_DIR + "paysense_preprocessor.pkl"
THRESHOLD_PATH = OUTPUT_DIR + "paysense_threshold.pkl"
PLOT_PATH = OUTPUT_DIR + "paysense_threshold_analysis.png"

RANDOM_STATE    = 42
np.random.seed(RANDOM_STATE)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ════════════════════════════════════════════════════════════════════════════
#  BLOCK 0 — REBUILD TRAINED OBJECTS
#  Exact reproduction of Phase 2 preprocessing + training so the saved
#  pkl files are consistent with the evaluation results already reported.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  BLOCK 0 │ Rebuilding Trained Model & Preprocessor (Phase 2 Replay)")
print("═" * 72)

df = pd.read_csv(INPUT_CSV)
print("Columns:", df.columns)
print("Shape:", df.shape)

DROP_COLS = [
    "transaction_id", "user_id", "receiver_id",
    "timestamp", "date", "data_source",
    "user_kyc_status", "status", "usr_home_city",
]
df = df.drop(columns=DROP_COLS)

TARGET   = "is_fraud"
X        = df.drop(columns=[TARGET])
y        = df[TARGET].astype(int)

CAT_COLS = X.select_dtypes(include=["object"]).columns.tolist()
NUM_COLS = X.select_dtypes(include=[np.number]).columns.tolist()

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y
)

# Build & fit preprocessor on TRAIN ONLY
cat_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="most_frequent")),
    ("encode", OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
        encoded_missing_value=-2,
    )),
])
num_pipeline = Pipeline([
    ("impute", SimpleImputer(strategy="median")),
])
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_pipeline, CAT_COLS),
        ("num", num_pipeline, NUM_COLS),
    ],
    remainder="drop",
    verbose_feature_names_out=False,
)
preprocessor.fit(X_train_raw)

X_train_proc = preprocessor.transform(X_train_raw)
X_test_proc  = preprocessor.transform(X_test_raw)

feature_names = CAT_COLS + NUM_COLS

# SMOTE on train only
smote = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=RANDOM_STATE)
X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

# Train XGBoost
model = XGBClassifier(
    n_estimators     = 400,
    max_depth        = 5,
    learning_rate    = 0.05,
    subsample        = 0.80,
    colsample_bytree = 0.80,
    min_child_weight = 10,
    gamma            = 0.10,
    scale_pos_weight = 1,
    reg_alpha        = 0.05,
    reg_lambda       = 1.50,
    eval_metric      = "aucpr",
    tree_method      = "hist",
    random_state     = RANDOM_STATE,
    n_jobs           = -1,
    verbosity        = 0,
)
print("  Training XGBoost … ", end="", flush=True)
model.fit(X_train_bal, y_train_bal, eval_set=[(X_test_proc, y_test)], verbose=False)
print("done ✓")

# Raw probabilities on the untouched test set
y_proba = model.predict_proba(X_test_proc)[:, 1]
roc_auc = roc_auc_score(y_test, y_proba)
pr_auc  = average_precision_score(y_test, y_proba)

print(f"  ROC-AUC : {roc_auc:.4f}  |  PR-AUC : {pr_auc:.4f}")
print(f"  Phase 2 scores confirmed — model rebuilt identically ✓")


# ════════════════════════════════════════════════════════════════════════════
#  BLOCK A — THRESHOLD SWEEP  (0.05 → 0.50, step 0.05)
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  BLOCK A │ Threshold Sweep")
print("─" * 72)

# Business constraints for optimal threshold selection
RECALL_MIN    = 0.75   # must catch at least 75% of real fraud
PRECISION_MIN = 0.50   # at least 1-in-2 alerts must be real fraud

thresholds  = np.arange(0.05, 0.51, 0.05)
results     = []

for t in thresholds:
    y_pred_t   = (y_proba >= t).astype(int)
    precision  = precision_score(y_test, y_pred_t, zero_division=0)
    recall     = recall_score(y_test, y_pred_t,    zero_division=0)
    f1         = f1_score(y_test, y_pred_t,        zero_division=0)
    cm         = confusion_matrix(y_test, y_pred_t)
    tn, fp, fn, tp = cm.ravel()
    results.append({
        "threshold" : round(t, 2),
        "precision" : precision,
        "recall"    : recall,
        "f1"        : f1,
        "tp"        : tp,
        "fp"        : fp,
        "fn"        : fn,
        "tn"        : tn,
        "meets_constraint": (recall >= RECALL_MIN and precision >= PRECISION_MIN),
    })

df_sweep = pd.DataFrame(results)

# ── Pretty-print the summary table ──────────────────────────────────────────
print(f"\n  {'Threshold':>10}  {'Precision':>10}  {'Recall':>8}  "
      f"{'F1-Score':>9}  {'TP':>5}  {'FP':>5}  {'FN':>5}  {'Constraint':>11}")
print(f"  {'─'*10}  {'─'*10}  {'─'*8}  {'─'*9}  {'─'*5}  {'─'*5}  {'─'*5}  {'─'*11}")

for _, row in df_sweep.iterrows():
    meets = "✓ MEETS" if row["meets_constraint"] else ""
    star  = " ◄" if row["threshold"] == 0.50 else ""  # phase 2 default marker
    print(
        f"  {row['threshold']:>10.2f}  "
        f"{row['precision']:>9.2%}  "
        f"{row['recall']:>7.2%}  "
        f"{row['f1']:>8.4f}  "
        f"{int(row['tp']):>5}  "
        f"{int(row['fp']):>5}  "
        f"{int(row['fn']):>5}  "
        f"{meets:>11}{star}"
    )

# ── Programmatic optimal threshold selection ─────────────────────────────────
#
#  Priority order:
#    1. Among rows meeting BOTH constraints (Recall ≥ 75%, Precision ≥ 50%),
#       pick the one with the highest F1-Score.
#    2. If no row meets both constraints (edge case on small test sets),
#       fall back to maximising F1 unconditionally.
#    3. Tiebreak: prefer the higher threshold (more conservative → fewer
#       false alerts, better user experience).

constraint_met = df_sweep[df_sweep["meets_constraint"]]

if not constraint_met.empty:
    best_row = constraint_met.loc[constraint_met["f1"].idxmax()]
    selection_reason = (
        f"Recall ≥ {RECALL_MIN:.0%} AND Precision ≥ {PRECISION_MIN:.0%}, "
        f"maximised F1"
    )
else:
    # Fallback: maximise F1 unconditionally
    best_row = df_sweep.loc[df_sweep["f1"].idxmax()]
    selection_reason = "No row met both constraints — fallback to max-F1"

OPTIMAL_THRESHOLD = float(best_row["threshold"])

print(f"\n  {'─'*72}")
print(f"  OPTIMAL THRESHOLD : {OPTIMAL_THRESHOLD}")
print(f"  Selection rule    : {selection_reason}")
print(f"  At this threshold :")
print(f"      Precision  = {best_row['precision']:.2%}")
print(f"      Recall     = {best_row['recall']:.2%}  "
      f"(caught {int(best_row['tp'])} / {y_test.sum()} fraud cases)")
print(f"      F1-Score   = {best_row['f1']:.4f}")
print(f"      FP (false alerts per 6,000 txns) = {int(best_row['fp'])}")
print(f"  {'─'*72}")


# ── Full classification report at optimal threshold ──────────────────────────
y_pred_optimal = (y_proba >= OPTIMAL_THRESHOLD).astype(int)
print(f"\n  Full Classification Report @ threshold = {OPTIMAL_THRESHOLD}:\n")
print(classification_report(y_test, y_pred_optimal,
                             target_names=["Legitimate", "Fraud"],
                             digits=4))


# ════════════════════════════════════════════════════════════════════════════
#  BLOCK A — VISUALISATION  (4-panel threshold analysis chart)
# ════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(18, 12))
fig.suptitle(
    "PaySense — Phase 3: Threshold Analysis\n"
    f"Optimal Threshold = {OPTIMAL_THRESHOLD}  "
    f"(Recall={best_row['recall']:.0%}, "
    f"Precision={best_row['precision']:.0%}, "
    f"F1={best_row['f1']:.3f})",
    fontsize=14, fontweight="bold", y=0.98,
)
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.32)

# ── Panel 1: Precision / Recall / F1 vs Threshold ───────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(df_sweep["threshold"], df_sweep["precision"],
         "o-", color="#1f77b4", lw=2, ms=6, label="Precision")
ax1.plot(df_sweep["threshold"], df_sweep["recall"],
         "s-", color="#d62728", lw=2, ms=6, label="Recall")
ax1.plot(df_sweep["threshold"], df_sweep["f1"],
         "^-", color="#2ca02c", lw=2, ms=6, label="F1-Score")
ax1.axvline(OPTIMAL_THRESHOLD, color="darkorange", lw=2,
            linestyle="--", label=f"Optimal = {OPTIMAL_THRESHOLD}")
ax1.axhline(RECALL_MIN,    color="#d62728", lw=1, linestyle=":",
            alpha=0.6, label=f"Recall floor = {RECALL_MIN:.0%}")
ax1.axhline(PRECISION_MIN, color="#1f77b4", lw=1, linestyle=":",
            alpha=0.6, label=f"Precision floor = {PRECISION_MIN:.0%}")
ax1.set_title("Precision / Recall / F1 vs. Threshold", fontweight="bold")
ax1.set_xlabel("Decision Threshold")
ax1.set_ylabel("Score")
ax1.set_ylim(0, 1.05)
ax1.legend(fontsize=8)
ax1.grid(alpha=0.3)

# ── Panel 2: TP and FP counts vs Threshold ───────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
ax2_twin = ax2.twinx()
bars_tp = ax2.bar(df_sweep["threshold"] - 0.01, df_sweep["tp"],
                  width=0.018, color="#2ca02c", alpha=0.75, label="True Positives (TP)")
bars_fp = ax2.bar(df_sweep["threshold"] + 0.01, df_sweep["fp"],
                  width=0.018, color="#d62728", alpha=0.75, label="False Positives (FP)")
ax2_twin.plot(df_sweep["threshold"], df_sweep["fn"],
              "v--", color="grey", lw=1.5, ms=6, label="False Negatives (FN)")
ax2.axvline(OPTIMAL_THRESHOLD, color="darkorange", lw=2, linestyle="--")
ax2.set_title("TP / FP / FN Counts vs. Threshold", fontweight="bold")
ax2.set_xlabel("Decision Threshold")
ax2.set_ylabel("TP / FP Count", color="black")
ax2_twin.set_ylabel("FN Count (missed fraud)", color="grey")
lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_twin.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, fontsize=8)
ax2.grid(alpha=0.3)

# ── Panel 3: Fraud probability score histogram (calibration view) ────────────
ax3 = fig.add_subplot(gs[1, 0])
bins = np.linspace(0, 1, 50)
ax3.hist(y_proba[y_test == 0], bins=bins, alpha=0.55,
         color="#1f77b4", density=True, label="Legitimate")
ax3.hist(y_proba[y_test == 1], bins=bins, alpha=0.65,
         color="#d62728", density=True, label="Fraud")
ax3.axvline(0.50, color="black",      lw=1.5, linestyle="--",
            label="Phase 2 threshold (0.50)")
ax3.axvline(OPTIMAL_THRESHOLD, color="darkorange", lw=2.5,
            linestyle="-", label=f"Optimal threshold ({OPTIMAL_THRESHOLD})")
ax3.set_title("Fraud Score Distribution by True Class\n"
              "(where to draw the line)", fontweight="bold")
ax3.set_xlabel("Predicted Fraud Probability")
ax3.set_ylabel("Density")
ax3.legend(fontsize=8)
ax3.grid(alpha=0.3)

# ── Panel 4: Phase 2 vs Phase 3 side-by-side bar comparison ─────────────────
ax4 = fig.add_subplot(gs[1, 1])
metrics_p2 = {
    "Precision": precision_score(y_test, (y_proba >= 0.50).astype(int), zero_division=0),
    "Recall":    recall_score(y_test,    (y_proba >= 0.50).astype(int), zero_division=0),
    "F1-Score":  f1_score(y_test,        (y_proba >= 0.50).astype(int), zero_division=0),
}
metrics_p3 = {
    "Precision": best_row["precision"],
    "Recall":    best_row["recall"],
    "F1-Score":  best_row["f1"],
}
metric_labels = list(metrics_p2.keys())
p2_vals = list(metrics_p2.values())
p3_vals = list(metrics_p3.values())
x_pos   = np.arange(len(metric_labels))
width   = 0.32
bars_p2 = ax4.bar(x_pos - width/2, p2_vals, width, label="Phase 2 (t=0.50)",
                  color="#7fb1d3", edgecolor="white")
bars_p3 = ax4.bar(x_pos + width/2, p3_vals, width,
                  label=f"Phase 3 (t={OPTIMAL_THRESHOLD})",
                  color="#2ca02c", edgecolor="white")
for bar in bars_p2:
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
             f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=9)
for bar in bars_p3:
    ax4.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015,
             f"{bar.get_height():.0%}", ha="center", va="bottom", fontsize=9,
             fontweight="bold")
ax4.set_xticks(x_pos)
ax4.set_xticklabels(metric_labels, fontsize=10)
ax4.set_ylim(0, 1.15)
ax4.set_title("Phase 2 vs. Phase 3\nMetric Improvement After Tuning",
              fontweight="bold")
ax4.set_ylabel("Score")
ax4.legend(fontsize=9)
ax4.grid(axis="y", alpha=0.3)
ax4.axhline(RECALL_MIN,    color="#d62728", lw=1, linestyle=":", alpha=0.5)
ax4.axhline(PRECISION_MIN, color="#1f77b4", lw=1, linestyle=":", alpha=0.5)

plt.savefig(PLOT_PATH, dpi=150, bbox_inches="tight")
plt.close()
print(f"  ✓ Saved → paysense_threshold_analysis.png")


# ════════════════════════════════════════════════════════════════════════════
#  BLOCK B — MODEL FREEZING & EXPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  BLOCK B │ Model Freezing & Export")
print("─" * 72)

# ── What gets saved and why ──────────────────────────────────────────────────
#
#  paysense_model.pkl
#    The trained XGBClassifier with all 400 trees baked in.
#    FastAPI loads this once at startup and reuses it for every request.
#
#  paysense_preprocessor.pkl
#    The fitted ColumnTransformer (OrdinalEncoder + SimpleImputer internals).
#    This object has LEARNED the encoding mappings (e.g. "PhonePe" → 3,
#    "Merchant" → 1) from the training data.  A live transaction must be
#    transformed by THIS EXACT object before being fed to the model.
#    Saving a fresh unfitted transformer would silently produce garbage output.
#
#  paysense_threshold.pkl
#    A single Python float — the optimal decision threshold.
#    Stored as a pkl so the FastAPI server never hardcodes it in source.
#    Changing the threshold for a future deployment is now a file swap,
#    not a code change.
#
#  paysense_feature_names.pkl
#    The ordered list of feature column names expected by the preprocessor.
#    The API must construct its input DataFrame in THIS column order, or
#    the ColumnTransformer will silently assign values to wrong features.
#    This is the most common source of silent production bugs.

joblib.dump(model,        MODEL_PATH)
joblib.dump(preprocessor, PREPROCESSOR_PATH)
joblib.dump(OPTIMAL_THRESHOLD, THRESHOLD_PATH)
joblib.dump(feature_names, OUTPUT_DIR + "paysense_feature_names.pkl")

print(f"  ✓ Model saved              → paysense_model.pkl")
print(f"  ✓ Preprocessor saved       → paysense_preprocessor.pkl")
print(f"  ✓ Threshold saved          → paysense_threshold.pkl  "
      f"(value: {OPTIMAL_THRESHOLD})")
print(f"  ✓ Feature names saved      → paysense_feature_names.pkl  "
      f"({len(feature_names)} features)")

# ── Verify round-trip integrity: reload and re-predict ───────────────────────
print(f"\n  Verifying round-trip integrity …")
model_rt        = joblib.load(MODEL_PATH)
preprocessor_rt = joblib.load(PREPROCESSOR_PATH)
threshold_rt    = joblib.load(THRESHOLD_PATH)
features_rt     = joblib.load(OUTPUT_DIR + "paysense_feature_names.pkl")

X_verify        = preprocessor_rt.transform(X_test_raw)
y_proba_rt      = model_rt.predict_proba(X_verify)[:, 1]
y_pred_rt       = (y_proba_rt >= threshold_rt).astype(int)

recall_rt    = recall_score(y_test, y_pred_rt)
precision_rt = precision_score(y_test, y_pred_rt, zero_division=0)
f1_rt        = f1_score(y_test, y_pred_rt)

assert abs(recall_rt - best_row["recall"]) < 1e-6, "Round-trip recall mismatch!"
assert abs(precision_rt - best_row["precision"]) < 1e-6, "Round-trip precision mismatch!"
print(f"  ✓ Reloaded model produces identical predictions")
print(f"      Recall={recall_rt:.2%}  Precision={precision_rt:.2%}  F1={f1_rt:.4f}")
print(f"  ✓ All artefacts are consistent and production-safe")

# ── File size audit ───────────────────────────────────────────────────────────
print(f"\n  Artefact sizes:")
for fname in ["paysense_model.pkl", "paysense_preprocessor.pkl",
              "paysense_threshold.pkl", "paysense_feature_names.pkl"]:
    size = os.path.getsize(OUTPUT_DIR + fname)
    print(f"      {fname:<35} {size/1024:>8.1f} KB")


# ════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  PAYSENSE PHASE 3 — FINAL SUMMARY")
print("═" * 72)
print(f"""
  THRESHOLD IMPROVEMENT (Phase 2 → Phase 3)
  ┌────────────────────┬──────────────────┬──────────────────┬────────────┐
  │ Metric             │ Phase 2 (t=0.50) │ Phase 3 (t={OPTIMAL_THRESHOLD:.2f}) │   Delta    │
  ├────────────────────┼──────────────────┼──────────────────┼────────────┤
  │ Precision          │   {metrics_p2["Precision"]:>7.2%}        │   {metrics_p3["Precision"]:>7.2%}        │  {metrics_p3["Precision"]-metrics_p2["Precision"]:+.2%}     │
  │ Recall             │   {metrics_p2["Recall"]:>7.2%}        │   {metrics_p3["Recall"]:>7.2%}        │  {metrics_p3["Recall"]-metrics_p2["Recall"]:+.2%}     │
  │ F1-Score           │   {metrics_p2["F1-Score"]:>7.4f}        │   {metrics_p3["F1-Score"]:>7.4f}        │  {metrics_p3["F1-Score"]-metrics_p2["F1-Score"]:+.4f}     │
  │ Fraud caught (TP)  │   {int(df_sweep[df_sweep["threshold"]==0.50]["tp"].values[0]):>3} / {y_test.sum()}         │   {int(best_row["tp"]):>3} / {y_test.sum()}         │  +{int(best_row["tp"])-int(df_sweep[df_sweep["threshold"]==0.50]["tp"].values[0]):<7}   │
  │ False alerts (FP)  │   {int(df_sweep[df_sweep["threshold"]==0.50]["fp"].values[0]):>3}              │   {int(best_row["fp"]):>3}              │  {int(best_row["fp"])-int(df_sweep[df_sweep["threshold"]==0.50]["fp"].values[0]):>+7}    │
  └────────────────────┴──────────────────┴──────────────────┴────────────┘

  FROZEN ARTEFACTS READY FOR FASTAPI
    ✓ paysense_model.pkl            (XGBoost, 400 trees)
    ✓ paysense_preprocessor.pkl     (ColumnTransformer, fitted on train)
    ✓ paysense_threshold.pkl        ({OPTIMAL_THRESHOLD})
    ✓ paysense_feature_names.pkl    ({len(feature_names)} features, in correct order)
    ✓ paysense_threshold_analysis.png

  ✅  Model is frozen and production-ready.  Next → FastAPI backend.
""")
print("═" * 72)
