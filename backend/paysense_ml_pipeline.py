"""
================================================================================
  PaySense — XGBoost Fraud Detection ML Pipeline
  Author  : Senior ML Engineer
  Project : PaySense — ML-Based UPI Fraud Detection
  Input   : paysense_master_dataset.csv  (30,000 rows × 50 cols)
  Output  : Trained XGBoost model + evaluation plots + SHAP analysis

  PIPELINE STAGES
  ───────────────
  A │ Final Preprocessing   — drop non-predictive cols, encode categoricals
  B │ Split & Balance       — stratified 80/20 split → SMOTE on train ONLY
  C │ XGBoost Training      — calibrated for imbalanced financial data
  D │ Evaluation            — Classification Report, ROC-AUC, PR-AUC
  E │ Explainability        — SHAP summary + XGBoost feature importance

  STRICT CONSTRAINT:
  SMOTE is applied ONLY after the train/test split and ONLY to X_train/y_train.
  X_test remains completely untouched throughout the entire pipeline.
================================================================================
"""

# ── Standard library ─────────────────────────────────────────────────────────
import os
import warnings
warnings.filterwarnings("ignore")

# ── Data & ML ────────────────────────────────────────────────────────────────
import numpy  as np
import pandas as pd

from sklearn.model_selection     import train_test_split, StratifiedKFold
from sklearn.preprocessing       import OrdinalEncoder
from sklearn.impute              import SimpleImputer
from sklearn.compose             import ColumnTransformer
from sklearn.pipeline            import Pipeline
from sklearn.metrics             import (
    classification_report,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
)
from imblearn.over_sampling      import SMOTE
from xgboost                     import XGBClassifier, plot_importance
import shap

# ── Plotting ─────────────────────────────────────────────────────────────────
import matplotlib
matplotlib.use("Agg")           # non-interactive backend (safe for scripts)
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── Paths ────────────────────────────────────────────────────────────────────
INPUT_CSV   = "/mnt/user-data/outputs/paysense_master_dataset.csv"
OUTPUT_DIR  = "/mnt/user-data/outputs/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


# ════════════════════════════════════════════════════════════════════════════
#  STAGE A — FINAL PREPROCESSING
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  STAGE A │ Final Preprocessing")
print("═" * 72)

df = pd.read_csv(INPUT_CSV)
print(f"  Loaded master dataset             : {df.shape[0]:,} rows × {df.shape[1]} cols")

# ── A.1  Drop non-predictive columns ────────────────────────────────────────
#
#  IDENTIFIERS  → contain no predictive signal; would cause model to memorise
#                 individual transactions rather than learn fraud patterns.
#  TEMPORAL     → timestamp and date are ordered; including them leaks the
#                 chronological structure of fraud into a non-temporal model.
#  METADATA     → data_source was an audit trail column, not a real feature.
#  REDUNDANT    → user_kyc_status (string) is fully captured by kyc_verified_flag
#                 (binary). status (string) is fully captured by txn_success_flag.
#                 Keeping both would double-weight these signals.
#  HIGH-CARD    → usr_home_city has 38 unique city strings. Without spatial
#                 feature engineering (e.g., city → lat/lon → distance to
#                 merchant), the raw string adds noise, not signal.

DROP_COLS = [
    # Identifiers
    "transaction_id", "user_id", "receiver_id",
    # Temporal (leakage vector in non-temporal models)
    "timestamp", "date",
    # Metadata
    "data_source",
    # Redundant string columns (binary flags already capture these)
    "user_kyc_status", "status",
    # High-cardinality geographic string (no spatial engineering done)
    "usr_home_city",
]

df = df.drop(columns=DROP_COLS)
print(f"  After dropping {len(DROP_COLS)} non-predictive cols  : {df.shape[1]} cols remain")

# ── A.2  Separate features and target ────────────────────────────────────────
TARGET = "is_fraud"
X = df.drop(columns=[TARGET])
y = df[TARGET].astype(int)

print(f"  Feature matrix shape              : {X.shape}")
print(f"  Target vector shape               : {y.shape}")
print(f"  Class distribution in full dataset:\n"
      f"      Legitimate (0) : {(y==0).sum():,}  ({(y==0).mean()*100:.2f}%)\n"
      f"      Fraud      (1) : {(y==1).sum():,}  ({(y==1).mean()*100:.2f}%)")

# ── A.3  Classify columns by dtype for the ColumnTransformer ────────────────
#
#  CATEGORICAL columns → OrdinalEncoder (not OHE)
#  DESIGN CHOICE: OrdinalEncoder is preferred over OneHotEncoder here because:
#    1. XGBoost uses tree splits, not linear algebra → it does NOT need the
#       geometric independence that OHE provides for linear models.
#    2. OHE on 9 categorical cols (some with 15 categories) inflates the
#       feature space from ~41 to ~65+ columns, diluting feature importance.
#    3. OrdinalEncoder keeps the schema compact and SHAP values interpretable
#       against the original column names — critical for your report charts.
#  handle_unknown='use_encoded_value' with unknown_value=-1 means any
#  production inference call with a new category degrades gracefully.

CAT_COLS = X.select_dtypes(include=["object"]).columns.tolist()
NUM_COLS = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\n  Categorical columns ({len(CAT_COLS)}):")
for c in CAT_COLS:
    print(f"      {c:<30} ({X[c].nunique()} unique values)")
print(f"\n  Numerical columns ({len(NUM_COLS)}) — includes intentional NaN columns:")
nan_num = {c: X[c].isnull().sum() for c in NUM_COLS if X[c].isnull().sum() > 0}
for c, n in nan_num.items():
    print(f"      {c:<35} ({n:,} NaN — will be median-imputed)")


# ════════════════════════════════════════════════════════════════════════════
#  STAGE B — STRATIFIED SPLIT  →  SMOTE ON TRAIN ONLY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STAGE B │ Stratified Split → SMOTE (train only)")
print("─" * 72)

# ── B.1  Stratified 80/20 split — BEFORE any preprocessing fit ──────────────
#  stratify=y ensures the 4.21% fraud rate is preserved in both partitions.
#  The ColumnTransformer is fitted ONLY on X_train below — never X_test.
#  This is the correct order to prevent test-set leakage into encoder state.

X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X, y,
    test_size    = 0.20,
    random_state = RANDOM_STATE,
    stratify     = y,
)

print(f"  Train set  : {X_train_raw.shape[0]:,} rows "
      f"| Fraud: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
print(f"  Test set   : {X_test_raw.shape[0]:,} rows  "
      f"| Fraud: {y_test.sum():,}  ({y_test.mean()*100:.2f}%)")
print(f"  ✓ Stratification preserved fraud rate in both partitions")

# ── B.2  Build preprocessing transformer (fit on TRAIN, transform both) ─────
cat_pipeline = Pipeline([
    ("impute",  SimpleImputer(strategy="most_frequent")),
    ("encode",  OrdinalEncoder(
                    handle_unknown     = "use_encoded_value",
                    unknown_value      = -1,
                    encoded_missing_value = -2,
                )),
])

num_pipeline = Pipeline([
    # median imputation for mrc_rating, device_risk_score, ip_risk_score
    ("impute", SimpleImputer(strategy="median")),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", cat_pipeline, CAT_COLS),
        ("num", num_pipeline, NUM_COLS),
    ],
    remainder = "drop",
    verbose_feature_names_out = False,
)

# Fit on TRAIN ONLY — this is the correct pipeline discipline
preprocessor.fit(X_train_raw)

X_train_proc = preprocessor.transform(X_train_raw)
X_test_proc  = preprocessor.transform(X_test_raw)

# Recover feature names after ColumnTransformer
feature_names = CAT_COLS + NUM_COLS
print(f"\n  Preprocessed train shape          : {X_train_proc.shape}")
print(f"  Preprocessed test shape           : {X_test_proc.shape}")

# ── B.3  SMOTE — ONLY on training partition ───────────────────────────────────
#
#  k_neighbors=5: the default; generates synthetic fraud samples by
#  interpolating between each minority-class sample and its 5 nearest
#  neighbors in feature space.
#
#  sampling_strategy='auto': resamples minority class to match majority class
#  (1:1 ratio). In a later iteration you may prefer 0.3 or 0.5 to reduce the
#  risk of synthetic-sample overfitting on a small true-fraud population.

smote = SMOTE(
    sampling_strategy = "auto",
    k_neighbors       = 5,
    random_state      = RANDOM_STATE,
)

print(f"\n  Before SMOTE (train set):")
print(f"      Legitimate (0) : {(y_train==0).sum():,}")
print(f"      Fraud      (1) : {(y_train==1).sum():,}")

X_train_bal, y_train_bal = smote.fit_resample(X_train_proc, y_train)

print(f"\n  After SMOTE (train set only — X_test untouched):")
print(f"      Legitimate (0) : {(y_train_bal==0).sum():,}")
print(f"      Fraud      (1) : {(y_train_bal==1).sum():,}")
print(f"      New train size : {len(y_train_bal):,}")
print(f"\n  ✓ X_test ({X_test_proc.shape[0]:,} rows) was NEVER touched by SMOTE")
print(f"  ✓ No leakage: test fraud rate still {y_test.mean()*100:.2f}%")


# ════════════════════════════════════════════════════════════════════════════
#  STAGE C — XGBOOST TRAINING
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STAGE C │ XGBoost Training")
print("─" * 72)

# ── HYPERPARAMETER RATIONALE (see end of script for full explanation) ────────
#
#  n_estimators   = 400  — sufficient for a 30K dataset; more risks overfit
#  max_depth      = 5    — shallow trees combat overfit on synthetic SMOTE data
#  learning_rate  = 0.05 — small steps → better generalization; use with high n
#  subsample      = 0.80 — row-level bagging: each tree sees 80% of train rows
#  colsample_bytree=0.80 — feature-level bagging: 80% of features per tree
#  min_child_weight=10   — a leaf needs ≥10 samples: critical guard against
#                          overfitting to synthetic minority samples near
#                          decision boundaries created by SMOTE
#  gamma          = 0.1  — minimum loss-reduction to make a split; prunes
#                          spurious splits on low-signal features
#  scale_pos_weight=1    — set to 1 because SMOTE already balanced the
#                          training set (1:1). Using the original imbalance
#                          ratio here would DOUBLE-CORRECT and bias toward
#                          fraud recall at the expense of precision
#  reg_alpha      = 0.05 — L1 regularization: drives weak feature weights
#                          toward exactly zero (sparse model)
#  reg_lambda     = 1.5  — L2 regularization: shrinks all weights; reduces
#                          variance on the inflated SMOTE feature space
#  eval_metric    = 'aucpr'— area under Precision-Recall curve. Optimising
#                          for AUC-PR (not AUC-ROC) during training is correct
#                          for imbalanced fraud data because PR curves are not
#                          distorted by the large true-negative count

model = XGBClassifier(
    n_estimators      = 400,
    max_depth         = 5,
    learning_rate     = 0.05,
    subsample         = 0.80,
    colsample_bytree  = 0.80,
    min_child_weight  = 10,
    gamma             = 0.10,
    scale_pos_weight  = 1,        # 1 because SMOTE already balanced training data
    reg_alpha         = 0.05,
    reg_lambda        = 1.50,
    eval_metric       = "aucpr",
    use_label_encoder = False,
    tree_method       = "hist",   # faster histogram-based algorithm
    random_state      = RANDOM_STATE,
    n_jobs            = -1,
    verbosity         = 0,
)

print("  Training XGBoost …  (400 trees × depth-5 on SMOTE-balanced data)")
model.fit(
    X_train_bal, y_train_bal,
    eval_set              = [(X_test_proc, y_test)],
    verbose               = False,
)
print("  ✓ Training complete")


# ════════════════════════════════════════════════════════════════════════════
#  STAGE D — EVALUATION & EXPLAINABILITY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  STAGE D │ Evaluation on Untouched X_test")
print("─" * 72)

# ── D.1  Predictions ─────────────────────────────────────────────────────────
y_pred       = model.predict(X_test_proc)
y_pred_proba = model.predict_proba(X_test_proc)[:, 1]

# ── D.2  Core metrics ────────────────────────────────────────────────────────
roc_auc  = roc_auc_score(y_test, y_pred_proba)
pr_auc   = average_precision_score(y_test, y_pred_proba)

print(f"\n  {'─'*40}")
print(f"  ROC-AUC Score         : {roc_auc:.4f}")
print(f"  PR-AUC Score          : {pr_auc:.4f}  ← primary metric for imbalanced fraud")
print(f"  {'─'*40}")
print(f"\n  Classification Report (X_test — never seen by model during training):\n")
print(classification_report(y_test, y_pred, target_names=["Legitimate", "Fraud"],
                             digits=4))

# ── D.3  Confusion matrix values for report ──────────────────────────────────
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"  Confusion Matrix breakdown:")
print(f"      True  Negatives (TN) : {tn:,}  — correctly identified legitimate txns")
print(f"      False Positives (FP) : {fp:,}   — legitimate txns falsely flagged")
print(f"      False Negatives (FN) : {fn:,}   — MISSED fraud (critical to minimise)")
print(f"      True  Positives (TP) : {tp:,}   — correctly caught fraud")
print(f"\n  Fraud Catch Rate  (Recall)  : {tp/(tp+fn)*100:.2f}%  of all frauds caught")
print(f"  Alert Precision             : {tp/(tp+fp)*100:.2f}%  of flagged alerts are real fraud")

# ── D.4  Composite visualisation — 4-panel report figure ────────────────────
fig = plt.figure(figsize=(18, 14))
fig.suptitle("PaySense — XGBoost Fraud Detection Model\nEvaluation Report",
             fontsize=16, fontweight="bold", y=0.98)

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.38)

# Panel 1: Confusion Matrix
ax1 = fig.add_subplot(gs[0, 0])
cmd = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=["Legitimate", "Fraud"])
cmd.plot(ax=ax1, colorbar=False, cmap="Blues")
ax1.set_title("Confusion Matrix\n(X_test — 6,000 rows)", fontweight="bold", pad=10)
ax1.set_xlabel("Predicted Label", fontsize=10)
ax1.set_ylabel("True Label",      fontsize=10)

# Panel 2: ROC Curve
ax2 = fig.add_subplot(gs[0, 1])
RocCurveDisplay.from_predictions(
    y_test, y_pred_proba, ax=ax2,
    name=f"XGBoost  (AUC = {roc_auc:.4f})",
    color="#1f77b4",
)
ax2.plot([0, 1], [0, 1], "k--", lw=1, label="Random Classifier")
ax2.set_title("ROC Curve", fontweight="bold", pad=10)
ax2.legend(fontsize=9)
ax2.grid(alpha=0.3)

# Panel 3: Precision-Recall Curve
ax3 = fig.add_subplot(gs[0, 2])
PrecisionRecallDisplay.from_predictions(
    y_test, y_pred_proba, ax=ax3,
    name=f"XGBoost  (PR-AUC = {pr_auc:.4f})",
    color="#d62728",
)
baseline = y_test.mean()
ax3.axhline(baseline, color="k", linestyle="--", lw=1,
            label=f"Baseline (fraud rate = {baseline:.3f})")
ax3.set_title("Precision-Recall Curve\n(primary metric for imbalanced data)",
              fontweight="bold", pad=10)
ax3.legend(fontsize=9)
ax3.grid(alpha=0.3)

# Panel 4: XGBoost built-in Feature Importance (top 15) — weight-based
ax4 = fig.add_subplot(gs[1, :2])
plot_importance(
    model,
    ax            = ax4,
    importance_type = "gain",     # 'gain' = avg information gain per split
    max_num_features = 15,
    show_values   = True,
    title         = "Top 15 Features by Average Gain (XGBoost Built-in)",
    xlabel        = "Average Information Gain",
    height        = 0.65,
    color         = "#2ca02c",
)
ax4.set_yticklabels(
    [t.get_text().replace("num__", "").replace("cat__", "")
     for t in ax4.get_yticklabels()],
    fontsize=9,
)
ax4.grid(axis="x", alpha=0.3)

# Panel 5: Score Distribution by class
ax5 = fig.add_subplot(gs[1, 2])
scores_legit = y_pred_proba[y_test == 0]
scores_fraud = y_pred_proba[y_test == 1]
ax5.hist(scores_legit, bins=40, alpha=0.65, color="#1f77b4",
         label=f"Legitimate (n={len(scores_legit):,})", density=True)
ax5.hist(scores_fraud, bins=40, alpha=0.65, color="#d62728",
         label=f"Fraud (n={len(scores_fraud):,})", density=True)
ax5.axvline(0.5, color="black", linestyle="--", lw=1.5, label="Threshold = 0.5")
ax5.set_title("Predicted Probability\nDistribution by True Class",
              fontweight="bold", pad=10)
ax5.set_xlabel("Fraud Probability Score", fontsize=10)
ax5.set_ylabel("Density",                 fontsize=10)
ax5.legend(fontsize=9)
ax5.grid(alpha=0.3)

plt.savefig(OUTPUT_DIR + "paysense_evaluation_report.png",
            dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  ✓ Saved → paysense_evaluation_report.png")

# ── D.5  SHAP Explainability ─────────────────────────────────────────────────
#
#  SHAP (SHapley Additive exPlanations) computes each feature's CONTRIBUTION
#  to an individual prediction using game-theory-based attribution.
#  TreeExplainer is O(TLD) — exact for tree models, not an approximation.
#
#  We explain on a SAMPLE of X_test (500 rows) for speed.
#  The insights generalise because SHAP aggregates across all samples.

print("\n  Computing SHAP values …  (TreeExplainer on 500 X_test samples)")
X_test_df = pd.DataFrame(X_test_proc, columns=feature_names)

shap_sample_size = min(500, X_test_proc.shape[0])
X_shap_sample    = X_test_df.sample(n=shap_sample_size, random_state=RANDOM_STATE)

explainer   = shap.TreeExplainer(model)
shap_values = explainer(X_shap_sample)
print("  ✓ SHAP values computed")

# --- SHAP Plot 1: Beeswarm (feature impact distribution across samples) ------
fig_shap1, ax_s1 = plt.subplots(figsize=(12, 8))
shap.plots.beeswarm(
    shap_values,
    max_display   = 15,
    show          = False,
    color_bar     = True,
)
plt.title("SHAP Beeswarm — Top 15 Features\n"
          "Each dot = one transaction | Color = feature value | "
          "X-axis = impact on fraud probability",
          fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "paysense_shap_beeswarm.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved → paysense_shap_beeswarm.png")

# --- SHAP Plot 2: Bar chart (mean absolute SHAP — global importance) ---------
fig_shap2, ax_s2 = plt.subplots(figsize=(11, 7))
shap.plots.bar(
    shap_values,
    max_display = 15,
    show        = False,
    ax          = ax_s2,
)
ax_s2.set_title("SHAP Global Feature Importance\n"
                "Mean |SHAP value| — Average magnitude of impact on model output",
                fontsize=11, fontweight="bold", pad=12)
plt.tight_layout()
plt.savefig(OUTPUT_DIR + "paysense_shap_bar.png",
            dpi=150, bbox_inches="tight")
plt.close()
print("  ✓ Saved → paysense_shap_bar.png")

# ── D.6  Print top-15 SHAP feature ranking to console ────────────────────────
shap_importance = pd.DataFrame({
    "feature"          : feature_names,
    "mean_abs_shap"    : np.abs(shap_values.values).mean(axis=0),
}).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)

print(f"\n  SHAP Global Feature Importance (Top 15):")
print(f"  {'Rank':<5} {'Feature':<35} {'Mean |SHAP|':>12}")
print(f"  {'─'*55}")
for i, row in shap_importance.head(15).iterrows():
    print(f"  {i+1:<5} {row['feature']:<35} {row['mean_abs_shap']:>12.5f}")


# ════════════════════════════════════════════════════════════════════════════
#  FINAL SUMMARY
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  PAYSENSE ML PIPELINE — FINAL SUMMARY")
print("═" * 72)
print(f"""
  ┌─────────────────────────────────┬────────────────────────────────┐
  │ Dataset                         │ 30,000 rows, 41 features       │
  │ Train / Test split              │ 24,000 / 6,000  (80/20)        │
  │ SMOTE applied on                │ X_train ONLY (correct)         │
  │ Post-SMOTE train size           │ {len(y_train_bal):,}                         │
  │ X_test contamination            │ NONE                           │
  ├─────────────────────────────────┼────────────────────────────────┤
  │ ROC-AUC                         │ {roc_auc:.4f}                        │
  │ PR-AUC  ★ PRIMARY               │ {pr_auc:.4f}                        │
  │ Fraud Recall  ★ CRITICAL        │ {tp/(tp+fn)*100:.2f}%                       │
  │ Fraud Precision                 │ {tp/(tp+fp)*100:.2f}%                       │
  │ Fraud F1-Score                  │ {2*tp/(2*tp+fp+fn)*100:.2f}%                       │
  ├─────────────────────────────────┼────────────────────────────────┤
  │ True  Positives (caught fraud)  │ {tp:,}                           │
  │ False Negatives (missed fraud)  │ {fn:,}                            │
  │ False Positives (false alerts)  │ {fp:,}                           │
  └─────────────────────────────────┴────────────────────────────────┘

  Output files:
    ✓ paysense_evaluation_report.png  — 5-panel evaluation dashboard
    ✓ paysense_shap_beeswarm.png      — SHAP beeswarm (for presentation)
    ✓ paysense_shap_bar.png           — SHAP global importance bar chart
""")
print("  ✅  ML Pipeline complete. Model is ready for Phase 3 (threshold tuning).")
print("═" * 72)
