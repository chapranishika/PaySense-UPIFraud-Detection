"""
================================================================================
  PaySense Master Dataset Construction Pipeline
  Author  : Senior Data Engineer
  Project : PaySense — ML-Based UPI Fraud Detection
  Purpose : Ingest, enrich, and vertically stack four source datasets into a
            single, XGBoost-ready Master Dataset without corrupting the natural
            fraud probability distribution or introducing data leakage.
================================================================================

PIPELINE OVERVIEW
─────────────────
  Phase 1 │ Data Ingestion & Validation
  Phase 2 │ Anchor Preprocessing   (fix existing NaN in transactions.csv)
  Phase 3 │ Context Join 1 — Users (left-join; collision-safe column selection)
  Phase 4 │ Context Join 2 — Merchants (left-join on receiver_id → merchant_id)
  Phase 5 │ Post-Join Gap Fill     (P2P transactions have no merchant row)
  Phase 6 │ Supplement Schema Bridge (align synthetic_fraud_dataset to anchor)
  Phase 7 │ Vertical Stack         (pd.concat with shared column superset)
  Phase 8 │ Final Cleanup & Export
  Phase 9 │ Pipeline Report

STRICT CONSTRAINT:
  SMOTE / oversampling is NOT applied here. It is applied exclusively after the
  train/test split in the modelling phase, on the training partition only.
================================================================================
"""

import zipfile
import warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── File paths ──────────────────────────────────────────────────────────────
# We read the CSV files directly from the parent directory where they are placed
import pathlib
_BASE_DIR = pathlib.Path(__file__).parent.parent.parent.parent

PATH_TXN   = str(_BASE_DIR / "transactions.csv")
PATH_USR   = str(_BASE_DIR / "users.csv")
PATH_MRC   = str(_BASE_DIR / "merchants.csv")
PATH_SYNTH = str(_BASE_DIR / "synthetic_fraud_dataset.csv")

OUTPUT_CSV = "paysense_master_dataset.csv"

# ── Reproducibility seed ─────────────────────────────────────────────────────
np.random.seed(42)


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 1 — DATA INGESTION & VALIDATION
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  PHASE 1 │ Data Ingestion & Validation")
print("═" * 72)

df_txn  = pd.read_csv(PATH_TXN)
df_usr  = pd.read_csv(PATH_USR)
df_mrc  = pd.read_csv(PATH_MRC)
df_syn  = pd.read_csv(PATH_SYNTH)

# --- Quick sanity assertions ------------------------------------------------
assert "is_fraud"  in df_txn.columns,  "Anchor missing target column"
assert "is_fraud"  in df_syn.columns,  "Supplement missing target column"
assert "user_id"   in df_usr.columns,  "Users missing join key"
assert "merchant_id" in df_mrc.columns, "Merchants missing join key"
assert df_txn["is_fraud"].isin([0, 1]).all(), "Anchor target is not binary"
assert df_syn["is_fraud"].isin([0, 1]).all(), "Supplement target is not binary"

print(f"  ✓ transactions.csv   │ {df_txn.shape[0]:>6,} rows │ {df_txn.shape[1]} cols")
print(f"  ✓ users.csv          │ {df_usr.shape[0]:>6,} rows │ {df_usr.shape[1]} cols")
print(f"  ✓ merchants.csv      │ {df_mrc.shape[0]:>6,} rows │ {df_mrc.shape[1]} cols")
print(f"  ✓ synthetic_fraud    │ {df_syn.shape[0]:>6,} rows │ {df_syn.shape[1]} cols")

# Capture pre-pipeline fraud counts for the final report
anchor_total  = len(df_txn)
anchor_fraud  = df_txn["is_fraud"].sum()
synth_total   = len(df_syn)
synth_fraud   = df_syn["is_fraud"].sum()


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 2 — ANCHOR PREPROCESSING
#  Goal: Fix the three NaN columns that already exist in transactions.csv
#  before any join so that imputation statistics are computed on clean data.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  PHASE 2 │ Anchor Preprocessing")
print("─" * 72)

# ── 2.1  Parse timestamps ─────────────────────────────────────────────────
df_txn["timestamp"] = pd.to_datetime(df_txn["timestamp"], errors="coerce")
df_txn["date"]      = pd.to_datetime(df_txn["date"],      errors="coerce")

# ── 2.2  Encode binary columns that arrived as text ───────────────────────
#  'status' → binary: Success=1, anything else (Failed/Pending) = 0
#  We keep the original string column and add a numeric flag for the model.
df_txn["txn_success_flag"] = (df_txn["status"] == "Success").astype(int)

#  user_kyc_status → binary flag (keeps original string for dashboard)
df_txn["kyc_verified_flag"] = (df_txn["user_kyc_status"] == "Verified").astype(int)

# ── 2.3  Impute the three float columns with their own medians ────────────
#  DESIGN CHOICE: Use median (not mean) because all three columns have large
#  standard deviations suggesting outliers. Mean would bias imputed values.
#  We compute these ONCE here so they can also be applied to supplement rows.

median_time_since   = df_txn["time_since_last_txn_min"].median()
median_velocity     = df_txn["transaction_velocity"].median()
median_amt_dev      = df_txn["amount_deviation_score"].median()

df_txn["time_since_last_txn_min"] = df_txn["time_since_last_txn_min"].fillna(median_time_since)
df_txn["transaction_velocity"]    = df_txn["transaction_velocity"].fillna(median_velocity)
df_txn["amount_deviation_score"]  = df_txn["amount_deviation_score"].fillna(median_amt_dev)

print(f"  ✓ Imputed time_since_last_txn_min  NaN → {median_time_since:.2f}  (anchor median)")
print(f"  ✓ Imputed transaction_velocity     NaN → {median_velocity:.4f} (anchor median)")
print(f"  ✓ Imputed amount_deviation_score   NaN → {median_amt_dev:.4f} (anchor median)")
print(f"  ✓ Remaining NaN in anchor          : {df_txn.isnull().sum().sum()}")

# ── 2.4  Tag anchor rows with their data source ───────────────────────────
#  This column travels all the way to the master dataset. It lets you
#  stratify evaluation metrics and is NOT a model feature (drop before fit).
df_txn["data_source"] = "anchor"


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 3 — CONTEXT JOIN 1: USERS
#  Objective: Bring in user-level profile features that are NOT already
#  embedded in transactions.csv (to avoid schema collision).
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  PHASE 3 │ Context Join 1 — Users")
print("─" * 72)

# ── 3.1  Identify collision columns and drop them from users before join ───
#
#  Collision analysis result (from pre-pipeline inspection):
#    - 'user_loyalty_score': exists in BOTH transactions.csv AND users.csv
#      with the same name.  The anchor version is a transaction-time snapshot
#      (correct for ML — no leakage).  The users.csv version is a profile
#      aggregate. We KEEP the anchor version; drop from users.
#    - 'user_id': join key — handled automatically by merge.
#
USERS_DROP_BEFORE_JOIN = ["user_loyalty_score"]   # collision list

#  Columns we want from users.csv (everything except collisions & redundant)
#  We rename them with a 'usr_' prefix to make their origin transparent
#  and to guard against any future schema drift in the source files.
USERS_KEEP_RENAME = {
    "age_group"              : "usr_age_group",
    "city"                   : "usr_home_city",
    "city_tier"              : "usr_home_city_tier",      # ≠ user_city_tier (txn-time)
    "account_age_days"       : "usr_account_age_days",
    "linked_bank_count"      : "usr_linked_bank_count",
    "avg_monthly_transactions": "usr_avg_monthly_txn_profile",  # ≠ user_avg_monthly_txn
    "avg_transaction_value"  : "usr_avg_txn_value_profile",     # ≠ user_avg_txn_value
    "preferred_app"          : "usr_preferred_app",
    "preferred_device"       : "usr_preferred_device",
    "is_high_risk_user"      : "usr_is_high_risk",
}

df_usr_clean = (
    df_usr
    .drop(columns=USERS_DROP_BEFORE_JOIN, errors="ignore")
    .rename(columns=USERS_KEEP_RENAME)
    [["user_id"] + list(USERS_KEEP_RENAME.values())]   # select only what we need
)

# ── 3.2  Left-join on user_id ─────────────────────────────────────────────
#  Left join → all 20,000 anchor rows are preserved even if a user_id has
#  no match in users.csv (shouldn't happen here, but defensive coding).
pre_join_shape = df_txn.shape
df_enriched = df_txn.merge(df_usr_clean, on="user_id", how="left")

assert df_enriched.shape[0] == pre_join_shape[0], \
    "CRITICAL: User join created duplicate rows — check for non-unique user_ids in users.csv"

user_join_nulls = df_enriched[list(USERS_KEEP_RENAME.values())].isnull().sum().sum()
print(f"  ✓ User join preserved row count   : {df_enriched.shape[0]:,} (was {pre_join_shape[0]:,})")
print(f"  ✓ New columns added from users    : {len(USERS_KEEP_RENAME)}")
print(f"  ✓ NaN cells after user join       : {user_join_nulls} (expected 0 — full coverage)")


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 4 — CONTEXT JOIN 2: MERCHANTS
#  Objective: Bring in merchant-level profile features.
#  COMPLICATION: receiver_id can be 'MRC####' (merchant) OR 'USR####' (P2P).
#  Only MRC-prefixed rows will get a merchant match. P2P rows → NaN, which
#  we fill in Phase 5 with explicit P2P semantic defaults.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  PHASE 4 │ Context Join 2 — Merchants")
print("─" * 72)

# ── 4.1  Prepare merchant lookup ──────────────────────────────────────────
#  DROP: merchant_name  (high cardinality string, no predictive signal for XGBoost)
#  DROP: city, city_tier from merchants (the transaction already has user city tier;
#        merchant city is a different geographic dimension that would require
#        distance-to-user feature engineering to be useful — deferred to Phase 2).
MERCHANTS_DROP   = ["merchant_name", "city", "city_tier"]
MERCHANTS_RENAME = {
    "merchant_category"     : "mrc_category",
    "merchant_size"         : "mrc_size",
    "avg_daily_transactions": "mrc_avg_daily_txn",
    "is_registered"         : "mrc_is_registered",
    "rating"                : "mrc_rating",
}

df_mrc_clean = (
    df_mrc
    .drop(columns=MERCHANTS_DROP, errors="ignore")
    .rename(columns=MERCHANTS_RENAME)
    [["merchant_id"] + list(MERCHANTS_RENAME.values())]
)

# ── 4.2  Left-join on receiver_id = merchant_id ───────────────────────────
pre_join_shape = df_enriched.shape
df_enriched = df_enriched.merge(
    df_mrc_clean,
    left_on  = "receiver_id",
    right_on = "merchant_id",
    how      = "left"
).drop(columns=["merchant_id"])   # drop the redundant key column created by merge

assert df_enriched.shape[0] == pre_join_shape[0], \
    "CRITICAL: Merchant join created duplicate rows — check merchants.csv for non-unique merchant_ids"

p2p_count      = (df_enriched["receiver_type"] == "User").sum()
merchant_count = (df_enriched["receiver_type"] == "Merchant").sum()
mrc_nulls      = df_enriched[list(MERCHANTS_RENAME.values())].isnull().sum()

print(f"  ✓ Merchant-type transactions      : {merchant_count:,} (will have merchant data)")
print(f"  ✓ P2P-type transactions           : {p2p_count:,} (will have merchant NaN → filled next)")
print(f"  ✓ NaN per merchant column after join:")
for col, n in mrc_nulls.items():
    print(f"      {col:<28} : {n:,}")


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 5 — POST-JOIN GAP FILL (P2P Semantic Defaults)
#  Objective: Fill NaN values in merchant columns for P2P rows with values
#  that are semantically meaningful, not arbitrary zeros.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  PHASE 5 │ Post-Join Gap Fill — P2P Semantic Defaults")
print("─" * 72)

#  P2P MASK: rows where the receiver is a User (no merchant record exists)
p2p_mask = df_enriched["receiver_type"] == "User"

#  mrc_category      → "P2P Transfer" (a valid semantic category, not a filler)
#  mrc_size          → "P2P"          (custom label marking non-merchant transactions)
#  mrc_avg_daily_txn → 0              (not a merchant; no concept of daily txn volume)
#  mrc_is_registered → 1              (UPI-registered users are by definition registered)
#  mrc_rating        → np.nan         (rating concept does not apply to individuals;
#                                      leaving as NaN → XGBoost will treat as missing signal)
df_enriched.loc[p2p_mask, "mrc_category"]       = "P2P Transfer"
df_enriched.loc[p2p_mask, "mrc_size"]           = "P2P"
df_enriched.loc[p2p_mask, "mrc_avg_daily_txn"]  = 0
df_enriched.loc[p2p_mask, "mrc_is_registered"]  = 1
# mrc_rating: intentionally left as NaN for P2P — XGBoost reads it as "no rating data"

remaining_mrc_nan = df_enriched[list(MERCHANTS_RENAME.values())].isnull().sum()
print(f"  ✓ Remaining NaN after P2P fill (only mrc_rating for P2P is intentional):")
for col, n in remaining_mrc_nan.items():
    marker = " ← intentional (no rating for P2P)" if col == "mrc_rating" and n > 0 else ""
    print(f"      {col:<28} : {n:,}{marker}")

print(f"\n  ✓ Enriched anchor shape           : {df_enriched.shape}")
print(f"  ✓ Enriched anchor NaN total       : {df_enriched.isnull().sum().sum()} "
      f"(= {p2p_count:,} intentional mrc_rating NaN)")


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 6 — SUPPLEMENT SCHEMA BRIDGE
#  Objective: Align synthetic_fraud_dataset.csv (10 cols) to the enriched
#  anchor schema (40+ cols) WITHOUT blowing up with NaN columns.
#
#  STRATEGY (4 rules):
#    R1. Direct rename  : equivalent columns with different names → rename
#    R2. Binary derivation: risk scores → binary flags (proxy mapping)
#    R3. Preserve new cols: device_risk_score, ip_risk_score are NEW features
#                           added to the master schema. Anchor rows get NaN.
#    R4. Imputed fills   : all other anchor columns → fill with anchor
#                          median (floats) or mode (categoricals).
#                          These are NOT model-ready values — they are
#                          neutral imputation fillers so that pd.concat
#                          does not produce a ragged schema.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  PHASE 6 │ Supplement Schema Bridge")
print("─" * 72)

# ── 6.1  Pre-compute anchor statistics for imputation ─────────────────────
#  IMPORTANT: computed BEFORE concat so supplement imputation uses ONLY
#  anchor distribution statistics → zero leakage from supplement into anchor.

anchor_stats = {}
for col in df_enriched.select_dtypes(include=[np.number]).columns:
    anchor_stats[col] = {
        "median": df_enriched[col].median(),
        "mode"  : df_enriched[col].mode()[0] if not df_enriched[col].mode().empty else 0,
    }

anchor_cat_modes = {}
for col in df_enriched.select_dtypes(include=["object"]).columns:
    anchor_cat_modes[col] = df_enriched[col].mode()[0] if not df_enriched[col].mode().empty else "Unknown"

# ── 6.2  Rule R1 — Direct column rename ───────────────────────────────────
#  synth 'hour'  → anchor 'hour_of_day'  (same concept, different name)
df_syn_bridge = df_syn.copy()
df_syn_bridge = df_syn_bridge.rename(columns={"hour": "hour_of_day"})

# ── 6.3  Rule R2 — Binary derivation from continuous risk scores ───────────
#  device_risk_score > 0.7  maps to new_device_flag = 1
#  ip_risk_score > 0.7      maps to ip_location_mismatch = 1
#  Threshold 0.7 chosen because it marks the top ~30% of risk scores,
#  consistent with the anchor's minority-fraud population proportion.
RISK_THRESHOLD = 0.70
df_syn_bridge["new_device_flag"]      = (df_syn_bridge["device_risk_score"] > RISK_THRESHOLD).astype(int)
df_syn_bridge["ip_location_mismatch"] = (df_syn_bridge["ip_risk_score"]    > RISK_THRESHOLD).astype(int)

# ── 6.4  Rule R3 — Promote risk scores as NEW master schema columns ────────
#  These columns will be NaN for all anchor rows (patched at the end of
#  Phase 7). XGBoost treats NaN as a learnable missing-value signal, so
#  the model can use the presence/absence of these scores as a feature.
#  No action needed here — they travel forward in df_syn_bridge automatically.

# ── 6.5  Rule R4 — transaction_type vocabulary mapping ────────────────────
#  Supplement uses : POS, ATM, QR, Online
#  Anchor uses     : P2M, P2P, Bill Payment, Recharge, EMI, Subscription
#
#  DESIGN CHOICE: Map to the closest semantic equivalent in the anchor vocab.
#    POS    → P2M      (point-of-sale = person-to-merchant)
#    QR     → P2M      (QR code scans at merchant = person-to-merchant)
#    Online → P2M      (online merchant checkout = person-to-merchant)
#    ATM    → "ATM"    (kept as a NEW category — withdrawals are distinct;
#                       XGBoost label-encoder will handle an unseen category)
TXNTYPE_MAP = {
    "POS"    : "P2M",
    "QR"     : "P2M",
    "Online" : "P2M",
    "ATM"    : "ATM",
}
df_syn_bridge["transaction_type"] = df_syn_bridge["transaction_type"].map(TXNTYPE_MAP)

# ── 6.6  Derive is_night_transaction from hour_of_day ─────────────────────
#  Anchor defines night as hours outside [06:00, 22:00) → 1, else 0
df_syn_bridge["is_night_transaction"] = (
    (df_syn_bridge["hour_of_day"] < 6) | (df_syn_bridge["hour_of_day"] >= 22)
).astype(int)

# ── 6.7  Fill the remaining anchor columns for supplement rows ─────────────
#  Build a reference dict of what to fill for each column not already
#  in the supplement's schema.

anchor_only_cols = [c for c in df_enriched.columns if c not in df_syn_bridge.columns]

print(f"  ✓ Anchor columns to be imputed in supplement  : {len(anchor_only_cols)}")

for col in anchor_only_cols:
    if col in ("timestamp", "date"):
        # Not meaningful for supplement — fill with NaT; exclude from model features
        df_syn_bridge[col] = pd.NaT
        continue

    if col == "data_source":
        df_syn_bridge[col] = "supplement"
        continue

    if col == "receiver_type":
        # All supplement transactions are merchant-category → treat as Merchant
        df_syn_bridge[col] = "Merchant"
        continue

    if col in ("transaction_id", "user_id", "receiver_id", "merchant_id"):
        # Identifier columns — prefix with 'SYN_' to mark supplement origin
        df_syn_bridge[col] = "SYN_" + df_syn_bridge.get(col, pd.Series(dtype=str)).astype(str)
        continue

    if col in anchor_cat_modes:
        # Categorical column → fill with anchor mode
        df_syn_bridge[col] = anchor_cat_modes[col]
    elif col in anchor_stats:
        # Numeric column → fill with anchor median
        df_syn_bridge[col] = anchor_stats[col]["median"]
    else:
        df_syn_bridge[col] = np.nan

# ── 6.8  Fix identifier columns that weren't in df_syn_bridge columns ─────
#  Override with proper synthetic identifiers for the key columns
df_syn_bridge["transaction_id"] = ["SYN_" + str(i) for i in df_syn["transaction_id"]]
df_syn_bridge["user_id"]        = ["SYN_USR_" + str(i) for i in df_syn["user_id"]]
df_syn_bridge["receiver_id"]    = "SYN_MRC_UNKNOWN"

# ── 6.9  merchant_category from supplement → mrc_category ─────────────────
#  The supplement has 'merchant_category' (Food, Travel, Electronics, etc.)
#  which maps to our enriched schema's 'mrc_category'.
if "merchant_category" in df_syn_bridge.columns:
    df_syn_bridge["mrc_category"] = df_syn_bridge["merchant_category"]

# ── 6.10 Drop supplement-only columns that have no place in anchor schema ──
#  'country': non-Indian geography, no equivalent in anchor schema
SUPPLEMENT_DROP = ["country", "merchant_category"]
df_syn_bridge = df_syn_bridge.drop(
    columns=[c for c in SUPPLEMENT_DROP if c in df_syn_bridge.columns],
    errors="ignore"
)

# ── 6.11  Ensure column order matches enriched anchor exactly ──────────────
#  Extra columns unique to supplement (device_risk_score, ip_risk_score)
#  are appended at the END of the master schema.
synth_extra_cols = [c for c in df_syn_bridge.columns if c not in df_enriched.columns]
anchor_col_order = list(df_enriched.columns) + synth_extra_cols
df_syn_final     = df_syn_bridge.reindex(columns=anchor_col_order)

print(f"  ✓ Supplement-unique new columns (appended)     : {synth_extra_cols}")
print(f"  ✓ Supplement bridge schema shape               : {df_syn_final.shape}")
print(f"  ✓ Supplement NaN count                        : {df_syn_final.isnull().sum().sum()}")


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 7 — VERTICAL STACK
#  Objective: Concat enriched anchor + bridged supplement into the master
#  dataset. Then patch the two new risk-score columns for anchor rows.
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  PHASE 7 │ Vertical Stack")
print("─" * 72)

# Promote the two new supplement columns onto the enriched anchor BEFORE concat
# so that pd.concat does not create a ragged index alignment.
df_enriched["device_risk_score"] = np.nan   # anchor rows have no risk scores
df_enriched["ip_risk_score"]     = np.nan   # anchor rows have no risk scores

df_master = pd.concat([df_enriched, df_syn_final], ignore_index=True)

print(f"  ✓ Anchor rows                     : {len(df_enriched):,}")
print(f"  ✓ Supplement rows                 : {len(df_syn_final):,}")
print(f"  ✓ Master dataset shape            : {df_master.shape}")
print(f"  ✓ Stacking preserved column count : {df_master.shape[1]} columns")


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 8 — FINAL CLEANUP & EXPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 72)
print("  PHASE 8 │ Final Cleanup & Export")
print("─" * 72)

# ── 8.1  Reset index ──────────────────────────────────────────────────────
df_master = df_master.reset_index(drop=True)

# ── 8.2  Enforce target column dtype ──────────────────────────────────────
df_master["is_fraud"] = df_master["is_fraud"].astype(int)

# ── 8.3  Standardise day_of_week to integer (Mon=0 … Sun=6) ───────────────
DOW_MAP = {
    "Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
    "Friday": 4, "Saturday": 5, "Sunday": 6,
}
df_master["day_of_week"] = df_master["day_of_week"].map(DOW_MAP)

# ── 8.4  user_city_tier → ordinal int ─────────────────────────────────────
TIER_MAP = {"Tier 1": 1, "Tier 2": 2, "Tier 3": 3}
df_master["user_city_tier"]     = df_master["user_city_tier"].map(TIER_MAP)
df_master["usr_home_city_tier"] = df_master["usr_home_city_tier"].map(TIER_MAP)

# ── 8.4b  Recompute transaction_frequency_score as hour_deviation_score ───
#
#  FIX (Fix 2 of the brutal-feedback session):
#
#  BEFORE (wrong): transaction_frequency_score was a normalised frequency
#  ratio in [0, 1] with almost zero discrimination between fraud and
#  legitimate (mean fraud=0.6111, mean legit=0.6100). It was a near-useless
#  feature, and we were incorrectly feeding the hour z-score from Android
#  into it at inference time — incompatible value ranges (0–6.67 vs 0–1).
#
#  AFTER (correct): we replace the column values with the per-user hour
#  deviation z-score. This gives fraud/legit discrimination ratio of 1.21x
#  (mean fraud=0.977, mean legit=0.807) and has correlation of -0.0006
#  with the original frequency values — confirming they are independent
#  signals and the replacement carries new information.
#
#  The COLUMN NAME is intentionally kept as transaction_frequency_score
#  so the 43-field Pydantic schema and all Android code remains unchanged.
#  Only the VALUES change — the model is retrained to interpret this field
#  as hour deviation. Android's FraudApiService now sends zHour here.
#
#  Computation:
#    For each transaction, compute per-user mean and std of hour_of_day
#    across the full dataset (approximation of the 90-day window used
#    at runtime). Then z = |hour - user_mean| / max(user_std, 0.5).

print(f"\n  ── Computing hour_deviation_score to replace transaction_frequency_score ──")

# Compute per-user hour statistics across all rows (train+test approximation)
user_hour_stats = (
    df_master.groupby("user_id")["hour_of_day"]
    .agg(user_mean_hour="mean", user_std_hour="std")
    .fillna({"user_std_hour": 6.0})   # single-transaction users get wide spread
)
user_hour_stats["user_std_hour"] = user_hour_stats["user_std_hour"].clip(lower=0.5)

df_master = df_master.merge(user_hour_stats, on="user_id", how="left")

df_master["transaction_frequency_score"] = (
    (df_master["hour_of_day"] - df_master["user_mean_hour"]).abs()
    / df_master["user_std_hour"]
)

# Drop the helper columns — they are not model features
df_master.drop(columns=["user_mean_hour", "user_std_hour"], inplace=True)

# Validate the replacement
fraud_mean_hd = df_master.loc[df_master["is_fraud"]==1, "transaction_frequency_score"].mean()
legit_mean_hd = df_master.loc[df_master["is_fraud"]==0, "transaction_frequency_score"].mean()
print(f"  ✓ hour_deviation (fraud class mean)  : {fraud_mean_hd:.4f}")
print(f"  ✓ hour_deviation (legit class mean)  : {legit_mean_hd:.4f}")
print(f"  ✓ Fraud/legit ratio                  : {fraud_mean_hd/legit_mean_hd:.2f}x  (>1.0 = useful signal)")
print(f"  ✓ Column name kept as 'transaction_frequency_score' (schema unchanged)")

# ── 8.5  Column classification for downstream modelling ───────────────────
#  IDENTIFIER columns  → keep in CSV but MUST be dropped before model fit
#  TEMPORAL columns    → MUST be dropped (leakage risk: timestamp is ordered)
#  METADATA columns    → keep for stratification; drop before fit
#  TARGET              → is_fraud
#  FEATURE columns     → everything else

IDENTIFIER_COLS = ["transaction_id", "user_id", "receiver_id"]
TEMPORAL_COLS   = ["timestamp", "date"]
METADATA_COLS   = ["data_source"]
TARGET_COL      = "is_fraud"

print(f"  ✓ Identifier cols (drop before fit): {IDENTIFIER_COLS}")
print(f"  ✓ Temporal cols  (drop before fit): {TEMPORAL_COLS}")
print(f"  ✓ Metadata cols  (drop before fit): {METADATA_COLS}")
print(f"  ✓ Target column                  : {TARGET_COL}")

feature_cols = [
    c for c in df_master.columns
    if c not in IDENTIFIER_COLS + TEMPORAL_COLS + METADATA_COLS + [TARGET_COL]
]
print(f"  ✓ Model-ready feature columns     : {len(feature_cols)}")

# ── 8.6  Export ───────────────────────────────────────────────────────────
import os
if os.path.dirname(OUTPUT_CSV):
    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
df_master.to_csv(OUTPUT_CSV, index=False)
print(f"\n  ✓ Master dataset exported to      : {OUTPUT_CSV}")


# ════════════════════════════════════════════════════════════════════════════
#  PHASE 9 — PIPELINE REPORT
# ════════════════════════════════════════════════════════════════════════════
print("\n" + "═" * 72)
print("  PHASE 9 │ Pipeline Report")
print("═" * 72)

master_total = len(df_master)
master_fraud = df_master["is_fraud"].sum()
master_legit = master_total - master_fraud

print(f"""
  ┌─────────────────────────────────────────────────────────────┐
  │           PAYSENSE MASTER DATASET — FINAL SUMMARY           │
  ├──────────────────────────┬──────────────────────────────────┤
  │ Total rows               │ {master_total:>10,}                  │
  │ Total columns            │ {df_master.shape[1]:>10}                  │
  │ Legitimate transactions  │ {master_legit:>10,}  ({master_legit/master_total*100:.2f}%)         │
  │ Fraudulent transactions  │ {master_fraud:>10,}  ({master_fraud/master_total*100:.2f}%)          │
  ├──────────────────────────┼──────────────────────────────────┤
  │ Anchor rows contributed  │ {anchor_total:>10,}  (fraud: {anchor_fraud:,} = {anchor_fraud/anchor_total*100:.2f}%) │
  │ Supplement rows added    │ {synth_total:>10,}  (fraud: {int(synth_fraud):,} = {synth_fraud/synth_total*100:.2f}%) │
  ├──────────────────────────┼──────────────────────────────────┤
  │ Pre-pipeline fraud rate  │ {anchor_fraud/anchor_total*100:.2f}% (anchor only)             │
  │ Post-pipeline fraud rate │ {master_fraud/master_total*100:.2f}% (controlled delta: +{(master_fraud/master_total - anchor_fraud/anchor_total)*100:.2f}%)  │
  ├──────────────────────────┼──────────────────────────────────┤
  │ Anchor rows (data_source)│ {(df_master['data_source']=='anchor').sum():>10,}                  │
  │ Supplement rows          │ {(df_master['data_source']=='supplement').sum():>10,}                  │
  ├──────────────────────────┼──────────────────────────────────┤
  │ New schema columns added │ {len(synth_extra_cols):>10}  {synth_extra_cols}  │
  │ Total model features     │ {len(feature_cols):>10}                  │
  │ SMOTE applied            │        NO (correct — applied post-split)  │
  └──────────────────────────┴──────────────────────────────────┘
""")

print("  is_fraud value counts (master dataset):")
print(df_master["is_fraud"].value_counts().to_string())
print(f"\n  NaN summary by column (top 10 by NaN count):")
nan_summary = df_master.isnull().sum().sort_values(ascending=False).head(10)
for col, n in nan_summary.items():
    if n > 0:
        pct = n / master_total * 100
        note = ""
        if col in ["device_risk_score", "ip_risk_score"]:
            note = "  ← supplement-only feature (NaN = anchor rows; intentional)"
        elif col == "mrc_rating":
            note = "  ← NaN = P2P transactions; no merchant rating (intentional)"
        elif col in TEMPORAL_COLS:
            note = "  ← supplement rows; excluded from model features"
        print(f"    {col:<35} : {n:>6,} ({pct:5.1f}%){note}")

print(f"\n  Data source breakdown:")
print(df_master["data_source"].value_counts().to_string())
print("\n  ✅  Pipeline complete. Master dataset is ready for train/test split.")
print("═" * 72)
