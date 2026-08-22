"""
================================================================================
  generate_grounded_synthetic_dataset.py
  ────────────────────────────────────────────────────────────────────────────
  Builds an INDEPENDENTLY-GENERATED synthetic dataset carrying the FULL
  40-feature schema the frozen PaySense model expects (all columns present
  in artefacts/paysense_feature_names.pkl), so it can be scored through
  src.fraud_model.score() without the "85% of features missing" confound
  that limited GENERALIZATION_CHECK.md's two external datasets (only 6/40
  and 2/40 features honestly mappable there).

  WHY THIS IS A GENUINELY SEPARATE DRAW, NOT A RE-RUN OF paysense_pipeline.py
  ────────────────────────────────────────────────────────────────────────────
  1. Different RNG seed (SEED below) vs. paysense_pipeline.py's np.random.seed(42)
     and paysense_ml_pipeline.py's RANDOM_STATE=42.
  2. Different row count (25,000 vs. the master dataset's 30,000).
  3. Different generative *structure*, not just different random draws of the
     same structure:
       - The original pipeline builds the anchor from pre-existing
         transactions.csv/users.csv/merchants.csv, blends in a 10k-row
         "supplement" via a rule-based schema bridge, and derives
         new_device_flag/ip_location_mismatch from continuous
         device_risk_score/ip_risk_score via a hard 0.70 threshold
         (paysense_pipeline.py lines ~318-324).
       - This script instead samples the two binary flags directly as
         independent low-probability events, THEN derives the continuous
         device_risk_score/ip_risk_score FROM those flags (opposite causal
         direction — see _derive_risk_scores below).
       - is_fraud here comes from a single calibrated logistic risk model
         over 9 standardized signals + Gaussian noise (see
         _simulate_fraud_label), not from a two-source anchor/supplement
         blend. This is a different correlation structure between features
         and the label by construction.
       - amount_deviation_score is computed here as an actual per-row
         z-score-like statistic against the user's own profile average
         (|amount - user_avg_txn_value| / scale), not sampled independently.

  GROUNDING — every distribution below cites either:
    (a) data_dictionary.csv (the authoritative schema spec, E:\\Projects\\upi\\
        data_dictionary.csv) for columns where it gives an exact rule/rate, or
    (b) an explicitly-labelled ASSUMPTION where the dictionary gives only a
        range/category list without a shape or baseline rate. Assumptions are
        never presented as if they were cited figures.

  FRAUD RATE — DELIBERATE ENRICHMENT, NOT REAL-WORLD PREVALENCE
  ────────────────────────────────────────────────────────────────────────────
  Real UPI fraud rate by transaction count (government figures, see
  SYNTHETIC_GROUNDING.md §1 for full derivation and sources):
      FY24: 13.42 lakh incidents / 131.1 billion txns ≈ 0.00102%
      FY25: 12.64 lakh incidents / 185.9 billion txns ≈ 0.00068%
  That is ~3-4 orders of magnitude below any usable positive-class sample
  size. TARGET_FRAUD_RATE below is deliberately set to 4.0% — in the same
  band as data_dictionary.csv's documented "~3.8% positive rate" for
  transactions.csv and paysense_ml_pipeline.py's actual master-dataset rate
  (~4.21%, per its own Stage-B log message) — NOT because 4% reflects
  real-world prevalence (it is ~5,000-6,000x the real FY25 population rate),
  but because this dataset exists to test "does the model generalize past
  its own training generation process," which requires enough positive
  examples to measure recall/precision at all. This choice is made
  deliberately and stated here, not mixed silently with a population-rate
  claim. See SYNTHETIC_GROUNDING.md for the full argument either way.

  Run
  ───
      venv\\Scripts\\python.exe generate_grounded_synthetic_dataset.py
================================================================================
"""

from __future__ import annotations

import io
import os
import sys
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_CSV = os.path.join(BASE_DIR, "synthetic_grounded_dataset.csv")

# ── Independent generation parameters (deliberately different from the
#    original pipeline's SEED=42 / N=30,000) ────────────────────────────────
SEED = 918273
N_USERS = 3000
N_MERCHANTS = 400
N_TXN = 25000
TARGET_FRAUD_RATE = 0.040  # see docstring above — deliberate enrichment

# ── Grounded categorical vocabularies (data_dictionary.csv) ────────────────
AGE_GROUPS = ["18-24", "25-34", "35-44", "45-54", "55+"]
AGE_WEIGHTS = [0.22, 0.33, 0.22, 0.14, 0.09]  # ASSUMPTION: dict gives buckets, not shares

APPS = ["GPay", "PhonePe", "Paytm", "BHIM", "Amazon Pay", "WhatsApp Pay"]
APP_WEIGHTS = [0.34, 0.32, 0.19, 0.06, 0.05, 0.04]  # ASSUMPTION: ordinal ranking only

DEVICES = ["Android", "iOS", "Web"]
DEVICE_WEIGHTS = [0.75, 0.20, 0.05]  # ASSUMPTION

MERCHANT_CATS = [
    "Grocery", "Food & Dining", "Travel", "Shopping", "Entertainment",
    "Bill Payment", "Healthcare", "Education", "Fuel", "Electronics", "Others",
]  # 11 categories, per data_dictionary.csv merchants.csv row ("11 categories total")

TXN_TYPES = ["P2P", "P2M", "Bill Payment", "Recharge", "EMI", "Subscription"]
TXN_TYPE_WEIGHTS = [0.38, 0.34, 0.12, 0.08, 0.05, 0.03]  # ASSUMPTION

# Cities — independent list from the original pipeline's (undocumented) 38.
# Tier assignment is illustrative of real Indian city-tier classification,
# not copied from any specific source file.
TIER1_CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
TIER2_CITIES = ["Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna",
                 "Vadodara", "Coimbatore", "Visakhapatnam", "Surat", "Ranchi"]
TIER3_CITIES = ["Jodhpur", "Mysuru", "Guwahati", "Raipur", "Amritsar", "Dehradun",
                 "Shimla", "Siliguri", "Jabalpur", "Rajkot", "Nashik"]
CITY_TIER_WEIGHTS = [0.45, 0.35, 0.20]  # ASSUMPTION


def _tier_to_int(tier_str):
    """
    'Tier 1'/'Tier 2'/'Tier 3' -> 1/2/3. Like day_of_week, data_dictionary.csv
    documents user_city_tier/usr_home_city_tier as STRINGS, but
    artefacts/paysense_preprocessor.pkl's fitted ColumnTransformer routes
    both through its 31-column NUMERIC pipeline (verified: paysense_master_
    dataset.csv stores them as int64 {1,2,3}), so the frozen preprocessor
    would median-impute-crash on the string form. Numeric is what the frozen
    model actually expects; recorded as another documented dictionary/
    artefact mismatch, not silently smoothed over.
    """
    return np.array([int(str(t).split()[-1]) for t in tier_str])


def _weighted_choice(rng, options, weights, size):
    return rng.choice(options, size=size, p=np.array(weights) / np.sum(weights))


def _sample_cities(rng, size):
    """Sample (city, city_tier) pairs consistent with CITY_TIER_WEIGHTS."""
    tiers = _weighted_choice(rng, ["Tier 1", "Tier 2", "Tier 3"], CITY_TIER_WEIGHTS, size)
    pools = {"Tier 1": TIER1_CITIES, "Tier 2": TIER2_CITIES, "Tier 3": TIER3_CITIES}
    cities = np.array([rng.choice(pools[t]) for t in tiers])
    return cities, tiers


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 1 — USER PROFILE TABLE  (grounds data_dictionary.csv users.csv rows)
# ════════════════════════════════════════════════════════════════════════════
def _generate_users(rng, n_users):
    user_id = np.array([f"SGU{idx:06d}" for idx in range(n_users)])
    age_group = _weighted_choice(rng, AGE_GROUPS, AGE_WEIGHTS, n_users)
    city, city_tier = _sample_cities(rng, n_users)

    # kyc_status: ~87% Verified — data_dictionary.csv users.csv row (exact figure)
    kyc_status = np.where(rng.random(n_users) < 0.87, "Verified", "Not Verified")

    # account_age_days: 30-2500 (dict range). ASSUMPTION on shape: triangular,
    # skewed toward younger accounts (mode near 300 days) — no shape given by dict.
    account_age_days = rng.triangular(30, 300, 2500, n_users).round().astype(int)

    # linked_bank_count: ~50% have only 1 (dict exact figure); remaining 50%
    # split roughly evenly across 2/3/4 (ASSUMPTION on the split of that 50%).
    one_bank = rng.random(n_users) < 0.50
    other = rng.choice([2, 3, 4], size=n_users)
    linked_bank_count = np.where(one_bank, 1, other)

    # avg_monthly_transactions: Tier 1 users average ~45/month (dict exact
    # figure); Tier 2/3 means are an ASSUMPTION (no figure given for them).
    tier_mean = np.select(
        [city_tier == "Tier 1", city_tier == "Tier 2", city_tier == "Tier 3"],
        [45.0, 25.0, 12.0],
    )
    avg_monthly_transactions = np.clip(
        rng.gamma(shape=3.0, scale=tier_mean / 3.0), 1, None
    ).round().astype(int)

    # avg_transaction_value: 50-10000, log-normal (dict exact distribution family)
    avg_transaction_value = np.clip(
        rng.lognormal(mean=np.log(500), sigma=0.9, size=n_users), 50, 10000
    )

    preferred_app = _weighted_choice(rng, APPS, APP_WEIGHTS, n_users)
    preferred_device = _weighted_choice(rng, DEVICES, DEVICE_WEIGHTS, n_users)

    # user_loyalty_score: 0.1-1.0 (dict range). ASSUMPTION: beta(2,2) rescaled.
    user_loyalty_score = 0.1 + 0.9 * rng.beta(2, 2, n_users)

    # is_high_risk_user: ~5% overall (dict exact figure), with a mild tilt
    # toward unverified-KYC users, rescaled so the population mean stays 5%.
    p_raw = np.where(kyc_status == "Verified", 0.03, 0.15)
    p_highrisk = np.clip(p_raw * (0.05 / p_raw.mean()), 0, 1)
    is_high_risk_user = (rng.random(n_users) < p_highrisk).astype(int)

    return pd.DataFrame({
        "user_id": user_id,
        "usr_age_group": age_group,
        "usr_home_city": city,
        "usr_home_city_tier": city_tier,
        "usr_kyc_status": kyc_status,
        "usr_account_age_days": account_age_days,
        "usr_linked_bank_count": linked_bank_count,
        "usr_avg_monthly_txn_profile": avg_monthly_transactions,
        "usr_avg_txn_value_profile": avg_transaction_value,
        "usr_preferred_app": preferred_app,
        "usr_preferred_device": preferred_device,
        "usr_loyalty_score_profile": user_loyalty_score,
        "usr_is_high_risk": is_high_risk_user,
    })


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 2 — MERCHANT PROFILE TABLE  (grounds merchants.csv rows)
# ════════════════════════════════════════════════════════════════════════════
def _generate_merchants(rng, n_merchants):
    merchant_id = np.array([f"SGM{idx:05d}" for idx in range(n_merchants)])
    merchant_category = rng.choice(MERCHANT_CATS, size=n_merchants)

    # merchant_size: ~55% Small (dict exact figure); Medium/Enterprise split
    # is an ASSUMPTION for the remaining 45%.
    size_roll = rng.random(n_merchants)
    merchant_size = np.select(
        [size_roll < 0.55, size_roll < 0.85], ["Small", "Medium"], default="Enterprise"
    )
    city, city_tier = _sample_cities(rng, n_merchants)

    # avg_daily_transactions: Enterprise merchants process most (dict exact
    # qualitative statement). Per-size means are an ASSUMPTION.
    size_mean = np.select(
        [merchant_size == "Small", merchant_size == "Medium"], [5.0, 40.0], default=250.0
    )
    avg_daily_transactions = np.clip(
        rng.gamma(shape=2.0, scale=size_mean / 2.0), 1, None
    ).round().astype(int)

    # is_registered: ~94% (dict exact figure), mild tilt by size, rescaled.
    p_raw = np.select(
        [merchant_size == "Enterprise", merchant_size == "Medium"], [0.99, 0.96], default=0.90
    )
    p_reg = np.clip(p_raw * (0.94 / p_raw.mean()), 0, 1)
    is_registered = (rng.random(n_merchants) < p_reg).astype(int)

    # rating: 2.5-5.0 (dict range), skewed high — ASSUMPTION on shape.
    rating = 2.5 + 2.5 * rng.beta(5, 2, n_merchants)

    return pd.DataFrame({
        "receiver_id": merchant_id,
        "mrc_category": merchant_category,
        "mrc_size": merchant_size,
        "mrc_city": city,
        "mrc_city_tier": city_tier,
        "mrc_avg_daily_txn": avg_daily_transactions,
        "mrc_is_registered": is_registered,
        "mrc_rating": rating,
    })


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 3 — RAW/CONTINUOUS RISK SCORES DERIVED FROM BINARY FLAGS
#  (reverse of paysense_pipeline.py's direction: there, continuous scores
#   >0.70 DEFINE the binary flags. Here, the binary flags are sampled first
#   as independent low-probability events, and the continuous scores are
#   generated to be CONSISTENT WITH the flags — opposite causal order.)
# ════════════════════════════════════════════════════════════════════════════
def _derive_risk_scores(rng, flag):
    n = len(flag)
    high = rng.beta(8, 2, n)   # concentrated toward 1.0
    low = rng.beta(2, 8, n)    # concentrated toward 0.0
    return np.where(flag == 1, high, low)


# ════════════════════════════════════════════════════════════════════════════
#  STAGE 4 — FRAUD LABEL: CALIBRATED LOGISTIC RISK MODEL
# ════════════════════════════════════════════════════════════════════════════
def _simulate_fraud_label(rng, signals: dict, target_rate: float):
    """
    signals: dict of raw (unstandardized) per-row numpy arrays for the 9
    fraud-relevant inputs. Weights are a documented, hand-set ASSUMPTION
    (there is no ground-truth coefficient vector to cite), chosen only to
    (a) push fraud probability up with each documented "key fraud signal"
    and (b) avoid the near-deterministic separability that
    GENERALIZATION_CHECK.md flagged as disqualifying in its rejected
    Dataset 2 (§2.2) — verified post-hoc in SYNTHETIC_GROUNDING.md.
    """
    n = len(signals["new_device_flag"])
    weights = {
        "new_device_flag": 1.8,
        "ip_location_mismatch": 1.4,
        "failed_attempts_scaled": 1.2,   # failed_attempts_last_24h / 5
        "amount_deviation_scaled": 1.5,  # amount_deviation_score / 10
        "is_night_transaction": 0.5,
        "usr_is_high_risk": 1.6,
        "kyc_unverified": 1.0,           # 1 - kyc_verified_flag
        "new_account": 0.8,              # account_age_days < 30
        "velocity_scaled": 1.0,          # transaction_velocity / 4
    }
    logit0 = np.zeros(n)
    for key, w in weights.items():
        logit0 += w * signals[key]
    noise = rng.normal(0, 0.6, size=n)
    logit0 = logit0 + noise

    # Binary-search an intercept so mean(sigmoid(logit0 + c)) == target_rate.
    lo, hi = -20.0, 10.0
    for _ in range(60):
        mid = (lo + hi) / 2
        p = 1 / (1 + np.exp(-(logit0 + mid)))
        if p.mean() > target_rate:
            hi = mid
        else:
            lo = mid
    intercept = (lo + hi) / 2
    p_final = 1 / (1 + np.exp(-(logit0 + intercept)))
    is_fraud = (rng.random(n) < p_final).astype(int)
    return is_fraud, p_final


# ════════════════════════════════════════════════════════════════════════════
#  MAIN GENERATOR
# ════════════════════════════════════════════════════════════════════════════
def generate(
    n_txn: int = N_TXN,
    n_users: int = N_USERS,
    n_merchants: int = N_MERCHANTS,
    seed: int = SEED,
    target_fraud_rate: float = TARGET_FRAUD_RATE,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    users = _generate_users(rng, n_users)
    merchants = _generate_merchants(rng, n_merchants)

    # ── Sample sender users, weighted by activity (more active users
    #    generate proportionally more transactions) ─────────────────────────
    activity_w = users["usr_avg_monthly_txn_profile"].to_numpy(dtype=float)
    activity_w = activity_w / activity_w.sum()
    sender_idx = rng.choice(len(users), size=n_txn, p=activity_w)
    sender = users.iloc[sender_idx].reset_index(drop=True)

    # ── transaction_type drives receiver_type (P2P -> User, else Merchant) ──
    transaction_type = _weighted_choice(rng, TXN_TYPES, TXN_TYPE_WEIGHTS, n_txn)
    receiver_type = np.where(transaction_type == "P2P", "User", "Merchant")

    # Merchant receivers, weighted by avg_daily_transactions (busier
    # merchants receive proportionally more transactions).
    mrc_w = merchants["mrc_avg_daily_txn"].to_numpy(dtype=float)
    mrc_w = mrc_w / mrc_w.sum()
    mrc_pick_idx = rng.choice(len(merchants), size=n_txn, p=mrc_w)
    mrc_pick = merchants.iloc[mrc_pick_idx].reset_index(drop=True)

    # User receivers: a different random user than the sender.
    user_pick_idx = rng.integers(0, len(users), size=n_txn)
    same_as_sender = user_pick_idx == sender_idx
    user_pick_idx[same_as_sender] = (user_pick_idx[same_as_sender] + 1) % len(users)
    user_receiver_id = users["user_id"].to_numpy()[user_pick_idx]

    receiver_id = np.where(receiver_type == "Merchant", mrc_pick["receiver_id"].to_numpy(), user_receiver_id)

    # ── amount: log-normal centered on the sender's own profile average,
    #    capped at 100,000 — data_dictionary.csv exact distribution + cap ────
    amount = np.clip(
        rng.lognormal(
            mean=np.log(np.clip(sender["usr_avg_txn_value_profile"].to_numpy(), 50, None)),
            sigma=0.7,
            size=n_txn,
        ),
        1, 100000,
    )

    # ── timestamp: full calendar year 2025, hour-of-day weighted toward
    #    daytime/evening (ASSUMPTION on shape — no distribution given) ───────
    hour_weights = np.array([
        0.6, 0.4, 0.3, 0.3, 0.4, 0.8,     # 0-5
        1.8, 2.6, 3.4, 3.6, 3.2, 3.0,     # 6-11
        3.8, 3.6, 3.0, 2.8, 2.9, 3.3,     # 12-17
        4.2, 4.6, 4.0, 3.2, 2.2, 1.2,     # 18-23
    ])
    hour_of_day = _weighted_choice(rng, np.arange(24), hour_weights, n_txn).astype(int)
    day_offset = rng.integers(0, 365, size=n_txn)
    base_date = datetime(2025, 1, 1)
    dates = [base_date + timedelta(days=int(d)) for d in day_offset]
    minute = rng.integers(0, 60, size=n_txn)
    second = rng.integers(0, 60, size=n_txn)
    timestamp = [
        d.replace(hour=int(h), minute=int(m), second=int(s))
        for d, h, m, s in zip(dates, hour_of_day, minute, second)
    ]
    # day_of_week: data_dictionary.csv documents this as a STRING ("Monday",
    # "Saturday", ...), but the frozen artefacts do not match that. Inspecting
    # artefacts/paysense_preprocessor.pkl's fitted ColumnTransformer shows
    # day_of_week routed through the NUMERIC pipeline (31 num cols), and
    # paysense_master_dataset.csv itself stores it as int64 matching pandas'
    # `.dt.dayofweek` convention (Monday=0 ... Sunday=6, verified: rows with
    # day_of_week in {5,6} are exactly the rows with is_weekend=1). This is a
    # real, honest discrepancy between the documented schema and what the
    # model was actually trained on — recorded here and in
    # SYNTHETIC_GROUNDING.md rather than silently "fixed" to match the
    # dictionary, because matching the FROZEN model's real expectations is
    # what makes this dataset scoreable at all.
    day_of_week_name = np.array([t.strftime("%A") for t in timestamp])
    is_weekend = np.isin(day_of_week_name, ["Saturday", "Sunday"]).astype(int)
    day_of_week = np.array([t.weekday() for t in timestamp])  # Monday=0 ... Sunday=6
    # is_night_transaction: 1 = hour<6 or hour>=22 — data_dictionary.csv exact rule
    is_night_transaction = ((hour_of_day < 6) | (hour_of_day >= 22)).astype(int)

    # ── time_since_last_txn_min: exponential, mean inversely tied to the
    #    sender's monthly activity (ASSUMPTION on shape); ~2% missing
    #    (dict exact figure, standing in for "first transaction") ───────────
    mean_gap_min = 43200.0 / np.clip(sender["usr_avg_monthly_txn_profile"].to_numpy(), 1, None)
    time_since_last_txn_min = rng.exponential(mean_gap_min)
    missing_gap = rng.random(n_txn) < 0.02
    time_since_last_txn_min = np.where(missing_gap, np.nan, time_since_last_txn_min)

    # ── payment_app / device_type: usually the sender's preferred choice,
    #    occasionally different (ASSUMPTION: 85%/90% "sticky" rates) ────────
    payment_app = np.where(
        rng.random(n_txn) < 0.85, sender["usr_preferred_app"].to_numpy(),
        _weighted_choice(rng, APPS, APP_WEIGHTS, n_txn),
    )
    device_type = np.where(
        rng.random(n_txn) < 0.90, sender["usr_preferred_device"].to_numpy(),
        _weighted_choice(rng, DEVICES, DEVICE_WEIGHTS, n_txn),
    )
    # new_device_flag: transaction initiated from an unrecognized device —
    # here that's a real, meaningful condition (device != preferred device),
    # not a threshold on an unrelated continuous score.
    new_device_flag = (device_type != sender["usr_preferred_device"].to_numpy()).astype(int)

    # ip_location_mismatch: baseline low-probability independent event
    # (ASSUMPTION: no baseline rate given by the dictionary beyond "key
    # fraud signal"); 8% chosen as a plausible minority-event rate.
    ip_location_mismatch = (rng.random(n_txn) < 0.08).astype(int)

    # ── status: ~88% Success / ~9% Failed / ~3% Pending — dict exact figures
    status = _weighted_choice(rng, ["Success", "Failed", "Pending"], [0.88, 0.09, 0.03], n_txn)
    txn_success_flag = (status == "Success").astype(int)

    # ── failed_attempts_last_24h: ~72% are 0 (dict exact figure); remaining
    #    28% distributed 1-5 with decreasing weight (ASSUMPTION on shape) ───
    fail_roll = rng.random(n_txn)
    failed_attempts_last_24h = np.zeros(n_txn, dtype=int)
    cum = 0.72
    for val, share in [(1, 0.12), (2, 0.08), (3, 0.04), (4, 0.025), (5, 0.015)]:
        nxt = cum + share
        failed_attempts_last_24h = np.where((fail_roll >= cum) & (fail_roll < nxt), val, failed_attempts_last_24h)
        cum = nxt

    # ── transaction_velocity: 0-4 (dict range); ~2% missing (dict exact) ────
    transaction_velocity = np.clip(rng.poisson(0.6, n_txn), 0, 4).astype(float)
    missing_vel = rng.random(n_txn) < 0.02
    transaction_velocity_out = np.where(missing_vel, np.nan, transaction_velocity)

    # ── recurring_payment_flag: 1 for Bill Payment/Subscription/EMI — dict exact rule
    recurring_payment_flag = np.isin(transaction_type, ["Bill Payment", "Subscription", "EMI"]).astype(int)

    # ── balance_after_transaction: synthetic 500-80000 (dict exact range,
    #    explicitly "for modeling purposes only") ────────────────────────────
    balance_after_transaction = rng.uniform(500, 80000, n_txn)

    # ── transaction_frequency_score: EXACT formula from data_dictionary.csv
    #    ("user_avg_monthly_txn / 50, capped at 1") ──────────────────────────
    user_avg_monthly_txn = sender["usr_avg_monthly_txn_profile"].to_numpy()
    transaction_frequency_score = np.clip(user_avg_monthly_txn / 50.0, 0, 1)

    # ── amount_deviation_score: REAL per-row deviation from the user's own
    #    profile average (not sampled independently), clipped to [0,10];
    #    ~2% missing (dict exact figure) ──────────────────────────────────────
    user_avg_txn_value = sender["usr_avg_txn_value_profile"].to_numpy()
    amount_deviation_score = np.clip(
        np.abs(amount - user_avg_txn_value) / (user_avg_txn_value * 0.5 + 50), 0, 10
    )
    missing_dev = rng.random(n_txn) < 0.02
    amount_deviation_score_out = np.where(missing_dev, np.nan, amount_deviation_score)

    kyc_verified_flag = (sender["usr_kyc_status"].to_numpy() == "Verified").astype(int)

    # ── device_risk_score / ip_risk_score — derived FROM the binary flags
    #    (reverse causal direction vs. the original pipeline; see Stage 3) ───
    device_risk_score = _derive_risk_scores(rng, new_device_flag)
    ip_risk_score = _derive_risk_scores(rng, ip_location_mismatch)

    # ── FRAUD LABEL — calibrated logistic risk model (Stage 4) ─────────────
    signals = {
        "new_device_flag": new_device_flag.astype(float),
        "ip_location_mismatch": ip_location_mismatch.astype(float),
        "failed_attempts_scaled": failed_attempts_last_24h.astype(float) / 5.0,
        "amount_deviation_scaled": amount_deviation_score / 10.0,  # pre-missing values, for internal calc only
        "is_night_transaction": is_night_transaction.astype(float),
        "usr_is_high_risk": sender["usr_is_high_risk"].to_numpy(dtype=float),
        "kyc_unverified": 1.0 - kyc_verified_flag.astype(float),
        "new_account": (sender["usr_account_age_days"].to_numpy() < 30).astype(float),
        "velocity_scaled": transaction_velocity / 4.0,  # pre-missing values, for internal calc only
    }
    is_fraud, fraud_prob = _simulate_fraud_label(rng, signals, target_fraud_rate)

    # ── Assemble the 50-column master schema (30 transaction cols +
    #    20 user/merchant/engineered bridge cols, matching
    #    paysense_master_dataset.csv / MASTER_SCHEMA_COLUMNS) ────────────────
    transaction_id = np.array([f"SGT{idx:07d}" for idx in range(n_txn)])

    # mrc_* fields only apply to Merchant-receiver rows; NaN for P2P (User) rows
    is_merchant = receiver_type == "Merchant"
    mrc_category = np.where(is_merchant, mrc_pick["mrc_category"].to_numpy(), None)
    mrc_size = np.where(is_merchant, mrc_pick["mrc_size"].to_numpy(), None)
    mrc_avg_daily_txn = np.where(is_merchant, mrc_pick["mrc_avg_daily_txn"].to_numpy(), np.nan)
    mrc_is_registered = np.where(is_merchant, mrc_pick["mrc_is_registered"].to_numpy(), np.nan)
    mrc_rating = np.where(is_merchant, mrc_pick["mrc_rating"].to_numpy(), np.nan)

    df = pd.DataFrame({
        # transactions.csv 30-column schema (exact order per data_dictionary.csv)
        "transaction_id": transaction_id,
        "user_id": sender["user_id"].to_numpy(),
        "receiver_id": receiver_id,
        "receiver_type": receiver_type,
        "amount": amount,
        "timestamp": [t.strftime("%Y-%m-%d %H:%M:%S") for t in timestamp],
        "date": [t.strftime("%Y-%m-%d") for t in timestamp],
        "hour_of_day": hour_of_day,
        "day_of_week": day_of_week,
        "is_weekend": is_weekend,
        "is_night_transaction": is_night_transaction,
        "time_since_last_txn_min": time_since_last_txn_min,
        "transaction_type": transaction_type,
        "payment_app": payment_app,
        "device_type": device_type,
        "status": status,
        "user_city_tier": _tier_to_int(sender["usr_home_city_tier"].to_numpy()),
        "user_kyc_status": sender["usr_kyc_status"].to_numpy(),
        "user_avg_monthly_txn": user_avg_monthly_txn,
        "user_avg_txn_value": user_avg_txn_value,
        "user_loyalty_score": sender["usr_loyalty_score_profile"].to_numpy(),
        "new_device_flag": new_device_flag,
        "ip_location_mismatch": ip_location_mismatch,
        "failed_attempts_last_24h": failed_attempts_last_24h,
        "transaction_velocity": transaction_velocity_out,
        "amount_deviation_score": amount_deviation_score_out,
        "is_fraud": is_fraud,
        "recurring_payment_flag": recurring_payment_flag,
        "balance_after_transaction": balance_after_transaction,
        "transaction_frequency_score": transaction_frequency_score,
        # Bridge columns (match paysense_master_dataset.csv's 20 extra columns)
        "txn_success_flag": txn_success_flag,
        "kyc_verified_flag": kyc_verified_flag,
        "data_source": "synthetic_grounded_v2",
        "usr_age_group": sender["usr_age_group"].to_numpy(),
        "usr_home_city": sender["usr_home_city"].to_numpy(),
        "usr_home_city_tier": _tier_to_int(sender["usr_home_city_tier"].to_numpy()),
        "usr_account_age_days": sender["usr_account_age_days"].to_numpy(),
        "usr_linked_bank_count": sender["usr_linked_bank_count"].to_numpy(),
        "usr_avg_monthly_txn_profile": sender["usr_avg_monthly_txn_profile"].to_numpy(),
        "usr_avg_txn_value_profile": sender["usr_avg_txn_value_profile"].to_numpy(),
        "usr_preferred_app": sender["usr_preferred_app"].to_numpy(),
        "usr_preferred_device": sender["usr_preferred_device"].to_numpy(),
        "usr_is_high_risk": sender["usr_is_high_risk"].to_numpy(),
        "mrc_category": mrc_category,
        "mrc_size": mrc_size,
        "mrc_avg_daily_txn": mrc_avg_daily_txn,
        "mrc_is_registered": mrc_is_registered,
        "mrc_rating": mrc_rating,
        "device_risk_score": device_risk_score,
        "ip_risk_score": ip_risk_score,
    })

    # Attach fraud_prob only as an in-memory diagnostic column dropped before
    # saving — useful for the generation-time sanity report, not part of the
    # model schema.
    df.attrs["fraud_prob"] = fraud_prob
    return df


def main():
    # Force UTF-8 stdout regardless of the Windows console codepage, same
    # rationale as generalization_check_ensemble.py — the bullet characters
    # printed below don't survive cp1252. Done only inside main(), not at
    # module import time, so importing this module (e.g. from the test
    # suite) never mutates global sys.stdout and breaks pytest's capture.
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print(f"Generating grounded synthetic dataset: N={N_TXN:,} rows, "
          f"seed={SEED}, target_fraud_rate={TARGET_FRAUD_RATE:.1%}")
    df = generate()
    fraud_prob = df.attrs["fraud_prob"]

    n_fraud = int(df["is_fraud"].sum())
    print(f"\nRealised fraud rate: {n_fraud:,}/{len(df):,} = {n_fraud/len(df)*100:.3f}%")
    print(f"Mean fraud_prob (pre-Bernoulli-sample): {fraud_prob.mean()*100:.3f}%")

    print("\n── Sanity: feature-vs-label separation (should NOT be near-1.0) ──")
    for col in ["new_device_flag", "ip_location_mismatch"]:
        rate_when_flag = df.loc[df[col] == 1, "is_fraud"].mean()
        rate_when_noflag = df.loc[df[col] == 0, "is_fraud"].mean()
        print(f"  {col:<22} fraud rate | flag=1: {rate_when_flag*100:5.2f}%  "
              f"flag=0: {rate_when_noflag*100:5.2f}%")
    corr_amt_dev = df["amount_deviation_score"].corr(df["is_fraud"].astype(float))
    print(f"  amount_deviation_score correlation with is_fraud: {corr_amt_dev:.3f}")

    print("\n── Distribution checks vs. documented rates ──")
    print(f"  status split           : {df['status'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"  user_kyc_status Verified: {(df['user_kyc_status']=='Verified').mean()*100:.2f}% (target ~87%)")
    print(f"  failed_attempts==0      : {(df['failed_attempts_last_24h']==0).mean()*100:.2f}% (target ~72%)")
    print(f"  time_since_last_txn NaN : {df['time_since_last_txn_min'].isna().mean()*100:.2f}% (target ~2%)")
    print(f"  amount_deviation_score NaN: {df['amount_deviation_score'].isna().mean()*100:.2f}% (target ~2%)")
    print(f"  transaction_velocity NaN: {df['transaction_velocity'].isna().mean()*100:.2f}% (target ~2%)")
    print(f"  amount max              : {df['amount'].max():.2f} (cap 100,000)")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
