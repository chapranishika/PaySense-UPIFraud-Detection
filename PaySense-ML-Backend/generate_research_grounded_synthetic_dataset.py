"""
================================================================================
  generate_research_grounded_synthetic_dataset.py
  ────────────────────────────────────────────────────────────────────────────
  TRACK B of REAL_DATA_AND_RESEARCH_GROUNDING.md.

  SYNTHETIC_GROUNDING.md grounded its dataset in real POPULATION-PREVALENCE
  statistics (RBI/NPCI: how much UPI fraud happens). It did not cite any
  research on which BEHAVIORAL SIGNALS actually discriminate fraud, or on how
  different fraud MODUS OPERANDI (typologies) produce different feature
  signatures -- its label was a single calibrated logistic risk model summing
  all signals with fixed hand-set weights, the same shape of correlation
  structure regardless of "what kind" of fraud a row represents.

  This script instead grounds the FRAUD-GENERATING PROCESS ITSELF in published
  research and government reporting on UPI/digital-payment fraud typologies,
  and implements it as a genuinely different generative structure: a finite
  MIXTURE MODEL over three distinct fraud typologies, each with its own
  feature-conditional distribution, rather than one continuous latent score.
  This is a structural difference, not a re-parameterization of the same
  model with a different seed.

  RESEARCH GROUNDING (real sources, cited as found -- not re-derived numbers
  dressed up as precise statistics; qualitative findings are used to decide
  the SHAPE of the model, not to fabricate exact coefficients)
  ────────────────────────────────────────────────────────────────────────────
  1. RBI Annual Report FY2024-25 (via Business Standard coverage, cited
     already in SYNTHETIC_GROUNDING.md §1.1): digital payment fraud cases
     rose 34% YoY. Separately (Ministry of Home Affairs data, via Business
     Standard reporting, Dec 2025 and widely syndicated): the primary
     technical mechanisms behind high-value UPI fraud are phishing links,
     counterfeit QR codes, remote-access apps, and SIM-swap operations --
     i.e. a MIX of distinct attack mechanisms, not one uniform pattern.
  2. SIM-swap fraud is a well-documented ACCOUNT-TAKEOVER attack that defeats
     OTP/2FA by porting the victim's number to an attacker-controlled SIM
     (Wikipedia "SIM swap attack"; "SIM Swap Fraud in India: A Digital
     Forensic Perspective", IJERD vol 22 issue 4). Once the attacker controls
     the OTP channel, the transaction is technically initiated from a device
     and network the legitimate user never used -- i.e. this typology's
     organic signature is a NEW DEVICE + IP/LOCATION MISMATCH, frequently
     preceded by failed authentication attempts while the attacker completes
     the takeover.
  3. Authorized Push Payment (APP) / social-engineering fraud is documented
     as a DISTINCT typology from account takeover precisely because the
     victim authorizes the transfer themselves under deception (fake
     customer-support calls, fraudulent UPI "collect" requests, romance/
     investment scams) -- see Zigram, "Authorized Push Payment Fraud:
     Detection & Prevention", and the general APP-fraud literature it
     summarizes. Because the victim's own device/session/KYC session is
     used, the organic signature is DEVICE/IP CLEAN, KYC VERIFIED, NO
     FAILED ATTEMPTS -- but the amount and/or receiver are atypical relative
     to the victim's own profile, because the victim is coached to send a
     specific, unusual amount to a specific, unfamiliar payee. This is the
     typology this project's own EDA already found empirically
     (`EDA_FEATURE_ENGINEERING.md` §1.2: 87.3% of the frozen model's
     "invisible" fraud rows have both hard flags clean) without having a
     name for it -- APP/social-engineering fraud is a documented real-world
     candidate explanation for that population, used here deliberately.
  4. Victim-demographics reporting (indiadatamap.com / psuconnect coverage of
     Ministry/NCRP-adjacent 2025 reporting, cited qualitatively): senior
     citizens are disproportionately targeted via impersonation/coercion
     (an account-takeover- and social-engineering-adjacent pattern), younger
     users via fake job/part-time-income schemes, rural households via
     phishing-led drains. Used here only to justify that typology
     probabilities are not independent of user age/profile (an ASSUMPTION on
     the specific tilt, not a copied percentage).
  5. Mule-network / money-laundering fraud is characterized in the general
     fraud-ops literature (e.g. Zigram's APP-fraud writeup; standard
     "velocity check" fraud-prevention literature such as Databricks'
     "Payment Fraud Detection" blog and paymentsandrisk.com's "Velocity
     Rules") by RAPID PASS-THROUGH of received funds and use of a shared/
     newly-registered receiving account -- i.e. elevated TRANSACTION
     VELOCITY and a receiver with lower registration/rating standing, with
     device/IP looking unremarkable (the mule's own device).
  6. Feature-importance literature on UPI-specific ML fraud models (search
     synthesis of "Enhancing UPI Fraud Detection: A Machine Learning
     Approach Using Stacked Generalization", ResearchGate) repeatedly
     surfaces transaction amount/value, transaction frequency, and failed
     attempts as top-ranked features. NOTE, stated plainly: the paper's full
     text could not be independently fetched (ResearchGate returned HTTP 403
     to automated retrieval in this environment), so the exact numeric
     feature-importance ranking/values are NOT reproduced or invented here --
     only the qualitative claim (these three categories of signal recur as
     top-ranked across UPI fraud ML studies) is used, and it is used only to
     decide which signals matter per typology below, not to set a precise
     coefficient.

  WHAT IS GENUINELY DIFFERENT FROM generate_grounded_synthetic_dataset.py
  ────────────────────────────────────────────────────────────────────────────
  - That script: ONE calibrated logistic model, `logit = sum(w_i * signal_i)
    + noise`, over 9 signals, the same functional form for every fraud row.
    A fraud row there is just "a legitimate row's signals, cranked up by a
    shared linear combination."
  - This script: fraud rows are drawn from a 3-COMPONENT MIXTURE. Which
    component (typology) a fraud row belongs to is itself random (drawn per
    row, with a documented, ASSUMPTION-flagged mix informed by point 1/4
    above), and each component has a DIFFERENT feature-conditional
    distribution -- not just a different mean on a single shared axis, but a
    genuinely different SET of features being perturbed (T1 perturbs
    device/IP/failed-attempts; T2 perturbs amount-deviation/receiver
    novelty while leaving device/IP/KYC/failed-attempts untouched; T3
    perturbs velocity/receiver-registration). This produces a label with a
    fundamentally different (multi-modal, feature-subset-conditional)
    correlation structure than a single logistic surface can express, which
    is the entire point of testing it as a separate hypothesis from
    `OOD_GENERALIZATION_REMEDIATION.md`'s Variant A (which blended in MORE
    data from the SAME single-logistic structure, different seed only).

  CONTAMINATION DISCIPLINE
  ────────────────────────────────────────────────────────────────────────────
  This script was designed and written WITHOUT reading the contents of
  `synthetic_grounded_dataset.csv` (seed 918273, the held-out eval set) or
  `category_generalization_test_set.csv` at any point -- only their
  documented schemas (from SYNTHETIC_GROUNDING.md and
  generate_grounded_synthetic_dataset.py's own column list, both already
  fully public within this project's own docs) were used, exactly the same
  degree of exposure every other script in this project that scores against
  those files already has. SEED below (771029) is disjoint from 42 (original
  pipeline), 918273 (held-out synthetic eval), and 445566
  (OOD_GENERALIZATION_REMEDIATION.md's blend seed).

  Run
  ───
      venv\\Scripts\\python.exe generate_research_grounded_synthetic_dataset.py
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
OUTPUT_CSV = os.path.join(BASE_DIR, "research_grounded_synthetic_dataset.csv")

SEED = 771029  # disjoint from 42 / 918273 / 445566 -- see docstring
N_USERS = 3000
N_MERCHANTS = 400
N_TXN = 15000
TARGET_FRAUD_RATE = 0.040  # same enrichment band as generate_grounded_synthetic_dataset.py,
                           # for apples-to-apples comparison against Variant A -- not a
                           # population-realism claim (see SYNTHETIC_GROUNDING.md §1 for why).

# ── Typology mix (ASSUMPTION on exact shares -- informed by, not copied from,
#    the qualitative "four primary mechanisms" / victim-demographics reporting
#    cited in the module docstring; no published source gives an exact %
#    breakdown by typology for UPI specifically) ──────────────────────────────
TYPOLOGY_NAMES = ["account_takeover", "social_engineering", "mule_network"]
TYPOLOGY_WEIGHTS = [0.35, 0.45, 0.20]  # social-engineering largest, per point 1/3 above

# ── Grounded categorical vocabularies (data_dictionary.csv) -- identical
#    baseline population facts to generate_grounded_synthetic_dataset.py;
#    what differs in this script is the FRAUD mechanism below, not these
#    shared, independently-true demographic baselines. ─────────────────────
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

TIER1_CITIES = ["Mumbai", "Delhi", "Bengaluru", "Hyderabad", "Chennai", "Kolkata", "Pune", "Ahmedabad"]
TIER2_CITIES = ["Jaipur", "Lucknow", "Kanpur", "Nagpur", "Indore", "Bhopal", "Patna",
                 "Vadodara", "Coimbatore", "Visakhapatnam", "Surat", "Ranchi"]
TIER3_CITIES = ["Jodhpur", "Mysuru", "Guwahati", "Raipur", "Amritsar", "Dehradun",
                 "Shimla", "Siliguri", "Jabalpur", "Rajkot", "Nashik"]
CITY_TIER_WEIGHTS = [0.45, 0.35, 0.20]  # ASSUMPTION


def _tier_to_int(tier_str):
    """See generate_grounded_synthetic_dataset.py's identical function for the
    full explanation of why this must be numeric (1/2/3), not the string form
    data_dictionary.csv documents -- the frozen preprocessor expects numeric."""
    return np.array([int(str(t).split()[-1]) for t in tier_str])


def _weighted_choice(rng, options, weights, size):
    return rng.choice(options, size=size, p=np.array(weights) / np.sum(weights))


def _sample_cities(rng, size):
    tiers = _weighted_choice(rng, ["Tier 1", "Tier 2", "Tier 3"], CITY_TIER_WEIGHTS, size)
    pools = {"Tier 1": TIER1_CITIES, "Tier 2": TIER2_CITIES, "Tier 3": TIER3_CITIES}
    cities = np.array([rng.choice(pools[t]) for t in tiers])
    return cities, tiers


def _generate_users(rng, n_users):
    user_id = np.array([f"RGU{idx:06d}" for idx in range(n_users)])
    age_group = _weighted_choice(rng, AGE_GROUPS, AGE_WEIGHTS, n_users)
    city, city_tier = _sample_cities(rng, n_users)
    kyc_status = np.where(rng.random(n_users) < 0.87, "Verified", "Not Verified")
    account_age_days = rng.triangular(30, 300, 2500, n_users).round().astype(int)
    one_bank = rng.random(n_users) < 0.50
    other = rng.choice([2, 3, 4], size=n_users)
    linked_bank_count = np.where(one_bank, 1, other)
    tier_mean = np.select(
        [city_tier == "Tier 1", city_tier == "Tier 2", city_tier == "Tier 3"], [45.0, 25.0, 12.0],
    )
    avg_monthly_transactions = np.clip(
        rng.gamma(shape=3.0, scale=tier_mean / 3.0), 1, None
    ).round().astype(int)
    avg_transaction_value = np.clip(
        rng.lognormal(mean=np.log(500), sigma=0.9, size=n_users), 50, 10000
    )
    preferred_app = _weighted_choice(rng, APPS, APP_WEIGHTS, n_users)
    preferred_device = _weighted_choice(rng, DEVICES, DEVICE_WEIGHTS, n_users)
    user_loyalty_score = 0.1 + 0.9 * rng.beta(2, 2, n_users)
    p_raw = np.where(kyc_status == "Verified", 0.03, 0.15)
    p_highrisk = np.clip(p_raw * (0.05 / p_raw.mean()), 0, 1)
    is_high_risk_user = (rng.random(n_users) < p_highrisk).astype(int)
    return pd.DataFrame({
        "user_id": user_id, "usr_age_group": age_group, "usr_home_city": city,
        "usr_home_city_tier": city_tier, "usr_kyc_status": kyc_status,
        "usr_account_age_days": account_age_days, "usr_linked_bank_count": linked_bank_count,
        "usr_avg_monthly_txn_profile": avg_monthly_transactions,
        "usr_avg_txn_value_profile": avg_transaction_value,
        "usr_preferred_app": preferred_app, "usr_preferred_device": preferred_device,
        "usr_loyalty_score_profile": user_loyalty_score, "usr_is_high_risk": is_high_risk_user,
    })


def _generate_merchants(rng, n_merchants):
    merchant_id = np.array([f"RGM{idx:05d}" for idx in range(n_merchants)])
    merchant_category = rng.choice(MERCHANT_CATS, size=n_merchants)
    size_roll = rng.random(n_merchants)
    merchant_size = np.select(
        [size_roll < 0.55, size_roll < 0.85], ["Small", "Medium"], default="Enterprise"
    )
    city, city_tier = _sample_cities(rng, n_merchants)
    size_mean = np.select(
        [merchant_size == "Small", merchant_size == "Medium"], [5.0, 40.0], default=250.0
    )
    avg_daily_transactions = np.clip(
        rng.gamma(shape=2.0, scale=size_mean / 2.0), 1, None
    ).round().astype(int)
    p_raw = np.select(
        [merchant_size == "Enterprise", merchant_size == "Medium"], [0.99, 0.96], default=0.90
    )
    p_reg = np.clip(p_raw * (0.94 / p_raw.mean()), 0, 1)
    is_registered = (rng.random(n_merchants) < p_reg).astype(int)
    rating = 2.5 + 2.5 * rng.beta(5, 2, n_merchants)
    return pd.DataFrame({
        "receiver_id": merchant_id, "mrc_category": merchant_category, "mrc_size": merchant_size,
        "mrc_city": city, "mrc_city_tier": city_tier, "mrc_avg_daily_txn": avg_daily_transactions,
        "mrc_is_registered": is_registered, "mrc_rating": rating,
    })


def _derive_risk_scores(rng, flag):
    n = len(flag)
    high = rng.beta(8, 2, n)
    low = rng.beta(2, 8, n)
    return np.where(flag == 1, high, low)


# ════════════════════════════════════════════════════════════════════════════
#  THE STRUCTURALLY DIFFERENT PART: typology-conditioned fraud generation
# ════════════════════════════════════════════════════════════════════════════
def _assign_typology(rng, n_fraud, sender_age_group):
    """
    Draws a typology per fraud row. Base rates from TYPOLOGY_WEIGHTS, with a
    documented ASSUMPTION tilt: senior (55+) users draw account_takeover /
    social_engineering more often (impersonation/coercion targeting, per the
    victim-demographics reporting cited in the module docstring); 18-24 users
    draw social_engineering slightly more often (fake job/income schemes).
    This is a real, cited qualitative pattern implemented as a modest,
    documented tilt -- not an invented precise number.
    """
    base = np.array(TYPOLOGY_WEIGHTS)
    out = np.empty(n_fraud, dtype=object)
    for i in range(n_fraud):
        w = base.copy()
        if sender_age_group[i] == "55+":
            w[0] *= 1.4  # account_takeover
            w[1] *= 1.3  # social_engineering
            w[2] *= 0.5  # mule_network
        elif sender_age_group[i] == "18-24":
            w[1] *= 1.3  # social_engineering
        w = w / w.sum()
        out[i] = rng.choice(TYPOLOGY_NAMES, p=w)
    return out


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

    activity_w = users["usr_avg_monthly_txn_profile"].to_numpy(dtype=float)
    activity_w = activity_w / activity_w.sum()
    sender_idx = rng.choice(len(users), size=n_txn, p=activity_w)
    sender = users.iloc[sender_idx].reset_index(drop=True)

    transaction_type = _weighted_choice(rng, TXN_TYPES, TXN_TYPE_WEIGHTS, n_txn)
    receiver_type = np.where(transaction_type == "P2P", "User", "Merchant")

    mrc_w = merchants["mrc_avg_daily_txn"].to_numpy(dtype=float)
    mrc_w = mrc_w / mrc_w.sum()
    mrc_pick_idx = rng.choice(len(merchants), size=n_txn, p=mrc_w)
    mrc_pick = merchants.iloc[mrc_pick_idx].reset_index(drop=True)

    user_pick_idx = rng.integers(0, len(users), size=n_txn)
    same_as_sender = user_pick_idx == sender_idx
    user_pick_idx[same_as_sender] = (user_pick_idx[same_as_sender] + 1) % len(users)
    user_receiver_id = users["user_id"].to_numpy()[user_pick_idx]
    receiver_id = np.where(receiver_type == "Merchant", mrc_pick["receiver_id"].to_numpy(), user_receiver_id)

    # ── STEP 1: decide is_fraud FIRST (not via a continuous risk score) ─────
    is_fraud = (rng.random(n_txn) < target_fraud_rate).astype(int)
    n_fraud = int(is_fraud.sum())
    fraud_idx = np.where(is_fraud == 1)[0]
    typology = np.full(n_txn, "", dtype=object)
    if n_fraud > 0:
        typology[fraud_idx] = _assign_typology(rng, n_fraud, sender["usr_age_group"].to_numpy()[fraud_idx])
    is_T1 = typology == "account_takeover"
    is_T2 = typology == "social_engineering"
    is_T3 = typology == "mule_network"

    # ── amount: baseline log-normal around the sender's own profile average
    #    (same shape as generate_grounded_synthetic_dataset.py for legit rows
    #    and as a starting point for fraud rows -- typology below then
    #    perturbs it multiplicatively, which is where the structural
    #    difference actually lives) ───────────────────────────────────────
    base_amount = np.clip(
        rng.lognormal(
            mean=np.log(np.clip(sender["usr_avg_txn_value_profile"].to_numpy(), 50, None)),
            sigma=0.7, size=n_txn,
        ), 1, 100000,
    )
    amount = base_amount.copy()
    # T1 (account takeover): "sweep" pattern -- large multiple of profile avg,
    # grounded in the real-world account-takeover pattern of draining an
    # account once control is gained (qualitative, not a cited exact figure).
    amount[is_T1] = np.clip(base_amount[is_T1] * rng.uniform(3.0, 7.0, is_T1.sum()), 1, 100000)
    # T2 (social engineering): moderately elevated, atypical-for-this-user
    # amount -- the victim is coached to send a specific unusual amount.
    amount[is_T2] = np.clip(base_amount[is_T2] * rng.uniform(1.8, 3.5, is_T2.sum()), 1, 100000)
    # T3 (mule network): mildly elevated only -- velocity is this typology's
    # real signal, not amount.
    amount[is_T3] = np.clip(base_amount[is_T3] * rng.uniform(1.1, 1.8, is_T3.sum()), 1, 100000)

    # ── timestamp ────────────────────────────────────────────────────────
    hour_weights = np.array([
        0.6, 0.4, 0.3, 0.3, 0.4, 0.8, 1.8, 2.6, 3.4, 3.6, 3.2, 3.0,
        3.8, 3.6, 3.0, 2.8, 2.9, 3.3, 4.2, 4.6, 4.0, 3.2, 2.2, 1.2,
    ])
    hour_of_day = _weighted_choice(rng, np.arange(24), hour_weights, n_txn).astype(int)
    # T1 (account takeover) is night-skewed -- attackers act once OTP control
    # is gained, often outside the victim's normal waking hours (qualitative,
    # consistent with SIM-swap forensic case descriptions cited above).
    night_hours = np.array([0, 1, 2, 3, 4, 22, 23])
    if is_T1.sum() > 0:
        force_night = rng.random(is_T1.sum()) < 0.55
        night_draw = rng.choice(night_hours, size=is_T1.sum())
        hour_of_day_T1 = np.where(force_night, night_draw, hour_of_day[is_T1])
        hour_of_day[is_T1] = hour_of_day_T1

    day_offset = rng.integers(0, 365, size=n_txn)
    base_date = datetime(2025, 1, 1)
    dates = [base_date + timedelta(days=int(d)) for d in day_offset]
    minute = rng.integers(0, 60, size=n_txn)
    second = rng.integers(0, 60, size=n_txn)
    timestamp = [d.replace(hour=int(h), minute=int(m), second=int(s))
                 for d, h, m, s in zip(dates, hour_of_day, minute, second)]
    day_of_week_name = np.array([t.strftime("%A") for t in timestamp])
    is_weekend = np.isin(day_of_week_name, ["Saturday", "Sunday"]).astype(int)
    day_of_week = np.array([t.weekday() for t in timestamp])
    is_night_transaction = ((hour_of_day < 6) | (hour_of_day >= 22)).astype(int)

    mean_gap_min = 43200.0 / np.clip(sender["usr_avg_monthly_txn_profile"].to_numpy(), 1, None)
    time_since_last_txn_min = rng.exponential(mean_gap_min)
    missing_gap = rng.random(n_txn) < 0.02
    time_since_last_txn_min = np.where(missing_gap, np.nan, time_since_last_txn_min)

    payment_app = np.where(
        rng.random(n_txn) < 0.85, sender["usr_preferred_app"].to_numpy(),
        _weighted_choice(rng, APPS, APP_WEIGHTS, n_txn),
    )

    # ── device_type / new_device_flag -- THE T1-SPECIFIC HARD SIGNAL.
    #    Baseline: mostly the preferred device (as before). T1 rows are
    #    FORCED to a different device with high probability (the SIM-swap/
    #    account-takeover organic signature per point 2 in the docstring).
    #    T2/T3 rows use the VICTIM'S/MULE'S OWN device -- i.e. deliberately
    #    NOT perturbed, which is the entire structural point of T2 (clean
    #    hard flags) vs. T1 (dirty hard flags). ──────────────────────────
    device_type = np.where(
        rng.random(n_txn) < 0.90, sender["usr_preferred_device"].to_numpy(),
        _weighted_choice(rng, DEVICES, DEVICE_WEIGHTS, n_txn),
    )
    if is_T1.sum() > 0:
        force_new_device = rng.random(is_T1.sum()) < 0.85
        alt_device = _weighted_choice(rng, DEVICES, DEVICE_WEIGHTS, is_T1.sum())
        device_type[is_T1] = np.where(force_new_device, alt_device, device_type[is_T1])
    new_device_flag = (device_type != sender["usr_preferred_device"].to_numpy()).astype(int)

    # ── ip_location_mismatch -- same T1-only treatment as new_device_flag.
    #    Baseline rate 8% (independent low-probability event, as in
    #    generate_grounded_synthetic_dataset.py); T1 forced to ~80% (SIM-swap/
    #    remote-access attacks originate from a different network/location);
    #    T2/T3 left at baseline (clean for T2 by design; T3's mule may
    #    plausibly transact locally). ───────────────────────────────────
    ip_location_mismatch = (rng.random(n_txn) < 0.08).astype(int)
    if is_T1.sum() > 0:
        ip_location_mismatch[is_T1] = (rng.random(is_T1.sum()) < 0.80).astype(int)

    status = _weighted_choice(rng, ["Success", "Failed", "Pending"], [0.88, 0.09, 0.03], n_txn)
    txn_success_flag = (status == "Success").astype(int)

    # ── failed_attempts_last_24h -- T1-only elevation (failed OTP/login
    #    attempts while the SIM-swap/takeover completes, per point 2). T2/T3
    #    left at the ordinary baseline distribution -- this is precisely
    #    what makes T2 "invisible" to a model trained to expect fraud to
    #    show up here (matching EDA_FEATURE_ENGINEERING.md §1.2's finding
    #    empirically, by construction here rather than by accident). ──────
    fail_roll = rng.random(n_txn)
    failed_attempts_last_24h = np.zeros(n_txn, dtype=int)
    cum = 0.72
    for val, share in [(1, 0.12), (2, 0.08), (3, 0.04), (4, 0.025), (5, 0.015)]:
        nxt = cum + share
        failed_attempts_last_24h = np.where((fail_roll >= cum) & (fail_roll < nxt), val, failed_attempts_last_24h)
        cum = nxt
    if is_T1.sum() > 0:
        elevated = rng.integers(2, 6, size=is_T1.sum())
        force_elevated = rng.random(is_T1.sum()) < 0.65
        failed_attempts_last_24h[is_T1] = np.where(force_elevated, elevated, failed_attempts_last_24h[is_T1])

    # ── transaction_velocity -- THE T3-SPECIFIC HARD SIGNAL (rapid pass-
    #    through of received funds, per point 5). T1/T2 left at the ordinary
    #    baseline. ─────────────────────────────────────────────────────────
    transaction_velocity = np.clip(rng.poisson(0.6, n_txn), 0, 4).astype(float)
    if is_T3.sum() > 0:
        transaction_velocity[is_T3] = np.clip(rng.poisson(3.2, is_T3.sum()), 2, 4).astype(float)
    missing_vel = rng.random(n_txn) < 0.02
    transaction_velocity_out = np.where(missing_vel, np.nan, transaction_velocity)

    recurring_payment_flag = np.isin(transaction_type, ["Bill Payment", "Subscription", "EMI"]).astype(int)
    balance_after_transaction = rng.uniform(500, 80000, n_txn)
    user_avg_monthly_txn = sender["usr_avg_monthly_txn_profile"].to_numpy()
    transaction_frequency_score = np.clip(user_avg_monthly_txn / 50.0, 0, 1)

    # ── amount_deviation_score -- real per-row statistic (as in the existing
    #    generator), computed AFTER the typology-conditioned amount above, so
    #    T1/T2's amount inflation naturally produces elevated deviation too
    #    (a real derived consequence, not independently re-perturbed) --
    #    except T2 gets an EXTRA explicit boost, because social-engineering
    #    fraud's defining real-world signature (per point 3) is specifically
    #    "unusual amount for this user," and the amount multiplier alone
    #    (1.8-3.5x) undersells that if the user's own profile average is low
    #    variance. ──────────────────────────────────────────────────────────
    user_avg_txn_value = sender["usr_avg_txn_value_profile"].to_numpy()
    amount_deviation_score = np.clip(
        np.abs(amount - user_avg_txn_value) / (user_avg_txn_value * 0.5 + 50), 0, 10
    )
    if is_T2.sum() > 0:
        amount_deviation_score[is_T2] = np.clip(
            amount_deviation_score[is_T2] * rng.uniform(1.3, 2.0, is_T2.sum()), 0, 10
        )
    missing_dev = rng.random(n_txn) < 0.02
    amount_deviation_score_out = np.where(missing_dev, np.nan, amount_deviation_score)

    # ── kyc_verified_flag -- deliberately LEFT AT THE USER'S OWN PROFILE
    #    VALUE for every typology, including T1. This is a deliberate
    #    modeling choice, not an oversight: SIM-swap/account-takeover exploits
    #    an ALREADY-KYC'd account (that is the entire point of taking it
    #    over), so forcing kyc_verified_flag down for T1 would fabricate a
    #    correlation the real-world mechanism does not imply. ──────────────
    kyc_verified_flag = (sender["usr_kyc_status"].to_numpy() == "Verified").astype(int)

    device_risk_score = _derive_risk_scores(rng, new_device_flag)
    ip_risk_score = _derive_risk_scores(rng, ip_location_mismatch)

    # ── mrc_is_registered / mrc_rating -- T3-only perturbation (mule/
    #    unregistered-receiver signature, per point 5). Only applies to
    #    Merchant-receiver rows; P2P mule transfers are left as NaN like any
    #    other P2P row (merchant fields are meaningless there either way). ──
    is_merchant = receiver_type == "Merchant"
    mrc_is_registered_out = np.where(is_merchant, mrc_pick["mrc_is_registered"].to_numpy().astype(float), np.nan)
    mrc_rating_out = np.where(is_merchant, mrc_pick["mrc_rating"].to_numpy(), np.nan)
    t3_merchant = is_T3 & is_merchant
    if t3_merchant.sum() > 0:
        mrc_is_registered_out[t3_merchant] = (rng.random(t3_merchant.sum()) < 0.35).astype(float)
        mrc_rating_out[t3_merchant] = np.clip(rng.normal(2.6, 0.4, t3_merchant.sum()), 1.0, 5.0)

    transaction_id = np.array([f"RGT{idx:07d}" for idx in range(n_txn)])
    mrc_category = np.where(is_merchant, mrc_pick["mrc_category"].to_numpy(), None)
    mrc_size = np.where(is_merchant, mrc_pick["mrc_size"].to_numpy(), None)
    mrc_avg_daily_txn = np.where(is_merchant, mrc_pick["mrc_avg_daily_txn"].to_numpy(), np.nan)

    df = pd.DataFrame({
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
        "txn_success_flag": txn_success_flag,
        "kyc_verified_flag": kyc_verified_flag,
        "data_source": "research_grounded_v1",
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
        "mrc_is_registered": mrc_is_registered_out,
        "mrc_rating": mrc_rating_out,
        "device_risk_score": device_risk_score,
        "ip_risk_score": ip_risk_score,
    })
    df.attrs["typology"] = typology
    return df


def main():
    if __name__ == "__main__":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print(f"Generating research-grounded synthetic dataset: N={N_TXN:,} rows, "
          f"seed={SEED}, target_fraud_rate={TARGET_FRAUD_RATE:.1%}")
    df = generate()
    typology = df.attrs["typology"]

    n_fraud = int(df["is_fraud"].sum())
    print(f"\nRealised fraud rate: {n_fraud:,}/{len(df):,} = {n_fraud/len(df)*100:.3f}%")
    print("\n-- Typology mix among fraud rows --")
    for t in TYPOLOGY_NAMES:
        n_t = int((typology == t).sum())
        print(f"  {t:<20}: {n_t:,} ({n_t/max(n_fraud,1)*100:.1f}%)")

    print("\n-- Sanity: feature-vs-label separation (should NOT be near-1.0 overall,"
          " but SHOULD be starkly different between typologies) --")
    for col in ["new_device_flag", "ip_location_mismatch"]:
        rate_when_flag = df.loc[df[col] == 1, "is_fraud"].mean()
        rate_when_noflag = df.loc[df[col] == 0, "is_fraud"].mean()
        print(f"  {col:<22} fraud rate | flag=1: {rate_when_flag*100:5.2f}%  "
              f"flag=0: {rate_when_noflag*100:5.2f}%")
    both_clean = (df["new_device_flag"] == 0) & (df["ip_location_mismatch"] == 0)
    clean_fraud_share = ((df["is_fraud"] == 1) & both_clean).sum() / max(n_fraud, 1)
    print(f"  Share of fraud rows with BOTH hard flags clean: {clean_fraud_share*100:.1f}% "
          f"(this project's own EDA found 87.3% for the frozen model's 'invisible' fraud "
          f"population -- this dataset deliberately builds a comparable-order population "
          f"by construction via the social_engineering typology, not by accident)")
    corr_amt_dev = df["amount_deviation_score"].corr(df["is_fraud"].astype(float))
    corr_vel = df["transaction_velocity"].corr(df["is_fraud"].astype(float))
    print(f"  amount_deviation_score correlation with is_fraud: {corr_amt_dev:.3f}")
    print(f"  transaction_velocity correlation with is_fraud: {corr_vel:.3f}")

    print("\n-- Distribution checks vs. documented rates --")
    print(f"  status split           : {df['status'].value_counts(normalize=True).round(3).to_dict()}")
    print(f"  user_kyc_status Verified: {(df['user_kyc_status']=='Verified').mean()*100:.2f}% (target ~87%)")
    print(f"  amount max              : {df['amount'].max():.2f} (cap 100,000)")

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nSaved: {OUTPUT_CSV}  ({df.shape[0]:,} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    main()
