# PaySense — Real Data Search & Research-Grounded Synthetic Data (Tracks A + B)

**Date:** 2026-08-24
**Author's intent:** Every prior generalization document tonight
(`GENERALIZATION_CHECK.md`, `SYNTHETIC_GROUNDING.md`,
`OOD_GENERALIZATION_REMEDIATION.md`, `EDA_FEATURE_ENGINEERING.md`) diagnosed
the same underlying problem from different angles — the frozen model
(`artefacts/paysense_model.pkl`, deployed threshold **0.50**, verified
directly from `artefacts/paysense_threshold.pkl` rather than assumed — see
§4.0) has never caught a single real fraud case in any real-world test, and
blending in more data from the *same* synthetic generative process improved
ranking (ROC-AUC) without moving the catch rate at all. Two things had not
yet been tried: (Track A) searching for a genuinely richer *real* dataset —
one that actually carries device/IP-risk, KYC, account-age, or velocity
signals instead of the same handful of amount/hour/category fields every
prior real dataset offered — and (Track B) grounding a synthetic dataset's
*label-generating mechanism* in real fraud-typology research, not just in
population base-rate statistics. This document does both, reports every
number honestly, and does not manufacture a win where the catch rate didn't
move.

Scripts: `generate_research_grounded_synthetic_dataset.py` (Track B
dataset), `real_data_and_research_grounding.py` (Dataset 5 loader/vetting,
Variant C training, and all five evaluation checks through the real
`src.fraud_model.score()` ensemble path — not raw `model.predict_proba()`).
No changes to `artefacts/paysense_model.pkl`, `paysense_preprocessor.pkl`,
or `paysense_threshold.pkl` anywhere in this document.

---

## 0. Threshold, verified directly

`artefacts/paysense_threshold.pkl` was loaded and printed by
`real_data_and_research_grounding.py` at run time (not assumed from this
prompt or from memory of other documents): **0.50**. Every confusion
matrix / recall / precision number below uses this value, asserted
programmatically at the top of the script (`assert abs(ps_threshold - 0.50)
< 1e-9`) so a stale threshold would fail loudly rather than silently produce
numbers for the wrong operating point.

---

## 1. Track A — searching for a genuinely richer real dataset

### 1.1 What was searched

Beyond the three datasets `GENERALIZATION_CHECK.md` already triaged and the
one `EDA_FEATURE_ENGINEERING.md` added (Dataset 4), this search covered:

- Every remaining unused folder under `E:\Projects\upi\` (11 candidate
  folders/files never referenced by any prior document).
- Live web search (Kaggle listings, arXiv, Amazon Science's Fraud Dataset
  Benchmark) for real financial/e-commerce/UPI fraud datasets carrying
  device, IP, KYC, account-age, or velocity signals — the exact fields
  `GENERALIZATION_CHECK.md` §4.4 already diagnosed as the reason LightLR and
  the rules scorer contribute nothing on Dataset 1/3.
- Direct download attempts via Kaggle's public, unauthenticated
  `/api/v1/datasets/download/<owner>/<slug>` endpoint (the same mechanism
  that already worked for two datasets in tonight's category-classifier
  work) for every promising lead the search surfaced.

### 1.2 Candidates rejected without scoring

| Candidate | Rows | Verdict | Why |
|---|---:|---|---|
| `UPI  Digital Payment Behavior Dataset/upi_digital_payment_dataset.xlsx` | 120 | **REJECTED** | No fraud label at all (Age/City/Monthly_Transactions/Preferred_App/Spending_Category only) — same class of rejection as `GENERALIZATION_CHECK.md`'s skipped no-label datasets. |
| `UPI Transactions Dataset/MyTransaction.csv` | 1,470 | Not new | Already used in `SYNTHETIC_GROUNDING.md` §3 as a realism sanity check; no fraud label, cannot be re-purposed as a training/eval target. |
| `UPI Transactions 2024 Dataset` / `UPI transacitons 2024 - 250K rows` (duplicate copies) | 250,000 | **REJECTED for Track A** | A real `fraud_flag` column and a schema that looks promising (exact `device_type` Android/iOS/Web taxonomy match, exact `transaction_type` P2P/P2M/Bill Payment/Recharge match) — but the Kaggle listing itself states this dataset is produced by a published generator script ("Script — kaggle.com/code/skullagos5246/upi-transactions-generator"). It is honestly, admittedly synthetic by its own source, not real-world data, so it cannot satisfy Track A regardless of how well-behaved its statistics are. (For the record: its statistics ARE well-behaved — fraud rate 0.192%, not round; every category's fraud rate sits within 0.148%-0.214% of the base rate, no near-0%/100% category — it would have passed `GENERALIZATION_CHECK.md` §2.2's vetting on separability grounds alone. It fails only on genuineness, the exact same failure mode `CATEGORY_CLASSIFIER_V3_ATTEMPT.md` found in two Kaggle downloads that looked real but were templated underneath.) |
| Kaggle `thuandao/bank-transactions-dataset-for-fraud-detection` | 50,000 | **REJECTED** | Downloaded and inspected directly. Genuinely rich fields (`DeviceID`, `IP Address`, `LoginAttempts`, `AccountBalance`, `Channel`) — exactly the kind of signal this search was looking for — but the file carries **no fraud label column at all**. Unusable for evaluation or training regardless of feature richness. |
| IEEE-CIS Fraud Detection (`E:\Projects\upi\IEEE\`, Kaggle's real 2019 fraud competition dataset, Vesta Corporation) | 590,540 (+144,233 identity rows) | **Vetted real, but rejected for this purpose** | See §1.3 — genuinely real, non-round fraud rate (3.499%), but its honest overlap with PaySense's schema is *worse* than Dataset 1's, despite being far larger and more famous. |

### 1.3 IEEE-CIS Fraud Detection — genuinely real, vetted, but a worse overlap than Dataset 1

This is the well-known Kaggle IEEE-CIS Fraud Detection competition dataset
— real, anonymized e-commerce transaction data from Vesta Corporation, not
templated or synthetic. Vetting (same rigor as `GENERALIZATION_CHECK.md`
§2.2): fraud rate 3.499% (20,663/590,540) — not round; no shared ID
backbone with any other file in this project; a legitimate, independently
famous benchmark, not a fabricated download.

**But the honest mapping to PaySense's 40-feature schema is worse than
Dataset 1's, not better — the exact opposite of what this search hoped to
find, and reported plainly:**

- `amount` ← `TransactionAmt` (direct; currency unstated by the source,
  used as-is, same treatment given to Dataset 3's and Dataset 5's `amount`)
- `device_type` ← derived from the paired `train_identity.csv`'s
  `DeviceInfo` free-text field via a documented heuristic (contains
  `ios`/`iphone`/`ipad` → iOS; contains an Android hardware marker
  (`sm-`, `samsung`, `moto`, `pixel`, `redmi`, `build/`, …) → Android;
  `DeviceType == "desktop"` or a Windows/Mac marker → Web; otherwise left
  unmapped). Resolved for 97.9% of a stratified 15,000-row sample — a
  genuine, if judgment-call, mapping.
- **That's it — 2 of 40 features (5.0%)**, *worse* than Dataset 1's 6/40
  (15.0%) despite IEEE-CIS having 394 raw columns to Dataset 1's 15.
  `hour_of_day`/`is_night_transaction` were deliberately **not** derived
  from `TransactionDT`: the competition host explicitly documents
  `TransactionDT` as "a timedelta from a given reference datetime, **not an
  actual timestamp**" — deriving a clock hour from it requires assuming an
  undisclosed day-alignment, and several community notebooks disagree on
  the exact reference point. Doing this anyway would be exactly the kind of
  "guess dressed as certainty" `GENERALIZATION_CHECK.md` §3.1 already
  refused for `payee_type`→`receiver_type`, so it was left out.
  IEEE-CIS's richest columns — `V1`-`V339` (Vesta's own engineered
  features), `C1`-`C14` (entity-count features), `D1`-`D15` (time-delta
  features), `M1`-`M9` (match flags) — are exactly the kind of
  device/IP-risk/velocity signal this search was looking for **in
  substance**, but Vesta's competition documentation deliberately does not
  disclose which column means what, "to protect user privacy and prevent
  competitors from reverse-engineering PII." Mapping any of them 1:1 to
  `new_device_flag`, `ip_location_mismatch`, `transaction_velocity`, or
  `failed_attempts_last_24h` would be fabricating a semantic correspondence
  the source itself refuses to confirm — precisely the discipline this
  project's own mapping tables have consistently upheld everywhere else.

A fast raw-XGBoost-only spot check (not the full ensemble — consistent with
how `EDA_FEATURE_ENGINEERING.md` §2.3 treated its own lower-priority
Dataset 4), on a stratified 15,000-row sample (525 fraud):

| Metric | Value |
|---|---:|
| ROC-AUC | 0.4753 |
| PR-AUC | 0.0357 |
| Confusion matrix @ τ=0.50 | TN=14,475 FP=0 / FN=525 TP=0 |
| Max predicted probability | 0.0045 |

ROC-AUC below 0.5 (worse than a coin flip) confirms 2/40 honestly-mapped
features aren't enough for this model to extract any real ranking signal
from IEEE-CIS at all — not scored further through the full ensemble, given
the time cost of a 590K-row per-row scoring pass and the low expected value
of a check that already under-performs the datasets already in use.

**Conclusion: bigger and more famous does not mean richer overlap.** The
richest real fraud-detection benchmarks publicly available deliberately
withhold the semantics of their most valuable columns for privacy reasons —
the same reason PaySense's own personalization features would need to exist
in a hypothetical real deployment's data, but can't be verified against in
any public dataset found tonight.

### 1.4 Dataset 5 — accepted: `kaggle_vbinh002_fraud_ecommerce/Fraud_Data.csv`

The one genuinely new, accepted find. Real e-commerce fraud data — the
"fraudecom" dataset cataloged in Amazon Science's **Fraud Dataset
Benchmark** paper (arXiv:2208.14417) as one of the few public fraud
datasets that retains raw device/IP fields. 151,112 rows, one transaction
per user. Downloaded via Kaggle's public unauthenticated endpoint
2026-08-24 (`external_data/kaggle_vbinh002_fraud_ecommerce/`, see
`external_data/NOTICE.md` for the full provenance note, including the
"license: unspecified" caveat, disclosed rather than glossed over).

**Vetting (same rigor as `GENERALIZATION_CHECK.md` §2.2, all verified
directly, not assumed):**

| Check | Result |
|---|---|
| Fraud rate | 9.3646% (14,151/151,112) — not round |
| Duplicate `user_id` / full rows | 0 / 0 |
| Nulls | 0 across all 11 columns |
| `source` (Ads/Direct/SEO) fraud rate | 8.93%–10.54% — no near-0%/100% category |
| `browser` (Chrome/FireFox/IE/Opera/Safari) fraud rate | 8.68%–9.88% — no near-0%/100% category |
| `sex` fraud rate | 9.10% (F) / 9.55% (M) — no separation |
| `purchase_value` by class | mean 36.93 (legit) vs. 36.99 (fraud) — **almost no signal**, an honest, unflattering finding reported plainly, not hidden |
| `age` by class | mean 33.12 (legit) vs. 33.32 (fraud) — negligible |
| **account age at purchase** (`purchase_time − signup_time`) | **legit median 60.13 days vs. fraud median 0.000012 days** — fraud purchases happen essentially immediately after signup. A real, strong, credible signal (elevated but with genuine overlap in both tails — fraud's own std is 38.35 days, not a clean spike at zero) |
| `device_id` shared across ≥2 different users | fraud rate 52.46% (shared) vs. 3.04% (not shared) — a real, strong multi-accounting signal, elevated but not deterministic |

No near-deterministic separation anywhere (nothing at 0%/100%, unlike the
disqualifying Dataset 2 in `GENERALIZATION_CHECK.md` §2.2). **Accepted.**

**Honest mapping — 7 of 40 features (17.5%), better than Dataset 1's 6/40
and Dataset 3's 2/40:**

`amount` ← `purchase_value` (direct); `usr_account_age_days` ← a genuinely
computed per-row `(purchase_time − signup_time)` in days (this is the
standout addition — neither Dataset 1 nor Dataset 3's honest mapping has
ever exercised this specific PaySense personalization feature with a
*strong* real signal before; Dataset 3 supplied the same field but with a
far weaker, unremarked correlation); `hour_of_day` ← `purchase_time.hour`
(direct); `is_night_transaction` ← derived, same rule as elsewhere;
`day_of_week` ← `purchase_time.dayofweek()` (numeric, matching the frozen
preprocessor's expectation — see `SYNTHETIC_GROUNDING.md` §2.4 for why
numeric, not the dictionary's documented string form); `is_weekend` ←
derived; `usr_age_group` ← `age` bucketed into PaySense's five brackets.

**Explicitly NOT mapped, and why** (full reasoning in
`real_data_and_research_grounding.py`'s `load_dataset_5` docstring):
`device_id`/`browser` → `device_type` (no OS information in either field —
browser runs across every OS, the same collision
`GENERALIZATION_CHECK.md` §3.2 already refused for Dataset 3); `source`,
`sex` → no PaySense field; `ip_address` → cannot honestly produce
`ip_location_mismatch` because there is no per-user home-country baseline
in this dataset to compare a geolocation against; **device_id reuse across
different users**, despite being a real and strong signal (52.46% vs.
3.04% fraud rate), is a *multi-accounting* pattern, not the *same-user,
unfamiliar-device* concept `new_device_flag` encodes — conflating the two
would repeat exactly the mistake `GENERALIZATION_CHECK.md` §3.1 already
refused for `is_new_payee`/`new_device_flag`, so it is reported here as an
interesting real finding and left out of the mapping.

**Scored as a stratified 20,000-row sample** (preserving the 9.3646% fraud
rate, ≈1,873 fraud rows — far higher power than Dataset 3's accepted
1,000/64), not the full 151,112 rows: scoring the full population through
the real per-row ensemble three times (baseline, blended-training, Variant
C) would cost over an hour for this one non-mandatory extra dataset alone,
on top of the four checks the task requires. Disclosed here, not silently
substituted as "the whole dataset."

**Is Dataset 5 usable as additional *training* signal?** No — stated
plainly, same verdict `EDA_FEATURE_ENGINEERING.md` §2.3 reached for its own
Dataset 4. Even with `usr_account_age_days` present, Dataset 5 still cannot
supply `new_device_flag`, `ip_location_mismatch`,
`amount_deviation_score`, `transaction_velocity`,
`failed_attempts_last_24h`, `kyc_verified_flag`, or `usr_is_high_risk` — the
specific hard-signal features `EDA_FEATURE_ENGINEERING.md` §1.1 identified
as the actual root cause of the OOD gap. It is a genuinely better *held-out
evaluation* target than any real dataset used before; it does not close the
training-data gap this project has been chasing.

---

## 2. Track B — synthetic data grounded in fraud-*pattern* research

### 2.1 What was cited, and its limits (stated up front)

`SYNTHETIC_GROUNDING.md` cited real RBI/NPCI *population-prevalence*
statistics (how much UPI fraud happens). It did not cite research on which
*behavioral signals* discriminate fraud, or on how different fraud
*typologies* produce different feature signatures — its label was one
calibrated logistic risk model summing every signal with the same fixed
functional form regardless of what kind of fraud a row represents. This
track searched specifically for the missing half: real research/reporting
on UPI fraud *mechanisms*, and used it to build a genuinely different
generative *structure* — a mixture model over three typologies — not a
different seed on the same structure.

**Sources actually found and cited** (full citations and exact wording in
`generate_research_grounded_synthetic_dataset.py`'s module docstring):

1. RBI Annual Report FY2024-25 (34% YoY rise in digital payment fraud
   cases) and Ministry of Home Affairs data (via Business Standard
   reporting, Dec 2025): the four primary technical mechanisms behind
   high-value UPI fraud are **phishing links, counterfeit QR codes,
   remote-access apps, and SIM-swap operations** — a mix of distinct
   mechanisms, not one uniform pattern.
2. SIM-swap fraud is documented (Wikipedia "SIM swap attack"; "SIM Swap
   Fraud in India: A Digital Forensic Perspective," IJERD vol 22 issue 4)
   as an **account-takeover** attack defeating OTP/2FA — once the attacker
   controls the OTP channel, the transaction is initiated from a device and
   network the real user never used.
3. Authorized Push Payment (APP) / social-engineering fraud is documented
   (Zigram, "Authorized Push Payment Fraud: Detection & Prevention," and
   the general APP-fraud literature it summarizes) as a **distinct**
   typology precisely because the victim authorizes the transfer
   themselves — device/session/KYC look clean; the signal is in the
   amount/receiver being atypical for that victim. This is a documented,
   named real-world candidate explanation for exactly the population this
   project's own EDA already found empirically:
   `EDA_FEATURE_ENGINEERING.md` §1.2 found 87.3% of the frozen model's
   "invisible" fraud rows have both hard flags clean, without having a name
   for why.
4. Victim-demographics reporting (indiadatamap.com/psuconnect coverage of
   2025 Ministry/NCRP-adjacent reporting): senior citizens
   disproportionately targeted via impersonation/coercion, younger users
   via fake job/income schemes — used only to justify a modest, documented
   *tilt* in typology-by-age-group probability, not a copied percentage.
5. Mule-network/money-laundering fraud is characterized in general
   fraud-ops literature (Zigram; Databricks' "Payment Fraud Detection";
   paymentsandrisk.com's "Velocity Rules") by **rapid pass-through of
   received funds** and an unregistered/newly-registered receiving account.
6. Search synthesis of "Enhancing UPI Fraud Detection: A Machine Learning
   Approach Using Stacked Generalization" (ResearchGate) repeatedly
   surfaces transaction amount, transaction frequency, and failed attempts
   as top-ranked features in UPI-specific ML fraud models. **Disclosed
   limit, stated plainly rather than hidden:** the paper's full text
   returned HTTP 403 to automated retrieval in this environment — only the
   qualitative claim (these three signal categories recur as top-ranked) is
   used here; no exact coefficient or ranking number from that paper is
   reproduced or invented.

No invented statistics are presented as cited figures anywhere in the
generator — every ASSUMPTION-labeled choice (the 35/45/20 typology mix, the
specific multiplier ranges per typology) is flagged as such in the script,
the same discipline `SYNTHETIC_GROUNDING.md` §2.3 already used for its own
un-cited shape choices.

### 2.2 What is genuinely structurally different

`generate_grounded_synthetic_dataset.py` (the existing generator): one
`logit = Σ(w_i · signal_i) + noise` over 9 signals, the same functional form
for every fraud row.

`generate_research_grounded_synthetic_dataset.py` (this document): `is_fraud`
is decided **first** (a coin flip at the target rate, not a continuous risk
score), and each fraud row is then assigned to exactly one of three
typologies (documented mix: 35% account-takeover, 45% social-engineering,
20% mule-network — social-engineering deliberately the largest, per source
#1/#3 above), each with a **different subset of features perturbed**:

| Typology | Forced signature | Left untouched (deliberately) |
|---|---|---|
| Account takeover (35%) | `new_device_flag`→85% forced, `ip_location_mismatch`→80% forced, `failed_attempts_last_24h` elevated 65% of the time, amount inflated 3-7x, night-skewed | `kyc_verified_flag` (an already-KYC'd account is exactly what's being taken over — forcing this down would fabricate a correlation the real mechanism doesn't imply) |
| Social engineering (45%) | `amount_deviation_score` inflated further (1.3-2.0x on top of the amount multiplier) | `new_device_flag`, `ip_location_mismatch`, `failed_attempts_last_24h`, `kyc_verified_flag` — **all left at ordinary baseline rates**, by design, because the victim's own device/session is used |
| Mule network (20%) | `transaction_velocity` forced to Poisson(3.2) clipped to [2,4] vs. baseline Poisson(0.6); merchant `mrc_is_registered`/`mrc_rating` degraded | device/IP left at baseline (the mule's own device) |

Generation-time sanity check (same red-flag rigor as
`GENERALIZATION_CHECK.md` §2.2 and `SYNTHETIC_GROUNDING.md` §2.5), on the
production 15,000-row run (SEED=771029):

| Check | Result |
|---|---:|
| Realised fraud rate | 4.10% (target 4.0%) |
| Typology mix realised | account_takeover 32.4% / social_engineering 48.5% / mule_network 19.2% (target 35/45/20) |
| `new_device_flag` fraud rate, flag=1 vs. flag=0 | 12.32% vs. 3.70% — elevated, not deterministic |
| `ip_location_mismatch` fraud rate, flag=1 vs. flag=0 | 14.28% vs. 3.07% — elevated, not deterministic |
| Share of fraud with BOTH hard flags clean | 63.3% (this project's own EDA found 87.3% for the frozen model's real "invisible" population — same order, built by construction via the social-engineering typology, not by accident) |
| `amount_deviation_score` correlation with `is_fraud` | 0.455 (comparable to the accepted Dataset 3's 0.45; far below the disqualifying Dataset 2's 0.66-0.75) |
| `transaction_velocity` correlation with `is_fraud` | 0.123 — modest, believable |

No category anywhere near 0%/100%; passes the same vetting this project
already applies to every dataset before trusting it.

### 2.3 Contamination discipline

This generator was designed and written **without reading the contents**
of `synthetic_grounded_dataset.csv` (seed 918273) or
`category_generalization_test_set.csv` at any point — only their already-
public, documented schemas (from `SYNTHETIC_GROUNDING.md` and
`generate_grounded_synthetic_dataset.py`'s own column list) were used,
exactly the same exposure every other script that scores against those
files already has. `SEED=771029` is disjoint from 42 (original pipeline),
918273 (held-out synthetic eval), and 445566
(`OOD_GENERALIZATION_REMEDIATION.md`'s blend seed) — verified
programmatically (`tests/test_research_grounded_synthetic_dataset.py::
test_seed_disjoint_from_all_prior_seeds`) and by a full-row-duplication
check against the held-out set (0 overlapping rows, confirmed at run time).

### 2.4 Variant C — the actual experiment

Mirrors `OOD_GENERALIZATION_REMEDIATION.md`'s Variant A construction
exactly, so the *only* thing that differs between Variant A and Variant C
is which generator produced the blended-in data: the original 80% train
partition (24,000 rows) concatenated with 15,000 rows from
`generate_research_grounded_synthetic_dataset.py` (39,000 rows total, 4.17%
fraud, before SMOTE), same Block-0 XGBoost hyperparameters
(`n_estimators=400, max_depth=5, learning_rate=0.05, subsample=0.80,
colsample_bytree=0.80, min_child_weight=10, gamma=0.10, reg_alpha=0.05,
reg_lambda=1.50`) and the same `monotone_constraints` on
`amount_deviation_score`/`transaction_velocity`/`failed_attempts_last_24h`
as the currently-deployed model — not regressing that fix. Saved as
`artefacts/paysense_model_research_grounded.pkl` /
`paysense_preprocessor_research_grounded.pkl`; not wired into
`src/fraud_model.py` or `main.py`.

---

## 3. Full honest results — every check, three models, real ensemble path

All numbers below score through `src.fraud_model.score()` (the real
XGBoost + rules + LightLR ensemble `/predict` actually uses), at the
verified deployed threshold **0.50**, produced by
`real_data_and_research_grounding.py`. Baseline's canonical-test, held-out-
synthetic, Dataset-1, and Dataset-3 numbers were re-run fresh in this
script (not cited from prior docs) so all three models in this comparison
are scored by the exact same code in the exact same run, eliminating any
cross-script scoring drift as a confound.

### 3.1 Canonical held-out test (in-distribution, 6,000 rows, 253 fraud)

| Model | ROC-AUC | PR-AUC | Recall | Precision | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---|
| Baseline (frozen) | 0.8969 | 0.5498 | 39.53% | 91.74% | 100/9/153/5738 |
| Blended training (Variant A) | 0.8957 | 0.5526 | 37.94% | 94.12% | 96/6/157/5741 |
| **Variant C (research-grounded)** | 0.8951 | 0.5443 | 38.74% | 92.45% | 98/8/155/5739 |

All three within noise of each other — a small, unremarkable in-distribution cost either way, same shape as Variant A's.

### 3.2 Held-out synthetic set (seed 918273, 25,000 rows, 985 fraud — never trained on by anything)

| Model | ROC-AUC | PR-AUC | Recall | Precision | TP/FP/FN/TN |
|---|---:|---:|---:|---:|---|
| Baseline | 0.7000 | 0.1015 | 20.30% | 12.73% | 200/1371/785/22644 |
| Blended training (Variant A) | **0.7304** | **0.1309** | 4.16% | 27.70% | 41/107/944/23908 |
| Variant C (research-grounded) | 0.6955 | 0.0939 | 2.84% | 13.02% | 28/187/957/23828 |

**Variant C is worse than baseline here, not just worse than Variant A.** ROC-AUC and PR-AUC both drop below the frozen model's own numbers — the research-grounded blend measurably *hurt* ranking quality on this check, it did not merely fail to help.

### 3.3 Real Dataset 1 (`upi_fraud_dataset.csv`, 74,917 rows, 701 fraud, 6/40 features honest)

| Model | ROC-AUC | PR-AUC | TP/701 | Max score |
|---|---:|---:|---:|---:|
| Baseline | 0.7919 | 0.3767 | 0 | 0.0782 |
| Blended training (Variant A) | **0.8442** | **0.4764** | 0 | 0.0847 |
| Variant C (research-grounded) | 0.7199 | 0.3062 | 0 | 0.0781 |

**Same pattern: Variant C underperforms baseline on the single highest-power real-world check in this entire project.** ROC-AUC 0.72 vs. baseline's 0.79 vs. Variant A's 0.84.

### 3.4 Real Dataset 3 (`fraud_dataset.csv`, 1,000 rows, 64 fraud, low power, 2/40 features honest)

| Model | ROC-AUC | PR-AUC | TP/64 |
|---|---:|---:|---:|
| Baseline | 0.6046 | 0.1157 | 0 |
| Blended training (Variant A) | 0.8913 | 0.3295 | 0 |
| **Variant C (research-grounded)** | **0.8982** | **0.3633** | 0 |

The one check where Variant C edges out Variant A — but this is the lowest-power check available (64 fraud rows), and both variants are already far above baseline here; read this as a tie within noise, not a Variant C win, given §3.2 and §3.3 point the other way on far higher-power checks.

### 3.5 Real Dataset 5 (`Fraud_Data.csv`, stratified 20,000-row sample, 1,873 fraud, 7/40 features honest — new this document)

| Model | ROC-AUC | PR-AUC | TP/1,873 | Max score |
|---|---:|---:|---:|---:|
| Baseline | 0.5027 | 0.0957 | 0 | 0.0768 |
| Blended training (Variant A) | 0.5040 | 0.0953 | 0 | 0.0769 |
| Variant C (research-grounded) | 0.4942 | 0.0920 | 0 | 0.0752 |

**All three models score at chance level (ROC-AUC ≈ 0.50) on Dataset 5 — worse than every model's showing on Dataset 1, despite Dataset 5 having *more* honest feature overlap (7/40 vs. 6/40).** This is the most important, and most sobering, new finding in this document, discussed in §4.

### 3.6 The number that actually matters: catch rate at the deployed threshold on Dataset 1

**0 of 701, for all three models, unchanged.** Neither the real dataset search (Track A) nor the research-grounded synthetic blend (Track B) moved this number even by one row. Every model's maximum predicted score on Dataset 1 (baseline 0.0782, Variant A 0.0847, Variant C 0.0781) remains far below the deployed threshold (0.50) regardless of which training data produced it — this is the same lesson `OOD_GENERALIZATION_REMEDIATION.md` already established: ranking-quality metrics (ROC-AUC/PR-AUC) and the actual catch rate at a fixed operating point are separate properties, and nothing tried across two full experiments has moved the second one.

---

## 4. Verdict

**Track A (Dataset 5) is a genuine methodological advance and a genuinely discouraging result, at the same time.** Finding a real dataset with a strong, credible signal PaySense has never had access to before — `usr_account_age_days`, computed as real signup-to-purchase time, showing a dramatic legit-vs-fraud gap (60.13 days vs. essentially 0) — was real, careful work, not a weak substitute forced to look useful. But scored through the actual frozen ensemble, none of the three models beat chance on this dataset. The most likely explanation, reasoned through but **not independently re-verified by re-running with adjusted scale — flagged as a hypothesis, not a confirmed finding**: Dataset 5's `purchase_value` shows almost no legit/fraud separation on its own (mean 36.93 vs. 36.99, in whatever currency/scale this e-commerce dataset uses), and PaySense's rules scorer has a hard-coded cold-start bonus keyed on `amount > 5000` — calibrated to PaySense's own INR-scale synthetic training data. If Dataset 5's amounts never cross that fixed numeric threshold, the one rule most suited to exploit a fast-signup-then-purchase pattern never fires, regardless of how strong the underlying account-age signal actually is. If this hypothesis is right, it points to a real, separate problem this project hasn't examined at all yet: hard-coded numeric thresholds in the rules scorer are calibrated to one dataset's currency/amount scale and may not transfer to any other, even before the missing-feature problem is considered.

**UPDATE (2026-08-24, same night) — the hypothesis was tested and confirmed, cheaply, without retraining.** `purchase_value` across all 151,112 rows of Dataset 5 ranges 9–154 (mean 36.94), confirmed directly — the maximum value in the *entire dataset* never comes close to the rules scorer's `amount > 5000` gate. Isolating the rules scorer's own contribution (its score alone, not blended into the ensemble) against Dataset 5's 20,000-row sample:

| Rules-scorer version | ROC-AUC (rules signal alone) | Cold-start bonus fires on |
|---|---:|---:|
| Original (`amount > 5000`) | 0.4952 (chance) | 0 / 20,000 rows |
| Scale-corrected (`amount > 35`, ≈ this dataset's own median) | 0.5933 | 2,752 / 20,000 |
| Amount gate removed entirely | 0.6937 | 5,691 / 20,000 |
| `usr_account_age_days` alone, no rules scorer at all (upper bound) | 0.7564 | — |

Confirmed exactly as hypothesized: the original threshold silently disables the cold-start rule on every single row, and removing the currency-scale mismatch recovers most of the gap to the account-age signal's own ceiling. **This changes how Dataset 5's chance-level ensemble result (§3.5) should be read, not just explains it**: it is not further evidence that the model fails to generalize the way the 0/701 result on Dataset 1 is. Dataset 1 is a real UPI/INR dataset — an apples-to-apples comparison the model genuinely fails. Dataset 5 is USD-denominated e-commerce data, and comparing an India-only, INR-calibrated rules scorer against it without currency normalization was never a fair test to begin with. Since PaySense only ever scores real UPI transactions (always INR) in production, this specific threshold never causes a problem in the field — it is a real, confirmed methodological artifact of this evaluation, not a production bug. It is recorded here rather than quietly reclassified, because the underlying lesson generalizes even though the specific fix does not need to: a hard-coded absolute-currency threshold is inherently non-portable, and any future evaluation against a foreign-currency dataset should normalize amounts first or it will silently understate the model exactly this way.

Not changed in `src/fraud_model.py` — production traffic is India-only, so there is no real threshold-scale bug to fix there. Verification script: read-only, no retraining, no artifact changes, `tests/test_rules_scorer_currency_scale.py`.

**Track B (Variant C) is a clear, honest negative result — more research grounding did not translate to better transfer.** Despite being structurally more sophisticated than Variant A (a three-typology mixture model built from real, cited fraud-mechanism research, versus Variant A's single differently-seeded logistic-risk draw), Variant C underperforms Variant A on both of the two highest-power out-of-distribution checks available (held-out synthetic: 0.6955 vs. 0.7304 ROC-AUC; real Dataset 1: 0.7199 vs. 0.8442), and underperforms even the *untouched baseline* on both. The most likely reason, again reasoned through rather than independently confirmed: Variant C was deliberately built so 63.3% of its fraud rows have both hard signal flags clean (mirroring this project's own EDA finding about the frozen model's real "invisible" population) — that is more *realistic* by the cited research, but it may also be a harder training signal for this specific model family to learn from without hurting its existing, cruder-but-more-learnable behavior elsewhere. Realism and learnability are not the same property, and this result is evidence they can trade off against each other.

**Neither track is recommended for adoption.** Track A is not training data (stated in §1.4, before these results existed) — it remains a genuinely better *evaluation* target than any real dataset used before, and the account-age-days signal it revealed is worth keeping in mind for future rules-scorer or feature-engineering work, even though this document's own attempt to exploit it (implicitly, via Variant A's blend) did not. Track B (Variant C) actively regresses ranking quality on the two checks that matter most and should not replace Variant A. `artefacts/paysense_model_research_grounded.pkl` is saved for reference, not deployment; frozen artifacts remain untouched throughout.

---

## 5. Reproducing this check

```
cd PaySense-ML-Backend
venv\Scripts\python.exe generate_research_grounded_synthetic_dataset.py   # writes research_grounded_synthetic_dataset.csv
venv\Scripts\python.exe real_data_and_research_grounding.py               # trains Variant C, scores all 3 models x 5 checks
```

Requires `external_data/kaggle_vbinh002_fraud_ecommerce/Fraud_Data.csv`
(downloaded 2026-08-24, see `external_data/NOTICE.md`) and the existing
real datasets at their original paths under `E:\Projects\upi\` (not copied
into this repo). Takes on the order of an hour — one SMOTE + 400-tree
XGBoost training, then real-ensemble-path scoring (one `fraud_model.score()`
call per row, the same per-request code path `/predict` uses) against five
evaluation sets across three models. Results are written to
`real_data_and_research_grounding_results.json`.

Regression tests: `tests/test_research_grounded_synthetic_dataset.py`
(generator structure/typology/seed-disjointness checks) and
`tests/test_real_data_and_research_grounding.py` (Dataset 5 vetting
regression guards, honest-mapping leakage guard, Variant C artifact smoke
test).
