# Feature validity audit -- source-domain forensics (2026-08-27)

Every one of the 38 kept features answered against 8 real-inference/leakage questions and classified as VALID_SIGNAL / DOMAIN_SHIFT / COLLECTION_ARTIFACT / SYNTHETIC_ARTIFACT / LEAKAGE / UNKNOWN. `legitimate_at_inference` is VERIFIED for all 38 -- every one is a literal field in `main.py`'s real `/predict` request schema (37 required, `mrc_rating` optional), checked directly against the source file, not assumed.

**Classification counts:** SYNTHETIC_ARTIFACT=30, DOMAIN_SHIFT=5, VALID_SIGNAL=3

## `receiver_type`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cramers_v_fraud_vs_legit):** 0.0023
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cramers_v_fraud_vs_legit=0.0023, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `amount`

- **Classification:** DOMAIN_SHIFT
- **Decision:** KEEP (legitimate distributional difference, not an artifact)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.0579
- **Reasoning:** Varies naturally in both sources but with a real distributional difference (effect size 0.818) -- consistent with the two sources representing somewhat different transaction populations (organic real-style data vs. an external synthetic dataset's generation assumptions), not a data-quality problem. Domain shift is not, on its own, a reason to remove a feature -- it is available at real inference time and not evidence of leakage.

## `hour_of_day`

- **Classification:** DOMAIN_SHIFT
- **Decision:** KEEP (legitimate distributional difference, not an artifact)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** -0.1270
- **Reasoning:** Varies naturally in both sources but with a real distributional difference (effect size 0.436) -- consistent with the two sources representing somewhat different transaction populations (organic real-style data vs. an external synthetic dataset's generation assumptions), not a data-quality problem. Domain shift is not, on its own, a reason to remove a feature -- it is available at real inference time and not evidence of leakage.

## `day_of_week`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.0188
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=0.0188, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `is_weekend`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.0069
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=0.0069, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `is_night_transaction`

- **Classification:** DOMAIN_SHIFT
- **Decision:** KEEP (legitimate distributional difference, not an artifact)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.2665
- **Reasoning:** Varies naturally in both sources but with a real distributional difference (effect size 0.499) -- consistent with the two sources representing somewhat different transaction populations (organic real-style data vs. an external synthetic dataset's generation assumptions), not a data-quality problem. Domain shift is not, on its own, a reason to remove a feature -- it is available at real inference time and not evidence of leakage.

## `time_since_last_txn_min`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.0312
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=0.0312, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `transaction_type`

- **Classification:** DOMAIN_SHIFT
- **Decision:** KEEP (legitimate distributional difference, not an artifact)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cramers_v_fraud_vs_legit):** 0.0084
- **Reasoning:** Varies naturally in both sources but with a real distributional difference (effect size 0.676) -- consistent with the two sources representing somewhat different transaction populations (organic real-style data vs. an external synthetic dataset's generation assumptions), not a data-quality problem. Domain shift is not, on its own, a reason to remove a feature -- it is available at real inference time and not evidence of leakage.

## `payment_app`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cramers_v_fraud_vs_legit):** 0.0140
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cramers_v_fraud_vs_legit=0.0140, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `device_type`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cramers_v_fraud_vs_legit):** 0.0062
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cramers_v_fraud_vs_legit=0.0062, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `user_city_tier`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.0144
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=0.0144, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `user_avg_monthly_txn`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** -0.0015
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=-0.0015, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `user_avg_txn_value`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** -0.0070
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=-0.0070, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `user_loyalty_score`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.0206
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=0.0206, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `new_device_flag`

- **Classification:** VALID_SIGNAL
- **Decision:** KEEP (no meaningful source-separability contribution found)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.4527
- **Reasoning:** Varies naturally in both sources with no material distributional difference detected (effect size 0.064) -- behaves like a genuine, source-independent fraud-relevant feature.

## `ip_location_mismatch`

- **Classification:** VALID_SIGNAL
- **Decision:** KEEP (no meaningful source-separability contribution found)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.2504
- **Reasoning:** Varies naturally in both sources with no material distributional difference detected (effect size 0.056) -- behaves like a genuine, source-independent fraud-relevant feature.

## `failed_attempts_last_24h`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** KEEP (source-artifact explains separability, but shows real organic fraud signal -- not removed)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.1996
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact, not a real-world pattern (the supplement is schema-bridged from an external synthetic dataset that repeats one templated profile). This alone explains why it separates organic from supplement almost perfectly. However, within anchor-only (organic) rows, this feature shows a non-negligible association with is_fraud (cohens_d_fraud_vs_legit=0.1996) -- real organic predictive value independent of the source-leakage question. Removing it would discard genuine signal to fix a dataset-generation problem, not a feature-validity problem.

## `transaction_velocity`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** KEEP (source-artifact explains separability, but shows real organic fraud signal -- not removed)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.4346
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact, not a real-world pattern (the supplement is schema-bridged from an external synthetic dataset that repeats one templated profile). This alone explains why it separates organic from supplement almost perfectly. However, within anchor-only (organic) rows, this feature shows a non-negligible association with is_fraud (cohens_d_fraud_vs_legit=0.4346) -- real organic predictive value independent of the source-leakage question. Removing it would discard genuine signal to fix a dataset-generation problem, not a feature-validity problem.

## `amount_deviation_score`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** KEEP (source-artifact explains separability, but shows real organic fraud signal -- not removed)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.1424
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact, not a real-world pattern (the supplement is schema-bridged from an external synthetic dataset that repeats one templated profile). This alone explains why it separates organic from supplement almost perfectly. However, within anchor-only (organic) rows, this feature shows a non-negligible association with is_fraud (cohens_d_fraud_vs_legit=0.1424) -- real organic predictive value independent of the source-leakage question. Removing it would discard genuine signal to fix a dataset-generation problem, not a feature-validity problem.

## `recurring_payment_flag`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** -0.0194
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=-0.0194, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `balance_after_transaction`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.0617
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=0.0617, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `transaction_frequency_score`

- **Classification:** VALID_SIGNAL
- **Decision:** KEEP (no meaningful source-separability contribution found)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.1386
- **Reasoning:** Varies naturally in both sources with no material distributional difference detected (effect size 0.004) -- behaves like a genuine, source-independent fraud-relevant feature.

## `txn_success_flag`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** -0.0147
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=-0.0147, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `kyc_verified_flag`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** KEEP (source-artifact explains separability, but shows real organic fraud signal -- not removed)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** -0.1861
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact, not a real-world pattern (the supplement is schema-bridged from an external synthetic dataset that repeats one templated profile). This alone explains why it separates organic from supplement almost perfectly. However, within anchor-only (organic) rows, this feature shows a non-negligible association with is_fraud (cohens_d_fraud_vs_legit=-0.1861) -- real organic predictive value independent of the source-leakage question. Removing it would discard genuine signal to fix a dataset-generation problem, not a feature-validity problem.

## `usr_age_group`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cramers_v_fraud_vs_legit):** 0.0098
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cramers_v_fraud_vs_legit=0.0098, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `usr_home_city_tier`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.0144
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=0.0144, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `usr_account_age_days`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** -0.0005
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=-0.0005, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `usr_linked_bank_count`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** -0.0038
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=-0.0038, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `usr_avg_monthly_txn_profile`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** -0.0015
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=-0.0015, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `usr_avg_txn_value_profile`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** -0.0070
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=-0.0070, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `usr_preferred_app`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cramers_v_fraud_vs_legit):** 0.0164
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cramers_v_fraud_vs_legit=0.0164, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `usr_preferred_device`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cramers_v_fraud_vs_legit):** 0.0055
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cramers_v_fraud_vs_legit=0.0055, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `usr_is_high_risk`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** KEEP (source-artifact explains separability, but shows real organic fraud signal -- not removed)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.1669
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact, not a real-world pattern (the supplement is schema-bridged from an external synthetic dataset that repeats one templated profile). This alone explains why it separates organic from supplement almost perfectly. However, within anchor-only (organic) rows, this feature shows a non-negligible association with is_fraud (cohens_d_fraud_vs_legit=0.1669) -- real organic predictive value independent of the source-leakage question. Removing it would discard genuine signal to fix a dataset-generation problem, not a feature-validity problem.

## `mrc_category`

- **Classification:** DOMAIN_SHIFT
- **Decision:** KEEP (legitimate distributional difference, not an artifact)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cramers_v_fraud_vs_legit):** 0.0322
- **Reasoning:** Varies naturally in both sources but with a real distributional difference (effect size 0.848) -- consistent with the two sources representing somewhat different transaction populations (organic real-style data vs. an external synthetic dataset's generation assumptions), not a data-quality problem. Domain shift is not, on its own, a reason to remove a feature -- it is available at real inference time and not evidence of leakage.

## `mrc_size`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cramers_v_fraud_vs_legit):** 0.0068
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cramers_v_fraud_vs_legit=0.0068, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `mrc_avg_daily_txn`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.0523
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=0.0523, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `mrc_is_registered`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** 0.0330
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=0.0330, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

## `mrc_rating`

- **Classification:** SYNTHETIC_ARTIFACT
- **Decision:** REMOVE (source-artifact, negligible organic fraud signal)
- **Available at real SMS inference time:** Yes (verified: required/optional field in main.py request schema)
- **Within-anchor fraud-signal (cohens_d_fraud_vs_legit):** -0.0501
- **Reasoning:** Constant (or effectively constant) in the supplement source -- a synthetic-generation artifact. Within anchor-only rows, this feature shows negligible association with is_fraud (cohens_d_fraud_vs_legit=-0.0501, below the 0.1 effect-size threshold) -- no established organic predictive value to weigh against removing a clear source-separability driver.

