# Model benchmark -- clean organic evaluation (2026-08-27)

All models share the identical anchor-only (organic) 60/20/20 split (`random_state=42`), fit preprocessing on TRAIN only, select threshold on VALIDATION only, and are evaluated on the SAME untouched final TEST set exactly once. Text-based model families are not included -- `paysense_master_dataset.csv` has no text field (see `feature_audit.md`).

| Model | Features | Threshold | ROC-AUC | PR-AUC | Precision | Recall | F1 | Latency (ms/row) | Business constraint met |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| A_XGBoost_current | structured (shared ordinal-encoded) | 0.1 | 0.7050 | 0.0945 | 8.82% | 21.05% | 0.1243 | 0.0050 | No |
| D_RandomForest | structured (shared ordinal-encoded) | 0.15 | 0.7210 | 0.0860 | 8.61% | 27.63% | 0.1313 | 0.0342 | No |
| E_LightGBM | structured (shared ordinal-encoded) | 0.1 | 0.6904 | 0.0980 | 10.34% | 13.82% | 0.1183 | 0.0203 | No |
| F_CatBoost | structured (native categorical, no SMOTE -- auto_class_weights='Balanced') | 0.35 | 0.6628 | 0.0785 | 12.56% | 18.42% | 0.1493 | 0.0135 | No |

## 5-fold cross-validation (TRAIN+VAL only, never touches TEST)

| Model | Mean PR-AUC | Std PR-AUC |
|---|---:|---:|
| A_XGBoost_current | 0.0827 | 0.0053 |
| D_RandomForest | 0.0963 | 0.0053 |
| E_LightGBM | 0.0806 | 0.0048 |

## Precision at fixed recall targets (threshold selected on validation only)


**A_XGBoost_current**

| Recall target | Achievable on validation | Threshold | Test precision | Test recall |
|---|---|---:|---:|---:|
| recall_50 | No | -- | -- | -- |
| recall_60 | No | -- | -- | -- |
| recall_75 | No | -- | -- | -- |
| recall_80 | No | -- | -- | -- |

**D_RandomForest**

| Recall target | Achievable on validation | Threshold | Test precision | Test recall |
|---|---|---:|---:|---:|
| recall_50 | Yes | 0.05 | 6.06% | 86.84% |
| recall_60 | Yes | 0.05 | 6.06% | 86.84% |
| recall_75 | Yes | 0.05 | 6.06% | 86.84% |
| recall_80 | Yes | 0.05 | 6.06% | 86.84% |

**E_LightGBM**

| Recall target | Achievable on validation | Threshold | Test precision | Test recall |
|---|---|---:|---:|---:|
| recall_50 | No | -- | -- | -- |
| recall_60 | No | -- | -- | -- |
| recall_75 | No | -- | -- | -- |
| recall_80 | No | -- | -- | -- |

**F_CatBoost**

| Recall target | Achievable on validation | Threshold | Test precision | Test recall |
|---|---|---:|---:|---:|
| recall_50 | Yes | 0.1 | 5.61% | 60.53% |
| recall_60 | Yes | 0.05 | 4.80% | 86.18% |
| recall_75 | Yes | 0.05 | 4.80% | 86.18% |
| recall_80 | No | -- | -- | -- |

## Where is the signal coming from? (top-10 feature importance)


**A_XGBoost_current**: is_night_transaction (0.184), transaction_velocity (0.104), new_device_flag (0.101), recurring_payment_flag (0.067), is_weekend (0.067), mrc_size (0.050), usr_linked_bank_count (0.045), device_type (0.040), receiver_type (0.033), kyc_verified_flag (0.032)

**D_RandomForest**: transaction_velocity (0.140), is_night_transaction (0.103), new_device_flag (0.100), mrc_size (0.059), recurring_payment_flag (0.051), usr_linked_bank_count (0.048), failed_attempts_last_24h (0.040), usr_home_city_tier (0.037), receiver_type (0.034), user_city_tier (0.032)

**E_LightGBM**: time_since_last_txn_min (950.000), balance_after_transaction (759.000), usr_account_age_days (709.000), transaction_frequency_score (668.000), user_loyalty_score (666.000), amount (608.000), user_avg_monthly_txn (534.000), user_avg_txn_value (515.000), mrc_category (399.000), mrc_avg_daily_txn (390.000)

**F_CatBoost**: user_loyalty_score (7.438), balance_after_transaction (6.756), amount (6.542), usr_account_age_days (6.392), time_since_last_txn_min (6.349), transaction_frequency_score (6.083), mrc_avg_daily_txn (4.075), hour_of_day (4.047), day_of_week (3.799), mrc_rating (3.663)
