# External data sources

## kaggle_vbinh002_fraud_ecommerce/ — used for REAL_DATA_AND_RESEARCH_GROUNDING.md (Track A)

- Source: https://www.kaggle.com/datasets/vbinh002/fraud-ecommerce
- Owner: vbinh002
- License: unspecified on Kaggle ("None") — used here only for internal
  model evaluation in a student project, the same way this project already
  scores against other unlicensed/unclear-license local CSVs; not
  redistributed.
- Downloaded 2026-08-24 via Kaggle's public, unauthenticated
  `/api/v1/datasets/download/<owner>/<slug>` endpoint (same mechanism as the
  two category-classifier downloads below).
- Files: `Fraud_Data.csv` (151,112 rows: user_id, signup_time, purchase_time,
  purchase_value, device_id, source, browser, sex, age, ip_address, class),
  `IpAddress_to_Country.csv` (138,846-row IP block → country lookup table,
  not used in the final honest mapping — see REAL_DATA_AND_RESEARCH_GROUNDING.md
  §2 for why: no per-user home-country baseline exists to compare an IP
  geolocation against, so `ip_location_mismatch` cannot be honestly derived
  from it).
- Genuinely real, not templated: a well-known e-commerce fraud benchmark,
  cataloged as real fraud data ("fraudecom") in Amazon Science's Fraud
  Dataset Benchmark paper (arXiv:2208.14417) — 151,112 unique transactions
  (one per user), 9.36% fraud rate (not a round number), no duplicate rows,
  no nulls, no categorical near-determinism (source/browser/sex all within
  8.7-10.5% of the 9.36% base rate). See
  REAL_DATA_AND_RESEARCH_GROUNDING.md §2 for the full vetting write-up.

## kaggle_bhavya_financial_transaction/ and kaggle_coderanand_indian_banking/ — used for the V2 category classifier attempt

See `CATEGORY_CLASSIFIER_V2_ATTEMPT.md` for the full account. Both datasets
below were downloaded on 2026-08-23 via Kaggle's public, unauthenticated
`/api/v1/datasets/download/<owner>/<slug>` endpoint (no `kaggle.json`
credential was available or needed for these two public datasets).

## kaggle_coderanand_indian_banking/

- Source: https://www.kaggle.com/datasets/coderanand/indian-banking-transaction-text-dataset
- Owner: Coder_Anand
- License: Apache 2.0
- Files: `financial_transaction_train.csv` (10,000 rows), `financial_transaction_test.csv` (1,000 rows)
- Columns: `Transaction_Text`, `Label` (Food/Travel/EMI/Investment/Shopping — same 5 classes as FinText-6K)
- **Important**: the dataset's own description states it is "**11,000
  synthetic** financial transaction narrations." Verified independently:
  digit/ref-collapsed normalization finds only **29 unique narration
  phrases** across all 11,000 rows (5-7 per class) — this is templated
  synthetic data from a different generator than FinText-6K, not real bank
  SMS text. Used here as an additional, differently-vocabularied synthetic
  source (new brand names: Zerodha, Groww, IRCTC, IndiGo, MakeMyTrip,
  RedBus, Myntra, Reliance Digital, IKEA, McDonald's, Bajaj Finserv, PPF —
  none of which appear in FinText-6K's 40 templates), not as real-world data.

## kaggle_bhavya_financial_transaction/

- Source: https://www.kaggle.com/datasets/bhavyasingh25/financial-transaction-description-dataset
- Owner: bhavyasingh25
- License: MIT
- Files: `train_transactions.csv` (5,000 rows), `test_transactions.csv` (1,000 rows)
- Columns: `transaction_text`, `category` — **9 classes**: food, travel, emi,
  investment, shopping, entertainment, healthcare, education, utilities.
  Only the 5 that map onto PaySense's taxonomy (food/travel/emi/investment/
  shopping) are used; entertainment/healthcare/education/utilities rows are
  dropped entirely (PaySense's classifier does not have and was not asked to
  gain those classes — forcing them in would require inventing a mapping
  the task brief didn't ask for and CATEGORY_CLASSIFIER.md §"category-
  vocabulary mismatch" already treats as out of scope).
- Also self-evidently synthetic/templated: 28 unique normalized phrases
  across the 5 usable classes (5-7 per class). Same treatment as the
  coderanand dataset above — used for its distinct vocabulary/generator,
  not represented as real-world text.

## Datasets investigated and rejected (documented for honesty, not used)

- `apoorvwatsky/bank-transaction-data` (CC0, real corporate bank statement
  export, 116,201 rows) — genuinely real narrations, but every row is an
  internal fund transfer / settlement between corporate accounts (e.g. "TRF
  FROM Indiaforensic SERVICES", "INDO GIBL ... STL01071"). No row carries
  any consumer spending-category signal (no food/travel/EMI/investment/
  shopping semantics at all) — does not map onto PaySense's taxonomy by any
  honest reading. Rejected, not forced.
- `engreemali/bank-transactions-sms-datasetss` (license unknown, real SMS
  from real UAE bank customers, ADIB, collected for a Khalifa University
  research project, 1,894 rows) — genuinely real bank SMS text with real
  merchant narrations (e.g. "Trx. of AED 40.00 on your a/c ****0535 at
  ZOMATO ORDER DUBAI AE"). This is the most realistic real-world text found
  in this search. But it carries **zero category labels** (`Label` column
  is 100% null in the shipped file) and is UAE/AED-denominated, not Indian/
  UPI. Using it would require hand-labeling every row ourselves, which is
  self-authored labeling dressed up as a "real dataset" and was rejected on
  that basis rather than forced in.
