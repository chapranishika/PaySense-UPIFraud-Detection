# External data sources used for the V2 category classifier attempt

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
