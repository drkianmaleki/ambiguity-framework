# Raw Data

All datasets live in this directory (`data/raw/`). Processed/cleaned versions go in `data/processed/`.

> **These files are not included in the repository.** Download each dataset from the links below and place the CSV files here before running any experiments.

---

## Quick Reference

| File | Domain | Rows | Features | Target | Positive rate | Balance | Size | Experiment |
|------|--------|-----:|--------:|--------|:-------------:|---------|-----:|-----------|
| `creditcard.csv` | Finance | 284,807 | 30 | `Class` | 0.17% | Severe | 143.8 MB | `run_credit.py` |
| `UCI_Credit_Card.csv` | Finance | 30,000 | 23 | `default.payment.next.month` | 22.1% | Moderate | 2.7 MB | `run_uci_credit.py` |
| `heart.csv` | Healthcare | 918 | 11 | `HeartDisease` | 55.3% | Balanced | <1 MB | `run_heart.py` |
| `heart_disease_health_indicators_BRFSS2015.csv` | Healthcare | 253,680 | 21 | `HeartDiseaseorAttack` | 9.0% | Moderate | 21.7 MB | `run_brfss.py` |
| `hospital_readmissions.csv` | Healthcare | 25,000 | 16 | `readmitted` | 46.4% | Balanced | 1.9 MB | `run_readmissions.py` |
| `nsl_kdd_dataset.csv` | Cybersecurity | 4,431 | 41 | `label` | 80.0% (attack) | Artificial | 3.4 MB | `run_nslkdd.py` |

> **Balance guide:** Balanced ≈ 40–60% positive; Moderate ≈ 10–40%; Severe < 10%; Artificial = dataset was intentionally equalized.

---

## General Preprocessing Notes

These steps are applied consistently by `experiments/data_loader.py`:

| Step | Datasets affected |
|------|-------------------|
| Drop ID column | `UCI_Credit_Card.csv` (`ID`) |
| Label-encode string categoricals | `heart.csv`, `hospital_readmissions.csv`, `nsl_kdd_dataset.csv` |
| Ordinal age bracket → midpoint int | `hospital_readmissions.csv` (`age`) |
| Binarize multi-class label | `nsl_kdd_dataset.csv` — `normal`→0, all attacks→1 |
| Binarize string target | `hospital_readmissions.csv` — `yes`→1, `no`→0 |
| No encoding needed | `creditcard.csv`, `heart_disease_health_indicators_BRFSS2015.csv` |

All `load_dataset()` calls return `(X: ndarray, y: ndarray, feature_names: list)` with `X` fully numeric and `y` integer-valued `{0, 1}`.

---

## Dataset Details

### Credit Card Fraud Detection (`creditcard.csv`)
- **Download:** https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud
- **Source:** ULB Machine Learning Group / Worldline
- **Experiment:** `python experiments/run_credit.py`
- **Rows:** 284,807 transactions (September 2013, European cardholders)
- **Target:** `Class` — 0=legitimate (284,315), 1=fraud (492)
- **Class balance:** 99.83% / 0.17% — severely imbalanced
- **Features (30):** `Time`, `Amount`, `V1`–`V28` (PCA-transformed; original features withheld for confidentiality)
- **Citation:** Dal Pozzolo et al. (2015). Calibrating Probability with Undersampling for Unbalanced Classification. IEEE CIDM.

---

### UCI Credit Card Default (`UCI_Credit_Card.csv`)
- **Download:** https://www.kaggle.com/datasets/uciml/default-of-credit-card-clients-dataset
- **Source:** UCI ML Repository — Taiwan credit card clients (Apr–Sep 2005)
- **Experiment:** `python experiments/run_uci_credit.py`
- **Rows:** 30,000 clients
- **Target:** `default.payment.next.month` — 0=no default (23,364), 1=default (6,636)
- **Class balance:** 77.9% / 22.1%
- **Features (23, after dropping `ID`):**
  - Credit: `LIMIT_BAL`
  - Demographics: `SEX`, `EDUCATION`, `MARRIAGE`, `AGE`
  - Repayment status Apr–Sep 2005: `PAY_0`, `PAY_2`–`PAY_6`
  - Bill amounts Apr–Sep 2005: `BILL_AMT1`–`BILL_AMT6`
  - Payment amounts Apr–Sep 2005: `PAY_AMT1`–`PAY_AMT6`
- **Citation:** Lichman, M. (2013). UCI Machine Learning Repository. UC Irvine.

---

### Heart Failure Prediction (`heart.csv`)
- **Download:** https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction
- **Source:** fedesoriano (Kaggle) — combined from 5 UCI heart disease datasets
- **Experiment:** `python experiments/run_heart.py`
- **Rows:** 918 observations
- **Target:** `HeartDisease` — 0=normal (410), 1=disease (508)
- **Class balance:** 44.7% / 55.3%
- **Features (11):**
  - Numeric: `Age`, `RestingBP`, `Cholesterol`, `MaxHR`, `Oldpeak`
  - Binary: `FastingBS`
  - Categoricals (label-encoded): `Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`
- **Citation:** fedesoriano. (September 2021). Heart Failure Prediction Dataset. Kaggle.

---

### BRFSS 2015 Heart Disease Health Indicators (`heart_disease_health_indicators_BRFSS2015.csv`)
- **Download:** https://www.kaggle.com/datasets/alexteboul/heart-disease-health-indicators-dataset
- **Source:** CDC BRFSS 2015 annual telephone survey
- **Experiment:** `python experiments/run_brfss.py`
- **Rows:** 253,680 survey responses
- **Target:** `HeartDiseaseorAttack` — 0=no (229,787), 1=yes (23,893)
- **Class balance:** 91.0% / 9.0%
- **Features (21):** All numeric binary/ordinal survey responses — no encoding required.
- **Citation:** Centers for Disease Control and Prevention. Behavioral Risk Factor Surveillance System Survey Data. 2015.

---

### Hospital Readmissions (`hospital_readmissions.csv`)
- **Download:** https://www.kaggle.com/datasets/dubradave/hospital-readmissions
- **Source:** 10-year diabetic inpatient encounters
- **Experiment:** `python experiments/run_readmissions.py`
- **Rows:** 25,000 records
- **Target:** `readmitted` — no (13,246), yes (11,754)
- **Class balance:** 53.6% / 46.4%
- **Features (16):**
  - Numeric: `time_in_hospital`, `n_lab_procedures`, `n_procedures`, `n_medications`, `n_outpatient`, `n_inpatient`, `n_emergency`
  - Categoricals (label-encoded): `age`, `medical_specialty`, `diag_1`, `diag_2`, `diag_3`, `glucose_test`, `A1Ctest`, `change`, `diabetes_med`
- **Note:** `age` is an ordinal bracket (e.g. `[50-60)`) mapped to midpoint integer. Target binarized from `yes`/`no` strings.

---

### NSL-KDD Intrusion Detection (`nsl_kdd_dataset.csv`)
- **Download:** https://www.kaggle.com/datasets/programmer3/nsl-kdd-intrusion-detection-dataset
- **Source:** Simulated NSL-KDD benchmark network traffic
- **Experiment:** `python experiments/run_nslkdd.py`
- **Rows:** 4,431 records
- **Target:** `label` — multi-class (`normal`, `DoS`, `Probe`, `U2R`, `R2L`)
- **Binary framing:** `normal`→0, any attack→1 (results in 80% positive rate)
- **Features (41):** TCP/IP connection properties; `protocol_type`, `service`, `flag` require label encoding
- **Note:** This Kaggle version simulates the NSL-KDD structure with 4,431 rows. The original full dataset is no longer publicly available from its primary source.
