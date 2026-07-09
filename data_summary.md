# ANNITIA Data Challenge 2026 — Dataset Summary

## Overview

The ANNITIA Data Challenge 2026 provides a fully synthetic, longitudinal clinical dataset designed for AI-based risk stratification in metabolic dysfunction-associated steatotic liver disease (MASLD). The dataset models real-world hospital data while preserving patient anonymity and privacy.

---

## Dataset Structure

- **Patient Cohort:** Thousands of synthetic patients with MASLD, based on anonymized records from Pitié-Salpêtrière Hospital, Paris.
- **Longitudinal Data:**  
  - Up to **22 visits** (time points) per patient  
  - Visits span years **2001–2025**  
  - Each visit has repeated measurements of clinical features and non-invasive liver tests (NITs).
- **Synthetic Data:** All clinical events, biomarker values, and outcomes are synthesized to prevent re-identification.

---

## Features

### Patient-Visit Level Features

- **subject_id:** Unique identifier for each patient
- **visit_number / visit_date:** Indicates time sequence and age at each data point
- **age:** Patient age at each visit
- **BMI:** Body Mass Index
- **Cardiometabolic markers:** (details may include blood pressure, glucose, etc.)

### Non-Invasive Tests (NITs)  
(at each time point; maximum available per patient)

- **FibroScan measurements (e.g., liver stiffness):** Quantitative NIT for liver fibrosis
- **FibroTest scores:** Composite biomarker score
- **Aixplorer measurements:** Alternative elastography-based NIT

### Biochemical and Hematological Markers

- **ALT, AST:** Liver enzymes
- **GGT:** Gamma-glutamyl transferase
- **Bilirubin:** Liver function indicator
- **Platelets:** May indicate advanced fibrosis/cirrhosis

### Outcome Variables

- **hepatic_event:**  
  - Binary indicator of major liver-related event occurrence (composite endpoint: ascites, jaundice, hepatic encephalopathy, etc.)
  - Associated event time (relative to baseline/enrollment)

- **death_status:**  
  - All-cause mortality indicator (binary)
  - Time of death (if applicable) or censoring

---

## Data Characteristics

- **Missingness:** Some features may be missing at random or systematically, simulating real-world EHR data.
- **Temporal Resolution:** Not all patients have the full 22 visits; follow-up duration varies.
- **Censoring:** Applies to survival/event variables; not all patients have experienced events or death by last follow-up.

---

## Usage Notes

- **Time-Aware Modeling:** Participants should leverage longitudinal feature dynamics (e.g., trends, slopes, recent values).
- **Handling Missing Data:** Robust imputation and informative missingness indicators are encouraged.
- **Prediction Targets:**  
  - Patient-level risk scores for major hepatic events and all-cause mortality.

- **Official Metric:** C-index for each outcome, combined as:  
  `Final Score = 0.3 × C-index_death + 0.7 × C-index_hepatic`

- **Open Science:** Dataset and models to be released under the MIT license.

---

## Example Schema

| subject_id | visit_number | age | BMI | FibroScan | FibroTest | Aixplorer | ALT | AST | GGT | bilirubin | platelets | [other markers] | hepatic_event | time_to_event | death_status | time_to_death |
|------------|-------------|-----|-----|-----------|-----------|-----------|-----|-----|-----|-----------|-----------|----------------|---------------|---------------|--------------|---------------|
|   string   |     int     |float|float|   float   |   float   |   float   |float|float|float|   float   |   float   |    ...         |    0/1        |    float      |    0/1       |    float      |

---

## References

- For detailed variable definitions and baseline code, see the [official portal](https://app.trustii.io).
- More background on clinical biomarkers and NITs in MASLD: consult literature on FibroScan, FibroTest, elastography, and liver fibrosis progression.

---