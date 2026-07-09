# ANNITIA Data Challenge 2026 — Dataset Summary

## Overview

The ANNITIA Data Challenge 2026 provides a fully synthetic, longitudinal dataset designed to enable AI-based risk stratification in metabolic dysfunction-associated steatotic liver disease (MASLD). The dataset closely models real-world clinical trajectories while ensuring complete patient anonymity.

---

## Dataset Structure

- **Cohort Origin:** Synthetic patients based on anonymized data distributions from Pitié-Salpêtrière Hospital, Paris
- **Longitudinal Follow-up:** Up to **22 visits** (2001–2025) per patient
- **Data Privacy:** All data is synthetic and fully de-identified

---

## Core Features

### Non-Invasive Tests (NITs) (repeated measures)
- **FibroScan:** Quantitative measurement of liver fibrosis (liver stiffness)
- **FibroTest:** Composite biomarker score for fibrosis
- **Aixplorer:** Alternative elastography-based NIT

### Biochemical & Clinical Parameters
- **ALT, AST, GGT:** Liver enzymes (alanine/aspartate transaminase, gamma-GT)
- **Bilirubin:** Marker of liver function
- **Platelets:** Low count can indicate advanced fibrosis or cirrhosis

### Anthropometric & Cardiometabolic Features
- **BMI:** Body mass index
- **Cardiometabolic markers:** May include glucose, blood pressure, etc.

### Outcomes (per patient)
- **hepatic_event:** Major liver-related events: ascites, jaundice, hepatic encephalopathy, etc.
- **death_status:** All-cause mortality indicator
- **Event times:** Time (in months/years) to hepatic event and/or death; censoring applies

---

## Data Properties

- **Synthetic missingness:** Missing values and irregular visit schedules to mimic real-world EHR data
- **Variable follow-up:** Not all patients have full 22 visits; follow-up length varies due to events/censoring
- **Longitudinal design:** Same features measured at multiple time points per patient

---

## Example Table Schema

| trustii_id | visit_no | age | BMI | FibroScan | FibroTest | Aixplorer | ALT | AST | GGT | bilirubin | platelets | ... | hepatic_event | event_time | death_status | death_time |
|------------|----------|-----|-----|-----------|-----------|-----------|-----|-----|-----|-----------|-----------|-----|---------------|------------|--------------|------------|
|    int     |   int    |float|float|   float   |   float   |   float   |float|float|float|   float   |   float   | ... |    0 or 1     |   float    |   0 or 1     |   float    |

---

## Usage and Modeling Notes

- **Survival prediction:** Predict individual risk for hepatic events and all-cause mortality using longitudinal feature histories
- **Time-aware features:** Consider trends, slopes, recency-weighted means, and time-to-event/censor information
- **Handle missingness:** Use robust imputation strategies, add missingness indicators if informative
- **Validation:** Only use patient-level splits to avoid data leakage

---

## Licensing & Access

- **Open source:** All data and required code must be released under the MIT license
- **Access & Documentation:** Full challenge details, variable definitions, and starter code available at [challenge portal](https://app.trustii.io) (hypothetical link)

---

## Additional Information

- **Metric:**  
  `Final Score = 0.3 × C-index_death + 0.7 × C-index_hepatic` (see [scikit-survival docs](https://scikit-survival.readthedocs.io/en/stable/generated/sksurv.metrics.concordance_index_censored.html))
- **Clinical context:** MASLD affects ~30% of the world’s population and can progress from steatosis to advanced cirrhosis and hepatic events

---

*For questions or updates, see the official challenge portal.*