# ANNITIA Data Challenge 2026 — Rules Summary

## Overview

- **Goal:** Develop AI models to predict (1) major hepatic events and (2) all-cause mortality in synthetic patients with MASLD, using longitudinal trajectories of non-invasive tests (NITs), biomarkers, BMI, and cardiometabolic data.
- **Dataset:** Synthetic, based on real-world hospital data, de-identified for privacy, with up to 22 visits per patient (years 2001–2025).
- **Modeling Scope:** Make risk predictions from longitudinal data, leveraging repeated NITs and clinical features.

---

## Data & Features

- **Synthetic Dataset:** No real patient data; all records are anonymous and privacy-protected.
- **Repeated Measures:** Up to 22 time points per patient, with repeated NITs (FibroScan, FibroTest, Aixplorer), ALT, AST, GGT, bilirubin, platelets, BMI, other cardiometabolic features.
- **Outcomes to Predict:**
  - Major hepatic (liver-related) events — composite endpoint
  - All-cause mortality

---

## Submission & Evaluation

- **Predictions:** Submit patient-level risk scores for both hepatic events and mortality, suitable for time-to-event (survival) analysis.
- **Official Metric:**  
  `Final Score = 0.3 × C-index_death + 0.7 × C-index_hepatic`  
  (using scikit-survival’s `concordance_index_censored`)
- **Qualitative Score:** A panel will judge model interpretability, clinical insight, and justifiability.
- **Total Final Score:**  
  `Total = 70% Quantitative (metric) + 30% Qualitative (interpretability/clinical value)`

---

## Timeline

- **Challenge Launch:** March 16th, 2026
- **Submission Deadline:** May 8th, 2026
- **Winners Announced:** May 2026

---

## Prizes

- **Total Prize Pool:** €14,000  
  - 1st Place: €8,000  
  - 2nd Place: €4,000  
  - 3rd Place: €2,000

- **Additional Recognition:** Top teams may be invited for collaboration and further development.

---

## Integrity & Fair Play

- **No Cheating:** Do **not** attempt to access hidden test labels, reverse-engineer the scoring, exploit leakage, or submit without genuine modeling.
- **Privacy & Ethics:** Never use real, non-synthetic patient data.
- **Reproducibility:** Prepare submissions with open-source, MIT-licensed code; ensure all results are replicable.

---

## Modeling Guidance

- Leverage longitudinal structure: use time-aware features (slopes, trends, recency).
- Handle missing data robustly (imputation, informative missingness).
- Strictly avoid data leakage. Use only information available up to prediction/censoring/event time.
- Perform validation using patient-level splits, **not** row-wise or timepoint-based splits.
- Only use features allowed by the challenge dataset.

---

## License & Open Science

- **All submissions and models must be released under the MIT open-source license.**
- The synthetic dataset and developed models will be made publicly available to support further research and education.

---

## Further Information

- Full challenge details, data access, baseline code, and updates are provided via [ANNITIA Challenge Portal](https://app.trustii.io).
- Direct questions to the organizers through official channels. Do not seek or share unfair advantages.

---