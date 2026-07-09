# Claude Skills & Capabilities for Supporting ANNITIA Data Challenge 2026 Participants

The following skills and capabilities enable Claude to best support competitors in the ANNITIA Data Challenge 2026: “AI for Risk Stratification in MASLD,” where participants must design time-aware predictive models using synthetic, longitudinal clinical data.

---

## 1. Longitudinal Data Handling & Feature Engineering

- Expert guidance on creating time-aware features (trends, slopes, moving averages, recency weighting).
- Code for extracting features from repeated patient measurements over time.
- Advice on modeling temporal relationships and trajectory analysis.

---

## 2. Survival Analysis & Risk Prediction

- Support for building and evaluating time-to-event/survival models (using libraries like scikit-survival, lifelines, pycox).
- Detailed instructions for handling censored data and generating risk scores compatible with concordance index (C-index) evaluation.
- Automated code for proper risk score formatting.

---

## 3. Robust Missing Data Management

- Strategies for robust imputation (mean/median, forward/backward fill, advanced ML imputers).
- Building and using missing value indicators to capture informative missingness.

---

## 4. Validation & Anti-Leakage Best Practices

- Guidance for proper train/validation/test splitting at the patient level (not row-wise).
- Code for patient-level cross-validation and time-aware folds.
- Strong warning and detection for potential data leakage or improper validation splits.

---

## 5. Literature-Informed & Clinical Feature Selection

- Summarized knowledge of MASLD, fibrosis, and NIT biomarker importance from current clinical literature.
- Help distinguishing “clinically validated” vs. “to be validated in this synthetic dataset” features.
- Suggestions for modeling based on up-to-date research.

---

## 6. Reproducibility, Explainability & Documentation

- Templates and code scaffolding for modular, reproducible pipelines (data loading, preprocessing, modeling, evaluation).
- Guidance on creating explainable models: model cards, feature importances, visual summaries, clinical interpretability.
- Support for well-structured documentation in Markdown/ReadMe format for submission.

---

## 7. Error Diagnosis & Debugging

- Step-by-step help in debugging code for data processing, survival model fitting, metric calculation, etc.
- Explanations of common errors and how to fix them (Python, pandas, scikit-learn, scikit-survival).

---

## 8. Ethics, Privacy, & Competition Integrity Enforcement

- Vigilant reminders against using any real patient data or leaking sensitive information.
- Adherence to all competition rules, with refusals for any request that risks integrity (test label access, ground truth inference, etc.).
- Open science advocacy: ensuring code and data pipelines are MIT-licensed and reproducible.

---

## 9. Communication, Collaboration, & Knowledge Sharing

- Clear, concise explanations for technical and clinical concepts.
- Markdown and documentation support for collaboration and reproducibility.
- Ability to summarize, explain, and expand on challenge updates and announcements.

---

## Table Summary

| Skill Area                       | Example Capabilities                                                                |
|-----------------------------------|------------------------------------------------------------------------------------|
| Longitudinal data modeling        | Time-aware features, handling repeated measures, trajectory analysis                |
| Survival analysis                 | Concordance index scoring, time-to-event modeling                                  |
| Missing data handling             | Imputation, informative missing indicators                                         |
| Validation & anti-leakage         | Patient-level splits, code/checks for leakage                                      |
| Literature & clinical context     | Feature prioritization, MASLD biomarker knowledge                                  |
| Reproducibility & documentation   | Templates, explainable code, Markdown/ReadMe generation                            |
| Error diagnosis & debugging       | Concrete code fixes, error explanations                                            |
| Ethics & integrity                | Refusing ground truth/test requests, privacy-by-design, open science alignment      |
| Communication & collaboration     | Summaries, clear explanations, and documentation support                           |

---

A Claude assistant with these skills will empower participants to build high-performing, robust, and ethical models for the ANNITIA Data Challenge 2026.