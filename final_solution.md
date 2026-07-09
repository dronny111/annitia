# 🏆 ANNITIA Data Challenge 2026 — Final Notebook Submission Instructions

Carefully follow this prompt to construct your final, **valid, and competitive** notebook for submission to the ANNITIA Data Challenge 2026. Strict adherence to challenge rules, best practices, and formatting requirements is required. Optimizing for both quantitative and qualitative evaluation will maximize your chance of a top placement.

---

## Notebook Sections & Guidance

### 1. **Title and Metadata**
- Title: Clearly indicate your team and your model (“ANNITIA 2026: [Team Name] Final Solution”).
- Authors, date, and brief summary of core approach.

---

### 2. **Setup and Environment**
- List all software and library dependencies (e.g., Python version, pandas, numpy, scikit-learn, scikit-survival, lifelines, matplotlib/seaborn if used).
- Ensure all code is suitable for MIT-licensed release and fully reproducible.

---

### 3. **Data Loading and Exploration**
- Import the provided ANNITIA synthetic dataset.
- Perform minimal exploratory data analysis (EDA) to show awareness of missingness, longitudinal timeline structure, and distributions.
- No peeking at test labels or usage of outcome information not allowed by the rules.

---

### 4. **Preprocessing and Feature Engineering**
- Handle longitudinal and temporal structure robustly:
  - Impute missing data (with strategy explained and missingness indicators if used).
  - Engineer time-aware features (e.g., trends, slopes, recency, windows, summary stats per patient).
- [**Required:**] Use at least one NIT trajectory as input.
- Ensure **no use of post-event or post-censoring information** for any prediction.

---

### 5. **Model Construction**
- Clearly outline your modeling pipeline:
  - Justify model selection (e.g., survival ensemble, time-series NN, or classical Cox model with engineered features).
  - Document hyperparameters and why chosen (grid search, literature, etc.).
  - Describe how outcome/censoring information is handled.
- Include comments for code clarity and reproducibility.

---

### 6. **Validation Strategy**
- Perform rigorous **patient-level** cross-validation (e.g., k-fold or holdout), explicitly preventing data leakage.
- Report **cross-validated C-index** scores for both hepatic event and death prediction on validation sets.
- Demonstrate model robustness and handling of missingness and longitudinal variability.

---

### 7. **Final Training & Prediction**
- Retrain final model on all training data, using only allowable (pre-event) features.
- Generate patient-level risk scores for both major hepatic events and all-cause mortality for the test set.
- Output must be formatted precisely as required for submission (see rules).

---

### 8. **Explainability and Clinical Insight**
- Assess and discuss feature importance and model interpretability (e.g., SHAP values, coefficients, important time-based features).
- Include plots or tables highlighting clinically relevant predictors and trends.
- Reflect on clinical plausibility and actionable insight from your model.

---

### 9. **Documentation & License**
- Provide clear, step-by-step instructions for running the notebook and reproducing your results.
- Explicitly include the MIT license at the start or end of the notebook.
- Describe any additional requirements or configuration for reproducibility.

---

### 10. **Appendix (Optional)**
- Show ablation studies, alternative model attempts, error analysis, or research justification for choices.
- Add any hyperparameter tuning details or additional EDA not shown in main sections.

---

## Checklist Before Submission

- [ ] **No test labels, no data leakage, and full patient-level splits throughout**
- [ ] **All key sections above are present and clearly documented**
- [ ] **Risk scores are output in the required format**
- [ ] **Code is readable, modular, and MIT-licensed**
- [ ] **Reproducibility validated: everything runs from scratch**
- [ ] **Notebook includes model interpretability/clinical insight**

---

## Optimization Tips

- Cross-check for robustness on missing/irregular longitudinal data.
- Favor validated modeling approaches (see baseline or literature), but feel free to experiment if justified.
- Focus on both C-index optimization **and** clinical explainability for best total score (per 70% quantitative, 30% qualitative rules).

---

**By following this structure and guidance, you will maximize both your compliance with ANNITIA Data Challenge 2026 rules and your opportunity for a winning submission.**