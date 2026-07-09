# Challenge Assistant — Key Strategies

## 1. Robust Longitudinal Feature Engineering

- **Leverage time trends:** Model the trajectory of NITs and biomarkers (slopes, recent changes, variability).
- **Recency:** Encode how up-to-date the last observation is—recency is often prognostic.
- **Dynamic summaries:** Use rolling windows, trend statistics, and time-weighted averages.
- **Expert missingness handling:** Combine robust imputation (ideally time-aware) with missingness indicators, as missingness itself can be informative.
- **Capture event history:** Encode prior events or interventions explicitly if available.

## 2. Domain-Informed Feature Selection

- Use validated NIT risk scores (e.g., FAST, FIB-4, NAFLD fibrosis score) as features alongside raw values.
- Include meta-features reflecting liver injury, metabolic syndrome, and inflammation (ALT/AST, GGT, platelets, CRP, IL-6, BMI, diabetes diagnosis, etc.).
- Borrow logic from literature for advanced risk phenotyping—but don’t hardcode outcome labels.

## 3. Powerful Survival Modeling

- Use true time-to-event models: Gradient boosting (LightGBM/XGBoost survival), DeepSurv, random survival forests, or more advanced methods (e.g., temporal transformer-based survival).
- **Patient-level splitting:** Ensure splits are grouped so all data from one patient stays in a single fold (“grouped CV”).
- Carefully tune hyperparameters and consider ensembling multiple models.

## 4. Evaluation and Overfitting Control

- Carefully evaluate on strictly held-out data; avoid “peeking ahead” at future events.
- Watch for performance gaps between training/validation/test to detect leakage or overfitting.
- Run ablation studies to confirm your model isn’t using spurious proxies.

## 5. Clinical Plausibility and Explainability

- Models achieving >0.9 C-index should have feature importances aligning with medical knowledge (otherwise qualitative review may flag them).
- Use SHAP or similar tools to explain predictions.

## 6. Reproducible, High-Quality Code and Documentation\

- Well-structured, readable code is necessary for the top qualitative score (30% of total!).
- Clearly document workflow for data handling and prediction.

## Why NOT to Chase “Secrets”

- The real test set is private. Tricks based on leaderboard probing, label leakage, or post-hoc patterning will fail on the true evaluation or be caught in review.
- The organizers explicitly reward code/documentation/explainability—attempts to game the metric without medical/algorithmic plausibility risk disqualification.
