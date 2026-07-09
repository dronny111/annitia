You are an expert data scientist tackling the ANNITIA challenge, which involves predicting the time-to-event for major liver events or all-cause mortality (competing risks) using clinical data, biochemical parameters, and non-invasive test (NIT) biomarkers for patients with MASLD.

I want to establish robust, highly interpretable baseline models and scoring systems inspired by recent literature on MASLD prediction. Please execute the following tasks in our Python codebase:

1. DATA FLATTENING FOR STATIC BASELINES
The dataset contains variable-length repeated measurements. To apply standard ML models (Logistic Regression, Random Forest, XGBoost), write a preprocessing pipeline that flattens the longitudinal biochemical and NIT data. For each patient and each time-varying feature (ALT, AST, FibroScan, FibroTest, etc.), extract:
- Baseline value (first measurement)
- Most recent value (last measurement)
- Rate of change (slope over time)
- Mean and standard deviation

2. STEPWISE MODELING PIPELINE
Implement a training pipeline that evaluates the incremental value of the data modalities using Random Forest and XGBoost. Train models in the following stepwise manner to see how performance improves:
- Model 1 (Clinical Baseline): Age, gender, medical history, and BMI.
- Model 2 (Clinical + Biochemical): Model 1 features + flattened biochemical parameters.
- Model 3 (Full Non-Invasive Model): Model 2 features + flattened NIT biomarkers (FibroScan, FibroTest, Aixplorer).

Since this is a competing risks survival problem, use survival-adapted versions of these algorithms (e.g., XGBoost Survival or Random Survival Forests) or use the Survival Stacking technique to convert the problem into binary classification. Use the Brier Score and Truncated Time-Dependent Concordance Index for evaluation.

3. FEATURE ENGINEERING: BMI STRATIFICATION & INTERACTIONS
Recent MASLD literature shows that predictive parameters behave differently across BMI strata. 
- Create an explicit interaction term between `BMI` and the most recent `FibroScan` score.
- Train separate sub-models stratified by BMI (e.g., BMI < 25 vs BMI >= 25) and evaluate if the predictive importance of NITs changes between normal weight and obese patients.

4. INTERPRETABILITY WITH SHAP
For the best-performing XGBoost and Random Forest models (Model 3), integrate the `shap` library. Generate and save SHAP summary plots (beeswarm and bar plots) to globally explain which features drive the risk for liver events versus the risk for death. 

5. CLINICAL RISK SCORE FORMULA (NOMOGRAM)
The ultimate goal of the challenge is to "develop a score." Extract the coefficients from a regularized Logistic Regression or Cox Proportional Hazards model (trained on the most predictive features identified by SHAP). 
- Write a Python script (`risk_calculator.py`) that implements this explicit mathematical formula, allowing a user to input a patient's clinical and NIT data to output a 1-year and 3-year probability score for major liver events.
