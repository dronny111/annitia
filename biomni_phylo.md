Now I have everything I need for a thorough, data-grounded analysis. Let me write the full response:

---

## ANNITIA 2026 — Repo Understanding, Risk Audit & Adaptation Roadmap

---

### Repo Understanding Summary

**Data**: 1,253 train / 423 test patients. Wide-format with 13 longitudinal feature families across up to 22 visits (`_v1`–`_v22`). Only 6 baseline (non-visit) columns: `gender`, `T2DM`, `Hypertension`, `Dyslipidaemia`, `bariatric_surgery`, `bariatric_surgery_age`. Missingness is extreme — median column is 93% missing; only early visits (v1–v5) are usable at ≤50% missing.

**Targets**: Hepatic events: 47/1,253 (3.8%). Death: 76/984 known (7.7%); 269 patients have unknown death status (NaN). Both are right-censored survival outcomes measured in age-years.

**Baseline (hello_world notebook)**: Keeps 35 features (50% missing threshold), uses median imputation + StandardScaler, fits Elastic Net Cox (alpha via 5-fold CV) and RSF (min_samples_leaf=20). Submits RSF predictions. Train C-index: ~0.77 (Cox) / ~0.91 (RSF — heavily overfit). EPV at 35 features: 1.3 for hepatic, 2.2 for death — both dangerously low.

---

### Top 3 Integrity / Modeling Risks

**Risk 1 — Post-event visit leakage (CRITICAL)**
27/47 hepatic event patients have visit data recorded *after* their event age. If raw visit columns (e.g., `alt_v5`) are used without masking post-event rows, the model can learn from measurements taken after the outcome occurred. Death events have zero post-death visits (biologically correct), but hepatic events do not. Any feature engineering that uses `last()`, `max()`, or `slope()` across all visits for event patients will leak. **Fix**: Before computing longitudinal summaries, mask all visit slots where `Age_v{i} > event_age_occur` for event patients.

**Risk 2 — EPV collapse with feature engineering (HIGH)**
With only 47 hepatic events, the safe feature budget is ~5–10 features (EPV ≥ 5). The hello-world already uses 35 (EPV 1.3). Any feature engineering that naively adds slopes, deltas, and derived markers per family will explode to 100+ features, making the model memorize training noise. **Fix**: Use aggressive regularization (LASSO/Elastic Net), feature selection via permutation importance on held-out folds, or dimensionality reduction before fitting. Target ≤10 active features for the hepatic model.

**Risk 3 — Death unknown status handling (MEDIUM)**
269/1,253 patients have `death = NaN`. The hello-world drops them entirely, reducing the death model to 984 patients. These patients are likely censored (alive at last contact) but the label is ambiguous. Dropping them is conservative but loses 21% of the cohort. Treating them as censored (death=0, time=last_observed_age−Age_v1) would recover them but introduces label noise if some are truly deceased. **Fix**: Sensitivity analysis — fit with and without NaN-as-censored; report both CV C-indices. If they agree within ±0.01, use the larger cohort.

---

### Top 5 High-Value Next Steps (Ranked)

**1. Leakage-safe longitudinal feature engineering** *(highest expected upside)*

The hello-world uses only raw visit columns v1–v4 (those passing the 50% threshold). The real signal is in *trajectories* — how ALT, fibrosis stiffness, or BMI change over time. Implement in `annitia_longitudinal_features.py`:

```python
def safe_longitudinal_features(df, feature_families, event_age_col=None):
    """
    For each patient and feature family, compute:
      - first_obs, last_obs (pre-event only if event_age_col provided)
      - mean, std, min, max across observed visits
      - slope (linear regression on age-indexed values, ≥2 obs required)
      - delta (last - first)
      - n_obs (visit count — informative missingness signal)
      - recency_weighted_mean (exponential decay from last visit)
    
    CRITICAL: if event_age_col is not None, mask all visit slots
    where Age_v{i} > row[event_age_col] before computing summaries.
    """
```

This converts 13 families × ~8 summary stats = ~104 candidate features, then regularize down to ≤10 active. Expected gain: +0.03–0.08 C-index over raw visit columns, based on standard longitudinal survival literature.

**2. Clinically motivated derived markers (FIB-4, APRI, AST/ALT)**

These are validated MASLD fibrosis staging tools with direct biological relevance to hepatic decompensation:

- **FIB-4** = (Age × AST) / (PLT × √ALT) — validated cutoffs: <1.3 low risk, >2.67 high risk for advanced fibrosis
- **APRI** = (AST / 40) / PLT × 100 — simpler, correlated with portal hypertension
- **AST/ALT ratio** — >1 suggests cirrhosis (Pugh sign), available for 64% of patients at v1

Compute these at each visit, then apply the same longitudinal summaries (first, last, slope). FIB-4 is available for 26% of patients at v1 (all four components present), but the slope of FIB-4 over time is a stronger predictor than a single value. Add to `annitia_longitudinal_features.py` as a `compute_derived_markers()` function. **Leakage note**: FIB-4 uses Age, which is fine — Age at visit is not a target-derived quantity.

**3. Recency-weighted NIT trajectory (required by rules)**

The challenge rules require preserving at least one NIT trajectory. `fibs_stiffness_med_BM_1` (liver stiffness by FibroScan) is the most clinically powerful — it directly measures fibrosis stage and is the primary predictor of hepatic decompensation in MASLD. Available for 70% of patients at v1.

Implement a recency-weighted mean: `Σ(value_i × exp(-λ × (last_age - age_i))) / Σ(exp(-λ × (last_age - age_i)))` with λ tuned by CV (try 0.1, 0.3, 0.5 per year). This captures that recent stiffness measurements are more prognostically relevant than old ones. Add to `annitia_longitudinal_features.py`. Also compute the **slope of liver stiffness** — progressive stiffening is a stronger predictor than absolute value.

**4. Proper cross-validation with C-index scoring**

The hello-world reports only train C-index (RSF: 0.91, Cox: 0.77). The RSF train C-index is heavily overfit — the true generalization gap is unknown. Implement stratified 5-fold CV with patient-level splits in `annitia_pipeline.py`:

```python
from sklearn.model_selection import StratifiedKFold
# Stratify by event status to ensure events in each fold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Score: 0.7 * cv_ci_hepatic + 0.3 * cv_ci_death
```

This is the only reliable signal for model selection. Without it, you cannot distinguish real improvement from overfitting. **Critical**: fit imputer and scaler inside each fold, not outside.

**5. Gradient boosted survival model (XGBoost-AFT or LightGBM)**

With ~35 features and 47 events, tree-based survival models can outperform RSF when properly regularized. XGBoost's AFT (Accelerated Failure Time) loss handles censoring natively and is less prone to overfitting than RSF on small event counts. Add to `ensemble.py`:

```python
import xgboost as xgb
# xgb.train with objective='survival:aft', eval_metric='aft-nloglik'
# Key hyperparams: max_depth=3, min_child_weight=10, subsample=0.8
```

Alternatively, `lifelines.CoxPHFitter` with penalizer tuning is simpler and more interpretable. Ensemble Cox + RSF + XGBoost-AFT predictions (rank-average) for the final submission.

---

### Clinically Motivated Feature Ideas

| Feature | Biological rationale | Implementation note |
|---|---|---|
| **FIB-4 slope** | Progressive fibrosis → decompensation. Rising FIB-4 over visits predicts cirrhosis transition | Compute FIB-4 at each visit, fit linear slope vs. age |
| **Liver stiffness trajectory** (fibs_stiffness_med_BM_1) | FibroScan kPa directly measures fibrosis stage; >12 kPa = cirrhosis threshold | Last value + slope; threshold crossing (ever >12 kPa) |
| **GGT/ALT ratio** | Elevated GGT with normal ALT suggests alcoholic or metabolic steatohepatitis pattern | Compute at each visit, use last and mean |
| **BMI trajectory** | Weight loss in MASLD can reverse fibrosis; weight gain accelerates it | Slope of BMI; delta BMI from v1 to last visit |
| **Bariatric surgery interaction** | Post-bariatric patients have dramatically different fibrosis trajectories | Interaction: bariatric_surgery × BMI_slope |
| **T2DM × fibrosis stiffness** | Diabetes is the strongest modifier of MASLD-to-cirrhosis progression | Interaction term: T2DM × fibs_stiffness_last |
| **Visit frequency** | Irregular follow-up may indicate clinical deterioration or loss to follow-up | n_visits, inter-visit interval mean/std |
| **Fibrotest trajectory** | Serum fibrosis marker; slope more informative than single value | Same as stiffness: last + slope |

---

### Adaptation Table

| Source idea | Why it matters | ANNITIA fit | Adaptation plan | Validation plan | Risks / exclusions |
|---|---|---|---|---|---|
| **Temporal feature engineering** (first/last/mean/slope/delta) | Raw visit columns discard trajectory information; slopes capture disease progression | **Directly transferable** | Add `safe_longitudinal_features()` to `annitia_longitudinal_features.py`; mask post-event visits before computing summaries | 5-fold CV C-index before/after; check EPV stays ≥5 | Must mask post-event visits for hepatic events (27/47 affected) |
| **Missing-rate filter** (hello-world: 50% threshold) | Prevents constant-imputed features from dominating | **Directly transferable, improve it** | Replace hard threshold with missingness as a feature (indicator columns) + MICE or iterative imputer for early visits | Compare CV C-index with/without missingness indicators | Fit imputer on train fold only; never on full train+test |
| **Elastic Net Cox regularization** | Enforces EPV constraint via sparsity; interpretable coefficients | **Directly transferable** | Already in hello-world; extend alpha grid, try l1_ratio ∈ {0.5, 0.7, 0.9, 1.0} | CV C-index across alpha/l1_ratio grid | No leakage risk; standard survival model |
| **RSF with min_samples_leaf tuning** | Reduces variance on rare events | **Transferable with modification** | Increase min_samples_leaf to 30–50 for hepatic (47 events); tune via CV | OOB C-index + 5-fold CV | Train C-index is misleading (0.91 is overfit); use OOB only |
| **Derived clinical markers** (FIB-4, APRI, AST/ALT) | Validated MASLD staging tools; encode domain knowledge | **Directly transferable** | Add `compute_derived_markers()` to `annitia_longitudinal_features.py`; compute at each visit then summarize | Feature importance in Cox model; clinical plausibility check | FIB-4 requires PLT (26% availability); handle NaN gracefully |
| **Recency-weighted mean** | Recent measurements more prognostically relevant than old ones | **Directly transferable** | Implement exponential decay weighting in `annitia_longitudinal_features.py`; tune λ by CV | Compare vs. simple mean in CV | Requires age-indexed visits; use Age_v{i} as time axis |
| **Ensemble (rank-average)** | Reduces variance across model families | **Directly transferable** | Add to `ensemble.py`: rank-normalize Cox, RSF, XGBoost-AFT predictions; weighted average (tune weights by CV) | Weighted C-index on held-out fold | Ensure all models trained on same fold splits |
| **Stratified K-fold CV** | Only reliable model selection signal | **Critical, missing from baseline** | Add to `annitia_pipeline.py`; stratify by event indicator; score = 0.7×CI_hep + 0.3×CI_death | This IS the validation plan | Patient-level splits only; no row-level leakage |
| **Missingness as informative feature** | In clinical data, missing labs often indicate clinical state (e.g., no PLT ordered = no cirrhosis suspicion) | **Transferable with caution** | Add binary indicator columns `{feature}_observed` for NIT and PLT columns | Check if indicators improve CV C-index | Do not use missingness patterns that correlate with outcome timing |
| **XGBoost-AFT / LightGBM survival** | Handles non-linear interactions; competitive with RSF on tabular data | **Transferable with modification** | Add to `ensemble.py`; use `xgb` AFT loss or `lifelines` CoxPH with penalizer | CV C-index; compare to RSF baseline | Requires careful max_depth/min_child_weight tuning to avoid EPV collapse |
| **Interaction features** (T2DM × stiffness, bariatric × BMI slope) | MASLD progression is strongly modified by metabolic comorbidities | **Transferable, medium priority** | Add to `feature_revolution.py`; compute pairwise products of top clinical predictors | Permutation importance in Cox model | Only add if EPV budget allows (≤2–3 interactions for hepatic model) |
| **Threshold-crossing features** (ever stiffness >12 kPa) | Cirrhosis threshold crossing is a clinical milestone | **Directly transferable** | Add to `annitia_longitudinal_features.py`; binary flag per NIT threshold | Feature importance; clinical validation | Requires knowing thresholds; use published MASLD cutoffs |
| **Visit frequency / inter-visit interval** | Irregular follow-up is informative about clinical trajectory | **Directly transferable** | Compute n_visits, mean/std inter-visit interval from Age_v{i} columns | Check correlation with event status | Do not use total follow-up duration (correlated with censoring time) |

**Excluded ideas** (not applicable to ANNITIA):
- Imaging-based features (no MRI/CT/histology in dataset)
- Row-level time-series models (LSTM, Transformer) — patient-level survival, not sequence classification; also EPV too low
- Post-event lab values as features — directly excluded by leakage rules
- Test-set label inference or pseudo-labeling — explicitly prohibited
- Biopsy-derived fibrosis stage (NAS score, Metavir) — not in dataset

---

### Immediate Action Priority

1. **Fix leakage first** — implement post-event visit masking before any feature engineering. This is a correctness issue, not an optimization.
2. **Add 5-fold CV** — you currently have no reliable model selection signal.
3. **Implement safe longitudinal summaries** — first/last/slope/delta for all 13 families with leakage masking.
4. **Add FIB-4, APRI, AST/ALT** as derived features at each visit, then summarize.
5. **Ensemble Cox + RSF** with rank-averaging as the submission strategy.

The weighted metric (0.7 × hepatic + 0.3 × death) means hepatic C-index improvements are worth 2.3× more than equivalent death improvements. Prioritize features and models that improve hepatic prediction — liver stiffness trajectory and FIB-4 slope are the highest-value candidates clinically.