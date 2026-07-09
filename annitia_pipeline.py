"""
ANNITIA Data Challenge 2026 — Full Survival Analysis Pipeline
=============================================================
AI-based risk stratification for MASLD (metabolic dysfunction-associated
steatotic liver disease). Predicts two survival endpoints:

  1. Major hepatic events (ascites, jaundice, hepatic encephalopathy)
  2. All-cause mortality

Score = 0.7 × C-index_hepatic + 0.3 × C-index_death

Pipeline stages:
  1. Data loading and post-event masking
  2. Feature engineering (longitudinal summaries + clinical thresholds + interactions)
  3. Survival modeling (XGBoost AFT + CatBoost SurvivalAft ensemble)
  4. Patient-level stratified k-fold cross-validation
  5. SHAP-based model explainability
  6. Submission CSV generation

Usage:
    python annitia_pipeline.py                # Full pipeline
    python annitia_pipeline.py --cv-only      # Cross-validation only
    python annitia_pipeline.py --submit-only  # Skip CV, generate submission
    python annitia_pipeline.py --no-shap      # Skip SHAP analysis

MIT License — ANNITIA 2026
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sksurv.metrics import concordance_index_censored

import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

RANDOM_STATE = 42
MISSINGNESS_THRESHOLD = 0.50
N_SPLITS = 5

OUTCOME_COLS = [
    "evenements_hepatiques_majeurs",
    "evenements_hepatiques_age_occur",
    "death",
    "death_age_occur",
]

# Longitudinal feature column prefixes (maps friendly name → column prefix)
LONGITUDINAL_FEATURES = {
    "Age": "Age_v",
    "BMI": "BMI_v",
    "ALT": "alt_v",
    "AST": "ast_v",
    "Bilirubin": "bilirubin_v",
    "Cholesterol": "chol_v",
    "GGT": "ggt_v",
    "Glucose": "gluc_fast_v",
    "Platelets": "plt_v",
    "Triglycerides": "triglyc_v",
    "FibroScan": "fibs_stiffness_med_BM_1_v",
    "FibroTest": "fibrotest_BM_2_v",
    "Aixplorer": "aixp_aix_result_BM_3_v",
}

# Feature ablation: remove volatility and derived scores (best public config)
VOLATILITY_SUFFIXES = ("_std", "_range")
DERIVED_PREFIXES = ("FIB4_", "APRI_", "AST_ALT_ratio_")


# ══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ══════════════════════════════════════════════════════════════════════════════

def find_data_files() -> tuple[Path, Path | None]:
    """Detect training and test CSVs automatically."""
    data_files = sorted(Path("data").glob("DB-*.csv"))
    train_path = test_path = None
    for f in data_files:
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        if "evenements_hepatiques_majeurs" in cols:
            train_path = f
        else:
            test_path = f
    if train_path is None:
        raise FileNotFoundError("Could not find training CSV with outcome columns in data/.")
    return train_path, test_path


def list_visit_numbers(columns) -> list[int]:
    """Extract sorted visit numbers from Age_v# columns."""
    pattern = re.compile(r"_v(\d+)$")
    visits = {
        int(m.group(1))
        for col in columns
        if col.startswith("Age_v") and (m := pattern.search(col))
    }
    return sorted(visits)


def mask_post_event_wide(
    df: pd.DataFrame,
    visit_numbers: list[int],
    event_age_col: str | None = None,
) -> pd.DataFrame:
    """Zero out visit data that occurs after the event/censoring age."""
    df = df.copy()
    if event_age_col and event_age_col in df.columns:
        event_age = df[event_age_col]
        for v in visit_numbers:
            age_col = f"Age_v{v}"
            if age_col not in df.columns:
                continue
            visit_cols = [c for c in df.columns if c.endswith(f"_v{v}") and c != age_col]
            if not visit_cols:
                continue
            mask = event_age.notna() & df[age_col].notna() & (df[age_col] > event_age)
            if mask.any():
                df.loc[mask, visit_cols] = np.nan
    return df


# ══════════════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════════

def safe_polyfit_slope(values: pd.Series, time_index: pd.Series) -> float:
    """Fit a linear trend and return the slope; 0.0 if insufficient data."""
    valid_mask = values.notna() & time_index.notna()
    x = time_index.loc[valid_mask].astype(float).to_numpy()
    y = values.loc[valid_mask].astype(float).to_numpy()
    if len(y) < 2 or np.unique(x).size < 2:
        return 0.0
    try:
        return float(np.polyfit(x, y, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def recency_weighted_mean(values: pd.Series, decay: float = 0.3) -> float:
    """Exponentially-weighted mean emphasizing recent observations."""
    observed = values.dropna().astype(float)
    if observed.empty:
        return np.nan
    weights = np.exp(decay * np.arange(len(observed), dtype=float))
    return float(np.average(observed.to_numpy(), weights=weights))


def summarize_series(values: pd.Series, time_index: pd.Series, prefix: str) -> dict:
    """Compute per-biomarker patient-level summary statistics."""
    observed = values.dropna().astype(float)
    if observed.empty:
        summary = {
            f"{prefix}_first": np.nan, f"{prefix}_last": np.nan,
            f"{prefix}_mean": np.nan, f"{prefix}_min": np.nan,
            f"{prefix}_max": np.nan, f"{prefix}_slope": np.nan,
            f"{prefix}_rwmean": np.nan, f"{prefix}_delta": np.nan,
        }
    else:
        first = float(observed.iloc[0])
        last = float(observed.iloc[-1])
        summary = {
            f"{prefix}_first": first,
            f"{prefix}_last": last,
            f"{prefix}_mean": float(observed.mean()),
            f"{prefix}_min": float(observed.min()),
            f"{prefix}_max": float(observed.max()),
            f"{prefix}_slope": safe_polyfit_slope(values, time_index),
            f"{prefix}_rwmean": recency_weighted_mean(observed),
            f"{prefix}_delta": last - first,
        }
    summary[f"{prefix}_missing_rate"] = float(values.isna().mean()) if len(values) else np.nan
    return summary


def build_long_format(
    df_wide: pd.DataFrame,
    event_age_col: str | None = None,
) -> pd.DataFrame:
    """Convert wide-format to long-format with one row per patient-visit."""
    visit_numbers = list_visit_numbers(df_wide.columns)
    frames = []

    for visit in visit_numbers:
        frame = pd.DataFrame(index=df_wide.index)
        frame["source_row"] = df_wide.index
        frame["visit_number"] = visit

        for feature_name, prefix in LONGITUDINAL_FEATURES.items():
            col = f"{prefix}{visit}"
            frame[feature_name] = df_wide[col].values if col in df_wide.columns else np.nan

        if event_age_col and event_age_col in df_wide.columns:
            frame["_event_age"] = df_wide[event_age_col].values

        frames.append(frame)

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.loc[long_df["Age"].notna()].copy()

    if event_age_col and event_age_col in df_wide.columns:
        keep = long_df["_event_age"].isna() | (long_df["Age"] <= long_df["_event_age"])
        long_df = long_df.loc[keep].copy()
        long_df.drop(columns="_event_age", inplace=True)

    long_df.sort_values(["source_row", "visit_number"], inplace=True)
    long_df.reset_index(drop=True, inplace=True)
    return long_df


def engineer_longitudinal_summaries(
    df_wide: pd.DataFrame,
    event_age_col: str | None = None,
) -> pd.DataFrame:
    """Build per-patient longitudinal summary features from wide-format data."""
    long_df = build_long_format(df_wide, event_age_col=event_age_col)
    feature_groups = list(LONGITUDINAL_FEATURES.keys())

    patient_records = []
    for source_row, visits in long_df.groupby("source_row", sort=True):
        visits = visits.sort_values("visit_number")
        record: dict[str, Any] = {"source_row": int(source_row), "n_visits": len(visits)}

        age_index = visits["Age"].astype(float)
        for feat_name in feature_groups:
            record.update(summarize_series(visits[feat_name], age_index, feat_name))

        patient_records.append(record)

    df_feat = pd.DataFrame(patient_records)
    df_feat.index = df_feat["source_row"].astype(df_wide.index.dtype)
    df_feat.drop(columns="source_row", inplace=True)
    df_feat.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df_feat


def add_clinical_threshold_features(
    df_wide: pd.DataFrame,
    event_age_col: str | None = None,
) -> pd.DataFrame:
    """Build clinically-motivated binary/count features from longitudinal data.

    These features encode hepatology domain knowledge about risk thresholds
    for FibroScan, FibroTest, liver enzymes, platelets, BAMI, and glucose.
    """
    long_df = build_long_format(df_wide, event_age_col=event_age_col)
    records = []

    for source_row, visits in long_df.groupby("source_row", sort=True):
        visits = visits.sort_values("visit_number")
        feat: dict[str, float] = {"source_row": int(source_row)}

        # --- FibroScan thresholds (liver stiffness kPa) ---
        fs = visits["FibroScan"].dropna()
        feat["ever_fibroscan_ge8"] = float((fs >= 8).any()) if len(fs) else np.nan
        feat["ever_fibroscan_ge12"] = float((fs >= 12).any()) if len(fs) else np.nan
        feat["max_fibroscan"] = float(fs.max()) if len(fs) else np.nan
        feat["last_fibroscan"] = float(fs.iloc[-1]) if len(fs) else np.nan

        # --- FibroTest thresholds (F2 ≥ 0.48, F3-F4 ≥ 0.74) ---
        ft = visits["FibroTest"].dropna()
        feat["ever_fibrotest_ge048"] = float((ft >= 0.48).any()) if len(ft) else np.nan
        feat["ever_fibrotest_ge074"] = float((ft >= 0.74).any()) if len(ft) else np.nan
        feat["max_fibrotest"] = float(ft.max()) if len(ft) else np.nan

        # --- ALT thresholds (ULN ~35 U/L) ---
        alt = visits["ALT"].dropna()
        feat["alt_above_uln_count"] = float((alt > 35).sum()) if len(alt) else np.nan
        feat["alt_above_uln_ratio"] = float((alt > 35).mean()) if len(alt) else np.nan
        feat["ever_alt_gt_2x_uln"] = float((alt > 70).any()) if len(alt) else np.nan

        # --- AST thresholds ---
        ast = visits["AST"].dropna()
        feat["ever_ast_gt_uln"] = float((ast > 35).any()) if len(ast) else np.nan

        # --- GGT thresholds (ULN ~50 U/L) ---
        ggt = visits["GGT"].dropna()
        feat["ever_ggt_elevated"] = float((ggt > 50).any()) if len(ggt) else np.nan
        feat["max_ggt"] = float(ggt.max()) if len(ggt) else np.nan

        # --- Bilirubin thresholds (ULN ~20 μmol/L) ---
        bili = visits["Bilirubin"].dropna()
        feat["ever_bili_elevated"] = float((bili > 20).any()) if len(bili) else np.nan

        # --- BMI thresholds ---
        bmi = visits["BMI"].dropna()
        feat["last_bmi"] = float(bmi.iloc[-1]) if len(bmi) else np.nan
        feat["last_bmi_ge30"] = float(bmi.iloc[-1] >= 30) if len(bmi) else np.nan
        feat["max_bmi"] = float(bmi.max()) if len(bmi) else np.nan
        feat["bmi_change"] = float(bmi.iloc[-1] - bmi.iloc[0]) if len(bmi) >= 2 else np.nan

        # --- Platelet thresholds (thrombocytopenia: <150, severe: <100) ---
        plt_vals = visits["Platelets"].dropna()
        feat["ever_plt_lt150"] = float((plt_vals < 150).any()) if len(plt_vals) else np.nan
        feat["ever_plt_lt100"] = float((plt_vals < 100).any()) if len(plt_vals) else np.nan
        feat["last_platelets"] = float(plt_vals.iloc[-1]) if len(plt_vals) else np.nan

        # --- NIT trajectory direction (increasing = worsening) ---
        if len(fs) >= 2:
            feat["fibroscan_direction"] = 1.0 if fs.iloc[-1] > fs.iloc[0] else (
                -1.0 if fs.iloc[-1] < fs.iloc[0] else 0.0)
        else:
            feat["fibroscan_direction"] = np.nan

        if len(ft) >= 2:
            feat["fibrotest_direction"] = 1.0 if ft.iloc[-1] > ft.iloc[0] else (
                -1.0 if ft.iloc[-1] < ft.iloc[0] else 0.0)
        else:
            feat["fibrotest_direction"] = np.nan

        records.append(feat)

    df_clinical = pd.DataFrame(records)
    df_clinical.index = df_clinical["source_row"].astype(df_wide.index.dtype)
    df_clinical.drop(columns="source_row", inplace=True)
    return df_clinical


def add_visit_pattern_features(
    df_wide: pd.DataFrame,
    event_age_col: str | None = None,
) -> pd.DataFrame:
    """Features about follow-up duration, visit density, and NIT coverage."""
    visit_numbers = list_visit_numbers(df_wide.columns)
    age_cols = [f"Age_v{v}" for v in visit_numbers]

    records = []
    for idx in range(len(df_wide)):
        row = df_wide.iloc[idx]
        feat: dict[str, float] = {}

        event_age = None
        if event_age_col and event_age_col in df_wide.columns:
            ea = row.get(event_age_col)
            if pd.notna(ea):
                event_age = ea

        ages = []
        for ac in age_cols:
            a = row.get(ac)
            if pd.notna(a):
                if event_age is not None and a > event_age:
                    continue
                ages.append(float(a))

        n_visits = len(ages)
        feat["follow_up_years"] = ages[-1] - ages[0] if n_visits >= 2 else 0.0
        feat["visit_density"] = (
            n_visits / feat["follow_up_years"]
            if feat["follow_up_years"] > 0 else float(n_visits)
        )
        feat["age_at_first_visit"] = ages[0] if n_visits else np.nan
        feat["age_at_last_visit"] = ages[-1] if n_visits else np.nan

        # NIT assessment coverage
        fs_cols = [c for c in df_wide.columns if c.startswith("fibs_stiffness_med_BM_1_v")]
        ft_cols = [c for c in df_wide.columns if c.startswith("fibrotest_BM_2_v")]
        n_fs = sum(1 for c in fs_cols if pd.notna(row.get(c)))
        n_ft = sum(1 for c in ft_cols if pd.notna(row.get(c)))
        feat["had_fibroscan"] = float(n_fs > 0)
        feat["had_fibrotest"] = float(n_ft > 0)
        feat["n_nit_visits"] = float(n_fs + n_ft)
        feat["nit_coverage_ratio"] = (n_fs + n_ft) / (2 * max(n_visits, 1))

        records.append(feat)

    return pd.DataFrame(records, index=df_wide.index)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Clinically-motivated interaction features for MASLD risk."""
    out = df.copy()

    if "T2DM" in df.columns and "last_bmi_ge30" in df.columns:
        out["T2DM_x_obese"] = df["T2DM"].fillna(0) * df["last_bmi_ge30"].fillna(0)

    if "Hypertension" in df.columns and "T2DM" in df.columns:
        out["MetSyn_proxy"] = df["Hypertension"].fillna(0) * df["T2DM"].fillna(0)

    comorbidity_cols = ["T2DM", "Hypertension", "Dyslipidaemia"]
    if all(c in df.columns for c in comorbidity_cols):
        out["n_comorbidities"] = df[comorbidity_cols].fillna(0).sum(axis=1)

    if "age_at_last_visit" in df.columns and "last_fibroscan" in df.columns:
        out["age_x_fibroscan"] = df["age_at_last_visit"] * df["last_fibroscan"]

    if "ever_alt_gt_2x_uln" in df.columns and "ever_ast_gt_uln" in df.columns:
        out["both_transaminases_elevated"] = (
            df["ever_alt_gt_2x_uln"].fillna(0) * df["ever_ast_gt_uln"].fillna(0)
        )

    if "ever_plt_lt150" in df.columns and "ever_fibroscan_ge8" in df.columns:
        out["plt_low_x_fs_high"] = (
            df["ever_plt_lt150"].fillna(0) * df["ever_fibroscan_ge8"].fillna(0)
        )

    return out


def build_full_feature_matrix(
    df_wide: pd.DataFrame,
    event_age_col: str | None = None,
) -> pd.DataFrame:
    """Orchestrate all feature engineering steps into a single matrix.

    Steps:
      1. Static demographics/comorbidities
      2. Longitudinal summary statistics (mean, slope, delta, etc.)
      3. Clinical threshold features (FibroScan ≥8, platelets <150, etc.)
      4. Visit pattern features (follow-up, density, NIT coverage)
      5. Interaction features (T2DM×obesity, platelet×FibroScan, etc.)
      6. Filter: remove volatility suffixes and derived scores (best config)
    """
    # Static features
    static_exclude = {"patient_id_anon", "trustii_id", *OUTCOME_COLS}
    static_cols = [
        c for c in df_wide.columns
        if c not in static_exclude and not re.search(r"_v\d+$", c)
    ]
    df_static = df_wide[static_cols].copy()

    # Longitudinal summaries
    df_longitudinal = engineer_longitudinal_summaries(df_wide, event_age_col)

    # Clinical thresholds
    df_clinical = add_clinical_threshold_features(df_wide, event_age_col)

    # Visit patterns
    df_visit = add_visit_pattern_features(df_wide, event_age_col)

    # Combine all
    df_features = pd.concat([df_static, df_longitudinal, df_clinical, df_visit], axis=1)
    df_features = df_features.loc[:, ~df_features.columns.duplicated()]

    # Interaction features (need combined columns)
    df_features = add_interaction_features(df_features)

    # Apply winning filter: remove volatility and derived scores
    keep_cols = [
        c for c in df_features.columns
        if not any(c.endswith(s) for s in VOLATILITY_SUFFIXES)
        and not any(c.startswith(p) for p in DERIVED_PREFIXES)
    ]
    df_features = df_features[keep_cols]

    df_features.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df_features


# ══════════════════════════════════════════════════════════════════════════════
# 3. PREPROCESSING
# ══════════════════════════════════════════════════════════════════════════════

def preprocess_features(
    X_train_raw: pd.DataFrame,
    X_val_raw: pd.DataFrame | None = None,
    X_test_raw: pd.DataFrame | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame | None, pd.DataFrame | None, list[str]]:
    """Impute missing values (median) and filter high-missingness columns.

    Imputer is fit ONLY on training data to prevent leakage.
    """
    missing_rate = X_train_raw.isna().mean()
    keep_cols = missing_rate[missing_rate <= MISSINGNESS_THRESHOLD].index.tolist()
    if not keep_cols:
        raise ValueError("No features passed the missingness filter.")

    imputer = SimpleImputer(strategy="median")
    X_train = pd.DataFrame(
        imputer.fit_transform(X_train_raw[keep_cols]),
        columns=keep_cols,
        index=X_train_raw.index,
    )

    def _apply(df):
        if df is None:
            return None
        return pd.DataFrame(
            imputer.transform(df.reindex(columns=keep_cols, fill_value=np.nan)),
            columns=keep_cols,
            index=df.index,
        )

    return X_train, _apply(X_val_raw), _apply(X_test_raw), keep_cols


# ══════════════════════════════════════════════════════════════════════════════
# 4. SURVIVAL MODELS
# ══════════════════════════════════════════════════════════════════════════════

class XGBoostAFTModel:
    """XGBoost Accelerated Failure Time (survival:aft).

    Best-performing single model in the ANNITIA challenge.
    Directly models the survival time distribution with interval-censored labels.
    """

    def __init__(
        self,
        distribution: str = "normal",
        num_boost_round: int = 250,
        max_depth: int = 4,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: float = 5.0,
    ):
        self.distribution = distribution
        self.num_boost_round = num_boost_round
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.booster = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "XGBoostAFTModel":
        self.feature_names_ = list(X.columns)
        dtrain = xgb.DMatrix(X, feature_names=self.feature_names_)

        # Interval-censored labels: [time, time] for events, [time, ∞) for censored
        lower = y["time"].astype(float)
        upper = np.where(y["event"], y["time"], np.inf).astype(float)
        dtrain.set_float_info("label_lower_bound", lower)
        dtrain.set_float_info("label_upper_bound", upper)

        params = {
            "objective": "survival:aft",
            "eval_metric": "aft-nloglik",
            "aft_loss_distribution": self.distribution,
            "aft_loss_distribution_scale": 1.0,
            "tree_method": "hist",
            "learning_rate": self.learning_rate,
            "max_depth": self.max_depth,
            "subsample": self.subsample,
            "colsample_bytree": self.colsample_bytree,
            "min_child_weight": self.min_child_weight,
            "seed": RANDOM_STATE,
            "nthread": -1,
            "verbosity": 0,
        }
        self.booster = xgb.train(params, dtrain, num_boost_round=self.num_boost_round)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        dm = xgb.DMatrix(X, feature_names=self.feature_names_)
        # AFT predicts survival time; negate so higher = higher risk
        return -self.booster.predict(dm)

    def get_feature_importance(self) -> dict[str, float]:
        return self.booster.get_score(importance_type="gain") if self.booster else {}


class CatBoostSurvivalAFTModel:
    """CatBoost with SurvivalAft loss for interval-censored survival data.

    Strong complementary model to XGBoost AFT, enabling robust ensembles.
    """

    def __init__(
        self,
        iterations: int = 250,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 3.0,
    ):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.model: CatBoostRegressor | None = None
        self.feature_names_: list[str] | None = None

    def fit(self, X: pd.DataFrame, y: np.ndarray) -> "CatBoostSurvivalAFTModel":
        self.feature_names_ = list(X.columns)

        # SurvivalAft expects 2D target: [lower_bound, upper_bound]
        lower = y["time"].astype(float)
        upper = np.where(y["event"], y["time"], 1e6).astype(float)
        labels = np.column_stack([lower, upper])

        self.model = CatBoostRegressor(
            loss_function="SurvivalAft",
            iterations=self.iterations,
            depth=self.depth,
            learning_rate=self.learning_rate,
            l2_leaf_reg=self.l2_leaf_reg,
            random_seed=RANDOM_STATE,
            verbose=False,
            allow_writing_files=False,
        )
        self.model.fit(X, labels)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        preds = self.model.predict(X)
        # SurvivalAft predicts survival time; negate so higher = higher risk
        return -preds

    def get_feature_importance(self) -> np.ndarray:
        return self.model.get_feature_importance() if self.model else np.array([])


# ══════════════════════════════════════════════════════════════════════════════
# 5. CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def combined_score(c_hep: float, c_death: float) -> float:
    return 0.7 * c_hep + 0.3 * c_death


def cv_single_endpoint(
    model_class,
    model_kwargs: dict,
    X_raw: pd.DataFrame,
    y: np.ndarray,
    n_splits: int = N_SPLITS,
) -> np.ndarray:
    """Run patient-level stratified k-fold CV for one endpoint."""
    strat = y["event"].astype(int)
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scores = []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_raw, strat), 1):
        X_tr, X_te, _, _ = preprocess_features(X_raw.iloc[tr_idx], X_raw.iloc[te_idx])
        model = model_class(**model_kwargs)
        model.fit(X_tr, y[tr_idx])
        preds = model.predict(X_te)
        c = concordance_index_censored(y[te_idx]["event"], y[te_idx]["time"], preds)[0]
        scores.append(c)

    return np.array(scores)


def cross_validate_full(
    X_hep: pd.DataFrame,
    y_hep: np.ndarray,
    X_death: pd.DataFrame,
    y_death: np.ndarray,
) -> dict[str, Any]:
    """Run cross-validation for both models on both endpoints."""
    models_to_eval = {
        "XGBoost AFT": (XGBoostAFTModel, {}),
        "CatBoost SurvivalAft": (CatBoostSurvivalAFTModel, {}),
    }

    results = {}
    for name, (model_class, kwargs) in models_to_eval.items():
        print(f"\n  {name}")
        hep_scores = cv_single_endpoint(model_class, kwargs, X_hep, y_hep)
        death_scores = cv_single_endpoint(model_class, kwargs, X_death, y_death)
        comb = combined_score(hep_scores.mean(), death_scores.mean())

        print(f"    Hepatic C-index: {hep_scores.mean():.4f} ± {hep_scores.std():.4f}  "
              f"[{', '.join(f'{s:.4f}' for s in hep_scores)}]")
        print(f"    Death C-index:   {death_scores.mean():.4f} ± {death_scores.std():.4f}  "
              f"[{', '.join(f'{s:.4f}' for s in death_scores)}]")
        print(f"    Combined Score:  {comb:.4f}")

        results[name] = {
            "hep_scores": hep_scores,
            "death_scores": death_scores,
            "combined": comb,
        }

    # Ensemble CV (70% XGB + 30% CatBoost rank-avg)
    print("\n  Ensemble (70% XGBoost + 30% CatBoost, rank-averaged)")
    strat_hep = y_hep["event"].astype(int)
    strat_death = y_death["event"].astype(int)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)

    ens_hep_scores, ens_death_scores = [], []

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_hep, strat_hep), 1):
        X_tr, X_te, _, _ = preprocess_features(X_hep.iloc[tr_idx], X_hep.iloc[te_idx])
        xgb_model = XGBoostAFTModel()
        xgb_model.fit(X_tr, y_hep[tr_idx])
        cat_model = CatBoostSurvivalAFTModel()
        cat_model.fit(X_tr, y_hep[tr_idx])
        preds_xgb = xgb_model.predict(X_te)
        preds_cat = cat_model.predict(X_te)
        # Rank-average ensemble
        from scipy.stats import rankdata
        rank_xgb = rankdata(preds_xgb) / len(preds_xgb)
        rank_cat = rankdata(preds_cat) / len(preds_cat)
        preds_ens = 0.7 * rank_xgb + 0.3 * rank_cat
        c = concordance_index_censored(y_hep[te_idx]["event"], y_hep[te_idx]["time"], preds_ens)[0]
        ens_hep_scores.append(c)

    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_death, strat_death), 1):
        X_tr, X_te, _, _ = preprocess_features(X_death.iloc[tr_idx], X_death.iloc[te_idx])
        xgb_model = XGBoostAFTModel()
        xgb_model.fit(X_tr, y_death[tr_idx])
        cat_model = CatBoostSurvivalAFTModel()
        cat_model.fit(X_tr, y_death[tr_idx])
        preds_xgb = xgb_model.predict(X_te)
        preds_cat = cat_model.predict(X_te)
        from scipy.stats import rankdata
        rank_xgb = rankdata(preds_xgb) / len(preds_xgb)
        rank_cat = rankdata(preds_cat) / len(preds_cat)
        preds_ens = 0.7 * rank_xgb + 0.3 * rank_cat
        c = concordance_index_censored(y_death[te_idx]["event"], y_death[te_idx]["time"], preds_ens)[0]
        ens_death_scores.append(c)

    ens_hep = np.array(ens_hep_scores)
    ens_death = np.array(ens_death_scores)
    ens_comb = combined_score(ens_hep.mean(), ens_death.mean())

    print(f"    Hepatic C-index: {ens_hep.mean():.4f} ± {ens_hep.std():.4f}")
    print(f"    Death C-index:   {ens_death.mean():.4f} ± {ens_death.std():.4f}")
    print(f"    Combined Score:  {ens_comb:.4f}")

    results["Ensemble 70/30"] = {
        "hep_scores": ens_hep,
        "death_scores": ens_death,
        "combined": ens_comb,
    }

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6. SUBMISSION GENERATION
# ══════════════════════════════════════════════════════════════════════════════

def generate_submission(
    X_hep: pd.DataFrame,
    y_hep: np.ndarray,
    X_death: pd.DataFrame,
    y_death: np.ndarray,
    X_test: pd.DataFrame,
    trustii_ids: pd.Series,
    filename: str = "submission_pipeline.csv",
    ensemble: bool = True,
) -> pd.DataFrame:
    """Train on all data, predict on test set, generate submission CSV.

    If ensemble=True (default), uses rank-averaged 70% XGB + 30% CatBoost.
    """
    from scipy.stats import rankdata

    # Preprocess hepatic endpoint
    X_hep_tr, _, X_hep_te, _ = preprocess_features(X_hep, X_test_raw=X_test)
    # Preprocess death endpoint
    X_death_tr, _, X_death_te, _ = preprocess_features(X_death, X_test_raw=X_test)

    if ensemble:
        # XGBoost AFT — hepatic
        xgb_hep = XGBoostAFTModel()
        xgb_hep.fit(X_hep_tr, y_hep)
        risk_hep_xgb = xgb_hep.predict(X_hep_te)

        # CatBoost — hepatic
        cat_hep = CatBoostSurvivalAFTModel()
        cat_hep.fit(X_hep_tr, y_hep)
        risk_hep_cat = cat_hep.predict(X_hep_te)

        # Rank-average ensemble
        rank_hep_xgb = rankdata(risk_hep_xgb) / len(risk_hep_xgb)
        rank_hep_cat = rankdata(risk_hep_cat) / len(risk_hep_cat)
        risk_hepatic = 0.7 * rank_hep_xgb + 0.3 * rank_hep_cat

        # XGBoost AFT — death
        xgb_death = XGBoostAFTModel()
        xgb_death.fit(X_death_tr, y_death)
        risk_death_xgb = xgb_death.predict(X_death_te)

        # CatBoost — death
        cat_death = CatBoostSurvivalAFTModel()
        cat_death.fit(X_death_tr, y_death)
        risk_death_cat = cat_death.predict(X_death_te)

        rank_death_xgb = rankdata(risk_death_xgb) / len(risk_death_xgb)
        rank_death_cat = rankdata(risk_death_cat) / len(risk_death_cat)
        risk_death = 0.7 * rank_death_xgb + 0.3 * rank_death_cat
    else:
        # Single XGBoost AFT model
        xgb_hep = XGBoostAFTModel()
        xgb_hep.fit(X_hep_tr, y_hep)
        risk_hepatic = xgb_hep.predict(X_hep_te)

        xgb_death = XGBoostAFTModel()
        xgb_death.fit(X_death_tr, y_death)
        risk_death = xgb_death.predict(X_death_te)

    sub = pd.DataFrame({
        "trustii_id": trustii_ids.values,
        "risk_hepatic_event": risk_hepatic,
        "risk_death": risk_death,
    }).sort_values("trustii_id").reset_index(drop=True)

    sub.to_csv(filename, index=False)
    print(f"\n  Submission saved: {filename}")
    print(f"  Shape: {sub.shape}, trustii_id range: [{sub['trustii_id'].min()}, {sub['trustii_id'].max()}]")
    return sub


# ══════════════════════════════════════════════════════════════════════════════
# 7. SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════════════════════════

def run_shap_analysis(
    X_hep: pd.DataFrame,
    y_hep: np.ndarray,
    X_death: pd.DataFrame,
    y_death: np.ndarray,
) -> None:
    """Generate SHAP summary plots for both endpoints.

    Trains final models on full data and explains feature importance.
    """
    try:
        import shap
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  SHAP or matplotlib not available. Skipping explainability.")
        return

    for endpoint_name, X_raw, y in [
        ("Hepatic Events", X_hep, y_hep),
        ("All-Cause Mortality", X_death, y_death),
    ]:
        print(f"\n  SHAP Analysis — {endpoint_name}")

        # Preprocess
        X_tr, _, _, keep_cols = preprocess_features(X_raw)

        # Train XGBoost AFT model
        model = XGBoostAFTModel()
        model.fit(X_tr, y)

        # SHAP explainer
        explainer = shap.TreeExplainer(model.booster)
        shap_values = explainer.shap_values(xgb.DMatrix(X_tr, feature_names=keep_cols))

        # Summary plot
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            shap_values,
            X_tr,
            feature_names=keep_cols,
            max_display=20,
            show=False,
        )
        slug = endpoint_name.lower().replace(" ", "_").replace("-", "_")
        fname = f"shap_summary_{slug}.png"
        plt.tight_layout()
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"    Saved: {fname}")

        # Top 15 features by mean absolute SHAP value
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[::-1][:15]
        print(f"    Top 15 features by |SHAP|:")
        for rank, i in enumerate(top_idx, 1):
            print(f"      {rank:2d}. {keep_cols[i]:40s}  {mean_abs_shap[i]:.4f}")

    # Bar chart comparison
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    for ax_idx, (endpoint_name, X_raw, y) in enumerate([
        ("Hepatic Events", X_hep, y_hep),
        ("All-Cause Mortality", X_death, y_death),
    ]):
        X_tr, _, _, keep_cols = preprocess_features(X_raw)
        model = XGBoostAFTModel()
        model.fit(X_tr, y)
        importance = model.booster.get_score(importance_type="gain")
        top20 = sorted(importance.items(), key=lambda x: x[1], reverse=True)[:20]
        names, values = zip(*top20) if top20 else ([], [])
        axes[ax_idx].barh(range(len(names)), values, color="steelblue")
        axes[ax_idx].set_yticks(range(len(names)))
        axes[ax_idx].set_yticklabels(names, fontsize=8)
        axes[ax_idx].invert_yaxis()
        axes[ax_idx].set_xlabel("Gain")
        axes[ax_idx].set_title(f"Feature Importance — {endpoint_name}")

    plt.tight_layout()
    plt.savefig("feature_importance_comparison.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Saved: feature_importance_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN PIPELINE
# ══════════════════════════════════════════════════════════════════════════════

def load_and_prepare_data() -> dict[str, Any]:
    """Load raw data, build targets, and engineer features."""
    train_path, test_path = find_data_files()
    print(f"  Train: {train_path}")
    print(f"  Test:  {test_path}")

    df_raw = pd.read_csv(train_path)
    df_test_raw = pd.read_csv(test_path) if test_path else None

    visit_numbers = list_visit_numbers(df_raw.columns)
    print(f"  Patients: {len(df_raw)} train, {len(df_test_raw) if df_test_raw is not None else 0} test")
    print(f"  Visits: v1–v{max(visit_numbers)}")

    # ── Post-event masking ────────────────────────────────────────────────
    df_hep_masked = mask_post_event_wide(df_raw, visit_numbers, "evenements_hepatiques_age_occur")
    df_death_masked = mask_post_event_wide(df_raw, visit_numbers, "death_age_occur")
    df_test_masked = mask_post_event_wide(df_test_raw, visit_numbers) if df_test_raw is not None else None

    # ── Build survival targets ────────────────────────────────────────────
    # Hepatic events
    age_cols = [f"Age_v{v}" for v in visit_numbers]

    def get_age_range(df):
        age_frame = df[age_cols]
        first = age_frame.bfill(axis=1).iloc[:, 0]
        last = age_frame.ffill(axis=1).iloc[:, -1]
        return first, last

    hep_first, hep_last = get_age_range(df_hep_masked)
    hep_unknown = df_raw["evenements_hepatiques_majeurs"].isna()
    hep_invalid = df_raw["evenements_hepatiques_majeurs"].eq(1) & df_raw["evenements_hepatiques_age_occur"].isna()
    hep_valid = ~hep_unknown & ~hep_invalid & hep_first.notna() & hep_last.notna()

    df_hep = df_hep_masked.loc[hep_valid].copy()
    df_hep["_age_first"] = hep_first.loc[hep_valid]
    df_hep["_age_last"] = hep_last.loc[hep_valid]
    hepatic_event = df_raw.loc[hep_valid, "evenements_hepatiques_majeurs"].astype(bool).values
    hepatic_time = np.where(
        hepatic_event,
        df_raw.loc[hep_valid, "evenements_hepatiques_age_occur"],
        df_hep["_age_last"],
    ) - df_hep["_age_first"].values
    hepatic_time = np.clip(hepatic_time, 0.01, None)

    y_hepatic = np.array(
        list(zip(hepatic_event, hepatic_time)),
        dtype=[("event", bool), ("time", float)],
    )

    # Death events
    death_first, death_last = get_age_range(df_death_masked)
    death_unknown = df_raw["death"].isna()
    death_invalid = df_raw["death"].eq(1) & df_raw["death_age_occur"].isna()
    death_valid = ~death_unknown & ~death_invalid & death_first.notna() & death_last.notna()

    df_death = df_death_masked.loc[death_valid].copy()
    df_death["_age_first"] = death_first.loc[death_valid]
    df_death["_age_last"] = death_last.loc[death_valid]
    death_event = df_raw.loc[death_valid, "death"].astype(bool).values
    death_time = np.where(
        death_event,
        df_raw.loc[death_valid, "death_age_occur"],
        df_death["_age_last"],
    ) - df_death["_age_first"].values
    death_time = np.clip(death_time, 0.01, None)

    y_death = np.array(
        list(zip(death_event, death_time)),
        dtype=[("event", bool), ("time", float)],
    )

    print(f"  Hepatic events: {hepatic_event.sum()}/{len(hepatic_event)} "
          f"({hepatic_event.mean()*100:.1f}%)")
    print(f"  Death events:   {death_event.sum()}/{len(death_event)} "
          f"({death_event.mean()*100:.1f}%)")

    # ── Feature engineering ───────────────────────────────────────────────
    print("\n  Engineering features...")
    X_hep = build_full_feature_matrix(df_hep, event_age_col="evenements_hepatiques_age_occur")
    X_death = build_full_feature_matrix(df_death, event_age_col="death_age_occur")
    X_test = build_full_feature_matrix(df_test_masked) if df_test_masked is not None else None

    # Align test columns to union of train columns
    if X_test is not None:
        union_cols = X_hep.columns.union(X_death.columns)
        X_test = X_test.reindex(columns=union_cols, fill_value=np.nan)

    print(f"  Feature matrix: {X_hep.shape[1]} features (hepatic), {X_death.shape[1]} (death)")

    # trustii_id mapping
    trustii_ids = df_test_raw["trustii_id"] if df_test_raw is not None else None

    return {
        "X_hep": X_hep, "X_death": X_death, "X_test": X_test,
        "y_hepatic": y_hepatic, "y_death": y_death,
        "trustii_ids": trustii_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="ANNITIA Pipeline — MASLD Risk Stratification")
    parser.add_argument("--cv-only", action="store_true", help="Run cross-validation only")
    parser.add_argument("--submit-only", action="store_true", help="Generate submission without CV")
    parser.add_argument("--no-shap", action="store_true", help="Skip SHAP analysis")
    parser.add_argument("--single-model", action="store_true",
                        help="Use single XGBoost AFT instead of ensemble")
    parser.add_argument("--output", type=str, default="submission_pipeline.csv",
                        help="Submission filename")
    args = parser.parse_args()

    print("=" * 70)
    print("ANNITIA DATA CHALLENGE 2026 — MASLD Risk Stratification Pipeline")
    print("=" * 70)

    # ── Stage 1: Load Data ────────────────────────────────────────────────
    print("\n[1/5] Loading and preparing data...")
    data = load_and_prepare_data()

    X_hep, X_death = data["X_hep"], data["X_death"]
    y_hep, y_death = data["y_hepatic"], data["y_death"]
    X_test, trustii_ids = data["X_test"], data["trustii_ids"]

    # ── Stage 2: Cross-Validation ─────────────────────────────────────────
    if not args.submit_only:
        print("\n[2/5] Cross-validating models...")
        cv_results = cross_validate_full(X_hep, y_hep, X_death, y_death)

        # Summary table
        print("\n" + "=" * 70)
        print("CROSS-VALIDATION SUMMARY")
        print("=" * 70)
        for name, res in sorted(cv_results.items(), key=lambda x: -x[1]["combined"]):
            print(f"  {name:30s}  Combined={res['combined']:.4f}  "
                  f"Hepatic={res['hep_scores'].mean():.4f}  "
                  f"Death={res['death_scores'].mean():.4f}")
    else:
        print("\n[2/5] Skipping cross-validation (--submit-only)")

    # ── Stage 3: SHAP Explainability ──────────────────────────────────────
    if not args.no_shap and not args.submit_only:
        print("\n[3/5] Running SHAP explainability analysis...")
        run_shap_analysis(X_hep, y_hep, X_death, y_death)
    else:
        print("\n[3/5] Skipping SHAP analysis")

    # ── Stage 4: Generate Submission ──────────────────────────────────────
    if not args.cv_only and X_test is not None and trustii_ids is not None:
        print(f"\n[4/5] Generating submission → {args.output}")
        sub = generate_submission(
            X_hep, y_hep, X_death, y_death, X_test, trustii_ids,
            filename=args.output,
            ensemble=not args.single_model,
        )
    else:
        print("\n[4/5] Skipping submission generation")

    # ── Stage 5: Summary ──────────────────────────────────────────────────
    print("\n[5/5] Pipeline complete!")
    print("=" * 70)
    if not args.submit_only and 'cv_results' in dir():
        best = max(cv_results.items(), key=lambda x: x[1]["combined"])
        print(f"  Best CV model: {best[0]} (Combined={best[1]['combined']:.4f})")
    if not args.cv_only and X_test is not None:
        print(f"  Submission: {args.output}")
    print("=" * 70)


if __name__ == "__main__":
    main()
