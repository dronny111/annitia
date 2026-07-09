"""Phase 1: Feature revolution for the ANNITIA challenge.

Addresses the core overfitting problem: 378 features with 47 hepatic events (EPV=0.12).

Key changes:
  1. Drop raw visit columns — use only static + engineered summaries
  2. Add clinical threshold features (hepatology-grounded)
  3. Add visit-pattern features
  4. Add interaction features
  5. Ablation across feature set variants

Usage:
    python3 feature_revolution.py                   # Run all ablations + submit best
    python3 feature_revolution.py --variant X        # Run specific variant only
    python3 feature_revolution.py --submit           # Generate submission CSVs
"""

from __future__ import annotations

import argparse
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sksurv.metrics import concordance_index_censored

import xgboost as xgb
from catboost import CatBoostRegressor

from annitia_longitudinal_features import (
    engineer_patient_features, LONGITUDINAL_FEATURES, list_visit_numbers,
    build_long_format,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
MISSINGNESS_THRESHOLD = 0.50
N_SPLITS = 5

OUTCOME_COLS = [
    "evenements_hepatiques_majeurs",
    "evenements_hepatiques_age_occur",
    "death",
    "death_age_occur",
]

VOLATILITY_SUFFIXES = ("_std", "_range")
DERIVED_PREFIXES = ("FIB4_", "APRI_", "AST_ALT_ratio_")


# ---------------------------------------------------------------------------
# Clinical threshold features
# ---------------------------------------------------------------------------

def add_clinical_threshold_features(df_wide: pd.DataFrame,
                                     event_age_col: str | None = None) -> pd.DataFrame:
    """Add clinically-motivated binary/count features from raw visits."""
    long_df = build_long_format(df_wide, event_age_col=event_age_col)
    visit_numbers = list_visit_numbers(df_wide.columns)

    records = []
    for source_row, visits in long_df.groupby("source_row", sort=True):
        visits = visits.sort_values("visit_number")
        feat: dict[str, float] = {"source_row": int(source_row)}

        # --- FibroScan thresholds ---
        fs = visits["FibroScan"].dropna()
        feat["ever_fibroscan_ge8"] = float((fs >= 8).any()) if len(fs) > 0 else np.nan
        feat["ever_fibroscan_ge12"] = float((fs >= 12).any()) if len(fs) > 0 else np.nan
        feat["max_fibroscan"] = float(fs.max()) if len(fs) > 0 else np.nan
        feat["last_fibroscan"] = float(fs.iloc[-1]) if len(fs) > 0 else np.nan
        feat["last_fibroscan_ge8"] = float(fs.iloc[-1] >= 8) if len(fs) > 0 else np.nan

        # --- FibroTest thresholds (F2 ≥ 0.48, F3-F4 ≥ 0.74) ---
        ft = visits["FibroTest"].dropna()
        feat["ever_fibrotest_ge048"] = float((ft >= 0.48).any()) if len(ft) > 0 else np.nan
        feat["ever_fibrotest_ge074"] = float((ft >= 0.74).any()) if len(ft) > 0 else np.nan
        feat["max_fibrotest"] = float(ft.max()) if len(ft) > 0 else np.nan

        # --- ALT > ULN (upper limit of normal ~35 U/L) ---
        alt = visits["ALT"].dropna()
        feat["alt_above_uln_count"] = float((alt > 35).sum()) if len(alt) > 0 else np.nan
        feat["alt_above_uln_ratio"] = float((alt > 35).mean()) if len(alt) > 0 else np.nan
        feat["ever_alt_gt_2x_uln"] = float((alt > 70).any()) if len(alt) > 0 else np.nan

        # --- AST > ULN (~35 U/L) ---
        ast = visits["AST"].dropna()
        feat["ever_ast_gt_uln"] = float((ast > 35).any()) if len(ast) > 0 else np.nan

        # --- GGT > ULN (~50 U/L for males, ~35 U/L for females; use 50 as conservative) ---
        ggt = visits["GGT"].dropna()
        feat["ever_ggt_elevated"] = float((ggt > 50).any()) if len(ggt) > 0 else np.nan
        feat["max_ggt"] = float(ggt.max()) if len(ggt) > 0 else np.nan

        # --- Bilirubin > ULN (~1.2 mg/dL = ~20 μmol/L; use 20 for μmol/L data) ---
        bili = visits["Bilirubin"].dropna()
        feat["ever_bili_elevated"] = float((bili > 20).any()) if len(bili) > 0 else np.nan

        # --- BMI thresholds ---
        bmi = visits["BMI"].dropna()
        feat["last_bmi"] = float(bmi.iloc[-1]) if len(bmi) > 0 else np.nan
        feat["last_bmi_ge30"] = float(bmi.iloc[-1] >= 30) if len(bmi) > 0 else np.nan
        feat["last_bmi_ge35"] = float(bmi.iloc[-1] >= 35) if len(bmi) > 0 else np.nan
        feat["max_bmi"] = float(bmi.max()) if len(bmi) > 0 else np.nan
        feat["bmi_change"] = float(bmi.iloc[-1] - bmi.iloc[0]) if len(bmi) >= 2 else np.nan

        # --- Platelet thresholds (thrombocytopenia < 150, severe < 100) ---
        plt_vals = visits["Platelets"].dropna()
        feat["ever_plt_lt150"] = float((plt_vals < 150).any()) if len(plt_vals) > 0 else np.nan
        feat["ever_plt_lt100"] = float((plt_vals < 100).any()) if len(plt_vals) > 0 else np.nan
        feat["last_platelets"] = float(plt_vals.iloc[-1]) if len(plt_vals) > 0 else np.nan

        # --- Glucose thresholds (diabetes: fasting > 126 mg/dL = ~7 mmol/L) ---
        gluc = visits["Glucose"].dropna()
        feat["ever_gluc_gt7"] = float((gluc > 7).any()) if len(gluc) > 0 else np.nan

        # --- NIT trajectory direction ---
        if len(fs) >= 2:
            feat["fibroscan_direction"] = 1.0 if fs.iloc[-1] > fs.iloc[0] else (-1.0 if fs.iloc[-1] < fs.iloc[0] else 0.0)
        else:
            feat["fibroscan_direction"] = np.nan

        if len(ft) >= 2:
            feat["fibrotest_direction"] = 1.0 if ft.iloc[-1] > ft.iloc[0] else (-1.0 if ft.iloc[-1] < ft.iloc[0] else 0.0)
        else:
            feat["fibrotest_direction"] = np.nan

        records.append(feat)

    return pd.DataFrame(records)


def add_visit_pattern_features(df_wide: pd.DataFrame,
                                event_age_col: str | None = None) -> pd.DataFrame:
    """Add features about visit patterns and follow-up."""
    visit_numbers = list_visit_numbers(df_wide.columns)
    age_cols = [f"Age_v{v}" for v in visit_numbers]

    records = []
    for idx in range(len(df_wide)):
        row = df_wide.iloc[idx]
        feat: dict[str, float] = {}

        # Age series (masked if event_age_col provided)
        ages = []
        event_age = None
        if event_age_col and event_age_col in df_wide.columns:
            event_age = row.get(event_age_col)
            if pd.isna(event_age):
                event_age = None

        for ac in age_cols:
            a = row.get(ac)
            if pd.notna(a):
                if event_age is not None and a > event_age:
                    continue
                ages.append(float(a))

        n_visits = len(ages)
        feat["follow_up_years"] = ages[-1] - ages[0] if n_visits >= 2 else 0.0
        feat["visit_density"] = (n_visits / feat["follow_up_years"]) if feat["follow_up_years"] > 0 else float(n_visits)
        feat["age_at_first_visit"] = ages[0] if n_visits > 0 else np.nan
        feat["age_at_last_visit"] = ages[-1] if n_visits > 0 else np.nan

        # NIT assessment coverage
        fs_cols = [c for c in df_wide.columns if c.startswith("fibs_stiffness_med_BM_1_v")]
        ft_cols = [c for c in df_wide.columns if c.startswith("fibrotest_BM_2_v")]
        aix_cols = [c for c in df_wide.columns if c.startswith("aixp_aix_result_BM_3_v")]

        n_fs = sum(1 for c in fs_cols if pd.notna(row.get(c)))
        n_ft = sum(1 for c in ft_cols if pd.notna(row.get(c)))
        n_aix = sum(1 for c in aix_cols if pd.notna(row.get(c)))

        feat["had_fibroscan"] = float(n_fs > 0)
        feat["had_fibrotest"] = float(n_ft > 0)
        feat["had_aixplorer"] = float(n_aix > 0)
        feat["n_nit_visits"] = float(n_fs + n_ft + n_aix)
        feat["nit_coverage_ratio"] = (n_fs + n_ft + n_aix) / (3 * max(n_visits, 1))

        records.append(feat)

    return pd.DataFrame(records, index=df_wide.index)


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add clinically-motivated interaction features."""
    out = df.copy()

    # T2DM × obesity — multiplicative MASLD risk
    if "T2DM" in df.columns and "last_bmi_ge30" in df.columns:
        out["T2DM_x_obese"] = df["T2DM"].fillna(0) * df["last_bmi_ge30"].fillna(0)
    if "T2DM" in df.columns and "last_bmi" in df.columns:
        out["T2DM_x_bmi"] = df["T2DM"].fillna(0) * df["last_bmi"].fillna(0)

    # Hypertension × T2DM — metabolic syndrome proxy
    if "Hypertension" in df.columns and "T2DM" in df.columns:
        out["MetSyn_proxy"] = df["Hypertension"].fillna(0) * df["T2DM"].fillna(0)

    # Comorbidity count
    comorbidity_cols = ["T2DM", "Hypertension", "Dyslipidaemia"]
    if all(c in df.columns for c in comorbidity_cols):
        out["n_comorbidities"] = df[comorbidity_cols].fillna(0).sum(axis=1)

    # Age × FibroScan (age-adjusted liver stiffness)
    if "age_at_last_visit" in df.columns and "last_fibroscan" in df.columns:
        out["age_x_fibroscan"] = df["age_at_last_visit"] * df["last_fibroscan"]

    # FibroScan slope × BMI slope (co-progression)
    if "FibroScan_slope" in df.columns and "BMI_slope" in df.columns:
        out["fibroscan_slope_x_bmi_slope"] = df["FibroScan_slope"] * df["BMI_slope"]

    # ALT × AST ratio elevated (both elevated = worse liver damage)
    if "ever_alt_gt_2x_uln" in df.columns and "ever_ast_gt_uln" in df.columns:
        out["both_transaminases_elevated"] = df["ever_alt_gt_2x_uln"].fillna(0) * df["ever_ast_gt_uln"].fillna(0)

    # Platelets low + FibroScan high = cirrhosis signal
    if "ever_plt_lt150" in df.columns and "ever_fibroscan_ge8" in df.columns:
        out["plt_low_x_fs_high"] = df["ever_plt_lt150"].fillna(0) * df["ever_fibroscan_ge8"].fillna(0)

    return out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def find_data_files() -> tuple[Path, Path | None]:
    data_files = sorted(Path("data").glob("DB-*.csv"))
    train_path = test_path = None
    for f in data_files:
        cols = pd.read_csv(f, nrows=0).columns.tolist()
        if "evenements_hepatiques_majeurs" in cols:
            train_path = f
        else:
            test_path = f
    if train_path is None:
        raise FileNotFoundError("Could not find training CSV.")
    return train_path, test_path


def load_and_prepare(feature_mode: str = "engineered_only") -> dict:
    """Load data and prepare feature matrices.

    feature_mode:
      - "combined_baseline": Original 378 features (base visits + engineered, no vol/derived)
      - "engineered_only": Static + engineered summaries only (~124 features)
      - "engineered_clinical": Engineered + clinical thresholds
      - "engineered_clinical_interactions": Engineered + clinical + visit patterns + interactions
    """
    train_path, test_path = find_data_files()
    df_raw = pd.read_csv(train_path)
    df_test_raw = pd.read_csv(test_path) if test_path else None

    age_cols = sorted(
        [c for c in df_raw.columns if re.fullmatch(r"Age_v\d+", c)],
        key=lambda c: int(c.split("_v")[1]),
    )
    visit_numbers = [int(c.split("_v")[1]) for c in age_cols]
    visit_feature_cols_by_visit = {
        v: [c for c in df_raw.columns if c.endswith(f"_v{v}") and c != f"Age_v{v}"]
        for v in visit_numbers
    }

    def add_age_window(df):
        df = df.copy()
        age_frame = df[age_cols]
        df["age_first"] = age_frame.bfill(axis=1).iloc[:, 0]
        df["age_last"]  = age_frame.ffill(axis=1).iloc[:, -1]
        return df

    def mask_post_event_wide(df, event_age_col=None):
        df = df.copy()
        if event_age_col and event_age_col in df.columns:
            event_age = df[event_age_col]
            for v in visit_numbers:
                ac = f"Age_v{v}"
                vcols = visit_feature_cols_by_visit.get(v, [])
                if ac not in df.columns or not vcols:
                    continue
                mask = event_age.notna() & df[ac].notna() & (df[ac] > event_age)
                if mask.any():
                    df.loc[mask, vcols] = np.nan
        return add_age_window(df)

    def build_base(df, event_age_col=None):
        base = mask_post_event_wide(df, event_age_col)
        base = base.copy()
        base["patient_id"] = base["patient_id_anon"]
        return base

    hep_base = build_base(df_raw, "evenements_hepatiques_age_occur")
    death_base = build_base(df_raw, "death_age_occur")
    df_base_test = build_base(df_test_raw) if df_test_raw is not None else None

    # Targets
    hep_unknown = hep_base["evenements_hepatiques_majeurs"].isna()
    hep_invalid = hep_base["evenements_hepatiques_majeurs"].eq(1) & hep_base["evenements_hepatiques_age_occur"].isna()
    df_model_hep = hep_base.loc[
        ~hep_unknown & ~hep_invalid & hep_base["age_first"].notna() & hep_base["age_last"].notna()
    ].copy()
    df_model_hep["hepatic_event"] = df_model_hep["evenements_hepatiques_majeurs"].astype(bool)
    df_model_hep["hepatic_time"] = np.where(
        df_model_hep["hepatic_event"],
        df_model_hep["evenements_hepatiques_age_occur"],
        df_model_hep["age_last"],
    ) - df_model_hep["age_first"]
    df_model_hep["hepatic_time"] = df_model_hep["hepatic_time"].clip(lower=0.01)

    death_unknown = death_base["death"].isna()
    death_invalid = death_base["death"].eq(1) & death_base["death_age_occur"].isna()
    df_model_death = death_base.loc[
        ~death_unknown & ~death_invalid & death_base["age_first"].notna() & death_base["age_last"].notna()
    ].copy()
    df_model_death["death_event"] = df_model_death["death"].astype(bool)
    df_model_death["death_time"] = np.where(
        df_model_death["death_event"],
        df_model_death["death_age_occur"],
        df_model_death["age_last"],
    ) - df_model_death["age_first"]
    df_model_death["death_time"] = df_model_death["death_time"].clip(lower=0.01)

    y_hepatic = np.array(
        list(zip(df_model_hep["hepatic_event"].values, df_model_hep["hepatic_time"].values)),
        dtype=[("event", bool), ("time", float)],
    )
    y_death = np.array(
        list(zip(df_model_death["death_event"].values, df_model_death["death_time"].values)),
        dtype=[("event", bool), ("time", float)],
    )

    # Build feature matrices based on mode
    static_cols = [c for c in ["gender", "T2DM", "Hypertension", "Dyslipidaemia",
                                "bariatric_surgery", "bariatric_surgery_age"]
                   if c in df_raw.columns]

    def build_features(df_model, df_test, event_age_col, mode):
        # Engineered features
        eng = engineer_patient_features(
            df_model, event_age_col=event_age_col
        ).drop(columns=["source_row", "patient_id_anon"], errors="ignore")

        eng_test = None
        if df_test is not None:
            eng_test = engineer_patient_features(
                df_test, event_age_col=None
            ).drop(columns=["source_row", "patient_id_anon", "trustii_id"], errors="ignore")

        # Filter: no volatility, no derived scores
        def filter_features(df):
            keep = [
                c for c in df.columns
                if not any(c.endswith(s) for s in VOLATILITY_SUFFIXES)
                and not any(c.startswith(p) for p in DERIVED_PREFIXES)
            ]
            return df[keep]

        eng = filter_features(eng)
        if eng_test is not None:
            eng_test = filter_features(eng_test)

        if mode == "combined_baseline":
            # Original: base visit cols + engineered
            base_exclude = {"patient_id", "patient_id_anon", "trustii_id",
                           *OUTCOME_COLS, "age_first", "age_last",
                           "hepatic_event", "hepatic_time", "death_event", "death_time"}
            base_visit = [c for c in df_model.columns if re.search(r"_v\d+$", c)
                         and c not in [f"Age_v{v}" for v in range(1, 23)]]
            base_static = [c for c in static_cols if c in df_model.columns]
            base_feature_cols = [c for c in base_static + base_visit if c not in base_exclude]

            X_base = df_model[base_feature_cols].copy().replace([np.inf, -np.inf], np.nan)
            X = pd.concat([X_base, eng], axis=1)
            X = X.loc[:, ~X.columns.duplicated()]

            if df_test is not None:
                X_test_base = df_test.reindex(columns=base_feature_cols).copy().replace([np.inf, -np.inf], np.nan)
                X_test = pd.concat([X_test_base, eng_test], axis=1)
                X_test = X_test.loc[:, ~X_test.columns.duplicated()]
            else:
                X_test = None
            return X, X_test

        # For all engineered modes: start with static + engineered
        X_static = df_model[static_cols].copy() if static_cols else pd.DataFrame(index=df_model.index)
        X = pd.concat([X_static, eng], axis=1)
        X = X.loc[:, ~X.columns.duplicated()]

        if df_test is not None:
            X_test_static = df_test[static_cols].copy() if static_cols else pd.DataFrame(index=df_test.index)
            X_test = pd.concat([X_test_static, eng_test], axis=1)
            X_test = X_test.loc[:, ~X_test.columns.duplicated()]
        else:
            X_test = None

        if mode == "engineered_only":
            return X, X_test

        # Add clinical threshold features
        clinical_train = add_clinical_threshold_features(df_model, event_age_col)
        # Align by source_row to handle patients with zero valid visits
        clinical_train = clinical_train.set_index("source_row").reindex(df_model.index).reset_index(drop=True)
        clinical_train.index = df_model.index
        X = pd.concat([X, clinical_train], axis=1)
        X = X.loc[:, ~X.columns.duplicated()]

        if df_test is not None:
            clinical_test = add_clinical_threshold_features(df_test, None)
            clinical_test = clinical_test.set_index("source_row").reindex(df_test.index).reset_index(drop=True)
            clinical_test.index = df_test.index
            X_test = pd.concat([X_test, clinical_test], axis=1)
            X_test = X_test.loc[:, ~X_test.columns.duplicated()]

        # Add visit pattern features
        visit_pat_train = add_visit_pattern_features(df_model, event_age_col)
        visit_pat_train.index = df_model.index
        X = pd.concat([X, visit_pat_train], axis=1)
        X = X.loc[:, ~X.columns.duplicated()]

        if df_test is not None:
            visit_pat_test = add_visit_pattern_features(df_test, None)
            visit_pat_test.index = df_test.index
            X_test = pd.concat([X_test, visit_pat_test], axis=1)
            X_test = X_test.loc[:, ~X_test.columns.duplicated()]

        if mode == "engineered_clinical":
            return X, X_test

        # Add interaction features
        X = add_interaction_features(X)
        if X_test is not None:
            X_test = add_interaction_features(X_test)
            # Align columns
            for c in X.columns:
                if c not in X_test.columns:
                    X_test[c] = np.nan
            X_test = X_test[X.columns]

        return X, X_test

    X_hep, X_test_hep = build_features(df_model_hep, df_base_test,
                                         "evenements_hepatiques_age_occur", feature_mode)
    X_death, X_test_death = build_features(df_model_death, df_base_test,
                                             "death_age_occur", feature_mode)

    # Align test to union of columns
    if X_test_hep is not None:
        union_cols = pd.Index(X_hep.columns).union(pd.Index(X_death.columns))
        X_test = X_test_hep.reindex(columns=union_cols, fill_value=np.nan)
    else:
        X_test = None

    # Remove inf
    X_hep = X_hep.replace([np.inf, -np.inf], np.nan)
    X_death = X_death.replace([np.inf, -np.inf], np.nan)
    if X_test is not None:
        X_test = X_test.replace([np.inf, -np.inf], np.nan)

    trustii_ids = None
    if df_test_raw is not None and "trustii_id" in df_test_raw.columns:
        trustii_ids = df_test_raw["trustii_id"]

    print(f"  Feature mode: {feature_mode}")
    print(f"  Hepatic: {X_hep.shape[0]} patients × {X_hep.shape[1]} features ({y_hepatic['event'].sum()} events)")
    print(f"  Death:   {X_death.shape[0]} patients × {X_death.shape[1]} features ({y_death['event'].sum()} events)")
    print(f"  EPV (hepatic): {y_hepatic['event'].sum() / X_hep.shape[1]:.2f}")

    return {
        "X_hep": X_hep, "X_death": X_death, "X_test": X_test,
        "y_hepatic": y_hepatic, "y_death": y_death,
        "trustii_ids": trustii_ids,
        "df_model_hep": df_model_hep, "df_model_death": df_model_death,
        "df_base_test": df_base_test,
    }


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------

def transform_matrices(X_train_raw, X_valid_raw=None, X_apply_raw=None, scale=False):
    missing_rate = X_train_raw.isna().mean()
    keep_cols = missing_rate[missing_rate <= MISSINGNESS_THRESHOLD].index.tolist()
    if not keep_cols:
        raise ValueError("No features passed the missingness filter.")

    imputer = SimpleImputer(strategy="median")
    X_train_imp = imputer.fit_transform(X_train_raw[keep_cols])
    scaler = StandardScaler() if scale else None
    X_train_arr = scaler.fit_transform(X_train_imp) if scaler else X_train_imp
    X_train = pd.DataFrame(X_train_arr, columns=keep_cols, index=X_train_raw.index)

    def _transform(df):
        if df is None:
            return None
        imp = imputer.transform(df.reindex(columns=keep_cols, fill_value=np.nan))
        arr = scaler.transform(imp) if scaler else imp
        return pd.DataFrame(arr, columns=keep_cols, index=df.index)

    return X_train, _transform(X_valid_raw), _transform(X_apply_raw), keep_cols


# ---------------------------------------------------------------------------
# Model wrappers
# ---------------------------------------------------------------------------

class XGBoostAFTWrapper:
    def __init__(self, distribution="normal", num_boost_round=250, max_depth=4,
                 learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
                 min_child_weight=5.0):
        self.distribution = distribution
        self.num_boost_round = num_boost_round
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.colsample_bytree = colsample_bytree
        self.min_child_weight = min_child_weight
        self.booster = None
        self.feature_names_ = None

    def fit(self, X, y):
        self.feature_names_ = list(X.columns)
        dtrain = xgb.DMatrix(X, feature_names=self.feature_names_)
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

    def predict(self, X):
        dm = xgb.DMatrix(X, feature_names=self.feature_names_)
        return -self.booster.predict(dm)


class CatBoostSurvivalAftWrapper:
    def __init__(self, iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=3.0):
        self.iterations = iterations
        self.depth = depth
        self.learning_rate = learning_rate
        self.l2_leaf_reg = l2_leaf_reg
        self.model = None

    def fit(self, X, y):
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

    def predict(self, X):
        return -self.model.predict(X)


# ---------------------------------------------------------------------------
# CV framework
# ---------------------------------------------------------------------------

def combined_score(c_hep, c_death):
    return 0.7 * c_hep + 0.3 * c_death


def cv_endpoint(model_fn, X_raw, y, scale=False):
    strat = y["event"].astype(int)
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = []
    for fold, (tr_idx, te_idx) in enumerate(skf.split(X_raw, strat), 1):
        X_tr, X_te, _, keep_cols = transform_matrices(
            X_raw.iloc[tr_idx], X_raw.iloc[te_idx], scale=scale)
        model = model_fn()
        model.fit(X_tr, y[tr_idx])
        preds = model.predict(X_te)
        c = concordance_index_censored(y[te_idx]["event"], y[te_idx]["time"], preds)[0]
        scores.append(c)
    return np.array(scores)


def run_experiment(name, model_fn, X_hep, y_hep, X_death, y_death, scale=False):
    print(f"\n  {name}")
    try:
        hep_scores = cv_endpoint(model_fn, X_hep, y_hep, scale=scale)
        print(f"    Hepatic: {hep_scores.mean():.4f} ± {hep_scores.std():.4f}  "
              f"[{', '.join(f'{s:.4f}' for s in hep_scores)}]")
    except Exception as e:
        print(f"    Hepatic: FAILED — {e}")
        hep_scores = np.full(N_SPLITS, np.nan)

    try:
        death_scores = cv_endpoint(model_fn, X_death, y_death, scale=scale)
        print(f"    Death:   {death_scores.mean():.4f} ± {death_scores.std():.4f}  "
              f"[{', '.join(f'{s:.4f}' for s in death_scores)}]")
    except Exception as e:
        print(f"    Death:   FAILED — {e}")
        death_scores = np.full(N_SPLITS, np.nan)

    c_h = float(np.nanmean(hep_scores))
    c_d = float(np.nanmean(death_scores))
    comb = combined_score(c_h, c_d)
    print(f"    Combined: {comb:.4f}  (0.7×{c_h:.4f} + 0.3×{c_d:.4f})")
    return {
        "name": name, "c_hepatic": c_h, "c_hepatic_std": float(np.nanstd(hep_scores)),
        "c_death": c_d, "c_death_std": float(np.nanstd(death_scores)),
        "combined": comb, "hep_folds": hep_scores.tolist(), "death_folds": death_scores.tolist(),
    }


# ---------------------------------------------------------------------------
# Submission
# ---------------------------------------------------------------------------

def generate_submission(model_fn, X_hep, y_hep, X_death, y_death,
                        X_test, trustii_ids, filename, scale=False):
    X_hep_tr, _, X_hep_te, _ = transform_matrices(X_hep, X_apply_raw=X_test, scale=scale)
    X_death_tr, _, X_death_te, _ = transform_matrices(X_death, X_apply_raw=X_test, scale=scale)

    model_hep = model_fn()
    model_hep.fit(X_hep_tr, y_hep)
    risk_hep = model_hep.predict(X_hep_te)

    model_death = model_fn()
    model_death.fit(X_death_tr, y_death)
    risk_death = model_death.predict(X_death_te)

    sub = pd.DataFrame({
        "trustii_id": trustii_ids.values if hasattr(trustii_ids, "values") else trustii_ids,
        "risk_hepatic_event": risk_hep,
        "risk_death": risk_death,
    }).sort_values("trustii_id").reset_index(drop=True)
    sub.to_csv(filename, index=False)
    print(f"  Submission saved: {filename}")
    return sub


def generate_ensemble_submission(submissions: list[tuple[str, float]],
                                  filename: str):
    """Rank-average ensemble from existing submission CSVs with weights."""
    dfs = []
    weights = []
    for path, w in submissions:
        dfs.append(pd.read_csv(path))
        weights.append(w)
    weights = np.array(weights) / np.sum(weights)

    ref = dfs[0]
    for endpoint in ["risk_hepatic_event", "risk_death"]:
        ranks = [rankdata(df[endpoint].values) for df in dfs]
        ref[endpoint] = sum(w * r for w, r in zip(weights, ranks))

    ref.sort_values("trustii_id").to_csv(filename, index=False)
    print(f"  Ensemble submission saved: {filename}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Feature revolution ablation")
    parser.add_argument("--variant", choices=[
        "combined_baseline", "engineered_only", "engineered_clinical",
        "engineered_clinical_interactions", "all"
    ], default="all")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args()

    print("=" * 70)
    print("FEATURE REVOLUTION — ANNITIA")
    print("Closing the gap from 0.875 to 0.90+")
    print("=" * 70)

    variants = (
        ["combined_baseline", "engineered_only",
         "engineered_clinical", "engineered_clinical_interactions"]
        if args.variant == "all"
        else [args.variant]
    )

    all_results = []

    for variant in variants:
        print(f"\n{'=' * 50}")
        print(f"VARIANT: {variant}")
        print(f"{'=' * 50}")

        data = load_and_prepare(feature_mode=variant)

        # XGB AFT normal
        result = run_experiment(
            f"XGB AFT [{variant}]",
            lambda: XGBoostAFTWrapper(distribution="normal"),
            data["X_hep"], data["y_hepatic"],
            data["X_death"], data["y_death"],
        )
        result["variant"] = variant
        result["model"] = "xgb_aft"
        all_results.append(result)

        # CatBoost SurvivalAft
        result = run_experiment(
            f"CatBoost AFT [{variant}]",
            lambda: CatBoostSurvivalAftWrapper(),
            data["X_hep"], data["y_hepatic"],
            data["X_death"], data["y_death"],
        )
        result["variant"] = variant
        result["model"] = "catboost_aft"
        all_results.append(result)

        # Generate submissions for this variant if requested
        if args.submit and data["X_test"] is not None and data["trustii_ids"] is not None:
            safe_name = variant.replace("_", "-")

            print(f"\n  Generating submissions for {variant}...")
            xgb_file = f"submission_featrev_xgb_{safe_name}.csv"
            generate_submission(
                lambda: XGBoostAFTWrapper(distribution="normal"),
                data["X_hep"], data["y_hepatic"],
                data["X_death"], data["y_death"],
                data["X_test"], data["trustii_ids"],
                xgb_file,
            )

            cat_file = f"submission_featrev_cat_{safe_name}.csv"
            generate_submission(
                lambda: CatBoostSurvivalAftWrapper(),
                data["X_hep"], data["y_hepatic"],
                data["X_death"], data["y_death"],
                data["X_test"], data["trustii_ids"],
                cat_file,
            )

            # 70/30 ensemble
            ens_file = f"submission_featrev_ens70_{safe_name}.csv"
            generate_ensemble_submission(
                [(xgb_file, 0.7), (cat_file, 0.3)],
                ens_file,
            )

    # Summary table
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    all_results.sort(key=lambda x: -x["combined"])
    print(f"\n{'Rank':<5} {'Name':<55} {'C_hep':>7} {'C_death':>7} {'Combined':>9}")
    print("-" * 85)
    for i, r in enumerate(all_results, 1):
        print(f"{i:<5} {r['name']:<55} {r['c_hepatic']:>7.4f} {r['c_death']:>7.4f} {r['combined']:>9.4f}")

    # Save results
    df_results = pd.DataFrame([{
        "variant": r.get("variant", ""), "model": r.get("model", ""),
        "c_hepatic": r["c_hepatic"], "c_death": r["c_death"],
        "combined": r["combined"],
    } for r in all_results])
    df_results.to_csv("feature_revolution_results.csv", index=False)
    print(f"\nResults saved to feature_revolution_results.csv")


if __name__ == "__main__":
    main()
