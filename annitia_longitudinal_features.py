from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler


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

SUMMARY_SUFFIXES = (
    "first",
    "last",
    "mean",
    "std",
    "min",
    "max",
    "slope",
    "rwmean",
    "delta",
    "range",
)

VISIT_PATTERN = re.compile(r"_v(\d+)$")
IDENTIFIER_COLUMNS = ("patient_id_anon", "trustii_id")
FEATURE_METADATA_COLUMNS = ("source_row", *IDENTIFIER_COLUMNS)


def list_visit_numbers(columns: Iterable[str]) -> list[int]:
    visits = {
        int(match.group(1))
        for column in columns
        if (match := VISIT_PATTERN.search(column)) and column.startswith("Age_v")
    }
    return sorted(visits)


def detect_default_train_path(data_dir: str | Path = "data") -> Path:
    data_paths = sorted(Path(data_dir).glob("DB-*.csv"))
    for path in data_paths:
        header = pd.read_csv(path, nrows=0)
        if "evenements_hepatiques_majeurs" in header.columns:
            return path
    raise FileNotFoundError("Could not find a training CSV with outcome columns in data/.")


def safe_polyfit_slope(values: pd.Series, time_index: pd.Series) -> float:
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
    observed = values.dropna().astype(float)
    if observed.empty:
        return np.nan
    weights = np.exp(decay * np.arange(len(observed), dtype=float))
    return float(np.average(observed.to_numpy(), weights=weights))


def summarize_series(values: pd.Series, time_index: pd.Series, prefix: str) -> dict[str, float]:
    observed = values.dropna().astype(float)
    if observed.empty:
        summary = {f"{prefix}_{suffix}": np.nan for suffix in SUMMARY_SUFFIXES}
    else:
        first = float(observed.iloc[0])
        last = float(observed.iloc[-1])
        summary = {
            f"{prefix}_first": first,
            f"{prefix}_last": last,
            f"{prefix}_mean": float(observed.mean()),
            f"{prefix}_std": float(observed.std()) if len(observed) > 1 else 0.0,
            f"{prefix}_min": float(observed.min()),
            f"{prefix}_max": float(observed.max()),
            f"{prefix}_slope": safe_polyfit_slope(values, time_index),
            f"{prefix}_rwmean": recency_weighted_mean(observed),
            f"{prefix}_delta": last - first,
            f"{prefix}_range": float(observed.max() - observed.min()),
        }
    summary[f"{prefix}_missing_rate"] = float(values.isna().mean()) if len(values) else np.nan
    return summary


def add_derived_longitudinal_features(long_df: pd.DataFrame) -> pd.DataFrame:
    derived = long_df.copy()
    alt = derived["ALT"].astype(float)
    ast = derived["AST"].astype(float)
    age = derived["Age"].astype(float)
    platelets = derived["Platelets"].astype(float)

    fib4 = (age * ast) / (platelets * np.sqrt(alt))
    apri = ((ast / 40.0) / platelets) * 100.0
    ast_alt_ratio = ast / alt

    derived["FIB4"] = fib4.replace([np.inf, -np.inf], np.nan)
    derived["APRI"] = apri.replace([np.inf, -np.inf], np.nan)
    derived["AST_ALT_ratio"] = ast_alt_ratio.replace([np.inf, -np.inf], np.nan)
    return derived


def build_long_format(
    df_wide: pd.DataFrame,
    *,
    event_age_col: str | None = None,
) -> pd.DataFrame:
    visit_numbers = list_visit_numbers(df_wide.columns)
    if not visit_numbers:
        raise ValueError("Could not find any Age_v# columns in the provided DataFrame.")

    id_columns = [column for column in IDENTIFIER_COLUMNS if column in df_wide.columns]
    frames: list[pd.DataFrame] = []

    for visit in visit_numbers:
        frame = pd.DataFrame(index=df_wide.index)
        frame["source_row"] = df_wide.index
        for column in id_columns:
            frame[column] = df_wide[column].values
        frame["visit_number"] = visit

        for feature_name, prefix in LONGITUDINAL_FEATURES.items():
            column_name = f"{prefix}{visit}"
            if column_name in df_wide.columns:
                frame[feature_name] = df_wide[column_name].values
            else:
                frame[feature_name] = np.nan

        if event_age_col is not None:
            if event_age_col not in df_wide.columns:
                raise ValueError(f"Missing requested event age column: {event_age_col}")
            frame["_event_age"] = df_wide[event_age_col].values

        frames.append(frame)

    long_df = pd.concat(frames, ignore_index=True)
    long_df = long_df.loc[long_df["Age"].notna()].copy()

    if event_age_col is not None:
        keep_mask = long_df["_event_age"].isna() | (long_df["Age"] <= long_df["_event_age"])
        long_df = long_df.loc[keep_mask].copy()
        long_df.drop(columns="_event_age", inplace=True)

    long_df.sort_values(["source_row", "visit_number"], inplace=True)
    long_df.reset_index(drop=True, inplace=True)
    return add_derived_longitudinal_features(long_df)


def engineer_patient_features(
    df_wide: pd.DataFrame,
    *,
    event_age_col: str | None = None,
    decay: float = 0.3,
) -> pd.DataFrame:
    long_df = build_long_format(df_wide, event_age_col=event_age_col)
    feature_groups = list(LONGITUDINAL_FEATURES) + ["FIB4", "APRI", "AST_ALT_ratio"]
    id_columns = [column for column in IDENTIFIER_COLUMNS if column in df_wide.columns]

    patient_frames: list[pd.Series] = []
    for source_row, patient_visits in long_df.groupby("source_row", sort=True):
        patient_visits = patient_visits.sort_values("visit_number")
        summary: dict[str, float | int | str] = {"source_row": int(source_row), "n_visits": int(len(patient_visits))}

        for column in id_columns:
            if column in patient_visits.columns:
                summary[column] = patient_visits.iloc[0][column]

        age_index = patient_visits["Age"].astype(float)
        for feature_name in feature_groups:
            feature_series = patient_visits[feature_name].astype(float)
            feature_summary = summarize_series(feature_series, age_index, feature_name)
            if decay != 0.3:
                feature_summary[f"{feature_name}_rwmean"] = recency_weighted_mean(
                    feature_series.dropna().astype(float),
                    decay=decay,
                )
            summary.update(feature_summary)

        patient_frames.append(pd.Series(summary))

    aggregated_df = pd.DataFrame(patient_frames)
    base_df = pd.DataFrame({"source_row": df_wide.index})
    for column in id_columns:
        base_df[column] = df_wide[column].values

    feature_df = base_df.merge(aggregated_df, on=["source_row", *id_columns], how="left")
    feature_df["n_visits"] = feature_df["n_visits"].fillna(0).astype(int)
    missing_rate_columns = [f"{feature_name}_missing_rate" for feature_name in feature_groups]
    for column in missing_rate_columns:
        if column in feature_df.columns:
            feature_df[column] = feature_df[column].fillna(1.0)

    feature_df.sort_values("source_row", inplace=True)
    feature_df.index = feature_df["source_row"].astype(df_wide.index.dtype, copy=False)
    feature_df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return feature_df


def summarize_feature_groups(feature_df: pd.DataFrame) -> pd.Series:
    group_names = list(LONGITUDINAL_FEATURES) + ["FIB4", "APRI", "AST_ALT_ratio"]
    allowed_suffixes = "|".join([*SUMMARY_SUFFIXES, "missing_rate"])
    counts = {
        group_name: sum(
            re.fullmatch(rf"{re.escape(group_name)}_(?:{allowed_suffixes})", column) is not None
            for column in feature_df.columns
        )
        for group_name in group_names
    }
    return pd.Series(counts).sort_values(ascending=False)


def get_model_feature_columns(feature_df: pd.DataFrame) -> list[str]:
    return [column for column in feature_df.columns if column not in FEATURE_METADATA_COLUMNS]


class TrajectoryRepresentationBuilder:
    def __init__(
        self,
        *,
        n_pca_components: int = 6,
        n_clusters: int = 6,
        random_state: int = 42,
    ) -> None:
        self.n_pca_components = n_pca_components
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.feature_columns_: list[str] | None = None
        self.imputer_: SimpleImputer | None = None
        self.scaler_: StandardScaler | None = None
        self.pca_: PCA | None = None
        self.clusterer_: KMeans | None = None

    def fit(self, feature_df: pd.DataFrame) -> "TrajectoryRepresentationBuilder":
        feature_columns = get_model_feature_columns(feature_df)
        if not feature_columns:
            raise ValueError("No model feature columns available for trajectory representation fitting.")

        X = feature_df[feature_columns]
        valid_feature_columns = X.columns[X.notna().any(axis=0)].tolist()
        if not valid_feature_columns:
            raise ValueError("No non-empty model feature columns available for trajectory representation fitting.")

        X = X[valid_feature_columns]
        self.feature_columns_ = valid_feature_columns
        self.imputer_ = SimpleImputer(strategy="median")
        X_imp = self.imputer_.fit_transform(X)

        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_imp)

        pca_components = max(1, min(self.n_pca_components, X_scaled.shape[0], X_scaled.shape[1]))
        self.pca_ = PCA(n_components=pca_components, random_state=self.random_state)
        X_pca = self.pca_.fit_transform(X_scaled)

        cluster_count = max(2, min(self.n_clusters, X_pca.shape[0]))
        self.clusterer_ = KMeans(n_clusters=cluster_count, n_init=10, random_state=self.random_state)
        self.clusterer_.fit(X_pca)
        return self

    def transform(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        if not all(
            component is not None
            for component in (self.feature_columns_, self.imputer_, self.scaler_, self.pca_, self.clusterer_)
        ):
            raise ValueError("TrajectoryRepresentationBuilder must be fit before calling transform().")

        X = feature_df.reindex(columns=self.feature_columns_, fill_value=np.nan)
        X_imp = self.imputer_.transform(X)
        X_scaled = self.scaler_.transform(X_imp)
        X_pca = self.pca_.transform(X_scaled)
        cluster_labels = self.clusterer_.predict(X_pca)
        cluster_distances = self.clusterer_.transform(X_pca)

        repr_df = pd.DataFrame(index=feature_df.index)
        for idx in range(X_pca.shape[1]):
            repr_df[f"traj_pca_{idx + 1}"] = X_pca[:, idx]
        repr_df["traj_cluster"] = cluster_labels.astype(int)
        for idx in range(cluster_distances.shape[1]):
            repr_df[f"traj_cluster_dist_{idx + 1}"] = cluster_distances[:, idx]
        return repr_df

    def fit_transform(self, feature_df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(feature_df).transform(feature_df)


def build_trajectory_representation_features(
    train_feature_df: pd.DataFrame,
    apply_feature_df: pd.DataFrame | None = None,
    *,
    n_pca_components: int = 6,
    n_clusters: int = 6,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame | None, TrajectoryRepresentationBuilder]:
    builder = TrajectoryRepresentationBuilder(
        n_pca_components=n_pca_components,
        n_clusters=n_clusters,
        random_state=random_state,
    )
    train_repr = builder.fit_transform(train_feature_df)
    apply_repr = builder.transform(apply_feature_df) if apply_feature_df is not None else None
    return train_repr, apply_repr, builder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Engineer patient-level longitudinal features from ANNITIA wide-format data."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        default=None,
        help="Path to an ANNITIA CSV. Defaults to the detected training CSV in data/.",
    )
    parser.add_argument(
        "--event-age-col",
        default="evenements_hepatiques_age_occur",
        help="Event age column used to exclude post-event visits. Use '' to disable masking.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Optional CSV path to save the engineered feature matrix.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_path = args.dataset_path or detect_default_train_path()
    event_age_col = args.event_age_col or None

    df_wide = pd.read_csv(dataset_path)
    feature_df = engineer_patient_features(df_wide, event_age_col=event_age_col)
    group_counts = summarize_feature_groups(feature_df)

    print(f"Loaded dataset: {dataset_path}")
    print(f"Input shape: {df_wide.shape}")
    print(f"Engineered feature matrix shape: {feature_df.shape}")
    print("Top feature groups created:")
    print(group_counts.to_string())

    if args.output_path is not None:
        feature_df.to_csv(args.output_path, index=False)
        print(f"Saved engineered features to: {args.output_path}")


if __name__ == "__main__":
    main()
