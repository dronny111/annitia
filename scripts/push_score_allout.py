"""All-out score push: multiple parallel strategies to maximize public LB score.

Strategies:
  A) 20-seed + 30-seed averaging (more variance reduction)
  B) Per-endpoint optimized XGB/Cat weights (fine-grained grid)
  C) 3-model ensemble: XGB + Cat + GBSA (model diversity)
  D) HeLM blending on each new base at optimal weight
  E) Cross-base rank averaging of top variants

Usage:
    python push_score_allout.py --cv-only        # CV evaluation only
    python push_score_allout.py --submit          # Generate all submissions
    python push_score_allout.py --submit --quick  # 10-seed only (faster)
"""
from __future__ import annotations

import argparse
import warnings
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedGroupKFold
from sksurv.metrics import concordance_index_censored

import xgboost as xgb
from catboost import CatBoostRegressor

from feature_revolution import load_and_prepare, MISSINGNESS_THRESHOLD

warnings.filterwarnings("ignore")

# ── Seeds ──────────────────────────────────────────────────────────────
SEEDS_30 = [42, 0, 7, 13, 99, 17, 37, 53, 71, 89,
            2, 5, 11, 23, 31, 41, 59, 67, 73, 83,
            3, 19, 29, 43, 61, 79, 97, 101, 107, 113]
SEEDS_20 = SEEDS_30[:20]
SEEDS_10 = SEEDS_30[:10]

N_SPLITS = 10
FEATURE_MODE = "engineered_clinical_interactions"


# ── Model fitting ──────────────────────────────────────────────────────
def transform_matrices(X_train_raw, X_valid_raw):
    missing_rate = X_train_raw.isna().mean()
    keep_cols = missing_rate[missing_rate <= MISSINGNESS_THRESHOLD].index.tolist()
    imputer = SimpleImputer(strategy="median")
    X_tr = pd.DataFrame(
        imputer.fit_transform(X_train_raw[keep_cols]),
        columns=keep_cols, index=X_train_raw.index)
    X_val = pd.DataFrame(
        imputer.transform(X_valid_raw.reindex(columns=keep_cols, fill_value=np.nan)),
        columns=keep_cols, index=X_valid_raw.index)
    return X_tr, X_val, imputer, keep_cols


def apply_transform(X_raw, imputer, keep_cols):
    arr = imputer.transform(X_raw.reindex(columns=keep_cols, fill_value=np.nan))
    return pd.DataFrame(arr, columns=keep_cols, index=X_raw.index)


def fit_xgb(X_tr, y_tr, seed):
    feature_names = list(X_tr.columns)
    dtrain = xgb.DMatrix(X_tr, feature_names=feature_names)
    lower = y_tr["time"].astype(float)
    upper = np.where(y_tr["event"], y_tr["time"], np.inf).astype(float)
    dtrain.set_float_info("label_lower_bound", lower)
    dtrain.set_float_info("label_upper_bound", upper)
    params = {
        "objective": "survival:aft",
        "eval_metric": "aft-nloglik",
        "aft_loss_distribution": "normal",
        "aft_loss_distribution_scale": 1.0,
        "tree_method": "hist",
        "learning_rate": 0.05,
        "max_depth": 4,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 5.0,
        "seed": seed,
        "nthread": -1,
        "verbosity": 0,
    }
    booster = xgb.train(params, dtrain, num_boost_round=250)
    return lambda X_te: -booster.predict(
        xgb.DMatrix(X_te, feature_names=feature_names))


def fit_cat(X_tr, y_tr, seed):
    lower = y_tr["time"].astype(float)
    upper = np.where(y_tr["event"], y_tr["time"], 1e6).astype(float)
    labels = np.column_stack([lower, upper])
    model = CatBoostRegressor(
        loss_function="SurvivalAft",
        iterations=250, depth=6, learning_rate=0.05, l2_leaf_reg=3.0,
        random_seed=seed, verbose=False, allow_writing_files=False,
    )
    model.fit(X_tr, labels)
    return lambda X_te: -model.predict(X_te)


def fit_gbsa(X_tr, y_tr, seed):
    """Gradient Boosted Survival Analysis from sksurv."""
    from sksurv.ensemble import GradientBoostingSurvivalAnalysis
    y_structured = np.array(
        [(bool(e), float(t)) for e, t in zip(y_tr["event"], y_tr["time"])],
        dtype=[("event", bool), ("time", float)]
    )
    model = GradientBoostingSurvivalAnalysis(
        n_estimators=100, learning_rate=0.05, max_depth=3,
        subsample=0.8, min_samples_leaf=10,
        random_state=seed,
    )
    model.fit(X_tr.values, y_structured)
    return lambda X_te: model.predict(X_te.values)


def seed_averaged_predict_3model(X_tr, y_tr, X_te, seeds, use_gbsa=False):
    """Returns dict of model_name -> averaged predictions."""
    preds = {"xgb": [], "cat": []}
    if use_gbsa:
        preds["gbsa"] = []
    for seed in seeds:
        preds["xgb"].append(fit_xgb(X_tr, y_tr, seed)(X_te))
        preds["cat"].append(fit_cat(X_tr, y_tr, seed)(X_te))
        if use_gbsa:
            preds["gbsa"].append(fit_gbsa(X_tr, y_tr, seed)(X_te))
    return {k: np.mean(v, axis=0) for k, v in preds.items()}


def rank_blend(pred_dict, weights):
    """Rank-blend predictions with given weights dict."""
    result = np.zeros_like(list(pred_dict.values())[0], dtype=float)
    for model, w in weights.items():
        if model in pred_dict and w > 0:
            result += w * rankdata(pred_dict[model])
    return result


# ── Weight optimization ────────────────────────────────────────────────
def optimize_2model_weight(oof_xgb, oof_cat, y_event, y_time, grid_step=0.02):
    """Fine-grained XGB weight search for 2-model blend."""
    best_w, best_c = 0.5, -1.0
    for w in np.arange(0.20, 0.82, grid_step):
        blended = w * rankdata(oof_xgb) + (1 - w) * rankdata(oof_cat)
        c = concordance_index_censored(y_event, y_time, blended)[0]
        if c > best_c:
            best_w, best_c = w, c
    return best_w, best_c


def optimize_3model_weights(oof_dict, y_event, y_time, step=0.05):
    """Grid search over 3-model weights."""
    best_w, best_c = {"xgb": 0.5, "cat": 0.5, "gbsa": 0.0}, -1.0
    for w_xgb in np.arange(0.2, 0.8, step):
        for w_cat in np.arange(0.1, 0.8 - w_xgb, step):
            w_gbsa = 1.0 - w_xgb - w_cat
            if w_gbsa < 0:
                continue
            blended = (w_xgb * rankdata(oof_dict["xgb"]) +
                       w_cat * rankdata(oof_dict["cat"]) +
                       w_gbsa * rankdata(oof_dict["gbsa"]))
            c = concordance_index_censored(y_event, y_time, blended)[0]
            if c > best_c:
                best_w = {"xgb": w_xgb, "cat": w_cat, "gbsa": w_gbsa}
                best_c = c
    return best_w, best_c


# ── Cross-validation ───────────────────────────────────────────────────
def run_cv(X_hep, y_hep, X_death, y_death, seeds, use_gbsa=False):
    print(f"\n{'='*65}")
    print(f"CV: {len(seeds)} seeds × {N_SPLITS} folds | GBSA={'ON' if use_gbsa else 'OFF'}")
    print(f"{'='*65}")

    sgkf_hep = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    sgkf_death = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
    groups_hep = np.arange(len(X_hep))
    groups_death = np.arange(len(X_death))

    # OOF storage
    n_hep, n_death = len(X_hep), len(X_death)
    oof = {
        "xgb_hep": np.zeros(n_hep), "cat_hep": np.zeros(n_hep),
        "xgb_death": np.zeros(n_death), "cat_death": np.zeros(n_death),
    }
    if use_gbsa:
        oof["gbsa_hep"] = np.zeros(n_hep)
        oof["gbsa_death"] = np.zeros(n_death)

    fold_records = []

    for fold, ((_, (tr_h, te_h)), (_, (tr_d, te_d))) in enumerate(
        zip(
            enumerate(sgkf_hep.split(X_hep, y_hep["event"].astype(int), groups=groups_hep)),
            enumerate(sgkf_death.split(X_death, y_death["event"].astype(int), groups=groups_death)),
        ), 1
    ):
        X_hep_tr, X_hep_te, _, _ = transform_matrices(X_hep.iloc[tr_h], X_hep.iloc[te_h])
        X_death_tr, X_death_te, _, _ = transform_matrices(X_death.iloc[tr_d], X_death.iloc[te_d])

        t0 = time.time()
        preds_h = seed_averaged_predict_3model(X_hep_tr, y_hep[tr_h], X_hep_te, seeds, use_gbsa)
        preds_d = seed_averaged_predict_3model(X_death_tr, y_death[tr_d], X_death_te, seeds, use_gbsa)
        dt = time.time() - t0

        # Store OOF
        for model in preds_h:
            oof[f"{model}_hep"][te_h] = preds_h[model]
            oof[f"{model}_death"][te_d] = preds_d[model]

        # 50/50 XGB+Cat blend
        blend_hep = 0.5 * rankdata(preds_h["xgb"]) + 0.5 * rankdata(preds_h["cat"])
        blend_death = 0.5 * rankdata(preds_d["xgb"]) + 0.5 * rankdata(preds_d["cat"])

        c_h = concordance_index_censored(y_hep[te_h]["event"], y_hep[te_h]["time"], blend_hep)[0]
        c_d = concordance_index_censored(y_death[te_d]["event"], y_death[te_d]["time"], blend_death)[0]
        c_comb = 0.7 * c_h + 0.3 * c_d
        fold_records.append({"fold": fold, "hepatic": c_h, "death": c_d, "combined": c_comb})
        print(f"  Fold {fold}: hep={c_h:.4f} death={c_d:.4f} comb={c_comb:.4f} ({dt:.0f}s)")

    df_folds = pd.DataFrame(fold_records)
    m_h, m_d = df_folds["hepatic"].mean(), df_folds["death"].mean()
    m_c = 0.7 * m_h + 0.3 * m_d
    print(f"\n  Mean 50/50: hep={m_h:.4f} death={m_d:.4f} combined={m_c:.4f}")
    print(f"  Std: comb={df_folds['combined'].std():.4f}")

    # Per-endpoint 2-model optimization
    w_h, c_h_opt = optimize_2model_weight(oof["xgb_hep"], oof["cat_hep"],
                                           y_hep["event"], y_hep["time"])
    w_d, c_d_opt = optimize_2model_weight(oof["xgb_death"], oof["cat_death"],
                                           y_death["event"], y_death["time"])
    opt_2m = 0.7 * c_h_opt + 0.3 * c_d_opt
    print(f"\n  Per-endpoint 2-model optimal:")
    print(f"    Hepatic: XGB {w_h:.2f} / Cat {1-w_h:.2f} → C={c_h_opt:.4f}")
    print(f"    Death:   XGB {w_d:.2f} / Cat {1-w_d:.2f} → C={c_d_opt:.4f}")
    print(f"    Combined: {opt_2m:.4f} (Δ={opt_2m - m_c:+.4f})")

    result = {
        "oof": oof, "m_h": m_h, "m_d": m_d, "m_c": m_c,
        "w_h_2m": w_h, "w_d_2m": w_d, "opt_2m": opt_2m,
    }

    if use_gbsa:
        oof_h_dict = {k.replace("_hep", ""): oof[k] for k in ["xgb_hep", "cat_hep", "gbsa_hep"]}
        oof_d_dict = {k.replace("_death", ""): oof[k] for k in ["xgb_death", "cat_death", "gbsa_death"]}
        w3_h, c3_h = optimize_3model_weights(oof_h_dict, y_hep["event"], y_hep["time"])
        w3_d, c3_d = optimize_3model_weights(oof_d_dict, y_death["event"], y_death["time"])
        opt_3m = 0.7 * c3_h + 0.3 * c3_d
        print(f"\n  Per-endpoint 3-model optimal:")
        print(f"    Hepatic: XGB {w3_h['xgb']:.2f} / Cat {w3_h['cat']:.2f} / GBSA {w3_h['gbsa']:.2f} → C={c3_h:.4f}")
        print(f"    Death:   XGB {w3_d['xgb']:.2f} / Cat {w3_d['cat']:.2f} / GBSA {w3_d['gbsa']:.2f} → C={c3_d:.4f}")
        print(f"    Combined: {opt_3m:.4f} (Δ vs 50/50={opt_3m - m_c:+.4f})")
        result.update({"w3_h": w3_h, "w3_d": w3_d, "opt_3m": opt_3m})

    return result


# ── Submission generation ──────────────────────────────────────────────
def generate_submission(X_hep, y_hep, X_death, y_death, X_test_raw, trustii_ids,
                        seeds, w_hep_dict, w_death_dict, output_csv, use_gbsa=False):
    """Generate a submission with arbitrary model weight dicts."""
    print(f"\n  Generating: {output_csv} ({len(seeds)} seeds)")

    # Hepatic
    keep_h_rate = X_hep.isna().mean()
    keep_h = keep_h_rate[keep_h_rate <= MISSINGNESS_THRESHOLD].index.tolist()
    imp_h = SimpleImputer(strategy="median")
    X_hep_imp = pd.DataFrame(imp_h.fit_transform(X_hep[keep_h]), columns=keep_h, index=X_hep.index)
    X_test_hep = apply_transform(X_test_raw, imp_h, keep_h)
    preds_h = seed_averaged_predict_3model(X_hep_imp, y_hep, X_test_hep, seeds, use_gbsa)
    risk_hep = rank_blend(preds_h, w_hep_dict)

    # Death
    keep_d_rate = X_death.isna().mean()
    keep_d = keep_d_rate[keep_d_rate <= MISSINGNESS_THRESHOLD].index.tolist()
    imp_d = SimpleImputer(strategy="median")
    X_death_imp = pd.DataFrame(imp_d.fit_transform(X_death[keep_d]), columns=keep_d, index=X_death.index)
    X_test_death = apply_transform(X_test_raw, imp_d, keep_d)
    preds_d = seed_averaged_predict_3model(X_death_imp, y_death, X_test_death, seeds, use_gbsa)
    risk_death = rank_blend(preds_d, w_death_dict)

    sub = pd.DataFrame({
        "trustii_id": trustii_ids.values if hasattr(trustii_ids, "values") else trustii_ids,
        "risk_hepatic_event": risk_hep,
        "risk_death": risk_death,
    }).sort_values("trustii_id").reset_index(drop=True)
    sub.to_csv(output_csv, index=False)
    print(f"    Saved: {output_csv} ({len(sub)} rows)")
    return sub


def helm_blend(base_csv, helm_csv, weight_helm, output_csv):
    """Rank-blend a base submission with HeLM at given weight."""
    base = pd.read_csv(base_csv)
    helm = pd.read_csv(helm_csv)
    merged = base.merge(helm, on="trustii_id", suffixes=("_b", "_h"))
    for col in ["risk_hepatic_event", "risk_death"]:
        r_b = rankdata(merged[f"{col}_b"])
        r_h = rankdata(merged[f"{col}_h"])
        merged[col] = (1 - weight_helm) * r_b + weight_helm * r_h
    merged[["trustii_id", "risk_hepatic_event", "risk_death"]].to_csv(output_csv, index=False)
    print(f"    HeLM blend w={weight_helm:.2f}: {output_csv}")


def cross_blend(csv_a, csv_b, weight_b, output_csv):
    """Rank-blend two submission CSVs."""
    a = pd.read_csv(csv_a)
    b = pd.read_csv(csv_b)
    merged = a.merge(b, on="trustii_id", suffixes=("_a", "_b"))
    for col in ["risk_hepatic_event", "risk_death"]:
        r_a = rankdata(merged[f"{col}_a"])
        r_b = rankdata(merged[f"{col}_b"])
        merged[col] = (1 - weight_b) * r_a + weight_b * r_b
    merged[["trustii_id", "risk_hepatic_event", "risk_death"]].to_csv(output_csv, index=False)
    print(f"    Cross blend w_b={weight_b:.2f}: {output_csv}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cv-only", action="store_true")
    parser.add_argument("--submit", action="store_true")
    parser.add_argument("--quick", action="store_true", help="10-seed only")
    parser.add_argument("--no-gbsa", action="store_true", help="Skip GBSA (faster)")
    args = parser.parse_args()

    print("=" * 70)
    print("ALL-OUT SCORE PUSH — ANNITIA")
    print("=" * 70)

    # Load data
    print("\nLoading data...")
    data = load_and_prepare(feature_mode=FEATURE_MODE)
    X_hep = data["X_hep"].replace([np.inf, -np.inf], np.nan)
    X_death = data["X_death"].replace([np.inf, -np.inf], np.nan)
    y_hep = data["y_hepatic"]
    y_death = data["y_death"]

    print(f"  Hepatic: {X_hep.shape[0]} pts × {X_hep.shape[1]} feats ({y_hep['event'].sum()} events)")
    print(f"  Death:   {X_death.shape[0]} pts × {X_death.shape[1]} feats ({y_death['event'].sum()} events)")

    use_gbsa = not args.no_gbsa
    seed_configs = [("10seed", SEEDS_10)]
    if not args.quick:
        seed_configs.append(("20seed", SEEDS_20))

    all_results = {}
    for tag, seeds in seed_configs:
        result = run_cv(X_hep, y_hep, X_death, y_death, seeds, use_gbsa=use_gbsa)
        all_results[tag] = result

    # ── Summary ────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    for tag, r in all_results.items():
        print(f"\n  [{tag}]")
        print(f"    50/50 2-model: hep={r['m_h']:.4f} death={r['m_d']:.4f} combined={r['m_c']:.4f}")
        print(f"    Per-ep 2-model: combined={r['opt_2m']:.4f} (hep w_xgb={r['w_h_2m']:.2f}, death w_xgb={r['w_d_2m']:.2f})")
        if "opt_3m" in r:
            print(f"    Per-ep 3-model: combined={r['opt_3m']:.4f}")

    # ── Submissions ────────────────────────────────────────────────────
    if args.submit and data["X_test"] is not None:
        X_test = data["X_test"].replace([np.inf, -np.inf], np.nan)
        trustii_ids = data["trustii_ids"]

        helm_path = "submission_helm_haiku_zeroshot_precomputed.csv"
        helm_exists = Path(helm_path).exists()

        for tag, seeds in seed_configs:
            r = all_results[tag]

            # A) Uniform 50/50
            base_csv = f"submission_allout_{tag}_50_50.csv"
            generate_submission(
                X_hep, y_hep, X_death, y_death, X_test, trustii_ids, seeds,
                w_hep_dict={"xgb": 0.5, "cat": 0.5},
                w_death_dict={"xgb": 0.5, "cat": 0.5},
                output_csv=base_csv, use_gbsa=False,
            )

            # B) Per-endpoint optimized 2-model
            perep_csv = f"submission_allout_{tag}_perep2m.csv"
            generate_submission(
                X_hep, y_hep, X_death, y_death, X_test, trustii_ids, seeds,
                w_hep_dict={"xgb": r["w_h_2m"], "cat": 1 - r["w_h_2m"]},
                w_death_dict={"xgb": r["w_d_2m"], "cat": 1 - r["w_d_2m"]},
                output_csv=perep_csv, use_gbsa=False,
            )

            # C) 3-model ensemble (if GBSA used)
            if use_gbsa and "w3_h" in r:
                tri_csv = f"submission_allout_{tag}_3model.csv"
                generate_submission(
                    X_hep, y_hep, X_death, y_death, X_test, trustii_ids, seeds,
                    w_hep_dict=r["w3_h"], w_death_dict=r["w3_d"],
                    output_csv=tri_csv, use_gbsa=True,
                )

            # D) HeLM blends on each base
            if helm_exists:
                for base, base_name in [(base_csv, "50_50"), (perep_csv, "perep2m")]:
                    for hw in [0.03, 0.05, 0.07]:
                        out = f"submission_allout_{tag}_{base_name}_helm{int(hw*100):02d}.csv"
                        helm_blend(base, helm_path, hw, out)

        # E) Cross-blend: best 10-seed with new 20-seed (if both exist)
        if not args.quick:
            best_10 = "submission_10seed_50xgb.csv"
            new_20 = "submission_allout_20seed_50_50.csv"
            if Path(best_10).exists() and Path(new_20).exists():
                for wb in [0.3, 0.5, 0.7]:
                    cross_blend(best_10, new_20, wb,
                                f"submission_allout_cross_10v20_w{int(wb*10)}.csv")

        print(f"\n{'='*70}")
        print("All submissions generated. Upload to leaderboard for scoring.")
        print(f"{'='*70}")

    elif not args.submit:
        print("\n  (use --submit to generate submission files)")


if __name__ == "__main__":
    main()
