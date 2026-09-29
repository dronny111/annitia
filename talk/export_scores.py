"""Run the repo's own cross-validation, as is and window-free, keeping only aggregate C-indices.

Needs the challenge CSVs under ../data/. Writes talk/data/cv_scores.csv with one row per model, and talk/data/cohort_stats.csv with
event counts:
no patient IDs, rows or predictions leave this script.

    python talk/export_scores.py            # 10 seeds, as in the final base
    python talk/export_scores.py --seeds 3  # quicker
"""
import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
from scipy.stats import rankdata
from sksurv.metrics import concordance_index_censored

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "talk" / "data" / "cv_scores.csv"
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)                                   # the loader globs data/ relative to cwd

import push_score_allout as P                    # noqa: E402
from feature_revolution import load_and_prepare  # noqa: E402


# What the pipeline dropped before the leak was fixed in feature_revolution.load_and_prepare.
OLD_PROXIES = ("follow_up_years", "Age_delta")


def features(drop_window: bool):
    """Feature matrices, window-free (the fixed pipeline) or as the pipeline ran before the fix."""
    d = load_and_prepare(feature_mode=P.FEATURE_MODE, drop_window=drop_window)
    X = {k: d[f"X_{k}"].replace([np.inf, -np.inf], np.nan) for k in ("hep", "death")}
    if not drop_window:
        X = {k: v.drop(columns=[c for c in OLD_PROXIES if c in v.columns]) for k, v in X.items()}
    return d, X


def cindex(y, score):
    return concordance_index_censored(y["event"], y["time"], score)[0]


def cv_rows(X, y, seeds, stage, suffix):
    """The repo's own CV; one aggregate row per model and one for their rank blend."""
    oof = P.run_cv(X["hep"], y["hep"], X["death"], y["death"], P.SEEDS_30[:seeds])["oof"]
    rows = [(stage, f"{m} ({seeds} seeds){suffix}", cindex(y["hep"], oof[f"{m}_hep"]),
             cindex(y["death"], oof[f"{m}_death"])) for m in ("xgb", "cat")]
    h = cindex(y["hep"], rankdata(oof["xgb_hep"]) + rankdata(oof["cat_hep"]))
    d = cindex(y["death"], rankdata(oof["xgb_death"]) + rankdata(oof["cat_death"]))
    return rows + [(stage, f"xgb + cat rank blend{suffix}", h, d)]


def proxy_rows(X, y):
    """C-index of the follow-up span alone (last-visit age minus first-visit age): a leak probe."""
    c = {}
    for ep in ("hep", "death"):
        span = (X[ep]["Age_last"] - X[ep]["Age_first"]).to_numpy(float)
        ok = ~np.isnan(span)
        c[ep] = concordance_index_censored(y[ep]["event"][ok], y[ep]["time"][ok], -span[ok])[0]
    return [("proxy", "follow-up span alone (leak probe)", c["hep"], c["death"])]


def write_rows(rows, mode="w"):
    OUT.parent.mkdir(exist_ok=True)
    new = mode == "w" or not OUT.is_file()
    with OUT.open(mode, newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["stage", "model", "c_hep", "c_death", "weighted"])
        for stage, model, h, d in rows:
            w.writerow([stage, model, f"{h:.4f}", f"{d:.4f}", f"{0.7 * h + 0.3 * d:.4f}"])


def write_cohort_stats(y):
    """Patients and events per endpoint, for the gate noise floor."""
    stats = OUT.parent / "cohort_stats.csv"
    with stats.open("w", newline="") as f:
        csv.writer(f).writerows([["endpoint", "n_patients", "events"],
                                 *[[ep, len(y[ep]), int(y[ep]["event"].sum())] for ep in ("hep", "death")]])
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    data, X = features(drop_window=False)                             # as the pipeline ran
    _, X_clean = features(drop_window=True)                           # the fixed pipeline
    y = {"hep": data["y_hepatic"], "death": data["y_death"]}

    rows = cv_rows(X, y, args.seeds, "cv", "")
    rows += cv_rows(X_clean, y, args.seeds, "cv_clean", ", window-free")

    effnet = ROOT / "ablation_finetune_efficientnet_results.csv"
    if effnet.is_file():                          # aggregate ablation table, one row per cell
        with effnet.open(newline="") as f:
            for rec in csv.DictReader(f):
                if rec.get("cell_id") == "cox-sgd-none-fuse":
                    rows.append(("cv", "EfficientNet-B0 cox-sgd-none-fuse",
                                 float(rec["c_hep"]), float(rec["c_death"])))

    rows += proxy_rows(X, y)
    write_rows(rows)
    print(f"wrote {len(rows)} aggregate rows -> {OUT}")
    print(f"wrote aggregate event counts -> {write_cohort_stats(y)}")


if __name__ == "__main__":
    main()
