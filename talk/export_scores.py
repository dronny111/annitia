"""Run the repo's own cross-validation and keep only aggregate C-index numbers for the talk.

Needs the challenge CSVs under ../data/. Writes talk/data/cv_scores.csv with one row per model:
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


def cindex(y, score):
    return concordance_index_censored(y["event"], y["time"], score)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    args = ap.parse_args()

    data = load_and_prepare(feature_mode=P.FEATURE_MODE)
    X = {k: data[f"X_{k}"].replace([np.inf, -np.inf], np.nan) for k in ("hep", "death")}
    X = {k: v.drop(columns=list(P.LABEL_PROXIES & set(v.columns))) for k, v in X.items()}
    y = {"hep": data["y_hepatic"], "death": data["y_death"]}

    r = P.run_cv(X["hep"], y["hep"], X["death"], y["death"], P.SEEDS_30[:args.seeds])
    oof = r["oof"]
    rows = []
    for model in ("xgb", "cat"):
        h, d = cindex(y["hep"], oof[f"{model}_hep"]), cindex(y["death"], oof[f"{model}_death"])
        rows.append(("cv", f"{model} ({args.seeds} seeds)", h, d))
    h = cindex(y["hep"], rankdata(oof["xgb_hep"]) + rankdata(oof["cat_hep"]))
    d = cindex(y["death"], rankdata(oof["xgb_death"]) + rankdata(oof["cat_death"]))
    rows.append(("cv", "xgb + cat rank blend", h, d))

    effnet = ROOT / "ablation_finetune_efficientnet_results.csv"
    if effnet.is_file():                          # aggregate ablation table, one row per cell
        with effnet.open(newline="") as f:
            for rec in csv.DictReader(f):
                if rec.get("cell_id") == "cox-sgd-none-fuse":
                    rows.append(("cv", "EfficientNet-B0 cox-sgd-none-fuse",
                                 float(rec["c_hep"]), float(rec["c_death"])))

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stage", "model", "c_hep", "c_death", "weighted"])
        for stage, model, h, d in rows:
            w.writerow([stage, model, f"{h:.4f}", f"{d:.4f}", f"{0.7 * h + 0.3 * d:.4f}"])
    print(f"wrote {len(rows)} aggregate rows -> {OUT}")


if __name__ == "__main__":
    main()
