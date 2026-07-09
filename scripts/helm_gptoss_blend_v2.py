"""Blend an arbitrary gpt-oss HeLM test-score file with the EffB0 stage-2 base.

Generalised version of helm_gptoss_blend.py: accepts --helm-path and
--weights on the CLI so each Tier-3 axis variant (temp-avg, split, few-shot,
CoT, variant-ensemble) can reuse the same blender without editing constants.

Rank-average blend: final rank = (1-w) * rank(base) + w * rank(helm),
performed per-endpoint and per-column. Output CSVs are sorted by trustii_id.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


DEFAULT_BASE = "submission_blend_base_effb0blend_w02.csv"


def _rank01(a: np.ndarray) -> np.ndarray:
    r = rankdata(a, method="average")
    return (r - 1) / max(1, len(a) - 1)


def _load_helm(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Accept several naming conventions for the score columns.
    rename = {}
    if "risk_hepatic" in df.columns:
        rename["risk_hepatic"] = "h_helm"
    elif "risk_hepatic_event" in df.columns:
        rename["risk_hepatic_event"] = "h_helm"
    if "risk_death" in df.columns:
        rename["risk_death"] = "d_helm"
    df = df.rename(columns=rename)
    if "trustii_id" not in df.columns:
        raise SystemExit(f"{path}: missing trustii_id column")
    df["trustii_id"] = df["trustii_id"].astype(int)
    return df[["trustii_id", "h_helm", "d_helm"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--helm-path", required=True, type=Path,
                    help="CSV with columns trustii_id, risk_hepatic(_event), risk_death")
    ap.add_argument("--base-path", type=Path, default=Path(DEFAULT_BASE))
    ap.add_argument("--weights", type=str, default="0.05,0.07,0.10",
                    help="Comma-separated blend weights")
    ap.add_argument("--tag", type=str, required=True,
                    help="Short identifier used in the output filename: "
                         "submission_helm_<tag>_w<WW>.csv")
    args = ap.parse_args()

    base = pd.read_csv(args.base_path)
    helm = _load_helm(args.helm_path)
    m = base.merge(helm, on="trustii_id", how="left")
    for col in ("h_helm", "d_helm"):
        med = m[col].median()
        n_missing = int(m[col].isna().sum())
        if n_missing:
            print(f"  [WARN] {n_missing} missing HeLM {col} values; filling with median {med:.3f}")
        m[col] = m[col].fillna(med)

    r_base_h = _rank01(m["risk_hepatic_event"].to_numpy())
    r_base_d = _rank01(m["risk_death"].to_numpy())
    r_helm_h = _rank01(m["h_helm"].to_numpy())
    r_helm_d = _rank01(m["d_helm"].to_numpy())

    rho_h = spearmanr(r_base_h, r_helm_h)[0]
    rho_d = spearmanr(r_base_d, r_helm_d)[0]
    print(f"Spearman HeLM-vs-base: hep={rho_h:.3f}  death={rho_d:.3f}")
    print(f"HeLM unique values: hep={m['h_helm'].nunique()}/{len(m)}  "
          f"death={m['d_helm'].nunique()}/{len(m)}")

    m = m.sort_values("trustii_id").reset_index(drop=True)
    # Re-rank after sorting to keep index alignment clean.
    r_base_h = _rank01(m["risk_hepatic_event"].to_numpy())
    r_base_d = _rank01(m["risk_death"].to_numpy())
    r_helm_h = _rank01(m["h_helm"].to_numpy())
    r_helm_d = _rank01(m["d_helm"].to_numpy())

    weights = [float(w) for w in args.weights.split(",")]
    for w in weights:
        h = (1 - w) * r_base_h + w * r_helm_h
        d = (1 - w) * r_base_d + w * r_helm_d
        out = pd.DataFrame({
            "trustii_id": m["trustii_id"].values,
            "risk_hepatic_event": h,
            "risk_death": d,
        }).sort_values("trustii_id")
        path = f"submission_helm_{args.tag}_w{int(round(w*100)):02d}.csv"
        out.to_csv(path, index=False)
        # Sanity vs base rankings (how far this blend moves away from base).
        rho_sh = spearmanr(base["risk_hepatic_event"], out["risk_hepatic_event"])[0]
        rho_sd = spearmanr(base["risk_death"], out["risk_death"])[0]
        print(f"  w={w:.02f}  rho_vs_base: hep={rho_sh:.4f} death={rho_sd:.4f}  -> {path}")


if __name__ == "__main__":
    main()
