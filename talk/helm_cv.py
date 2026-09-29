"""Test the GPT-OSS HeLM blend inside the window-free CV: the replication the talk asks for.

Phase 1 (CPU, about 30 min): window-free 10-seed CV of the XGBoost + CatBoost base, saving
per-patient out-of-fold ranks to ../data/window_free_oof.csv (git-ignored).
Phase 2: for each finished variant (split: one prompt per endpoint; joint: one prompt for
both), merge the zero-shot HeLM scores from helm_train_scores.py, blend at the submitted
GPT-OSS weight 0.07 (fixed in advance), and write aggregates only:

* talk/data/cv_scores.csv   rows with stage "cv_helm"
* talk/data/cv_folds.csv    per-fold paired gains, for the sign test
* talk/data/cv_tests.csv    patient-bootstrap mean and SE of each paired gain

Also compares XGBoost alone against the pipeline's 50/50 XGBoost + CatBoost blend, a weight the
leaky CV approved. That part needs only phase 1.

    python talk/helm_cv.py
"""
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.model_selection import StratifiedGroupKFold

sys.argv = sys.argv[:1]
sys.path.insert(0, str(Path(__file__).resolve().parent))
import export_scores as E  # noqa: E402  (sets cwd and paths, imports the repo pipeline)

OOF = E.ROOT / "data" / "window_free_oof.csv"
HELM = E.ROOT / "data" / "helm_train_scores.csv"
FOLDS = E.OUT.parent / "cv_folds.csv"
TESTS = E.OUT.parent / "cv_tests.csv"
W = 0.07                          # the GPT-OSS weight used in the submitted blend, not tuned here
N_BOOT = 1000


def rank01(a):
    return (rankdata(a) - 1) / max(1, len(a) - 1)


def load():
    d, X = E.features(drop_window=True)
    ids = {"hep": d["df_model_hep"]["patient_id"].to_numpy(), "death": d["df_model_death"]["patient_id"].to_numpy()}
    return X, {"hep": d["y_hepatic"], "death": d["y_death"]}, ids


def phase1(X, y, ids):
    oof = E.P.run_cv(X["hep"], y["hep"], X["death"], y["death"], E.P.SEEDS_10)["oof"]
    rows = []
    for ep in ("hep", "death"):
        xgb, cat = rankdata(oof[f"xgb_{ep}"]), rankdata(oof[f"cat_{ep}"])
        rows += [(ep, pid, x + c, x, c) for pid, x, c in zip(ids[ep], xgb, cat)]   # base = 50/50 blend
    pd.DataFrame(rows, columns=["endpoint", "patient_id_anon", "base", "xgb", "cat"]).to_csv(OOF, index=False)


def fold_gains(e, t, new, old):
    """Per-fold C(new) - C(old) on the pipeline's own splitter (same seed, same folds)."""
    split = StratifiedGroupKFold(n_splits=E.P.N_SPLITS, shuffle=True, random_state=42)
    out = []
    for _, te in split.split(new.reshape(-1, 1), e.astype(int), groups=np.arange(len(e))):
        c = lambda s: E.concordance_index_censored(e[te], t[te], s[te])[0]
        out.append(c(new) - c(old))
    return out


def boot_se(e, t, new, old, seed=0):
    """Patient-level paired bootstrap SE of C(new) - C(old)."""
    rng, boot = np.random.default_rng(seed), []
    while len(boot) < N_BOOT:
        i = rng.integers(0, len(e), len(e))
        if e[i].any():
            c = lambda s: E.concordance_index_censored(e[i], t[i], s[i])[0]
            boot.append(c(new) - c(old))
    return np.array(boot)


def write_folds(comparison, folds):
    new = not FOLDS.is_file()
    keep = [] if new else [r for r in csv.reader(FOLDS.open()) if r and r[0] not in ("comparison", comparison)]
    with FOLDS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "endpoint", "fold", "delta"])
        w.writerows(keep + [[comparison, *r] for r in folds])


def write_test(comparison, boot):
    """Store one comparison's patient-bootstrap mean and SE, replacing any earlier row for it."""
    keep = [] if not TESTS.is_file() else [r for r in csv.reader(TESTS.open()) if r and r[0] not in ("comparison", comparison)]
    with TESTS.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["comparison", "estimate", "se"])
        w.writerows(keep + [[comparison, f"{boot.mean():.5f}", f"{boot.std(ddof=1):.5f}"]])


def weights_check(y, ids):
    """XGBoost alone against the 50/50 blend, inside the window-free CV."""
    oof = pd.read_csv(OOF)
    folds, boots = [], {}
    for ep in ("hep", "death"):
        o = oof[oof.endpoint == ep].set_index("patient_id_anon").loc[ids[ep]]
        e, t = y[ep]["event"], y[ep]["time"]
        xgb, base = o["xgb"].to_numpy(float), o["base"].to_numpy(float)
        folds += [(ep, k, d) for k, d in enumerate(fold_gains(e, t, xgb, base), 1)]
        boots[ep] = boot_se(e, t, xgb, base)
    write_folds("xgb_only_vs_blend", folds)
    b = 0.7 * boots["hep"] + 0.3 * boots["death"]
    write_test("xgb_only_vs_blend", b)
    print(f"XGBoost alone minus 50/50 blend, weighted: {b.mean():+.4f} ± {b.std(ddof=1):.4f} (bootstrap)")


# variant -> (scores file, stage in cv_scores.csv, comparison name in cv_folds/cv_tests)
# Other LLMs, scored the same way by helm_train_scores.py: name -> file stem.
OTHERS = {"nemotron": "nvidia-nemotron-3-super-120b-a12b-free_think",   # OpenRouter free, reasoning on
          "medgemma": "medgemma1-5-latest"}                              # local Ollama, thinking off
VARIANTS = {"split": (HELM, "cv_helm", "helm_blend_vs_base"),
            "joint": (E.ROOT / "data" / "helm_train_scores_joint.csv", "cv_helm_joint", "helm_joint_blend_vs_base")}
PAIRS = [("split", "joint", "helm_joint_vs_split_blend")]              # (split, joint, comparison)
for name, stem in OTHERS.items():
    for mode in ("split", "joint"):
        VARIANTS[f"{name}_{mode}"] = (E.ROOT / "data" / f"helm_train_scores_{mode}_{stem}.csv",
                                      f"cv_helm_{name}_{mode}", f"helm_{name}_{mode}_blend_vs_base")
    PAIRS.append((f"{name}_split", f"{name}_joint", f"helm_{name}_joint_vs_split_blend"))


def blends(y, ids, path):
    """Per endpoint: event, time, base ranks and the base + GPT-OSS blend at weight W."""
    oof, helm = pd.read_csv(OOF), pd.read_csv(path).set_index("patient_id_anon")
    out = {}
    for ep in ("hep", "death"):
        o = oof[oof.endpoint == ep].set_index("patient_id_anon").loc[ids[ep]]
        h = helm[f"helm_{ep}"].reindex(ids[ep])
        h = h.fillna(h.median()).to_numpy()           # failed calls get the median rank
        base, llm = rank01(o["base"].to_numpy()), rank01(h)
        out[ep] = (y[ep]["event"], y[ep]["time"], base, llm, (1 - W) * base + W * llm)
    return out


def phase2(y, ids, variant):
    path, stage, comparison = VARIANTS[variant]
    b = blends(y, ids, path)
    c = {ep: {k: E.cindex(y[ep], v) for k, v in zip(("base", "llm", "blend"), b[ep][2:])} for ep in b}
    folds = [(ep, k, d) for ep in b for k, d in enumerate(fold_gains(b[ep][0], b[ep][1], b[ep][4], b[ep][2]), 1)]
    boot_w = 0.7 * boot_se(*b["hep"][:2], b["hep"][4], b["hep"][2]) + 0.3 * boot_se(*b["death"][:2], b["death"][4], b["death"][2])
    kept = [r for r in pd.read_csv(E.OUT).itertuples(index=False) if r.stage != stage]
    E.write_rows([(r.stage, r.model, r.c_hep, r.c_death) for r in kept])      # rerun-safe
    E.write_rows([(stage, "window-free base", c["hep"]["base"], c["death"]["base"]),
                  (stage, "GPT-OSS HeLM alone, window-free text", c["hep"]["llm"], c["death"]["llm"]),
                  (stage, f"base + GPT-OSS at weight {W}", c["hep"]["blend"], c["death"]["blend"])],
                 mode="a")
    write_folds(comparison, folds)
    write_test(comparison, boot_w)
    w = lambda k: 0.7 * c["hep"][k] + 0.3 * c["death"][k]
    print(f"{variant}: base {w('base'):.4f} | LLM alone {w('llm'):.4f} | blend {w('blend'):.4f}"
          f" | gain {w('blend') - w('base'):+.4f} ± {boot_w.std(ddof=1):.4f} (bootstrap)")
    return b


def joint_vs_split(y, ids, split, joint, comparison):
    """Paired test of the two blends against each other: does one prompt beat two?"""
    folds, boot = [], {}
    for ep in ("hep", "death"):
        e, t, _, _, bs = split[ep]
        bj = joint[ep][4]
        folds += [(ep, k, d) for k, d in enumerate(fold_gains(e, t, bj, bs), 1)]
        boot[ep] = boot_se(e, t, bj, bs)
    write_folds(comparison, folds)
    b = 0.7 * boot["hep"] + 0.3 * boot["death"]
    write_test(comparison, b)
    print(f"{comparison}: {b.mean():+.4f} ± {b.std(ddof=1):.4f} (bootstrap)")


def main():
    X, y, ids = load()
    if not OOF.is_file() or "xgb" not in pd.read_csv(OOF, nrows=0).columns:
        phase1(X, y, ids)
    weights_check(y, ids)
    done = {}
    for variant, (path, _, _) in VARIANTS.items():
        if path.is_file() and pd.read_csv(path).shape[0] >= len(ids["hep"]):
            done[variant] = phase2(y, ids, variant)
        else:
            print(f"{variant} HeLM training scores incomplete; rerun after helm_train_scores.py finishes")
    for sp, jt, comparison in PAIRS:
        if sp in done and jt in done:
            joint_vs_split(y, ids, done[sp], done[jt], comparison)


if __name__ == "__main__":
    main()
