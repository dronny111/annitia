from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
import time
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable


REPO = Path(__file__).resolve().parents[1]
MODEL_PATH = REPO / "autoannitia_model.py"
RESULTS_PATH = REPO / "autoannitia_results.tsv"
RUN_LOG_PATH = REPO / "run.log"
SESSION_DIR = REPO / ".autoannitia"
STATE_PATH = SESSION_DIR / "autoannitia_runner_state.json"
LOG_PATH = SESSION_DIR / "autoannitia_runner.log"
PID_PATH = SESSION_DIR / "autoannitia_runner.pid"


@dataclass
class EvalResult:
    commit: str
    cv_combined: float
    cv_hepatic: float
    cv_death: float
    n_features: int
    w_xgb: float
    seeds: list[int]
    time_s: float
    fold_scores_hep: list[float]
    fold_scores_dth: list[float]


def log(message: str) -> None:
    SESSION_DIR.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as fh:
        fh.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}\n")
    print(message, flush=True)


def git(*args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "--no-pager", *args],
        cwd=REPO,
        text=True,
        capture_output=True,
        check=False,
    )
    if check and proc.returncode != 0:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip() or f"git {' '.join(args)} failed")
    return proc.stdout.strip()


def sanitize_description(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text).strip("_")
    return text or "experiment"


def config_signature(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True)


def exec_config_text(text: str) -> dict[str, Any]:
    namespace: dict[str, Any] = {}
    exec(text, namespace)
    return namespace["get_config"]()


def load_config_from_rev(rev: str) -> tuple[dict[str, Any], str]:
    text = git("show", f"{rev}:autoannitia_model.py")
    return exec_config_text(text), text


def load_current_config() -> tuple[dict[str, Any], str]:
    text = MODEL_PATH.read_text(encoding="utf-8")
    return exec_config_text(text), text


def format_config(config: dict[str, Any]) -> str:
    fg = config["feature_groups"]
    xgb_params = config["xgb_params"]
    cat_params = config["cat_params"]
    return f'''"""
autoannitia_model.py — MUTABLE EXPERIMENT FILE
================================================
This is the ONLY file the autonomous research agent modifies.
Everything else in the repository is off-limits.

Interface contract:
    get_config() -> dict   (read by autoannitia_eval.py)

Current baseline: 165-feature clinical-only set, 50/50 XGB+Cat ensemble.
Public score: 0.89104 (submission_clinical_only_w50.csv, 2026-03-29)
"""


def get_config() -> dict:
    return {{
        # ── Ensemble ─────────────────────────────────────────────────────────
        # w_xgb: weight for XGBoost AFT in rank-average blend (0.0 = pure CatBoost)
        # Current best: 0.5 (50/50). Range to explore: 0.3 – 0.8
        "w_xgb": {repr(config["w_xgb"])},

        # ── Seeds ─────────────────────────────────────────────────────────────
        # Multiple seeds: each seed runs a full 5-fold CV; results are averaged.
        # More seeds = more stable estimates but slower evaluation.
        # Single seed (~60-90s). Three seeds (~3-4 min). Five seeds (~6-8 min).
        "seeds": {repr(config["seeds"])},

        # ── Feature groups ───────────────────────────────────────────────────
        # True  = include the feature group
        # False = exclude the feature group
        # NEVER change: label proxies (follow_up_years, Age_delta) are always excluded
        "feature_groups": {{
            # Core longitudinal summaries (first/last/mean/min/max/delta)
            # Disabling this would remove the main signal — don't disable
            "longitudinal_core": {repr(fg["longitudinal_core"])},

            # Slope features (_slope suffix)
            # Historical: removing slopes improved public score in some runs
            "slope": {repr(fg["slope"])},

            # Recency-weighted mean (_rwmean suffix)
            "rwmean": {repr(fg["rwmean"])},

            # Missingness indicators (_missing_rate suffix)
            "missing_flags": {repr(fg["missing_flags"])},

            # Volatility features (_std, _range suffixes)
            # Currently OFF — removing these improved public score from 0.86759 to 0.87279
            "volatility": {repr(fg["volatility"])},

            # Continuous derived scores (FIB4_*, APRI_*, AST_ALT_ratio_*)
            # Currently OFF — these are collinear and hurt when used as continuous summaries
            "derived_scores_continuous": {repr(fg["derived_scores_continuous"])},

            # Clinical threshold features (ever_fibroscan_ge8, ever_plt_lt150, etc.)
            "clinical_thresholds": {repr(fg["clinical_thresholds"])},

            # Visit pattern features (visit_density, age_at_first/last_visit, NIT coverage)
            # NOTE: follow_up_years is always excluded regardless of this flag
            "visit_patterns": {repr(fg["visit_patterns"])},

            # Interaction features (T2DM_x_obese, MetSyn_proxy, etc.)
            "interactions": {repr(fg["interactions"])},

            # Age-robust features (_slope_per_yr, _rel_change, _last_age_resid)
            "age_robust": {repr(fg["age_robust"])},

            # NIT timing features: age-of-first-crossing thresholds for FibroScan/FibroTest
            # Log-Rank significant (FibroScan ≥8 kPa: p=7.8e-4 hepatic, p=9.3e-10 death)
            # Currently OFF — not yet tested in CV
            "nit_timing": {repr(fg["nit_timing"])},

            # Clinical binary thresholds: APRI_ever_ge1, FIB4_ever_ge267, AST_2xULN,
            # GGT/ALT age-of-first-ULN (from Log-Rank analysis, Entry 007)
            # Currently OFF — not yet tested in CV
            "clinical_binary_thresholds": {repr(fg["clinical_binary_thresholds"])},
        }},

        # ── XGBoost AFT hyperparameters ───────────────────────────────────────
        # Baseline: the defaults used in annitia_pipeline.py that produced 0.89104
        # DO NOT tune with Optuna (history: Optuna CV 0.9625 → public 0.761)
        # Change ONE parameter at a time and check fold stability
        "xgb_params": {{
            "num_boost_round": {repr(xgb_params["num_boost_round"])},
            "max_depth": {repr(xgb_params["max_depth"])},
            "learning_rate": {repr(xgb_params["learning_rate"])},
            "subsample": {repr(xgb_params["subsample"])},
            "colsample_bytree": {repr(xgb_params["colsample_bytree"])},
            "min_child_weight": {repr(xgb_params["min_child_weight"])},
        }},

        # ── CatBoost SurvivalAft hyperparameters ──────────────────────────────
        "cat_params": {{
            "iterations": {repr(cat_params["iterations"])},
            "depth": {repr(cat_params["depth"])},
            "learning_rate": {repr(cat_params["learning_rate"])},
            "l2_leaf_reg": {repr(cat_params["l2_leaf_reg"])},
        }},
    }}
'''


def write_config(config: dict[str, Any]) -> None:
    MODEL_PATH.write_text(format_config(config), encoding="utf-8")


def read_results_rows() -> list[dict[str, str]]:
    if not RESULTS_PATH.exists():
        return []
    with RESULTS_PATH.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh, delimiter="\t"))


def append_results_row(row: dict[str, str]) -> None:
    header = [
        "commit",
        "cv_combined",
        "cv_hepatic",
        "cv_death",
        "n_features",
        "w_xgb",
        "seeds",
        "status",
        "description",
    ]
    exists = RESULTS_PATH.exists()
    with RESULTS_PATH.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=header, delimiter="\t")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def run_eval() -> EvalResult:
    with RUN_LOG_PATH.open("w", encoding="utf-8") as fh:
        proc = subprocess.run(
            [sys.executable, "autoannitia_eval.py"],
            cwd=REPO,
            stdout=fh,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    text = RUN_LOG_PATH.read_text(encoding="utf-8")
    if proc.returncode != 0 or "cv_combined:" not in text:
        tail = "\n".join(text.splitlines()[-30:])
        raise RuntimeError(f"evaluation failed\n{tail}")
    head = git("rev-parse", "--short", "HEAD")
    return parse_run_log(text, head)


def parse_run_log(text: str, commit: str) -> EvalResult:
    def get_float(key: str) -> float:
        match = re.search(rf"^{re.escape(key)}:\s*([0-9.]+)", text, re.MULTILINE)
        if not match:
            raise RuntimeError(f"missing {key} in run.log")
        return float(match.group(1))

    def get_int(key: str) -> int:
        match = re.search(rf"^{re.escape(key)}:\s*(\d+)", text, re.MULTILINE)
        if not match:
            raise RuntimeError(f"missing {key} in run.log")
        return int(match.group(1))

    def get_list(key: str) -> list[float]:
        match = re.search(rf"^{re.escape(key)}:\s*([0-9.,-]+)", text, re.MULTILINE)
        if not match:
            raise RuntimeError(f"missing {key} in run.log")
        return [float(x) for x in match.group(1).split(",") if x]

    seeds_match = re.search(r"^seeds:\s*([0-9,]+)", text, re.MULTILINE)
    if not seeds_match:
        raise RuntimeError("missing seeds in run.log")
    seeds = [int(x) for x in seeds_match.group(1).split(",") if x]
    return EvalResult(
        commit=commit,
        cv_combined=get_float("cv_combined"),
        cv_hepatic=get_float("cv_hepatic"),
        cv_death=get_float("cv_death"),
        n_features=get_int("n_features"),
        w_xgb=get_float("w_xgb"),
        seeds=seeds,
        time_s=get_float("time_s"),
        fold_scores_hep=get_list("fold_scores_hep"),
        fold_scores_dth=get_list("fold_scores_dth"),
    )


def row_from_eval(result: EvalResult, status: str, description: str) -> dict[str, str]:
    return {
        "commit": result.commit,
        "cv_combined": f"{result.cv_combined:.4f}",
        "cv_hepatic": f"{result.cv_hepatic:.4f}",
        "cv_death": f"{result.cv_death:.4f}",
        "n_features": str(result.n_features),
        "w_xgb": f"{result.w_xgb:.2f}",
        "seeds": ",".join(str(s) for s in result.seeds),
        "status": status,
        "description": description,
    }


def get_best_logged_row(rows: list[dict[str, str]]) -> dict[str, str]:
    candidates = [r for r in rows if r["cv_combined"] != "NA" and r["status"] in {"keep", "SUBMIT_CANDIDATE"}]
    return max(candidates, key=lambda r: float(r["cv_combined"]))


def count_fold_improvements(candidate: list[float], baseline: list[float]) -> int:
    return sum(1 for c, b in zip(candidate, baseline) if c > b)


def load_logged_signatures(rows: list[dict[str, str]]) -> set[str]:
    signatures: set[str] = set()
    for row in rows:
        commit = row["commit"]
        if not commit or commit == "NA":
            continue
        try:
            config, _ = load_config_from_rev(commit)
        except Exception:
            continue
        signatures.add(config_signature(config))
    return signatures


def keep_status(candidate: EvalResult, best_result: EvalResult) -> tuple[str, str]:
    if candidate.time_s > 300:
        return "discard", "too_expensive"
    if candidate.cv_combined <= best_result.cv_combined:
        return "discard", "no_improvement"
    hep_improved = count_fold_improvements(candidate.fold_scores_hep, best_result.fold_scores_hep)
    dth_improved = count_fold_improvements(candidate.fold_scores_dth, best_result.fold_scores_dth)
    if hep_improved < 3 or dth_improved < 3:
        return "discard", f"fold_unstable_hep_{hep_improved}_dth_{dth_improved}"
    if candidate.n_features <= 200 and candidate.cv_combined >= best_result.cv_combined + 0.003:
        return "SUBMIT_CANDIDATE", "submit_candidate"
    return "keep", "new_best"


def save_state(state: dict[str, Any]) -> None:
    STATE_PATH.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def load_state() -> dict[str, Any]:
    if not STATE_PATH.exists():
        return {}
    return json.loads(STATE_PATH.read_text(encoding="utf-8"))


def commit_if_needed(message: str) -> str:
    subprocess.run(["git", "--no-pager", "add", "autoannitia_model.py"], cwd=REPO, check=True)
    diff = subprocess.run(
        ["git", "--no-pager", "diff", "--cached", "--quiet"],
        cwd=REPO,
        check=False,
    )
    if diff.returncode == 0:
        return git("rev-parse", "--short", "HEAD")
    subprocess.run(
        [
            "git",
            "commit",
            "-m",
            f"{message}\n\nCo-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>",
        ],
        cwd=REPO,
        text=True,
        check=True,
    )
    return git("rev-parse", "--short", "HEAD")


def set_param(config: dict[str, Any], path: tuple[str, ...], value: Any) -> dict[str, Any]:
    cfg = deepcopy(config)
    target: Any = cfg
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value
    return cfg


def candidate_plan(best_config: dict[str, Any]) -> Iterable[tuple[str, dict[str, Any]]]:
    # Original Tier 1/Tier 2/Tier 3/Tier 4/Tier 5 plan. Existing configs will be skipped.
    for w in [0.30, 0.35, 0.40, 0.45, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
        yield (f"w_xgb_{w:.2f}", set_param(best_config, ("w_xgb",), round(w, 2)))
    for seeds in ([42, 7], [42, 7, 99], [42, 0, 7, 13, 99]):
        yield (f"seeds_{'_'.join(map(str, seeds))}", set_param(best_config, ("seeds",), list(seeds)))
    for feature_key in ["clinical_binary_thresholds", "nit_timing"]:
        yield (
            f"{feature_key}_true",
            set_param(best_config, ("feature_groups", feature_key), True),
        )
    for feature_key in [
        "slope",
        "rwmean",
        "missing_flags",
        "age_robust",
        "interactions",
        "clinical_thresholds",
        "visit_patterns",
    ]:
        yield (
            f"{feature_key}_false",
            set_param(best_config, ("feature_groups", feature_key), False),
        )
    for value in [150, 200, 300, 400]:
        yield ("xgb_num_boost_round_" + str(value), set_param(best_config, ("xgb_params", "num_boost_round"), value))
    for value in [3, 5]:
        yield ("xgb_max_depth_" + str(value), set_param(best_config, ("xgb_params", "max_depth"), value))
    for value in [3, 7]:
        yield (
            "xgb_min_child_weight_" + str(value),
            set_param(best_config, ("xgb_params", "min_child_weight"), float(value)),
        )
    for value in [0.7, 0.9]:
        yield ("xgb_subsample_" + str(value), set_param(best_config, ("xgb_params", "subsample"), value))
    for value in [0.7, 0.9]:
        yield (
            "xgb_colsample_bytree_" + str(value),
            set_param(best_config, ("xgb_params", "colsample_bytree"), value),
        )

    # Follow-up micro-searches around the current best known region.
    for w in [0.52, 0.53, 0.54, 0.56, 0.58]:
        yield (f"w_xgb_{w:.2f}_micro", set_param(best_config, ("w_xgb",), w))
    for seeds in ([42, 7, 99, 1], [42, 7, 99, 13], [42, 7, 99, 21]):
        yield (f"seeds_{'_'.join(map(str, seeds))}_micro", set_param(best_config, ("seeds",), list(seeds)))
    for value in [60, 45]:
        yield ("cat_iterations_" + str(value) + "_micro", set_param(best_config, ("cat_params", "iterations"), value))
    for value in [2.5, 3.5]:
        yield (
            "xgb_min_child_weight_" + str(value).replace(".", "p") + "_micro",
            set_param(best_config, ("xgb_params", "min_child_weight"), value),
        )
    for value in [0.85, 0.95]:
        yield (
            "xgb_colsample_bytree_" + str(value).replace(".", "p") + "_micro",
            set_param(best_config, ("xgb_params", "colsample_bytree"), value),
        )


def bootstrap_state() -> dict[str, Any]:
    rows = read_results_rows()
    if not rows:
        raise RuntimeError("autoannitia_results.tsv is empty; baseline is required")

    best_row = get_best_logged_row(rows)
    best_config, best_text = load_config_from_rev(best_row["commit"])
    best_signature = config_signature(best_config)
    best_logged_result = EvalResult(
        commit=best_row["commit"],
        cv_combined=float(best_row["cv_combined"]),
        cv_hepatic=float(best_row["cv_hepatic"]),
        cv_death=float(best_row["cv_death"]),
        n_features=int(best_row["n_features"]),
        w_xgb=float(best_row["w_xgb"]),
        seeds=[int(x) for x in best_row["seeds"].split(",") if x],
        time_s=0.0,
        fold_scores_hep=[],
        fold_scores_dth=[],
    )

    state = {
        "best_commit": best_row["commit"],
        "best_description": best_row["description"],
        "best_signature": best_signature,
        "best_score": best_logged_result.cv_combined,
        "best_config": best_config,
        "best_config_text": best_text,
        "logged_signatures": list(load_logged_signatures(rows)),
    }

    write_config(best_config)
    log(f"Evaluating best config {state['best_commit']} for fold baseline")
    best_eval = run_eval()
    state["best_result"] = asdict(best_eval)

    head_short = git("rev-parse", "--short", "HEAD")
    logged_commits = {r["commit"] for r in rows}

    if head_short not in logged_commits:
        log(f"HEAD {head_short} is not logged; evaluating current config first")
        head_result = run_eval()
        head_message = sanitize_description(git("log", "-1", "--pretty=%s"))
        status, reason = keep_status(head_result, best_eval)
        append_results_row(row_from_eval(head_result, status, f"{head_message}_{reason}"))
        state["logged_signatures"].append(config_signature(load_current_config()[0]))
        if status in {"keep", "SUBMIT_CANDIDATE"}:
            state["best_commit"] = head_result.commit
            state["best_description"] = f"{head_message}_{reason}"
            state["best_signature"] = config_signature(load_current_config()[0])
            state["best_score"] = head_result.cv_combined
            state["best_config"], state["best_config_text"] = load_current_config()
            state["best_result"] = asdict(head_result)
        else:
            write_config(best_config)
            log(f"Restored working tree to best logged config {best_row['commit']}")
    save_state(state)
    return state


def next_candidate(state: dict[str, Any]) -> tuple[str, dict[str, Any]] | None:
    seen = set(state["logged_signatures"])
    best_config = state["best_config"]
    for desc, config in candidate_plan(best_config):
        if config_signature(config) in seen:
            continue
        return desc, config
    return None


def run_loop(max_iterations: int | None = None, bootstrap_only: bool = False) -> None:
    PID_PATH.write_text(str(os.getpid()), encoding="utf-8")
    state = bootstrap_state()
    if bootstrap_only:
        log("Bootstrap complete")
        return

    iterations = 0
    while True:
        candidate = next_candidate(state)
        if candidate is None:
            log("No unseen candidate configs remain; sleeping before retry")
            time.sleep(300)
            state = load_state() or state
            continue

        desc, config = candidate
        write_config(state["best_config"])
        write_config(config)
        commit_msg = desc.replace("_", " ")
        commit_hash = commit_if_needed(commit_msg)
        log(f"Running {commit_hash} {desc}")

        try:
            result = run_eval()
            best_result = EvalResult(**state["best_result"])
            status, reason = keep_status(result, best_result)
            append_results_row(row_from_eval(result, status, f"{desc}_{reason}"))
            state["logged_signatures"].append(config_signature(config))
            if status in {"keep", "SUBMIT_CANDIDATE"}:
                state["best_commit"] = commit_hash
                state["best_description"] = f"{desc}_{reason}"
                state["best_signature"] = config_signature(config)
                state["best_score"] = result.cv_combined
                state["best_config"] = config
                state["best_config_text"] = format_config(config)
                state["best_result"] = asdict(result)
                log(f"KEEP {commit_hash} score={result.cv_combined:.4f} desc={desc}")
            else:
                write_config(state["best_config"])
                log(f"DISCARD {commit_hash} score={result.cv_combined:.4f} reason={reason}")
            save_state(state)
        except Exception as exc:
            current_short = git("rev-parse", "--short", "HEAD")
            append_results_row({
                "commit": current_short,
                "cv_combined": "NA",
                "cv_hepatic": "NA",
                "cv_death": "NA",
                "n_features": "?",
                "w_xgb": f"{config['w_xgb']:.2f}",
                "seeds": ",".join(map(str, config["seeds"])),
                "status": "crash",
                "description": f"{desc}_crash",
            })
            state["logged_signatures"].append(config_signature(config))
            write_config(state["best_config"])
            save_state(state)
            log(f"CRASH {current_short} {desc}: {exc}")

        iterations += 1
        if max_iterations is not None and iterations >= max_iterations:
            log("Reached requested iteration limit")
            return


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-iterations", type=int, default=None)
    parser.add_argument("--bootstrap-only", action="store_true")
    args = parser.parse_args()
    run_loop(max_iterations=args.max_iterations, bootstrap_only=args.bootstrap_only)


if __name__ == "__main__":
    main()
