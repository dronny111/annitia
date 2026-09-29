"""Score every training patient with the split-endpoint GPT-OSS HeLM prompts, for a CV test.

Reuses the repo's serializer and prompts (scripts/annitia_helm_submit.py) with two changes that
keep the scores leak-free, matching the talk's window-free CV:

* each endpoint's text is built from visits up to that endpoint's event, as the base pipeline
  masks them (27 of 47 hepatic-event patients have later visits);
* the text drops the visit count, the age range and per-marker visit counts, which encode how
  long the patient was followed.

Zero-shot, so no training label reaches the model and every score is effectively out-of-fold.
Writes ../data/helm_train_scores.csv, or helm_train_scores_joint.csv with --joint (git-ignored),
and resumes where it stopped.

    python talk/helm_train_scores.py --n 3            # smoke test
    python talk/helm_train_scores.py --workers 4      # split: one prompt per endpoint
    python talk/helm_train_scores.py --joint          # joint: one prompt for both endpoints
    OPENROUTER_API_KEY=... python talk/helm_train_scores.py --provider openrouter \
        --model google/gemma-4-31b-it:free --think --joint --workers 2
"""
import argparse
import csv
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
os.chdir(ROOT)
import annitia_helm_submit as H  # noqa: E402

MODEL = "gpt-oss:120b-cloud"
EVENT_AGE = {"hep": "evenements_hepatiques_age_occur", "death": "death_age_occur"}
PROMPT = {"hep": H.SCORE_PROMPT_HEPATIC, "death": H.SCORE_PROMPT_DEATH}
KEY = {"hep": "hepatic_risk", "death": "death_risk"}


def masked(row: pd.Series, event_age_col: str) -> pd.Series:
    """Blank every visit recorded after the event, as feature_revolution.mask_post_event_wide does."""
    event_age = row.get(event_age_col)
    if pd.isna(event_age):
        return row
    row = row.copy()
    for v in H.VISIT_NUMS:
        age = row.get(f"Age_v{v}")
        if pd.notna(age) and age > event_age:
            row[[c for c in row.index if c.endswith(f"_v{v}")]] = np.nan
    return row


def window_free(text: str) -> str:
    """Remove follow-up length from the serialized text: visit counts and the age range."""
    text = re.sub(r"age ([\d.]+)–[\d.]+ yr, \d+ clinic visits\.", r"age \1 yr at first visit.", text)
    text = re.sub(r"age ([\d.]+) yr, \d+ clinic visits\.", r"age \1 yr at first visit.", text)
    return re.sub(r"n=\d+, ", "", text)


def patient_text(row: pd.Series, ep: str) -> str:
    """ep "joint" masks at the earliest event, so neither endpoint sees post-event visits."""
    if ep == "joint":
        ages = [row[c] for c in EVENT_AGE.values() if pd.notna(row.get(c))]
        r = row if not ages else masked(pd.concat([row, pd.Series({"_first_event": min(ages)})]), "_first_event")
    else:
        r = masked(row, EVENT_AGE[ep])
    text = window_free(H.serialize_patient(r, H._fib4(r)))
    # ponytail: pinned to the label the first (split) run used, so split and joint differ only in
    # prompt structure. The repo's serializer now says liver stiffness, per the data dictionary.
    return text.replace("Aixplorer liver stiffness", "AIxplorer arterial stiffness")


class RateLimited(Exception):
    """The provider keeps refusing calls: stop cleanly and resume later, rather than save failures."""


STOP = threading.Event()


def ask(prompt, args, client):
    """One model call, returning the parsed risk scores. Retries transient errors with backoff."""
    keys = H._detect_keys(prompt)
    for attempt in range(6):
        if STOP.is_set():
            raise RateLimited()
        try:
            if args.provider == "ollama":            # the repo's own call path (thinking off)
                return H._call_model(prompt, args.model, "ollama", client, 1, 1.0)
            body = {"model": args.model, "temperature": 0, "response_format": {"type": "json_object"},
                    "messages": [{"role": "system", "content": H.SYSTEM_PROMPT},
                                 {"role": "user", "content": prompt}]}
            if args.think:
                body["reasoning"] = {"enabled": True}
            r = requests.post("https://openrouter.ai/api/v1/chat/completions", json=body, timeout=300,
                              headers={"Authorization": f"Bearer {os.environ['OPENROUTER_API_KEY']}"})
            if r.status_code == 429:
                raise RateLimited(r.text[:200])
            r.raise_for_status()
            return H._parse_json(r.json()["choices"][0]["message"].get("content") or "", keys)
        except RateLimited:
            if attempt == 5:
                STOP.set()
                raise
            time.sleep(min(300, 15 * 2 ** attempt))   # free pools refill slowly
        except Exception as exc:
            if attempt == 5:
                print(f"  [WARN] {exc}", flush=True)
                return {k: float("nan") for k in keys}
            time.sleep(5 * (attempt + 1))


def job(row, args, client):
    """(patient, hepatic score, death score), or None if the provider stopped the run."""
    try:
        if args.joint:
            prompt = H.SCORE_PROMPT_COMBINED.format(preamble="", patient_data=patient_text(row, "joint"),
                                                    cot_instruction="")
            r = ask(prompt, args, client)
            return row["patient_id_anon"], r.get("hepatic_risk"), r.get("death_risk")
        risk = []
        for ep in ("hep", "death"):
            prompt = PROMPT[ep].format(preamble="", patient_data=patient_text(row, ep), cot_instruction="")
            risk.append(ask(prompt, args, client).get(KEY[ep]))
        return row["patient_id_anon"], *risk
    except RateLimited:
        return None


def out_path(args):
    """The first two runs keep their names; other models get one file per model, mode and thinking."""
    mode = "joint" if args.joint else "split"
    if args.provider == "ollama" and args.model == MODEL and not args.think:
        return ROOT / "data" / ("helm_train_scores_joint.csv" if args.joint else "helm_train_scores.csv")
    slug = re.sub(r"[^a-z0-9]+", "-", args.model.lower()).strip("-")
    return ROOT / "data" / f"helm_train_scores_{mode}_{slug}{'_think' if args.think else ''}.csv"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None, help="score only the first N patients")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--joint", action="store_true", help="one combined prompt for both endpoints")
    ap.add_argument("--provider", choices=["ollama", "openrouter"], default="ollama")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--think", action="store_true", help="reasoning on (OpenRouter only)")
    args = ap.parse_args()
    if args.think and args.provider != "openrouter":
        raise SystemExit("--think needs --provider openrouter; the repo's Ollama path runs thinking off")
    OUT = out_path(args)

    train = pd.read_csv(H.TRAIN_PATH)
    if args.n:
        train = train.head(args.n)
    done = set()
    if OUT.is_file():                             # drop failed calls so they are scored again
        prev = pd.read_csv(OUT).dropna()
        prev.to_csv(OUT, index=False)
        done = set(prev["patient_id_anon"])
    todo = train[~train["patient_id_anon"].isin(done)]
    print(f"{OUT.name}: {len(done)} already scored, {len(todo)} to go, {args.workers} workers", flush=True)
    client = H._init_client("ollama") if args.provider == "ollama" else None

    new = not OUT.is_file()
    with OUT.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["patient_id_anon", "helm_hep", "helm_death"])
        t0, n_done = time.time(), 0
        with ThreadPoolExecutor(args.workers) as pool:
            futures = [pool.submit(job, row, args, client) for _, row in todo.iterrows()]
            for fut in as_completed(futures):
                res = fut.result()
                if res is None:
                    continue
                w.writerow(res); f.flush(); n_done += 1
                if n_done % 25 == 0:
                    rate = n_done / (time.time() - t0)
                    print(f"  {n_done}/{len(futures)} | {rate * 60:.1f} patients/min"
                          f" | eta {(len(futures) - n_done) / rate / 60:.0f} min", flush=True)
    if STOP.is_set():
        print(f"STOPPED: provider rate limit persisted. {n_done} scored this run; rerun to resume.")
    s = pd.read_csv(OUT)
    print(f"scored {len(s)} patients | missing hep {s['helm_hep'].isna().sum()}"
          f" death {s['helm_death'].isna().sum()} | unique hep {s['helm_hep'].nunique()}")


if __name__ == "__main__":
    main()
