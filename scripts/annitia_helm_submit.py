"""
annitia_helm_submit.py — HeLM Zero-Shot / Few-Shot Risk Scoring for Test Set
=============================================================================
Generates submission-ready risk scores by serializing each test patient's
clinical features and querying an LLM to predict hepatic event and death risk.

This is Stage 1 of the HeLM adaptation (Belyaeva et al. 2023):
  - Features serialized as structured text (analogous to HeLM tabular encoder path)
  - LLM outputs continuous risk scores 0–100 (used as monotonic ranking scores)
  - Scores combined into submission CSV

Ablation flags:
  --model         Model ID (default: claude-haiku-4-5-20251001)
                  Examples: claude-sonnet-4-6, gpt-4.1-mini, gemini-2.5-pro
  --provider      API provider: auto | anthropic | openai | google | ollama
  --few-shot      Prepend 5 labeled training examples to each prompt (A2)
  --split-endpoints  Separate prompts for hepatic vs. death endpoints (A3)
  --n-samples N   Average N API calls per patient at temperature=0.7 (A4)
  --cot           Chain-of-thought reasoning before JSON output (A5)

Requires one provider key:
           ANTHROPIC_API_KEY for Claude
           OPENAI_API_KEY for GPT
           GEMINI_API_KEY or GOOGLE_API_KEY for Gemini
           pip install anthropic openai google-genai

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python annitia_helm_submit.py                                  # Haiku zero-shot
    python annitia_helm_submit.py --model claude-sonnet-4-6        # A1: Sonnet
    python annitia_helm_submit.py --model claude-sonnet-4-6 --few-shot  # A1+A2
    python annitia_helm_submit.py --model claude-sonnet-4-6 --few-shot --split-endpoints  # A1+A2+A3
    python annitia_helm_submit.py --model gpt-4.1-mini             # OpenAI
    python annitia_helm_submit.py --model gemini-2.5-pro           # Gemini
    python annitia_helm_submit.py --cot                            # A5: CoT
    python annitia_helm_submit.py --n-samples 3                    # A4: temp ensemble
    python annitia_helm_submit.py --n 20 --dry-run                 # test first 20, no API
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ── Paths ─────────────────────────────────────────────────────────────────────
TRAIN_PATH = Path("data/DB-1773398340961.csv")
TEST_PATH  = Path("data/DB-1773398340961 (1).csv")
OOF_PATH   = Path("oof_predictions.csv")

VISIT_NUMS = list(range(1, 23))
DEFAULT_MODEL = "claude-haiku-4-5-20251001"   # fast + cheap; use --model for sonnet
DEFAULT_PROVIDER = "auto"


# ── Feature extraction ────────────────────────────────────────────────────────

def _traj(row: pd.Series, prefix: str) -> dict:
    vals = pd.Series(
        [row.get(f"{prefix}_v{n}") for n in VISIT_NUMS], dtype=float
    ).dropna()
    if vals.empty:
        return {"n": 0, "first": None, "last": None, "max": None, "slope": None}
    slope = None
    if len(vals) >= 2:
        xs = np.arange(len(vals), dtype=float)
        try:
            slope = float(np.polyfit(xs, vals.values, 1)[0])
        except Exception:
            pass
    return {
        "n":     int(len(vals)),
        "first": round(float(vals.iloc[0]), 2),
        "last":  round(float(vals.iloc[-1]), 2),
        "max":   round(float(vals.max()), 2),
        "slope": round(slope, 4) if slope is not None else None,
    }


def _fib4(row: pd.Series) -> float | None:
    for n in reversed(VISIT_NUMS):
        age = row.get(f"Age_v{n}"); ast = row.get(f"ast_v{n}")
        plt = row.get(f"plt_v{n}"); alt = row.get(f"alt_v{n}")
        if all(v is not None and not pd.isna(v) for v in [age, ast, plt, alt]):
            if float(plt) > 0 and float(alt) > 0:
                return round(float(age) * float(ast) / (float(plt) * float(alt) ** 0.5), 2)
    return None


def _fmt(t: dict, u: str = "") -> str:
    if t["n"] == 0:
        return "missing"
    parts = [f"n={t['n']}"]
    if t["first"] is not None: parts.append(f"first={t['first']}{u}")
    if t["last"]  is not None: parts.append(f"last={t['last']}{u}")
    if t["slope"] is not None:
        d = "rising" if t["slope"] > 0.05 else ("falling" if t["slope"] < -0.05 else "stable")
        parts.append(d)
    return ", ".join(parts)


def serialize_patient(row: pd.Series, fib4: float | None) -> str:
    """Build a compact clinical text for LLM scoring."""
    g     = "female" if row.get("gender", 0) == 0 else "male"
    t2dm  = bool(row.get("T2DM", 0))
    htn   = bool(row.get("Hypertension", 0))
    dysl  = bool(row.get("Dyslipidaemia", 0))
    bari  = bool(row.get("bariatric_surgery", 0))

    age  = _traj(row, "Age")
    bmi  = _traj(row, "BMI")
    fs   = _traj(row, "fibs_stiffness_med_BM_1")
    ft   = _traj(row, "fibrotest_BM_2")
    aix  = _traj(row, "aixp_aix_result_BM_3")
    alt  = _traj(row, "alt")
    ast  = _traj(row, "ast")
    ggt  = _traj(row, "ggt")
    plt_ = _traj(row, "plt")
    bili = _traj(row, "bilirubin")
    gluc = _traj(row, "gluc_fast")
    trig = _traj(row, "triglyc")

    age_str = (f"{age['first']}–{age['last']} yr" if age["n"] > 1
               else f"{age['first']} yr" if age["n"] == 1 else "unknown")
    comorbidities = [x for x in (
        "T2DM" if t2dm else "",
        "hypertension" if htn else "",
        "dyslipidaemia" if dysl else "",
        "bariatric surgery" if bari else "",
    ) if x]

    lines = [
        f"MASLD patient. {g.capitalize()}, age {age_str}, {age['n']} clinic visits.",
        f"Comorbidities: {', '.join(comorbidities) if comorbidities else 'none'}.",
        f"BMI: {_fmt(bmi, ' kg/m2')}.",
        f"",
        f"Non-invasive tests (NITs — liver fibrosis staging):",
        f"  FibroScan liver stiffness (kPa; F2=7.0, F3=9.5, F4/cirrhosis=12.5): {_fmt(fs, ' kPa')}",
        f"  FibroTest composite score (0–1; F4 >0.74): {_fmt(ft)}",
        f"  AIxplorer arterial stiffness: {_fmt(aix)}",
        f"  FIB-4 index (low <1.30, high >2.67): {fib4 if fib4 else 'unavailable'}",
        f"",
        f"Liver biochemistry:",
        f"  ALT (U/L; ULN 40): {_fmt(alt, ' U/L')}",
        f"  AST (U/L; ULN 40): {_fmt(ast, ' U/L')}",
        f"  GGT (U/L): {_fmt(ggt, ' U/L')}",
        f"  Bilirubin (umol/L): {_fmt(bili)}",
        f"  Platelets (G/L; thrombocytopenia <150): {_fmt(plt_, ' G/L')}",
        f"",
        f"Metabolic markers:",
        f"  Fasting glucose (mmol/L): {_fmt(gluc)}",
        f"  Triglycerides (mmol/L): {_fmt(trig)}",
    ]
    return "\n".join(lines)


# ── Prompt templates ──────────────────────────────────────────────────────────

# Base zero-shot prompt (combined endpoints)
SCORE_PROMPT_COMBINED = """{preamble}{patient_data}

---

You are a hepatologist assessing risk for this MASLD patient.
Provide two continuous risk scores on a 0–100 scale:
- hepatic_risk: probability of a major hepatic event (ascites, encephalopathy, variceal bleed, HCC, liver transplant) occurring during follow-up
- death_risk: probability of all-cause mortality during follow-up

Higher scores = higher risk. Base scores on the NIT values, liver biochemistry, and metabolic burden shown above.
Key clinical signals: FibroScan >12.5 kPa = cirrhosis (very high risk); FIB-4 >2.67 = advanced fibrosis; platelet decline = portal hypertension; persistently elevated ALT/GGT = ongoing injury.
Missing NIT data (e.g. no FibroScan) is itself a risk signal for sparse follow-up.
{cot_instruction}
Return only valid JSON — no other text:
{{"hepatic_risk": <float 0-100>, "death_risk": <float 0-100>}}"""

# A3: endpoint-specific prompts
SCORE_PROMPT_HEPATIC = """{preamble}{patient_data}

---

You are a hepatologist assessing liver-specific risk for this MASLD patient.
Provide a single continuous hepatic_risk score on a 0–100 scale.

This score reflects the probability of a major hepatic event: ascites, hepatic encephalopathy, variceal bleed, HCC, or liver transplant.

Key clinical signals for hepatic risk:
- FibroScan >12.5 kPa = cirrhosis (very high risk); >9.5 kPa = F3 (high risk); >7.0 kPa = F2
- FIB-4 >2.67 = advanced fibrosis; FIB-4 1.30–2.67 = intermediate
- FibroTest >0.74 = F4; >0.58 = F3
- Platelet decline below 150 G/L = portal hypertension sign
- Rising FibroScan slope = active fibrosis progression
- Persistently elevated ALT/GGT = ongoing hepatic injury
- T2DM accelerates fibrosis progression in MASLD
- Missing NITs (sparse follow-up) is a risk signal
{cot_instruction}
Return only valid JSON — no other text:
{{"hepatic_risk": <float 0-100>}}"""

SCORE_PROMPT_DEATH = """{preamble}{patient_data}

---

You are a clinician assessing all-cause mortality risk for this MASLD patient.
Provide a single continuous death_risk score on a 0–100 scale.

This score reflects the probability of all-cause mortality during follow-up (any cause).

Key clinical signals for all-cause mortality:
- Advanced age is the dominant driver
- T2DM + hypertension + dyslipidaemia = high cardiovascular risk
- Bariatric surgery history = prior severe obesity
- Elevated fasting glucose and triglycerides = metabolic syndrome
- High BMI trajectory
- Liver cirrhosis (FibroScan >12.5, low platelets) = hepatic mortality contribution
- Sparse follow-up (few visits) = possible frailty or rapid deterioration
{cot_instruction}
Return only valid JSON — no other text:
{{"death_risk": <float 0-100>}}"""

COT_INSTRUCTION_COMBINED = """
First, reason briefly through the key risk signals in <reasoning>...</reasoning> tags.
Then output the JSON on its own line after </reasoning>.
"""

COT_INSTRUCTION_SINGLE = """
First, reason briefly through the key risk signals in <reasoning>...</reasoning> tags.
Then output the JSON on its own line after </reasoning>.
"""


# ── Few-shot example builder (A2) ─────────────────────────────────────────────

def build_few_shot_examples(train: pd.DataFrame, oof: pd.DataFrame, n_examples: int = 5) -> str:
    """
    Select n_examples representative training patients across risk strata
    and format them as labeled examples for few-shot prompting.

    Selects: lowest, low, median, high, highest OOF hepatic risk.
    Includes known outcomes to anchor the LLM's calibration.
    """
    merged = train.merge(oof[["patient_id_anon", "oof_hepatic", "oof_death"]],
                         on="patient_id_anon", how="inner")
    if merged.empty:
        return ""

    merged = merged.sort_values("oof_hepatic").reset_index(drop=True)
    n = len(merged)
    # Pick indices spread across the distribution
    indices = [int(n * f) for f in [0.05, 0.25, 0.50, 0.75, 0.95]]
    indices = [min(max(i, 0), n - 1) for i in indices]
    examples = merged.iloc[indices[:n_examples]]

    lines = ["Below are example MASLD patients from the training cohort with known outcomes. "
             "Use these to calibrate your risk scores for the new patient.\n"]

    for rank_i, (_, ex) in enumerate(examples.iterrows(), 1):
        fib4 = _fib4(ex)
        text = serialize_patient(ex, fib4)
        hep_score = round(float(ex["oof_hepatic"]) * 100, 1)
        dth_score = round(float(ex["oof_death"]) * 100, 1)
        outcome_h = bool(ex.get("evenements_hepatiques_majeurs", 0))
        outcome_d = bool(ex.get("death", 0))
        lines.append(f"[Example {rank_i}]")
        lines.append(text)
        lines.append(f'Known outcome: hepatic_event={outcome_h}, death={outcome_d}')
        lines.append(f'Answer: {{"hepatic_risk": {hep_score}, "death_risk": {dth_score}}}')
        lines.append("")

    lines.append("[New patient to score]")
    return "\n".join(lines)


# ── OOF loader ────────────────────────────────────────────────────────────────

def load_oof() -> pd.DataFrame | None:
    """
    Load OOF predictions and return a DataFrame with columns:
      patient_id_anon, oof_hepatic [0-1], oof_death [0-1]

    Handles the pipeline's long format: columns endpoint, fold, patient_idx,
    xgb_pred, cat_pred. Also handles simpler wide formats with explicit oof_* cols.
    """
    if not OOF_PATH.exists():
        return None
    oof = pd.read_csv(OOF_PATH)

    # ── Long format (pipeline output) ────────────────────────────────────────
    if {"endpoint", "fold", "xgb_pred", "cat_pred"}.issubset(oof.columns):
        id_col = "patient_idx" if "patient_idx" in oof.columns else "patient_id_anon"
        # Blend xgb + cat (50/50) as risk proxy; AFT scores: more negative = sooner event = higher risk
        oof["blend"] = 0.5 * oof["xgb_pred"] + 0.5 * oof["cat_pred"]
        # Average across folds per patient × endpoint
        grp = oof.groupby([id_col, "endpoint"])["blend"].mean().reset_index()
        hep = grp[grp["endpoint"] == "hepatic"].rename(columns={"blend": "oof_hepatic"})
        dth = grp[grp["endpoint"] == "death"].rename(columns={"blend": "oof_death"})
        merged = hep[[id_col, "oof_hepatic"]].merge(dth[[id_col, "oof_death"]], on=id_col)

        # If patient_idx, map back to patient_id_anon via training data row order
        if id_col == "patient_idx":
            train = pd.read_csv(TRAIN_PATH)
            train["patient_idx"] = range(len(train))
            merged = merged.merge(train[["patient_idx", "patient_id_anon"]], on="patient_idx")

        # Min-max normalize to [0, 1]
        for col in ["oof_hepatic", "oof_death"]:
            lo, hi = merged[col].min(), merged[col].max()
            merged[col] = (merged[col] - lo) / (hi - lo + 1e-9)

        return merged[["patient_id_anon", "oof_hepatic", "oof_death"]]

    # ── Wide format fallback ──────────────────────────────────────────────────
    col_map = {}
    for c in oof.columns:
        lc = c.lower()
        if "hepatic" in lc and "oof" in lc:
            col_map[c] = "oof_hepatic"
        elif "death" in lc and "oof" in lc:
            col_map[c] = "oof_death"
        elif "patient" in lc and "id" in lc:
            col_map[c] = "patient_id_anon"
    oof = oof.rename(columns=col_map)
    required = {"patient_id_anon", "oof_hepatic", "oof_death"}
    if not required.issubset(oof.columns):
        print(f"  [WARN] OOF file missing columns. Found: {list(oof.columns)}")
        return None
    for col in ["oof_hepatic", "oof_death"]:
        if oof[col].max() > 1.5:
            oof[col] = (oof[col] - oof[col].min()) / (oof[col].max() - oof[col].min() + 1e-9)
    return oof[["patient_id_anon", "oof_hepatic", "oof_death"]]


# ── JSON parsing ──────────────────────────────────────────────────────────────

def _parse_json(raw: str, keys: list[str]) -> dict[str, float]:
    """Extract JSON from model response, handling CoT reasoning tags."""
    # Strip reasoning block if present
    raw = re.sub(r"<reasoning>.*?</reasoning>", "", raw, flags=re.DOTALL).strip()
    m = re.search(r"\{[^}]+\}", raw)
    if m:
        try:
            data = json.loads(m.group())
            result = {}
            for key in keys:
                val = data.get(key) or data.get(key.replace("_risk", "")) or data.get(key.split("_")[0])
                result[key] = float(np.clip(float(val), 0, 100)) if val is not None else 50.0
            return result
        except Exception:
            pass
    # Fallback: extract numbers directly via regex (handles truncated/malformed JSON)
    result = {}
    for key in keys:
        pat = re.search(rf'"{key}"\s*:\s*([0-9]+(?:\.[0-9]+)?)', raw)
        if pat:
            result[key] = float(np.clip(float(pat.group(1)), 0, 100))
        else:
            raise ValueError(f"No JSON in response: {raw!r}")
    return result


# ── API scoring ───────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a clinical risk scorer. You output ONLY valid JSON — no prose, "
    "no reasoning, no explanation, no markdown. Your entire response must be "
    "a single JSON object matching the exact format requested."
)


def _detect_keys(prompt: str) -> list[str]:
    if "hepatic_risk" in prompt and "death_risk" in prompt:
        return ["hepatic_risk", "death_risk"]
    if "hepatic_risk" in prompt:
        return ["hepatic_risk"]
    return ["death_risk"]


def _infer_provider(model: str, provider: str) -> str:
    if provider != "auto":
        return provider
    model_l = model.lower()
    if model_l.startswith("claude"):
        return "anthropic"
    if model_l.startswith("gemini"):
        return "google"
    # Ollama-served local models (check before "gpt" so gpt-oss:* routes to ollama)
    if (
        model_l.startswith(("gpt-oss", "medgemma", "llama", "qwen", "mistral", "phi"))
        or ":" in model_l  # ollama tag syntax e.g. "model:120b"
    ):
        return "ollama"
    if model_l.startswith(("gpt", "o1", "o3", "o4")):
        return "openai"
    raise ValueError(
        f"Could not infer provider from model '{model}'. "
        "Pass --provider anthropic|openai|google|ollama."
    )


def _model_tag(provider: str, model: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", model.lower()).strip("-")
    if provider == "anthropic" and model.lower().startswith("claude-"):
        parts = [p for p in slug.split("-") if p]
        if len(parts) >= 2:
            return parts[1]
    if provider == "ollama":
        return f"oll-{slug}"
    return slug


def _provider_key(provider: str) -> tuple[str, str]:
    if provider == "anthropic":
        key = os.environ.get("ANTHROPIC_API_KEY", "")
        return key, "ANTHROPIC_API_KEY"
    if provider == "openai":
        key = os.environ.get("OPENAI_API_KEY", "")
        return key, "OPENAI_API_KEY"
    if provider == "google":
        key = os.environ.get("GEMINI_API_KEY", "") or os.environ.get("GOOGLE_API_KEY", "")
        return key, "GEMINI_API_KEY or GOOGLE_API_KEY"
    if provider == "ollama":
        # No API key needed; return a non-empty placeholder so _init_client passes.
        return "ollama-local", "OLLAMA_URL"
    raise ValueError(f"Unsupported provider: {provider}")


def _init_client(provider: str):
    key, env_name = _provider_key(provider)
    if not key:
        raise SystemExit(
            f"\n{env_name} not set.\n"
            f"Run: export {env_name.split(' or ')[0]}=<api-key>\n"
            "Then re-run this script."
        )
    if provider == "anthropic":
        import anthropic
        return anthropic.Anthropic(api_key=key)
    if provider == "openai":
        from openai import OpenAI
        return OpenAI(api_key=key)
    if provider == "google":
        from google import genai
        return genai.Client(api_key=key)
    if provider == "ollama":
        url = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434")
        # Probe reachability; fail fast with an actionable message.
        import requests  # local import keeps non-ollama paths clean
        try:
            requests.get(f"{url.rstrip('/')}/api/tags", timeout=5).raise_for_status()
        except Exception as exc:
            raise SystemExit(
                f"\nOllama not reachable at {url}: {exc}\n"
                "Install Ollama, run `ollama serve`, then `ollama pull <model>`."
            )
        return {"url": url.rstrip("/")}
    raise ValueError(f"Unsupported provider: {provider}")


def _extract_openai_text(resp) -> str:
    choice = resp.choices[0]
    msg = getattr(choice, "message", None)
    if msg is None:
        raise ValueError("OpenAI response missing message")
    content = getattr(msg, "content", None)
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        texts = []
        for part in content:
            text = getattr(part, "text", None)
            if text:
                texts.append(text)
            elif isinstance(part, dict) and part.get("type") == "text":
                texts.append(part.get("text", ""))
        if texts:
            return "\n".join(texts).strip()
    raise ValueError("OpenAI response missing text content")


def _extract_google_text(resp) -> str:
    text = getattr(resp, "text", None)
    if text:
        return text.strip()
    candidates = getattr(resp, "candidates", None) or []
    for cand in candidates:
        content = getattr(cand, "content", None)
        parts = getattr(content, "parts", None) or []
        texts = [getattr(part, "text", None) for part in parts if getattr(part, "text", None)]
        if texts:
            return "\n".join(texts).strip()
    raise ValueError("Gemini response missing text content")


def _call_model_once(prompt: str, model: str, provider: str, client, temperature: float | None) -> str:
    if provider == "anthropic":
        kwargs: dict = dict(
            model=model,
            max_tokens=64,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": prompt}],
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        msg = client.messages.create(**kwargs)
        return msg.content[0].text.strip()

    if provider == "openai":
        kwargs = dict(
            model=model,
            max_completion_tokens=64,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        if temperature is not None:
            kwargs["temperature"] = temperature
        resp = client.chat.completions.create(**kwargs)
        return _extract_openai_text(resp)

    if provider == "google":
        cfg: dict = {
            "system_instruction": SYSTEM_PROMPT,
            "max_output_tokens": 64,
        }
        if temperature is not None:
            cfg["temperature"] = temperature
        resp = client.models.generate_content(
            model=model,
            contents=prompt,
            config=cfg,
        )
        return _extract_google_text(resp)

    if provider == "ollama":
        import requests  # local import
        url = f"{client['url']}/api/generate"
        # CoT prompts contain free-form <reasoning>...</reasoning> text before the JSON.
        # Ollama's `format=json` mode forces the entire response to be JSON and would
        # clobber CoT reasoning, so we only apply it for non-CoT prompts.
        is_cot = "<reasoning>" in prompt
        # gpt-oss and similar "thinking" models spend most tokens on hidden reasoning
        # (returned separately in the 'thinking' field) before emitting the JSON.
        # For long prompts (few-shot) the default budget is fully consumed by
        # reasoning and the `response` comes back empty — hence the generous
        # headroom and `reasoning=low` for JSON-mode calls.
        num_predict = 2048 if is_cot else 4096
        options: dict = {"num_predict": num_predict}
        if temperature is not None:
            options["temperature"] = float(temperature)
        else:
            options["temperature"] = 0.0
        if not is_cot:
            # Cap hidden chain-of-thought so the token budget is spent on the JSON payload.
            options["reasoning"] = "low"
        body: dict = {
            "model": model,
            "prompt": prompt,
            "system": SYSTEM_PROMPT,
            "stream": False,
            "options": options,
        }
        if not is_cot:
            body["format"] = "json"
            body["think"] = False
        r = requests.post(url, json=body, timeout=300)
        r.raise_for_status()
        data = r.json()
        text = (data.get("response") or "").strip()
        # Do NOT fall back to `thinking`: for reasoning models, thinking is hidden
        # prose (not JSON) and "parsing" it just produces misleading warnings.
        # An empty response means the token budget was exhausted — surface that.
        if not text:
            raise ValueError(
                f"Empty response (done_reason={data.get('done_reason')!r}, "
                f"thinking_len={len(data.get('thinking') or '')})"
            )
        return text

    raise ValueError(f"Unsupported provider: {provider}")


def _call_model(prompt: str, model: str, provider: str, client, n_samples: int,
                temperature: float) -> dict[str, float]:
    """Call an LLM provider, optionally averaging n_samples calls."""
    keys = _detect_keys(prompt)

    results: list[dict] = []
    for _ in range(n_samples):
        raw = _call_model_once(
            prompt=prompt,
            model=model,
            provider=provider,
            client=client,
            temperature=temperature if n_samples > 1 else None,
        )
        results.append(_parse_json(raw, keys))

    # Average across samples
    averaged: dict[str, float] = {}
    for k in results[0]:
        averaged[k] = float(np.mean([r[k] for r in results]))
    return averaged


def score_patient_combined(patient_text: str, preamble: str, model: str, provider: str, client,
                           n_samples: int, temperature: float, cot: bool) -> tuple[float, float]:
    """Single combined prompt for both endpoints."""
    cot_instr = COT_INSTRUCTION_COMBINED if cot else ""
    prompt = SCORE_PROMPT_COMBINED.format(
        preamble=preamble, patient_data=patient_text, cot_instruction=cot_instr
    )
    result = _call_model(prompt, model, provider, client, n_samples, temperature)
    return result.get("hepatic_risk", 50.0), result.get("death_risk", 50.0)


def score_patient_split(patient_text: str, preamble: str, model: str, provider: str, client,
                        n_samples: int, temperature: float, cot: bool,
                        inter_call_delay: float = 0.5) -> tuple[float, float]:
    """Separate prompts for hepatic and death endpoints (A3)."""
    cot_instr = COT_INSTRUCTION_SINGLE if cot else ""
    prompt_h = SCORE_PROMPT_HEPATIC.format(
        preamble=preamble, patient_data=patient_text, cot_instruction=cot_instr
    )
    prompt_d = SCORE_PROMPT_DEATH.format(
        preamble=preamble, patient_data=patient_text, cot_instruction=cot_instr
    )
    res_h = _call_model(prompt_h, model, provider, client, n_samples, temperature)
    time.sleep(inter_call_delay)  # gap between the two endpoint calls
    res_d = _call_model(prompt_d, model, provider, client, n_samples, temperature)
    return res_h.get("hepatic_risk", 50.0), res_d.get("death_risk", 50.0)


# ── Main ──────────────────────────────────────────────────────────────────────

def main(n: int | None, model: str, provider: str, dry_run: bool, delay: float,
         few_shot: bool, split_endpoints: bool, n_samples: int,
         cot: bool) -> Path:

    provider = _infer_provider(model, provider)

    # Google free tier: 15 RPM → need ≥4s between calls; auto-bump if user left default
    if provider == "google" and delay < 4.0:
        delay = 4.5
        print(f"[INFO] Google provider detected — auto-setting delay={delay}s (free tier ≤15 RPM)")
    # Local Ollama has no rate limit; zero out delay unless user explicitly raised it.
    if provider == "ollama" and delay <= 0.1:
        delay = 0.0

    if not dry_run:
        client = _init_client(provider)
    else:
        client = None

    print("Loading data...", flush=True)
    test  = pd.read_csv(TEST_PATH)
    train = pd.read_csv(TRAIN_PATH)

    if n is not None:
        test = test.head(n)

    # Build few-shot preamble (A2)
    preamble = ""
    if few_shot:
        oof = load_oof()
        if oof is not None:
            preamble = build_few_shot_examples(train, oof) + "\n"
            print(f"  Few-shot: loaded {preamble.count('[Example')} examples from OOF")
        else:
            print("  [WARN] --few-shot requested but oof_predictions.csv not found; skipping")

    # Describe configuration
    ablations = []
    model_short = _model_tag(provider, model)
    if few_shot:             ablations.append("fewshot")
    if split_endpoints:      ablations.append("split")
    if n_samples > 1:        ablations.append(f"t{n_samples}x")
    if cot:                  ablations.append("cot")
    if n is not None:        ablations.append(f"n{n}")
    if dry_run:              ablations.append("dryrun")
    tag_parts = [model_short] + ablations
    tag = "_".join(tag_parts)

    total = len(test)
    print(f"Scoring {total} test patients | provider={provider} | model={model} | "
          f"ablations={ablations or ['zero-shot']} | n_samples={n_samples} | "
          f"split_endpoints={split_endpoints} | dry_run={dry_run}",
          flush=True)

    trustii_ids, hep_scores, dth_scores = [], [], []
    errors = 0

    for i, (_, row) in enumerate(test.iterrows()):
        tid  = int(row["trustii_id"])
        fib4 = _fib4(row)
        text = serialize_patient(row, fib4)

        if dry_run:
            rng = np.random.default_rng(seed=tid % 10000)
            h = float(40 + rng.uniform(0, 40))
            d = float(30 + rng.uniform(0, 40))
        else:
            try:
                if split_endpoints:
                    # inter_call_delay = gap between hepatic/death calls within a patient
                    if provider == "google":
                        inter = delay
                    elif provider == "ollama":
                        inter = 0.0
                    else:
                        inter = 0.5
                    h, d = score_patient_split(
                        text, preamble, model, provider, client, n_samples,
                        temperature=0.7 if n_samples > 1 else 1.0, cot=cot,
                        inter_call_delay=inter,
                    )
                else:
                    h, d = score_patient_combined(
                        text, preamble, model, provider, client, n_samples,
                        temperature=0.7 if n_samples > 1 else 1.0, cot=cot
                    )
                if delay > 0:
                    time.sleep(delay)
            except Exception as e:
                print(f"  [WARN] patient {tid} failed: {e}", flush=True)
                h, d = 50.0, 50.0
                errors += 1

        trustii_ids.append(tid)
        hep_scores.append(h)
        dth_scores.append(d)

        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  {i+1}/{total} done | last hep={h:.1f} death={d:.1f}", flush=True)

    sub = pd.DataFrame({
        "trustii_id":        trustii_ids,
        "risk_hepatic_event": hep_scores,
        "risk_death":         dth_scores,
    }).sort_values("trustii_id").reset_index(drop=True)

    out_path = Path(f"submission_helm_{tag}.csv")
    sub.to_csv(out_path, index=False)
    print(f"\nSaved {len(sub)} rows → {out_path}")
    if errors:
        print(f"WARNING: {errors} patients used fallback score 50.0 due to API errors")

    hep_std = sub["risk_hepatic_event"].std()
    dth_std = sub["risk_death"].std()
    print(f"\nHep score stats: mean={sub['risk_hepatic_event'].mean():.1f}  std={hep_std:.2f}")
    print(f"Death score stats: mean={sub['risk_death'].mean():.1f}  std={dth_std:.2f}")
    if hep_std < 5:
        print("  [WARN] Low hepatic score std — LLM may be clustering scores; check prompt calibration")
    if dth_std < 5:
        print("  [WARN] Low death score std — LLM may be clustering scores; check prompt calibration")

    return out_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n",               type=int,   default=None,
                        help="Limit to first N patients")
    parser.add_argument("--model",           type=str,   default=DEFAULT_MODEL,
                        help="Model ID (e.g. claude-sonnet-4-6, gpt-4.1-mini, gemini-2.5-pro)")
    parser.add_argument("--provider",        type=str,   default=DEFAULT_PROVIDER,
                        choices=["auto", "anthropic", "openai", "google", "ollama"],
                        help="LLM provider; default auto-infers from --model")
    parser.add_argument("--dry-run",         action="store_true",
                        help="Skip API, use deterministic pseudo-random scores")
    parser.add_argument("--delay",           type=float, default=0.1,
                        help="Seconds between API calls")
    parser.add_argument("--few-shot",        action="store_true",
                        help="A2: Prepend 5 labeled training examples from OOF")
    parser.add_argument("--split-endpoints", action="store_true",
                        help="A3: Separate prompts for hepatic vs. death endpoints")
    parser.add_argument("--n-samples",       type=int,   default=1,
                        help="A4: Average N API calls per patient at temperature=0.7")
    parser.add_argument("--cot",             action="store_true",
                        help="A5: Chain-of-thought reasoning before JSON output")
    args = parser.parse_args()
    main(
        n=args.n,
        model=args.model,
        provider=args.provider,
        dry_run=args.dry_run,
        delay=args.delay,
        few_shot=args.few_shot,
        split_endpoints=args.split_endpoints,
        n_samples=args.n_samples,
        cot=args.cot,
    )
