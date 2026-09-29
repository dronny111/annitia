"""Seeded synthetic stand-ins for MASLD longitudinal records and for the models ranking them.

Nothing here comes from the challenge. Patients are drawn from a toy disease model whose column
names mimic the real wide schema (``alt_v1`` .. ``alt_v22``), so the repo's own serializer runs on
them unchanged:

* a latent fibrosis level rises at a patient-specific rate and drives the liver markers and the
  hepatic hazard;
* age and cardiometabolic comorbidities drive the death hazard;
* visits stop at the end of follow-up, so sicker patients have fewer visits;
* event times are Weibull, right-censored by a uniform administrative end date.
"""
import numpy as np
import pandas as pd

MAX_VISITS = 22


def _sigmoid(x):
    return 1 / (1 + np.exp(-x))


def cohort(n: int = 400, seed: int = 0) -> pd.DataFrame:
    """One wide row per patient, plus outcome columns ``hep_event, hep_time, death_event, death_time``."""
    rng = np.random.default_rng(seed)
    age0 = rng.uniform(40, 72, n)
    male = rng.random(n) < 0.5
    t2dm = rng.random(n) < 0.35
    htn = rng.random(n) < 0.45
    dysl = rng.random(n) < 0.5
    bari = rng.random(n) < 0.05
    fib0 = rng.normal(0, 1, n) + 0.4 * t2dm           # latent fibrosis at baseline
    rate = np.clip(rng.normal(0.08, 0.08, n) + 0.05 * t2dm, -0.05, None)

    # Hazards: hepatic from fibrosis level and progression; death from age and metabolic load.
    lin_h = 1.3 * fib0 + 6 * rate - 2.9
    lin_d = 0.07 * (age0 - 55) + 0.5 * t2dm + 0.35 * htn + 0.3 * fib0 - 2.8
    weib = lambda lin, k: (-np.log(rng.random(n)) / np.exp(lin)) ** (1 / k) * 6
    t_hep, t_death, t_cens = weib(lin_h, 1.4), weib(lin_d, 1.6), rng.uniform(3, 14, n)
    follow = np.minimum.reduce([t_hep, t_death, t_cens])

    rows = []
    for i in range(n):
        gaps = rng.uniform(0.4, 1.4, MAX_VISITS)
        times = np.concatenate([[0], np.cumsum(gaps)[:-1]])
        times = times[times < follow[i]][:MAX_VISITS]
        row = {"gender": int(male[i]), "T2DM": int(t2dm[i]), "Hypertension": int(htn[i]),
               "Dyslipidaemia": int(dysl[i]), "bariatric_surgery": int(bari[i])}
        bmi0 = rng.normal(31, 4)
        for v, t in enumerate(times, start=1):
            f = fib0[i] + rate[i] * t
            noisy = lambda mu, sd: float(mu + rng.normal(0, sd))
            row[f"Age_v{v}"] = age0[i] + t
            row[f"BMI_v{v}"] = noisy(bmi0 - 0.1 * t, 0.8)
            row[f"alt_v{v}"] = max(8.0, noisy(38 + 12 * f, 8))
            row[f"ast_v{v}"] = max(8.0, noisy(30 + 11 * f, 6))
            row[f"ggt_v{v}"] = max(8.0, noisy(55 + 25 * f, 15))
            row[f"plt_v{v}"] = max(40.0, noisy(245 - 38 * f, 20))
            row[f"bilirubin_v{v}"] = max(3.0, noisy(10 + 3 * max(f, 0), 2))
            row[f"gluc_fast_v{v}"] = noisy(5.4 + 1.6 * t2dm[i], 0.5)
            row[f"triglyc_v{v}"] = max(0.4, noisy(1.5 + 0.5 * dysl[i], 0.3))
            if rng.random() < 0.7:                    # NITs are not measured every visit
                row[f"fibs_stiffness_med_BM_1_v{v}"] = max(3.0, noisy(6.5 * np.exp(0.3 * f), 0.8))
                row[f"fibrotest_BM_2_v{v}"] = float(np.clip(noisy(_sigmoid(f - 0.5), 0.05), 0, 1))
            if rng.random() < 0.3:
                row[f"aixp_aix_result_BM_3_v{v}"] = noisy(20 + 0.3 * (age0[i] - 55), 5)
        row.update(hep_event=bool(t_hep[i] <= min(t_death[i], t_cens[i])),
                   death_event=bool(t_death[i] <= min(t_hep[i], t_cens[i])),
                   hep_time=follow[i], death_time=follow[i],
                   true_hep=lin_h[i], true_death=lin_d[i])
        rows.append(row)
    return pd.DataFrame(rows)


def rankers(truth: np.ndarray, seed: int, tab_noise: float = 0.6, llm_noise: float = 1.0,
            shared: float = 0.0, llm_levels: int = 20) -> tuple[np.ndarray, np.ndarray]:
    """A strong tabular ranker and a weaker LLM ranker of the same true risk.

    ``shared`` is the correlation between their errors (0 = independent mistakes). The LLM
    score is squeezed onto ``llm_levels`` values of a 0-100 scale, which creates the ties real
    LLM risk scores have.
    """
    rng = np.random.default_rng(seed)
    e_tab, e_own = rng.normal(0, 1, (2, len(truth)))
    tab = truth + tab_noise * e_tab
    llm = truth + llm_noise * (shared * e_tab + np.sqrt(1 - shared ** 2) * e_own)
    pct = pd.Series(llm).rank(pct=True).to_numpy() * 100
    return tab, np.round(pct / (100 / llm_levels)) * (100 / llm_levels)
