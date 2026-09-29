"""Analytic and mechanistic checks against false-positive gates on a C-index leaderboard.

A false-positive gate says a change helped when it did not. Five criteria, used in the talk:

1. resolution:   gain > 2 x the paired standard error on the gate's own sample
2. multiplicity: gain > se * sqrt(2 ln K) after K looks at one fixed sample
3. mechanism:    gain <= what error independence can deliver (Gaussian ranker model)
4. ceiling:      score below an attainable ceiling (oracle, baseline-only, red-flag CV)
5. replication:  gain also seen on independent evidence (CV folds, the other endpoint)

The mechanism model: each ranker is ``truth + noise`` with noise sd ``s``. A rank blend of two
rankers with error correlation ``r`` behaves like one ranker whose noise sd follows from the
inverse-variance formula below. The C-index of a noise level is read off a calibration cohort.
"""
import numpy as np
from scipy.stats import rankdata
from sksurv.metrics import concordance_index_censored


def cidx(event, time, score) -> float:
    return concordance_index_censored(np.asarray(event, bool), np.asarray(time, float), score)[0]


def rank01(a):
    return (rankdata(a) - 1) / max(1, len(a) - 1)


# ── 1 · Resolution ────────────────────────────────────────────────────────────
def hanley_mcneil_se(c: float, n_events: int, n_non: int) -> float:
    """SE of one AUC-like concordance from n_events positives and n_non negatives (Hanley-McNeil 1982)."""
    q1, q2 = c / (2 - c), 2 * c * c / (1 + c)
    var = c * (1 - c) + (n_events - 1) * (q1 - c * c) + (n_non - 1) * (q2 - c * c)
    return float(np.sqrt(var / (n_events * n_non)))


def paired_delta(cal, s_a, s_c, w, r, n, n_boot=300, seed=0):
    """Mean and SD of C(blend) - C(A) over random n-patient subsets of the calibration cohort.

    A = truth + s_a noise. The blend puts rank weight w on C = truth + s_c noise, whose noise
    has correlation r with A's. The SD is the sampling SE of the gate on an n-patient subset.
    """
    truth, event, time = cal
    rng = np.random.default_rng([seed, 2])        # a stream distinct from the cohort's
    e_a, e_own = rng.normal(size=(2, len(truth)))
    a = truth + s_a * e_a
    c = truth + s_c * (r * e_a + np.sqrt(1 - r * r) * e_own)
    b = (1 - w) * rank01(a) + w * rank01(c)
    deltas = []
    while len(deltas) < n_boot:
        idx = rng.choice(len(truth), n, replace=False)
        if not event[idx].any():                  # a gate with no events cannot score; redraw
            continue
        deltas.append(cidx(event[idx], time[idx], b[idx]) - cidx(event[idx], time[idx], a[idx]))
    return float(np.mean(deltas)), float(np.std(deltas, ddof=1))


# ── 2 · Multiplicity ──────────────────────────────────────────────────────────
def null_max_gain(se: float, k: int) -> float:
    """Typical best 'gain' among k null changes with noise se: the look-elsewhere bar."""
    return se * np.sqrt(2 * np.log(k)) if k > 1 else 0.0


# ── 3 · Mechanism ─────────────────────────────────────────────────────────────
def calibration_cohort(c_oracle: float, n=4000, event_rate=0.10, seed=0):
    """Latent-risk survival cohort whose true risk scores C = c_oracle (bisection on the hazard slope).

    truth ~ N(0, 1); exponential event times with hazard exp(b * truth); administrative censoring
    at the time quantile that leaves ``event_rate`` of patients with an observed event.
    """
    rng = np.random.default_rng([seed, 0])
    z, u = rng.normal(size=n), rng.random(n)
    def make(b):
        t = -np.log(u) / np.exp(b * z)
        cut = np.quantile(t, event_rate)
        return t <= cut, np.minimum(t, cut)
    lo, hi = 0.0, 8.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cidx(*make(mid), z) < c_oracle else (lo, mid)
    event, time = make((lo + hi) / 2)
    return z, event, time


def c_of_sigma(cal, s: float, seed=1) -> float:
    truth, event, time = cal
    return cidx(event, time, truth + s * np.random.default_rng([seed, 1]).normal(size=len(truth)))


def sigma_from_c(cal, c: float) -> float:
    """Noise sd that brings a ranker of the calibration cohort's truth down to C-index c."""
    lo, hi = 0.0, 50.0
    for _ in range(30):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if c_of_sigma(cal, mid) > c else (lo, mid)
    return (lo + hi) / 2


def blend_sigma(s1, s2, r, v):
    """Noise sd of the value blend (1-v) A + v B."""
    return float(np.sqrt((1 - v) ** 2 * s1 * s1 + v * v * s2 * s2 + 2 * v * (1 - v) * r * s1 * s2))


def optimal_value_weight(s1, s2, r):
    """Inverse-variance weight on B: (s1^2 - r s1 s2) / (s1^2 + s2^2 - 2 r s1 s2), clipped to [0, 1]."""
    den = s1 * s1 + s2 * s2 - 2 * r * s1 * s2
    return float(np.clip((s1 * s1 - r * s1 * s2) / den, 0, 1)) if den > 0 else 0.0


def value_to_rank_weight(v, s1, s2, var_truth=1.0):
    """Ranks divide each score by its spread, so a value weight v is a rank weight scaled by spread."""
    sd1, sd2 = np.sqrt(var_truth + s1 * s1), np.sqrt(var_truth + s2 * s2)
    return float(v * sd2 / ((1 - v) * sd1 + v * sd2))


def rank_to_value_weight(w, s1, s2, var_truth=1.0):
    sd1, sd2 = np.sqrt(var_truth + s1 * s1), np.sqrt(var_truth + s2 * s2)
    return float((w / sd2) / ((1 - w) / sd1 + w / sd2))


def predicted_gain(cal, s1, s2, r, w=None):
    """Predicted C gain of blending B into A at rank weight w (None = the optimal weight).

    Returns (gain, rank weight used).
    """
    var_truth = float(np.var(cal[0]))
    v = optimal_value_weight(s1, s2, r) if w is None else rank_to_value_weight(w, s1, s2, var_truth)
    w_used = value_to_rank_weight(v, s1, s2, var_truth) if w is None else w
    return c_of_sigma(cal, blend_sigma(s1, s2, r, v)) - c_of_sigma(cal, s1), w_used


# ── The scorecard ─────────────────────────────────────────────────────────────
def verdict(delta, se, k, mech_max, replicated, below_ceiling=True) -> str:
    if not below_ceiling:
        return "implausible: above ceiling"
    if delta > mech_max + 2 * se:
        return "implausible: no mechanism"
    if delta < 2 * se:
        return "within noise"
    if delta < null_max_gain(se, k):
        return "within noise after K looks"
    return "credible" if replicated else "unreplicated"
