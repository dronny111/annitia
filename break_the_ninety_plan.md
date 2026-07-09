## The Key to 0.90: It's Not Post-Processing. It's the NITs.

Here's the honest, data-grounded answer.

---

### The Single Biggest Finding

**`aixp_aix_result_BM_3` (BM_3) has a univariate C-index of 0.85 alone.** `fibs_stiffness_med_BM_1` reaches 0.78 at v3. These two NITs are the dominant signal in this dataset — not clever imputation, not post-processing, not model architecture.

The hello-world baseline uses raw visit columns (v1–v4) with a 50% missingness filter. It keeps these NITs, but treats them as flat point-in-time values. The moment you convert them into **trajectories** (last, max, slope, delta), the CV C-index jumps from ~0.77 to **0.879 (Cox) → 0.928 (RSF) → 0.929 (ensemble)** — all with clean, leakage-safe features.

---

### What Actually Drives 0.90

**Ranked by measured impact:**

| Lever | CV C-index gain | Why it works |
|---|---|---|
| **NIT trajectories** (last, max, slope of fibs + fibrotest + aixp) | +0.10–0.15 | Liver stiffness 2× higher in event patients; trajectory captures progression, not just snapshot |
| **Leakage-safe aggregation** (mask post-event visits) | Integrity, not gain | 27/47 event patients have post-event data; using it inflates train score but hurts test |
| **RSF over raw Cox** on these features | +0.05 | Non-linear interactions between NIT + enzyme trajectories |
| **Rank-ensemble Cox + RSF** | +0.005–0.01 | Variance reduction; ensemble CV = 0.929 |
| **FIB-4 / AST/ALT derived markers** | +0.01–0.03 | Adds signal for patients missing NIT data |

---

### Why Post-Processing Won't Get You There

Post-processing (rank calibration, isotonic regression, Platt scaling) adjusts the *scale* of risk scores but **cannot change the ranking** — and C-index is purely rank-based. If your model already ranks patients correctly, post-processing does nothing to C-index. If it doesn't, post-processing can't fix it. This is a dead end for this metric.

---

### Why "Secret Missingness Handling" Is Mostly a Red Herring

The missingness pattern here is structural: early visits (v1–v5) have 35–55% missing; late visits (v10–v22) have 97%+ missing. No imputation strategy recovers signal from 97% missing columns — they are genuinely absent. The real missingness insight is:

1. **Use `n_obs` as a feature** — patients with more NIT measurements are likely under closer clinical surveillance (informative missingness). This is a real signal.
2. **Don't impute late visits** — median imputation on 97% missing columns creates a constant feature that the model memorizes. The hello-world's 50% filter is correct; the mistake is not going further with the kept features.
3. **MICE/iterative imputation** for early visits (35–55% missing) can help marginally, but the gain is small compared to trajectory engineering.

---

### The Remaining Gap: 0.929 → 0.95+

The CV mean is already 0.929. The gap to 0.95+ is mostly **variance from 47 events** — with only 9–10 events per fold, a single misranked patient moves the C-index by ~0.01. The bootstrap 95% CI on train is [0.95, 0.98], confirming the model is already near ceiling on training data.

To push further on test:
1. **Fit on all 1,253 train patients** (not just CV folds) before submitting — this is standard and legitimate
2. **Tune RSF `min_samples_leaf`** — try 10, 15, 20 by CV; smaller leaves capture more signal but risk overfitting with 47 events
3. **Add `aixp` more aggressively** — it has the highest univariate C-index (0.85) but only 310 patients have v1 data. Its `last` and `max` across all visits cover more patients and are the single highest-value features in the dataset
4. **Death model** — the weighted metric is 0.7×hepatic + 0.3×death. The death model (76 events, better EPV) likely has more headroom; improving it from ~0.75 to ~0.85 adds 0.03 to the composite score

---

### The Honest Ceiling

With 47 hepatic events and a synthetic dataset, a leaderboard score of 0.90 is **achievable and realistic** with the trajectory-based RSF approach shown above (CV mean 0.929). Scores above 0.95 on the leaderboard would require either (a) the test set happening to have a favorable event distribution, (b) a feature that captures something the NITs miss, or (c) some form of leakage — which you should not do.

The path is: **NIT trajectories + leakage-safe aggregation + RSF + ensemble**. That's it.