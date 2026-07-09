# Biomni CV Ideas

## 1. Image structure

The median patient has only 3 observations per biomarker, and 10% have just 1 visit. A GASF/MTF image from a 3-point series is a 3×3 matrix — essentially 9 pixels, which carries almost no spatial structure for a CNN to learn from. How do you want to handle this?

- **Option A (recommended): Multi-biomarker 2D grid image** — treat the 12 biomarkers as rows and time slots (padded to a fixed length, e.g. 22 columns) as columns. Each patient becomes a 12×22 image. GASF/MTF are then applied across the time axis per biomarker row. This gives a meaningful 2D structure even for patients with few visits.
- **Option B: Per-biomarker images, padded to 22×22** — one GASF+GADF+MTF triple-channel image per biomarker, padded with zeros where visits are missing. 12 images per patient, processed by 12 parallel CNN streams or a single CNN with 36 input channels.
- **Option C: Concatenated multi-biomarker series** — flatten all 12 biomarkers into one long series (length up to 12×22=264), then encode as a single large image. Loses the biomarker identity structure.
- **Option D: You decide** — I'll choose based on what's most likely to work given the data sparsity.

## 2. Irregular timing

The visits are irregularly spaced (inter-visit gap: median 1 year, p95 = 4 years). Standard GASF/MTF treat observations as equally spaced in time — which is wrong here. How should we handle the irregular timing?

- **Option A (recommended): Time-aware interpolation** — before encoding, interpolate each biomarker onto a regular annual grid (e.g. years 0, 1, 2, … up to max follow-up), using linear interpolation between observed visits and NaN-fill outside the observation window. This makes the temporal axis meaningful.
- **Option B: Ignore timing, use visit order** — treat v1, v2, v3… as equally spaced regardless of actual age gaps. Simpler, but the image encodes visit sequence not calendar time.
- **Option C: Include Age_v* as an explicit channel** — use visit order for the image but add a 4th channel encoding the actual time gap at each position.
- **Option D: You decide**

## 3. Survival layer

For the survival/competing risks layer on top of the CNN, which approach do you want?

- **Option A (recommended): DeepHit** — discretises time into bins, outputs a cause-specific hazard matrix for both hepatic events and death simultaneously. Handles competing risks natively. The competition metric (0.7×C_hep + 0.3×C_death) maps directly onto this. Requires choosing a time grid (e.g. 0–21 years in 1-year bins).
- **Option B: Two independent DeepSurv heads** — one CNN backbone, two Cox-PH output heads (one per endpoint). Simpler, but treats the endpoints as independent (ignores competing risks).
- **Option C: You decide**

## 4. Model capacity

Given that we have only 47 hepatic events and 1253 patients total, a CNN trained from scratch will almost certainly overfit catastrophically — far worse than the gradient boosting we already tried. How do you want to handle the model capacity?

- **Option A (recommended): Tiny custom CNN from scratch** — 2–3 conv layers, heavy dropout (0.5), L2 regularisation, trained only on this data. No pretrained weights (images are not natural images). This is the most principled approach but may still overfit.
- **Option B: Transfer learning from a pretrained model (e.g. EfficientNet-B0)** — fine-tune only the final layers. Pretrained weights encode natural image priors that don't apply here, but the architecture's inductive biases (local feature detection) may still help.
- **Option C: Linear probe only** — freeze a pretrained CNN backbone, train only a linear survival head. Minimal overfitting risk but the features are completely mismatched to clinical time-series images.
- **Option D: You decide**

---

## Your response

- **Image structure:** A — 12×22 grid image
- **Irregular timing:** A — Regular grid interpolation
- **Survival layer:** A — DeepHit (competing risks)
- **Model capacity:** A — Tiny CNN from scratch

---

# Enhanced brainstorm (2026-04-17)

This section expands the original four design choices with concrete implementation
detail, alternative formulations, risk mitigation, and a realistic go/no-go
assessment against the confirmed 0.89442 ceiling. All ideas respect the
integrity constraints: patient-level splits, post-event masking, train-only
preprocessing, no test-label inference.

## 0. Context anchor — what "Biomni CV" must beat

| Anchor | Public | CV | Notes |
|---|---|---|---|
| 10-seed XGB AFT + CatBoost SurvivalAft (50/50) | **0.89379** | 0.9464 | pure-tabular ceiling |
| EfficientNet-B0 embeddings blend @ w=0.01 | **0.89442** | n/a | best confirmed overall |
| ViT-Small embeddings blend (best weight) | 0.841 | 0.9544 | emb-only degrades |
| Image backbones emb-only (4 variants) | 0.744–0.779 | 0.79–0.82 | all below ceiling |

**Precedent:** 4×4 tiled GASF/MTF images + ImageNet backbones already tested and
*closed*. They contribute ≤+0.0006 public when blended at w≈0.01; pure emb-only
is 0.11–0.15 below the tabular base. Any new Biomni CV proposal must either
(a) substantially improve over emb-only (i.e., push emb-only above ~0.85 so the
blend can contribute >0.001), or (b) generate fundamentally different residuals
than existing trees so w≈0.05–0.10 becomes useful.

## 1. Image structure — deeper analysis

### Chosen: Option A (12×22 biomarker × time grid)

**Concrete spec:**
- Rows = 12 biomarkers in clinically-grouped order:
  1. NITs (highest signal): `fibs_stiffness_med_BM_1`, `fibrotest_BM_2`, `aixp_aix_result_BM_3`
  2. Liver enzymes: `alt`, `ast`, `ggt`, `bilirubin`
  3. Metabolic: `chol`, `gluc_fast`, `triglyc`
  4. Hematology/composition: `plt`, `BMI`
- Columns = 22 annual time slots (age-anchored, see §2)
- Channel 1: standardized biomarker value (z-score on train, winsorized to ±3σ)
- Channel 2: observation mask (1 = observed, 0 = imputed/missing)
- Channel 3: age at that time slot (normalized)
- **Not GASF/MTF at this stage** — the raw 12×22 clinical-matrix image carries
  interpretable structure per-row. GASF/MTF buys nothing at 22 time points and
  destroys clinical interpretability.

**Why this beats Option B (per-biomarker GASF):**
- Cross-biomarker interactions (e.g. ALT-AST ratio, FIB-4 numerator/denominator)
  are encoded spatially when rows are adjacent.
- CNN kernels can learn local 3×3 patches = (3 biomarkers × 3 years) — matches
  clinical reasoning ("ALT+AST+platelets trending over 3 years").
- 12×22 = 264 pixels/channel is information-dense; 22×22 GASF is mostly zero
  padding for patients with 3 visits.

**Ordering experiments to run (cheap):**
1. Clinical grouping (above) — default
2. Hierarchical-clustering order of biomarker correlations on train
3. Learned order via a single extra 1×1 conv that permutes rows
4. Augmentation: random row permutation within clinical groups (regularizer)

### Alternative structures worth reserving as ablations

- **A+: 13×22 with Age_v as an explicit row** — adds the age trajectory as a
  first-class feature row (currently Channel 3). May help kernels learn
  "trajectories conditional on age" without needing cross-channel attention.
- **A++: 14×22 adding gender+comorbidity broadcast rows** (binary repeated
  across time). Lets the CNN learn gender/T2DM-conditional fibrosis dynamics
  without requiring a separate tabular branch.
- **A-tiny: 12×8 "recent-visits-only"** — last 8 annual slots. Reduces
  parameter count 3× and aligns with the fact that recency dominates hepatic
  C-index (per `learnings.md`).

## 2. Irregular timing — deeper analysis

### Chosen: Option A (regular grid interpolation)

**Concrete spec:**
- Anchor: age at visit v1 (baseline age) is time slot 0; each subsequent slot
  is +1 year. Max slot = 21 (22 total).
- Linear interpolation between observed annual values; NaN outside observation
  window (before v1 or after last visit / event-censor age).
- **Critical leakage guard:** the "last observation" for a patient is the last
  visit age ≤ event_age − ε (ε = 0 days for hepatic, event_age − ε for death
  endpoint); interpolation window is clipped accordingly per-endpoint.
  Two image tensors needed: `images_train_hep.npy` and `images_train_death.npy`
  (infrastructure already exists).

**Refinements to consider:**
- **Piecewise-constant (last-observation-carried-forward) alternative** —
  clinically more realistic than linear for biomarkers that are episodic
  (e.g., FibroScan done once every 2–3 years). Could be a channel of its own.
- **Gap-aware channel** — Channel 4 = time-since-last-real-observation at each
  slot (0 at observed slots, 1, 2, … in interpolated ones). This lets the CNN
  down-weight interpolated pixels implicitly.
- **Missing-window masking as explicit loss weight** — if >50% of a patient's
  image is interpolated, apply a lower loss weight (√fraction-observed).

### Risk: interpolation hallucinates signal

Linear interpolation between (alt=30 at age 40) and (alt=90 at age 48) creates
7 intermediate "synthetic" observations. If the model learns from these, CV
improves but public stays flat because the same hallucination happens equally
on test. **Mitigation:** always include Channel 2 (observation mask). Consider
an ablation where interpolated pixels are zeroed rather than filled, forcing
the CNN to learn only from real observations.

## 3. Survival layer — deeper analysis

### Chosen: Option A (DeepHit)

**Reality check from prior work:** DeepHit + Dynamic DeepHit have **already been
tested and closed** on this dataset (see `learnings.md`):
- DeepHit CV 0.7398, public 0.729 — gradient starvation at 8–10 events/batch.
- All DDH blends with the 10-seed base closed (0.853–0.879 public).

This does NOT mean the Biomni CV DeepHit is doomed — the prior failure was
with tabular inputs, not an image backbone — but it does mean we must engineer
the loss to survive N_events=47.

**Concrete DeepHit spec for Biomni CV:**
- Time bins: 22 annual bins covering years 0–21 from age_at_v1.
- Output: 2 × 22 = 44 logits (hepatic cause-specific hazard + death
  cause-specific hazard).
- Loss components:
  1. Negative log-likelihood (NLL) of observed event-time-cause.
  2. Ranking loss (pairwise concordance) with weight λ_rank — crucial at
     small N_events; DeepHit's original paper uses λ_rank=1.0.
  3. Competition-weighted term: `0.7 × rank_loss_hep + 0.3 × rank_loss_death`
     to align gradient with the evaluation metric.
- Censoring handling: standard DeepHit (censored patients contribute only
  via the "no event by T_cens" likelihood term).

**Mitigations for 47-event fragility:**
- **Warm-start from tabular pseudo-labels.** Train the CNN first as a
  regressor onto XGB AFT out-of-fold log-risk scores (both endpoints). This
  gives a well-shaped feature extractor; only then swap in the DeepHit head.
- **Loss weighting by event rarity.** Up-weight event samples by
  (1 − event_rate)/event_rate to combat gradient starvation.
- **Large effective batch via gradient accumulation.** Effective batch ≥256
  ensures each step sees ≈10 hepatic events on average.
- **Focal-style modulation** on the NLL to prevent saturated easy-censor
  gradients from dominating (γ=1–2).

### Alternatives to DeepHit worth considering

| Option | Strength | Weakness |
|---|---|---|
| **Two independent DeepSurv (Cox-PH) heads (Opt B)** | Battle-tested; 1 number per patient per endpoint; cleanest blend with tabular ranks | Ignores competing risks; PH assumption violated (see Cox public 0.658) |
| **AFT-MLP head** (log-normal, same as XGB) | Matches tabular loss family → easier blend; CV-public gap predictable | No competing-risks modeling |
| **MTLR (Multi-Task Logistic Regression)** | Discrete-time, smoother than DeepHit, handles censoring properly | Still rank-based gradients; similar EPV concerns |
| **Survival-stacked BCE** (long-format per-bin binary classification) | Simple; each bin is a large dataset; easy pretraining | Does not directly optimize C-index |
| **Dual heads: AFT (log-time) + DeepHit-rank (auxiliary)** | Best of both — AFT supplies stable gradient, DeepHit ranks refine | More hyperparameters; risk of task interference |

**Recommendation:** run DeepHit as planned **and** a parallel AFT-MLP head
variant. Compare emb-only C-index and blend stability. Pick whichever yields
larger post-blend public uplift.

## 4. Model capacity — deeper analysis

### Chosen: Option A (tiny CNN from scratch)

**Concrete spec (baseline tiny CNN):**

```
Input: (B, 4, 12, 22)       # 4 channels: value, mask, age, gap
Conv2d(4→16, 3×3, pad=1) + BN + ReLU + Dropout2d(0.3)
Conv2d(16→32, 3×3, pad=1) + BN + ReLU + Dropout2d(0.3)
AvgPool2d(2×2) → (32, 6, 11)
Conv2d(32→32, 3×3, pad=1) + BN + ReLU + Dropout2d(0.5)
AdaptiveAvgPool2d(1) → (32,)
Linear(32 → 44)  # DeepHit head: 22 bins × 2 causes
```

Param count ≈ 15k — matches N_events=47 at ~300 params/event, which is the
conventional ceiling. With L2=1e-4 and dropout 0.3–0.5, this is the minimum
viable architecture.

**Regularization stack (non-negotiable at 47 events):**
1. Dropout 0.3–0.5 on every conv block.
2. Weight decay 1e-4 to 1e-3 (Adam/AdamW).
3. **Data augmentation:**
   - Biomarker-row jitter: Gaussian noise on value channel (σ=0.05 z-scores).
   - Time dropout: randomly zero 10–20% of observed columns (simulate missing
     visits, regularizes against over-reliance on full trajectories).
   - Row permutation within clinical groups (see §1).
   - MixUp at the image-tensor level across pairs of same-event-status
     patients (preserves competing-risks labels).
4. **Early stopping** on StratifiedGroupKFold OOF C-index (fold-averaged).
5. **SWA / EMA** of weights to reduce final-epoch noise.
6. **Ensemble of 5–10 seeds** by default — parallels the tabular pipeline.

### Alternative capacities to try

| Variant | Rationale |
|---|---|
| **Linear probe on pretrained EfficientNet-B0 tiled image** | Already partly done (w=0.01 blend); include as lower baseline. |
| **Tiny 1D CNN per biomarker row + late fusion** | Respects that within-row signal dominates cross-row. |
| **Depthwise-separable tiny CNN** | ~3× fewer params; may be enough at 47 events. |
| **TabTransformer over (biomarker × time) tokens** | Attention over 264 tokens; explicitly encodes biomarker identity via row-embedding. |
| **Tiny Conv + tabular-branch fusion** | Concatenate the 32-dim CNN features with the 165 engineered tabular features into a joint MLP head. Best chance of exceeding tabular ceiling. |

## 5. New idea — joint image+tabular fusion (not in original Q&A)

The single largest missing piece: the current plan builds a **pure image
model**, but our evidence (EfficientNet-B0 blend at w=0.01) shows the image
signal is <5% of usable signal. A fusion architecture is much more likely
to exceed 0.89442:

```
image  → tiny CNN → emb_img  (32-dim)
tabular→ MLP(165→64) → emb_tab (64-dim)
concat([emb_img, emb_tab]) → MLP(96→64→44)  # DeepHit head
```

Trained end-to-end, the image branch only has to learn *residual* signal that
the tabular pipeline misses. This is a much lower bar than beating the tabular
pipeline outright.

**Expected behavior:**
- If image branch is useful: fusion C-index > tabular C-index alone.
- If image branch is not useful: backprop through the image path should shrink
  its gradient to ~0 and fusion converges to tabular-alone (safe failure).

Risk: co-adaptation between branches. Mitigation: train tabular-branch first,
freeze it for the first 20 epochs of CNN training, then unfreeze with lower LR.

## 6. New idea — contrastive / self-supervised pretraining

With 1,253 unlabeled patient-images (train+test union = 1,676), we can
pretrain the CNN without using test labels:

- **SimCLR-style:** two augmented views of the same patient image → pull
  together; different patients → push apart. Augmentations as in §4.
- **Masked autoencoder (MAE):** mask 50% of time-slot columns, reconstruct.
  Forces the CNN to learn trajectory continuity.
- **Time-contrastive:** same patient at two truncation lengths → pull
  together. Encourages features invariant to visit count.

These are integrity-safe (no test labels used). Prior supervised
image-backbone attempts failed because ImageNet priors are irrelevant;
SSL on clinical images learns the right priors from scratch.

**Expected gain:** worth testing; high-uncertainty direction. SSL has been a
consistent win in low-label tabular medical imaging literature.

## 7. New idea — explicit NIT-aware image layout

Three NITs (FibroScan, FibroTest, Aixplorer) carry the dominant signal
(per `learnings.md`: "aixp alone achieves C=0.85"). A CNN averaging over
12 biomarker rows dilutes this. Instead:

- Put the 3 NITs as the top 3 rows (already in §1 ordering).
- **Apply a 3×K row-wise asymmetric conv first** (kernel spans exactly the
  3 NIT rows × K time slots), before any cross-biomarker mixing.
- Branch architecture: (a) NIT-only 3×22 sub-image → small CNN → emb_nit;
  (b) full 12×22 → main CNN → emb_full. Concatenate.

This gives the model an explicit "NIT highway" that cannot be blurred by
pooling across biomarker rows.

## 8. Blending strategy (not in original Q&A but decisive)

Based on the EfficientNet-B0 experience:

- **Step 1 emb-only target:** beat 0.82 public (current emb-only ceiling for
  any image approach). If <0.80, the image model is not useful and
  improvement in step 2 is capped at ≤+0.0003.
- **Step 2 blend grid:** rank-average with 10-seed tabular base at
  w ∈ {0.005, 0.01, 0.02, 0.03, 0.05, 0.10}. Expect peak around w=0.01–0.05.
- **Step 3 combine with HeLM blend:** if Biomni CV yields +0.002 public
  (over 0.89442), test triple-blend with HeLM at w_heLM=0.05 (empirically
  optimal per prior work).

**Kill criteria:**
- Emb-only public < 0.78 → stop; do not blend.
- Best blend Δ < +0.0005 over 0.89442 after 6 weight values → stop.
- OOF C-index instability > ±0.02 across 5 folds → architecture too large.

## 9. Honest feasibility assessment

**Base rate for beating 0.89442 with a new DL approach:** very low.
- 4 image backbones already tried → all below 0.80 emb-only.
- DeepHit on tabular already tried → 0.729 public.
- The ceiling at 0.891 has been validated across ~40 closed directions.

**What is genuinely new here vs. prior attempts:**
1. **12×22 clinical-matrix layout** (vs. 4×4 tiled GASF/MTF). This is the
   single most-different design decision; all prior image work used pyts
   transforms on full 224×224 images.
2. **Training from scratch with leakage-proof masks** (vs. ImageNet transfer).
3. **Fusion with tabular branch** (§5) — not previously attempted.
4. **SSL pretraining on clinical images** (§6) — not previously attempted.

**Realistic outcome distribution:**
- P(public < 0.80): ~45% — tiny CNN overfits catastrophically on 47 events.
- P(0.80 ≤ public < 0.89): ~45% — matches prior image backbones.
- P(0.89 ≤ public < 0.895): ~8% — emb-only surprisingly useful; small blend gain.
- P(public > 0.895): ~2% — tail scenario requires fusion architecture to
  capture genuine residual signal.

**Primary value may be qualitative (30% score component), not quantitative:**
- Clinical-matrix image + SHAP-on-CNN (Grad-CAM) → highly interpretable
  per-patient trajectory explanations.
- "Biomni" framing fits the qualitative narrative (multi-modal biomedical AI).
- This is worth pursuing **even if quantitative gain is zero**.

## 10. Prioritized experimental protocol

**Phase 1 — minimum viable Biomni CV** (1–2 runs):
1. Build 12×22×4 image tensors with per-endpoint post-event masking.
2. Train tiny CNN + AFT-MLP head (simpler than DeepHit, easier to debug).
3. 5-fold StratifiedGroupKFold; report OOF hepatic+death C-index.
4. Gate: OOF combined ≥ 0.90 → continue; < 0.88 → architecture too weak,
   try fusion (§5) before abandoning.

**Phase 2 — DeepHit head + augmentation** (2–3 runs):
1. Swap head to DeepHit with competing-risks NLL + ranking loss.
2. Add augmentation stack from §4.
3. Gate: OOF combined ≥ 0.92 → continue; else stop.

**Phase 3 — blend with tabular** (cheap):
1. Generate test-set predictions; rank-normalize; blend at w grid.
2. Submit best 2–3 weights.

**Phase 4 — fusion and SSL** (conditional on Phase 3 gain ≥ 0).
1. Implement §5 fusion model.
2. Implement §6 SSL pretraining.

**Phase 5 — qualitative assets** (regardless of quantitative outcome).
1. Grad-CAM / integrated gradients overlays on 12×22 images for 5 canonical
   patients (high-risk converter, low-risk stable, etc.).
2. Narrative linking CNN-salient regions to known MASLD progression.
3. Include in final notebook as evidence of multi-modal reasoning.

## 11. Stop criteria (explicit)

- Phase 1 OOF combined < 0.88 AND fusion (§5) also < 0.88 → abandon.
- Phase 3 best blend public < 0.89442 across all weights → abandon
  quantitative direction; keep qualitative assets from Phase 5.
- Any CV > 0.965 → overfitting flag; do not submit (per repo CV reliability
  rules).

## 12. Deliverables even on quantitative failure

1. `biomni_cv_model.py` — reproducible training script.
2. `image_tensors_12x22.npz` — saved tensors for train+test.
3. Grad-CAM PNGs for 5 canonical patients (qualitative report).
4. A section in the final notebook describing the clinical-matrix image
   representation and what the CNN learned — this is novel relative to all
   prior submissions and has qualitative-score value regardless of public
   leaderboard impact.

# new
GASF/MTF vs grid

A geometric conflict exists in the original plan that needs a decision. Standard GASF/GADF/MTF each produce a square T×T matrix from a single time series of length T. But you chose a 12×T grid image (biomarkers as rows, time as columns). These two shapes are incompatible: you can’t fit a T×T matrix into a 12×T grid. There are two principled ways to resolve this: **Option A (recommended): Keep the 12×T grid, use 3 direct signal channels instead of GASF/MTF** Each patient becomes a (12, T, 3) tensor where: - Channel 0: normalised biomarker values on the annual grid (the raw trajectory) - Channel 1: first-order differences (year-over-year change — captures rate of progression) - Channel 2: a missingness mask (1 = observed or interpolated, 0 = outside observation window) This preserves the 12×T structure you chose, is not leaky, and gives the CNN interpretable local features (e.g., a 3×3 conv kernel can detect ‘rising FibroScan + rising AST over 3 years’). The GASF/MTF encoding is abandoned but the spirit — encoding temporal dynamics as a 2D image — is preserved. **Option B: Keep GASF/GADF/MTF, switch to per-biomarker square images** Each biomarker gets its own T×T image (3 channels: GASF, GADF, MTF). 12 images per patient, each (T, T, 3). A single CNN processes all 12 images by treating them as a (12×T, T, 3) stacked image or using 12 parallel streams. This is the original GASF/MTF idea but requires switching away from the 12×T grid you chose. Note: with T=15 (covering 96% of follow-up), each per-biomarker image is 15×15 pixels — still very small for a CNN.

Grid width T

The annual grid width T determines the image width and the DeepHit time bins. The data shows: - T=10 covers 85% of follow-up times (15% of patients truncated) - T=15 covers 96% of follow-up times (4% truncated) - T=21 covers 100% but makes the image wider and adds many empty columns for short-follow-up patients For DeepHit, the same T is used as the number of discrete time bins (e.g., T=15 means bins at years 1, 2, …, 15). Events beyond T are treated as censored at T. Note: 8 of the 47 hepatic events occur in year 0 (time ≈ 0.001) — these are patients where the event age equals Age_v1. These are likely data artefacts (event already present at baseline). They need special handling regardless of T.

Zero-time events

8 of the 47 hepatic events have time ≈ 0 (event age = Age_v1, i.e., the event was already present at the first visit). These patients have no at-risk period and will confuse any survival model. How should they be handled? This is a real data quality issue, not a modelling choice — but it affects 17% of your hepatic events so the decision matters.

Submission strategy

Should the image-based model be trained and submitted as a standalone replacement, or blended with the RSF baseline (hello_world, public LB 0.7865)? Given the severe event-count constraint (47 hepatic events), a blend is almost always safer: if the CNN overfits, the RSF anchor prevents catastrophic collapse. If the CNN genuinely learns something new, the blend captures it.

Your response:

GASF/MTF vs grid: A: 12×T grid + direct channels
Grid width T: T=15
Zero-time events: ablate all the different techniques and choose the best - drop them, clip to 0.5 years or keep as-is
Submission strategy: CNN standalone