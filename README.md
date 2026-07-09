# ANNITIA 2026 Final Submission Code

Lean reproduction code for my best ANNITIA Data Challenge 2026 submission:

- public leaderboard score: `0.90364`
- final file: `submission_helm_s1w50_base_w15.csv`
- metric: `0.7 * C-index_hepatic + 0.3 * C-index_death`

The solution predicts major hepatic events and all-cause mortality in synthetic
longitudinal MASLD patient records.

## Core Idea

This project treats a tabular clinical survival problem as more than a
spreadsheet modeling task. The strongest part of the solution is a GPT-OSS
HeLM-style branch that serializes each patient's longitudinal biomarker history
into compact clinical text, asks a reasoning LLM to produce endpoint-specific
risk scores, and then uses those scores as an orthogonal ranking signal in a
survival ensemble.

The LLM is not used as a chatbot wrapper around the model. It acts as a learned
clinical pattern recognizer over irregular tabular trajectories: rising enzymes,
falling platelets, fibrosis-test dynamics, cardiometabolic comorbidities, and
endpoint-specific risk cues that tree models may compress away. The final gain
comes from combining that clinical-reasoning signal with strong survival models,
not from replacing classical modeling.

## Why GPT-OSS Was Useful

The base model is deliberately strong and conventional: XGBoost AFT plus
CatBoost SurvivalAft on leakage-safe engineered longitudinal features. GPT-OSS
adds value because it makes a different kind of error. Its risk rankings were
weak as a standalone submission, but sufficiently different from the tabular
base to improve the final rank ensemble.

Key design choices:

- Patient records are converted from wide longitudinal tables into clinically
  readable summaries.
- Hepatic-event risk and death risk are scored with split-endpoint prompts.
- GPT-OSS scores are treated as monotonic rankings, not calibrated
  probabilities.
- Rank blending is used because leaderboard metric is C-index, so ordering
  matters more than absolute scale.

This is the research idea I would emphasize: foundation models can be useful for
structured medical prediction when they are used as reasoning-based feature or
ranking generators, then disciplined by survival modeling, leakage controls,
and validation.

## Research Fit

For a PhD application, especially to a lab working on machine learning for
healthcare such as van der Schaar Lab, this project demonstrates the kind of
questions I want to work on:

- How can foundation models help with tabular and longitudinal EHR-style data,
  not just text or images?
- When should an LLM be a predictor, a representation generator, or an
  ensemble-diversity source?
- How do we combine clinical priors, survival analysis, and modern foundation
  models without leaking outcome information?
- How do we evaluate a high-capacity clinical reasoning signal when standalone
  accuracy is weaker than its value inside a robust ensemble?

The submission is therefore a compact example of clinically grounded ML:
survival modeling, longitudinal feature engineering, foundation-model reasoning,
and careful competition integrity in one pipeline.

## What Is Kept

```text
scripts/
  push_score_allout.py              # 10-seed XGBoost AFT + CatBoost base
  ablation_finetune_efficientnet.py # EfficientNet-B0 image/tabular fine-tune
  annitia_helm_submit.py            # GPT-OSS/HeLM risk scoring via Ollama
  helm_gptoss_blend_v2.py           # rank-blend utility for final ensemble
  feature_revolution.py             # engineered clinical feature matrix
  annitia_longitudinal_features.py  # longitudinal summaries and NIT features
  validate_submission.py            # simple CSV contract check
```

Raw challenge data, generated submissions, model artifacts, webapp assets, and
exploratory notes are intentionally excluded.

## Expected Data

Place the official challenge CSVs under `data/`:

```text
data/
  DB-1773398340961.csv      # train, with outcome columns
  DB-1773398340961 (1).csv  # test
```

Generated/frozen artifacts used by the final blend should live under
`artifacts/` when reproducing the exact final submission.

## Reproduce

Base tabular submission:

```bash
python scripts/push_score_allout.py --submit --quick --no-gbsa
```

EfficientNet-B0 fine-tune cell used in the final blend:

```bash
python scripts/ablation_finetune_efficientnet.py --cell cox-sgd-none-fuse
```

GPT-OSS split-endpoint HeLM scores:

```bash
ollama serve
ollama pull gpt-oss:120b-cloud
python scripts/annitia_helm_submit.py \
  --model gpt-oss:120b-cloud \
  --provider ollama \
  --split-endpoints \
  --delay 0
```

Two-stage final rank blend:

```bash
python scripts/helm_gptoss_blend_v2.py \
  --base-path artifacts/submission_finetune_cox-sgd-none-fuse_standalone.csv \
  --helm-path artifacts/submission_helm_oll-gpt-oss-120b-cloud_split.csv \
  --weights 0.50 \
  --tag splithelm_effb0

python scripts/helm_gptoss_blend_v2.py \
  --base-path artifacts/submission_allout_10seed_50_50.csv \
  --helm-path submission_helm_splithelm_effb0_w50.csv \
  --weights 0.15 \
  --tag s1w50_base
```

Validate a produced submission:

```bash
python scripts/validate_submission.py submission_helm_s1w50_base_w15.csv
```

## Integrity

- Patient-level splits only.
- Per-endpoint post-event visit masking before feature extraction.
- Train-fold-only imputation and missingness filtering.
- Label proxies such as `follow_up_years` and `Age_delta` are dropped.
- No raw challenge data or generated submissions are committed.
