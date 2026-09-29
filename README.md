# ANNITIA 2026 Submission Code


Metric: `0.7 * C-index_hepatic + 0.3 * C-index_death`

The solution predicts major hepatic events and all-cause mortality in synthetic
longitudinal MASLD patient records.

## Core Idea

This project treats a tabular clinical survival problem as more than a
tabular modeling task. The strongest part of the solution is a GPT-OSS
HeLM-style branch that serializes each patient's longitudinal biomarker history
into compact clinical text, asks a reasoning LLM to produce endpoint-specific
risk scores, and then uses those scores as an orthogonal ranking signal in a
survival ensemble.

The LLM is not used as a chatbot wrapper around the model. It acts as a learned
clinical model over irregular tabular trajectories including rising enzymes,
falling platelets, fibrosis-test dynamics, cardiometabolic comorbidities, and
endpoint-specific risk cues that tree models may miss to capture. The final gain
comes from combining that clinical-reasoning signal with strong survival models,
not from replacing classical modeling.

## Why GPT-OSS Was Useful

The base model is an ensemble of XGBoost AFT plus
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

This is a research idea I would argue for: foundation models can be useful for
structured medical prediction when they are used as reasoning-based feature or
ranking generators in right-censored survival modelling tasks.


## Repo Layout

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


## Follow-up leak (fixed)

`feature_revolution.load_and_prepare` now drops features computed over each patient's observation
window, such as visit counts, missing rates and last-visit age (`WINDOW_FEATURES`). For censored
patients, last-visit age minus first-visit age equals the target time, so these features let a
model read follow-up length. That span alone scored a CV C-index of 0.96. The fixed 10-seed base
scores 0.902 in CV, close to its public leaderboard score of 0.894. Submissions made before this
fix were trained with the leaky features, so rerunning the steps below will not reproduce them
exactly.

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
