# ANNITIA 2026 Final Submission Code

Lean reproduction code for my best ANNITIA Data Challenge 2026 submission:

- public leaderboard score: `0.90364`
- final file: `submission_helm_s1w50_base_w15.csv`
- metric: `0.7 * C-index_hepatic + 0.3 * C-index_death`

The solution predicts major hepatic events and all-cause mortality in synthetic
longitudinal MASLD patient records.

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
