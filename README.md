# ANNITIA MASLD Survival Modeling

Recovered research code and notes for the ANNITIA Data Challenge 2026, focused on
AI-based risk stratification for metabolic dysfunction-associated steatotic liver
disease (MASLD).

The project predicts two patient-level survival endpoints:

- major hepatic events
- all-cause mortality

The challenge score is a weighted C-index:

```text
0.7 * C-index_hepatic + 0.3 * C-index_death
```

## Highlights

- Longitudinal feature engineering from wide clinical visit data.
- Leakage-aware post-event visit masking.
- Clinical markers and thresholds including FibroScan, FibroTest, APRI, FIB-4,
  platelet thresholds, transaminase elevation, and visit-pattern summaries.
- Survival ensembles using XGBoost AFT and CatBoost SurvivalAft.
- Patient-level stratified cross-validation.
- Recovered experiment notes for tabular, EfficientNet-style image, HeLM/LLM,
  and rank-blending approaches.

Recovered run notes record:

- public leaderboard score: `0.90364`
- 10-fold tabular CV: combined `0.9474`, hepatic `0.9556`, death `0.9282`
- final blend recipe: tabular survival ensemble + EfficientNet-B0 finetune +
  HeLM rank blend

Those numbers come from recovered local history and session checkpoints after
the original project folder was deleted. They have not been rerun in this
reconstructed checkout because the official challenge data and final artifacts
are not included.

## Main Files

- `annitia_pipeline.py` - end-to-end survival pipeline.
- `annitia_longitudinal_features.py` - reusable longitudinal feature builder.
- `feature_revolution.py` - ablation-oriented feature engineering pipeline.
- `tools/autoannitia_runner.py` - autonomous experiment runner recovered from
  Copilot session files.
- `recovery_notes/` - recovered implementation and final-submission checkpoints.
- `RECOVERY_MANIFEST.json` - provenance for every restored file.

## Data Policy

Raw challenge CSVs, generated submissions, trained models, large artifacts, and
third-party reference notebooks are intentionally not tracked in git. Place
official ANNITIA CSV files under `data/` when running locally:

```text
data/
  DB-*.csv
```

The repository includes a starter notebook snapshot under `data/`, but not the
official train/test CSVs.

## Quick Start

Create an environment with the dependencies in `requirements.txt`, add the
official challenge CSV files to `data/`, then run:

```bash
python annitia_pipeline.py --cv-only
python feature_revolution.py --variant engineered_clinical_interactions
```

To generate a challenge submission when the official test CSV is present:

```bash
python annitia_pipeline.py --submit-only
```

## Recovery Notes

This repository was reconstructed from:

- VS Code local history snapshots
- Copilot session checkpoints
- Gemini project traces

The reconstruction prioritizes authored code, reproducible methodology, and
research documentation suitable for a public portfolio. Files likely to be
third-party reference material were restored locally under `reference_archive/`
but are ignored by git.

## License

MIT License. See `LICENSE`.
