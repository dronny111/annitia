# ANNITIA talk

This folder contains the final public materials for **LLM Risk Rankers and Agentic Scaffolding for Liver Disease Survival Modeling**, a technical talk based on the ANNITIA 2026 challenge.

The talk follows an apparent leaderboard improvement from adding an LLM to a strong tabular survival model. It explains why the blend initially looked plausible, then audits the result for statistical noise, repeated leaderboard testing, and leakage from follow-up duration. The leak-free replication does not support the apparent gain.

## Files

- [`annitia_talk.pdf`](annitia_talk.pdf) — the final 49-page presentation and the easiest version to read or present.
- [`annitia_talk.ipynb`](annitia_talk.ipynb) — the executed Jupyter notebook containing the narrative, code, figures, and saved outputs behind the presentation.

## Data note

Patient examples shown in the talk are synthetic. Published results use aggregate performance measurements; the folder contains no challenge patient-level data, predictions, submissions, or credentials.

The notebook is provided as an executed record of the analysis. Supporting build files and rehearsal materials are intentionally omitted from this public folder.
