# The Model That Lost Alone and Won the Ensemble: a talk with a live Jupyter walkthrough

A 50-minute technical talk on this repository's ANNITIA 2026 solution. Its argument: a reasoning
LLM that reads a patient's record as clinical text is a weak survival model on its own, but it
improves a strong tabular ensemble because its mistakes are different.

**Data safety.** The talk runs on a seeded synthetic cohort (`synthetic.py`) and our own aggregate
scores (`data/*.csv`: model names and C-index values only). `guard.py` makes any attempt to open a
challenge CSV, a submission, out-of-fold predictions or `reference_archive/` raise, and
`test_talk_data_safety.py` checks the sources and the score tables.

- `annitia_talk.py`: the source (jupytext percent format), the file to review and edit
- `annitia_talk.ipynb`: generated from it and executed headless for verification
- `export_scores.py`: runs the repo's own cross-validation and writes `data/cv_scores.csv`

```bash
pip install -r talk/requirements.txt
jupytext --to ipynb talk/annitia_talk.py
python -m nbconvert --execute --to notebook --inplace talk/annitia_talk.ipynb
cd talk && python -m nbconvert --to slides annitia_talk.ipynb        # annitia_talk.slides.html
# PDF handout: inject print.css, then print one slide per page with headless Chrome
python3 -c "from pathlib import Path; h=Path('annitia_talk.slides.html').read_text(); Path('annitia_talk.print.html').write_text(h.replace('</head>', '<style>'+Path('print.css').read_text()+'</style></head>', 1))"
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless=new --no-pdf-header-footer \
  --virtual-time-budget=20000 --print-to-pdf=annitia_talk.pdf "file://$PWD/annitia_talk.print.html?print-pdf"
rm annitia_talk.print.html
python -m unittest test_talk_data_safety.py
```

**Real numbers.** Without them, the hook chart and the results table use the synthetic cohort and
say so. To add them:

1. Place the challenge CSVs under `data/` and run `python talk/export_scores.py`. Pass
   `--seeds 3` for a quicker run. It runs the repo's CV twice: as the pipeline runs it, and
   window-free. The window-free run drops every feature computed over the observation window,
   such as visit counts, missing rates and age summaries other than first-visit age. The pipeline
   keeps `Age_first` and `Age_last`, and for censored patients their difference is the follow-up
   time. A leak-probe row scores that span alone. If `ablation_finetune_efficientnet_results.csv` exists in the repo
   root, its `cox-sgd-none-fuse` row is included.
2. `talk/data/leaderboard.csv` holds eight hand-picked leaderboard milestones from
   `talk/data/CHALLENGE_SUBS.md`, the full upload history. Each one maps to this repo's pipeline.
   The history's top scores (0.931 and 0.930) came from runs outside this repo, so the talk does
   not claim them. GPT-OSS was never scored on the training set, so its blend gain exists only as
   a leaderboard number.
3. The export also writes `data/cohort_stats.csv`: patient and event counts only. The
   false-positive gate section then uses real event rates instead of assumed ones (10% hepatic,
   8% death).
4. Re-execute the notebook and rebuild the slides.

**False-positive gates.** Section 8 and `gates.py` define five criteria a leaderboard gain must
clear: resolution, multiplicity, mechanism, ceiling and replication. They are applied to our own
upload decisions. The public leaderboard scored 30% of the 423 test patients, about 127.

**Timing (50 min):**
- hook and the argument: 4 min
- objective and rank blending: 3 min
- data and the leakage check: 4 min
- tabular base: 3 min
- serializing a patient and split prompts: 4 min
- why a weak ranker helps: 7 min
- orthogonality checks: 3 min
- the two-stage blend: 2 min
- false-positive gates and the scorecard: 9 min
- limits, generalising and the checklist: 4 min
- Q&A: 7 min

Every code cell runs in seconds on a CPU, offline and deterministic. No LLM is called live. Keep
the rendered `.slides.html` as a fallback in case the kernel fails.
