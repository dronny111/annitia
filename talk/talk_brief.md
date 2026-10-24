# Talk brief

Read by the `talk-coach` skill. Edit this file when a decision changes.

Title: LLM Risk Rankers and Agentic Scaffolding for Liver Disease Survival Modeling (ANNITIA 2026 case study)

Speaker: Dr. Ronny Polle, he/him. Bio in `speaker_bio.md`.

Audience: GAIN (Ghana AI Research Network) Monthly Webinar. African researchers, clinicians, and engineers with mixed ML depth. Define MASLD, C-index, HeLM, XGBoost, CatBoost, CV, and LLM on first use.

Slot: 30 minutes of talk, then 10 minutes of Q&A. Platform: Google Meet. The chat works, so ask a GAIN host or co-host to read out two or three answers while the speaker keeps presenting.

Throughline:
- Feel: "I can check this myself, and my judgment already counts."
- Best message (technical): a weak LLM looked like a win on a 127-patient leaderboard. A leak-free replication showed it was not, and cheap checks told us why.
- Big message (empowerment): rigorous clinical AI does not need a bigger model or a bigger budget. It needs a good check, and you can run one yourself.

Sources of truth:
- `annitia_talk.py` (slide source) and `annitia_talk.ipynb` (executed outputs, the numbers), `talk_abstract.md`, `data/*.csv` (aggregate scores only), `data/CHALLENGE_SUBS.md` (upload history: 280 uploads, 272 scored, 8 scoring errors).
- Unreliable: `talk_review_gemini.md`, an LLM-written review. It presents a simulated +0.0098 inflation as the effect of the 272 uploads, says the blend "lost in 9 of 10 folds" (true for hepatic folds only), claims zero data left the infrastructure (GPT-OSS ran on Ollama's cloud), and uses a 50-patient pilot where the talk says three.

Lens: agentic harness framing, from the YC Paper Club talk "Why The Harness Matters More Than The Model". Thread through the talk: scaffolding around the model, persistent state, pre-flight probes (Spearman rho and score granularity), local execution (MedGemma on a laptop), and verification gates. Say "error correlation", not "error orthogonality". Cite the source by its title only.

Data and privacy:
- All data in the talk is synthetic, including the 1,253 training patients GPT-OSS scored (speaker confirmed, 2026-10-10). The deck itself uses only the seeded synthetic cohort (`synthetic.py`) and aggregate scores, and `guard.py` enforces it. Say "synthetic" on the slides.
- Scores are rankings, not risks. Never imply clinical readiness.
- MedGemma ran locally. GPT-OSS ran on Ollama's cloud endpoint.
- `assets/` holds photos of the speaker and colleagues, not patients. Do not use them as patient images. They stay out of the talk repo (`talk/.gitignore`).

Contact and links: polleronny2@gmail.com (public, on the closing slide). Talk repo: separate repo, [add the link after pushing]. Paste the link into the Meet chat at the close.

Decisions already made:
- Presentation deck: `annitia_talk_recap.pptx`, built by `pptx/build_deck.js` (headlines and notes from the track, numbers and images from `pptx/figures.py`). The original `annitia_talk.pptx` stays unchanged. Light background, Calibri, Okabe-Ito colours, text of 18 pt or more, backups hidden. The speaker presents it from Keynote, so the build repairs the charts for Keynote (2026-10-24).
- Pilot: score three patients first, as a smoke test. Three patients are the same Patients A, B and C shown on the data slide (`patient_cards.py`).
- Patient A is the recurring case: data slide, tree slide (her path through the first tree), serialization slide, CV leak slide.
- Show and tell (2026-10-23): a model that appears on a slide is drawn with its own visual. The tree models get a real fitted tree (slide 7) and a tree icon wherever they appear; the LLM gets a note icon.
- Say "272 scored uploads", not "272 uploads" (the platform lists 280 uploads, 8 of which failed to score).
- Chat question after the hook slide, paid off on the resolution slide.
- Slide headlines in assertion style. Times come from `track_check.py`, targeting about 26 of 30 minutes.
- Three parts, not five (map slide and footers): an LLM can help if its mistakes differ; three tests show our gain was not real; you can run the same checks yourself.
- Recaps (2026-10-24): slide 14 recaps part 1 and slide 21 recaps part 2. Each begins with four muted statements, then sharpens one statement per click using the deck's Keynote-compatible fade animation.
- Three tests, not five gates, in the spoken talk and the pptx: bigger than the noise? (resolution, multiplicity), too good to be true? (mechanism, ceiling), does it replicate? The HTML deck lists the five criteria under these three tests.
- Echo lines: "A second opinion only helps if its mistakes differ." (part 1, added 2026-10-21) / "A model that reads follow-up duration isn't predicting disease. It's reading the label." / "A gain you can't replicate isn't a gain." / "A better check beats a bigger model."
- Four doors (2026-10-21): one shared aha moment, that the calendar beat the liver (span alone 0.9614, above the CV as run, 0.9502), plus one for each group: clinicians, model builders, both, and neither. The table is in the track's Throughline. The closing slide gives each group one action.
- Learning moves come from the `learn` skill: one question at a time, with the answer after it; guided discovery (data slide to cohort slide); predict then reveal (a worse/same/better guess in the chat at the end of the mechanism slide, read out before the click, paid off on the CV leak slide); recall of the three tests before the checklist.
- Visuals: new diagrams are drawn in a private storyboard artifact with BioRender briefs, then built natively in the pptx (harness, error correlation, resolution, CV leak and checklist slides). BioRender is not connected. A PNG made there can replace a native figure later.
- Rehearsal script: `speaker_script.md` and `speaker_script.pdf`, in the Beyond the Lump format, written from the track. Refresh it by hand after the track changes.

Open items:
- Add the repo link after pushing: paste it into the chat at the close, and optionally into the hidden placeholder on the checklist slide of `annitia_talk.py`.
- Recheck "about 35 minutes" for MedGemma and "about 1,000 requests a day" for free API tiers before presenting.
- Confirm the apor line on the cohort slide suits the audience.
- Optional: ask the GAIN host for the rough registrant mix, to name it in the opening.
- Add the talk date to the header of `speaker_script.md`.
