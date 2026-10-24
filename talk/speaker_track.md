# Speaker track: LLM Risk Rankers and Agentic Scaffolding for Liver Disease Survival Modeling

- **Trunk:** A weak LLM looked like a win on a 127-patient leaderboard. A leak-free replication showed it was not, and cheap checks told us why.
- **Branches:** (1) An LLM can help, if its mistakes differ (slides 5-14). (2) Three tests show our gain was not real (slides 15-21): bigger than the noise? too good to be true? does it replicate? (3) You can run the same checks yourself (slides 22-24).
- **Slides:** slides 1-4 state the trunk and ask the chat question; every later slide hangs off one branch and carries one assertion.

GAIN Monthly Webinar, on Google Meet. Plan: a 30-minute slot, then 10 minutes of Q&A. The script runs about 29 minutes 20 seconds, derived from word counts (see the timing check), which leaves about 40 seconds of slack for slower delivery and overruns. If you finish early, use the backup slides or give the time to Q&A.
Figures come from the saved outputs in `annitia_talk.ipynb` and the text of `annitia_talk.py`. Each visual is labelled measured, leaderboard, simulated, or synthetic. Items marked `[CONFIRM]` need you.
The Visual lines describe the recap pptx deck (`annitia_talk_recap.pptx`). The live HTML deck shows the same evidence in a different layout, and each Deck line names its section.

## Throughline

**Feel:** "I can check this myself, and my judgment already counts."

- **Best message (technical):** A weak LLM looked like a win on a 127-patient leaderboard. A leak-free replication showed it was not, and cheap checks told us why.
- **Big message (empowerment):** Rigorous clinical AI does not need a bigger model or a bigger budget. It needs a good check, and you can run one yourself.
- **Five Maxwell moves:**
  1. Open by serving: name the four groups in the room and promise each one check (opening slide).
  2. Ask them, then credit their answer: the chat question on the hook slide pays off on the resolution slide, and the prediction that ends the mechanism slide pays off on the CV leak slide.
  3. Make the subject visible: meet Patients A, B and C before any model. Patient A then walks the first tree, returns when her record is serialized, and A, B and C are the three patients you score first.
  4. Be vulnerable: our own CV fooled us (the two leak slides).
  5. Hand over a tool and invite one action: the checklist (closing slide).
- **Echo lines, each tied to a number:**
  - "A second opinion only helps if its mistakes differ." (the error-correlation slide: independent mistakes add up to +0.0064, an error correlation of 0.6 adds nothing)
  - "A model that reads follow-up duration isn't predicting disease. It's reading the label." (the CV leak slide)
  - "A gain you can't replicate isn't a gain." (the replication slide)
  - "A better check beats a bigger model." (the closing slide)
- **Four doors: one shared aha, then one per group.** Everyone: the best score came from reading the calendar, not the liver (follow-up span alone 0.9614, above the CV as run, 0.9502). Each group also gets its own aha:

| Group | Their aha | Lands on | Device | Take-home (closing slide) |
|---|---|---|---|---|
| Clinicians | A short record is a clinical signal, and a model can read it instead of the liver | Data slide, then the CV leak slide | Patient A; apor | Ask whoever built the model what it scores with the dates alone |
| Model builders | A prompt cannot hide a column, so the guard belongs in the harness; the LLMs share their mistakes (rank correlation about 0.8) | Harness slide; scorecard slide | A jury told to disregard; same-school second opinions | Run the span-only leak probe this week |
| Both | The three tests are the evidence rules you already use for trials | Resolution to replication slides | No drug is approved on five events | Run the checklist on one recent decision |
| Neither | Luck looks like skill when you try many times | Resolution and multiplicity slides | 272 fair coins; five patients carry 70% of the score | Ask the three questions of the next AI headline |

- **Learning moves (from the `learn` skill):** one question at a time, and the answer always follows. The audience guesses before the reveal (the CV leak slide), discovers the leak before it is named (data slide to cohort slide), and recalls the three tests before the speaker says them (takeaways slide to checklist).
- **Guard rails:** do not imply clinical readiness. Do not oversell local models: MedGemma ran on a laptop but ranked poorly (0.59 alone). Always say the patients on the slides are synthetic.

---

### Slide 1: Today you will leave with one check that catches a model reading the answers. (80 s)
**Deck:** Title slide.
**Visual:** Title slide: the talk title, subtitle, speaker and event on the left. On the right, three patient pictograms (one in the accent colour), each beside a follow-up lane with visit ticks and an end marker. Data: schematic.
**Caption:** ANNITIA 2026 challenge: predicting MASLD liver events and death.
**Hook (before the slide):** This talk is for you, not about me.
**Script:** Thank you for having me. Some of you treat patients. Some of you build models. Some of you do both, and some of you do neither. Whichever you are, you will leave with one check that catches a model reading the answers instead of the patient. `[PAUSE - 2s]` Our story focuses on MASLD—metabolic dysfunction-associated steatotic liver disease. You can think of it as a condition where excess fat builds up in the liver, often linked to diabetes or weight challenges. Our clinical goal was simple to state, but tough to solve: predict which patients will suffer a severe liver event or death. The catch? Patients don't visit the clinic on a neat schedule. Their records are like a book with missing chapters—uneven visits, long gaps, and tests taken at irregular times.
**[CONFIRM]:** Any personal greeting or thanks to the GAIN hosts. If the host can share the rough registrant mix, name it here in place of the four groups.

### Slide 2: Three questions: can an LLM help, did ours, and how would you check? (40 s)
**Deck:** "The argument, in one sentence" lists the steps in the live deck. The pptx groups them into three parts.
**Visual:** Three rows, each a thumbnail of that part's key slide beside its answer: 1 An LLM can help, if its mistakes differ. 2 Three tests show our gain was not real. 3 You can run the same checks yourself. Data: schematic.
**Caption:** Three parts, in the order we will take them.
**Hook (before the slide):** Here is the road we will take.
**Script:** Three questions today, one per row. First: can a weak language model help a strong one at all? Yes, if its mistakes differ. Second: did ours? Three tests say no. Third: how would you check your own work? That part is yours to take home. `[PAUSE - 2s]` I will tell you where we are as we go.

### Slide 3: An LLM ranks patients worse alone, yet seemed to improve our strong model. (45 s)
**Deck:** "The argument, in one sentence."
**Visual:** Flow diagram: Patient record (a pictogram and a follow-up lane, as on the title slide), then two rankers, each with the icon it keeps on later slides: tree models, "strong ranker", with a small tree, and an LLM, "weak, but different mistakes", with a small clinical note. Then a rank blend, and an accent-colour box "Seemed to improve". Data: schematic.
**Caption:** The LLM ranks worse alone, but its errors differ from the tree models'.
**Hook (before the slide):** If you remember one sentence from the opening, make it this one.
**Script:** The whole talk in one picture, left to right. The tree models read the table of measurements and rank patients well. The LLM reads the same record as text. It ranks worse, but its mistakes differ. Blend the two, and the leaderboard score seemed to improve. `[PAUSE - 2s]` I will show you why that was plausible, and how a careful repeat of the test overturned it.

### Slide 4: On the leaderboard, a weak LLM seemed to lift our pipeline to 0.904. (110 s)
**Deck:** The hook chart, before section 1.
**Visual:** Eight horizontal bars of weighted C-index, from our leaderboard milestones. Light bars are models alone: tabular base 0.894, Claude Haiku HeLM 0.753, EfficientNet-B0 0.844, GPT-OSS 120B 0.792. Orange bars are blends: base plus Haiku 0.898, base plus EfficientNet-B0 0.897, EfficientNet base plus GPT-OSS 0.902, final two-stage 0.904. The axis does not start at zero. Data: leaderboard.
**Caption:** Blended at a small rank weight, GPT-OSS gave the pipeline's best public score.
**Hook (before the slide):** Here is the number that looked like a win.
**Script:** The score on this chart is the C-index: 1 means every patient is in the right order, and 0.5 means random order. Blue bars are models alone. Orange bars are blends. Our tabular base alone scored 0.894. GPT-OSS, an open LLM with 120 billion parameters, scored 0.792 alone, far below it. Now look at the last orange bar. Blended in at a small weight, GPT-OSS gave this pipeline's best public score, 0.904. `[PAUSE - 2s]` A gap of 0.01 looks like progress, even on an axis that starts at random order. That is how it looked to us. Before I go on, a question for the chat. If a colleague showed you this result, what is the first thing you would ask? One line is enough. Type it in the Meet chat now, and I will come back to your answers shortly.
**Extra time:** +30 s for the chat.
**[CONFIRM]:** Ask a GAIN host or co-host to watch the Meet chat and read out two or three answers, so you can keep presenting. The chart is drawn from `leaderboard.csv`, and its title says "leaderboard". Add a personal anecdote here only if it is true.

### Slide 5: Because the metric only sees order, we blend ranks, not raw scores. (60 s)
**Deck:** Section 1, Objective.
**Visual:** Four bars of C-index on the synthetic cohort: a tabular score 0.7926, the same score exponentiated 0.7926, a raw 50/50 average of the tabular and LLM scores 0.7606, and a rank 50/50 average 0.7884. The raw-average bar is the accent colour. Data: synthetic.
**Caption:** Averaging raw scores lets the wider scale win. Averaging ranks weighs both fairly.
**Hook (before the slide):** The scoring rule decides the method.
**Script:** The challenge score is 0.7 times the C-index for liver events, plus 0.3 times the C-index for death. The C-index asks of every pair of patients we can compare: did the one with the earlier event get the higher risk? Like a race result, only the finishing order counts, not the times. Four bars, in two pairs. First: a score, and the same score stretched, give the same C-index, 0.7926. Second: average the raw scores, and the LLM's wider scale takes over, 0.7606. Average the ranks instead, and both models count: 0.7884. `[PAUSE - 2s]` So when the metric only sees order, blend ranks.

### Slide 6: Each patient's record has its own length, gaps and ending. (100 s)
**Deck:** Section 2, the new slide "Each patient's record has its own length, gaps and ending."
**Visual:** Three synthetic patients side by side, built by `talk/patient_cards.py`. Each has a pictogram (red for a hepatic event, blue for event-free), a FibroScan stiffness trajectory against age on one shared scale with the 12.5 kPa cirrhosis line, and a follow-up lane with one tick per visit and an end marker: a cross for the event, an open circle for "still event-free". Patient A: woman, 41, dyslipidaemia, 5 visits, FibroScan at 2, both above the cirrhosis line, hepatic event at 44. Patient B: man, 55, hypertension, 15 visits, FibroScan at 12, low readings, still event-free at 67. Patient C: man, 70, type 2 diabetes, 6 visits, FibroScan at 4, all above the cirrhosis line, still event-free at 75. Data: synthetic.
**Caption:** Visits: 5, 15 and 6. FibroScan at 2, 12 and 4 of them. One event.
**Hook (before the slide):** Before any model, meet three patients.
**Script:** Each patient is one row, with up to 22 visits, each with blood tests and liver stiffness, measured by FibroScan. Three synthetic patients, left to right, with the same columns as the real records. Patient A is a woman of 41. She has five visits, and FibroScan at only two of them. Both readings are above the cirrhosis line, where the liver is badly scarred. Then her record ends, because she has a hepatic event, a major liver event. `[PAUSE - 2s]` Patient B is a man of 55, with fifteen visits over about twelve years and low FibroScan readings at twelve of them. He is still event-free when follow-up ends. Patient C is a man of 70 with six visits. FibroScan at four, all above the cirrhosis line. He is also still event-free at his last visit, and we do not know what happens next. That is right censoring. Patient A's record is the shortest, because the event ended it. So what could a model learn from the length of a record alone? Hold that thought.

### Slide 7: Our tree models read Patient A's record as hundreds of small yes/no flowcharts. (65 s)
**Deck:** Section 3, the new subslide "Patient A, as the trees read her record."
**Visual:** Left: the first tree of the honest XGBoost model (first-visit features), drawn by `talk/tree_card.py` from the fitted model. Patient A's path is in her colour, the accent: FibroTest ≥ 0.64? yes; FibroTest ≥ 0.86? yes; BMI ≥ 31.6? no; type 2 diabetes? no; this tree: higher risk. The questions and ends she never meets are grey dots. Right: XGBoost, 250 trees, and CatBoost, 250 trees, each a small grid of tree icons, joining in a rank blend: 99% of the 600 test patients rank below Patient A. Data: synthetic.
**Caption:** Each tree asks a few yes/no questions. Added up, hundreds of them rank the patients.
**Hook (before the slide):** Before any LLM, see how our strong model reads Patient A.
**Script:** Our strong model is two boosted tree models, XGBoost and CatBoost. A tree is a flowchart of yes/no questions. Here is the first tree, with Patient A's path. Is her FibroTest above 0.64? Yes. Above 0.86? Yes. Two more questions, and this tree says higher risk. Grey dots are questions she never meets. `[PAUSE - 2s]` One tree is a weak guess. Boosting adds 250, each correcting the ones before it. CatBoost grows 250 more, and we blend the two by rank. Together, they rank Patient A above 99% of the test patients. This model sees only her first visit. What if it also saw how long her record runs?
**Time note:** If you run long, skip the CatBoost sentence.

### Slide 8: Follow-up stops at the event, so record length can leak the outcome. (95 s)
**Deck:** Section 2, the cohort plot and the leakage trap; section 3, The tabular base.
**Visual:** Two charts. Left: stacked columns of the number of visits per patient in the synthetic cohort, event-free in blue and hepatic event in the accent colour, with fewer visits where follow-up ends at the event. Right: three bars of hepatic C-index: a baseline-visit-only model 0.798, the true risk 0.825, and a model that sees the whole follow-up window 0.920. Data: synthetic.
**Caption:** A score that sees the whole follow-up window beats the true risk, so it reads the outcome.
**Hook (before the slide):** Patient A's record was short for a reason.
**Script:** Now the whole synthetic cohort. Two charts. On the left, the number of visits per patient. The orange bars are patients with a hepatic event, and they have fewer visits, because follow-up ends at the event. That is Patient A's story, many times over. So, to my question: the number of visits, and the length of follow-up, quietly encode the outcome. On the right, the same tree models. In synthetic data we know each patient's true risk, and ranking by it scores 0.825. That is the ceiling. The trees you just saw, limited to the first visit, score 0.798, just below it, as an honest model should. A model that sees the whole follow-up window scores 0.920, above the ceiling. A model that beats the truth is reading the answer. In Ghana we have a word for this: apor, the leaked exam paper. `[PAUSE - 2s]` I am telling you this now because we fell into this trap ourselves, and I will show you how we found out.
**[CONFIRM]:** Keep the apor line only if it suits the room. Listeners from outside Ghana get its meaning from "the leaked exam paper".

### Slide 9: We turn each patient into a clinical note and ask one question per endpoint. (90 s)
**Deck:** Section 4, Serializing a patient ("Patient A, as the LLM reads the record") and Split endpoints.
**Visual:** An excerpt of Patient A's real clinical note: age span and visits, comorbidity, FibroScan "n=2, first=15.18 kPa, last=15.22 kPa, stable", platelets "first=180.72 G/L, last=129.49 G/L, falling", and ALT "rising", with reference thresholds in blue and direction words in the accent colour. Beside it, two boxes, "Hepatic prompt" and "Death prompt", each returning a 0-100 score. Data: synthetic.
**Caption:** HeLM idea: a language model reads the record as text and returns a risk ranking.
**Hook (before the slide):** The model never sees the spreadsheet. It reads a clinical note.
**Script:** HeLM is a simple idea: give a language model a patient's record as text, and read a risk out of it. It comes from Belyaeva, Cosentino and colleagues, in 2023. On the left is Patient A again, as the model sees her: not a spreadsheet, but a short clinical note. FibroScan reads 15.18, then 15.22: stable. Platelets are falling, and the last value is below the 150 threshold. The note adds direction words, like rising and falling, normal ranges, and her other conditions. On the right, two prompts: one question per endpoint, meaning per outcome. Liver events are driven by fibrosis, the scarring of the liver. Death is driven mostly by age, and by heart and metabolic disease. One combined prompt blurs the two, so each outcome gets its own prompt and its own score from 0 to 100. `[PAUSE - 2s]` We treat those scores as rankings, never as probabilities. A C-index needs nothing more.

### Slide 10: A fixed harness, not an autonomous agent, decides what the model may see. (90 s)
**Deck:** Section 4, The scaffolding around the LLM.
**Visual:** Top: a pipeline of five stages with a one-line note under each: Leak guard (drops follow-up length, struck through in the accent colour), Prompt builder (one prompt per endpoint), Model call (one prompt in, one JSON out), Retry and rate limits (survives free-tier limits), and Score log (saves every score), drawn as a can for stored state. Bottom, dashed: a prompt-only guard. A "follow-up length" chip in the accent colour runs through a prompt that says "ignore follow-up length" and on into the model, labelled "still reaches the model". Data: schematic of `helm_train_scores.py`.
**Caption:** The model picks no tools and no next steps.
**Hook (before the slide):** Everything except the model call is plain code we wrote.
**Script:** Follow the top row: guard, prompt, one model call, retry, log. Only the blue box is the model. It sees one prompt and returns one JSON object. The rest is the harness. It decides what the model may see, keeps a long run alive on free API tiers, and saves every score, so a crash costs nothing. It is a fixed pipeline, not an autonomous agent: the model picks no tools and no next steps. `[PAUSE - 2s]` So put the guards in the harness, not in the prompt. Look at the dashed row: a prompt that says "ignore follow-up length" still contains it. That is like telling a jury to disregard what it just heard. It cannot unhear it. In the top row, the guard drops it before the prompt is written, so the model never sees it. One note on privacy: GPT-OSS ran on Ollama's cloud, so that text left the machine. The data here is synthetic; with real records, that matters.

### Slide 11: A weak ranker helps only when its mistakes differ from the strong model's. (80 s)
**Deck:** Section 5, Why a weak ranker helps.
**Visual:** Line chart of C-index against the weight on the LLM ranks, one line per error correlation (0, 0.3, 0.6). In place of a legend, a pair of overlapping circles beside the chart for each line, in that line's colour: small overlap, "mistakes differ" (0); medium, "some shared" (0.3); large, "mostly shared" (0.6). Independent mistakes rise above the base. Shared mistakes fall below it. Data: synthetic; the circles are schematic.
**Caption:** Independent mistakes: a weak model adds signal. Shared mistakes: it only adds noise.
**Hook (before the slide):** A weak model can still earn its place.
**Script:** Think of a second opinion. It helps only if the second doctor can catch what the first one missed. A ranker is any model that puts patients in order of risk. Take two simulated rankers of the same true risk: a strong tabular one, and a weaker LLM one with about 20 distinct scores, like real LLM scores. We turn one knob: how often they make the same mistakes. Each line is one setting, with its overlap drawn beside it. When the mistakes differ, the line climbs above the base as the LLM gets more weight. When they are shared, it only falls. A second opinion only helps if its mistakes differ. `[PAUSE - 2s]` So judge a new model by what it adds to the blend, not by its score alone. Judged alone, our LLM would have been thrown away.

### Slide 12: Two cheap checks show early whether a new ranker is worth its API calls. (55 s)
**Deck:** Section 6, Check rank correlation before spending API calls.
**Visual:** Left: Spearman rho between LLM and base ranks against best blend gain, falling as rho rises. Right: bars of unique scores for 5, 20 and 100 levels (6, 21 and 101 unique values). Data: synthetic.
**Caption:** Lower rank correlation means more new information.
**Hook (before the slide):** Measure before you spend.
**Script:** Scoring every patient costs two LLM calls, so test cheaply first. Start with a smoke test on Patients A, B and C: does the prompt run, does the JSON parse, do the scores differ? Then two checks, one per chart. On the left, the Spearman rank correlation: how closely the second opinion agrees with the first. The higher it is, the less the blend gains. On the right, count the distinct scores. Five score levels give only six distinct values, and a ranker with a handful of scores ties most pairs.
**[CONFIRM]:** Decided: score three patients first. Three patients are a smoke test, not an estimate: a Spearman correlation on three patients can only be 1, 0.5, -0.5 or -1. Say "the correlation needs a bigger first batch" if someone asks. In the synthetic demo, score granularity cost little (LLM alone 0.743 at 5 levels, 0.752 at 100), so correlation is the bigger lever.

### Slide 13: We submitted the LLM inside a two-stage rank blend, at a small weight. (25 s)
**Deck:** Section 7, The blend we submitted.
**Visual:** Two-stage diagram: EfficientNet-B0 (image + tabular, with an icon of image tiles) and GPT-OSS split-endpoint scores (the note icon) feed a Stage 1 50/50 rank blend. Stage 1 (weight 0.15) and the 10-seed tabular base (the tree icon, weight 0.85) feed a Stage 2 rank blend, then the final score. A large 0.075 states the LLM's effective weight. Data: recipe, weights computed from the submitted recipe.
**Caption:** Both stages are rank blends, so the LLM's effective weight is about 0.075.
**Hook (before the slide):** This is what we actually submitted.
**Script:** This is what we submitted: GPT-OSS blended with an image model, then into our tabular base, by rank both times. So the LLM's effective weight was about 0.075. Small.

### Slide 14: Part 1 recap: why an LLM could help. (25 s)
**Deck:** Part 1 recap.
**Visual:** Four muted statements are visible from the start. On each click, the next statement becomes crisp and gains a numbered blue badge: irregular MASLD records; outcome leakage through record length; removing leaked time information in the harness; and the need for different ranking errors. Data: recap of Part 1.
**Caption:** None.
**Hook (before the slide):** Before we test the apparent win, pause on what Part 1 established.
**Script:** MASLD records are irregular, and events can shorten them. That makes follow-up length a leak unless the harness removes it before prompting. A weak LLM can still help, but only when its ranking errors differ. Together, this evidence made the small-weight blend plausible. Part 2 tests whether the apparent gain was real.

### Slide 15: A 127-patient public leaderboard cannot tell a gain of 0.0056 from noise. (105 s)
**Deck:** Section 8, Gates that say yes too easily; 1 Resolution.
**Visual:** Left: the public leaderboard as 127 pictograms, 5 of them filled in the accent colour for the liver events expected at the measured 4% rate, the rest outlined in grey. Right: bars comparing our observed gain (+0.0056, short bar) with the credible gain on three samples: public n=127 (0.0111), private n=296 (0.0059), whole test n=423 (0.0048). Data: simulated from measured event rates (hepatic 4%, death 8%), with our observed leaderboard gain.
**Caption:** Our gain was one standard error. A credible gain needed two.
**Hook (before the slide):** So was the 0.904 real? Let me go back to your question.
**Script:** Earlier I asked what you would ask first. If you wrote "how big was the test set?" or "how many times did you try?", you were already running test one. If you wrote "was there a leak?", that is test two, and for us it mattered most. We held every change to three tests. Test one: is the gain bigger than the noise? On the left, the public leaderboard: 127 patients, and only about five liver events. Seventy percent of the score rests on how those five are ranked. No one would approve a drug on a trial with five events. On the right, our gain against the gain you would need on each sample. A standard error is the size of a typical chance swing, and a credible gain is two of them, about 0.011 here. Our gain from adding GPT-OSS is the short bar: 0.0056, one standard error. `[PAUSE - 2s]` Even on the 296 private patients, you would need 0.0059.
**Extra time:** +15 s if you read out chat answers.
**Note:** Why 70%: the score is 0.7 times the hepatic C-index plus 0.3 times the death C-index, and every hepatic pair the C-index can compare includes one of the patients with an event. Five is expected, not observed: 127 times the training rate of 47 in 1,253 is just under five. The public split's own count is hidden.

### Slide 16: With 272 tries on one sample, the best result looks good by chance alone. (65 s)
**Deck:** Section 8, 2 Multiplicity.
**Visual:** Four bars: our gain 0.0056 in the accent colour, and the typical best gain by chance for 5, 30 and 272 tries (0.0100, 0.0145, 0.0186). Data: simulated, with our observed leaderboard gain.
**Caption:** The best of many tries looks good by construction.
**Hook (before the slide):** Noise also grows with how many times you look.
**Script:** Test one has a second part: how many times did we look? Flip 272 fair coins ten times each. The luckiest coin will almost always land heads at least nine times. It is not a magic coin. It is the best of 272. Leaderboards work the same way. Four bars: our gain on the left, then the best gain that chance alone gives you, as the number of tries goes up. With five tries, it is 0.0100. With thirty, 0.0145. With 272, our scored uploads, it is 0.0186. `[PAUSE - 2s]` Our gain sits below all three. A benchmark win below the noise floor is not a win.
**Note:** The coin line: one coin shows 9 or more heads in 10 flips with chance 11 in 1,024, so all 272 miss with chance (1 - 11/1024)^272, about 5%. Submission seven (best public score) moved from the script to the likely questions. `data/CHALLENGE_SUBS.md` lists 280 uploads, of which 272 were scored (8 scoring errors). Say "scored uploads", not "uploads". The deck line was corrected to match. Submission seven is not the LLM blend, whose final upload scored 0.904 (`leaderboard.csv`). `data/CHALLENGE_SUBS.md` gives it the same listed score, 0.93081, as `submission_cv_0`, uploaded the same day, and no file in the repo says which pipeline made either. If asked, say "a different pipeline, not the LLM blend".

### Slide 17: Our gain was bigger than even fully independent mistakes could deliver. (110 s)
**Deck:** Section 8, 3 Mechanism, and the Ceiling part of 4 Ceiling and 5 Replication.
**Visual:** Left: predicted best blend gain against error correlation (0.0: +0.0064, 0.3: +0.0020, 0.6: 0.0000). Right: paired bars of observed against predicted-at-best gain for two blends: Haiku into base (+0.0046 against +0.0023), GPT-OSS into the EfficientNet base (+0.0056 against +0.0037). Data: left synthetic, right leaderboard against model prediction.
**Caption:** When reality beats the best case, suspect the measurement, not the model.
**Hook (before the slide):** Some gains are too good for the mechanism.
**Script:** Test two: is the gain too good to be true? Start with the mechanism: how much can a second model add at most? Two charts. On the left, a simulation: with independent mistakes, the best blend gain is about 0.006. At an error correlation of 0.3, about 0.002. At 0.6, nothing. On the right, our two blends. Adding GPT-OSS at weight 0.07, even with fully independent mistakes, should give about 0.0037. We saw 0.0056. `[PAUSE - 2s]` When reality beats the best case, suspect the measurement, not the model. A score above what is possible is a leak, not a win. Before I show you ours, a prediction for the chat. Picture a model that sees only one thing per patient: the gap between the first and the last visit. No blood tests, no scans. Will it rank patients worse than our whole model, about the same, or better? Type worse, same or better.
**Extra time:** +20 s: wait for the guesses, and have the host read two or three before you click. The next headline gives the answer away.

### Slide 18: Our own cross-validation was a false positive, because follow-up length alone scored 0.96. (100 s)
**Deck:** Section 8, 4 Ceiling and 5 Replication; "Our CV was a false-positive gate."
**Visual:** Horizontal bars of weighted C-index: public leaderboard 0.8938, window-free CV 0.9020, the red-flag threshold 0.947 in grey, CV as run 0.9502, and follow-up span only (no biology) 0.9614. The two leaky bars are in the accent colour. Data: measured CV and leaderboard.
**Caption:** Drop the 26 window features and the same base falls from 0.950 to 0.902.
**Hook (before the slide):** The leak was not only on the leaderboard. It was in our own CV.
**Script:** If you typed better, you just found the leak in our pipeline. If you typed worse, that is the sensible guess, and it is wrong. Look at the bottom bar: the gap between the first and last visit, with no biology at all, scored 0.9614. `[PAUSE - 2s]` That beats our whole model. A model that reads follow-up duration isn't predicting disease. It's reading the label. Patient A's short record is exactly what it reads. How did we get here? Cross-validation, or CV, tests a model on patients it did not train on: split the training data into ten folds, hold one out, train on the rest, and repeat. Our tabular base scored 0.9502 in CV, but 0.8938 on the leaderboard. And 0.9502 is above our red flag, 0.947, the score above which we suspect a leak. So we ran the probe you just predicted. This was our own pipeline and our own mistake. We dropped every feature computed over the follow-up window, 26 in all. The same base, now leak-free, scores 0.9020, below the red flag and within 0.008 of the leaderboard.

### Slide 19: Adding GPT-OSS slightly lowers the leak-free cross-validation score, so the gain does not replicate. (80 s)
**Deck:** Section 8, "A cheap replication check" and "The real replication: GPT-OSS inside the window-free CV."
**Visual:** Left: bars of window-free CV: base 0.9020, GPT-OSS alone 0.7465, and base plus GPT-OSS 0.8999 in the accent colour. Right: bars of the folds where the blend beat the base, out of 10: hepatic 1, death 2, and a synthetic cohort with real signal 9. Data: measured window-free 10-fold CV, with the synthetic contrast labelled.
**Caption:** Weight 0.07, fixed in advance: the blend loses in most folds. A synthetic cohort with real signal wins 9.
**Hook (before the slide):** Test three is replication, and this is the replication that mattered.
**Script:** Test three, replication: does the gain show up on data the leaderboard never saw? GPT-OSS scored all 1,253 training patients zero-shot, with no examples in the prompt. It saw only what the leak-free base sees: no follow-up length, no visits after the event. The blend weight is 0.07, the one we submitted, fixed in advance. Two charts. On the left, the leak-free CV: base 0.9020, base plus GPT-OSS 0.8999. On the right, the folds. The blend beat the base in just one hepatic fold and two death folds. For contrast, in a synthetic cohort where an LLM does add real signal, the same weight wins nine of ten. A gain you can't replicate isn't a gain. `[PAUSE - 2s]` One caveat: this is not an exact replay of the leaderboard run. The inputs are stricter, and the base is the tabular model.

### Slide 20: No model or prompt clears the noise, and the LLMs agree with each other. (80 s)
**Deck:** Section 8, "One prompt or two?" and "The scorecard."
**Visual:** Dot-and-whisker plot of the window-free CV, one row per model and prompt: GPT-OSS split, Nemotron joint with reasoning, MedGemma split, MedGemma joint. Each dot is the blend gain (-0.0018, +0.0010, -0.0038, -0.0025) and each whisker spans two standard errors (SE 0.0020, 0.0017, 0.0022, 0.0025). A dashed line marks zero, and every whisker crosses it. Nemotron, the only gain above zero, is in the accent colour with its value. The C-index alone (0.747, 0.789, 0.590, 0.616) is spoken, not shown. Data: measured window-free 10-fold CV.
**Caption:** The LLMs agree with each other: rank correlation about 0.8.
**Hook (before the slide):** Maybe a better prompt or a stronger model would change the answer.
**Script:** Four rows, one per model and prompt. Split means one prompt per outcome; joint, one for both. Each dot is what that LLM adds to our base, and each bar spans two standard errors, our bar for a credible gain. The dashed line is zero, and every bar crosses it. `[PAUSE - 2s]` Reasoning helps the LLM alone: Nemotron 3 Super, which works step by step before it answers, is the strongest, at 0.789. Its dot is the only one right of zero, but its bar still crosses it. And the LLMs agree with each other, at a rank correlation of about 0.8: second opinions from doctors trained in the same school. Swapping one for another changes little. The weak-ranker idea is real. Here, the LLM's mistakes were too close to the base's for it to help.

### Slide 21: Part 2 recap: why the gain did not hold. (25 s)
**Deck:** Part 2 recap.
**Visual:** Four muted statements are visible from the start. On each click, the next statement becomes crisp and gains a numbered vermillion badge: the gain within noise after many uploads; the observed lift above the mechanism's best case; the follow-up-span leak probe; and the leak-free comparison of base against blend. Data: recap of Part 2.
**Caption:** None.
**Hook (before the slide):** The audit changed the conclusion.
**Script:** The public gain was within noise after many tries. It was larger than the mechanism predicted, follow-up span exposed the leak, and the leak-free blend scored below the base. The apparent win did not hold. The audit changed the conclusion, so Part 3 converts the findings into practical habits.

### Slide 22: The limits mean we report rankings, not risks, and quote the leak-free 0.902. (60 s)
**Deck:** Section 9, Limits and honest reporting.
**Visual:** Left: a two-row table, "Limit" and "What it means": rankings, not risks; cost and drift. Right: horizontal bars titled "Limit 3: which score to quote": CV as run (leaky) 0.9502 in the accent colour, leak-free CV 0.9020, public leaderboard 0.8938. Data: measured CV and leaderboard.
**Caption:** Quote the leak-free 0.902, not the leaky 0.950.
**Hook (before the slide):** Here is what this result does not show.
**Script:** Two limits in the table, one in the chart. First, the LLM scores rank patients. They are not risks, and we never report them as risks. Second, two calls per patient cost money, and a hosted model can change under you, so store the raw scores and the model tag with every run. Third, in the chart: our own CV read follow-up length and scored 0.950. The number to quote is the leak-free one, 0.902. `[PAUSE - 2s]` None of this says LLMs are useless here. It says that here, this one did not help.

### Slide 23: Cheap habits, not big budgets, make clinical AI results trustworthy. (80 s)
**Deck:** Takeaways.
**Visual:** Three columns, each a numbered habit with its evidence beneath and one line under that. Start local: a large "35 min", with "1,253 patients, on a laptop". Test cheaply: a small line chart, best blend gain falling as the Spearman rank correlation with the base rises (synthetic rankers), with "Three patients first". Hunt for leaks: small horizontal bars, follow-up span alone 0.9614 and CV as run 0.9502 in the accent colour, leak-free CV 0.9020, with "Span alone beat the whole model". Data: measured run time, synthetic rankers, measured CV.
**Caption:** None on the slide; the line under each column does that job.
**Hook (before the slide):** Take one of these home.
**Script:** One habit per column, each with its evidence beneath it. One: start with a model you can run yourself, and know where your data goes. MedGemma scored all 1,253 patients on a laptop in about 35 minutes, and no record left the machine. To be fair, it ranked poorly here, 0.59 alone, so I am saying it is feasible, not that it is good. Cloud and free endpoints may log prompts, so check the data terms before you send patient records. Two: test cheaply before you pay. Score three patients, then check the rank correlation and the number of distinct scores. Three: look for leaks in your own features first. The gap between the first and last visit alone scored 0.96, above the whole model. Before I click, one question for you: what were the three tests? `[PAUSE - 2s]`
**[CONFIRM]:** Recheck the 35 minutes before you present. The free-tier quota (about 1,000 requests a day) moved to the likely questions; recheck it too.

### Slide 24: You can audit your own model with one checklist, starting this week. (95 s)
**Deck:** The checklist.
**Visual:** Top: a band of the three tests, each with a numbered badge: 1 Bigger than the noise? 2 Too good to be true? 3 Does it replicate? Below: the ten checklist items in a two-column table with check boxes, and the line "Questions or results: polleronny2@gmail.com". Data: schematic.
**Caption:** Pick one recent decision and run it through this list.
**Hook (before the slide):** Here are the three tests, and the checklist behind them.
**Script:** Is the gain bigger than the noise? Is it too good to be true? Does it replicate? You already ask these in your own work. The 120-billion-parameter model gave us the gain that did not replicate. A better check beats a bigger model. `[PAUSE - 2s]` So, one action each. If you treat patients, ask whoever built the model: what does it score with the dates alone? If you build models, run that probe this week. If you do neither, ask the three questions of the next AI headline you read. Whichever you choose: if the result survives, you have earned the right to trust it. If it does not, you have caught something before it reached a paper or a patient. If you try it, I would love to hear what you find. My email is polleronny2 at gmail dot com, and it is on the slide. I will paste the link to the repo, with the code and the checklist, in the chat. Thank you. I will take your questions.
**Note:** Keep this slide up during the questions.
**[CONFIRM]:** Paste the repo link into the Meet chat as you close. The deck has a hidden placeholder for the link: add it after you push, if you also want it on the slide.

---

## Timing check

Each slide's time is its word count at 120 words per minute, plus 10 s to point at the visual and 2 s per pause, rounded to 5 s, plus any extra time noted. Refresh the table with `python3 ~/.claude/skills/talk-coach/scripts/track_check.py talk/speaker_track.md --sources talk/annitia_talk.py talk/annitia_talk.ipynb --slot-min 30 --write`.

<!-- timing:start -->
| Slide | Header | Words | Seconds |
|---|---|---|---|
| 1 | Today you will leave with one check that catches a model reading the answers. | 131 | 80 |
| 2 | Three questions: can an LLM help, did ours, and how would you check? | 55 | 40 |
| 3 | An LLM ranks patients worse alone, yet seemed to improve our strong model. | 64 | 45 |
| 4 | On the leaderboard, a weak LLM seemed to lift our pipeline to 0.904. | 139 | 110 |
| 5 | Because the metric only sees order, we blend ranks, not raw scores. | 101 | 60 |
| 6 | Each patient's record has its own length, gaps and ending. | 174 | 100 |
| 7 | Our tree models read Patient A's record as hundreds of small yes/no flowcharts. | 107 | 65 |
| 8 | Follow-up stops at the event, so record length can leak the outcome. | 165 | 95 |
| 9 | We turn each patient into a clinical note and ask one question per endpoint. | 153 | 90 |
| 10 | A fixed harness, not an autonomous agent, decides what the model may see. | 161 | 90 |
| 11 | A weak ranker helps only when its mistakes differ from the strong model's. | 138 | 80 |
| 12 | Two cheap checks show early whether a new ranker is worth its API calls. | 91 | 55 |
| 13 | We submitted the LLM inside a two-stage rank blend, at a small weight. | 29 | 25 |
| 14 | Part 1 recap: why an LLM could help. | 55 | 25 |
| 15 | A 127-patient public leaderboard cannot tell a gain of 0.0056 from noise. | 160 | 105 |
| 16 | With 272 tries on one sample, the best result looks good by chance alone. | 105 | 65 |
| 17 | Our gain was bigger than even fully independent mistakes could deliver. | 151 | 110 |
| 18 | Our own cross-validation was a false positive, because follow-up length alone scored 0.96. | 180 | 100 |
| 19 | Adding GPT-OSS slightly lowers the leak-free cross-validation score, so the gain does not replicate. | 139 | 80 |
| 20 | No model or prompt clears the noise, and the LLMs agree with each other. | 134 | 80 |
| 21 | Part 2 recap: why the gain did not hold. | 50 | 25 |
| 22 | The limits mean we report rankings, not risks, and quote the leak-free 0.902. | 92 | 60 |
| 23 | Cheap habits, not big budgets, make clinical AI results trustworthy. | 137 | 80 |
| 24 | You can audit your own model with one checklist, starting this week. | 169 | 95 |
| | **Total** | 2880 | **1760 (29 min 20 s)** |

Slot: 30.0 min. Slack: 40.0 s.
<!-- timing:end -->

## Backup slides (use for Q&A, not in the main run)

- **The 50/50 blend (deck: "Another decision the leaky gate approved").** The leaky CV approved the 50/50 XGBoost and CatBoost blend on a margin of 0.0008. Window-free, XGBoost alone looks better: fold mean +0.0155 ± 0.0056 (2.8 SE), patient bootstrap +0.0103 ± 0.0070 (1.5 SE), and XGBoost beats the blend in 6 of 10 hepatic and 5 of 10 death folds. We chose the comparison after seeing the scores. Suggestive, not credible.
- **Generalising the idea (deck: "Generalising the idea").** The same pattern in credit risk, search, and triage: strong base, candidate ranker, rank blend, reciprocal rank fusion, or ordered escalation.
- **What each column means (deck: section 2 subslide).** The data dictionary.

## Likely questions

1. **"So are LLMs dead for survival modeling?"** No. The mechanism is real. Here, the LLM's errors were too close to the base's, with an error correlation near 0.5, and at that correlation the expected change is about minus 0.002, which is what the replication found. A candidate with lower correlation could help. Judge it by its gain in the ensemble, on held-out data.
2. **"Could this go to a clinic?"** No. The scores are rankings, not risks, the patients here are synthetic, and the gain failed replication.
3. **"Was it OK to send the data to a cloud endpoint?"** The data is synthetic, including the 1,253 training patients GPT-OSS scored, so no real patient record was sent. For real patient data, check the data terms before you send anything.
4. **"Why rank blending?"** The C-index only sees order. Any monotone map of a score gives an identical C-index, so ranks combine models with incompatible outputs safely.
5. **"Which model should I run locally?"** MedGemma 1.5 (3.3 GB) scored 1,253 patients on a laptop in about 35 minutes. As a ranker it was weak here (0.59 split, 0.616 joint) and added nothing to the blend.
6. **"Why only three patients in the pilot?"** Three patients are a smoke test: does the prompt run, does the JSON parse, are the scores different? A correlation on three patients can only be 1, 0.5, -0.5 or -1, so check it on a bigger first batch before the full run.
7. **"How often is the public winner the real winner?"** In a simulation of thirty near-tied submissions, the public winner was the private winner only 5% of the time, and its public score was inflated by 0.0098 on average. That is a simulation, not the effect of our 272 scored uploads.
8. **"How do you catch a parser that fails quietly?"** Count your silent defaults. Every time the parser falls back to 50, that is a tie.
9. **"How would I spot this leak in a paper or a vendor's tool?"** Ask what the model scores with only the dates or the visit counts, and whether anything recorded after the prediction time was masked. Here the follow-up span alone scored 0.9614, above the whole model.
10. **"Should I trust AI in hospitals?"** Trust results, not names. Ask the three questions: is it bigger than the noise, is it too good to be true, and does it replicate on new patients? Our own gain failed the third.
11. **"Would a bigger model fix it?"** Not here. The 120-billion-parameter GPT-OSS gave the gain that did not replicate, and the LLMs agree with each other at a rank correlation of about 0.8, so swapping one for another changed little. Judge any model by its gain in the blend, on held-out data.
12. **"The leaderboard gives the same score every time. How is that noise?"** The measurement is fixed; the sample is not. The noise is in which 127 patients the board holds, and with about five liver events, a different five would reorder the board.
13. **Lines moved out of the script, for questions.** Best leaderboard scores: submission seven was best on the public board at 0.9308, and a different submission was best on the private board at 0.9301; neither is the LLM blend. Claude Haiku showed the same too-good pattern: observed +0.0046 against a best case of +0.0023. Free API tiers allow about 1,000 requests a day (recheck), so save after every call.

## Notes for the speaker

- **Headers are proposals.** They follow Alley's assertion style. The live HTML deck still uses topic headers, except the new patient slide. The HTML deck is 48 pages, so each slide here covers one or more deck pages, named in the Deck line. The pptx takes its headlines from this file.
- **Three parts, and three tests inside part 2.** The talk groups the HTML deck's five gates into three tests: noise (resolution, slide 15, and multiplicity, slide 16), too good to be true (mechanism, slide 17, and ceiling, slide 18), and replication (slides 19 and 20). Slide 21 recaps the audit before part 3. The HTML deck groups its five criteria under the same three tests, so the two decks agree.
- **Review file errors.** `talk_review_gemini.md` presents the simulated +0.0098 inflation (30 near-tied submissions) as the effect of the 272 uploads. It says the LLM "lost in 9 of 10 folds", which is true for hepatic folds only (death: 8 of 10). Its Module 3 says zero data leave the infrastructure, which is false for GPT-OSS on Ollama's cloud. Its pilot size is 50, where the talk says three patients. Do not quote the review without checking the notebook.
- **Before you push the talk repo.** The deck imports `annitia_helm_submit`, `helm_gptoss_blend_v2` and `push_score_allout` from the parent repo's `scripts/` folder, so the notebook will not run in a talk-only repo until you copy those in. The safety test also imports `feature_revolution` from `scripts/`. `talk/.gitignore` keeps `assets/` (photos of you and of other people) and `talk_review_gemini.md` out of the repo. `data/CHALLENGE_SUBS.md` lists all 280 uploads by file name, and the talk's safety test forbids those names in code (it does not scan `.md` files), so consider leaving that file out too. The README mentions it.
- **If you run long, cut in this order:** the blend slide (say only "we submitted it at a small weight, about 0.075"), the caveat on the replication slide, the trial line on the resolution slide, then the chat prediction on the mechanism slide (just reveal the bar). Never cut the data slide, the CV leak reveal or the close.
