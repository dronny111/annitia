# LLM Risk Rankers: speaker script

**GAIN Monthly Webinar · [CONFIRM: date] · 30-minute talk + 10-minute Q&A · Google Meet**

Dr Ronny Polle · deck: `annitia_talk_recap.pptx` · written from `speaker_track.md`, which stays the source for the deck's notes

Spoken length about 29:20 including both chat moments, which leaves about 0:40 of slack.
Patients A, B and C are synthetic, and the script says so on slide 6.

## Before you start

**Memorise these 6 lines word for word.** They are in bold in the script.

1. Some of you treat patients. Some of you build models. Some of you do both, and some of you do neither. Whichever you are, you will leave with one check that catches a model reading the answers instead of the patient.
2. A second opinion only helps if its mistakes differ.
3. A model that reads follow-up duration isn't predicting disease. It's reading the label.
4. A gain you can't replicate isn't a gain.
5. A better check beats a bigger model.
6. If you treat patients, ask whoever built the model: what does it score with the dates alone? If you build models, run that probe this week. If you do neither, ask the three questions of the next AI headline you read.

**The four doors.** Everyone gets the shared aha on slide 18: the calendar beat the liver. Each group also gets its own:

- Clinicians: Patient A's short record, her path through one tree, and apor (slides 6–8).
- Model builders: the jury that cannot unhear (slide 10) and the same-school LLMs (slide 20).
- Both: no drug is approved on five events (slide 15).
- Neither: 272 fair coins (slide 16).

**Read the room on slide 4.** If the chat answers are clinical (how many patients, was it validated), slow down on Patient A and the trial line. If they are technical (a leak, CV, the size of the test set), keep the pace and lean on the harness and the scorecard. Either way, credit the answers on slide 15.

**Checkpoints**

| Leaving | Target | Late if after |
|---|---|---|
| Slide 14, end of part 1 | 16:00 | 16:40 |
| Slide 21, end of part 2 | 25:25 | 26:35 |
| Slide 24, the close | 29:20 | 29:50 |

**Chat moments.** Ask the host to read out two or three answers each time.

- Slide 4: "If a colleague showed you this result, what is the first thing you would ask?" Paid off on slide 15.
- End of slide 17: "worse, same or better?" Read out before the click; paid off on slide 18.

**If you run long, cut in this order**

1. Slide 13: say only "We submitted it at a small weight, about 0.075" (saves about 10 s).
2. Slide 7: skip the CatBoost sentence (about 5 s).
3. Slide 19: the caveat that this is not an exact replay (about 10 s).
4. Slide 15: the trial line (about 6 s).
5. Slide 17: skip the chat prediction and just reveal the bar on slide 18 (about 40 s).

Never cut slides 6, 18 or 24.

**Key:** **bold** = memorise · ⏸ = two-second pause · *[italics]* = action · *[Click]* = next slide; inside a slide, click to bring in each next point

## 1 · Today you will leave with one check that catches a model reading the answers.

`0:00 → 1:20` · about 80 s

This talk is for you, not about me.

Thank you for having me. **Some of you treat patients. Some of you build models. Some of you do both, and some of you do neither. Whichever you are, you will leave with one check that catches a model reading the answers instead of the patient.**

⏸ *pause: two seconds*

Our story focuses on **MASLD**—metabolic dysfunction-associated steatotic liver disease. You can think of it as a condition where excess fat builds up in the liver, often linked to diabetes or weight challenges.

Our clinical goal was simple to state, but tough to solve: predict which patients will suffer a severe liver event or death. The catch? Patients don't visit the clinic on a neat schedule. Their records are like a book with missing chapters—uneven visits, long gaps, and tests taken at irregular times.

*[Click]*

<small>Data: schematic.</small>

## 2 · Three questions: can an LLM help, did ours, and how would you check?

`1:20 → 2:00` · about 40 s

Here is the road we will take.

Three questions today, one per row. First: can a weak language model help a strong one at all? Yes, if its mistakes differ. Second: did ours? Three tests say no. Third: how would you check your own work? That part is yours to take home.

⏸ *pause: two seconds*

I will tell you where we are as we go.

*[Click]*

<small>Data: schematic.</small>

## 3 · An LLM ranks patients worse alone, yet seemed to improve our strong model.

`2:00 → 2:45` · about 45 s

If you remember one sentence from the opening, make it this one.

The whole talk in one picture, left to right. The tree models read the table of measurements and rank patients well. The LLM reads the same record as text. It ranks worse, but its mistakes differ. Blend the two, and the leaderboard score seemed to improve.

⏸ *pause: two seconds*

I will show you why that was plausible, and how a careful repeat of the test overturned it.

*[Click]*

<small>Data: schematic.</small>

## 4 · On the leaderboard, a weak LLM seemed to lift our pipeline to 0.904.

`2:45 → 4:35` · about 110 s, including 30 s for the chat

Here is the number that looked like a win.

The score on this chart is the C-index: 1 means every patient is in the right order, and 0.5 means random order. Blue bars are models alone. Orange bars are blends. Our tabular base alone scored 0.894. GPT-OSS, an open LLM with 120 billion parameters, scored 0.792 alone, far below it. Now look at the last orange bar. Blended in at a small weight, GPT-OSS gave this pipeline's best public score, 0.904.

⏸ *pause: two seconds*

A gap of 0.01 looks like progress, even on an axis that starts at random order. That is how it looked to us. Before I go on, a question for the chat. If a colleague showed you this result, what is the first thing you would ask? One line is enough. Type it in the Meet chat now, and I will come back to your answers shortly.

*[Leave the question up for the chat. The host reads two or three answers on slide 15.]*

<small>Data: leaderboard.</small>

## 5 · Because the metric only sees order, we blend ranks, not raw scores.

`4:35 → 5:35` · about 60 s

The scoring rule decides the method.

The challenge score is 0.7 times the C-index for liver events, plus 0.3 times the C-index for death. The C-index asks of every pair of patients we can compare: did the one with the earlier event get the higher risk? Like a race result, only the finishing order counts, not the times. Four bars, in two pairs. First: a score, and the same score stretched, give the same C-index, 0.7926. Second: average the raw scores, and the LLM's wider scale takes over, 0.7606. Average the ranks instead, and both models count: 0.7884.

⏸ *pause: two seconds*

So when the metric only sees order, blend ranks.

*[Click]*

<small>Data: synthetic.</small>

## 6 · Each patient's record has its own length, gaps and ending.

`5:35 → 7:15` · about 100 s

Before any model, meet three patients.

Each patient is one row, with up to 22 visits, each with blood tests and liver stiffness, measured by FibroScan. Three synthetic patients, left to right, with the same columns as the real records. Patient A is a woman of 41. She has five visits, and FibroScan at only two of them. Both readings are above the cirrhosis line, where the liver is badly scarred. Then her record ends, because she has a hepatic event, a major liver event.

⏸ *pause: two seconds*

Patient B is a man of 55, with fifteen visits over about twelve years and low FibroScan readings at twelve of them. He is still event-free when follow-up ends. Patient C is a man of 70 with six visits. FibroScan at four, all above the cirrhosis line. He is also still event-free at his last visit, and we do not know what happens next. That is right censoring. Patient A's record is the shortest, because the event ended it. So what could a model learn from the length of a record alone? Hold that thought.

*[Click]*

<small>Data: synthetic.</small>

## 7 · Our tree models read Patient A's record as hundreds of small yes/no flowcharts.

`7:15 → 8:20` · about 65 s

Before any LLM, see how our strong model reads Patient A.

Our strong model is two boosted tree models, XGBoost and CatBoost. A tree is a flowchart of yes/no questions. Here is the first tree, with Patient A's path. Is her FibroTest above 0.64? Yes. Above 0.86? Yes. Two more questions, and this tree says higher risk. Grey dots are questions she never meets.

⏸ *pause: two seconds*

One tree is a weak guess. Boosting adds 250, each correcting the ones before it. CatBoost grows 250 more, and we blend the two by rank. Together, they rank Patient A above 99% of the test patients. This model sees only her first visit. What if it also saw how long her record runs?

*[Click]*

<small>Data: synthetic.</small>

## 8 · Follow-up stops at the event, so record length can leak the outcome.

`8:20 → 9:55` · about 95 s

Patient A's record was short for a reason.

Now the whole synthetic cohort. Two charts. On the left, the number of visits per patient. The orange bars are patients with a hepatic event, and they have fewer visits, because follow-up ends at the event. That is Patient A's story, many times over. So, to my question: the number of visits, and the length of follow-up, quietly encode the outcome. On the right, the same tree models. In synthetic data we know each patient's true risk, and ranking by it scores 0.825. That is the ceiling. The trees you just saw, limited to the first visit, score 0.798, just below it, as an honest model should. A model that sees the whole follow-up window scores 0.920, above the ceiling. A model that beats the truth is reading the answer. In Ghana we have a word for this: apor, the leaked exam paper.

⏸ *pause: two seconds*

I am telling you this now because we fell into this trap ourselves, and I will show you how we found out.

*[Click]*

<small>Data: synthetic.</small>

## 9 · We turn each patient into a clinical note and ask one question per endpoint.

`9:55 → 11:25` · about 90 s

The model never sees the spreadsheet. It reads a clinical note.

HeLM is a simple idea: give a language model a patient's record as text, and read a risk out of it. It comes from Belyaeva, Cosentino and colleagues, in 2023. On the left is Patient A again, as the model sees her: not a spreadsheet, but a short clinical note. FibroScan reads 15.18, then 15.22: stable. Platelets are falling, and the last value is below the 150 threshold. The note adds direction words, like rising and falling, normal ranges, and her other conditions. On the right, two prompts: one question per endpoint, meaning per outcome. Liver events are driven by fibrosis, the scarring of the liver. Death is driven mostly by age, and by heart and metabolic disease. One combined prompt blurs the two, so each outcome gets its own prompt and its own score from 0 to 100.

⏸ *pause: two seconds*

We treat those scores as rankings, never as probabilities. A C-index needs nothing more.

*[Click]*

<small>Data: synthetic.</small>

## 10 · A fixed harness, not an autonomous agent, decides what the model may see.

`11:25 → 12:55` · about 90 s

Everything except the model call is plain code we wrote.

Follow the top row: guard, prompt, one model call, retry, log. Only the blue box is the model. It sees one prompt and returns one JSON object. The rest is the harness. It decides what the model may see, keeps a long run alive on free API tiers, and saves every score, so a crash costs nothing. It is a fixed pipeline, not an autonomous agent: the model picks no tools and no next steps.

⏸ *pause: two seconds*

So put the guards in the harness, not in the prompt. Look at the dashed row: a prompt that says "ignore follow-up length" still contains it. That is like telling a jury to disregard what it just heard. It cannot unhear it. In the top row, the guard drops it before the prompt is written, so the model never sees it. One note on privacy: GPT-OSS ran on Ollama's cloud, so that text left the machine. The data here is synthetic; with real records, that matters.

*[Click]*

<small>Data: schematic of `helm_train_scores.py`.</small>

## 11 · A weak ranker helps only when its mistakes differ from the strong model's.

`12:55 → 14:15` · about 80 s

A weak model can still earn its place.

Think of a second opinion. It helps only if the second doctor can catch what the first one missed. A ranker is any model that puts patients in order of risk. Take two simulated rankers of the same true risk: a strong tabular one, and a weaker LLM one with about 20 distinct scores, like real LLM scores. We turn one knob: how often they make the same mistakes. Each line is one setting, with its overlap drawn beside it. When the mistakes differ, the line climbs above the base as the LLM gets more weight. When they are shared, it only falls. **A second opinion only helps if its mistakes differ.**

⏸ *pause: two seconds*

So judge a new model by what it adds to the blend, not by its score alone. Judged alone, our LLM would have been thrown away.

*[Click]*

<small>Data: synthetic; the circles are schematic.</small>

## 12 · Two cheap checks show early whether a new ranker is worth its API calls.

`14:15 → 15:10` · about 55 s

Measure before you spend.

Scoring every patient costs two LLM calls, so test cheaply first. Start with a smoke test on Patients A, B and C: does the prompt run, does the JSON parse, do the scores differ? Then two checks, one per chart. On the left, the Spearman rank correlation: how closely the second opinion agrees with the first. The higher it is, the less the blend gains. On the right, count the distinct scores. Five score levels give only six distinct values, and a ranker with a handful of scores ties most pairs.

*[Click]*

<small>Data: synthetic.</small>

## 13 · We submitted the LLM inside a two-stage rank blend, at a small weight.

`15:10 → 15:35` · about 25 s

This is what we actually submitted.

This is what we submitted: GPT-OSS blended with an image model, then into our tabular base, by rank both times. So the LLM's effective weight was about 0.075. Small.

*[Click]*

<small>Data: recipe, weights computed from the submitted recipe.</small>

## 14 · Part 1 recap: why an LLM could help.

`15:35 → 16:00` · about 25 s

Before we test the apparent win, pause on what Part 1 established.

*[Click once per point. Each muted line becomes crisp.]*

MASLD records are irregular, and events can shorten them. That makes follow-up length a leak unless the harness removes it before prompting. A weak LLM can still help, but only when its ranking errors differ. Together, this evidence made the small-weight blend plausible. Part 2 tests whether the apparent gain was real.

*[Click]*

<small>Data: recap of Part 1.</small>

## 15 · A 127-patient public leaderboard cannot tell a gain of 0.0056 from noise.

`16:00 → 17:45` · about 105 s, including 15 s for the chat

So was the 0.904 real? Let me go back to your question.

Earlier I asked what you would ask first. If you wrote "how big was the test set?" or "how many times did you try?", you were already running test one. If you wrote "was there a leak?", that is test two, and for us it mattered most. We held every change to three tests. Test one: is the gain bigger than the noise? On the left, the public leaderboard: 127 patients, and only about five liver events. Seventy percent of the score rests on how those five are ranked. No one would approve a drug on a trial with five events. On the right, our gain against the gain you would need on each sample. A standard error is the size of a typical chance swing, and a credible gain is two of them, about 0.011 here. Our gain from adding GPT-OSS is the short bar: 0.0056, one standard error.

⏸ *pause: two seconds*

Even on the 296 private patients, you would need 0.0059.

*[Click]*

<small>Data: simulated from measured event rates (hepatic 4%, death 8%), with our observed leaderboard gain.</small>

## 16 · With 272 tries on one sample, the best result looks good by chance alone.

`17:45 → 18:50` · about 65 s

Noise also grows with how many times you look.

Test one has a second part: how many times did we look? Flip 272 fair coins ten times each. The luckiest coin will almost always land heads at least nine times. It is not a magic coin. It is the best of 272. Leaderboards work the same way. Four bars: our gain on the left, then the best gain that chance alone gives you, as the number of tries goes up. With five tries, it is 0.0100. With thirty, 0.0145. With 272, our scored uploads, it is 0.0186.

⏸ *pause: two seconds*

Our gain sits below all three. A benchmark win below the noise floor is not a win.

*[Click]*

<small>Data: simulated, with our observed leaderboard gain.</small>

## 17 · Our gain was bigger than even fully independent mistakes could deliver.

`18:50 → 20:40` · about 110 s, including 20 s for the chat

Some gains are too good for the mechanism.

Test two: is the gain too good to be true? Start with the mechanism: how much can a second model add at most? Two charts. On the left, a simulation: with independent mistakes, the best blend gain is about 0.006. At an error correlation of 0.3, about 0.002. At 0.6, nothing. On the right, our two blends. Adding GPT-OSS at weight 0.07, even with fully independent mistakes, should give about 0.0037. We saw 0.0056.

⏸ *pause: two seconds*

When reality beats the best case, suspect the measurement, not the model. A score above what is possible is a leak, not a win. Before I show you ours, a prediction for the chat. Picture a model that sees only one thing per patient: the gap between the first and the last visit. No blood tests, no scans. Will it rank patients worse than our whole model, about the same, or better? Type worse, same or better.

*[Wait about 20 seconds for the guesses. Have the host read two or three, then click. The next headline gives the answer away.]*

<small>Data: left synthetic, right leaderboard against model prediction.</small>

## 18 · Our own cross-validation was a false positive, because follow-up length alone scored 0.96.

`20:40 → 22:20` · about 100 s

The leak was not only on the leaderboard. It was in our own CV.

If you typed better, you just found the leak in our pipeline. If you typed worse, that is the sensible guess, and it is wrong. Look at the bottom bar: the gap between the first and last visit, with no biology at all, scored 0.9614.

⏸ *pause: two seconds*

That beats our whole model. **A model that reads follow-up duration isn't predicting disease. It's reading the label.** Patient A's short record is exactly what it reads. How did we get here? Cross-validation, or CV, tests a model on patients it did not train on: split the training data into ten folds, hold one out, train on the rest, and repeat. Our tabular base scored 0.9502 in CV, but 0.8938 on the leaderboard. And 0.9502 is above our red flag, 0.947, the score above which we suspect a leak. So we ran the probe you just predicted. This was our own pipeline and our own mistake. We dropped every feature computed over the follow-up window, 26 in all. The same base, now leak-free, scores 0.9020, below the red flag and within 0.008 of the leaderboard.

*[Click]*

<small>Data: measured CV and leaderboard.</small>

## 19 · Adding GPT-OSS slightly lowers the leak-free cross-validation score, so the gain does not replicate.

`22:20 → 23:40` · about 80 s

Test three is replication, and this is the replication that mattered.

Test three, replication: does the gain show up on data the leaderboard never saw? GPT-OSS scored all 1,253 training patients zero-shot, with no examples in the prompt. It saw only what the leak-free base sees: no follow-up length, no visits after the event. The blend weight is 0.07, the one we submitted, fixed in advance. Two charts. On the left, the leak-free CV: base 0.9020, base plus GPT-OSS 0.8999. On the right, the folds. The blend beat the base in just one hepatic fold and two death folds. For contrast, in a synthetic cohort where an LLM does add real signal, the same weight wins nine of ten. **A gain you can't replicate isn't a gain.**

⏸ *pause: two seconds*

One caveat: this is not an exact replay of the leaderboard run. The inputs are stricter, and the base is the tabular model.

*[Click]*

<small>Data: measured window-free 10-fold CV, with the synthetic contrast labelled.</small>

## 20 · No model or prompt clears the noise, and the LLMs agree with each other.

`23:40 → 25:00` · about 80 s

Maybe a better prompt or a stronger model would change the answer.

Four rows, one per model and prompt. Split means one prompt per outcome; joint, one for both. Each dot is what that LLM adds to our base, and each bar spans two standard errors, our bar for a credible gain. The dashed line is zero, and every bar crosses it.

⏸ *pause: two seconds*

Reasoning helps the LLM alone: Nemotron 3 Super, which works step by step before it answers, is the strongest, at 0.789. Its dot is the only one right of zero, but its bar still crosses it. And the LLMs agree with each other, at a rank correlation of about 0.8: second opinions from doctors trained in the same school. Swapping one for another changes little. The weak-ranker idea is real. Here, the LLM's mistakes were too close to the base's for it to help.

*[Click]*

<small>Data: measured window-free 10-fold CV.</small>

## 21 · Part 2 recap: why the gain did not hold.

`25:00 → 25:25` · about 25 s

The audit changed the conclusion.

*[Click once per point. Each muted line becomes crisp.]*

The public gain was within noise after many tries. It was larger than the mechanism predicted, follow-up span exposed the leak, and the leak-free blend scored below the base. The apparent win did not hold. The audit changed the conclusion, so Part 3 converts the findings into practical habits.

*[Click]*

<small>Data: recap of Part 2.</small>

## 22 · The limits mean we report rankings, not risks, and quote the leak-free 0.902.

`25:25 → 26:25` · about 60 s

Here is what this result does not show.

Two limits in the table, one in the chart. First, the LLM scores rank patients. They are not risks, and we never report them as risks. Second, two calls per patient cost money, and a hosted model can change under you, so store the raw scores and the model tag with every run. Third, in the chart: our own CV read follow-up length and scored 0.950. The number to quote is the leak-free one, 0.902.

⏸ *pause: two seconds*

None of this says LLMs are useless here. It says that here, this one did not help.

*[Click]*

<small>Data: measured CV and leaderboard.</small>

## 23 · Cheap habits, not big budgets, make clinical AI results trustworthy.

`26:25 → 27:45` · about 80 s

Take one of these home.

One habit per column, each with its evidence beneath it. One: start with a model you can run yourself, and know where your data goes. MedGemma scored all 1,253 patients on a laptop in about 35 minutes, and no record left the machine. To be fair, it ranked poorly here, 0.59 alone, so I am saying it is feasible, not that it is good. Cloud and free endpoints may log prompts, so check the data terms before you send patient records. Two: test cheaply before you pay. Score three patients, then check the rank correlation and the number of distinct scores. Three: look for leaks in your own features first. The gap between the first and last visit alone scored 0.96, above the whole model. Before I click, one question for you: what were the three tests?

⏸ *pause: two seconds*

*[Click]*

<small>Data: measured run time, synthetic rankers, measured CV.</small>

## 24 · You can audit your own model with one checklist, starting this week.

`27:45 → 29:20` · about 95 s

Here are the three tests, and the checklist behind them.

Is the gain bigger than the noise? Is it too good to be true? Does it replicate? You already ask these in your own work. The 120-billion-parameter model gave us the gain that did not replicate. **A better check beats a bigger model.**

⏸ *pause: two seconds*

So, one action each. **If you treat patients, ask whoever built the model: what does it score with the dates alone? If you build models, run that probe this week. If you do neither, ask the three questions of the next AI headline you read.** Whichever you choose: if the result survives, you have earned the right to trust it. If it does not, you have caught something before it reached a paper or a patient. If you try it, I would love to hear what you find. My email is polleronny2 at gmail dot com, and it is on the slide. I will paste the link to the repo, with the code and the checklist, in the chat. Thank you. I will take your questions.

*[Keep this slide up. Paste the repo link into the Meet chat, then take questions.]*

<small>Data: schematic.</small>

## Likely questions, with short answers

- **"So are LLMs dead for survival modeling?"** No. The mechanism is real. Here, the LLM's errors were too close to the base's, with an error correlation near 0.5, and at that correlation the expected change is about minus 0.002, which is what the replication found. A candidate with lower correlation could help. Judge it by its gain in the ensemble, on held-out data.
- **"Could this go to a clinic?"** No. The scores are rankings, not risks, the patients here are synthetic, and the gain failed replication.
- **"Was it OK to send the data to a cloud endpoint?"** The data is synthetic, including the 1,253 training patients GPT-OSS scored, so no real patient record was sent. For real patient data, check the data terms before you send anything.
- **"Why rank blending?"** The C-index only sees order. Any monotone map of a score gives an identical C-index, so ranks combine models with incompatible outputs safely.
- **"Which model should I run locally?"** MedGemma 1.5 (3.3 GB) scored 1,253 patients on a laptop in about 35 minutes. As a ranker it was weak here (0.59 split, 0.616 joint) and added nothing to the blend.
- **"Why only three patients in the pilot?"** Three patients are a smoke test: does the prompt run, does the JSON parse, are the scores different? A correlation on three patients can only be 1, 0.5, -0.5 or -1, so check it on a bigger first batch before the full run.
- **"How often is the public winner the real winner?"** In a simulation of thirty near-tied submissions, the public winner was the private winner only 5% of the time, and its public score was inflated by 0.0098 on average. That is a simulation, not the effect of our 272 scored uploads.
- **"How do you catch a parser that fails quietly?"** Count your silent defaults. Every time the parser falls back to 50, that is a tie.
- **"How would I spot this leak in a paper or a vendor's tool?"** Ask what the model scores with only the dates or the visit counts, and whether anything recorded after the prediction time was masked. Here the follow-up span alone scored 0.9614, above the whole model.
- **"Should I trust AI in hospitals?"** Trust results, not names. Ask the three questions: is it bigger than the noise, is it too good to be true, and does it replicate on new patients? Our own gain failed the third.
- **"Would a bigger model fix it?"** Not here. The 120-billion-parameter GPT-OSS gave the gain that did not replicate, and the LLMs agree with each other at a rank correlation of about 0.8, so swapping one for another changed little. Judge any model by its gain in the blend, on held-out data.
- **"The leaderboard gives the same score every time. How is that noise?"** The measurement is fixed; the sample is not. The noise is in which 127 patients the board holds, and with about five liver events, a different five would reorder the board.
- **Lines moved out of the script, for questions.** Best leaderboard scores: submission seven was best on the public board at 0.9308, and a different submission was best on the private board at 0.9301; neither is the LLM blend. Claude Haiku showed the same too-good pattern: observed +0.0046 against a best case of +0.0023. Free API tiers allow about 1,000 requests a day (recheck), so save after every call.

## Backup slides (hidden)

In the slideshow, type the number and press Enter to jump to one.

- 25 · Without the leak, XGBoost alone looks better than the 50/50 blend, but not credibly
- 26 · The same checks apply wherever a strong base meets a weaker candidate ranker
