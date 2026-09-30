### Proposed Talk Title

**When a Weak AI Model Helps: LLM Risk Rankers for Liver Disease**

*Orthogonal LLM Rankers and Agentic Scaffolding for MASLD Survival Modeling*

### Abstract

Metabolic dysfunction-associated steatotic liver disease (MASLD) is a common fatty liver disease linked to obesity and diabetes. Predicting who will suffer a major liver event or die is hard. Records are irregular and right-censored: many patients are still event-free when follow-up ends. Large language models (LLMs) are usually tested as clinical chatbots, and alone they rank patients poorly on such data.

We build on the HeLM framework (Belyaeva, Cosentino et al., 2023, *Multimodal LLMs for health grounded in individual-specific data*) and on agentic harness design. We turn each patient's visit history into a short clinical summary and ask an open-weights LLM for a risk score, one prompt per endpoint. We measure ranking quality with a weighted C-index, where 0.5 is chance and 1 is perfect. Alone, the LLM scores 0.792, well below a gradient-boosted survival model at 0.894. But its errors differ from those of tree models. Blending ranks rather than raw scores, this "weak" ranker lifts the pipeline to 0.904 on the ANNITIA 2026 public leaderboard.

We treat careful testing as part of responsible AI. We show cheap checks to run before spending compute: rank correlation with the base model and the number of distinct LLM scores. We also show how to guard against features that leak the answer, such as follow-up length. Finally, we share takeaways for researchers: open-weights models, some small enough to run on a laptop, let you test an LLM on health records cheaply, without sending data off-site.
