Topic: *LLM Risk Rankers and Agentic Scaffolding for Liver Disease Survival Modeling*

*Abstract*

Metabolic dysfunction-associated steatotic liver disease (MASLD) is a common fatty liver disease linked to obesity and diabetes. Predicting who will suffer a major liver event or die is hard. Records can be irregular and right-censored; that is, many patients are still event-free when follow-up ends. Large language models (LLMs) are usually tested as clinical chatbots, and alone they rank patients poorly on such data.

We build on the HeLM framework (Belyaeva, Cosentino et al., 2023, Multimodal LLMs for health grounded in individual-specific data) and on agentic harness design. We turn each patient's visit history into a short clinical summary and ask an open-weights LLM for a per-endpoint risk score. The LLM ranks worse than a gradient-boosted survival model, but its errors differ from those of tree models. We ask whether rank-based blending can combine LLM and tree-based rankers. Finally, we share practical takeaways for researchers deploying foundation models on structured health data.
