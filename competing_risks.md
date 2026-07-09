The competing risks analysis is complete. Here's a summary of what was produced and the key scientific findings:

---

## Competing Risks Analysis — Results

### What was done
- Linked the hepatic (n=1,253) and death (n=984) datasets to classify each patient's outcome: hepatic event (n=36), competing death without hepatic event (n=64), or censored (n=838)
- Ran Fine-Gray subdistribution hazard regression (R `cmprsk`) alongside the cause-specific Cox model
- Generated CIF curves overall and stratified by risk group; ran Gray's test

### CS-Cox vs Fine-Gray comparison

| Variable | CS-Cox HR (95% CI) | p | FG SHR (95% CI) | p | Change |
|---|---|---|---|---|---|
| Liver stiffness (kPa) | 1.137 (1.098–1.178) | <0.001 | 1.103 (1.064–1.144) | <0.001 | −3.0% |
| AST (U/L) | 1.022 (1.008–1.035) | 0.002 | 1.023 (1.011–1.036) | <0.001 | +0.2% |
| GGT (U/L) | 1.002 (1.000–1.004) | 0.102 | 1.002 (1.000–1.004) | 0.100 | ≈0% |
| Age (years) | 1.053 (1.016–1.091) | 0.005 | 1.023 (0.992–1.054) | 0.150 | −2.9% |
| T2DM | 0.943 (0.524–1.699) | 0.846 | 1.005 (0.581–1.737) | 0.990 | +6.5% |

### Key findings

**Liver stiffness**: The main finding holds robustly. SHR attenuated from 1.137 to 1.103 (−3%), but remains highly significant (p<0.001). This attenuation is expected and clinically meaningful — competing deaths are more common in high-stiffness patients, so the cause-specific model slightly overestimates the absolute hepatic risk.

**Age**: The most notable change. HR drops from 1.053 (p=0.005) to SHR 1.023 (p=0.15). Clinically plausible — older patients die from competing causes before reaching hepatic decompensation, so age is less prognostic for the hepatic endpoint specifically once competing deaths are accounted for. However, with EPV=7.2, this change may partly reflect noise rather than a true biological signal.

**AST, GGT, T2DM**: Essentially unchanged — competing risks do not materially affect their estimates.

**CIF curves**: Gray's test confirms strong risk group separation (p=1.4×10⁻¹⁰ for hepatic event). Low Risk patients have zero hepatic events throughout follow-up.

### Limitations to flag
1. **EPV=7.2** for both models — all estimates are underpowered; wide CIs should be interpreted cautiously
2. **269 patients** in the hepatic dataset are absent from the death dataset (short follow-up, ~1 year median) — treated as censored; a small number of undetected competing deaths is possible
3. **Cause of death is unknown** — the 64 competing deaths are assumed non-hepatic; any liver-related deaths not captured in the hepatic endpoint would be misclassified

---

**Files produced:**
- `competing_risks_analysis.png/.svg` — 3-panel figure
- `cs_cox_vs_fine_gray_comparison.csv` — full comparison table