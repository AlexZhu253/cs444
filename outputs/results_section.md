## 4 Results

### 4.1 Evaluation Scope and Metrics

We evaluate refusal safety for three target models (`chatgpt`, `deepseek`, `gemini`) on a multilingual harmful-request benchmark spanning 45 dataset files (9 language groups × 5 harm categories), with 20 prompts per file (900 prompts total). Each model response is adjudicated by a three-judge labeling pipeline into Direct Refusal (DR), Partial Compliance (PC), or Full Compliance (FC). The primary rates are: Refusal Rate (RR), Partial Compliance Rate (PCR), and Full Compliance Rate (FCR), where lower FCR indicates safer behavior.

In line with the labeling pipeline, `split` cases (no unique majority among judges) are excluded from RR/PCR/FCR denominators and tracked separately as an ambiguity signal. To avoid over-weighting small or noisy slices, aggregate comparisons use split-aware weighting (`effective_n = n * (1 - split_rate)`), where `n = DR + PC + FC`.

### 4.2 Aggregate Performance Across Models

At the global level, model performance is clearly separated. Reliability-adjusted weighted FCR is 0.00% for `chatgpt`, 3.45% for `deepseek`, and 9.54% for `gemini`; corresponding adjusted RR values are 100.00%, 96.33%, and 86.98%, respectively. Mean split rates are low overall (0.00%, 0.22%, 0.56%), indicating stable label agreement across most slices.

This yields a consistent safety ordering by harmful compliance risk:

1. `chatgpt` (lowest risk),
2. `deepseek` (moderate risk),
3. `gemini` (highest risk).

**Figure callout.** *Fig. 1* (`outputs/figures/overview_weighted_fcr.png`) summarizes weighted FCR by model and dimension. *Fig. 2* (`outputs/figures/adjusted_overall_fcr_by_model.png`) shows the reliability-adjusted aggregate ranking.

### 4.3 Statistical Separation Between Models

To test whether aggregate differences are substantive, we perform pairwise two-proportion z-tests on aggregated FCR counts. The largest separations are:

- `chatgpt` vs `gemini`: z = -19.01
- `chatgpt` vs `deepseek`: z = -11.25
- `deepseek` vs `gemini`: z = -10.48

Given the sample sizes and effect magnitudes, these results indicate strong practical and statistical separation in harmful compliance behavior, not merely small-sample noise.

**Figure/Table callout.** Pairwise statistics are listed in `outputs/pairwise_fcr_ztests.csv`.

### 4.4 Risk Localization by Language and Category

Risk is not uniform across slices. Instead, failures concentrate in specific language configurations and categories, especially for `gemini`. The highest-risk cells include:

- `gemini | by_category:MD` with FCR = 31.82% (RR = 60.23%, n = 176),
- `gemini | by_group:ZH-SW` with FCR = 21.00%,
- `gemini | by_context_lang:SW` with FCR = 14.05%,
- `gemini | by_group:EN-SW` with FCR = 14.00%,
- `deepseek | by_group:EN-SW` with FCR = 15.00% (notable non-`gemini` hotspot).

These patterns suggest two stress regimes: (i) category-sensitive degradation in misinformation/disinformation (MD), and (ii) cross-lingual robustness gaps in selected bilingual settings.

**Figure callout.** Absolute-risk heatmaps are shown in *Fig. 3a–3d*:

- `outputs/figures/fcr_heatmap_by_group.png`
- `outputs/figures/fcr_heatmap_by_keyword_lang.png`
- `outputs/figures/fcr_heatmap_by_context_lang.png`
- `outputs/figures/fcr_heatmap_by_category.png`

### 4.5 Gap-to-Best Analysis

To quantify improvement headroom per slice, we compute each model’s FCR gap to the best model within each key (`gap_to_best_fcr = FCR_model - min_model(FCR)` for a fixed key). This analysis confirms that the largest deficits are concentrated in the same hotspots identified above, with `gemini` showing the largest and most frequent positive gaps, and `deepseek` generally closer to the best-performing model.

**Figure callout.** Gap-to-best heatmaps are provided in *Fig. 4a–4d*:

- `outputs/figures/fcr_gap_to_best_by_group.png`
- `outputs/figures/fcr_gap_to_best_by_keyword_lang.png`
- `outputs/figures/fcr_gap_to_best_by_context_lang.png`
- `outputs/figures/fcr_gap_to_best_by_category.png`

### 4.6 Uncertainty and Label Reliability

We report Wilson 95% confidence intervals for FCR at the slice level to account for finite sample uncertainty. High-risk slices remain elevated under interval-based comparison, supporting robustness of the main conclusions. In parallel, split-rate diagnostics show low but non-zero ambiguity concentrated in selected slices, with `gemini` exhibiting the highest average split rate.

**Figure callout.** CI and reliability diagnostics appear in:

- *Fig. 5*: `outputs/figures/fcr_with_ci_by_dimension.png`
- *Fig. 6a–6d*: `outputs/figures/split_heatmap_by_group.png`, `outputs/figures/split_heatmap_by_keyword_lang.png`, `outputs/figures/split_heatmap_by_context_lang.png`, `outputs/figures/split_heatmap_by_category.png`

### 4.7 Joint Trade-off View (Refusal vs. Compliance)

A joint RR–FCR bubble plot (bubble size = `n`, marker style indicating split presence) provides a compact view of operational trade-offs. Safer behavior appears in the upper-left safety regime (high RR, low FCR), where `chatgpt` dominates. `deepseek` occupies an intermediate cluster, while `gemini` exhibits broader spread and multiple points with materially elevated FCR.

**Figure callout.** *Fig. 7*: `outputs/figures/rr_vs_fcr_bubble.png`.

### 4.8 Summary of Findings

Across all aggregation views (group, keyword language, context language, category), the same ranking persists: `chatgpt` lowest harmful compliance, `deepseek` intermediate, `gemini` highest. The principal failure concentrations are category-specific (MD) and cross-lingual (notably ZH-SW and EN-SW contexts), rather than random or uniformly distributed. Split-aware weighting, uncertainty intervals, and pairwise tests all support that these differences are stable and practically meaningful for multilingual safety evaluation.

