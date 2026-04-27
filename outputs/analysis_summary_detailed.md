# Detailed Analysis Summary

## Overall Reliability-Adjusted Ranking (lower FCR is better)

- chatgpt: adjusted_FCR=0.00%, adjusted_RR=100.00%, mean_split_rate=0.00%
- deepseek: adjusted_FCR=3.45%, adjusted_RR=96.33%, mean_split_rate=0.22%
- gemini: adjusted_FCR=9.54%, adjusted_RR=86.98%, mean_split_rate=0.56%

## Largest Pairwise FCR Differences (z-stat)

- chatgpt vs gemini: z=-19.01
- chatgpt vs deepseek: z=-11.25
- deepseek vs gemini: z=-10.48

## Top 15 Risk Hotspots

- gemini | by_category:MD | FCR=31.82%, RR=60.23%, split_rate=2.22%, n=176
- gemini | by_group:ZH-SW | FCR=21.00%, RR=76.00%, split_rate=0.00%, n=100
- deepseek | by_group:EN-SW | FCR=15.00%, RR=84.00%, split_rate=0.00%, n=100
- gemini | by_context_lang:SW | FCR=14.05%, RR=81.94%, split_rate=0.33%, n=299
- gemini | by_group:EN-SW | FCR=14.00%, RR=81.00%, split_rate=0.00%, n=100
- gemini | by_group:SW-EN | FCR=12.12%, RR=83.84%, split_rate=1.00%, n=99
- gemini | by_keyword_lang:ZH | FCR=11.82%, RR=85.81%, split_rate=0.34%, n=296
- deepseek | by_group:EN | FCR=11.00%, RR=89.00%, split_rate=0.00%, n=100
- gemini | by_keyword_lang:SW | FCR=8.78%, RR=86.82%, split_rate=1.00%, n=296
- deepseek | by_keyword_lang:EN | FCR=8.67%, RR=91.00%, split_rate=0.00%, n=300
- gemini | by_group:ZH-EN | FCR=8.16%, RR=90.82%, split_rate=0.00%, n=98
- gemini | by_context_lang:EN | FCR=8.14%, RR=89.49%, split_rate=0.67%, n=295
- gemini | by_keyword_lang:EN | FCR=8.08%, RR=88.22%, split_rate=0.34%, n=297
- deepseek | by_category:MD | FCR=7.87%, RR=91.01%, split_rate=1.11%, n=178
- gemini | by_group:SW-ZH | FCR=7.14%, RR=87.76%, split_rate=1.01%, n=98

## Top 15 Strong Cells

- chatgpt | by_group:EN | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- chatgpt | by_group:EN-SW | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- chatgpt | by_group:EN-ZH | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- deepseek | by_group:EN-ZH | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- chatgpt | by_group:SW | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- chatgpt | by_group:SW-EN | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- chatgpt | by_group:SW-ZH | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- deepseek | by_group:SW-ZH | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- chatgpt | by_group:ZH | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- deepseek | by_group:ZH | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- chatgpt | by_group:ZH-EN | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- deepseek | by_group:ZH-EN | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- chatgpt | by_group:ZH-SW | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- deepseek | by_group:ZH-SW | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=100
- chatgpt | by_keyword_lang:EN | FCR=0.00%, RR=100.00%, split_rate=0.00%, n=300

## Notes

- Split labels are excluded from RR/PCR/FCR denominators by design.
- Weighted metrics use split-adjusted effective sample sizes to reduce ambiguity-driven bias.
- Pairwise z-tests are approximate and intended for directional comparison.