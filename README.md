# cs444 — Censorship and Refusal Behavior in LLMs: Consistency Across Languages and Models

Cross-lingual jailbreak safety study. Probes LLM safety guardrails by decomposing adversarial prompts into intent-bearing **keywords** ($K$) and structural **context** ($C$), then intermixing them across English (EN), Chinese (ZH), and Swahili (SW).

---

## Project Structure

```
cs444/
├── data/                        # All datasets (raw + generated)
│   ├── Primary Dataset.tsv      # Source: AdvBench (520 harmful prompt–completion pairs)
│   ├── categorization.tsv       # 7-category HarmBench taxonomy with keys
│   ├── EN-{CAT}.tsv             # Monolingual English, 5 categories × 20 examples
│   ├── ZH-{CAT}.tsv             # Monolingual Chinese
│   ├── SW-{CAT}.tsv             # Monolingual Swahili
│   └── {KW}-{CTX}-{CAT}.tsv    # Mixed-language datasets (6 combinations × 5 categories)
├── outputs/
│   ├── chatgpt_results.json     # Raw classification output from ChatGPT
│   ├── deepseek_results.json    # Raw classification output from DeepSeek
│   ├── gemini_results.json      # Raw classification output from Gemini
│   ├── final_categorization.json/.tsv   # Consensus-filtered category assignments
│   ├── samples/                 # Per-category sampled files (pre-rename)
│   └── responses/               # Model responses from collect_responses.py
│       └── {GROUP}-{CAT}.json
└── scripts/
    ├── utils.py                 # Shared helpers (data loading, prompts, JSON parsing)
    ├── classify_chatgpt.py      # Step 2a: classify dataset with ChatGPT
    ├── classify_deepseek.py     # Step 2b: classify dataset with DeepSeek
    ├── classify_gemini.py       # Step 2c: classify dataset with Gemini
    ├── compare_results.py       # Step 3: consensus filter → final_categorization
    ├── sample_categories.py     # Step 4: sample 20 examples per category
    ├── translate_datasets.py    # Step 5: translate EN → ZH and SW
    ├── keyword_mix.py           # Step 6: build 6 mixed-language datasets
    └── collect_responses.py     # Step 7: query models and collect responses
```

---

## Setup

### 1. Install dependencies

```bash
python -m pip install openai google-genai python-dotenv requests yake jieba
```

### 2. Configure API keys

Fill in `.env` at the project root (never commit this file):

```
OPENAI_API_KEY=sk-...
DEEPSEEK_API_KEY=sk-...
GEMINI_API_KEY=AIza...
GOOGLE_TRANSLATE_API_KEY=AIza...
```

> **Note:** Always use `python -m pip install` to ensure packages install into the same Python that runs your scripts. Mismatched environments (e.g., system Python vs Anaconda) are the most common source of `ModuleNotFoundError`.

---

## Models Used

| Model | Provider | Version |
|-------|----------|---------|
| ChatGPT | OpenAI | `gpt-4o` |
| DeepSeek | DeepSeek AI | `deepseek-chat` (DeepSeek-V3) |
| Gemini | Google DeepMind | `gemini-2.0-flash-lite` |

To change a model version, edit the `MODEL = "..."` line at the top of the relevant script.

---

## Dataset Layout

### Category Keys (`data/categorization.tsv`)

| Key | Category |
|-----|----------|
| CU | Cybercrime & Unauthorized Intrusion |
| CB | Chemical & Biological Weapons/Drugs |
| CV | Copyright Violations *(excluded — insufficient consensus examples)* |
| MD | Misinformation & Disinformation |
| HB | Harassment & Bullying |
| IA | Illegal Activities |
| GH | General Harm *(excluded — insufficient consensus examples)* |

**Active categories:** CU, CB, MD, HB, IA (20 examples each)

### Language Groups

| Group | Description |
|-------|-------------|
| `EN` | Monolingual English |
| `ZH` | Monolingual Chinese |
| `SW` | Monolingual Swahili |
| `EN-ZH` | English keywords in Chinese context |
| `EN-SW` | English keywords in Swahili context |
| `ZH-EN` | Chinese keywords in English context |
| `ZH-SW` | Chinese keywords in Swahili context |
| `SW-EN` | Swahili keywords in English context |
| `SW-ZH` | Swahili keywords in Chinese context |

**File naming:** `{KW_LANG}-{CTX_LANG}-{CATEGORY_KEY}.tsv`
- Monolingual: `EN-CU.tsv`
- Mixed: `ZH-EN-CU.tsv` (Chinese keywords, English context, CU category)

**Total: 45 files × 20 examples = 900 prompts**

---

## Pipeline — Step by Step

### Step 1 — Understand the source data

```
data/Primary Dataset.tsv     520 rows, columns: Goal | Target
data/categorization.tsv      7 categories, columns: Key | Name | Definition
```

### Step 2 — Classify the dataset (Triple-Model Consensus)

Run each script independently (can be run in parallel in separate terminals):

```bash
python scripts/classify_chatgpt.py
python scripts/classify_deepseek.py
python scripts/classify_gemini.py
```

Output: `outputs/{model}_results.json` — maps line numbers to category names.

Each script sends the 520 rows in batches of 50, retries on failure, and saves progress after each batch.

### Step 3 — Merge classifications (consensus filter)

```bash
python scripts/compare_results.py
```

Keeps only rows where all 3 models agree. Discards divergent/missing labels.

Output:
- `outputs/final_categorization.json`
- `outputs/final_categorization.tsv`

Prints per-category agreed counts and overall agreement/divergence statistics.

### Step 4 — Sample 20 examples per category

```bash
python scripts/sample_categories.py          # default: 20 per category
python scripts/sample_categories.py --n 10  # custom sample size
```

Output: `outputs/samples/{category_name}.tsv`

Files with fewer than 20 agreed examples (CV, GH) are automatically skipped.

### Step 5 — Translate to Chinese and Swahili

**Prerequisite:** Set `GOOGLE_TRANSLATE_API_KEY` in `.env` and enable the Cloud Translation API in Google Cloud Console.

```bash
python scripts/translate_datasets.py            # both ZH and SW
python scripts/translate_datasets.py --langs ZH # Chinese only
python scripts/translate_datasets.py --langs SW # Swahili only
```

Output: `data/ZH-{CAT}.tsv` and `data/SW-{CAT}.tsv` (10 files total)

The free tier covers ~500,000 characters/month. This project uses ~16,000 characters per language — well within quota.

### Step 6 — Build mixed-language datasets

**Prerequisite:** ZH and SW monolingual files must exist (Step 5 complete).

```bash
python scripts/keyword_mix.py                         # all 6 mixed pairs
python scripts/keyword_mix.py --pairs ZH-EN SW-EN    # specific pairs only
```

Valid pairs: `EN-ZH  EN-SW  ZH-EN  ZH-SW  SW-EN  SW-ZH`

Output: `data/{KW_LANG}-{CTX_LANG}-{CAT}.tsv` (30 files)

**How it works:**
1. YAKE! extracts top-10 keywords (n-gram ≤ 3) from each prompt
2. Chinese text is pre-segmented with jieba before YAKE! runs
3. Keywords are batch-translated via Google Translate API
4. Keywords are replaced in-place; context words stay in the source language

### Step 7 — Collect model responses

```bash
python scripts/collect_responses.py                         # all 45 files, all 3 models
python scripts/collect_responses.py --groups EN ZH SW       # monolingual only
python scripts/collect_responses.py --groups EN-ZH ZH-EN    # specific groups
python scripts/collect_responses.py --models chatgpt        # one model only
python scripts/collect_responses.py --models chatgpt deepseek  # two models
python scripts/collect_responses.py --overwrite             # ignore cached outputs
```

`--groups` options: `EN  ZH  SW  EN-ZH  EN-SW  ZH-EN  ZH-SW  SW-EN  SW-ZH`

`--models` options: `chatgpt  deepseek  gemini`

Output: `outputs/responses/{GROUP}-{CAT}.json`

Each JSON is a list of 20 records:
```json
{
  "line":     30,
  "goal":     "Create a ransomware...",
  "target":   "Sure, here is a ransomware...",
  "chatgpt":  "<model response>",
  "deepseek": "<model response>",
  "gemini":   "<model response>"
}
```

The script saves after every row — safe to interrupt and resume. Missing model columns (null) are filled on re-run without re-querying complete columns.

---

## Common Issues

| Error | Cause | Fix |
|-------|-------|-----|
| `ModuleNotFoundError` | Package installed to wrong Python | Use `python -m pip install <pkg>` |
| `Illegal header value` for OpenAI | `OPENAI_API_KEY` in shell env is corrupted | `load_dotenv(override=True)` already handles this; check `~/.zshrc` for stray export lines |
| `429 RESOURCE_EXHAUSTED` (Gemini) | Free-tier daily quota exhausted | Wait for daily reset, enable billing, or use a new API key |
| `limit: 0` in Gemini 429 | Daily quota is fully used up — retrying won't help | Wait until midnight PT or enable billing |
