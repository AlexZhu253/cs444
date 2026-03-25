"""
Creates 6 mixed-language datasets via keyword injection.

Pipeline per row:
  1. Preprocess source text (jieba segmentation for ZH; identity for EN/SW)
  2. Extract keywords with YAKE! (tuned params from methodology)
  3. Deduplicate & batch-translate keywords to the target keyword language
  4. Replace keywords in-place; context words stay in source language

Naming convention  →  {KW_LANG}-{CTX_LANG}-{CATEGORY_KEY}.tsv
  KW_LANG  = language the keywords are translated INTO
  CTX_LANG = language the context words stay in (= source dataset language)

The 6 output groups (30 files total):
  EN-ZH   EN-SW
  ZH-EN   ZH-SW
  SW-EN   SW-ZH

Reads:   data/{EN,ZH,SW}-{CU,CB,MD,HB,IA}.tsv
Writes:  data/{KW_LANG}-{CTX_LANG}-{CU,CB,MD,HB,IA}.tsv

Usage:
  python scripts/keyword_mix.py                          # all 6 pairs
  python scripts/keyword_mix.py --pairs ZH-EN SW-EN      # specific pairs
"""

import argparse
import csv
import os
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import jieba
import requests
import yake
from dotenv import load_dotenv

load_dotenv(override=True)

from utils import DATA_DIR

# ── Configuration ─────────────────────────────────────────────────────────────

CATEGORY_KEYS = ["CU", "CB", "MD", "HB", "IA"]

# Google Cloud Translation API v2 language codes
LANG_CODES = {
    "EN": "en",
    "ZH": "zh-CN",
    "SW": "sw",
}

# All 6 (kw_lang, ctx_lang) pairs to generate
ALL_PAIRS = [
    ("EN", "ZH"),
    ("EN", "SW"),
    ("ZH", "EN"),
    ("ZH", "SW"),
    ("SW", "EN"),
    ("SW", "ZH"),
]

# YAKE! parameters (tuned for short safety-study prompts per methodology)
YAKE_N       = 3    # max n-gram size — captures "homemade explosive device"
YAKE_TOP     = 10   # keywords to return per text
YAKE_DEDUP   = 0.9  # keep distinct harmful terms
YAKE_WINDOW  = 2    # co-occurrence window for short prompts

TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"
TRANSLATE_BATCH = 50   # max segments per API call


# ── Text preprocessing ────────────────────────────────────────────────────────

def preprocess(text: str, lang: str) -> str:
    """
    Returns a space-tokenised form of text suitable for YAKE!
    ZH → jieba word segmentation.  EN/SW → unchanged (already space-separated).
    """
    if lang == "ZH":
        return " ".join(jieba.cut(text))
    return text


# ── Keyword extraction ────────────────────────────────────────────────────────

def extract_keywords(processed_text: str, lang: str) -> list[str]:
    """
    Runs YAKE! on pre-processed text and returns keyword strings.
    Sorted longest-first so multi-word phrases are replaced before
    their constituent single words (prevents partial replacements).
    """
    yake_lang = "zh" if lang == "ZH" else "en"
    extractor = yake.KeywordExtractor(
        lan=yake_lang,
        n=YAKE_N,
        dedupLim=YAKE_DEDUP,
        top=YAKE_TOP,
        windowsSize=YAKE_WINDOW,
    )
    keywords = [kw for kw, _score in extractor.extract_keywords(processed_text)]
    # Longest first → avoid substring collision when replacing
    keywords.sort(key=len, reverse=True)
    return keywords


# ── Translation ───────────────────────────────────────────────────────────────

def _translate_batch(texts: list[str], src: str, tgt: str, api_key: str) -> list[str]:
    resp = requests.post(
        TRANSLATE_URL,
        params={"key": api_key},
        json={"q": texts, "source": src, "target": tgt, "format": "text"},
        timeout=30,
    )
    resp.raise_for_status()
    return [t["translatedText"] for t in resp.json()["data"]["translations"]]


def translate_list(texts: list[str], src: str, tgt: str, api_key: str) -> list[str]:
    """Translates an arbitrary list by chunking into API-friendly batches."""
    results = []
    for i in range(0, len(texts), TRANSLATE_BATCH):
        chunk = texts[i : i + TRANSLATE_BATCH]
        results.extend(_translate_batch(chunk, src, tgt, api_key))
        if i + TRANSLATE_BATCH < len(texts):
            time.sleep(0.2)
    return results


# ── Keyword replacement ───────────────────────────────────────────────────────

def replace_keywords(processed_text: str,
                     keywords: list[str],
                     translations: list[str],
                     ctx_lang: str) -> str:
    """
    Replaces each keyword in processed_text with its translation.
    - ZH (space-tokenised):  exact token-boundary replacement
    - EN / SW (Latin):       case-insensitive whole-word replacement
    """
    result = processed_text
    for kw, tr in zip(keywords, translations):
        if not kw.strip() or kw == tr:
            continue
        if ctx_lang == "ZH":
            # In jieba-segmented text every token is already space-delimited
            pattern = r"(?<!\S)" + re.escape(kw) + r"(?!\S)"
        else:
            pattern = r"\b" + re.escape(kw) + r"\b"
        result = re.sub(pattern, tr, result, flags=re.IGNORECASE)
    return result


# ── Per-pair main loop ────────────────────────────────────────────────────────

def process_pair(kw_lang: str, ctx_lang: str, api_key: str) -> None:
    src_code = LANG_CODES[ctx_lang]
    tgt_code = LANG_CODES[kw_lang]
    label    = f"{kw_lang}-{ctx_lang}"

    print(f"\n[{label}]  keywords→{kw_lang}  context stays in {ctx_lang}")

    for cat_key in CATEGORY_KEYS:
        src_path = DATA_DIR / f"{ctx_lang}-{cat_key}.tsv"
        dst_path = DATA_DIR / f"{label}-{cat_key}.tsv"

        if not src_path.exists():
            print(f"  ✗  {src_path.name} not found — skipping.")
            continue

        # ── Load ─────────────────────────────────────────────────────────────
        rows = []
        with open(src_path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f, delimiter="\t"))

        print(f"  {src_path.name} → {dst_path.name} ...", end=" ", flush=True)

        # ── Preprocess + extract keywords ────────────────────────────────────
        goal_data   = []   # (keywords, processed_text) per row
        target_data = []

        for row in rows:
            g_proc = preprocess(row["Goal"],   ctx_lang)
            t_proc = preprocess(row["Target"], ctx_lang)
            goal_data.append((extract_keywords(g_proc, ctx_lang), g_proc))
            target_data.append((extract_keywords(t_proc, ctx_lang), t_proc))

        # ── Batch-translate unique keywords (minimises API calls) ─────────────
        all_kws = []
        for kws, _ in goal_data:
            all_kws.extend(kws)
        for kws, _ in target_data:
            all_kws.extend(kws)

        unique_kws = list(dict.fromkeys(all_kws))   # dedup, preserve order

        if unique_kws:
            translated = translate_list(unique_kws, src_code, tgt_code, api_key)
            kw_map = dict(zip(unique_kws, translated))
        else:
            kw_map = {}

        # ── Apply replacements ────────────────────────────────────────────────
        output_rows = []
        for i, row in enumerate(rows):
            g_kws,   g_proc = goal_data[i]
            t_kws,   t_proc = target_data[i]
            g_trans = [kw_map.get(kw, kw) for kw in g_kws]
            t_trans = [kw_map.get(kw, kw) for kw in t_kws]
            mixed_goal   = replace_keywords(g_proc, g_kws, g_trans, ctx_lang)
            mixed_target = replace_keywords(t_proc, t_kws, t_trans, ctx_lang)
            output_rows.append({
                "Line":   row["Line"],
                "Goal":   mixed_goal,
                "Target": mixed_target,
            })

        # ── Write ─────────────────────────────────────────────────────────────
        with open(dst_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["Line", "Goal", "Target"], delimiter="\t")
            writer.writeheader()
            writer.writerows(output_rows)

        print("done.")


# ── Entry point ───────────────────────────────────────────────────────────────

def main(pairs: list[tuple[str, str]]) -> None:
    api_key = os.environ.get("GOOGLE_TRANSLATE_API_KEY", "")
    if not api_key or api_key == "your_google_translate_api_key_here":
        print("ERROR: GOOGLE_TRANSLATE_API_KEY is not set in your .env file.")
        sys.exit(1)

    for kw_lang, ctx_lang in pairs:
        try:
            process_pair(kw_lang, ctx_lang, api_key)
        except requests.HTTPError as e:
            print(f"  HTTP error: {e.response.status_code} — {e.response.text[:300]}")
        except Exception as e:
            print(f"  Unexpected error for {kw_lang}-{ctx_lang}: {e}")

    print("\nAll done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Build mixed-language datasets via keyword injection."
    )
    parser.add_argument(
        "--pairs",
        nargs="+",
        default=[f"{kw}-{ctx}" for kw, ctx in ALL_PAIRS],
        metavar="KW-CTX",
        help="Pairs to generate (e.g. ZH-EN SW-EN).  Default: all 6.",
    )
    args = parser.parse_args()

    parsed_pairs = []
    for p in args.pairs:
        parts = p.split("-")
        if len(parts) != 2 or parts[0] not in LANG_CODES or parts[1] not in LANG_CODES:
            print(f"Invalid pair '{p}'. Must be one of: " +
                  ", ".join(f"{a}-{b}" for a, b in ALL_PAIRS))
            sys.exit(1)
        parsed_pairs.append(tuple(parts))

    main(parsed_pairs)
