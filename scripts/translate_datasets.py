"""
Translates each EN-XX.tsv dataset into Chinese (ZH) and Swahili (SW)
using the Google Cloud Translation API v2 (REST, API-key auth).

Reads:   data/EN-{CU,CB,MD,HB,IA}.tsv
Writes:  data/ZH-{CU,CB,MD,HB,IA}.tsv
         data/SW-{CU,CB,MD,HB,IA}.tsv

Usage:
  python scripts/translate_datasets.py
  python scripts/translate_datasets.py --langs ZH   # only Chinese
  python scripts/translate_datasets.py --langs SW   # only Swahili
"""

import argparse
import csv
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import requests
from dotenv import load_dotenv
import os

load_dotenv(override=True)

from utils import DATA_DIR

# ── Config ────────────────────────────────────────────────────────────────────

TRANSLATE_URL = "https://translation.googleapis.com/language/translate/v2"

CATEGORY_KEYS = ["CU", "CB", "MD", "HB", "IA"]

LANGUAGE_CODES = {
    "ZH": "zh-CN",   # Simplified Chinese
    "SW": "sw",       # Swahili
}

# Google Translation API v2 allows up to 128 segments per request.
BATCH_SIZE = 50


# ── Translation helpers ───────────────────────────────────────────────────────

def translate_batch(texts: list[str], target_lang: str, api_key: str) -> list[str]:
    """
    Calls Google Translation API v2 for a batch of texts.
    Returns translated strings in the same order.
    """
    response = requests.post(
        TRANSLATE_URL,
        params={"key": api_key},
        json={
            "q": texts,
            "source": "en",
            "target": target_lang,
            "format": "text",
        },
        timeout=30,
    )
    response.raise_for_status()
    return [t["translatedText"] for t in response.json()["data"]["translations"]]


def translate_list(texts: list[str], target_lang: str, api_key: str) -> list[str]:
    """Translates an arbitrary-length list by splitting into batches."""
    results = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i : i + BATCH_SIZE]
        results.extend(translate_batch(batch, target_lang, api_key))
        if i + BATCH_SIZE < len(texts):
            time.sleep(0.2)  # stay comfortably within rate limits
    return results


# ── Per-file translation ──────────────────────────────────────────────────────

def translate_file(category_key: str, lang_prefix: str, target_lang: str, api_key: str) -> None:
    src_path = DATA_DIR / f"EN-{category_key}.tsv"
    dst_path = DATA_DIR / f"{lang_prefix}-{category_key}.tsv"

    # Load source rows
    rows = []
    with open(src_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(row)

    if not rows:
        print(f"  {src_path.name} is empty, skipping.")
        return

    # Collect all texts to translate in two passes (Goal, then Target)
    goals   = [r["Goal"]   for r in rows]
    targets = [r["Target"] for r in rows]

    print(f"  Translating {src_path.name} → {dst_path.name} ...", end=" ", flush=True)
    translated_goals   = translate_list(goals,   target_lang, api_key)
    translated_targets = translate_list(targets, target_lang, api_key)
    print("done.")

    # Write output
    with open(dst_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Line", "Goal", "Target"])
        for row, t_goal, t_target in zip(rows, translated_goals, translated_targets):
            writer.writerow([row["Line"], t_goal, t_target])


# ── Main ──────────────────────────────────────────────────────────────────────

def main(langs: list[str]) -> None:
    api_key = os.environ.get("GOOGLE_TRANSLATE_API_KEY", "")
    if not api_key:
        print("ERROR: GOOGLE_TRANSLATE_API_KEY is not set in your .env file.")
        print("See the instructions in the README or ask your assistant for setup steps.")
        sys.exit(1)

    for lang_prefix in langs:
        target_lang = LANGUAGE_CODES[lang_prefix]
        print(f"\n[{lang_prefix}] Translating to '{target_lang}':")
        for key in CATEGORY_KEYS:
            try:
                translate_file(key, lang_prefix, target_lang, api_key)
            except requests.HTTPError as e:
                print(f"  HTTP error for {key}: {e.response.status_code} — {e.response.text[:200]}")
            except Exception as e:
                print(f"  Failed for {key}: {e}")

    print("\nAll done. Output files written to data/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Translate EN datasets to ZH and SW.")
    parser.add_argument(
        "--langs",
        nargs="+",
        choices=list(LANGUAGE_CODES.keys()),
        default=list(LANGUAGE_CODES.keys()),
        help="Which target languages to produce (default: ZH SW)",
    )
    args = parser.parse_args()
    main(langs=args.langs)
