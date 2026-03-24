"""
Classifies the primary dataset using OpenAI ChatGPT.
Sends the dataset in batches and saves per-row classifications to:
  outputs/chatgpt_results.json  →  {"2": "Category Name", "3": "Category Name", ...}

Usage:
  python scripts/classify_chatgpt.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from openai import OpenAI

from utils import (
    build_system_prompt,
    build_user_prompt,
    load_categories,
    load_dataset,
    make_batches,
    parse_json_response,
    save_results,
)

load_dotenv(override=True)

client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
MODEL = "gpt-4o"


def classify_batch(system_prompt: str, batch: list, retries: int = 2) -> dict:
    user_prompt = build_user_prompt(batch)
    for attempt in range(retries + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
            return parse_json_response(response.choices[0].message.content)
        except Exception as e:
            if attempt < retries:
                print(f"  Attempt {attempt + 1} failed: {e}. Retrying in 3s...")
                time.sleep(3)
            else:
                print(f"  Batch failed after {retries + 1} attempts: {e}")
                return {}


def main():
    categories = load_categories()
    dataset = load_dataset()
    system_prompt = build_system_prompt(categories)

    results = {}
    batches = list(make_batches(dataset))
    total = len(batches)

    print(f"ChatGPT ({MODEL}): classifying {len(dataset)} rows in {total} batches...")

    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i}/{total} (lines {batch[0][0]}–{batch[-1][0]})...", end=" ")
        batch_results = classify_batch(system_prompt, batch)
        results.update(batch_results)
        print(f"got {len(batch_results)} classifications.")
        if i < total:
            time.sleep(1)  # avoid rate-limit bursts

    print(f"\nTotal classified: {len(results)}/{len(dataset)} rows.")
    save_results(results, "chatgpt_results.json")


if __name__ == "__main__":
    main()
