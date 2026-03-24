"""
Classifies the primary dataset using Google Gemini.
Sends the dataset in batches and saves per-row classifications to:
  outputs/gemini_results.json  →  {"2": "Category Name", "3": "Category Name", ...}

Usage:
  python scripts/classify_gemini.py
"""

import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from dotenv import load_dotenv
from google import genai
from google.genai import types

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

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# gemini-2.0-flash-lite has a separate, more generous free-tier quota.
# Switch back to "gemini-2.0-flash" if you have a paid plan.
# MODEL = "gemini-2.0-flash-lite"
MODEL = "gemini-3-flash-preview"

# Seconds to wait between successful batches (free tier: 30 RPM)
INTER_BATCH_SLEEP = 10


def _parse_retry_delay(exc: Exception) -> float:
    """
    Gemini 429 errors include a suggested retryDelay in the response body.
    Extract it so we wait exactly as long as the API asks.
    Returns a float number of seconds, defaulting to 60 if not parseable.
    """
    import re
    text = str(exc)
    # Look for  'retryDelay': '11s'  or  'retryDelay': '11.3s'
    match = re.search(r"retryDelay['\"]:\s*['\"]([0-9.]+)s", text)
    if match:
        return float(match.group(1)) + 2  # add a small buffer
    return 60.0


def classify_batch(system_prompt: str, batch: list, retries: int = 5) -> dict:
    user_prompt = build_user_prompt(batch)
    for attempt in range(retries + 1):
        try:
            response = client.models.generate_content(
                model=MODEL,
                config=types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    response_mime_type="application/json",
                ),
                contents=user_prompt,
            )
            return parse_json_response(response.text)
        except Exception as e:
            is_rate_limit = "429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)
            if attempt < retries:
                if is_rate_limit:
                    wait = _parse_retry_delay(e)
                    print(f"  Rate limited (attempt {attempt + 1}). Waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
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

    print(f"Gemini ({MODEL}): classifying {len(dataset)} rows in {total} batches...")

    for i, batch in enumerate(batches, 1):
        print(f"  Batch {i}/{total} (lines {batch[0][0]}–{batch[-1][0]})...", end=" ", flush=True)
        batch_results = classify_batch(system_prompt, batch)
        results.update(batch_results)
        print(f"got {len(batch_results)} classifications.")
        if i < total:
            time.sleep(INTER_BATCH_SLEEP)

    print(f"\nTotal classified: {len(results)}/{len(dataset)} rows.")
    save_results(results, "gemini_results.json")


if __name__ == "__main__":
    main()
