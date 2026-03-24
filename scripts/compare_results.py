"""
Compares classifications from ChatGPT, DeepSeek, and Gemini.

For each data row, keeps it only when all three models agree on the category.
Rows where any model disagrees or didn't return a result are discarded.

Reads:
  outputs/chatgpt_results.json
  outputs/deepseek_results.json
  outputs/gemini_results.json

Writes:
  outputs/final_categorization.tsv   — human-readable, one category per line
  outputs/final_categorization.json  — machine-readable dict

Usage:
  python scripts/compare_results.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import OUTPUTS_DIR, load_categories, load_dataset


def load_results(filename: str) -> dict[int, str]:
    """Loads a results JSON and normalises all keys to int."""
    path = OUTPUTS_DIR / filename
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def main():
    categories = load_categories()
    category_names = [name for name, _ in categories]
    dataset = load_dataset()
    all_line_numbers = sorted(line_num for line_num, _ in dataset)

    print("Loading model results...")
    chatgpt = load_results("chatgpt_results.json")
    deepseek = load_results("deepseek_results.json")
    gemini = load_results("gemini_results.json")

    agreed = 0
    diverged = 0
    missing = 0
    unknown_category = 0

    final: dict[str, list[int]] = defaultdict(list)

    for line_num in all_line_numbers:
        c = chatgpt.get(line_num)
        d = deepseek.get(line_num)
        g = gemini.get(line_num)

        # Skip if any model didn't return a result for this line
        if c is None or d is None or g is None:
            missing += 1
            continue

        # Skip if any model returned an unrecognised category name
        if c not in category_names or d not in category_names or g not in category_names:
            unknown_category += 1
            continue

        if c == d == g:
            final[c].append(line_num)
            agreed += 1
        else:
            diverged += 1

    total = len(all_line_numbers)
    print(f"\nResults over {total} data rows:")
    print(f"  All three models agreed : {agreed}")
    print(f"  At least one disagreed  : {diverged}")
    print(f"  Missing from ≥1 model   : {missing}")
    print(f"  Unknown category name   : {unknown_category}")

    # ── TSV output (human-readable) ──────────────────────────────────────────
    tsv_path = OUTPUTS_DIR / "final_categorization.tsv"
    with open(tsv_path, "w", encoding="utf-8") as f:
        f.write("Category\tLine Numbers\n")
        for name, _ in categories:
            line_nums = final.get(name, [])
            f.write(f"{name}\t{', '.join(str(n) for n in line_nums)}\n")
    print(f"\nSaved TSV  → {tsv_path}")

    # ── JSON output (machine-readable) ───────────────────────────────────────
    json_path = OUTPUTS_DIR / "final_categorization.json"
    json_data = {name: final.get(name, []) for name, _ in categories}
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    print(f"Saved JSON → {json_path}")

    # ── Per-category summary ─────────────────────────────────────────────────
    print("\nPer-category agreed line counts:")
    for name, _ in categories:
        count = len(final.get(name, []))
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
