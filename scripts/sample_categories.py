"""
Randomly samples up to 20 examples from each category in the final
categorization and writes one TSV file per category to outputs/.

Reads:
  outputs/final_categorization.json   — produced by compare_results.py
  data/Primary Dataset.tsv            — original dataset

Writes (one file per category):
  outputs/samples/<sanitized_category_name>.tsv

Usage:
  python scripts/sample_categories.py
  python scripts/sample_categories.py --n 10   # sample a different number
"""

import argparse
import csv
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from utils import OUTPUTS_DIR, DATASET_FILE

SAMPLES_DIR = OUTPUTS_DIR / "samples"
FINAL_CATEGORIZATION = OUTPUTS_DIR / "final_categorization.json"
DEFAULT_N = 20
RANDOM_SEED = 42


def sanitize_filename(name: str) -> str:
    """Converts a category name to a safe filename."""
    return re.sub(r"[^\w]+", "_", name).strip("_").lower()


def load_final_categorization() -> dict[str, list[int]]:
    with open(FINAL_CATEGORIZATION, encoding="utf-8") as f:
        return json.load(f)


def load_dataset_by_line() -> dict[int, dict]:
    """Returns {line_number: {"Goal": ..., "Target": ...}}."""
    rows = {}
    with open(DATASET_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader, start=2):
            rows[i] = {"Goal": row["Goal"].strip(), "Target": row["Target"].strip()}
    return rows


def sample_and_write(category: str, line_numbers: list[int], dataset: dict, n: int) -> None:
    if not line_numbers:
        print(f"  [{category}] — no agreed examples, skipping.")
        return

    chosen = random.sample(line_numbers, min(n, len(line_numbers)))
    chosen.sort()

    filename = SAMPLES_DIR / f"{sanitize_filename(category)}.tsv"
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f, delimiter="\t")
        writer.writerow(["Line", "Goal", "Target"])
        for line_num in chosen:
            row = dataset[line_num]
            writer.writerow([line_num, row["Goal"], row["Target"]])

    print(f"  [{category}] — sampled {len(chosen)}/{len(line_numbers)} → {filename.name}")


def main(n: int = DEFAULT_N) -> None:
    random.seed(RANDOM_SEED)

    print(f"Loading final categorization from {FINAL_CATEGORIZATION}...")
    categorization = load_final_categorization()

    print(f"Loading primary dataset from {DATASET_FILE}...")
    dataset = load_dataset_by_line()

    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    print(f"\nSampling up to {n} examples per category → {SAMPLES_DIR}/\n")

    for category, line_numbers in categorization.items():
        sample_and_write(category, line_numbers, dataset, n)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Sample examples per category.")
    parser.add_argument(
        "--n",
        type=int,
        default=DEFAULT_N,
        help=f"Number of examples to sample per category (default: {DEFAULT_N})",
    )
    args = parser.parse_args()
    main(n=args.n)
