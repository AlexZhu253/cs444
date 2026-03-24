"""
Shared utilities for loading data, building prompts, and parsing model responses.
"""

import csv
import json
import re
from pathlib import Path

ROOT = Path(__file__).parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

CATEGORIES_FILE = DATA_DIR / "categorization.tsv"
DATASET_FILE = DATA_DIR / "Primary Dataset.tsv"

BATCH_SIZE = 50


def load_categories() -> list[tuple[str, str]]:
    """Returns a list of (name, definition) tuples from categorization.tsv."""
    categories = []
    with open(CATEGORIES_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            categories.append((row["Name"].strip(), row["Definition"].strip()))
    return categories


def load_dataset() -> list[tuple[int, str]]:
    """
    Returns a list of (line_number, goal) tuples from the primary dataset.
    Line numbers are 1-indexed as they appear in the file (header = line 1,
    first data row = line 2).
    """
    rows = []
    with open(DATASET_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for i, row in enumerate(reader, start=2):
            rows.append((i, row["Goal"].strip()))
    return rows


def build_system_prompt(categories: list[tuple[str, str]]) -> str:
    """
    Builds the system prompt that instructs the model how to classify requests.
    """
    lines = [
        "You are a content safety classifier. Classify each request into exactly one of the 7 categories below.",
        "",
        "Categories:",
    ]
    for i, (name, definition) in enumerate(categories, 1):
        lines.append(f"  {i}. {name}: {definition}")
    lines += [
        "",
        "You will receive a numbered list of requests.",
        "Respond ONLY with a valid JSON object that maps each line number (string key) to its category name (string value).",
        'Example: {"2": "Cybercrime & Unauthorized Intrusion", "3": "Illegal Activities"}',
        "Use ONLY the exact category names listed above. Do not add any explanation or extra text outside the JSON.",
    ]
    return "\n".join(lines)


def build_user_prompt(batch: list[tuple[int, str]]) -> str:
    """Builds the user turn prompt for a batch of (line_number, goal) pairs."""
    lines = ["Classify the following requests:"]
    for line_num, goal in batch:
        lines.append(f"Line {line_num}: {goal}")
    return "\n".join(lines)


def parse_json_response(text: str) -> dict[str, str]:
    """
    Extracts and parses a JSON object from a model response,
    handling markdown code fences if present.
    """
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return json.loads(text.strip())


def make_batches(rows: list, batch_size: int = BATCH_SIZE):
    """Yields successive batches of the given list."""
    for i in range(0, len(rows), batch_size):
        yield rows[i : i + batch_size]


def save_results(results: dict, filename: str) -> None:
    """Saves a results dict as JSON into the outputs directory."""
    OUTPUTS_DIR.mkdir(exist_ok=True)
    path = OUTPUTS_DIR / filename
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"Saved → {path}")
