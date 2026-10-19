# Copyright 2025 Individual Contributor:
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Convert a CSV of multi-turn tool-calling samples into verl's parquet format,
and write BOTH train.parquet and test.parquet to an output directory.

Input CSV is expected to have at least these columns:
- messages: Python-literal list of {"role", "content"}
- solution: Python/JSON list of {"name", "arguments"}
- dataset: short identifier (e.g., "xlam60k"). Used as data_source.

Output rows follow verl's expected schema:
{
  "data_source": str,
  "prompt": list[{"role": str, "content": str}],
  "ability": str,
  "reward_model": {"ground_truth": str},  # JSON string
  "extra_info": {...}
}

Usage:
  Just run the script. Edit the constants INPUT_CSV / OUT_DIR / TEST_COUNT below if needed.
"""

from __future__ import annotations
import ast
import json
import os
import sys
from typing import Any, Dict, List, Tuple

# ======== Simple, editable constants ========
# Path to your input CSV. Edit this to your dataset path.
INPUT_CSV = os.getenv("INPUT_CSV", "/path/to/your/input.csv")

# Output directory for the converted dataset. Edit as needed.
OUT_DIR = os.getenv("OUT_DIR", "./verl_data")

# Number of samples to also write into test.parquet.
# Train will always contain ALL rows; test will contain the first TEST_COUNT rows (may overlap with train).
TEST_COUNT = 8


def _parse_json_field(text: str) -> Any:
    """Parse a JSON field from CSV content (same logic as code1)."""
    if text is None or not str(text).strip():
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return ast.literal_eval(text)


def _read_csv_rows(path: str, limit: int | None = None) -> List[Dict[str, str]]:
    import csv

    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = []
        for i, row in enumerate(reader):
            rows.append(row)
            if limit is not None and i + 1 >= limit:
                break
        return rows


def _split_rows(
    rows: List[Dict[str, Any]], test_ratio: float = 0.1, seed: int = 42
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows into (train, test).

    If a top-level 'split' column exists and contains values like 'train'/'test',
    use that. Otherwise, perform a random split with the given ratio and seed.
    """
    # Prefer explicit split column if present
    has_split_col = len(rows) > 0 and ("split" in rows[0])
    if has_split_col:
        train_rows = [r for r in rows if str(r.get("split", "")).lower() == "train"]
        test_rows = [r for r in rows if str(r.get("split", "")).lower() == "test"]
        # If the split column is present but empty/invalid, fall back to random
        if train_rows or test_rows:
            return train_rows, test_rows

    # Random split fallback
    import random

    rng = random.Random(seed)
    idxs = list(range(len(rows)))
    rng.shuffle(idxs)
    n = len(rows)
    n_test = max(1, int(n * test_ratio)) if n > 1 else 0
    test_idx = set(idxs[:n_test])
    train_rows = [rows[i] for i in range(n) if i not in test_idx]
    test_rows = [rows[i] for i in range(n) if i in test_idx]
    return train_rows, test_rows


def _to_verl_row(src: Dict[str, Any]) -> Dict[str, Any]:
    """Convert a CSV row to VERL format (same logic as code1)."""
    messages = _parse_json_field(src.get("messages", "")) or []
    solution = _parse_json_field(src.get("solution", "[]")) or []

    if not isinstance(messages, list):
        raise ValueError("`messages` must be a list of {role, content} objects")
    for m in messages:
        if not isinstance(m, dict) or "role" not in m or "content" not in m:
            raise ValueError("each message must be a dict with keys 'role' and 'content'")

    ground_truth_json = json.dumps(solution, ensure_ascii=False)
    data_source = (src.get("dataset") or "custom/multi_turn_tool_calls").strip()

    extra_info = {
        "id": str(src.get("id", ""))
    }

    return {
        "data_source": data_source,
        "prompt": messages,  # already HF-chat-like [{role, content}, ...]
        "ability": "tool_use",
        "reward_model": {"ground_truth": ground_truth_json},
        "extra_info": extra_info,
    }


def main() -> None:
    # Read CSV
    rows_all = _read_csv_rows(INPUT_CSV, limit=None)

    # Convert all rows
    converted: List[Dict[str, Any]] = []
    errors = 0
    for i, r in enumerate(rows_all):
        try:
            converted.append(_to_verl_row(r))
        except Exception as e:
            errors += 1
            print(f"[warn] row {i} skipped: {e}", file=sys.stderr)

    # Train uses ALL converted rows
    train_rows = converted
    # Test uses first TEST_COUNT rows (overlap allowed with train)
    test_rows = converted[: min(TEST_COUNT, len(converted))]

    os.makedirs(OUT_DIR, exist_ok=True)
    train_out = os.path.join(OUT_DIR, "mts_data_sync_1129_1130_alldata_merged_cleaned_1405samples_train.parquet")
    test_out = os.path.join(OUT_DIR, "mts_data_sync_1129_1130_alldata_merged_cleaned_1405samples_test.parquet")

    try:
        from datasets import Dataset

        Dataset.from_list(train_rows).to_parquet(train_out)
        Dataset.from_list(test_rows).to_parquet(test_out)
        print(
            f"Converted: total={len(rows_all)}, ok={len(converted)}, skipped={errors}.\n"
            f"Wrote train={len(train_rows)} -> {train_out}; test={len(test_rows)} -> {test_out}"
        )
    except Exception as e:
        # Fallback: write JSONL if datasets/pyarrow unavailable
        train_jsonl = os.path.join(OUT_DIR, "train.jsonl")
        test_jsonl = os.path.join(OUT_DIR, "test.jsonl")
        with open(train_jsonl, "w", encoding="utf-8") as f:
            for x in train_rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        with open(test_jsonl, "w", encoding="utf-8") as f:
            for x in test_rows:
                f.write(json.dumps(x, ensure_ascii=False) + "\n")
        print(
            f"[fallback] Could not write parquet ({e}). Wrote JSONL train={train_jsonl} test={test_jsonl}.\n"
            f"Converted: total={len(rows_all)}, ok={len(converted)}, skipped={errors}."
        )


if __name__ == "__main__":
    main()
