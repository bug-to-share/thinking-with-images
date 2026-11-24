#!/usr/bin/env python3
"""Build a row-aligned rollout file for the rows selected by R3."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--selection", type=Path, required=True)
    parser.add_argument("--rollouts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    selected = [
        record for record in read_jsonl(args.selection) if bool(record.get("selected"))
    ]
    selected.sort(key=lambda record: int(record["row_index"]))
    rollouts = {
        int(record["row_index"]): record for record in read_jsonl(args.rollouts)
    }
    missing = [int(record["row_index"]) for record in selected if int(record["row_index"]) not in rollouts]
    if missing:
        raise ValueError(f"Missing rollout rows for selected indices: {missing[:5]}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for new_index, selection_record in enumerate(selected):
            record = dict(rollouts[int(selection_record["row_index"])])
            record["row_index"] = new_index
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"selected_rows={len(selected)} output={args.output}")


if __name__ == "__main__":
    main()
