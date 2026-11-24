#!/usr/bin/env python3
"""Build this experiment's English-only SLAKE parquet without touching old data."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from datasets import Dataset


def _text(value) -> str:
    return "" if value is None else str(value).strip()


def _row(record: dict, root: Path, index: int) -> dict | None:
    if _text(record.get("q_lang")).lower() != "en":
        return None
    image_path = root / "imgs" / _text(record.get("img_name"))
    if not image_path.is_file():
        return None
    question = _text(record.get("question"))
    answer = _text(record.get("answer"))
    prompt = "\n".join(["<image>", "Question:", question, "", "Use the available tools only when useful.", "Return the final answer inside `<answer>...</answer>`."])
    return {
        "data_source": "slake",
        "prompt": [{"role": "user", "content": prompt}],
        # Keep image bytes out of the parquet. verl resolves this repository-relative
        # path inside the mounted project, matching the existing SLAKE views.
        "images": [{"bytes": None, "path": str(image_path)}],
        "reward_model": {"style": "rule", "ground_truth": answer},
        "extra_info": {
            "index": index,
            "question_id": _text(record.get("qid", index)),
            "question": question,
            "answer_text": answer,
            "image_path": str(image_path),
            "source": "slake",
            "benchmark": "SLAKE",
            "q_lang": "en",
            "answer_type": record.get("answer_type"),
            "content_type": record.get("content_type"),
            "location": record.get("location"),
            "modality": record.get("modality"),
        },
        "agent_name": "tool_agent",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    records = json.loads((args.dataset_root / "train.json").read_text(encoding="utf-8"))
    rows = [item for index, record in enumerate(records) if (item := _row(record, args.dataset_root, index)) is not None]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(args.output))
    print(json.dumps({"output": str(args.output), "rows": len(rows), "language": "en"}, indent=2))


if __name__ == "__main__":
    main()
