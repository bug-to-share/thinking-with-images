from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Bucket curated VQA items by correct_count.")
    parser.add_argument("--input", required=True, help="Path to results.jsonl")
    parser.add_argument("--output-dir", required=True, help="Directory for bucketed jsonl outputs")
    parser.add_argument("--quarantine-max", type=int, default=0, help="0..quarantine-max => quarantine_hard")
    parser.add_argument("--sft-min", type=int, default=1, help="sft lower bound inclusive")
    parser.add_argument("--sft-max", type=int, default=3, help="sft upper bound inclusive")
    parser.add_argument("--rl-min", type=int, default=4, help="rl lower bound inclusive")
    parser.add_argument("--rl-max", type=int, default=6, help="rl upper bound inclusive")
    parser.add_argument("--easy-min", type=int, default=7, help="easy lower bound inclusive")
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def bucket_name(correct_count: int, args: argparse.Namespace) -> str:
    if correct_count <= args.quarantine_max:
        return "quarantine_hard"
    if args.sft_min <= correct_count <= args.sft_max:
        return "sft_candidate"
    if args.rl_min <= correct_count <= args.rl_max:
        return "rl_candidate"
    if correct_count >= args.easy_min:
        return "easy_filtered"
    raise ValueError(f"correct_count={correct_count} does not fit any configured bucket")


def augment_row(row: Dict[str, Any], bucket: str) -> Dict[str, Any]:
    output = dict(row)
    output["difficulty_bucket"] = bucket
    output["recommended_use"] = bucket
    return output


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = read_jsonl(input_path)
    buckets: Dict[str, List[Dict[str, Any]]] = {
        "easy_filtered": [],
        "rl_candidate": [],
        "sft_candidate": [],
        "quarantine_hard": [],
    }

    for row in rows:
        correct_count = int(row.get("correct_count", 0) or 0)
        bucket = bucket_name(correct_count, args)
        buckets[bucket].append(augment_row(row, bucket))

    for name, bucket_rows in buckets.items():
        write_jsonl(output_dir / f"{name}.jsonl", bucket_rows)

    summary = {
        "input": str(input_path),
        "output_dir": str(output_dir),
        "total_rows": len(rows),
        "bucket_thresholds": {
            "quarantine_hard": [0, args.quarantine_max],
            "sft_candidate": [args.sft_min, args.sft_max],
            "rl_candidate": [args.rl_min, args.rl_max],
            "easy_filtered": [args.easy_min, "max"],
        },
        "bucket_counts": {name: len(bucket_rows) for name, bucket_rows in buckets.items()},
    }
    (output_dir / "bucket_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
