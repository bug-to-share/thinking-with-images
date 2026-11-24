from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


BASE_DIR = Path(".")
DEFAULT_OMNI_SFT = BASE_DIR / "data_curation/results/omnimedvqa_filtered/buckets/sft_candidate.jsonl"
DEFAULT_OMNI_RL = BASE_DIR / "data_curation/results/omnimedvqa_filtered/buckets/rl_candidate.jsonl"
DEFAULT_OPEN_SFT = BASE_DIR / "data_curation/results/slake_en_vqa_rad_open_qa_full/buckets/sft_candidate.jsonl"
DEFAULT_OPEN_RL = BASE_DIR / "data_curation/results/slake_en_vqa_rad_open_qa_full/buckets/rl_candidate.jsonl"


PoolKey = Tuple[str, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build a deterministic with-answer trace-generation sample from "
            "Qwen3-VL-8B direct-answer difficulty buckets."
        )
    )
    parser.add_argument("--omni-sft", default=str(DEFAULT_OMNI_SFT), help="OmniMedVQA sft_candidate jsonl")
    parser.add_argument("--omni-rl", default=str(DEFAULT_OMNI_RL), help="OmniMedVQA rl_candidate jsonl")
    parser.add_argument("--open-sft", default=str(DEFAULT_OPEN_SFT), help="SLAKE/VQA-RAD sft_candidate jsonl")
    parser.add_argument("--open-rl", default=str(DEFAULT_OPEN_RL), help="SLAKE/VQA-RAD rl_candidate jsonl")
    parser.add_argument("--output", required=True, help="Output jsonl for with-answer trace construction")
    parser.add_argument("--summary-output", default="", help="Optional summary json path")
    parser.add_argument("--total", type=int, default=1000, help="Total rows to sample")
    parser.add_argument("--sft-ratio", type=float, default=0.8, help="Fraction sampled from sft_candidate buckets")
    parser.add_argument(
        "--open-ratio",
        type=float,
        default=0.3,
        help="Fraction sampled from SLAKE/VQA-RAD buckets; the rest comes from OmniMedVQA",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--dedupe-key",
        choices=["question_id", "image_question", "none"],
        default="question_id",
        help="Deduplicate rows before sampling",
    )
    parser.add_argument(
        "--include-source-fields",
        action="store_true",
        help="Add trace_sample_source/trace_sample_bucket fields to each output row",
    )
    parser.add_argument(
        "--max-dataset-fraction",
        type=float,
        default=0.2,
        help=(
            "Soft cap for any single dataset/benchmark in the sampled output. "
            "Use 0 to disable. Default 0.2 keeps RadImageNet-like sources from dominating."
        ),
    )
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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def row_key(row: Dict[str, Any], mode: str) -> str:
    if mode == "none":
        return ""
    question_id = str(row.get("question_id") or "")
    dataset = str(row.get("dataset") or row.get("benchmark") or "")
    if mode == "question_id":
        return f"{dataset}::{question_id}"
    image_path = str(row.get("image_path_abs") or row.get("image_path_rel") or "")
    question = str(row.get("question") or "")
    return f"{dataset}::{image_path}::{question}"


def dedupe_rows(rows: List[Dict[str, Any]], mode: str) -> List[Dict[str, Any]]:
    if mode == "none":
        return rows
    seen = set()
    output: List[Dict[str, Any]] = []
    for row in rows:
        key = row_key(row, mode)
        if key in seen:
            continue
        seen.add(key)
        output.append(row)
    return output


def load_pools(args: argparse.Namespace) -> Dict[PoolKey, List[Dict[str, Any]]]:
    specs = {
        ("omni", "sft_candidate"): Path(args.omni_sft),
        ("omni", "rl_candidate"): Path(args.omni_rl),
        ("open_medvqa", "sft_candidate"): Path(args.open_sft),
        ("open_medvqa", "rl_candidate"): Path(args.open_rl),
    }
    pools: Dict[PoolKey, List[Dict[str, Any]]] = {}
    for key, path in specs.items():
        rows = dedupe_rows(read_jsonl(path), args.dedupe_key)
        pools[key] = rows
    return pools


def validate_ratios(args: argparse.Namespace) -> None:
    if args.total <= 0:
        raise ValueError("--total must be positive")
    for name in ("sft_ratio", "open_ratio"):
        value = float(getattr(args, name))
        if value < 0.0 or value > 1.0:
            raise ValueError(f"--{name.replace('_', '-')} must be in [0, 1], got {value}")
    if args.max_dataset_fraction < 0.0 or args.max_dataset_fraction > 1.0:
        raise ValueError(f"--max-dataset-fraction must be in [0, 1], got {args.max_dataset_fraction}")


def initial_targets(total: int, sft_ratio: float, open_ratio: float) -> Dict[PoolKey, int]:
    open_total = round(total * open_ratio)
    omni_total = total - open_total
    open_sft = round(open_total * sft_ratio)
    omni_sft = round(omni_total * sft_ratio)
    return {
        ("omni", "sft_candidate"): omni_sft,
        ("omni", "rl_candidate"): omni_total - omni_sft,
        ("open_medvqa", "sft_candidate"): open_sft,
        ("open_medvqa", "rl_candidate"): open_total - open_sft,
    }


def sample_with_redistribution(
    pools: Dict[PoolKey, List[Dict[str, Any]]],
    targets: Dict[PoolKey, int],
    rng: random.Random,
    total: int,
    max_dataset_fraction: float,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    shuffled: Dict[PoolKey, List[Dict[str, Any]]] = {}
    for key, rows in pools.items():
        copied = list(rows)
        rng.shuffle(copied)
        shuffled[key] = copied

    selected: List[Dict[str, Any]] = []
    used_counts: Dict[PoolKey, int] = {}
    selected_keys = set()
    dataset_counts: Counter[str] = Counter()
    max_dataset_count = int(total * max_dataset_fraction) if max_dataset_fraction > 0 else 0

    def can_take(row: Dict[str, Any]) -> bool:
        key_value = row_key(row, "image_question")
        if key_value in selected_keys:
            return False
        if max_dataset_count > 0 and dataset_counts[dataset_key(row)] >= max_dataset_count:
            return False
        return True

    def take_row(key: PoolKey, row: Dict[str, Any]) -> None:
        selected_keys.add(row_key(row, "image_question"))
        dataset_counts[dataset_key(row)] += 1
        used_counts[key] = used_counts.get(key, 0) + 1
        selected.append(row)

    shortage = 0
    for key, target in targets.items():
        before = len(selected)
        for row in shuffled.get(key, []):
            if used_counts.get(key, 0) >= target:
                break
            if can_take(row):
                take_row(key, row)
        shortage += max(0, target - (len(selected) - before))

    if shortage > 0 or len(selected) < total:
        remainder: List[Tuple[PoolKey, Dict[str, Any]]] = []
        for key, rows in shuffled.items():
            for row in rows:
                remainder.append((key, row))
        rng.shuffle(remainder)
        for key, row in remainder:
            if len(selected) >= total:
                break
            if can_take(row):
                take_row(key, row)

    if len(selected) < total and max_dataset_count > 0:
        # If the cap is too strict for a small pool, fill the remainder without it.
        remainder = []
        for key, rows in shuffled.items():
            for row in rows:
                if row_key(row, "image_question") not in selected_keys:
                    remainder.append((key, row))
        rng.shuffle(remainder)
        for key, row in remainder:
            if len(selected) >= total:
                break
            take_row(key, row)

    rng.shuffle(selected)
    if len(selected) > total:
        selected = selected[:total]
        used_counts = defaultdict(int)
        for row in selected:
            source = str(row.get("trace_sample_source") or infer_source(row))
            bucket = str(row.get("difficulty_bucket") or row.get("recommended_use") or "")
            used_counts[(source, bucket)] += 1

    summary = {
        "targets": {"/".join(key): value for key, value in targets.items()},
        "selected_by_pool": {"/".join(key): value for key, value in sorted(used_counts.items())},
        "selected_total": len(selected),
        "max_dataset_fraction": max_dataset_fraction,
        "max_dataset_count": max_dataset_count,
    }
    return selected, summary


def infer_source(row: Dict[str, Any]) -> str:
    if row.get("dataset"):
        return "omni"
    return "open_medvqa"


def dataset_key(row: Dict[str, Any]) -> str:
    return str(row.get("dataset") or row.get("benchmark") or "unknown")


def annotate_rows(rows: List[Dict[str, Any]], include_source_fields: bool) -> List[Dict[str, Any]]:
    output: List[Dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        if include_source_fields:
            item["trace_sample_source"] = infer_source(item)
            item["trace_sample_bucket"] = item.get("difficulty_bucket") or item.get("recommended_use")
        output.append(item)
    return output


def summarize_rows(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    source_counter = Counter()
    bucket_counter = Counter()
    correct_counter = Counter()
    dataset_counter = Counter()
    benchmark_counter = Counter()
    missing_images = 0

    for row in rows:
        source_counter[infer_source(row)] += 1
        bucket_counter[str(row.get("difficulty_bucket") or row.get("recommended_use") or "unknown")] += 1
        correct_counter[str(row.get("correct_count", "unknown"))] += 1
        if row.get("dataset"):
            dataset_counter[str(row.get("dataset"))] += 1
        if row.get("benchmark"):
            benchmark_counter[str(row.get("benchmark"))] += 1
        image_path = row.get("image_path_abs")
        if image_path and not Path(str(image_path)).exists():
            missing_images += 1

    return {
        "num_rows": len(rows),
        "source_counts": dict(source_counter),
        "bucket_counts": dict(bucket_counter),
        "correct_count_distribution": dict(sorted(correct_counter.items(), key=lambda kv: int(kv[0]) if kv[0].isdigit() else 999)),
        "top_datasets": dict(dataset_counter.most_common(20)),
        "benchmark_counts": dict(benchmark_counter),
        "missing_image_paths": missing_images,
    }


def main() -> None:
    args = parse_args()
    validate_ratios(args)
    rng = random.Random(args.seed)

    pools = load_pools(args)
    targets = initial_targets(args.total, args.sft_ratio, args.open_ratio)
    selected, allocation_summary = sample_with_redistribution(
        pools,
        targets,
        rng,
        args.total,
        args.max_dataset_fraction,
    )
    output_rows = annotate_rows(selected, args.include_source_fields)

    output_path = Path(args.output)
    write_jsonl(output_path, output_rows)

    summary = {
        "output": str(output_path),
        "seed": args.seed,
        "total_requested": args.total,
        "sft_ratio": args.sft_ratio,
        "open_ratio": args.open_ratio,
        "dedupe_key": args.dedupe_key,
        "pool_sizes": {"/".join(key): len(value) for key, value in sorted(pools.items())},
        **allocation_summary,
        **summarize_rows(output_rows),
    }

    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_suffix(".summary.json")
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
