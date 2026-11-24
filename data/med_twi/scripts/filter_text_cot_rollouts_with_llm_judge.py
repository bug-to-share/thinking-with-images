from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib.util
import json
import os
from collections import Counter
from pathlib import Path
from typing import Any, Awaitable, Callable

import pyarrow as pa
import pyarrow.parquet as pq

from filter_text_cot_by_base_rollout import (
    load_rows,
    read_rollout_records,
    sha256_file,
    stable_json,
)


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "data/med_twi/text_cot/train.parquet"
DEFAULT_VALIDATION = PROJECT_ROOT / "eval/benchmarks/SLAKE/views/slake_validation/validation.parquet"
DEFAULT_ROLLOUTS = (
    PROJECT_ROOT / "data/med_twi/text_cot/base_rollout_qwen3_vl_4b_n8/rollouts.jsonl"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT
    / "data/med_twi/text_cot/base_rollout_qwen3_vl_4b_n8/llm_as_judge_qwen3_14b"
)

AsyncRewardFn = Callable[..., Awaitable[dict[str, float]]]


def load_async_reward(path: Path) -> AsyncRewardFn:
    spec = importlib.util.spec_from_file_location("med_twi_llm_judge_filter_reward", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import reward function from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    compute_score = module.compute_score
    if not asyncio.iscoroutinefunction(compute_score):
        raise TypeError(f"Expected async compute_score in {path}")
    return compute_score


def read_selection_records(path: Path, config_id: str) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("config_id") != config_id:
                raise ValueError(
                    f"Existing judge record at {path}:{line_number} uses a different configuration"
                )
            row_index = int(record["row_index"])
            if row_index in records:
                raise ValueError(f"Duplicate row_index {row_index} at {path}:{line_number}")
            records[row_index] = record
    return records


async def score_one_output(
    *,
    semaphore: asyncio.Semaphore,
    compute_score: AsyncRewardFn,
    row: dict[str, Any],
    output: str,
) -> dict[str, float]:
    async with semaphore:
        return await compute_score(
            data_source=row.get("data_source", ""),
            solution_str=output,
            ground_truth=row["reward_model"]["ground_truth"],
            extra_info=row.get("extra_info"),
        )


async def score_one_row(
    *,
    row_index: int,
    row: dict[str, Any],
    rollout_record: dict[str, Any],
    samples: int,
    semaphore: asyncio.Semaphore,
    compute_score: AsyncRewardFn,
    config_id: str,
    retries: int,
) -> dict[str, Any]:
    outputs = rollout_record["outputs"]
    if len(outputs) != samples:
        raise ValueError(f"Rollout row {row_index} has {len(outputs)} samples, expected {samples}")

    last_error: Exception | None = None
    for attempt in range(retries + 1):
        try:
            scores = await asyncio.gather(
                *(
                    score_one_output(
                        semaphore=semaphore,
                        compute_score=compute_score,
                        row=row,
                        output=output,
                    )
                    for output in outputs
                )
            )
            failed = [index for index, score in enumerate(scores) if score["judge_error"] > 0]
            if failed:
                raise RuntimeError(f"judge errors for sample indices {failed}")

            correctness = [int(score["correctness"] > 0.0) for score in scores]
            return {
                "row_index": row_index,
                "qid": (row.get("extra_info") or {}).get("qid"),
                "correct_count": sum(correctness),
                "correctness": correctness,
                "exact_match": [int(score["exact_match"] > 0.0) for score in scores],
                "judge_called": [int(score["judge_called"] > 0.0) for score in scores],
                "judge_positive": [int(score["judge_positive"] > 0.0) for score in scores],
                "format_valid": [int(score["format_reward"] > 0.0) for score in scores],
                "config_id": config_id,
            }
        except Exception as exc:
            last_error = exc
            if attempt < retries:
                await asyncio.sleep(1.0 * (2**attempt))
    raise RuntimeError(f"row {row_index} failed after {retries + 1} attempt(s): {last_error}")


def write_final_outputs(
    *,
    args: argparse.Namespace,
    table: pa.Table,
    records: dict[int, dict[str, Any]],
    input_sha256: str,
    rollout_sha256: str,
    reward_sha256: str,
    config_id: str,
) -> None:
    ordered = [records[index] for index in range(table.num_rows)]
    distribution = Counter(int(record["correct_count"]) for record in ordered)
    selected_indices = [
        int(record["row_index"])
        for record in ordered
        if args.min_correct <= int(record["correct_count"]) <= args.max_correct
    ]
    selected_set = set(selected_indices)

    with args.selection_output.open("w", encoding="utf-8") as handle:
        for record in ordered:
            final_record = dict(record)
            final_record["selected"] = int(record["row_index"]) in selected_set
            handle.write(json.dumps(final_record, ensure_ascii=False) + "\n")

    selected_table = table.take(pa.array(selected_indices, type=pa.int64()))
    pq.write_table(selected_table, args.output)

    exact_positive = sum(sum(record["exact_match"]) for record in ordered)
    judge_called = sum(sum(record["judge_called"]) for record in ordered)
    judge_positive = sum(sum(record["judge_positive"]) for record in ordered)
    format_valid = sum(sum(record["format_valid"]) for record in ordered)
    total_samples = table.num_rows * args.samples
    summary = {
        "method": "base_model_rollout_llm_as_judge_difficulty_filter",
        "input": {
            "path": str(args.input),
            "sha256": input_sha256,
            "rows": table.num_rows,
        },
        "rollouts": {
            "path": str(args.rollouts),
            "sha256": rollout_sha256,
            "samples_per_question": args.samples,
            "total_samples": total_samples,
        },
        "judge": {
            "base_url": os.environ.get("LLM_AS_A_JUDGE_BASE", ""),
            "model": os.environ.get("LLM_AS_A_JUDGE_MODEL", "auto-discovered"),
            "reward_function": str(args.reward_path),
            "reward_sha256": reward_sha256,
            "config_id": config_id,
            "format_valid_samples": format_valid,
            "exact_positive_samples": exact_positive,
            "judge_called_samples": judge_called,
            "judge_positive_samples": judge_positive,
            "total_correct_samples": exact_positive + judge_positive,
        },
        "selection": {
            "criterion": f"{args.min_correct} <= binary_correctness_count <= {args.max_correct}",
            "correct_count_distribution": {
                str(count): distribution.get(count, 0) for count in range(args.samples + 1)
            },
            "selected_rows": len(selected_indices),
            "excluded_zero_correct": distribution.get(0, 0),
            "excluded_all_correct": distribution.get(args.samples, 0),
            "output_path": str(args.output),
            "output_sha256": sha256_file(args.output),
            "selection_records": str(args.selection_output),
        },
        "validation": {
            "path": str(args.validation),
            "sha256": sha256_file(args.validation),
            "policy": "reused unchanged; never filtered",
        },
    }
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


async def run(args: argparse.Namespace) -> None:
    args.input = args.input.resolve()
    args.rollouts = args.rollouts.resolve()
    args.validation = args.validation.resolve()
    args.reward_path = args.reward_path.resolve()
    args.output = args.output.resolve()
    args.selection_output = args.selection_output.resolve()
    args.progress_output = args.progress_output.resolve()
    args.summary_output = args.summary_output.resolve()
    for path in (args.output, args.selection_output, args.progress_output, args.summary_output):
        path.parent.mkdir(parents=True, exist_ok=True)

    table, rows = load_rows(args.input)
    rollout_records = read_rollout_records(args.rollouts)
    expected_indices = set(range(len(rows)))
    if set(rollout_records) != expected_indices:
        raise ValueError("Saved rollouts do not provide complete, exact coverage of the input dataset")

    input_sha256 = sha256_file(args.input)
    rollout_sha256 = sha256_file(args.rollouts)
    reward_sha256 = sha256_file(args.reward_path)
    config = {
        "input_sha256": input_sha256,
        "rollout_sha256": rollout_sha256,
        "reward_sha256": reward_sha256,
        "judge_base_url": os.environ.get("LLM_AS_A_JUDGE_BASE", ""),
        "judge_model": os.environ.get("LLM_AS_A_JUDGE_MODEL", ""),
        "samples": args.samples,
        "min_correct": args.min_correct,
        "max_correct": args.max_correct,
    }
    config_id = hashlib.sha256(stable_json(config).encode("utf-8")).hexdigest()
    completed = read_selection_records(args.progress_output, config_id)
    pending = [index for index in range(len(rows)) if index not in completed]
    if args.limit > 0:
        pending = pending[: args.limit]
    print(
        f"input_rows={len(rows)} completed={len(completed)} pending_this_run={len(pending)} "
        f"concurrency={args.concurrency} progress={args.progress_output}",
        flush=True,
    )

    compute_score = load_async_reward(args.reward_path)
    semaphore = asyncio.Semaphore(args.concurrency)
    tasks = [
        asyncio.create_task(
            score_one_row(
                row_index=index,
                row=rows[index],
                rollout_record=rollout_records[index],
                samples=args.samples,
                semaphore=semaphore,
                compute_score=compute_score,
                config_id=config_id,
                retries=args.retries,
            )
        )
        for index in pending
    ]

    errors: list[str] = []
    finished_now = 0
    with args.progress_output.open("a", encoding="utf-8") as handle:
        for task in asyncio.as_completed(tasks):
            try:
                record = await task
            except Exception as exc:
                errors.append(str(exc))
                print(f"ERROR {exc}", flush=True)
                continue
            completed[int(record["row_index"])] = record
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            handle.flush()
            finished_now += 1
            if finished_now % args.progress_every == 0 or finished_now == len(pending):
                print(
                    f"judged_rows={finished_now}/{len(pending)} "
                    f"total_completed={len(completed)}/{len(rows)}",
                    flush=True,
                )

    if errors:
        raise RuntimeError(
            f"{len(errors)} rows failed; rerun to resume. First error: {errors[0]}"
        )
    if len(completed) != len(rows):
        print("Partial run complete; rerun without --limit to finish and build parquet.", flush=True)
        return

    write_final_outputs(
        args=args,
        table=table,
        records=completed,
        input_sha256=input_sha256,
        rollout_sha256=rollout_sha256,
        reward_sha256=reward_sha256,
        config_id=config_id,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Re-score saved n=8 VQA rollouts with an LLM judge and retain correctness counts 1-7."
    )
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--rollouts", type=Path, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    parser.add_argument("--reward-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_DIR / "train_correct_1_to_7.parquet")
    parser.add_argument("--selection-output", type=Path, default=DEFAULT_OUTPUT_DIR / "selection.jsonl")
    parser.add_argument("--progress-output", type=Path, default=DEFAULT_OUTPUT_DIR / "progress.jsonl")
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_OUTPUT_DIR / "summary.json")
    parser.add_argument("--samples", type=int, default=8)
    parser.add_argument("--min-correct", type=int, default=1)
    parser.add_argument("--max-correct", type=int, default=7)
    parser.add_argument("--concurrency", type=int, default=128)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--progress-every", type=int, default=50)
    parser.add_argument("--limit", type=int, default=0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not (0 <= args.min_correct <= args.max_correct <= args.samples):
        raise ValueError("Require 0 <= min-correct <= max-correct <= samples")
    if args.concurrency < 1:
        raise ValueError("--concurrency must be positive")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
