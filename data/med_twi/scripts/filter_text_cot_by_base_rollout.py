from __future__ import annotations

import argparse
import base64
import hashlib
import importlib.util
import json
import mimetypes
import os
import random
import time
from collections import Counter
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pyarrow as pa
import pyarrow.parquet as pq


PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = PROJECT_ROOT / "data/med_twi/text_cot/train.parquet"
DEFAULT_VALIDATION = PROJECT_ROOT / "eval/benchmarks/SLAKE/views/slake_validation/validation.parquet"
DEFAULT_ARTIFACT_DIR = (
    PROJECT_ROOT / "data/med_twi/text_cot/base_rollout_qwen3_vl_4b_n8"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def load_rows(path: Path) -> tuple[pa.Table, list[dict[str, Any]]]:
    table = pq.read_table(path)
    required = {"prompt", "images", "reward_model", "extra_info"}
    missing = required.difference(table.column_names)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return table, table.to_pylist()


def image_data_url(image: dict[str, Any], project_root: Path) -> str:
    raw_bytes = image.get("bytes")
    image_path = image.get("path")
    suffix = ""
    if raw_bytes is None:
        if not image_path:
            raise ValueError("Image entry contains neither bytes nor a path")
        path = Path(image_path)
        if not path.is_absolute():
            path = project_root / path
        raw_bytes = path.read_bytes()
        suffix = path.suffix
    elif image_path:
        suffix = Path(image_path).suffix

    mime_type = mimetypes.guess_type(f"image{suffix}")[0] or "image/jpeg"
    encoded = base64.b64encode(raw_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def multimodal_messages(row: dict[str, Any], project_root: Path) -> list[dict[str, Any]]:
    prompts = row["prompt"]
    images = row["images"] or []
    if not isinstance(prompts, list):
        raise TypeError(f"Expected prompt to be a list, got {type(prompts).__name__}")

    result: list[dict[str, Any]] = []
    image_index = 0
    for message in prompts:
        role = message["role"]
        content = message["content"]
        if role != "user" or "<image>" not in content:
            result.append({"role": role, "content": content})
            continue

        parts: list[dict[str, Any]] = []
        text_parts = content.split("<image>")
        for part_index, text_part in enumerate(text_parts):
            if part_index > 0:
                if image_index >= len(images):
                    raise ValueError("Prompt contains more <image> tags than image entries")
                parts.append(
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_data_url(images[image_index], project_root)
                        },
                    }
                )
                image_index += 1
            if text_part:
                parts.append({"type": "text", "text": text_part})
        result.append({"role": role, "content": parts})

    if image_index != len(images):
        raise ValueError(
            f"Dataset row contains {len(images)} images but prompt consumed {image_index}"
        )
    return result


def completion_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/chat/completions"):
        return base
    if not base.endswith("/v1"):
        base += "/v1"
    return base + "/chat/completions"


def completion_urls(base_urls: str) -> list[str]:
    urls = [completion_url(item.strip()) for item in base_urls.split(",") if item.strip()]
    if not urls:
        raise ValueError("--base-url must contain at least one endpoint")
    return urls


def request_json(url: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    request = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"},
        method="POST",
    )
    try:
        with urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code} from rollout server: {body[:1000]}") from exc
    except URLError as exc:
        raise RuntimeError(f"Cannot reach rollout server at {url}: {exc}") from exc


def generate_one(
    row_index: int,
    row: dict[str, Any],
    args: argparse.Namespace,
    input_sha256: str,
    config_id: str,
) -> dict[str, Any]:
    payload = {
        "model": args.model,
        "messages": multimodal_messages(row, args.project_root),
        "n": args.samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed + row_index,
    }
    last_error: Exception | None = None
    for attempt in range(args.retries + 1):
        try:
            response = request_json(
                args.urls[row_index % len(args.urls)], payload, args.timeout
            )
            choices = sorted(response.get("choices", []), key=lambda item: item["index"])
            outputs = [choice["message"]["content"] or "" for choice in choices]
            if len(outputs) != args.samples:
                raise RuntimeError(
                    f"Expected {args.samples} choices, server returned {len(outputs)}"
                )
            extra_info = row["extra_info"] or {}
            return {
                "row_index": row_index,
                "qid": extra_info.get("qid"),
                "question": extra_info.get("question"),
                "ground_truth": row["reward_model"]["ground_truth"],
                "outputs": outputs,
                "finish_reasons": [choice.get("finish_reason") for choice in choices],
                "input_sha256": input_sha256,
                "config_id": config_id,
            }
        except Exception as exc:  # The failed row remains resumable on the next run.
            last_error = exc
            if attempt < args.retries:
                delay = args.retry_delay * (2**attempt) + random.random()
                time.sleep(delay)
    raise RuntimeError(f"row {row_index} failed after {args.retries + 1} attempts: {last_error}")


def read_rollout_records(path: Path) -> dict[int, dict[str, Any]]:
    records: dict[int, dict[str, Any]] = {}
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            row_index = int(record["row_index"])
            if row_index in records:
                raise ValueError(f"Duplicate row_index {row_index} at {path}:{line_number}")
            records[row_index] = record
    return records


def rollout_command(args: argparse.Namespace) -> None:
    args.input = args.input.resolve()
    args.project_root = args.project_root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    _, rows = load_rows(args.input)
    input_sha256 = sha256_file(args.input)
    config = {
        "input_sha256": input_sha256,
        "model": args.model,
        "samples": args.samples,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
        "seed": args.seed,
        "reward_scope": "generation_only",
    }
    config_id = hashlib.sha256(stable_json(config).encode("utf-8")).hexdigest()

    completed = read_rollout_records(args.output)
    for row_index, record in completed.items():
        if record.get("input_sha256") != input_sha256 or record.get("config_id") != config_id:
            raise ValueError(
                f"Existing rollout record {row_index} was produced with a different dataset "
                "or sampling configuration; choose a new output path"
            )
    pending = [index for index in range(len(rows)) if index not in completed]
    if args.limit > 0:
        pending = pending[: args.limit]
    print(
        f"input_rows={len(rows)} completed={len(completed)} pending_this_run={len(pending)} "
        f"output={args.output}",
        flush=True,
    )
    if not pending:
        return

    errors: list[str] = []
    with args.output.open("a", encoding="utf-8") as output_handle:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures: dict[Future[dict[str, Any]], int] = {
                executor.submit(
                    generate_one,
                    row_index,
                    rows[row_index],
                    args,
                    input_sha256,
                    config_id,
                ): row_index
                for row_index in pending
            }
            finished_now = 0
            for future in as_completed(futures):
                row_index = futures[future]
                try:
                    record = future.result()
                except Exception as exc:
                    errors.append(str(exc))
                    print(f"ERROR {exc}", flush=True)
                    continue
                output_handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                output_handle.flush()
                os.fsync(output_handle.fileno())
                finished_now += 1
                if finished_now % args.progress_every == 0 or finished_now == len(pending):
                    print(
                        f"generated={finished_now}/{len(pending)} total_completed="
                        f"{len(completed) + finished_now}",
                        flush=True,
                    )
    if errors:
        raise RuntimeError(
            f"{len(errors)} rollout rows failed. Successful rows were saved; rerun to resume. "
            f"First error: {errors[0]}"
        )


def load_reward_function(path: Path) -> Callable[..., dict[str, float]]:
    spec = importlib.util.spec_from_file_location("med_twi_filter_reward", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import reward function from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.compute_score


def filter_command(args: argparse.Namespace) -> None:
    args.input = args.input.resolve()
    table, rows = load_rows(args.input)
    input_sha256 = sha256_file(args.input)
    records = read_rollout_records(args.rollouts)
    expected_indices = set(range(len(rows)))
    actual_indices = set(records)
    if actual_indices != expected_indices:
        missing = sorted(expected_indices - actual_indices)
        extra = sorted(actual_indices - expected_indices)
        raise ValueError(
            f"Rollout coverage does not match dataset: missing={len(missing)} {missing[:10]}, "
            f"extra={len(extra)} {extra[:10]}"
        )

    compute_score = load_reward_function(args.reward_path.resolve())
    selected_indices: list[int] = []
    distribution: Counter[int] = Counter()
    selection_records: list[dict[str, Any]] = []
    for row_index, row in enumerate(rows):
        record = records[row_index]
        if record.get("input_sha256") != input_sha256:
            raise ValueError(f"Rollout row {row_index} belongs to a different input parquet")
        outputs = record["outputs"]
        if len(outputs) != args.samples:
            raise ValueError(
                f"Rollout row {row_index} has {len(outputs)} samples, expected {args.samples}"
            )
        ground_truth = row["reward_model"]["ground_truth"]
        scores = [
            compute_score(
                data_source=row.get("data_source", ""),
                solution_str=output,
                ground_truth=ground_truth,
                extra_info=row.get("extra_info"),
            )
            for output in outputs
        ]
        correctness = [int(score["exact_match"] > 0.0) for score in scores]
        correct_count = sum(correctness)
        distribution[correct_count] += 1
        selected = args.min_correct <= correct_count <= args.max_correct
        if selected:
            selected_indices.append(row_index)
        selection_records.append(
            {
                "row_index": row_index,
                "qid": (row.get("extra_info") or {}).get("qid"),
                "correct_count": correct_count,
                "correctness": correctness,
                "selected": selected,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    selected_table = table.take(pa.array(selected_indices, type=pa.int64()))
    pq.write_table(selected_table, args.output)
    with args.selection_output.open("w", encoding="utf-8") as handle:
        for record in selection_records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")

    summary = {
        "method": "base_model_rollout_difficulty_filter",
        "input": {
            "path": str(args.input),
            "sha256": input_sha256,
            "rows": len(rows),
        },
        "rollouts": {
            "path": str(args.rollouts.resolve()),
            "samples_per_question": args.samples,
            "correct_count_distribution": {
                str(count): distribution.get(count, 0) for count in range(args.samples + 1)
            },
        },
        "selection": {
            "criterion": f"{args.min_correct} <= strict_exact_match_count <= {args.max_correct}",
            "reward_function": str(args.reward_path.resolve()),
            "selected_rows": len(selected_indices),
            "excluded_zero_correct": distribution.get(0, 0),
            "excluded_all_correct": distribution.get(args.samples, 0),
            "output_path": str(args.output.resolve()),
            "output_sha256": sha256_file(args.output),
            "selection_records": str(args.selection_output.resolve()),
        },
        "validation": {
            "path": str(args.validation.resolve()),
            "sha256": sha256_file(args.validation.resolve()),
            "policy": "reused unchanged; never filtered",
        },
    }
    args.summary_output.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate base-model VQA rollouts and retain questions with non-zero variance."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    rollout = subparsers.add_parser("rollout", help="Generate resumable N-sample rollouts.")
    rollout.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    rollout.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_DIR / "rollouts.jsonl")
    rollout.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    rollout.add_argument(
        "--base-url",
        default="http://127.0.0.1:30000",
        help="One endpoint or a comma-separated list of endpoints for round-robin dispatch.",
    )
    rollout.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    rollout.add_argument("--samples", type=int, default=8)
    rollout.add_argument("--temperature", type=float, default=1.0)
    rollout.add_argument("--top-p", type=float, default=1.0)
    rollout.add_argument("--top-k", type=int, default=-1)
    rollout.add_argument("--max-tokens", type=int, default=1024)
    rollout.add_argument("--seed", type=int, default=42)
    rollout.add_argument("--concurrency", type=int, default=64)
    rollout.add_argument("--timeout", type=float, default=300.0)
    rollout.add_argument("--retries", type=int, default=4)
    rollout.add_argument("--retry-delay", type=float, default=2.0)
    rollout.add_argument("--progress-every", type=int, default=50)
    rollout.add_argument("--limit", type=int, default=0, help="Generate only this many pending rows.")
    rollout.set_defaults(func=rollout_command)

    filter_parser = subparsers.add_parser(
        "filter", help="Score saved rollouts and write the selected training parquet."
    )
    filter_parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    filter_parser.add_argument(
        "--rollouts", type=Path, default=DEFAULT_ARTIFACT_DIR / "rollouts.jsonl"
    )
    filter_parser.add_argument(
        "--output", type=Path, default=DEFAULT_ARTIFACT_DIR / "train_correct_1_to_7.parquet"
    )
    filter_parser.add_argument(
        "--selection-output",
        type=Path,
        default=DEFAULT_ARTIFACT_DIR / "selection.jsonl",
    )
    filter_parser.add_argument(
        "--summary-output", type=Path, default=DEFAULT_ARTIFACT_DIR / "summary.json"
    )
    filter_parser.add_argument("--validation", type=Path, default=DEFAULT_VALIDATION)
    filter_parser.add_argument("--reward-path", type=Path, required=True)
    filter_parser.add_argument("--samples", type=int, default=8)
    filter_parser.add_argument("--min-correct", type=int, default=1)
    filter_parser.add_argument("--max-correct", type=int, default=7)
    filter_parser.set_defaults(func=filter_command)
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.command == "rollout":
        if args.samples < 2:
            raise ValueError("--samples must be at least 2")
        if args.concurrency < 1:
            raise ValueError("--concurrency must be positive")
        args.urls = completion_urls(args.base_url)
    else:
        if not (0 <= args.min_correct <= args.max_correct <= args.samples):
            raise ValueError("Require 0 <= min-correct <= max-correct <= samples")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.selection_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    validate_args(args)
    args.func(args)


if __name__ == "__main__":
    main()
