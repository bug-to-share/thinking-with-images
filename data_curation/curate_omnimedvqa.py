#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import io
import json
import mimetypes
import os
import re
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from openai import OpenAI
from PIL import Image


DEFAULT_DATASET_ROOT = "./dataset/OmniMedVQA/OmniMedVQA"
DEFAULT_OUTPUT_ROOT = "./data_curation/results"
DEFAULT_ARTIFACT_ROOT = "./data_curation/artifacts"
DEFAULT_BASE_URL = "http://127.0.0.1:31000/v1"
DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_NUM_SAMPLES = 8
DEFAULT_EASY_THRESHOLD = 7
DEFAULT_NUM_WORKERS = 2
DEFAULT_MAX_TOKENS = 512
DEFAULT_TEMPERATURE = 0.6
DEFAULT_TOP_P = 0.95
DEFAULT_TIMEOUT_RETRIES = 3
DEFAULT_REQUEST_TIMEOUT = 180
DEFAULT_SCOPE = "open_access"
DEFAULT_LOG_EVERY = 10
DEFAULT_MAX_IMAGE_SIDE = 1536

SYSTEM_PROMPT = """
You are a medical visual question answering assistant.
You will receive one medical image and one multiple-choice question.
Reason briefly from the visual evidence, then pick the single best option from the provided choices.
Return exactly two XML-style blocks in this exact order:
<reasoning>
brief explanation
</reasoning>
<answer>A</answer>
Rules:
- The answer must be exactly one option letter that appears in the choices.
- Do not output markdown.
- Do not output JSON.
- Do not output anything outside the two blocks.
""".strip()


_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class ManifestStats:
    total_raw_samples: int
    accessible_samples: int
    skipped_missing_image: int
    skipped_unresolved_answer: int
    source_file_count: int


def normalize_text(text: Any) -> str:
    if text is None:
        return ""
    return " ".join(str(text).strip().split())


def normalize_answer_key(text: Any) -> str:
    normalized = normalize_text(text).lower()
    normalized = normalized.replace("&", " and ")
    normalized = re.sub(r"[^a-z0-9]+", " ", normalized)
    return " ".join(normalized.split())


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_choices(item: Dict[str, Any]) -> List[Dict[str, str]]:
    choices: List[Dict[str, str]] = []
    for letter in "ABCDE":
        key = f"option_{letter}"
        value = normalize_text(item.get(key))
        if value:
            choices.append({"letter": letter, "text": value})
    return choices


def resolve_gold_letter(gt_answer: str, choices: List[Dict[str, str]]) -> Optional[str]:
    gold_norm = normalize_answer_key(gt_answer)
    if not gold_norm:
        return None

    exact_matches = [choice["letter"] for choice in choices if normalize_text(choice["text"]) == normalize_text(gt_answer)]
    if len(exact_matches) == 1:
        return exact_matches[0]

    normalized_matches = [choice["letter"] for choice in choices if normalize_answer_key(choice["text"]) == gold_norm]
    if len(normalized_matches) == 1:
        return normalized_matches[0]

    if re.fullmatch(r"[A-E]", normalize_text(gt_answer).upper()):
        letter = normalize_text(gt_answer).upper()
        if any(choice["letter"] == letter for choice in choices):
            return letter
    return None


def qa_files_for_scope(dataset_root: Path, scope: str) -> List[Path]:
    qa_root = dataset_root / "QA_information"
    if scope == "open_access":
        return sorted((qa_root / "Open-access").glob("*.json"))
    if scope == "restricted_access":
        return sorted((qa_root / "Restricted-access").glob("*.json"))
    if scope == "all":
        return sorted(qa_root.rglob("*.json"))
    raise ValueError(f"unsupported scope: {scope}")


def build_manifest(dataset_root: Path, scope: str) -> tuple[List[Dict[str, Any]], ManifestStats]:
    manifest: List[Dict[str, Any]] = []
    total_raw = 0
    skipped_missing_image = 0
    skipped_unresolved_answer = 0
    qa_files = qa_files_for_scope(dataset_root, scope)

    for qa_file in qa_files:
        rows = read_json(qa_file)
        source_group = qa_file.parent.name
        source_name = qa_file.name
        for item in rows:
            total_raw += 1
            image_rel = normalize_text(item.get("image_path"))
            if not image_rel or "${dataset_root_path}" in image_rel:
                skipped_missing_image += 1
                continue
            image_abs = (dataset_root / image_rel).resolve()
            if not image_abs.exists():
                skipped_missing_image += 1
                continue

            choices = build_choices(item)
            gt_answer = normalize_text(item.get("gt_answer"))
            gold_letter = resolve_gold_letter(gt_answer, choices)
            if not gold_letter:
                skipped_unresolved_answer += 1
                continue

            manifest.append(
                {
                    "question_id": normalize_text(item.get("question_id")),
                    "dataset": normalize_text(item.get("dataset")),
                    "question_type": normalize_text(item.get("question_type")),
                    "question": normalize_text(item.get("question")),
                    "gt_answer": gt_answer,
                    "gold_letter": gold_letter,
                    "choices": choices,
                    "num_choices": len(choices),
                    "image_path_rel": image_rel,
                    "image_path_abs": str(image_abs),
                    "modality_type": normalize_text(item.get("modality_type")),
                    "source_group": source_group,
                    "source_json": source_name,
                }
            )

    stats = ManifestStats(
        total_raw_samples=total_raw,
        accessible_samples=len(manifest),
        skipped_missing_image=skipped_missing_image,
        skipped_unresolved_answer=skipped_unresolved_answer,
        source_file_count=len(qa_files),
    )
    return manifest, stats


def encode_image_as_data_url(image_path: str, max_image_side: int) -> str:
    path = Path(image_path)
    if max_image_side <= 0:
        mime_type, _ = mimetypes.guess_type(str(path))
        if not mime_type:
            mime_type = "image/png"
        with path.open("rb") as file:
            encoded = base64.b64encode(file.read()).decode("utf-8")
        return f"data:{mime_type};base64,{encoded}"

    with Image.open(path) as image:
        image.load()
        width, height = image.size
        longest_side = max(width, height)
        if longest_side > max_image_side:
            scale = max_image_side / float(longest_side)
            new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
        if image.mode not in ("RGB", "L"):
            image = image.convert("RGB")
        elif image.mode == "L":
            image = image.convert("RGB")

        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=95)
        encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


def build_user_text(item: Dict[str, Any]) -> str:
    choice_lines = [f"{choice['letter']}. {choice['text']}" for choice in item["choices"]]
    return (
        f"Question:\n{item['question']}\n\n"
        f"Choices:\n" + "\n".join(choice_lines) + "\n\n"
        f"Return <reasoning>...</reasoning><answer>...</answer>."
    )


def parse_reasoning(raw_text: str) -> str:
    match = re.search(r"<reasoning>\s*(.*?)\s*</reasoning>", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if match:
        return normalize_text(match.group(1))
    return ""


def parse_answer(raw_text: str, valid_letters: set[str]) -> Optional[str]:
    text = normalize_text(raw_text)
    if not text:
        return None

    answer_match = re.search(r"<answer>\s*([A-E])\s*</answer>", raw_text, flags=re.IGNORECASE | re.DOTALL)
    if answer_match:
        letter = answer_match.group(1).upper()
        if letter in valid_letters:
            return letter

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw_text, flags=re.DOTALL)
    if fenced:
        text = fenced.group(1).strip()
    try:
        payload = json.loads(text)
        if isinstance(payload, dict):
            answer = normalize_text(payload.get("answer")).upper().replace(".", "")
            if answer in valid_letters:
                return answer
    except Exception:
        pass

    patterns = [
        r'"answer"\s*:\s*"([A-E])"',
        r"\b([A-E])\b",
    ]
    for pattern in patterns:
        match = re.search(pattern, raw_text.upper(), flags=re.IGNORECASE | re.DOTALL)
        if match:
            letter = match.group(1).upper()
            if letter in valid_letters:
                return letter
    return None


def get_thread_client(base_url: str, timeout: int) -> OpenAI:
    client = getattr(_THREAD_LOCAL, "client", None)
    client_base_url = getattr(_THREAD_LOCAL, "client_base_url", None)
    client_timeout = getattr(_THREAD_LOCAL, "client_timeout", None)
    if client is None or client_base_url != base_url or client_timeout != timeout:
        client = OpenAI(base_url=base_url, api_key=os.getenv("OPENAI_API_KEY", "EMPTY"), timeout=timeout)
        _THREAD_LOCAL.client = client
        _THREAD_LOCAL.client_base_url = base_url
        _THREAD_LOCAL.client_timeout = timeout
    return client


def sample_n(
    base_url: str,
    model: str,
    image_data_url: str,
    item: Dict[str, Any],
    num_samples: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int,
    timeout_retries: int,
) -> List[Dict[str, Any]]:
    client = get_thread_client(base_url, timeout)
    user_content = [
        {"type": "image_url", "image_url": {"url": image_data_url}},
        {"type": "text", "text": build_user_text(item)},
    ]
    valid_letters = {choice["letter"] for choice in item["choices"]}
    last_error: Optional[Exception] = None

    for attempt in range(timeout_retries + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                n=num_samples,
            )
            results: List[Dict[str, Any]] = []
            for sample_index, choice in enumerate(response.choices):
                raw_text = normalize_text(choice.message.content)
                answer = parse_answer(raw_text, valid_letters)
                reasoning = parse_reasoning(raw_text)
                results.append(
                    {
                        "sample_index": sample_index,
                        "raw_response": raw_text,
                        "reasoning": reasoning,
                        "pred_letter": answer,
                        "status": "ok" if answer else "parse_failed",
                        "attempt": attempt,
                    }
                )
            while len(results) < num_samples:
                results.append(
                    {
                        "sample_index": len(results),
                        "raw_response": "",
                        "reasoning": "",
                        "pred_letter": None,
                        "status": "request_failed",
                        "attempt": attempt,
                        "error": f"server returned only {len(response.choices)} choices for requested n={num_samples}",
                    }
                )
            return results
        except Exception as exc:
            last_error = exc
            if attempt < timeout_retries:
                time.sleep(1.0 + attempt)

    return [
        {
            "sample_index": sample_index,
            "raw_response": "",
            "reasoning": "",
            "pred_letter": None,
            "status": "request_failed",
            "attempt": timeout_retries,
            "error": str(last_error) if last_error else "unknown error",
        }
        for sample_index in range(num_samples)
    ]


def evaluate_item(
    item: Dict[str, Any],
    *,
    base_url: str,
    model: str,
    num_samples: int,
    easy_threshold: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    max_image_side: int,
    timeout: int,
    timeout_retries: int,
) -> Dict[str, Any]:
    image_data_url = encode_image_as_data_url(item["image_path_abs"], max_image_side=max_image_side)
    samples: List[Dict[str, Any]] = []
    correct_count = 0
    parse_failed_count = 0
    request_failed_count = 0

    sampled_results = sample_n(
        base_url=base_url,
        model=model,
        image_data_url=image_data_url,
        item=item,
        num_samples=num_samples,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        timeout=timeout,
        timeout_retries=timeout_retries,
    )
    for sample_index, result in enumerate(sampled_results):
        pred_letter = result.get("pred_letter")
        is_correct = pred_letter == item["gold_letter"]
        if is_correct:
            correct_count += 1
        if result["status"] == "parse_failed":
            parse_failed_count += 1
        if result["status"] == "request_failed":
            request_failed_count += 1
        samples.append(
            {
                "sample_index": sample_index,
                "pred_letter": pred_letter,
                "correct": is_correct,
                "status": result["status"],
                "attempt": result["attempt"],
                "error": result.get("error", ""),
                "reasoning": result.get("reasoning", ""),
                "raw_response": result["raw_response"],
            }
        )

    accuracy = correct_count / num_samples if num_samples else 0.0
    is_easy = correct_count >= easy_threshold
    return {
        **item,
        "num_samples": num_samples,
        "correct_count": correct_count,
        "accuracy": round(accuracy, 6),
        "easy_threshold": easy_threshold,
        "is_easy": is_easy,
        "decision": "filter_out_easy" if is_easy else "keep_for_rl",
        "parse_failed_count": parse_failed_count,
        "request_failed_count": request_failed_count,
        "samples": samples,
    }


def select_manifest_rows(
    manifest: List[Dict[str, Any]],
    *,
    num_shards: int,
    shard_index: int,
    start_index: int,
    max_items: int,
) -> List[Dict[str, Any]]:
    if num_shards <= 0:
        raise ValueError("num_shards must be positive")
    if shard_index < 0 or shard_index >= num_shards:
        raise ValueError("shard_index must be in [0, num_shards)")

    shard_rows = [row for index, row in enumerate(manifest) if index % num_shards == shard_index]
    if start_index < 0:
        raise ValueError("start_index must be non-negative")
    shard_rows = shard_rows[start_index:]
    if max_items > 0:
        shard_rows = shard_rows[:max_items]
    return shard_rows


def detect_model_name(base_url: str, fallback: str) -> str:
    try:
        with urllib.request.urlopen(f"{base_url.rstrip('/')}/models", timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
        model_ids = [item.get("id") for item in payload.get("data", []) if isinstance(item, dict)]
        for model_id in model_ids:
            if normalize_text(model_id):
                return str(model_id)
    except Exception:
        pass
    return fallback


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter OmniMedVQA with repeated Qwen3-VL-8B-Instruct sampling.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--artifact-root", default=DEFAULT_ARTIFACT_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", default="")
    parser.add_argument("--scope", choices=["open_access", "restricted_access", "all"], default=DEFAULT_SCOPE)
    parser.add_argument("--base-url", default=os.getenv("QWEN_BASE_URL", DEFAULT_BASE_URL))
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--easy-threshold", type=int, default=DEFAULT_EASY_THRESHOLD)
    parser.add_argument("--num-workers", type=int, default=DEFAULT_NUM_WORKERS)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--timeout", type=int, default=DEFAULT_REQUEST_TIMEOUT)
    parser.add_argument("--timeout-retries", type=int, default=DEFAULT_TIMEOUT_RETRIES)
    parser.add_argument("--max-image-side", type=int, default=DEFAULT_MAX_IMAGE_SIDE)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--manifest-only", action="store_true")
    parser.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    artifact_root = Path(args.artifact_root).resolve()
    output_root = Path(args.output_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    manifest_path = artifact_root / f"omnimedvqa_manifest_{args.scope}.jsonl"
    manifest_summary_path = artifact_root / f"omnimedvqa_manifest_{args.scope}_summary.json"

    manifest, manifest_stats = build_manifest(dataset_root, args.scope)
    write_jsonl(manifest_path, manifest)
    write_json(
        manifest_summary_path,
        {
            "dataset_root": str(dataset_root),
            "scope": args.scope,
            "total_raw_samples": manifest_stats.total_raw_samples,
            "accessible_samples": manifest_stats.accessible_samples,
            "skipped_missing_image": manifest_stats.skipped_missing_image,
            "skipped_unresolved_answer": manifest_stats.skipped_unresolved_answer,
            "source_file_count": manifest_stats.source_file_count,
        },
    )

    if args.manifest_only:
        print(json.dumps({"manifest_path": str(manifest_path), "manifest_rows": len(manifest)}, ensure_ascii=False, indent=2))
        return

    shard_rows = select_manifest_rows(
        manifest,
        num_shards=args.num_shards,
        shard_index=args.shard_index,
        start_index=args.start_index,
        max_items=args.max_items,
    )

    model_name = detect_model_name(args.base_url, args.model)
    run_name = args.run_name or (
        f"omnimedvqa_{args.scope}_qwen3_vl_8b_"
        f"s{args.num_samples}_thr{args.easy_threshold}_"
        f"shard{args.shard_index}of{args.num_shards}"
    )
    run_dir = output_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    keep_path = run_dir / "keep_for_rl.jsonl"
    easy_path = run_dir / "filtered_out_easy.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    existing_results = read_jsonl(results_path) if args.resume else []
    done_ids = {row["question_id"] for row in existing_results}
    pending_rows = [row for row in shard_rows if row["question_id"] not in done_ids]

    write_json(
        config_path,
        {
            "dataset_root": str(dataset_root),
            "artifact_root": str(artifact_root),
            "output_root": str(output_root),
            "scope": args.scope,
            "base_url": args.base_url,
            "model": model_name,
            "requested_model": args.model,
            "num_samples": args.num_samples,
            "easy_threshold": args.easy_threshold,
            "num_workers": args.num_workers,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "timeout": args.timeout,
            "timeout_retries": args.timeout_retries,
            "max_image_side": args.max_image_side,
            "num_shards": args.num_shards,
            "shard_index": args.shard_index,
            "start_index": args.start_index,
            "max_items": args.max_items,
            "resume": bool(args.resume),
            "manifest_path": str(manifest_path),
            "manifest_rows": len(manifest),
            "selected_rows": len(shard_rows),
            "pending_rows": len(pending_rows),
        },
    )

    if not args.resume:
        for path in (results_path, keep_path, easy_path, summary_path):
            if path.exists():
                path.unlink()

    total_results = list(existing_results)
    completed = len(existing_results)
    started_at = time.time()

    if pending_rows:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_map = {
                executor.submit(
                    evaluate_item,
                    row,
                    base_url=args.base_url,
                    model=model_name,
                    num_samples=args.num_samples,
                    easy_threshold=args.easy_threshold,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_image_side=args.max_image_side,
                    timeout=args.timeout,
                    timeout_retries=args.timeout_retries,
                ): row
                for row in pending_rows
            }

            for future in as_completed(future_map):
                row = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {
                        **row,
                        "num_samples": args.num_samples,
                        "correct_count": 0,
                        "accuracy": 0.0,
                        "easy_threshold": args.easy_threshold,
                        "is_easy": False,
                        "decision": "keep_for_rl",
                        "parse_failed_count": 0,
                        "request_failed_count": args.num_samples,
                        "samples": [],
                        "fatal_error": str(exc),
                    }

                total_results.append(result)
                append_jsonl(results_path, result)
                completed += 1
                if completed % max(args.log_every, 1) == 0 or completed == len(shard_rows):
                    elapsed = time.time() - started_at
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "selected_rows": len(shard_rows),
                                "elapsed_sec": round(elapsed, 2),
                                "last_question_id": result["question_id"],
                                "last_decision": result["decision"],
                                "last_correct_count": result["correct_count"],
                            },
                            ensure_ascii=False,
                        ),
                        flush=True,
                    )

    total_results.sort(key=lambda row: row["question_id"])
    write_jsonl(results_path, total_results)

    keep_rows = [row for row in total_results if row["decision"] == "keep_for_rl"]
    easy_rows = [row for row in total_results if row["decision"] == "filter_out_easy"]
    write_jsonl(keep_path, keep_rows)
    write_jsonl(easy_path, easy_rows)

    summary = {
        "run_name": run_name,
        "dataset_root": str(dataset_root),
        "scope": args.scope,
        "manifest_rows": len(manifest),
        "selected_rows": len(shard_rows),
        "completed_rows": len(total_results),
        "keep_for_rl_rows": len(keep_rows),
        "filtered_out_easy_rows": len(easy_rows),
        "keep_ratio": round(len(keep_rows) / len(total_results), 6) if total_results else 0.0,
        "easy_ratio": round(len(easy_rows) / len(total_results), 6) if total_results else 0.0,
        "mean_accuracy": round(sum(row["accuracy"] for row in total_results) / len(total_results), 6) if total_results else 0.0,
        "total_parse_failed": sum(row.get("parse_failed_count", 0) for row in total_results),
        "total_request_failed": sum(row.get("request_failed_count", 0) for row in total_results),
        "base_url": args.base_url,
        "model": model_name,
        "requested_model": args.model,
        "num_samples": args.num_samples,
        "easy_threshold": args.easy_threshold,
        "num_workers": args.num_workers,
        "max_tokens": args.max_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "timeout": args.timeout,
        "timeout_retries": args.timeout_retries,
        "max_image_side": args.max_image_side,
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "start_index": args.start_index,
        "max_items": args.max_items,
        "manifest_path": str(manifest_path),
        "results_path": str(results_path),
        "keep_path": str(keep_path),
        "easy_path": str(easy_path),
    }
    write_json(summary_path, summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
