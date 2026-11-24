#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import json
import logging
import os
import random
import re
import threading
import time
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, as_completed, wait
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import requests
import sglang as sgl
from openai import OpenAI
from PIL import Image
from tqdm.auto import tqdm
from transformers import AutoProcessor

from curate_omnimedvqa import (
    DEFAULT_ARTIFACT_ROOT,
    DEFAULT_EASY_THRESHOLD,
    DEFAULT_MAX_IMAGE_SIDE,
    DEFAULT_MAX_TOKENS,
    DEFAULT_NUM_SAMPLES,
    DEFAULT_OUTPUT_ROOT,
    DEFAULT_TEMPERATURE,
    DEFAULT_TOP_P,
    normalize_text,
    read_jsonl,
    write_json,
    write_jsonl,
)


LOGGER = logging.getLogger("curate_open_medvqa")

DEFAULT_DATASET_ROOT = "./dataset"
DEFAULT_MODEL_PATH = "external/Med_Vision/verl/model/Qwen3-VL-8B-Instruct"
DEFAULT_BATCH_SIZE = 64
DEFAULT_DP_SIZE = 4
DEFAULT_MEM_FRACTION_STATIC = 0.8
DEFAULT_BENCHMARKS = "slake,vqa_rad"
DEFAULT_SLAKE_LANGUAGES = "en"
DEFAULT_JUDGE_BASE_URL = os.environ.get("LLM_AS_A_JUDGE_BASE", "http://127.0.0.1:30000/v1")
DEFAULT_JUDGE_MAX_WORKERS = 16
DEFAULT_JUDGE_BATCH_SIZE = 16
DEFAULT_JUDGE_REQUEST_TIMEOUT = 120

SYSTEM_PROMPT = """
You are a medical visual question answering assistant.
You will receive one medical image and one medical question.
Reason briefly from the image, then give a short final answer.
Return exactly two XML-style blocks in this exact order:
<reasoning>
brief explanation
</reasoning>
<answer>short final answer</answer>
Rules:
- Keep the final answer concise.
- Do not output markdown.
- Do not output JSON.
- Do not output anything outside the two blocks.
""".strip()

ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
REASONING_PATTERN = re.compile(r"<reasoning>\s*(.*?)\s*</reasoning>", re.DOTALL | re.IGNORECASE)
_THREAD_LOCAL = threading.local()


@dataclass(frozen=True)
class ManifestStats:
    total_raw_samples: int
    accessible_samples: int
    skipped_missing_image: int
    benchmark_rows: Dict[str, int]


@dataclass(frozen=True)
class JudgeConfig:
    enabled: bool
    base_url: str
    model_name: str
    max_workers: int
    batch_size: int
    timeout: int


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_answer_text(text: Any) -> str:
    text = normalize_text(text).lower()
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"^[a-e][\.\):\-\s]+", "", text)
    text = re.sub(r"[\"'`]+", "", text)
    return text.strip(" .,:;!?")


def extract_binary_answer(text: Any) -> str | None:
    text = normalize_answer_text(text)
    if re.search(r"\byes\b", text):
        return "yes"
    if re.search(r"\bno\b", text):
        return "no"
    return None


def parse_reasoning(raw_text: str) -> str:
    match = REASONING_PATTERN.search(raw_text or "")
    return normalize_text(match.group(1)) if match else ""


def parse_answer_text(raw_text: str) -> tuple[str, bool]:
    match = ANSWER_PATTERN.search(raw_text or "")
    if match:
        return normalize_text(match.group(1)), True
    return normalize_text(raw_text), False


def load_image_resized(image_path: str, max_image_side: int) -> Image.Image:
    image = Image.open(image_path)
    image.load()
    image = image.convert("RGB")
    width, height = image.size
    longest_side = max(width, height)
    if max_image_side > 0 and longest_side > max_image_side:
        scale = max_image_side / float(longest_side)
        new_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    return image


def load_processor(model_path: str) -> AutoProcessor:
    return AutoProcessor.from_pretrained(model_path, trust_remote_code=True)


def build_user_text(item: Dict[str, Any]) -> str:
    hints: list[str] = []
    if item["benchmark"] == "slake":
        hints.append(f"Answer type: {item.get('answer_type', 'OPEN')}")
    return (
        f"Question:\n{item['question']}\n\n"
        + ("\n".join(hints) + "\n\n" if hints else "")
        + "Return <reasoning>...</reasoning><answer>...</answer>."
    )


def build_vlm_prompt(processor: AutoProcessor, item: Dict[str, Any]) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": build_user_text(item)},
            ],
        },
    ]
    return processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def normalize_engine_outputs(outputs: Any) -> List[Dict[str, Any]]:
    if isinstance(outputs, list):
        return outputs
    if isinstance(outputs, dict):
        return [outputs]
    raise TypeError(f"Unexpected engine output type: {type(outputs)!r}")


def parse_benchmarks(value: str) -> list[str]:
    benchmarks = []
    for item in value.split(","):
        name = item.strip().lower()
        if name:
            benchmarks.append(name)
    if not benchmarks:
        raise ValueError("No benchmarks selected.")
    invalid = [name for name in benchmarks if name not in {"slake", "vqa_rad"}]
    if invalid:
        raise ValueError(f"Unsupported benchmarks: {invalid}")
    return benchmarks


def parse_slake_languages(value: str) -> set[str]:
    raw = {item.strip().lower() for item in value.split(",") if item.strip()}
    if not raw or "all" in raw:
        return set()
    return raw


def iter_slake_rows(dataset_root: Path, languages: set[str]) -> Iterable[Dict[str, Any]]:
    slake_root = dataset_root / "SLAKE"
    for split in ("train", "validation", "test"):
        rows = json.loads((slake_root / f"{split}.json").read_text(encoding="utf-8"))
        for row in rows:
            q_lang = normalize_text(row.get("q_lang")).lower()
            if languages and q_lang not in languages:
                continue
            image_rel = normalize_text(row.get("img_name"))
            image_abs = (slake_root / "imgs" / image_rel).resolve()
            if not image_rel or not image_abs.exists():
                continue
            answer_type = normalize_text(row.get("answer_type")).upper() or "OPEN"
            answer = normalize_text(row.get("answer"))
            yield {
                "question_id": f"slake_{split}_{normalize_text(row.get('qid'))}",
                "benchmark": "slake",
                "split": split,
                "question": normalize_text(row.get("question")),
                "gt_answer": answer,
                "answer_type": answer_type,
                "image_path_abs": str(image_abs),
                "image_path_rel": f"SLAKE/imgs/{image_rel}",
                "q_lang": q_lang,
                "location": normalize_text(row.get("location")),
                "modality": normalize_text(row.get("modality")),
                "content_type": normalize_text(row.get("content_type")),
            }


def decode_vqa_rad_image(row: Dict[str, Any], image_path: Path) -> None:
    payload = row.get("image") or {}
    image_b64 = payload.get("bytes", "")
    if not image_b64:
        raise ValueError("Missing image bytes in VQA_RAD row.")
    image_path.parent.mkdir(parents=True, exist_ok=True)
    image_path.write_bytes(base64.b64decode(image_b64))


def iter_vqa_rad_rows(dataset_root: Path, image_cache_root: Path) -> Iterable[Dict[str, Any]]:
    rad_root = dataset_root / "VQA_RAD" / "data"
    split_files = {
        "train": rad_root / "train-00000-of-00001-eb8844602202be60.json",
        "test": rad_root / "test-00000-of-00001-e5bc3d208bb4deeb.json",
    }
    for split, json_path in split_files.items():
        rows = json.loads(json_path.read_text(encoding="utf-8"))
        for index, row in enumerate(rows):
            original_name = normalize_text((row.get("image") or {}).get("path")) or f"{split}_{index}.jpg"
            image_path = image_cache_root / split / original_name
            if not image_path.exists():
                decode_vqa_rad_image(row, image_path)
            answer = normalize_text(row.get("answer"))
            binary = extract_binary_answer(answer)
            yield {
                "question_id": f"vqa_rad_{split}_{index}",
                "benchmark": "vqa_rad",
                "split": split,
                "question": normalize_text(row.get("question")),
                "gt_answer": answer,
                "answer_type": "CLOSED" if binary else "OPEN",
                "image_path_abs": str(image_path.resolve()),
                "image_path_rel": f"artifacts/vqa_rad_images/{split}/{original_name}",
                "q_lang": "en",
                "location": "",
                "modality": "",
                "content_type": "",
            }


def build_manifest(dataset_root: Path, benchmarks: Sequence[str], slake_languages: set[str], artifact_root: Path) -> tuple[List[Dict[str, Any]], ManifestStats]:
    manifest: list[Dict[str, Any]] = []
    benchmark_rows = {name: 0 for name in benchmarks}
    total_raw = 0
    skipped_missing_image = 0

    if "slake" in benchmarks:
        for row in iter_slake_rows(dataset_root, slake_languages):
            total_raw += 1
            if not Path(row["image_path_abs"]).exists():
                skipped_missing_image += 1
                continue
            manifest.append(row)
            benchmark_rows["slake"] += 1

    if "vqa_rad" in benchmarks:
        cache_root = artifact_root / "vqa_rad_images"
        for row in iter_vqa_rad_rows(dataset_root, cache_root):
            total_raw += 1
            if not Path(row["image_path_abs"]).exists():
                skipped_missing_image += 1
                continue
            manifest.append(row)
            benchmark_rows["vqa_rad"] += 1

    return manifest, ManifestStats(
        total_raw_samples=total_raw,
        accessible_samples=len(manifest),
        skipped_missing_image=skipped_missing_image,
        benchmark_rows=benchmark_rows,
    )


def init_judge_config(base_url: str, timeout: int, max_workers: int) -> JudgeConfig:
    base_url = normalize_text(base_url).rstrip("/")
    if not base_url:
        return JudgeConfig(False, "", "", max_workers, DEFAULT_JUDGE_BATCH_SIZE, timeout)
    try:
        response = requests.get(f"{base_url}/models", headers={"Authorization": "Bearer EMPTY"}, timeout=20)
        response.raise_for_status()
        payload = response.json()
        model_name = payload.get("data", [{}])[0].get("id", "")
    except Exception as exc:
        LOGGER.warning("Failed to initialize remote judge at %s: %s", base_url, exc)
        model_name = ""
    return JudgeConfig(bool(model_name), base_url, model_name, max_workers, DEFAULT_JUDGE_BATCH_SIZE, timeout)


def get_judge_client(judge_config: JudgeConfig) -> OpenAI:
    client = getattr(_THREAD_LOCAL, "judge_client", None)
    client_base_url = getattr(_THREAD_LOCAL, "judge_client_base_url", None)
    if client is None or client_base_url != judge_config.base_url:
        client = OpenAI(api_key="EMPTY", base_url=judge_config.base_url)
        _THREAD_LOCAL.judge_client = client
        _THREAD_LOCAL.judge_client_base_url = judge_config.base_url
    return client


def normalize_judgement(value: Any) -> str:
    text = normalize_text(value).upper()
    if "CORRECT" in text and "INCORRECT" not in text:
        return "CORRECT"
    if "INCORRECT" in text:
        return "INCORRECT"
    return ""


def build_judge_batch_prompt(cases: Sequence[Dict[str, Any]]) -> str:
    sections = [
        "I will provide multiple cases. For each case, compare the standard answer and the model answer.",
        "Return JSON only. Use this exact schema:",
        '[{"case_id":"0","judgement":"CORRECT"},{"case_id":"1","judgement":"INCORRECT"}]',
        "Allowed values for judgement are only CORRECT or INCORRECT.",
        "Do not include explanation.",
        "",
        "Examples:",
        "Case 0",
        "Question: Are regions of the brain infarcted?",
        "Standard Answer: yes",
        "Model Answer: no",
        "",
        "Case 1",
        "Question: Which part of the human body is the organ located in the image?",
        "Standard Answer: Chest",
        "Model Answer: thorax",
        "",
    ]
    for case in cases:
        sections.extend(
            [
                f"Case {case['case_id']}",
                f"Question: {case['question']}",
                f"Standard Answer: {case['ground_truth']}",
                f"Model Answer: {case['pred_answer']}",
                "",
            ]
        )
    return "\n".join(sections).strip()


def parse_judge_batch_response(content: str, expected_case_ids: Sequence[str]) -> Dict[str, bool]:
    text = normalize_text(content)
    fenced = re.search(r"```(?:json)?\s*(.*?)```", content or "", flags=re.DOTALL | re.IGNORECASE)
    payload_text = fenced.group(1).strip() if fenced else content
    results: Dict[str, bool] = {}

    try:
        payload = json.loads(payload_text)
    except Exception:
        payload = None

    if isinstance(payload, dict):
        if isinstance(payload.get("results"), list):
            payload = payload["results"]
        else:
            for case_id, judgement in payload.items():
                normalized = normalize_judgement(judgement)
                if normalized:
                    results[str(case_id)] = normalized == "CORRECT"

    if isinstance(payload, list):
        for item in payload:
            if not isinstance(item, dict):
                continue
            case_id = normalize_text(item.get("case_id"))
            judgement = normalize_judgement(item.get("judgement"))
            if case_id and judgement:
                results[case_id] = judgement == "CORRECT"

    if results:
        return results

    for case_id in expected_case_ids:
        match = re.search(rf"(?:case[_\s-]*id|case)\s*[:#]?\s*{re.escape(case_id)}.*?\b(CORRECT|INCORRECT)\b", text, flags=re.IGNORECASE)
        if match:
            results[case_id] = match.group(1).upper() == "CORRECT"
    return results


def llm_answer_verify_batch(cases: Sequence[Dict[str, Any]], judge_config: JudgeConfig) -> Dict[str, bool]:
    if not judge_config.enabled:
        return {}
    client = get_judge_client(judge_config)
    system_prompt = (
        "You are an expert evaluator. For each case, compare the provided standard answer and model answer "
        "for the medical question. Judge correctness against the provided standard answer, not your own beliefs. "
        "The answering model had access to the image. Common medical synonyms, abbreviations, and equivalent "
        "anatomical or clinical terms should be treated as CORRECT when they clearly refer to the same concept. "
        "Any answer claiming the image was missing should be treated as INCORRECT unless the standard answer "
        "itself says that. Return JSON only."
    )
    user_prompt = build_judge_batch_prompt(cases)
    try:
        response = client.chat.completions.create(
            model=judge_config.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            seed=random.randint(0, 1_000_000),
            temperature=0.1,
            timeout=judge_config.timeout,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
        content = response.choices[0].message.content
    except Exception as exc:
        LOGGER.warning("LLM judge request failed: %s", exc)
        return {}
    results = parse_judge_batch_response(content, [case["case_id"] for case in cases])
    if not results:
        LOGGER.warning("Unexpected batch judge response: %s", normalize_text(content))
    return results


def quick_match_answer(prediction: str, ground_truth: str) -> tuple[bool, str]:
    pred_norm = normalize_answer_text(prediction)
    gt_norm = normalize_answer_text(ground_truth)
    if pred_norm and gt_norm and pred_norm == gt_norm:
        return True, "exact"
    pred_binary = extract_binary_answer(prediction)
    gt_binary = extract_binary_answer(ground_truth)
    if pred_binary and gt_binary and pred_binary == gt_binary:
        return True, "binary"
    return False, ""


def prepare_samples(item: Dict[str, Any], sample_outputs: List[Dict[str, Any]], num_samples: int) -> tuple[List[Dict[str, Any]], list[int]]:
    samples: list[Dict[str, Any]] = []
    unresolved_indices: list[int] = []
    for sample_index in range(num_samples):
        if sample_index < len(sample_outputs):
            output = sample_outputs[sample_index]
            raw_text = normalize_text(output.get("text", ""))
            answer_text, has_structured_answer = parse_answer_text(raw_text)
            reasoning = parse_reasoning(raw_text)
            status = "ok" if has_structured_answer else "parse_failed"
        else:
            raw_text = ""
            answer_text = ""
            reasoning = ""
            status = "request_failed"

        correct = False
        matched_by = ""
        if answer_text:
            correct, matched_by = quick_match_answer(answer_text, item["gt_answer"])
            if not correct:
                unresolved_indices.append(sample_index)
        samples.append(
            {
                "sample_index": sample_index,
                "pred_answer": answer_text,
                "correct": correct,
                "status": status,
                "attempt": 0,
                "error": "",
                "reasoning": reasoning,
                "raw_response": raw_text,
                "matched_by": matched_by,
                "judge_used": False,
            }
        )
    return samples, unresolved_indices


def build_judge_cases(batch_prepared: Sequence[tuple[Dict[str, Any], List[Dict[str, Any]]]]) -> List[Dict[str, Any]]:
    case_map: Dict[tuple[str, str, str], Dict[str, Any]] = {}
    for item, samples in batch_prepared:
        for sample in samples:
            if sample["correct"] or not sample["pred_answer"]:
                continue
            dedupe_key = (
                item["question"],
                normalize_answer_text(item["gt_answer"]),
                normalize_answer_text(sample["pred_answer"]),
            )
            case = case_map.get(dedupe_key)
            if case is None:
                case = {
                    "question": item["question"],
                    "ground_truth": item["gt_answer"],
                    "pred_answer": sample["pred_answer"],
                    "samples": [],
                }
                case_map[dedupe_key] = case
            case["samples"].append(sample)
    return list(case_map.values())


def chunked_cases(cases: Sequence[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(cases), batch_size):
        chunk = list(cases[start : start + batch_size])
        for index, case in enumerate(chunk):
            case["case_id"] = str(index)
        yield chunk


def run_judge_tasks(cases: List[Dict[str, Any]], judge_config: JudgeConfig) -> None:
    if not cases or not judge_config.enabled:
        return
    with ThreadPoolExecutor(max_workers=judge_config.max_workers) as executor:
        future_to_task = {
            executor.submit(llm_answer_verify_batch, chunk, judge_config): chunk
            for chunk in chunked_cases(cases, judge_config.batch_size)
        }
        for future in as_completed(future_to_task):
            try:
                results = future.result()
            except Exception as exc:
                results = {}
                LOGGER.warning("Batch judge worker failed: %s", exc)
            for case in future_to_task[future]:
                judged = results.get(case["case_id"], False)
                for sample in case["samples"]:
                    sample["judge_used"] = True
                    sample["matched_by"] = "judge"
                    sample["correct"] = judged
                    if case["case_id"] not in results:
                        sample["error"] = "missing_judge_result"


def judge_and_finalize_batch(
    batch_prepared: Sequence[tuple[Dict[str, Any], List[Dict[str, Any]]]],
    easy_threshold: int,
    judge_config: JudgeConfig,
) -> List[Dict[str, Any]]:
    judge_cases = build_judge_cases(batch_prepared)
    run_judge_tasks(judge_cases, judge_config)
    return [finalize_item_result(item, samples, easy_threshold) for item, samples in batch_prepared]


def finalize_item_result(item: Dict[str, Any], samples: List[Dict[str, Any]], easy_threshold: int) -> Dict[str, Any]:
    correct_count = sum(1 for sample in samples if sample["correct"])
    parse_failed_count = sum(1 for sample in samples if sample["status"] == "parse_failed")
    request_failed_count = sum(1 for sample in samples if sample["status"] == "request_failed")
    judge_used_count = sum(1 for sample in samples if sample["judge_used"])
    accuracy = correct_count / len(samples) if samples else 0.0
    is_easy = correct_count >= easy_threshold
    return {
        **item,
        "num_samples": len(samples),
        "correct_count": correct_count,
        "accuracy": round(accuracy, 6),
        "easy_threshold": easy_threshold,
        "is_easy": is_easy,
        "decision": "filter_out_easy" if is_easy else "keep_for_rl",
        "parse_failed_count": parse_failed_count,
        "request_failed_count": request_failed_count,
        "judge_used_count": judge_used_count,
        "samples": samples,
    }


def batched(rows: List[Dict[str, Any]], batch_size: int) -> Iterable[List[Dict[str, Any]]]:
    for start in range(0, len(rows), batch_size):
        yield rows[start : start + batch_size]


def build_failed_result(item: Dict[str, Any], *, num_samples: int, easy_threshold: int, error: str) -> Dict[str, Any]:
    samples = [
        {
            "sample_index": sample_index,
            "pred_answer": "",
            "correct": False,
            "status": "request_failed",
            "attempt": 0,
            "error": error,
            "reasoning": "",
            "raw_response": "",
            "matched_by": "",
            "judge_used": False,
        }
        for sample_index in range(num_samples)
    ]
    return finalize_item_result(item, samples, easy_threshold)


def init_summary(existing_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    completed = len(existing_results)
    keep = sum(1 for row in existing_results if row["decision"] == "keep_for_rl")
    easy = sum(1 for row in existing_results if row["decision"] == "filter_out_easy")
    return {
        "completed_rows": completed,
        "keep_for_rl_rows": keep,
        "filtered_out_easy_rows": easy,
        "total_accuracy": sum(float(row.get("accuracy", 0.0)) for row in existing_results),
        "total_parse_failed": sum(int(row.get("parse_failed_count", 0)) for row in existing_results),
        "total_request_failed": sum(int(row.get("request_failed_count", 0)) for row in existing_results),
        "total_judge_used": sum(int(row.get("judge_used_count", 0)) for row in existing_results),
    }


def update_summary(summary: Dict[str, Any], result: Dict[str, Any]) -> None:
    summary["completed_rows"] += 1
    summary["keep_for_rl_rows"] += int(result["decision"] == "keep_for_rl")
    summary["filtered_out_easy_rows"] += int(result["decision"] == "filter_out_easy")
    summary["total_accuracy"] += float(result["accuracy"])
    summary["total_parse_failed"] += int(result.get("parse_failed_count", 0))
    summary["total_request_failed"] += int(result.get("request_failed_count", 0))
    summary["total_judge_used"] += int(result.get("judge_used_count", 0))


def flush_done_futures(
    pending_futures: Dict[Future, int],
    *,
    wait_for_one: bool,
    results_path: Path,
    keep_path: Path,
    easy_path: Path,
    summary: Dict[str, Any],
    progress: tqdm,
    last_result_holder: Dict[str, Any],
) -> None:
    if not pending_futures:
        return
    done: set[Future]
    if wait_for_one:
        done, _ = wait(set(pending_futures.keys()), return_when=FIRST_COMPLETED)
    else:
        done = {future for future in pending_futures if future.done()}
        if not done:
            return

    for future in done:
        batch_size = pending_futures.pop(future)
        results = future.result()
        for result in results:
            append_jsonl(results_path, result)
            append_jsonl(keep_path if result["decision"] == "keep_for_rl" else easy_path, result)
            update_summary(summary, result)
            last_result_holder["result"] = result
        progress.update(batch_size)
        if last_result_holder.get("result"):
            result = last_result_holder["result"]
            progress.set_postfix(
                decision=result["decision"],
                correct=result["correct_count"],
                refresh=False,
            )


def write_benchmark_split_files(results_path: Path, run_dir: Path) -> Dict[str, Dict[str, str]]:
    rows_by_benchmark: Dict[str, List[Dict[str, Any]]] = {}
    with results_path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            benchmark = normalize_text(row.get("benchmark")).lower() or "unknown"
            rows_by_benchmark.setdefault(benchmark, []).append(row)

    output_paths: Dict[str, Dict[str, str]] = {}
    for benchmark, rows in rows_by_benchmark.items():
        benchmark_results_path = run_dir / f"{benchmark}_results.jsonl"
        benchmark_keep_path = run_dir / f"{benchmark}_keep_for_rl.jsonl"
        benchmark_easy_path = run_dir / f"{benchmark}_filtered_out_easy.jsonl"
        keep_rows = [row for row in rows if row["decision"] == "keep_for_rl"]
        easy_rows = [row for row in rows if row["decision"] == "filter_out_easy"]
        write_jsonl(benchmark_results_path, rows)
        write_jsonl(benchmark_keep_path, keep_rows)
        write_jsonl(benchmark_easy_path, easy_rows)
        output_paths[benchmark] = {
            "results_path": str(benchmark_results_path),
            "keep_path": str(benchmark_keep_path),
            "easy_path": str(benchmark_easy_path),
        }
    return output_paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline open-ended medical VQA curation with local SGLang and remote judge.")
    parser.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--benchmarks", default=DEFAULT_BENCHMARKS, help="Comma-separated subset of: slake,vqa_rad")
    parser.add_argument("--slake-languages", default=DEFAULT_SLAKE_LANGUAGES, help="Comma-separated SLAKE languages, or all.")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--judge-base-url", default=DEFAULT_JUDGE_BASE_URL)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--easy-threshold", type=int, default=DEFAULT_EASY_THRESHOLD)
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--top-p", type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--max-image-side", type=int, default=DEFAULT_MAX_IMAGE_SIDE)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dp-size", type=int, default=DEFAULT_DP_SIZE)
    parser.add_argument("--mem-fraction-static", type=float, default=DEFAULT_MEM_FRACTION_STATIC)
    parser.add_argument("--judge-max-workers", type=int, default=DEFAULT_JUDGE_MAX_WORKERS)
    parser.add_argument("--judge-batch-size", type=int, default=DEFAULT_JUDGE_BATCH_SIZE)
    parser.add_argument("--judge-timeout", type=int, default=DEFAULT_JUDGE_REQUEST_TIMEOUT)
    parser.add_argument("--judge-pipeline-depth", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_root = Path(args.dataset_root).resolve()
    artifact_root = Path(DEFAULT_ARTIFACT_ROOT).resolve()
    output_root = Path(args.output_root).resolve()
    artifact_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    benchmarks = parse_benchmarks(args.benchmarks)
    slake_languages = parse_slake_languages(args.slake_languages)
    benchmark_key = "_".join(benchmarks)
    language_key = "all" if not slake_languages else "-".join(sorted(slake_languages))

    manifest_path = artifact_root / f"open_medvqa_manifest_{benchmark_key}_{language_key}.jsonl"
    manifest_summary_path = artifact_root / f"open_medvqa_manifest_{benchmark_key}_{language_key}_summary.json"

    manifest, manifest_stats = build_manifest(dataset_root, benchmarks, slake_languages, artifact_root)
    write_jsonl(manifest_path, manifest)
    write_json(
        manifest_summary_path,
        {
            "dataset_root": str(dataset_root),
            "benchmarks": benchmarks,
            "slake_languages": [] if not slake_languages else sorted(slake_languages),
            "total_raw_samples": manifest_stats.total_raw_samples,
            "accessible_samples": manifest_stats.accessible_samples,
            "skipped_missing_image": manifest_stats.skipped_missing_image,
            "benchmark_rows": manifest_stats.benchmark_rows,
        },
    )

    selected_rows = manifest[: args.max_items] if args.max_items > 0 else manifest
    run_dir = output_root / args.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.jsonl"
    keep_path = run_dir / "keep_for_rl.jsonl"
    easy_path = run_dir / "filtered_out_easy.jsonl"
    summary_path = run_dir / "summary.json"
    config_path = run_dir / "config.json"

    existing_results = read_jsonl(results_path) if args.resume else []
    done_ids = {row["question_id"] for row in existing_results}
    pending_rows = [row for row in selected_rows if row["question_id"] not in done_ids]

    if not args.resume:
        for path in (results_path, keep_path, easy_path, summary_path):
            if path.exists():
                path.unlink()

    judge_config = init_judge_config(args.judge_base_url, args.judge_timeout, args.judge_max_workers)
    judge_config = JudgeConfig(
        enabled=judge_config.enabled,
        base_url=judge_config.base_url,
        model_name=judge_config.model_name,
        max_workers=args.judge_max_workers,
        batch_size=args.judge_batch_size,
        timeout=args.judge_timeout,
    )

    write_json(
        config_path,
        {
            "dataset_root": str(dataset_root),
            "artifact_root": str(artifact_root),
            "output_root": str(output_root),
            "benchmarks": benchmarks,
            "slake_languages": [] if not slake_languages else sorted(slake_languages),
            "model_path": args.model_path,
            "judge_base_url": judge_config.base_url,
            "judge_model_name": judge_config.model_name,
            "judge_enabled": judge_config.enabled,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "easy_threshold": args.easy_threshold,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_image_side": args.max_image_side,
            "max_items": args.max_items,
            "resume": bool(args.resume),
            "dp_size": args.dp_size,
            "mem_fraction_static": args.mem_fraction_static,
            "judge_max_workers": args.judge_max_workers,
            "judge_batch_size": args.judge_batch_size,
            "judge_timeout": args.judge_timeout,
            "judge_pipeline_depth": args.judge_pipeline_depth,
            "manifest_path": str(manifest_path),
            "manifest_rows": len(manifest),
            "selected_rows": len(selected_rows),
            "pending_rows": len(pending_rows),
        },
    )

    engine_started_at = time.time()
    processor = load_processor(args.model_path)
    engine = sgl.Engine(
        model_path=args.model_path,
        dp_size=args.dp_size,
        mem_fraction_static=args.mem_fraction_static,
        log_level="error",
    )
    engine_load_sec = time.time() - engine_started_at

    processing_started_at = time.time()
    summary = init_summary(existing_results)
    progress = tqdm(
        total=len(selected_rows),
        initial=summary["completed_rows"],
        desc="Curating",
        unit="item",
        dynamic_ncols=True,
    )
    pending_futures: Dict[Future, int] = {}
    last_result_holder: Dict[str, Any] = {}

    try:
        with ThreadPoolExecutor(max_workers=max(args.judge_pipeline_depth, 1)) as pipeline_executor:
            for batch_rows in batched(pending_rows, args.batch_size):
                prompts = [build_vlm_prompt(processor, row) for row in batch_rows]
                images = [load_image_resized(row["image_path_abs"], args.max_image_side) for row in batch_rows]
                sampling_params = {
                    "temperature": args.temperature,
                    "top_p": args.top_p,
                    "max_new_tokens": args.max_tokens,
                    "n": args.num_samples,
                }
                try:
                    outputs = normalize_engine_outputs(
                        engine.generate(prompt=prompts, image_data=images, sampling_params=sampling_params)
                    )
                except Exception as exc:
                    for row in batch_rows:
                        result = build_failed_result(
                            row,
                            num_samples=args.num_samples,
                            easy_threshold=args.easy_threshold,
                            error=str(exc),
                        )
                        append_jsonl(results_path, result)
                        append_jsonl(keep_path if result["decision"] == "keep_for_rl" else easy_path, result)
                        update_summary(summary, result)
                        last_result_holder["result"] = result
                    progress.update(len(batch_rows))
                    continue

                expected = len(batch_rows) * args.num_samples
                outputs = outputs[:expected]

                batch_prepared: list[tuple[Dict[str, Any], List[Dict[str, Any]]]] = []
                for batch_index, row in enumerate(batch_rows):
                    sample_outputs = outputs[batch_index * args.num_samples : (batch_index + 1) * args.num_samples]
                    samples, _ = prepare_samples(row, sample_outputs, args.num_samples)
                    batch_prepared.append((row, samples))

                future = pipeline_executor.submit(
                    judge_and_finalize_batch,
                    batch_prepared,
                    args.easy_threshold,
                    judge_config,
                )
                pending_futures[future] = len(batch_rows)

                flush_done_futures(
                    pending_futures,
                    wait_for_one=len(pending_futures) >= max(args.judge_pipeline_depth, 1),
                    results_path=results_path,
                    keep_path=keep_path,
                    easy_path=easy_path,
                    summary=summary,
                    progress=progress,
                    last_result_holder=last_result_holder,
                )

            while pending_futures:
                flush_done_futures(
                    pending_futures,
                    wait_for_one=True,
                    results_path=results_path,
                    keep_path=keep_path,
                    easy_path=easy_path,
                    summary=summary,
                    progress=progress,
                    last_result_holder=last_result_holder,
                )
    finally:
        progress.close()
        engine.shutdown()

    processing_sec = time.time() - processing_started_at
    completed_rows = summary["completed_rows"]
    keep_rows = summary["keep_for_rl_rows"]
    easy_rows = summary["filtered_out_easy_rows"]

    write_json(
        summary_path,
        {
            "run_name": args.run_name,
            "dataset_root": str(dataset_root),
            "artifact_root": str(artifact_root),
            "benchmarks": benchmarks,
            "slake_languages": [] if not slake_languages else sorted(slake_languages),
            "manifest_rows": len(manifest),
            "selected_rows": len(selected_rows),
            "completed_rows": completed_rows,
            "keep_for_rl_rows": keep_rows,
            "filtered_out_easy_rows": easy_rows,
            "keep_ratio": round(keep_rows / completed_rows, 6) if completed_rows else 0.0,
            "easy_ratio": round(easy_rows / completed_rows, 6) if completed_rows else 0.0,
            "mean_accuracy": round(summary["total_accuracy"] / completed_rows, 6) if completed_rows else 0.0,
            "total_parse_failed": summary["total_parse_failed"],
            "total_request_failed": summary["total_request_failed"],
            "total_judge_used": summary["total_judge_used"],
            "engine_load_sec": round(engine_load_sec, 3),
            "processing_sec": round(processing_sec, 3),
            "model_path": args.model_path,
            "judge_base_url": judge_config.base_url,
            "judge_model_name": judge_config.model_name,
            "judge_enabled": judge_config.enabled,
            "batch_size": args.batch_size,
            "num_samples": args.num_samples,
            "easy_threshold": args.easy_threshold,
            "max_tokens": args.max_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_image_side": args.max_image_side,
            "dp_size": args.dp_size,
            "mem_fraction_static": args.mem_fraction_static,
            "judge_max_workers": args.judge_max_workers,
            "judge_batch_size": args.judge_batch_size,
            "judge_timeout": args.judge_timeout,
            "judge_pipeline_depth": args.judge_pipeline_depth,
            "max_items": args.max_items,
            "manifest_path": str(manifest_path),
            "results_path": str(results_path),
            "keep_path": str(keep_path),
            "easy_path": str(easy_path),
        },
    )
    benchmark_output_paths = write_benchmark_split_files(results_path, run_dir)
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    summary_payload["benchmark_output_paths"] = benchmark_output_paths
    write_json(summary_path, summary_payload)


if __name__ == "__main__":
    main()
