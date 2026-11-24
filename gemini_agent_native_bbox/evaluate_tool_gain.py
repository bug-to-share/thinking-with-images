from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm

from judge_client import LLMJudgeClient
from loop_utils import normalize_text


DEFAULT_DIRECT_RESULTS = [
    "./data_curation/results/omnimedvqa_filtered/results.jsonl",
    "./data_curation/results/slake_en_vqa_rad_open_qa_full/results.jsonl",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Select tool-gain traces by joining Qwen3-VL-8B direct sampling buckets, "
            "Gemini with-answer tool traces, optional trace verification, and optional "
            "Qwen3-VL-8B with-tool-context evaluation."
        )
    )
    parser.add_argument("--trace-predictions", required=True, help="Generated with-answer predictions.jsonl")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--direct-results",
        action="append",
        default=[],
        help="Direct 8-sample results.jsonl. Can be passed multiple times. Defaults to known direct result files.",
    )
    parser.add_argument("--source-jsonl", default="", help="Optional sampled jsonl used to generate traces")
    parser.add_argument("--verify-results", default="", help="Optional trace_verify.py verify_results.jsonl")
    parser.add_argument("--require-verify", action="store_true", help="Reject traces without verify pass")
    parser.add_argument("--min-verify-score", type=float, default=8.0)
    parser.add_argument("--min-tool-calls", type=int, default=1)
    parser.add_argument("--direct-gain-max", type=int, default=3, help="correct_count <= this is high-value tool gain")
    parser.add_argument("--direct-rl-min", type=int, default=4)
    parser.add_argument("--direct-rl-max", type=int, default=6)
    parser.add_argument("--easy-min", type=int, default=7)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument(
        "--mode",
        choices=["proxy", "student_with_tool"],
        default="proxy",
        help="proxy uses existing trace correctness; student_with_tool also asks remote Qwen3-VL-8B using tool context.",
    )

    parser.add_argument("--student-base-url", default="", help="OpenAI-compatible Qwen3-VL service base, e.g. http://host:30001/v1")
    parser.add_argument("--student-model", default="", help="Model id. If empty, discover from /models")
    parser.add_argument("--student-api-key", default="EMPTY")
    parser.add_argument("--student-timeout", type=float, default=180.0)
    parser.add_argument("--student-max-tokens", type=int, default=128)
    parser.add_argument("--student-temperature", type=float, default=0.0)
    parser.add_argument("--student-top-p", type=float, default=0.9)
    parser.add_argument("--student-num-workers", type=int, default=4)
    parser.add_argument("--disable-thinking", action="store_true", help="Send chat_template_kwargs.enable_thinking=false")

    parser.add_argument("--judge-base-url", default="", help="Optional LLM-as-judge base URL for student answers")
    parser.add_argument("--judge-api-key", default="EMPTY")
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    return parser.parse_args()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
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


def identity_key(row: Dict[str, Any]) -> str:
    question_id = normalize_text(row.get("question_id"))
    dataset = normalize_text(row.get("dataset") or row.get("benchmark"))
    if dataset:
        return f"{dataset}::{question_id}"
    return question_id


def loose_question_id(row: Dict[str, Any]) -> str:
    return normalize_text(row.get("question_id"))


def build_index(rows: Iterable[Dict[str, Any]]) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    strict: Dict[str, Dict[str, Any]] = {}
    loose: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        if loose_question_id(row):
            loose.setdefault(loose_question_id(row), row)
        if identity_key(row):
            strict.setdefault(identity_key(row), row)
    return strict, loose


def get_indexed(
    row: Dict[str, Any],
    strict: Dict[str, Dict[str, Any]],
    loose: Dict[str, Dict[str, Any]],
) -> Dict[str, Any]:
    return strict.get(identity_key(row)) or loose.get(loose_question_id(row)) or {}


def verify_passes(verify_row: Dict[str, Any], min_score: float) -> Tuple[bool, List[str]]:
    if not verify_row:
        return False, ["missing_verify"]
    if verify_row.get("status") == "verify_failed":
        return False, ["verify_failed"]
    verify = verify_row.get("verify_result") or {}
    reasons: List[str] = []
    if normalize_text(verify.get("overall_verdict")).lower() != "pass":
        reasons.append("overall_verdict_not_pass")
    try:
        score = float(verify.get("score", 0) or 0)
    except Exception:
        score = 0.0
    if score < min_score:
        reasons.append("verify_score_too_low")
    expected = {
        "format_check": "pass",
        "tool_usage_check": "pass",
        "tool_choice_rationale_check": "pass",
        "obs_integration_check": "pass",
        "clinical_correctness": "pass",
        "supervision_leak_check": "pass",
        "tool_necessity_check": "pass",
    }
    for key, wanted in expected.items():
        value = normalize_text(verify.get(key)).lower()
        if value and value != wanted:
            reasons.append(f"{key}:{value}")
    answer_value = normalize_text(verify.get("final_answer_check")).lower()
    if answer_value and answer_value != "correct":
        reasons.append(f"final_answer_check:{answer_value}")
    return not reasons, reasons


def successful_tool_events(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [event for event in trace.get("event_trace", []) or [] if event.get("type") == "tool"]


def trace_quality_reasons(
    trace: Dict[str, Any],
    verify_row: Dict[str, Any],
    args: argparse.Namespace,
) -> Tuple[bool, List[str]]:
    reasons: List[str] = []
    if trace.get("status") != "ok":
        reasons.append(f"trace_status:{trace.get('status')}")
    if trace.get("correct") is not True:
        reasons.append("trace_answer_not_correct")
    if int(trace.get("tool_calls_count") or 0) < args.min_tool_calls:
        reasons.append("too_few_tool_calls")
    if len(successful_tool_events(trace)) < args.min_tool_calls:
        reasons.append("too_few_successful_tool_calls")
    if args.require_verify or verify_row:
        ok, verify_reasons = verify_passes(verify_row, args.min_verify_score)
        if not ok:
            reasons.extend(verify_reasons)
    return not reasons, reasons


def classify_proxy(
    direct_correct_count: Optional[int],
    trace_ok: bool,
    trace: Dict[str, Any],
    args: argparse.Namespace,
) -> str:
    if not trace_ok:
        return "rejected"
    if direct_correct_count is None:
        return "candidate_tool_gain_unknown_direct"
    if direct_correct_count == 0:
        return "hard_candidate_tool_gain"
    if 1 <= direct_correct_count <= args.direct_gain_max:
        return "candidate_tool_gain"
    if args.direct_rl_min <= direct_correct_count <= args.direct_rl_max:
        return "possible_tool_gain"
    if direct_correct_count >= args.easy_min:
        return "tool_neutral"
    return "candidate_tool_gain_unknown_direct"


def sanitize_for_prompt(value: Any, max_string: int = 1200) -> Any:
    if isinstance(value, dict):
        output = {}
        for key, item in value.items():
            key_text = str(key)
            if any(marker in key_text.lower() for marker in ("base64", "bytes", "image_data")):
                continue
            output[key_text] = sanitize_for_prompt(item, max_string=max_string)
        return output
    if isinstance(value, list):
        return [sanitize_for_prompt(item, max_string=max_string) for item in value[:20]]
    if isinstance(value, str):
        return value[:max_string]
    return value


def image_index_paths(trace: Dict[str, Any]) -> Dict[int, str]:
    paths: Dict[int, str] = {}
    for item in trace.get("image_inventory", []) or []:
        try:
            paths[int(item.get("image_index"))] = normalize_text(item.get("path"))
        except Exception:
            continue
    if trace.get("model_image_path"):
        paths.setdefault(0, normalize_text(trace.get("model_image_path")))
    elif trace.get("image_path"):
        paths.setdefault(0, normalize_text(trace.get("image_path")))
    return paths


def build_tool_context(trace: Dict[str, Any]) -> Tuple[str, List[str]]:
    blocks: List[str] = []
    image_paths: List[str] = []
    paths_by_idx = image_index_paths(trace)
    if paths_by_idx.get(0):
        image_paths.append(paths_by_idx[0])

    for event in trace.get("event_trace", []) or []:
        if event.get("type") != "tool":
            continue
        name = normalize_text(event.get("name"))
        args = sanitize_for_prompt(event.get("arguments") or {})
        result = sanitize_for_prompt(event.get("result") or {})
        new_indices = []
        for idx in event.get("new_indices", []) or []:
            try:
                int_idx = int(idx)
            except Exception:
                continue
            new_indices.append(int_idx)
            path = paths_by_idx.get(int_idx)
            if path:
                image_paths.append(path)
        blocks.append(
            "\n".join(
                [
                    f"Tool: {name}",
                    f"Arguments: {json.dumps(args, ensure_ascii=False)}",
                    f"Result: {json.dumps(result, ensure_ascii=False)}",
                    f"New images registered: {', '.join('image_' + str(i) for i in new_indices) if new_indices else 'none'}",
                ]
            )
        )
    return "\n\n".join(blocks), unique_existing_paths(image_paths)


def unique_existing_paths(paths: Iterable[str]) -> List[str]:
    seen = set()
    output: List[str] = []
    for path in paths:
        if not path or path in seen:
            continue
        if not Path(path).exists():
            continue
        seen.add(path)
        output.append(path)
    return output


def image_to_data_url(path: str) -> str:
    mime, _ = mimetypes.guess_type(path)
    if not mime:
        mime = "image/png"
    data = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime};base64,{data}"


class OpenAIChatClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        model: str,
        timeout: float,
        disable_thinking: bool,
    ):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key or "EMPTY"
        self.timeout = timeout
        self.disable_thinking = disable_thinking
        self.http = requests.Session()
        self.model = model or self._discover_model()
        self._lock = threading.Lock()

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _discover_model(self) -> str:
        response = self.http.get(f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        models = payload.get("data") or []
        if not models:
            raise RuntimeError(f"No models found at {self.base_url}")
        return normalize_text(models[0].get("id"))

    def chat(
        self,
        messages: List[Dict[str, Any]],
        max_tokens: int,
        temperature: float,
        top_p: float,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if self.disable_thinking:
            payload["chat_template_kwargs"] = {"enable_thinking": False}
        response = self.http.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        message = data["choices"][0]["message"]
        return normalize_text(message.get("content"))


def build_student_messages(trace: Dict[str, Any]) -> List[Dict[str, Any]]:
    tool_context, image_paths = build_tool_context(trace)
    question = normalize_text(trace.get("question"))
    text = (
        "You are answering a medical visual question.\n"
        "The images and tool observations below come from a real previous tool trajectory. "
        "Tool results may be imperfect, so use them as evidence only when they are relevant and consistent with the image.\n\n"
        f"Question: {question}\n\n"
        "<tool_observations>\n"
        f"{tool_context or 'No successful tool observations.'}\n"
        "</tool_observations>\n\n"
        "Answer the question concisely. Do not mention that a reference answer exists. "
        "Return only the final answer text."
    )
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for idx, path in enumerate(image_paths[:6]):
        label = "original working image" if idx == 0 else f"tool generated image {idx}"
        content.append({"type": "text", "text": f"\nImage {idx}: {label}"})
        content.append({"type": "image_url", "image_url": {"url": image_to_data_url(path)}})
    return [
        {
            "role": "system",
            "content": "You are a careful medical VQA assistant. Answer concisely based on the image and observations.",
        },
        {"role": "user", "content": content},
    ]


ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.IGNORECASE | re.DOTALL)


def clean_answer(text: str) -> str:
    text = normalize_text(text)
    match = ANSWER_TAG_RE.search(text)
    if match:
        text = match.group(1)
    text = re.sub(r"^\s*(answer|final answer)\s*[:：]\s*", "", text, flags=re.IGNORECASE)
    return normalize_text(text)


def exact_answer_match(prediction: str, gold: str) -> bool:
    return normalize_text(prediction).casefold() == normalize_text(gold).casefold()


def judge_student_answer(
    judge_client: Optional[LLMJudgeClient],
    question: str,
    prediction: str,
    gold: str,
) -> Tuple[bool, Dict[str, Any]]:
    if not prediction:
        return False, {"method": "empty_answer"}
    if exact_answer_match(prediction, gold):
        return True, {"method": "exact"}
    if judge_client is None:
        return False, {"method": "exact_only"}
    result = judge_client.judge_semantic_equivalence(question=question, prediction=prediction, gold=gold)
    return result.correct, {
        "method": "llm_judge",
        "judge_model": result.model_name,
        "judge_raw_text": result.raw_text,
    }


def classify_student(
    direct_correct_count: Optional[int],
    student_correct: bool,
    trace_ok: bool,
    args: argparse.Namespace,
) -> str:
    if not trace_ok:
        return "rejected"
    if direct_correct_count is None:
        return "tool_gain" if student_correct else "unsolved"
    if student_correct and direct_correct_count == 0:
        return "hard_tool_gain"
    if student_correct and direct_correct_count <= args.direct_gain_max:
        return "tool_gain"
    if student_correct and args.direct_rl_min <= direct_correct_count <= args.direct_rl_max:
        return "possible_tool_gain"
    if student_correct and direct_correct_count >= args.easy_min:
        return "tool_neutral"
    if (not student_correct) and direct_correct_count >= args.easy_min:
        return "tool_harmful"
    return "unsolved"


def merge_row(
    trace: Dict[str, Any],
    source_row: Dict[str, Any],
    direct_row: Dict[str, Any],
    verify_row: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    direct_correct_count = first_int(
        source_row.get("correct_count"),
        direct_row.get("correct_count"),
    )
    direct_accuracy = first_value(source_row.get("accuracy"), direct_row.get("accuracy"))
    trace_ok, rejection_reasons = trace_quality_reasons(trace, verify_row, args)
    category = classify_proxy(direct_correct_count, trace_ok, trace, args)
    verify = verify_row.get("verify_result") or {}
    return {
        "question_id": trace.get("question_id"),
        "category": category,
        "trace_ok": trace_ok,
        "rejection_reasons": rejection_reasons,
        "direct_correct_count": direct_correct_count,
        "direct_accuracy": direct_accuracy,
        "difficulty_bucket": first_value(source_row.get("difficulty_bucket"), direct_row.get("difficulty_bucket")),
        "recommended_use": first_value(source_row.get("recommended_use"), direct_row.get("recommended_use")),
        "trace_correct": trace.get("correct"),
        "trace_status": trace.get("status"),
        "trace_tool_calls_count": trace.get("tool_calls_count", 0),
        "trace_tool_call_counter": trace.get("tool_call_counter", {}),
        "verify_overall_verdict": verify.get("overall_verdict"),
        "verify_score": verify.get("score"),
        "verify_checks": verify,
        "question": trace.get("question"),
        "gold_answer": trace.get("gold_answer"),
        "trace_pred_answer": trace.get("pred_answer"),
        "image_path": trace.get("image_path"),
        "model_image_path": trace.get("model_image_path"),
        "trace_path": trace_file_path(trace),
        "source_row": compact_source_row(source_row or direct_row),
    }


def first_int(*values: Any) -> Optional[int]:
    for value in values:
        if value is None or value == "":
            continue
        try:
            return int(value)
        except Exception:
            continue
    return None


def first_value(*values: Any) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return None


def trace_file_path(trace: Dict[str, Any]) -> str:
    question_id = normalize_text(trace.get("question_id"))
    if not question_id:
        return ""
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", question_id).strip("_")
    jsonl_path = normalize_text(trace.get("jsonl_path"))
    # The trace path is not stored in predictions.jsonl, so leave best-effort empty.
    return "" if jsonl_path else ""


def compact_source_row(row: Dict[str, Any]) -> Dict[str, Any]:
    if not row:
        return {}
    keep = [
        "question_id",
        "dataset",
        "benchmark",
        "split",
        "question_type",
        "answer_type",
        "modality_type",
        "modality",
        "location",
        "content_type",
        "correct_count",
        "accuracy",
        "difficulty_bucket",
        "recommended_use",
        "image_path_abs",
        "image_path_rel",
    ]
    return {key: row.get(key) for key in keep if key in row}


def evaluate_student_one(
    row: Dict[str, Any],
    trace: Dict[str, Any],
    client: OpenAIChatClient,
    judge_client: Optional[LLMJudgeClient],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    if not row.get("trace_ok"):
        row["student_with_tool"] = {"skipped": True, "reason": "trace_not_ok"}
        return row
    try:
        messages = build_student_messages(trace)
        raw_answer = client.chat(
            messages,
            max_tokens=args.student_max_tokens,
            temperature=args.student_temperature,
            top_p=args.student_top_p,
        )
        pred_answer = clean_answer(raw_answer)
        correct, judge_info = judge_student_answer(
            judge_client=judge_client,
            question=normalize_text(trace.get("question")),
            prediction=pred_answer,
            gold=normalize_text(trace.get("gold_answer")),
        )
        row["student_with_tool"] = {
            "status": "ok",
            "model": client.model,
            "raw_answer": raw_answer,
            "pred_answer": pred_answer,
            "correct": correct,
            "judge": judge_info,
        }
        row["category"] = classify_student(row.get("direct_correct_count"), correct, bool(row.get("trace_ok")), args)
        return row
    except Exception as exc:
        row["student_with_tool"] = {"status": "failed", "error": str(exc)}
        row["category"] = "student_eval_failed"
        return row


def load_inputs(args: argparse.Namespace) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Dict[str, Any]]]:
    traces = read_jsonl(Path(args.trace_predictions))
    if args.max_items > 0:
        traces = traces[: args.max_items]

    direct_paths = args.direct_results or DEFAULT_DIRECT_RESULTS
    direct_rows: List[Dict[str, Any]] = []
    for path in direct_paths:
        direct_rows.extend(read_jsonl(Path(path)))

    source_rows = read_jsonl(Path(args.source_jsonl)) if args.source_jsonl else []
    verify_rows = read_jsonl(Path(args.verify_results)) if args.verify_results else []

    direct_strict, direct_loose = build_index(direct_rows)
    source_strict, source_loose = build_index(source_rows)
    verify_by_qid = {loose_question_id(row): row for row in verify_rows if loose_question_id(row)}
    return traces, direct_strict, direct_loose, source_strict, source_loose, verify_by_qid, {}


def write_outputs(rows: List[Dict[str, Any]], output_dir: Path, args: argparse.Namespace) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    category_rows: Dict[str, List[Dict[str, Any]]] = {}
    for row in rows:
        category_rows.setdefault(normalize_text(row.get("category")) or "unknown", []).append(row)

    for category, items in sorted(category_rows.items()):
        write_jsonl(output_dir / f"{category}.jsonl", items)
    write_jsonl(output_dir / "all_evaluated.jsonl", rows)

    summary = {
        "trace_predictions": args.trace_predictions,
        "source_jsonl": args.source_jsonl,
        "verify_results": args.verify_results,
        "mode": args.mode,
        "total": len(rows),
        "category_counts": {name: len(items) for name, items in sorted(category_rows.items())},
        "direct_correct_count_distribution": dict(Counter(str(row.get("direct_correct_count")) for row in rows)),
        "trace_tool_counter": dict(sum_counters(row.get("trace_tool_call_counter", {}) for row in rows)),
        "require_verify": args.require_verify,
        "min_verify_score": args.min_verify_score,
        "min_tool_calls": args.min_tool_calls,
        "student_base_url": args.student_base_url,
        "student_model": rows[0].get("student_with_tool", {}).get("model") if rows and args.mode == "student_with_tool" else "",
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def sum_counters(counters: Iterable[Dict[str, Any]]) -> Counter:
    output: Counter[str] = Counter()
    for counter in counters:
        for key, value in (counter or {}).items():
            try:
                output[str(key)] += int(value)
            except Exception:
                continue
    return output


def main() -> None:
    args = parse_args()
    traces, direct_strict, direct_loose, source_strict, source_loose, verify_by_qid, _ = load_inputs(args)

    rows: List[Dict[str, Any]] = []
    trace_by_qid: Dict[str, Dict[str, Any]] = {}
    for trace in traces:
        source_row = get_indexed(trace, source_strict, source_loose)
        direct_row = get_indexed(trace, direct_strict, direct_loose)
        verify_row = verify_by_qid.get(loose_question_id(trace), {})
        row = merge_row(trace, source_row, direct_row, verify_row, args)
        rows.append(row)
        trace_by_qid[loose_question_id(trace)] = trace

    if args.mode == "student_with_tool":
        if not args.student_base_url:
            raise ValueError("--student-base-url is required for student_with_tool mode")
        student_client = OpenAIChatClient(
            base_url=args.student_base_url,
            api_key=args.student_api_key,
            model=args.student_model,
            timeout=args.student_timeout,
            disable_thinking=args.disable_thinking,
        )
        judge_client = (
            LLMJudgeClient(base_url=args.judge_base_url, api_key=args.judge_api_key, timeout=args.judge_timeout)
            if args.judge_base_url
            else None
        )
        evaluated: List[Dict[str, Any]] = []
        with ThreadPoolExecutor(max_workers=max(1, args.student_num_workers)) as executor:
            futures = {
                executor.submit(
                    evaluate_student_one,
                    row,
                    trace_by_qid.get(normalize_text(row.get("question_id")), {}),
                    student_client,
                    judge_client,
                    args,
                ): row
                for row in rows
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="student-with-tool"):
                evaluated.append(future.result())
        rows = sorted(evaluated, key=lambda item: normalize_text(item.get("question_id")))

    summary = write_outputs(rows, Path(args.output_dir), args)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
