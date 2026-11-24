from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List

from gemini_client import DEFAULT_MODEL, GeminiClient
from loop_utils import normalize_text
from prompts import FORMAT_ERROR_TEXT, build_tool_feedback_text, build_user_feedback_text
from verify_prompts import build_verify_system_prompt, build_verify_user_prompt


DEFAULT_INPUT = "./gemini_agent_native_bbox/results/omnimedvqa_random20_seed42_43_rerun_with_answer"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify generated medical reasoning traces.")
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Trace json file or directory containing trace_*.json")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default="https://generativelanguage.googleapis.com/v1beta")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=1600)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    return parser.parse_args()


def discover_trace_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    paths = sorted(input_path.glob("trace_*.json"))
    return [p for p in paths if p.is_file()]


def load_record(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_trace_text(record: Dict[str, Any]) -> str:
    blocks: List[str] = []
    blocks.append("<image>")
    for event in record.get("event_trace", []) or []:
        event_type = event.get("type")
        response_text = normalize_text(event.get("response_text"))
        if event_type == "tool":
            if response_text:
                blocks.append(response_text)
            obs_payload = event.get("result", {})
            tool_name = normalize_text(event.get("name"))
            blocks.append(
                build_tool_feedback_text(
                    tool_name=tool_name,
                    payload=obs_payload,
                    image_descriptions=_verify_image_descriptions(event.get("new_indices", []) or []),
                )
            )
        elif event_type == "tool_error":
            if response_text:
                blocks.append(response_text)
            obs_payload = {"error": event.get("error", ""), "tool": event.get("name")}
            tool_name = normalize_text(event.get("name"))
            blocks.append(
                build_tool_feedback_text(
                    tool_name=tool_name,
                    payload=obs_payload,
                    image_descriptions=_verify_image_descriptions(event.get("new_indices", []) or []),
                )
            )
        elif event_type == "answer":
            if response_text:
                blocks.append(response_text)
        elif event_type == "format_error":
            if response_text:
                blocks.append(f"<invalid_response>{response_text}</invalid_response>")
            blocks.append(build_user_feedback_text(FORMAT_ERROR_TEXT))
        elif event_type == "exception":
            blocks.append(f"<exception>{normalize_text(event.get('error'))}</exception>")
    return "\n".join(blocks).strip()


def _verify_image_descriptions(indices: List[int]) -> List[str]:
    rows: List[str] = []
    for idx in indices:
        rows.append(f'<image image_index="{int(idx)}" python_var="image_{int(idx)}">')
    return rows


def extract_static_facts(record: Dict[str, Any]) -> Dict[str, Any]:
    event_counter = Counter()
    error_messages: List[str] = []
    leakage_markers = (
        "reference answer",
        "provided answer",
        "known answer",
        "gold answer",
        "ground truth",
        "standard answer",
        "trust the reference answer",
    )
    leakage_hits: List[str] = []
    for event in record.get("event_trace", []) or []:
        event_type = normalize_text(event.get("type"))
        event_counter[event_type] += 1
        if event_type in {"format_error", "tool_error", "exception"}:
            error_messages.append(normalize_text(event.get("error")))
        response_text = normalize_text(event.get("response_text")).lower()
        for marker in leakage_markers:
            if marker in response_text:
                leakage_hits.append(marker)
    return {
        "event_counts": dict(event_counter),
        "tool_call_counter": record.get("tool_call_counter", {}),
        "status": record.get("status"),
        "correct": record.get("correct"),
        "tool_calls_count": record.get("tool_calls_count", 0),
        "error_messages": error_messages,
        "supervision_leak_markers_found": sorted(set(leakage_hits)),
    }


def extract_json_object(text: str) -> Dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:].strip()
    try:
        return json.loads(text)
    except Exception:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start : end + 1])
        raise


def normalize_verify_result(payload: Dict[str, Any]) -> Dict[str, Any]:
    verify = dict(payload or {})
    alias_map = {
        "post_tool_reason": "obs_integration_check",
        "post_tool_reason_check": "obs_integration_check",
        "observation_integration_check": "obs_integration_check",
        "answer_check": "final_answer_check",
    }
    for src, dst in alias_map.items():
        if src in verify and dst not in verify:
            verify[dst] = verify.pop(src)
    return verify


def verify_one(
    client: GeminiClient,
    record: Dict[str, Any],
    max_tokens: int,
    temperature: float,
    top_p: float,
    top_k: int,
) -> Dict[str, Any]:
    trace_text = build_trace_text(record)
    static_facts = extract_static_facts(record)
    contents: List[Dict[str, Any]] = [
        {
            "role": "user",
            "parts": [{"text": build_verify_user_prompt(record, trace_text, static_facts)}],
        }
    ]

    response = client.generate(
        contents=contents,
        system_prompt=build_verify_system_prompt(),
        temperature=temperature,
        max_output_tokens=max_tokens,
        top_p=top_p,
        top_k=top_k,
    )
    parsed = normalize_verify_result(extract_json_object(response.text))
    return {
        "question_id": record.get("question_id"),
        "source_trace": record.get("question_id"),
        "verify_result": parsed,
        "raw_response_text": response.text,
        "static_facts": static_facts,
        "gold_answer": record.get("gold_answer"),
        "pred_answer": record.get("pred_answer"),
        "original_correct": record.get("correct"),
        "original_status": record.get("status"),
    }


def summarize(results: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    rows = list(results)
    score_sum = 0.0
    verdict_counter = Counter()
    check_fail_counter = Counter()
    for row in rows:
        verify = row.get("verify_result", {})
        if row.get("status") == "verify_failed":
            verdict_counter["verify_failed"] += 1
            check_fail_counter["verify_exception"] += 1
            continue
        score_sum += float(verify.get("score", 0))
        verdict_counter[normalize_text(verify.get("overall_verdict"))] += 1
        for key, value in verify.items():
            if key.endswith("_check") and normalize_text(value) in {"fail", "incorrect"}:
                check_fail_counter[key] += 1
    return {
        "attempted": len(rows),
        "average_score": round(score_sum / len(rows), 4) if rows else 0.0,
        "overall_verdict_counter": dict(verdict_counter),
        "failed_checks": dict(check_fail_counter),
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace_paths = discover_trace_paths(input_path)
    if args.max_items > 0:
        trace_paths = trace_paths[: args.max_items]

    client = GeminiClient(
        api_key=args.api_key,
        model=args.model,
        base_url=args.base_url,
        timeout=args.request_timeout,
    )

    results: List[Dict[str, Any]] = []
    for trace_path in trace_paths:
        record = load_record(trace_path)
        try:
            result = verify_one(
                client,
                record,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
            )
        except Exception as exc:
            result = {
                "question_id": record.get("question_id"),
                "source_trace": record.get("question_id"),
                "verify_result": {},
                "raw_response_text": "",
                "static_facts": extract_static_facts(record),
                "gold_answer": record.get("gold_answer"),
                "pred_answer": record.get("pred_answer"),
                "original_correct": record.get("correct"),
                "original_status": record.get("status"),
                "status": "verify_failed",
                "verify_error": str(exc),
            }
        result["trace_path"] = str(trace_path)
        results.append(result)

    jsonl_path = output_dir / "verify_results.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = summarize(results)
    summary["input"] = str(input_path)
    summary["model"] = args.model
    summary["num_traces"] = len(trace_paths)
    summary["top_p"] = args.top_p
    summary["top_k"] = args.top_k
    (output_dir / "verify_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
