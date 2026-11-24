from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


DEFAULT_TRACE1_DIR = "./outputs/qwen35_agent/trace_full"
DEFAULT_TRACE0_DIR = "./result/qwen3_direct_eval_full"
DEFAULT_OUTPUT_DIR = "./result/trace_merge_analysis"


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
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


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def load_result_map(prediction_path: Path) -> Dict[int, Dict[str, Any]]:
    rows = read_jsonl(prediction_path)
    result: Dict[int, Dict[str, Any]] = {}
    for row in rows:
        question_id = int(row["question_id"])
        result[question_id] = row
    return result


def event_counter(row: Dict[str, Any]) -> Dict[str, int]:
    counter = Counter()
    for event in row.get("event_trace", []) or []:
        counter[normalize_text(event.get("type"))] += 1
    return dict(counter)


def choose_bucket(trace0_row: Optional[Dict[str, Any]], trace1_row: Optional[Dict[str, Any]]) -> str:
    if trace0_row is None and trace1_row is None:
        return "missing_both"
    if trace0_row is None:
        return "missing_trace0"
    if trace1_row is None:
        return "missing_trace1"

    trace0_correct = bool(trace0_row.get("correct"))
    trace1_correct = bool(trace1_row.get("correct"))

    if trace0_correct and trace1_correct:
        return "A_simple_keep_trace0"
    if (not trace0_correct) and trace1_correct:
        return "B_tool_positive_keep_trace1"
    if trace0_correct and (not trace1_correct):
        return "C_tool_negative_need_trace2"
    return "D_both_wrong_need_trace2"


def candidate_keep(bucket: str) -> str:
    if bucket == "A_simple_keep_trace0":
        return "trace0"
    if bucket == "B_tool_positive_keep_trace1":
        return "trace1"
    if bucket in {"C_tool_negative_need_trace2", "D_both_wrong_need_trace2"}:
        return "trace2_needed"
    return "undetermined"


def simplified_trace0(row: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    return {
        "status": normalize_text(row.get("status")),
        "pred_letter": normalize_text(row.get("pred_letter")),
        "pred_text": normalize_text(row.get("pred_text")),
        "gold_letter": normalize_text(row.get("gold_letter")),
        "gold_text": normalize_text(row.get("gold_text")),
        "correct": bool(row.get("correct")),
        "raw_response": normalize_text(row.get("raw_response")),
    }


def simplified_trace1(row: Optional[Dict[str, Any]], trace1_dir: Path) -> Optional[Dict[str, Any]]:
    if row is None:
        return None
    tool_counter = row.get("tool_call_counter", {}) or {}
    event_stats = event_counter(row)
    question_id = int(row["question_id"])
    trace_json_path = trace1_dir / f"trace_q{question_id:06d}.json"
    trace_html_path = trace1_dir / f"trace_q{question_id:06d}.html"
    return {
        "status": normalize_text(row.get("status")),
        "pred_letter": normalize_text(row.get("pred_letter")),
        "pred_text": normalize_text(row.get("pred_text")),
        "gold_letter": normalize_text(row.get("gold_letter")),
        "gold_text": normalize_text(row.get("gold_text")),
        "correct": bool(row.get("correct")),
        "final_answer_text": normalize_text(row.get("final_answer_text")),
        "tool_call_counter": tool_counter,
        "tool_calls_count": int(row.get("tool_calls_count", 0)),
        "has_python": bool(tool_counter.get("python", 0) > 0),
        "has_retrieve": bool(tool_counter.get("retrieve", 0) > 0),
        "has_visual_tool": bool(tool_counter.get("biomedparse", 0) > 0 or tool_counter.get("medsam3", 0) > 0),
        "event_counter": event_stats,
        "trace_json_path": str(trace_json_path) if trace_json_path.exists() else "",
        "trace_html_path": str(trace_html_path) if trace_html_path.exists() else "",
    }


def build_merged_row(
    question_id: int,
    trace0_row: Optional[Dict[str, Any]],
    trace1_row: Optional[Dict[str, Any]],
    trace1_dir: Path,
) -> Dict[str, Any]:
    bucket = choose_bucket(trace0_row, trace1_row)
    base = trace1_row or trace0_row or {}
    return {
        "question_id": question_id,
        "image_path": normalize_text(base.get("image_path")),
        "question": normalize_text(base.get("question")),
        "source": normalize_text(base.get("source")),
        "gold_letter": normalize_text(base.get("gold_letter")),
        "gold_text": normalize_text(base.get("gold_text")),
        "trace0_exists": trace0_row is not None,
        "trace1_exists": trace1_row is not None,
        "trace0": simplified_trace0(trace0_row),
        "trace1": simplified_trace1(trace1_row, trace1_dir),
        "bucket": bucket,
        "candidate_keep": candidate_keep(bucket),
        "trace2_needed": bucket in {"C_tool_negative_need_trace2", "D_both_wrong_need_trace2"},
    }


def build_stats(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    bucket_counter = Counter(row["bucket"] for row in rows)
    candidate_counter = Counter(row["candidate_keep"] for row in rows)
    trace2_count = sum(1 for row in rows if row["trace2_needed"])
    overlap_count = sum(1 for row in rows if row["trace0_exists"] and row["trace1_exists"])
    trace1_tool_counter = Counter()
    trace1_event_counter = Counter()
    for row in rows:
        trace1 = row.get("trace1")
        if not trace1:
            continue
        trace1_tool_counter.update(trace1.get("tool_call_counter", {}))
        trace1_event_counter.update(trace1.get("event_counter", {}))
    return {
        "total_rows": len(rows),
        "overlap_rows": overlap_count,
        "trace2_needed_rows": trace2_count,
        "bucket_counter": dict(bucket_counter),
        "candidate_keep_counter": dict(candidate_counter),
        "trace1_tool_counter": dict(trace1_tool_counter),
        "trace1_event_counter": dict(trace1_event_counter),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align trace0 and trace1 predictions, then bucket them for trace selection.")
    parser.add_argument("--trace1-dir", default=DEFAULT_TRACE1_DIR)
    parser.add_argument("--trace0-dir", default=DEFAULT_TRACE0_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    trace1_dir = Path(args.trace1_dir)
    trace0_dir = Path(args.trace0_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trace1_map = load_result_map(trace1_dir / "predictions.jsonl")
    trace0_map = load_result_map(trace0_dir / "predictions.jsonl")
    all_question_ids = sorted(set(trace0_map) | set(trace1_map))

    merged_rows = [
        build_merged_row(
            question_id=question_id,
            trace0_row=trace0_map.get(question_id),
            trace1_row=trace1_map.get(question_id),
            trace1_dir=trace1_dir,
        )
        for question_id in all_question_ids
    ]

    merged_path = output_dir / "merged_trace_analysis.jsonl"
    write_jsonl(merged_path, merged_rows)

    bucket_groups: Dict[str, List[Dict[str, Any]]] = {}
    for row in merged_rows:
        bucket_groups.setdefault(row["bucket"], []).append(row)
    for bucket, rows in bucket_groups.items():
        write_jsonl(output_dir / f"{bucket}.jsonl", rows)

    stats = build_stats(merged_rows)
    stats.update(
        {
            "trace0_dir": str(trace0_dir),
            "trace1_dir": str(trace1_dir),
            "output_dir": str(output_dir),
            "trace0_count": len(trace0_map),
            "trace1_count": len(trace1_map),
            "merged_path": str(merged_path),
        }
    )
    with (output_dir / "stats.json").open("w", encoding="utf-8") as file:
        json.dump(stats, file, ensure_ascii=False, indent=2)

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
