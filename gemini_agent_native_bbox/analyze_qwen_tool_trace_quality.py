from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any, Dict, Iterable, List


VAGUE_TARGETS = {
    "ascites",
    "fluid",
    "lesion",
    "mass",
    "tumor",
    "organ",
    "liver",
    "kidney",
    "lung",
    "heart",
    "spleen",
}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_records(path_or_dir: str) -> Dict[str, Dict[str, Any]]:
    path = Path(path_or_dir)
    if path.is_dir():
        path = path / "predictions.jsonl"
    rows = load_jsonl(path)
    return {str(row.get("question_id")): row for row in rows}


def iter_events(record: Dict[str, Any], event_type: str | None = None) -> Iterable[Dict[str, Any]]:
    for event in record.get("event_trace", []) or []:
        if event_type is None or event.get("type") == event_type:
            yield event


def bbox_area_ratio(bbox: Any) -> float | None:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in bbox]
    except Exception:
        return None
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    return width * height / 1_000_000.0


def is_vague_text(value: Any) -> bool:
    text = str(value or "").strip().lower()
    if not text:
        return True
    if text in VAGUE_TARGETS:
        return True
    return len(text.split()) <= 1


def summarize_records(records: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    counters = Counter()
    sam3_scores: List[float] = []
    sam3_bbox_areas: List[float] = []
    tool_counter = Counter()

    for record in records.values():
        counters["total"] += 1
        if record.get("correct"):
            counters["correct"] += 1
        if record.get("status") == "no_answer":
            counters["no_answer"] += 1
        if record.get("status") == "failed":
            counters["failed"] += 1
        tool_counter.update(record.get("tool_call_counter", {}) or {})

        for event in iter_events(record):
            event_type = event.get("type")
            if event_type == "format_error":
                counters["format_errors"] += 1
            if event_type == "tool_error":
                counters["tool_errors"] += 1
            if event_type == "tool":
                counters["tool_success_events"] += 1
            if event.get("name") != "sam3":
                continue

            counters["sam3_attempt_events"] += 1
            args = event.get("arguments") or {}
            norm_args = event.get("normalized_arguments") or {}
            positive = args.get("positive_points")
            negative = args.get("negative_points")
            if isinstance(positive, list) and len(positive) == 3:
                counters["sam3_raw_3_positive"] += 1
            else:
                counters["sam3_raw_bad_positive_count"] += 1
            if isinstance(negative, list) and len(negative) == 3:
                counters["sam3_raw_3_negative"] += 1
            else:
                counters["sam3_raw_bad_negative_count"] += 1
            if is_vague_text(args.get("text")):
                counters["sam3_vague_text"] += 1
            area = bbox_area_ratio(args.get("bbox"))
            if area is not None:
                sam3_bbox_areas.append(area)
                if area > 0.40:
                    counters["sam3_loose_bbox_area_gt_40pct"] += 1

            norm_positive = norm_args.get("positive_points")
            norm_negative = norm_args.get("negative_points")
            if isinstance(norm_positive, list) and len(norm_positive) == 3 and isinstance(norm_negative, list) and len(norm_negative) == 3:
                counters["sam3_normalized_3pos_3neg"] += 1

            result = event.get("result") or {}
            if result.get("status") == "success":
                counters["sam3_success"] += 1
            try:
                score = float(result.get("top_score"))
                sam3_scores.append(score)
                if score < 0.70:
                    counters["sam3_score_lt_070"] += 1
                elif score < 0.80:
                    counters["sam3_score_070_080"] += 1
            except Exception:
                pass

    total = counters["total"] or 1
    sam3_attempts = counters["sam3_attempt_events"] or 1
    return {
        "counts": dict(counters),
        "rates": {
            "accuracy": counters["correct"] / total,
            "no_answer_rate": counters["no_answer"] / total,
            "failed_rate": counters["failed"] / total,
            "format_errors_per_item": counters["format_errors"] / total,
            "tool_errors_per_item": counters["tool_errors"] / total,
            "sam3_success_rate_per_attempt": counters["sam3_success"] / sam3_attempts,
            "sam3_raw_3pos_3neg_rate": min(counters["sam3_raw_3_positive"], counters["sam3_raw_3_negative"]) / sam3_attempts,
            "sam3_vague_text_rate": counters["sam3_vague_text"] / sam3_attempts,
            "sam3_loose_bbox_rate": counters["sam3_loose_bbox_area_gt_40pct"] / sam3_attempts,
        },
        "tool_call_counter": dict(tool_counter),
        "sam3_top_score_mean": mean(sam3_scores) if sam3_scores else None,
        "sam3_bbox_area_mean": mean(sam3_bbox_areas) if sam3_bbox_areas else None,
    }


def paired_summary(left: Dict[str, Dict[str, Any]], right: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    common = sorted(set(left) & set(right))
    counters = Counter()
    for qid in common:
        l_ok = bool(left[qid].get("correct"))
        r_ok = bool(right[qid].get("correct"))
        if l_ok and r_ok:
            counters["both_correct"] += 1
        elif l_ok and not r_ok:
            counters["left_only_correct"] += 1
        elif r_ok and not l_ok:
            counters["right_only_correct"] += 1
        else:
            counters["both_wrong"] += 1
    return {"common": len(common), **dict(counters)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True, help="Prompt-only result dir or predictions.jsonl")
    parser.add_argument("--right", required=True, help="Cold-start result dir or predictions.jsonl")
    parser.add_argument("--left-name", default="prompt_only")
    parser.add_argument("--right-name", default="cold_start")
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    left = load_records(args.left)
    right = load_records(args.right)
    report = {
        args.left_name: summarize_records(left),
        args.right_name: summarize_records(right),
        "paired": paired_summary(left, right),
    }
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
