from __future__ import annotations

import argparse
import base64
import json
import mimetypes
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import requests
from tqdm import tqdm

if __package__:
    from .loop_utils import normalize_text
    from .prompts import FORMAT_ERROR_TEXT, build_tool_feedback_text, build_user_feedback_text
    from .verify_prompts_v2 import build_verify_v2_system_prompt, build_verify_v2_user_prompt
else:
    from loop_utils import normalize_text
    from prompts import FORMAT_ERROR_TEXT, build_tool_feedback_text, build_user_feedback_text
    from verify_prompts_v2 import build_verify_v2_system_prompt, build_verify_v2_user_prompt


DEFAULT_INPUT = (
    "./gemini_agent_native_bbox/results/"
    "cold_start_2000_seed42_with_answer_prompt_v2"
)
DEFAULT_OUTPUT_DIR = (
    "./gemini_agent_native_bbox/results/"
    "trace_verify_v2"
)

ALLOWED_TOOLS = {"python_exec", "sam3"}
LEAK_PATTERNS = (
    r"\binternal supervised answer\b",
    r"\bverified target\b",
    r"\breference answer\b",
    r"\bprovided answer\b",
    r"\bknown answer\b",
    r"\bgold answer\b",
    r"\bground[- ]?truth\b",
    r"\bstandard answer\b",
    r"\bthe answer is given\b",
    r"\bi know the answer\b",
)
CAUTION_WORDS = re.compile(
    r"\b(weak|uncertain|uncertainty|unreliable|low[- ]?confidence|"
    r"insufficient|cautious|caution|not fully|partial|imperfect|"
    r"may not|might not|questionable|ambiguous|however|but|although)\b",
    re.IGNORECASE,
)
CONFIRM_WORDS = re.compile(
    r"\b(confirm|confirms|confirmed|successfully|supports|validates|"
    r"verified|proves|demonstrates|establishes)\b",
    re.IGNORECASE,
)
VISUAL_EVIDENCE_WORDS = re.compile(
    r"\b(image|crop|zoom|mask|segmentation|overlay|bbox|region|lesion|"
    r"opacity|signal|density|margin|border|shape|size|location|"
    r"left|right|anterior|posterior|superior|inferior|medial|lateral|"
    r"hyperintens|hypointens|hyperreflect|hyporeflect|enhancement|"
    r"retina|lung|liver|heart|bone|tissue|cell|vessel|organ)\b",
    re.IGNORECASE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evidence-aware verifier for generated medical visual tool traces."
    )
    parser.add_argument("--input", default=DEFAULT_INPUT, help="Trace json file or directory containing trace_*.json")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skip-llm", action="store_true", help="Only run deterministic precheck.")
    parser.add_argument("--llm-base-url", default=os.environ.get("LLM_AS_A_JUDGE_BASE", ""))
    parser.add_argument("--llm-api-key", default=os.environ.get("LLM_AS_A_JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--llm-model", default="", help="OpenAI-compatible model name. Auto-discover if empty.")
    parser.add_argument("--llm-timeout", type=float, default=120.0)
    parser.add_argument("--llm-max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--include-images", action="store_true", help="Attach original/tool images to the verifier request.")
    parser.add_argument("--max-images", type=int, default=6, help="Maximum images to attach per trace.")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def discover_trace_paths(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    return sorted(p for p in input_path.glob("trace_*.json") if p.is_file())


def load_record(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def encode_image_as_data_url(path_value: Any) -> str:
    path_text = normalize_text(path_value)
    if not path_text:
        return ""
    path = Path(path_text)
    if not path.exists() or not path.is_file():
        return ""
    mime_type, _ = mimetypes.guess_type(str(path))
    if not mime_type:
        mime_type = "image/png"
    with path.open("rb") as file:
        encoded = base64.b64encode(file.read()).decode("utf-8")
    return f"data:{mime_type};base64,{encoded}"


def collect_image_parts(record: Dict[str, Any], max_images: int) -> List[Dict[str, Any]]:
    parts: List[Dict[str, Any]] = []
    seen: set[str] = set()
    inventory = record.get("image_inventory") or []
    rows = [row for row in inventory if isinstance(row, dict)] if isinstance(inventory, list) else []
    if not rows:
        fallback = record.get("model_image_path") or record.get("image_path") or record.get("image_path_abs")
        if fallback:
            rows.append({"image_index": 0, "path": fallback})

    attached = 0
    for row in rows:
        if attached >= max_images:
            break
        image_path = normalize_text(row.get("path"))
        if not image_path or image_path in seen:
            continue
        data_url = encode_image_as_data_url(image_path)
        if not data_url:
            continue
        seen.add(image_path)
        attached += 1
        image_index = row.get("image_index", "?")
        parts.append({"type": "text", "text": f'Image below: image_index={image_index}, path="{image_path}"'})
        parts.append({"type": "image_url", "image_url": {"url": data_url}})
    return parts


def assistant_response_text(record: Dict[str, Any]) -> str:
    parts: List[str] = []
    for event in record.get("event_trace", []) or []:
        if event.get("type") in {"tool", "answer", "format_error"}:
            parts.append(normalize_text(event.get("response_text")))
    return "\n".join(part for part in parts if part)


def build_trace_text(record: Dict[str, Any]) -> str:
    blocks: List[str] = ["<image>"]
    for event in record.get("event_trace", []) or []:
        event_type = normalize_text(event.get("type"))
        response_text = normalize_text(event.get("response_text"))
        if event_type == "tool":
            if response_text:
                blocks.append(response_text)
            tool_name = normalize_text(event.get("name"))
            blocks.append(
                build_tool_feedback_text(
                    tool_name=tool_name,
                    payload=event.get("result", {}),
                    image_descriptions=_image_descriptions(event.get("new_indices", []) or []),
                )
            )
        elif event_type == "answer":
            if response_text:
                blocks.append(response_text)
        elif event_type == "format_error":
            if response_text:
                blocks.append(f"<invalid_response>{response_text}</invalid_response>")
            blocks.append(build_user_feedback_text(FORMAT_ERROR_TEXT))
        elif event_type in {"tool_error", "exception"}:
            blocks.append(f"<{event_type}>{normalize_text(event.get('error'))}</{event_type}>")
    return "\n".join(blocks).strip()


def _image_descriptions(indices: Iterable[Any]) -> List[str]:
    rows: List[str] = []
    for idx in indices:
        try:
            int_idx = int(idx)
        except Exception:
            continue
        rows.append(f'<image image_index="{int_idx}" python_var="image_{int_idx}">')
    return rows


def extract_json_object(text: str) -> Dict[str, Any]:
    text = normalize_text(text)
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


def normalize_answer(value: Any) -> str:
    text = normalize_text(value).lower()
    text = re.sub(r"\s+", " ", text)
    text = text.strip(" .。")
    return text


def final_answer_matches(record: Dict[str, Any]) -> bool:
    if record.get("correct") is True:
        return True
    pred = normalize_answer(record.get("pred_answer") or record.get("final_answer_text"))
    gold = normalize_answer(record.get("gold_answer"))
    return bool(pred and gold and (pred == gold or pred in gold or gold in pred))


def valid_gemini_bbox(bbox: Any) -> Tuple[bool, str]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False, "bbox_missing_or_not_4d"
    try:
        ymin, xmin, ymax, xmax = [float(x) for x in bbox]
    except Exception:
        return False, "bbox_non_numeric"
    if not (0 <= ymin <= 1000 and 0 <= xmin <= 1000 and 0 <= ymax <= 1000 and 0 <= xmax <= 1000):
        return False, "bbox_out_of_0_1000"
    if ymax <= ymin or xmax <= xmin:
        return False, "bbox_invalid_order"
    return True, ""


def valid_gemini_points(points: Any) -> Tuple[bool, str]:
    if points in (None, []):
        return False, "positive_points_missing"
    if not isinstance(points, list):
        return False, "positive_points_not_list"
    for point in points:
        if not isinstance(point, list) or len(point) != 2:
            return False, "point_not_2d"
        try:
            y, x = [float(v) for v in point]
        except Exception:
            return False, "point_non_numeric"
        if not (0 <= y <= 1000 and 0 <= x <= 1000):
            return False, "point_out_of_0_1000"
    return True, ""


def has_reason_and_action(event: Dict[str, Any]) -> Tuple[bool, str]:
    response = normalize_text(event.get("response_text"))
    event_type = normalize_text(event.get("type"))
    if not response:
        return False, "empty_response_text"
    if "<reason>" not in response or "</reason>" not in response:
        return False, "missing_reason_tag"
    if event_type == "tool":
        if response.count("<tool_call>") != 1 or response.count("</tool_call>") != 1:
            return False, "tool_call_tag_count_not_one"
        tail = response.split("</tool_call>", 1)[1].strip()
        if tail:
            return False, "text_after_tool_call"
    if event_type == "answer":
        if response.count("<answer>") != 1 or response.count("</answer>") != 1:
            return False, "answer_tag_count_not_one"
    return True, ""


def next_reason(events: List[Dict[str, Any]], index: int) -> str:
    for event in events[index + 1 :]:
        if event.get("type") in {"tool", "answer", "format_error"}:
            return normalize_text(event.get("reason") or event.get("response_text"))
    return ""


def deterministic_precheck(record: Dict[str, Any]) -> Dict[str, Any]:
    events = record.get("event_trace", []) or []
    failure_reasons: List[str] = []
    warnings: List[str] = []
    counters = Counter(normalize_text(event.get("type")) for event in events)
    tool_counter = Counter()
    sam3_scores: List[float] = []
    low_sam_events: List[Dict[str, Any]] = []
    tool_result_integration_failures: List[Dict[str, Any]] = []
    invalid_tool_events: List[Dict[str, Any]] = []

    assistant_text = assistant_response_text(record).lower()
    leak_hits = sorted({pat for pat in LEAK_PATTERNS if re.search(pat, assistant_text, re.IGNORECASE)})

    if record.get("status") != "ok":
        failure_reasons.append(f"trace_status_{record.get('status')}")
    if not final_answer_matches(record):
        failure_reasons.append("final_answer_not_marked_correct")
    if leak_hits:
        failure_reasons.append("supervision_leak")
    if counters.get("answer", 0) != 1:
        failure_reasons.append("answer_event_count_not_one")
    if counters.get("format_error", 0) > 0:
        warnings.append("format_error_recovered")
    if counters.get("exception", 0) > 0:
        failure_reasons.append("exception_event_present")
    if counters.get("tool_error", 0) > 0:
        failure_reasons.append("tool_error_event_present")

    for idx, event in enumerate(events):
        event_type = normalize_text(event.get("type"))
        if event_type in {"tool", "answer"}:
            ok, reason = has_reason_and_action(event)
            if not ok:
                if event_type == "tool":
                    failure_reasons.append(f"bad_tool_response_format:{reason}")
                else:
                    warnings.append(f"bad_answer_response_format:{reason}")
        if event_type != "tool":
            continue

        name = normalize_text(event.get("name"))
        tool_counter[name] += 1
        args = event.get("arguments") or {}
        result = event.get("result") or {}
        reason_text = normalize_text(event.get("reason"))
        nreason = next_reason(events, idx)

        if name not in ALLOWED_TOOLS:
            invalid_tool_events.append({"round": event.get("round"), "tool": name, "reason": "tool_not_allowed"})
            failure_reasons.append("tool_not_allowed")
        if not reason_text or len(reason_text.split()) < 6:
            failure_reasons.append(f"weak_tool_choice_rationale:{name}")

        if name == "python_exec":
            code = normalize_text(args.get("code"))
            if not code:
                invalid_tool_events.append({"round": event.get("round"), "tool": name, "reason": "missing_code"})
                failure_reasons.append("python_exec_missing_code")
            if result.get("status") != "success":
                failure_reasons.append("python_exec_not_success")
            if nreason and not VISUAL_EVIDENCE_WORDS.search(nreason):
                tool_result_integration_failures.append(
                    {"round": event.get("round"), "tool": name, "reason": "next_reason_lacks_visual_evidence_terms"}
                )

        if name == "sam3":
            if not normalize_text(args.get("text")):
                invalid_tool_events.append({"round": event.get("round"), "tool": name, "reason": "missing_text"})
                failure_reasons.append("sam3_missing_text")
            bbox_ok, bbox_reason = valid_gemini_bbox(args.get("bbox"))
            if not bbox_ok:
                invalid_tool_events.append({"round": event.get("round"), "tool": name, "reason": bbox_reason})
                failure_reasons.append(f"sam3_{bbox_reason}")
            points_ok, points_reason = valid_gemini_points(args.get("positive_points"))
            if not points_ok:
                invalid_tool_events.append({"round": event.get("round"), "tool": name, "reason": points_reason})
                failure_reasons.append(f"sam3_{points_reason}")
            if result.get("status") != "success":
                failure_reasons.append("sam3_not_success")

            score = result.get("top_score")
            if isinstance(score, (int, float)):
                sam3_scores.append(float(score))
                if score < 0.8:
                    next_has_caution = bool(CAUTION_WORDS.search(nreason))
                    next_has_confirm = bool(CONFIRM_WORDS.search(nreason))
                    low_row = {
                        "round": event.get("round"),
                        "score": float(score),
                        "text": args.get("text"),
                        "next_has_caution": next_has_caution,
                        "next_has_confirm": next_has_confirm,
                    }
                    low_sam_events.append(low_row)
                    if score < 0.7 and not next_has_caution:
                        failure_reasons.append("low_sam3_without_caution")
                    if next_has_confirm and not next_has_caution:
                        failure_reasons.append("low_sam3_overtrusted")
            else:
                warnings.append("sam3_missing_score")

            if nreason:
                mentions_tool = re.search(r"\b(sam3|segmentation|mask|overlay|score|bbox|image_\d+)\b", nreason, re.I)
                if not mentions_tool:
                    tool_result_integration_failures.append(
                        {"round": event.get("round"), "tool": name, "reason": "next_reason_does_not_mention_sam3_result"}
                    )

    if len(tool_result_integration_failures) > 0:
        failure_reasons.append("tool_result_not_integrated")
    if sum(tool_counter.values()) == 0:
        failure_reasons.append("no_tool_call")
    if sum(tool_counter.values()) > 4:
        warnings.append("many_tool_calls")

    final_reason = ""
    for event in reversed(events):
        if event.get("type") == "answer":
            final_reason = normalize_text(event.get("reason") or event.get("response_text"))
            break
    final_reason_words = len(final_reason.split())
    if final_reason_words < 30:
        warnings.append("short_final_reason")
    if final_reason and not VISUAL_EVIDENCE_WORDS.search(final_reason):
        failure_reasons.append("final_reason_lacks_visual_evidence")

    hard_fail_markers = {
        "supervision_leak",
        "final_answer_not_marked_correct",
        "answer_event_count_not_one",
        "exception_event_present",
        "tool_error_event_present",
        "no_tool_call",
    }
    unique_failures = sorted(set(failure_reasons))
    unique_warnings = sorted(set(warnings))

    score = 10
    for reason in unique_failures:
        if reason in hard_fail_markers:
            score -= 4
        elif reason in {"low_sam3_overtrusted", "low_sam3_without_caution", "tool_result_not_integrated"}:
            score -= 2
        else:
            score -= 1
    score -= min(2, len(unique_warnings))
    score = max(0, min(10, score))

    return {
        "event_counts": dict(counters),
        "tool_counter": dict(tool_counter),
        "format_check": "fail" if any(r.startswith("bad_tool_response_format") for r in unique_failures) else "pass",
        "answer_correctness_check": "pass" if final_answer_matches(record) else "fail",
        "supervision_leak_check": "fail" if leak_hits else "pass",
        "tool_call_validity_check": "fail" if invalid_tool_events else "pass",
        "tool_choice_rationale_check": "fail" if any(r.startswith("weak_tool_choice_rationale") for r in unique_failures) else "pass",
        "tool_result_integration_check": "fail" if tool_result_integration_failures else "pass",
        "tool_trust_calibration_check": "fail"
        if ("low_sam3_overtrusted" in unique_failures or "low_sam3_without_caution" in unique_failures)
        else "pass",
        "visual_evidence_consistency_check": "fail"
        if ("final_reason_lacks_visual_evidence" in unique_failures or "tool_result_not_integrated" in unique_failures)
        else "pass",
        "unnecessary_tool_use_check": "warn" if "many_tool_calls" in unique_warnings else "pass",
        "clinical_reasoning_quality_check": "warn" if "short_final_reason" in unique_warnings else "pass",
        "overall_verdict": "pass" if score >= 8 and not (set(unique_failures) & hard_fail_markers) else "fail",
        "score": score,
        "failure_reasons": unique_failures,
        "warnings": unique_warnings,
        "leak_hits": leak_hits,
        "invalid_tool_events": invalid_tool_events,
        "tool_result_integration_failures": tool_result_integration_failures,
        "sam3_score_count": len(sam3_scores),
        "sam3_min_score": min(sam3_scores) if sam3_scores else None,
        "sam3_mean_score": round(sum(sam3_scores) / len(sam3_scores), 4) if sam3_scores else None,
        "low_sam3_events": low_sam_events,
        "final_reason_words": final_reason_words,
    }


class OpenAICompatibleVerifier:
    def __init__(
        self,
        base_url: str,
        api_key: str = "EMPTY",
        model: str = "",
        timeout: float = 120.0,
        max_tokens: int = 1024,
        temperature: float = 0.0,
        include_images: bool = False,
        max_images: int = 6,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if not self.base_url:
            raise ValueError("--llm-base-url or LLM_AS_A_JUDGE_BASE is required unless --skip-llm is set")
        self.api_key = api_key or "EMPTY"
        self.timeout = float(timeout)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.include_images = bool(include_images)
        self.max_images = int(max_images)
        self.http = requests.Session()
        self.model = model or self._discover_model()

    def _headers(self) -> Dict[str, str]:
        return {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}

    def _discover_model(self) -> str:
        response = self.http.get(f"{self.base_url}/models", headers=self._headers(), timeout=self.timeout)
        response.raise_for_status()
        models = response.json().get("data") or []
        if not models:
            raise RuntimeError(f"No models found at {self.base_url}")
        return normalize_text(models[0].get("id"))

    def verify(self, record: Dict[str, Any], trace_text: str, precheck: Dict[str, Any]) -> Dict[str, Any]:
        user_prompt = build_verify_v2_user_prompt(record, trace_text, precheck)
        user_content: Any = user_prompt
        attached_images = 0
        if self.include_images:
            image_parts = collect_image_parts(record, self.max_images)
            attached_images = sum(1 for part in image_parts if part.get("type") == "image_url")
            user_content = [{"type": "text", "text": user_prompt}]
            user_content.extend(image_parts)
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": build_verify_v2_system_prompt()},
                {"role": "user", "content": user_content},
            ],
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        response = self.http.post(
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        raw = normalize_text(response.json()["choices"][0]["message"]["content"])
        return {
            "llm_verify_result": extract_json_object(raw),
            "raw_response_text": raw,
            "llm_model": self.model,
            "include_images": self.include_images,
            "attached_images": attached_images,
        }


def merge_verdict(precheck: Dict[str, Any], llm_result: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    llm_payload = (llm_result or {}).get("llm_verify_result") or {}
    if not llm_payload:
        return {
            "overall_verdict": precheck.get("overall_verdict"),
            "score": precheck.get("score"),
            "source": "deterministic_only",
            "recommended_action": "keep_tool_trace" if precheck.get("overall_verdict") == "pass" else "discard",
        }
    pre_score = float(precheck.get("score", 0) or 0)
    try:
        llm_score = float(llm_payload.get("score", 0) or 0)
    except Exception:
        llm_score = 0.0
    hard_fail = (
        precheck.get("supervision_leak_check") == "fail"
        or precheck.get("answer_correctness_check") == "fail"
        or precheck.get("format_check") == "fail"
        or normalize_text(llm_payload.get("supervision_leak_check")).lower() == "fail"
        or normalize_text(llm_payload.get("answer_correctness_check")).lower() == "fail"
        or normalize_text(llm_payload.get("format_check")).lower() == "fail"
    )
    score = min(pre_score, llm_score) if hard_fail else round((pre_score + llm_score) / 2, 2)
    recommended_action = normalize_text(llm_payload.get("recommended_action"))
    if recommended_action not in {"keep_tool_trace", "rewrite_as_direct", "discard"}:
        recommended_action = "discard" if hard_fail else ("keep_tool_trace" if score >= 8 else "rewrite_as_direct")
    verdict = "pass" if (score >= 8 and not hard_fail) or recommended_action == "rewrite_as_direct" else "fail"
    return {
        "overall_verdict": verdict,
        "score": score,
        "source": "deterministic_and_llm",
        "recommended_action": recommended_action,
    }


def verify_path(path: Path, verifier: Optional[OpenAICompatibleVerifier]) -> Dict[str, Any]:
    record = load_record(path)
    precheck = deterministic_precheck(record)
    trace_text = build_trace_text(record)
    llm_result: Dict[str, Any] = {}
    status = "ok"
    if verifier is not None:
        try:
            llm_result = verifier.verify(record, trace_text, precheck)
        except Exception as exc:
            status = "verify_failed"
            llm_result = {"llm_verify_error": str(exc)}
    final = merge_verdict(precheck, llm_result if status == "ok" else None)
    return {
        "question_id": record.get("question_id"),
        "trace_path": str(path),
        "status": status,
        "original_status": record.get("status"),
        "original_correct": record.get("correct"),
        "question": record.get("question"),
        "gold_answer": record.get("gold_answer"),
        "pred_answer": record.get("pred_answer"),
        "tool_call_counter": record.get("tool_call_counter", {}),
        "tool_calls_count": record.get("tool_calls_count", 0),
        "precheck": precheck,
        **llm_result,
        "final_verdict": final,
    }


def read_done_ids(path: Path) -> set[str]:
    done: set[str] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except Exception:
                continue
            qid = normalize_text(row.get("question_id"))
            if qid:
                done.add(qid)
    return done


def summarize(rows: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    items = list(rows)
    status = Counter(row.get("status") for row in items)
    verdict = Counter(((row.get("final_verdict") or {}).get("overall_verdict")) for row in items)
    action = Counter(((row.get("final_verdict") or {}).get("recommended_action")) for row in items)
    llm_action = Counter(normalize_text((row.get("llm_verify_result") or {}).get("recommended_action")) for row in items)
    pre_verdict = Counter(((row.get("precheck") or {}).get("overall_verdict")) for row in items)
    failures = Counter()
    warnings = Counter()
    tool_counter = Counter()
    score_sum = 0.0
    low_sam = 0
    for row in items:
        score_sum += float(((row.get("final_verdict") or {}).get("score")) or 0)
        for reason in (row.get("precheck") or {}).get("failure_reasons", []) or []:
            failures[reason] += 1
        for warning in (row.get("precheck") or {}).get("warnings", []) or []:
            warnings[warning] += 1
        tool_counter.update(row.get("tool_call_counter") or {})
        low_sam += len((row.get("precheck") or {}).get("low_sam3_events", []) or [])
    return {
        "num_rows": len(items),
        "status": dict(status),
        "final_verdict": dict(verdict),
        "recommended_action": dict(action),
        "llm_recommended_action": dict(llm_action),
        "precheck_verdict": dict(pre_verdict),
        "average_final_score": round(score_sum / len(items), 4) if items else 0.0,
        "failure_reasons": dict(failures.most_common()),
        "warnings": dict(warnings.most_common()),
        "tool_counter": dict(tool_counter),
        "low_sam3_event_count": low_sam,
    }


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_jsonl = output_dir / "verify_v2_results.jsonl"
    output_summary = output_dir / "verify_v2_summary.json"

    paths = discover_trace_paths(input_path)
    if args.max_items > 0:
        paths = paths[: args.max_items]
    if args.resume:
        done = read_done_ids(output_jsonl)
        paths = [p for p in paths if normalize_text(load_record(p).get("question_id")) not in done]

    verifier = None
    if not args.skip_llm:
        verifier = OpenAICompatibleVerifier(
            base_url=args.llm_base_url,
            api_key=args.llm_api_key,
            model=args.llm_model,
            timeout=args.llm_timeout,
            max_tokens=args.llm_max_tokens,
            temperature=args.temperature,
            include_images=args.include_images,
            max_images=args.max_images,
        )

    started = time.time()
    results: List[Dict[str, Any]] = []
    mode = "a" if args.resume and output_jsonl.exists() else "w"
    with output_jsonl.open(mode, encoding="utf-8") as file:
        if args.num_workers <= 1:
            iterator = (verify_path(path, verifier) for path in paths)
            for row in tqdm(iterator, total=len(paths), desc="trace-verify-v2"):
                results.append(row)
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
                file.flush()
        else:
            with ThreadPoolExecutor(max_workers=args.num_workers) as pool:
                futures = [pool.submit(verify_path, path, verifier) for path in paths]
                for future in tqdm(as_completed(futures), total=len(futures), desc="trace-verify-v2"):
                    row = future.result()
                    results.append(row)
                    file.write(json.dumps(row, ensure_ascii=False) + "\n")
                    file.flush()

    all_rows: List[Dict[str, Any]] = []
    if output_jsonl.exists():
        with output_jsonl.open("r", encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    all_rows.append(json.loads(line))
    summary = summarize(all_rows)
    summary.update(
        {
            "input": str(input_path),
            "output_dir": str(output_dir),
            "skip_llm": args.skip_llm,
            "include_images": args.include_images,
            "max_images": args.max_images,
            "num_workers": args.num_workers,
            "elapsed_sec_this_run": round(time.time() - started, 2),
        }
    )
    output_summary.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
