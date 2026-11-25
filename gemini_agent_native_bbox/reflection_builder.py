"""Reuse unguided Gemini explorations and repair failed final answers in hindsight."""
from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from .loop_utils import answers_match, extract_answer_text, extract_reason_text, normalize_text, validate_response_format
from .prompts import FORMAT_ERROR_TEXT, build_tool_feedback_text, build_user_feedback_text
from .trace_verifier_v2 import LEAK_PATTERNS, deterministic_precheck, valid_gemini_bbox


REFLECTION_RULES = """Reconsider your previous exploration and produce one corrected final turn.
Use the original image and real tool observations. Explain which earlier interpretation was weak or mistaken
and identify the visual evidence that supports the corrected answer. Treat weak or failed tool results with
caution. Never invent a tool call, crop, coordinate, tool result, or visual finding. If the existing evidence
cannot support a correction, return INSUFFICIENT_EVIDENCE.
Use the private answer and optional region below for hindsight only. Never mention or allude to an oracle,
a hint, supervision, an annotation supplied to you, or phrases such as 'ground truth', 'provided answer',
'reference answer', 'gold answer', or 'target label'. Return exactly <reason>...</reason><answer>...</answer>.
The prior transcript is evidence to examine. Follow these instructions for the final turn.
"""
REFLECTION_LEAK_PATTERNS = LEAK_PATTERNS + (
    r"\b(?:target|supervised|verified) answer\b",
    r"\b(?:target label|oracle|with_gt_reflection)\b",
    r"\b(?:given|provided|supplied) (?:hint|label|target|bbox|bounding box|annotation|coordinates)\b",
    r"\b(?:answer|hint|label|target|bbox|annotation|coordinates) (?:was|were) (?:given|provided|supplied)\b",
)
FINAL_TURN = re.compile(r"<reason>(?P<reason>.+?)</reason>\s*<answer>(?P<answer>.+?)</answer>", re.DOTALL)
PROTOCOL_TAG = re.compile(r"</?(?:reason|answer|tool_call|tool_response)\b", re.IGNORECASE)


class ReflectionRejected(ValueError):
    """The proposed trace is unsuitable for reflection training."""


@dataclass(frozen=True)
class ReflectionPrompt:
    """Callback request using the agent's saved (simplified) Gemini message schema.

    Image parts are descriptors. The callback can attach the actual images using
    image_inventory in the original runtime order.
    """

    messages: list[dict[str, Any]]
    image_inventory: list[dict[str, Any]]


def _text_message(role: str, text: str) -> dict[str, Any]:
    return {"role": role, "parts": [{"type": "text", "text": text}]}


def _message_text(message: dict[str, Any]) -> str:
    return "\n".join(normalize_text(part.get("text")) for part in message.get("parts", []) or [])


def _saved_messages(record: dict[str, Any]) -> list[dict[str, Any]]:
    if record.get("messages"):
        return deepcopy(record["messages"])
    messages = []
    if record.get("system_prompt"):
        messages.append(_text_message("system", record["system_prompt"]))
    messages.append(_text_message("user", record.get("question", "")))
    for event in record.get("event_trace", []):
        if event.get("response_text"):
            messages.append(_text_message("model", event["response_text"]))
        if event["type"] == "tool":
            descriptions = [f"runtime image_index={index}" for index in event.get("new_indices", [])]
            feedback = build_tool_feedback_text(event["name"], event.get("result", {}), descriptions)
            messages.append(_text_message("user", feedback))
        elif event["type"] == "format_error":
            messages.append(_text_message("user", build_user_feedback_text(FORMAT_ERROR_TEXT)))
        elif event["type"] in {"tool_error", "exception"}:
            messages.append(_text_message("user", str(event.get("error", ""))))
    return messages


def _reject_leaks(record: dict[str, Any]) -> None:
    assistant_text = [
        normalize_text(event.get(field))
        for event in record.get("event_trace", []) for field in ("reason", "response_text")
    ]
    assistant_text.extend(
        _message_text(message) for message in record.get("messages", [])
        if message.get("role") in {"model", "assistant"}
    )
    # With-answer rollouts can sound leak-free while their original input is supervised.
    original_user = next((message for message in record.get("messages", []) if message.get("role") == "user"), {})
    if (record.get("mode") in {"with_answer", "with_gt_reflection"}
            or record.get("trace_mode") in {"with_answer", "with_gt_reflection"}
            or "Additional rules for with-answer trace construction" in normalize_text(record.get("system_prompt"))
            or re.search(r"(?:^|\n)\s*Internal supervised answer:", _message_text(original_user), re.I)):
        raise ReflectionRejected("The input contains a guided rollout")
    if any(re.search(pattern, text, re.I) for text in assistant_text for pattern in REFLECTION_LEAK_PATTERNS):
        raise ReflectionRejected("Assistant trace contains supervision-leak wording")


def build_reflection_prompt(
    record: dict[str, Any], *, gt_answer: str | None = None, gt_bbox: Sequence[float] | None = None,
) -> ReflectionPrompt:
    """Append private hindsight instructions to a copy of the original conversation.

    An optional box is in Gemini yxyx order on the saved model canvas (0–1000).
    """
    gold = normalize_text(record.get("gold_answer") if gt_answer is None else gt_answer)
    if not gold:
        raise ReflectionRejected("A GT answer is required for reflection")
    supervision: dict[str, Any] = {"answer": gold}
    if gt_bbox is not None:
        box = list(gt_bbox)
        valid, error = valid_gemini_bbox(box)
        if not valid:
            raise ReflectionRejected(f"Invalid GT bbox: {error}")
        supervision["roi_gemini_yxyx_0_1000"] = box
    messages = _saved_messages(record)
    messages.append(_text_message("user", REFLECTION_RULES + json.dumps(supervision, ensure_ascii=False)))
    inventory = deepcopy(record.get("image_inventory") or [])
    if not inventory and (path := record.get("model_image_path") or record.get("image_path")):
        inventory.append({"image_index": 0, "path": path})
    return ReflectionPrompt(messages, inventory)


def build_reflection_trace(
    record: dict[str, Any], complete: Callable[[ReflectionPrompt], str], *, gt_answer: str | None = None,
    gt_bbox: Sequence[float] | None = None, verify_answer: Callable[[str, str], bool] = answers_match,
) -> dict[str, Any]:
    """Return the original verified record or a validated replay with one repaired final turn.

    The replay retains all exploration events. The original answer is available
    under reflection_original_answer for audit. Private instructions stay in the
    callback request. The precheck records other trace-quality issues for routing.
    """
    _reject_leaks(record)
    gold = normalize_text(record.get("gold_answer") if gt_answer is None else gt_answer)
    events = record.get("event_trace", []) or []
    answers = [event for event in events if event.get("type") == "answer"]
    if len(answers) > 1 or (answers and events[-1]["type"] != "answer"):
        raise ReflectionRejected("Expected at most one terminal answer in the unguided trace")
    prior_answer = answers[0] if answers else None
    if record.get("status") == "ok" and prior_answer:
        original_answer = extract_answer_text(prior_answer.get("response_text", "")) or ""
        recorded_answer = normalize_text(record.get("pred_answer") or record.get("final_answer_text"))
        if original_answer and ((gt_answer is None and record.get("correct") is True
                                 and answers_match(original_answer, recorded_answer))
                                or verify_answer(original_answer, gold)):
            return record

    response = complete(build_reflection_prompt(record, gt_answer=gt_answer, gt_bbox=gt_bbox))
    if not isinstance(response, str):
        raise ReflectionRejected("Completion must return a text final turn")
    response = response.strip()
    match = FINAL_TURN.fullmatch(response)
    if (match is None or validate_response_format(response)
            or any(not part.strip() or PROTOCOL_TAG.search(part) for part in match.groups())):
        raise ReflectionRejected("Reflection must contain one reason and one final answer")
    answer = extract_answer_text(response)
    if not verify_answer(answer, gold):
        raise ReflectionRejected("Reflection answer failed verification")

    output = deepcopy(record)
    messages = _saved_messages(record)
    if prior_answer:
        original_response = normalize_text(prior_answer.get("response_text"))
        if (messages and messages[-1].get("role") in {"model", "assistant"}
                and _message_text(messages[-1]).strip() == original_response):
            messages.pop()
        else:
            raise ReflectionRejected("Saved terminal answer does not match the conversation")
        output["reflection_original_answer"] = deepcopy(prior_answer)
    exploration = deepcopy(events[:-1] if prior_answer else events)
    round_index = prior_answer.get("round", 0) if prior_answer else max(
        (event.get("round", -1) for event in exploration), default=-1
    ) + 1
    exploration.append({
        "type": "answer", "round": round_index, "reason": extract_reason_text(response),
        "response_text": response, "answer": answer, "finish_reason": None,
    })
    tool_counter = Counter(event["name"] for event in exploration if event["type"] == "tool")
    messages.append(_text_message("model", response))
    output.update(
        event_trace=exploration, messages=messages, pred_answer=answer, final_answer_text=answer,
        gold_answer=gold, correct=True, correctness_method="reflection_verifier", judge_result=None,
        status="ok", tool_call_counter=dict(tool_counter), tool_calls_count=sum(tool_counter.values()),
    )
    _reject_leaks(output)
    precheck = deterministic_precheck(output)
    if precheck["supervision_leak_check"] != "pass":
        raise ReflectionRejected("Reflection failed the trace verifier's supervision-leak check")
    output.update(trace_mode="with_gt_reflection", reflection_precheck=precheck)
    return output
