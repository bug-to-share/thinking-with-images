"""Baseline independent format and exact-match reward."""

from __future__ import annotations

import json
import re
import unicodedata
from typing import Any


ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
FORMAT_PATTERN = re.compile(
    r"^\s*<reason>\s*(\S(?:.*?\S)?)\s*</reason>\s*"
    r"<answer>\s*(\S(?:.*?\S)?)\s*</answer>\s*$",
    re.DOTALL | re.IGNORECASE,
)
EDGE_PUNCTUATION = " \t\r\n.,!?;:，。！？；：、\"'“”‘’()[]{}"


def as_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        for key in ("text", "answer", "ground_truth", "letter"):
            if key in value:
                return as_text(value[key])
        return json.dumps(value, ensure_ascii=False, sort_keys=True)
    if isinstance(value, list):
        return " ".join(as_text(item) for item in value)
    return str(value)


def normalize_exact(value: Any) -> str:
    text = unicodedata.normalize("NFKC", as_text(value)).casefold()
    text = re.sub(r"\s+", " ", text).strip(EDGE_PUNCTUATION)
    return text


def strict_format_check(solution_str: str) -> bool:
    return FORMAT_PATTERN.fullmatch(as_text(solution_str)) is not None


def extract_answer(solution_str: str) -> str:
    text = as_text(solution_str)
    matches = ANSWER_PATTERN.findall(text)
    # A bare answer can still earn exact-match credit; format is scored separately.
    return matches[-1].strip() if matches else text.strip()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    del data_source, extra_info, kwargs

    prediction = extract_answer(solution_str)
    prediction_normalized = normalize_exact(prediction)
    target_normalized = normalize_exact(ground_truth)
    exact_match = float(bool(prediction_normalized) and prediction_normalized == target_normalized)
    format_reward = float(strict_format_check(solution_str))
    score = 0.7 * exact_match + 0.3 * format_reward

    return {
        "score": float(score),
        "exact_match": exact_match,
        "format_reward": format_reward,
    }
