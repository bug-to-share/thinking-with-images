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


def extract_strict_answer(solution_str: str) -> tuple[bool, str]:
    text = as_text(solution_str)
    normalized_tags = text.casefold()
    required_tags = ("<reason>", "</reason>", "<answer>", "</answer>")
    if any(normalized_tags.count(tag) != 1 for tag in required_tags):
        return False, ""

    match = FORMAT_PATTERN.fullmatch(text)
    if match is None:
        return False, ""
    answer_matches = ANSWER_PATTERN.findall(text)
    return True, answer_matches[-1].strip()


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    del data_source, extra_info, kwargs

    format_valid, prediction = extract_strict_answer(solution_str)
    raw_exact_match = float(
        bool(format_valid and normalize_exact(prediction))
        and normalize_exact(prediction) == normalize_exact(ground_truth)
    )
    format_reward = float(format_valid)
    exact_match = format_reward * raw_exact_match
    score = format_reward * (0.3 + 0.7 * raw_exact_match)

    return {
        "score": float(score),
        "exact_match": exact_match,
        "format_reward": format_reward,
    }
