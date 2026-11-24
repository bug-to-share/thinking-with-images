from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List

from gemini_bbox_format import load_image_size, xyxy_to_gemini_box
from loop_utils import normalize_text
from model_canvas import map_original_xyxy_to_canvas_xyxy
from prompts import build_system_prompt as build_base_system_prompt


def build_system_prompt(tool_specs: List[Dict[str, Any]]) -> str:
    base = build_base_system_prompt(tool_specs)
    appendix = """
Additional rules for with-answer trace construction:
- The user message may contain hidden supervision fields for data construction. Treat them as internal guidance only.
- Your final `<answer>...</answer>` must be semantically consistent with the supervised target. Exact wording is not required.
- Never mention that any answer, label, hint, supervision, target, or guidance was provided.
- Never mention phrases such as `reference answer`, `provided answer`, `gold answer`, `ground truth`, `standard answer`, `target answer`, `verified target`, `internal supervised answer`, `supervised answer`, `target label`, or `verified answer`.
- Never say that you are trusting, following, copying, or using a provided answer or hint.
- Use tools only to build a faithful evidence trace supporting the correct answer.
- Do not fabricate tool outputs, observations, coordinates, crop regions, or intermediate evidence.
- Do not force an unreliable tool result to support the answer. If a tool result is irrelevant, low-confidence, or anatomically inconsistent, say so and rely on better evidence or retry.
- In a tool-use turn, only explain why you need the tool now. Do not claim what it will prove before it runs.
- Only answer after the supporting evidence has appeared in prior tool responses or direct visual inspection.
- Before the final answer, explain the key visual finding, the medical/anatomical knowledge that makes it relevant, and why at least one plausible alternative is less likely when applicable.
- If your draft reasoning contains any supervision-related phrase, discard it and rewrite the turn before responding.
""".strip()
    return f"{base}\n\n{appendix}"


def build_question_text_with_answer(item: Dict[str, Any]) -> str:
    question = normalize_text(item.get('question'))
    gold_answer = normalize_text(item.get('gt_answer'))
    choices = item.get('choices')
    choice_lines: List[str] = []
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                letter = normalize_text(choice.get('letter'))
                text = normalize_text(choice.get('text'))
                if text:
                    prefix = f'{letter}. ' if letter else ''
                    choice_lines.append(f'{prefix}{text}')
            else:
                text = normalize_text(choice)
                if text:
                    choice_lines.append(text)
    choice_block = ''
    final_instruction = (
        'When you are ready, output a final answer inside <answer>...</answer> that is semantically consistent with the internal supervised answer. '
        'Exact wording is not required. Now start your reasoning.'
    )
    if choice_lines:
        choice_block = 'Choices:\n' + '\n'.join(f'- {line}' for line in choice_lines) + '\n\n'
        final_instruction = (
            'When you are ready, output the selected option text inside <answer>...</answer>, not only the option letter. '
            'Do not include the option letter prefix. '
            'The selected option must be semantically consistent with the internal supervised answer. Now start your reasoning.'
        )
    reference_lines: List[str] = []
    image_path = normalize_text(item.get('_model_image_path_abs') or item.get('image_path_abs'))
    image_size = load_image_size(image_path) if image_path else None
    canvas_info = item.get('_model_canvas_info')

    modality = normalize_text(item.get('modality'))
    if modality:
        reference_lines.append(f'- Reference modality: {modality}')

    location = normalize_text(item.get('location'))
    if location:
        reference_lines.append(f'- Reference body part / location: {location}')

    content_type = normalize_text(item.get('content_type'))
    if content_type:
        reference_lines.append(f'- Reference question type: {content_type}')

    detections = _extract_reference_detections(item)
    for label, bbox in detections:
        if isinstance(canvas_info, dict):
            bbox = map_original_xyxy_to_canvas_xyxy(bbox, canvas_info)
        if image_size is not None:
            model_bbox = xyxy_to_gemini_box(bbox, image_size)
            reference_lines.append(f'- Reference detection: {label} {model_bbox} (Gemini bbox: [ymin, xmin, ymax, xmax])')
        else:
            reference_lines.append(f'- Reference detection: {label} {bbox}')

    reference_block = ''
    if reference_lines:
        reference_block = 'Auxiliary context for internal use only:\n' + '\n'.join(reference_lines) + '\n\n'

    return (
        f'Question: {question}\n\n'
        f'{choice_block}'
        f'Internal supervised answer: {gold_answer}\n\n'
        f'{reference_block}'
        'The internal supervised answer above is correct. Any auxiliary context above is also correct and may guide your internal tool choice. '
        'Do not mention that this supervision exists. Do not quote or refer to the target or auxiliary context explicitly. '
        'Use tools only to construct a faithful evidence trail that independently supports the same conclusion. '
        f'{final_instruction}'
    )


def _extract_reference_detections(item: Dict[str, Any]) -> List[tuple[str, List[int]]]:
    direct_bbox = _extract_direct_bbox(item)
    if direct_bbox is not None:
        label = normalize_text(item.get('target_name')) or 'reference_target'
        return [(label, direct_bbox)]

    rows = _lookup_all_slake_detections(item)
    detections: List[tuple[str, List[int]]] = []
    for label, xywh in rows:
        x, y, w, h = xywh
        bbox = [int(round(x)), int(round(y)), int(round(x + w)), int(round(y + h))]
        detections.append((label, bbox))
    return detections


def _extract_direct_bbox(item: Dict[str, Any]) -> List[int] | None:
    for key in ('gt_bbox', 'reference_bbox', 'bbox', 'roi_bbox'):
        value = item.get(key)
        if isinstance(value, (list, tuple)) and len(value) == 4:
            try:
                return [int(round(float(v))) for v in value]
            except Exception:
                continue
    return None


def _lookup_all_slake_detections(item: Dict[str, Any]) -> List[tuple[str, List[float]]]:
    detection_path = _resolve_slake_detection_path(item)
    if detection_path is None:
        return []
    return _load_slake_detection(detection_path)


def _resolve_slake_detection_path(item: Dict[str, Any]) -> Path | None:
    image_path_abs = normalize_text(item.get('image_path_abs'))
    if image_path_abs:
        image_path = Path(image_path_abs)
        candidate = image_path.parent / 'detection.json'
        if candidate.exists():
            return candidate

    image_path_rel = normalize_text(item.get('image_path_rel'))
    if image_path_rel:
        candidate = Path('./dataset') / image_path_rel
        candidate = candidate.parent / 'detection.json'
        if candidate.exists():
            return candidate
    return None


@lru_cache(maxsize=2048)
def _load_slake_detection(path: Path) -> List[tuple[str, List[float]]]:
    try:
        raw = json.loads(path.read_text())
    except Exception:
        return []
    rows: List[tuple[str, List[float]]] = []
    if not isinstance(raw, list):
        return rows
    for item in raw:
        if not isinstance(item, dict):
            continue
        for label, value in item.items():
            if isinstance(value, list) and len(value) == 4:
                try:
                    rows.append((str(label), [float(v) for v in value]))
                except Exception:
                    continue
    return rows
