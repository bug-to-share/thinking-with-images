from __future__ import annotations

import base64
import ast
import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Dict, List, Optional


def normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, list):
        return ' '.join(normalize_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def collapse_spaces(text: str) -> str:
    return re.sub(r'\s+', ' ', normalize_text(text)).strip()


def canonical_answer(text: Any) -> str:
    value = collapse_spaces(text).lower()
    value = re.sub(r'^[a-z]\s*[\.\)\:]\s*', '', value)
    value = re.sub(r'^[\W_]+|[\W_]+$', '', value)
    value = re.sub(r'^(a|an|the)\s+', '', value)
    return value


def answers_match(prediction: Any, gold: Any) -> bool:
    pred = canonical_answer(prediction)
    ref = canonical_answer(gold)
    return bool(pred) and pred == ref


def build_question_text(item: Dict[str, Any]) -> str:
    question = normalize_text(item.get('question'))
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
    answer_instruction = (
        'When you are ready, output the final free-text answer inside <answer>...</answer>.'
    )
    if choice_lines:
        choice_block = 'Choices:\n' + '\n'.join(f'- {line}' for line in choice_lines) + '\n\n'
        answer_instruction = (
            'This is a multiple-choice question. When you are ready, output the selected option text exactly inside '
            '<answer>...</answer>. Do not answer with only the option letter. Do not include the option letter prefix.'
        )
    return (
        f'Question: {question}\n\n'
        f'{choice_block}'
        f'You may use tools if needed. {answer_instruction}'
    )


def extract_tag_block(text: str, tag: str) -> Optional[str]:
    matches = re.findall(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    if not matches:
        return None
    return matches[0].strip()


def extract_first_tag_block(text: str, tag: str) -> Optional[str]:
    matches = re.findall(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    if not matches:
        return None
    return matches[0].strip()


def extract_reason_text(text: str) -> Optional[str]:
    return extract_tag_block(text, 'reason')


def extract_answer_text(text: str) -> Optional[str]:
    return extract_tag_block(text, 'answer')


def strip_trailing_action_garbage(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return text
    pattern = re.compile(
        r'^\s*(<reason>.*?</reason>\s*(?:<tool_call>.*?</tool_call>|<answer>.*?</answer>))',
        re.DOTALL,
    )
    match = pattern.match(text)
    if not match:
        return text
    return match.group(1).strip()


def extract_tool_call(
    text: str,
) -> Optional[Dict[str, Any]]:
    block = extract_first_tag_block(text, 'tool_call')
    if not block:
        return None
    payload = _parse_json_or_literal(block)
    if not isinstance(payload, dict):
        return None
    name = normalize_text(payload.get('name'))
    payload = normalize_tool_call_payload(payload)
    if 'arguments' not in payload:
        return None
    arguments = payload.get('arguments')
    if not name:
        return None
    if not isinstance(arguments, dict):
        return None
    return {'name': name, 'arguments': arguments}


def normalize_tool_call_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize narrowly accepted tool-call variants.

    This intentionally only fixes the common python_exec shorthand:
    {"name": "python_exec", "code": "..."} -> {"name": "python_exec", "arguments": {"code": "..."}}
    Other malformed tools still fail format validation.
    """
    name = normalize_text(payload.get('name'))
    if name == 'python_exec' and 'arguments' not in payload and 'code' in payload:
        normalized = dict(payload)
        arguments = {'code': payload.get('code')}
        if 'timeout' in payload:
            arguments['timeout'] = payload.get('timeout')
        normalized['arguments'] = arguments
        return normalized
    return payload


def count_blocks(text: str, tag: str) -> int:
    return len(re.findall(rf'<{tag}>.*?</{tag}>', text, re.DOTALL))


def validate_response_format(text: str) -> Optional[str]:
    text = strip_trailing_action_garbage(text)
    if extract_reason_text(text) is None:
        return 'missing <reason>...</reason>'
    if '<tool_response' in normalize_text(text):
        return 'assistant output must not contain <tool_response>'
    tool_count = count_blocks(text, 'tool_call')
    answer_count = count_blocks(text, 'answer')
    if tool_count == 0 and answer_count == 1:
        pass
    elif tool_count >= 1 and answer_count == 0:
        pass
    else:
        return 'must contain exactly one of <tool_call> or <answer>'
    pattern = re.compile(r'^\s*<reason>.*?</reason>\s*(<tool_call>.*?</tool_call>|<answer>.*?</answer>)\s*$', re.DOTALL)
    if not pattern.match(text):
        return 'response must end immediately after one <tool_call> or one <answer>'
    return None


def file_to_inline_data_part(image_path: str) -> Dict[str, Any]:
    path = Path(image_path)
    mime_type, _ = mimetypes.guess_type(str(path))
    mime_type = mime_type or 'image/png'
    data = base64.b64encode(path.read_bytes()).decode('utf-8')
    return {
        'inlineData': {
            'mimeType': mime_type,
            'data': data,
        }
    }


def simplify_contents(contents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for message in contents:
        row: Dict[str, Any] = {'role': message.get('role')}
        parts = []
        for part in message.get('parts', []) or []:
            if 'text' in part:
                parts.append({'type': 'text', 'text': part.get('text', '')})
            elif 'inlineData' in part:
                inline = part.get('inlineData') or {}
                mime = inline.get('mimeType', '')
                size = len(normalize_text(inline.get('data')))
                parts.append({'type': 'image', 'mime_type': mime, 'base64_length': size})
        row['parts'] = parts
        rows.append(row)
    return rows


def safe_trace_stem(question_id: Any) -> str:
    raw = normalize_text(question_id) or 'unknown'
    safe = re.sub(r'[^0-9A-Za-z._-]+', '_', raw)
    return f'trace_{safe}'


def _parse_json_or_literal(text: str) -> Any:
    try:
        return json.loads(text)
    except Exception:
        try:
            return ast.literal_eval(text)
        except Exception:
            return None
