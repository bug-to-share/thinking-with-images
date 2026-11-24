from __future__ import annotations

import base64
import json
import math
import re
from io import BytesIO
from typing import Any, Dict, List, Optional

from PIL import Image


IMAGE_FACTOR = 28
MIN_PIXELS = 4 * 28 * 28
MAX_PIXELS = 16384 * 28 * 28


def normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, list):
        return ' '.join(normalize_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


def ceil_by_factor(number: int, factor: int) -> int:
    return math.ceil(number / factor) * factor


def floor_by_factor(number: int, factor: int) -> int:
    return math.floor(number / factor) * factor


def round_by_factor(number: int, factor: int) -> int:
    return round(number / factor) * factor


def smart_resize(
    height: int,
    width: int,
    factor: int = IMAGE_FACTOR,
    min_pixels: int = MIN_PIXELS,
    max_pixels: int = MAX_PIXELS,
) -> tuple[int, int]:
    h_bar = max(factor, round_by_factor(height, factor))
    w_bar = max(factor, round_by_factor(width, factor))
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    return h_bar, w_bar


def maybe_resize_image_for_qwen(image: Image.Image) -> Image.Image:
    height, width = image.height, image.width
    new_height, new_width = smart_resize(height, width)
    if new_height == height and new_width == width:
        return image
    return image.resize((new_width, new_height), Image.BILINEAR)


def pil_to_base64(image: Image.Image, image_format: str = 'PNG') -> str:
    buffer = BytesIO()
    image.convert('RGB').save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def pil_to_data_url(image: Image.Image, image_format: str = 'PNG') -> str:
    image_base64 = pil_to_base64(image, image_format=image_format)
    return f'data:image/{image_format.lower()};base64,{image_base64}'


def base64_to_pil_image(base64_string: str) -> Image.Image:
    payload = base64_string.split(',', 1)[1] if ',' in base64_string else base64_string
    image_data = base64.b64decode(payload)
    return Image.open(BytesIO(image_data)).convert('RGB')


def path_to_data_url(image_path: str) -> str:
    with Image.open(image_path).convert('RGB') as image:
        image = maybe_resize_image_for_qwen(image)
        return pil_to_data_url(image)


def extract_tag_block(text: str, tag: str) -> Optional[str]:
    matches = re.findall(rf'<{tag}>(.*?)</{tag}>', text, re.DOTALL)
    if not matches:
        return None
    return matches[-1].strip()


def extract_python_code(text: str) -> Optional[str]:
    block = extract_tag_block(text, 'code')
    if not block:
        return None
    fenced = re.findall(r'```python\s*(.*?)```', block, re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    fenced = re.findall(r'```\s*(.*?)```', block, re.DOTALL)
    if fenced:
        return fenced[-1].strip()
    return block.strip()


def extract_tool_call(text: str) -> Optional[Dict[str, Any]]:
    block = extract_tag_block(text, 'tool_call')
    if not block:
        marker = '<tool_call>'
        start = text.find(marker)
        if start < 0:
            return None
        start += len(marker)
        tail = text[start:]
        next_tag = tail.find('<')
        block = tail[:next_tag].strip() if next_tag >= 0 else tail.strip()
        if not block:
            return None
    try:
        payload = json.loads(block)
    except Exception:
        payload = None

    def normalize_payload(obj: Any) -> Optional[Dict[str, Any]]:
        if not isinstance(obj, dict):
            return None
        if 'name' not in obj:
            return None
        if 'arguments' not in obj:
            arguments = {key: value for key, value in obj.items() if key not in {'name'}}
            obj = {'name': obj['name'], 'arguments': arguments}
        obj.setdefault('arguments', {})
        if obj['arguments'] is None:
            obj['arguments'] = {}
        return obj

    normalized = normalize_payload(payload)
    if normalized is not None:
        return normalized

    decoder = json.JSONDecoder()
    idx = 0
    pieces: List[Any] = []
    raw = block.strip()
    while idx < len(raw):
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
        if idx >= len(raw):
            break
        try:
            obj, end = decoder.raw_decode(raw, idx)
        except Exception:
            break
        pieces.append(obj)
        idx = end

    if len(pieces) >= 2 and isinstance(pieces[0], dict) and 'name' in pieces[0] and isinstance(pieces[1], dict):
        return {
            'name': pieces[0]['name'],
            'arguments': pieces[1],
        }
    if len(pieces) == 1:
        return normalize_payload(pieces[0])
    return None


def extract_answer_text(text: str) -> Optional[str]:
    block = extract_tag_block(text, 'answer')
    return normalize_text(block) if block else None


def parse_answer_letter(text: str) -> Optional[str]:
    answer = normalize_text(text).upper().replace('.', '')
    if re.fullmatch(r'[A-E]', answer):
        return answer
    match = re.search(r'\b([A-E])\b', answer)
    return match.group(1) if match else None


def answer_letter(vqa_item: Dict[str, Any]) -> str:
    return normalize_text(vqa_item.get('vqa_data', {}).get('answer')).upper().replace('.', '')


def answer_text(vqa_item: Dict[str, Any]) -> str:
    answer = answer_letter(vqa_item)
    options = vqa_item.get('vqa_data', {}).get('options', [])
    index = ord(answer) - ord('A')
    if 0 <= index < len(options):
        option = normalize_text(options[index])
        return re.sub(r'^[A-Z]\.[ ]*', '', option).strip()
    return answer


def option_text_by_letter(vqa_item: Dict[str, Any], letter: Optional[str]) -> str:
    if not letter:
        return ''
    options = vqa_item.get('vqa_data', {}).get('options', [])
    index = ord(letter) - ord('A')
    if 0 <= index < len(options):
        option = normalize_text(options[index])
        return re.sub(r'^[A-Z]\.[ ]*', '', option).strip()
    return ''


def build_vqa_prompt(vqa_item: Dict[str, Any]) -> str:
    question = normalize_text(vqa_item.get('vqa_data', {}).get('question'))
    options = [normalize_text(option) for option in vqa_item.get('vqa_data', {}).get('options', [])]
    option_block = '\n'.join(options)
    return (
        f'{question}\n\n'
        f'Choices:\n{option_block}\n\n'
        'Reason step by step. Use <tool_call>...</tool_call> when needed, including the `python_exec` tool for python image analysis. '
        'When finished, output only one final answer inside <answer>...</answer>.'
    )


def serialize_messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    serialized: List[Dict[str, Any]] = []
    for message in messages:
        row: Dict[str, Any] = {'role': message.get('role')}
        content = message.get('content')
        if isinstance(content, str):
            row['content'] = content
        elif isinstance(content, list):
            simplified = []
            for item in content:
                if item.get('type') == 'text':
                    simplified.append({'type': 'text', 'text': item.get('text', '')})
                elif item.get('type') == 'image_url':
                    url = item.get('image_url', {}).get('url', '')
                    simplified.append({'type': 'image_url', 'url': f'[data-url length={len(url)}]'})
            row['content'] = simplified
        else:
            row['content'] = normalize_text(content)
        serialized.append(row)
    return serialized
