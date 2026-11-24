from __future__ import annotations

import base64
import json
import mimetypes
from html import escape
from pathlib import Path
from typing import Any, Dict, List, Optional


def image_to_data_uri(path: Path) -> str:
    mime, _ = mimetypes.guess_type(str(path))
    mime = mime or 'image/png'
    encoded = base64.b64encode(path.read_bytes()).decode('utf-8')
    return f'data:{mime};base64,{encoded}'


def pretty_json(payload: Any) -> str:
    return json.dumps(payload, ensure_ascii=False, indent=2)


def source_item_for_record(record: Dict[str, Any]) -> Dict[str, Any]:
    jsonl_path = record.get('jsonl_path')
    question_id = str(record.get('question_id') or '')
    if not jsonl_path or not question_id:
        return {}
    path = Path(str(jsonl_path))
    if not path.exists():
        return {}
    try:
        with path.open('r', encoding='utf-8') as file:
            for line in file:
                if not line.strip():
                    continue
                row = json.loads(line)
                if str(row.get('question_id') or '') == question_id:
                    return row
    except Exception:
        return {}
    return {}


def choices_for_record(record: Dict[str, Any]) -> List[str]:
    choices = record.get('choices')
    if not choices:
        choices = source_item_for_record(record).get('choices')
    rows: List[str] = []
    if isinstance(choices, list):
        for choice in choices:
            if isinstance(choice, dict):
                letter = str(choice.get('letter') or '').strip()
                text = str(choice.get('text') or '').strip()
                if text:
                    rows.append(f'{letter}. {text}' if letter else text)
            else:
                text = str(choice or '').strip()
                if text:
                    rows.append(text)
    return rows


def gold_letter_for_record(record: Dict[str, Any]) -> str:
    value = str(record.get('gold_letter') or '').strip()
    if value:
        return value
    return str(source_item_for_record(record).get('gold_letter') or '').strip()


def render_image_card(title: str, image_path: Optional[Path]) -> str:
    if image_path is None or not image_path.exists():
        return ''
    return f"""
    <div class="image-card">
      <div class="image-title">{escape(title)}</div>
      <img src="{image_to_data_uri(image_path)}" alt="{escape(title)}" />
      <div class="image-path">{escape(str(image_path))}</div>
    </div>
    """


def inventory_map(record: Dict[str, Any]) -> Dict[int, Path]:
    mapping: Dict[int, Path] = {}
    for item in record.get('image_inventory', []) or []:
        index = item.get('image_index')
        path = item.get('path')
        if index is None or not path:
            continue
        mapping[int(index)] = Path(path)
    return mapping


def event_attachments(event: Dict[str, Any], inventory: Dict[int, Path]) -> List[Path]:
    attachments: List[Path] = []
    for index in event.get('new_indices', []) or []:
        path = inventory.get(int(index))
        if path is not None and path.exists():
            attachments.append(path)
    return attachments


def render_trace_html(record: Dict[str, Any], output_path: Path) -> None:
    question_id = str(record.get('question_id', 'unknown'))
    inventory = inventory_map(record)
    original_image_path = Path(record['image_path'])
    choices = choices_for_record(record)
    choices_html = '<br />'.join(escape(choice) for choice in choices) if choices else 'N/A'
    gold_letter = gold_letter_for_record(record)

    gallery_parts: List[str] = []
    if original_image_path.exists():
        gallery_parts.append(render_image_card('Original Image (image_index=0)', original_image_path))
    for image_index in sorted(k for k in inventory.keys() if k != 0):
        gallery_parts.append(render_image_card(f'image_index={image_index}', inventory[image_index]))

    timeline_parts: List[str] = []
    for event in record.get('event_trace', []) or []:
        event_type = str(event.get('type', 'unknown'))
        round_idx = event.get('round')
        header = f'Round {round_idx} · {event_type}'
        if event.get('name'):
            header += f' · {event["name"]}'

        visible_parts: List[str] = []
        if event.get('reason'):
            visible_parts.append(f'<div class="block-title">Reason</div><pre>{escape(str(event["reason"]))}</pre>')
        if event.get('response_text'):
            visible_parts.append(f'<div class="block-title">Response</div><pre>{escape(str(event["response_text"]))}</pre>')
        if event_type == 'tool':
            payload = {
                'name': event.get('name'),
                'arguments': event.get('arguments'),
                'normalized_arguments': event.get('normalized_arguments'),
                'coordinate_matches': event.get('coordinate_matches'),
                'result': event.get('result'),
            }
            visible_parts.append(f'<div class="block-title">Tool Payload</div><pre>{escape(pretty_json(payload))}</pre>')
        elif event_type == 'tool_error':
            payload = {
                'name': event.get('name'),
                'arguments': event.get('arguments'),
                'normalized_arguments': event.get('normalized_arguments'),
                'coordinate_matches': event.get('coordinate_matches'),
                'error': event.get('error'),
            }
            visible_parts.append(f'<div class="block-title">Tool Error</div><pre>{escape(pretty_json(payload))}</pre>')
        elif event_type == 'answer':
            visible_parts.append(f'<div class="block-title">Answer</div><pre>{escape(str(event.get("answer", "")))}</pre>')

        attachments_html = ''
        attachments = event_attachments(event, inventory)
        if attachments:
            attachments_html = '<div class="attachment-grid">' + ''.join(
                render_image_card(path.name, path) for path in attachments
            ) + '</div>'

        timeline_parts.append(
            f"""
            <div class="step {escape(event_type)}">
              <div class="step-header">{escape(header)}</div>
              {''.join(visible_parts)}
              {attachments_html}
            </div>
            """
        )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>Gemini Agent Trace - {escape(question_id)}</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #111827;
      background: #f8fafc;
    }}
    .section {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 18px;
      margin-bottom: 20px;
    }}
    .meta-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
      gap: 12px;
    }}
    .kv {{
      background: #f3f4f6;
      border-radius: 8px;
      padding: 10px 12px;
    }}
    .gallery, .attachment-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(260px, 1fr));
      gap: 14px;
    }}
    .image-card {{
      border: 1px solid #d1d5db;
      border-radius: 10px;
      overflow: hidden;
      background: #fff;
    }}
    .image-title, .image-path {{
      padding: 8px 10px;
      font-size: 12px;
    }}
    .image-title {{
      font-weight: 700;
      background: #eff6ff;
    }}
    .image-path {{
      color: #4b5563;
      border-top: 1px solid #e5e7eb;
      word-break: break-all;
    }}
    img {{
      display: block;
      width: 100%;
      height: auto;
      background: #111827;
    }}
    .timeline {{
      display: flex;
      flex-direction: column;
      gap: 14px;
    }}
    .step {{
      background: white;
      border: 1px solid #e5e7eb;
      border-left: 6px solid #94a3b8;
      border-radius: 10px;
      padding: 14px;
    }}
    .step.tool {{ border-left-color: #059669; }}
    .step.answer {{ border-left-color: #2563eb; }}
    .step.tool_error, .step.format_error {{ border-left-color: #dc2626; }}
    .step-header {{
      font-weight: 700;
      margin-bottom: 12px;
    }}
    .block-title {{
      font-size: 12px;
      font-weight: 700;
      text-transform: uppercase;
      color: #475569;
      margin: 10px 0 6px 0;
    }}
    pre {{
      white-space: pre-wrap;
      word-break: break-word;
      background: #0f172a;
      color: #e5e7eb;
      border-radius: 8px;
      padding: 12px;
      overflow-x: auto;
    }}
  </style>
</head>
<body>
  <div class="section">
    <h1>Trace Visualization · {escape(question_id)}</h1>
    <div class="meta-grid">
      <div class="kv"><strong>Question</strong><br />{escape(record.get('question', ''))}</div>
      <div class="kv"><strong>Choices</strong><br />{choices_html}</div>
      <div class="kv"><strong>Gold Letter</strong><br />{escape(gold_letter or 'N/A')}</div>
      <div class="kv"><strong>Model Image</strong><br />{escape(str(record.get('model_image_path', '')))}</div>
      <div class="kv"><strong>Gold</strong><br />{escape(str(record.get('gold_answer', '')))}</div>
      <div class="kv"><strong>Pred</strong><br />{escape(str(record.get('pred_answer', '')))}</div>
      <div class="kv"><strong>Correctness Method</strong><br />{escape(str(record.get('correctness_method', '')))}</div>
      <div class="kv"><strong>Status</strong><br />{escape(str(record.get('status', '')))}</div>
      <div class="kv"><strong>Correct</strong><br />{escape(str(record.get('correct', False)))}</div>
      <div class="kv"><strong>Tool Calls</strong><br />{escape(pretty_json(record.get('tool_call_counter', {})))}</div>
      <div class="kv"><strong>Canvas</strong><br />{escape(pretty_json(record.get('model_canvas', {})))}</div>
    </div>
  </div>

  <div class="section">
    <h2>Image Gallery</h2>
    <div class="gallery">
      {''.join(gallery_parts)}
    </div>
  </div>

  <div class="section">
    <h2>Timeline</h2>
    <div class="timeline">
      {''.join(timeline_parts)}
    </div>
  </div>
</body>
</html>
"""
    output_path.write_text(html, encoding='utf-8')
