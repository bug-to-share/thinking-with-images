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
    question_id = int(record['question_id'])
    inventory = inventory_map(record)
    original_image_path = Path(record['image_path'])

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
        if event.get('reasoning'):
            visible_parts.append(f'<div class="block-title">Reasoning</div><pre>{escape(str(event["reasoning"]))}</pre>')
        if event.get('content'):
            visible_parts.append(f'<div class="block-title">Visible Content</div><pre>{escape(str(event["content"]))}</pre>')
        if event.get('action_text') and event.get('action_text') != event.get('content'):
            visible_parts.append(f'<div class="block-title">Parsed Action</div><pre>{escape(str(event["action_text"]))}</pre>')
        if event_type == 'python':
            visible_parts.append(f'<div class="block-title">Code</div><pre>{escape(str(event.get("code", "")))}</pre>')
            visible_parts.append(f'<div class="block-title">stdout</div><pre>{escape(str(event.get("stdout", "")))}</pre>')
            visible_parts.append(f'<div class="block-title">stderr</div><pre>{escape(str(event.get("stderr", "")))}</pre>')
        elif event_type == 'tool':
            payload = {
                'name': event.get('name'),
                'arguments': event.get('arguments'),
                'result': event.get('result'),
            }
            visible_parts.append(f'<div class="block-title">Tool Payload</div><pre>{escape(pretty_json(payload))}</pre>')
        elif event_type == 'tool_error':
            payload = {
                'name': event.get('name'),
                'arguments': event.get('arguments'),
                'error': event.get('error'),
            }
            visible_parts.append(f'<div class="block-title">Tool Error</div><pre>{escape(pretty_json(payload))}</pre>')

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
  <title>Qwen Agent Trace - q_{question_id:06d}</title>
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
    h1, h2 {{
      margin: 0 0 12px 0;
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
    .step.python {{ border-left-color: #7c3aed; }}
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
    <h1>Trace Visualization · q_{question_id:06d}</h1>
    <div class="meta-grid">
      <div class="kv"><strong>Question</strong><br />{escape(record.get('question', ''))}</div>
      <div class="kv"><strong>Gold</strong><br />{escape(str(record.get('gold_letter', '')))} · {escape(record.get('gold_text', ''))}</div>
      <div class="kv"><strong>Pred</strong><br />{escape(str(record.get('pred_letter', '')))} · {escape(record.get('pred_text', ''))}</div>
      <div class="kv"><strong>Status</strong><br />{escape(str(record.get('status', '')))}</div>
      <div class="kv"><strong>Tool Calls</strong><br />{escape(pretty_json(record.get('tool_call_counter', {})))}</div>
      <div class="kv"><strong>Reasoning Backend</strong><br />{escape(str(record.get('reasoning_backend', '')))}</div>
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
