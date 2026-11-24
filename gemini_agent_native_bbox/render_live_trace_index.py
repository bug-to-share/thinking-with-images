from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from datetime import datetime
from html import escape
from pathlib import Path
from typing import Any, Dict, List

from loop_utils import safe_trace_stem


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def count_events(rows: List[Dict[str, Any]], event_type: str) -> int:
    return sum(1 for row in rows for event in row.get("event_trace", []) or [] if event.get("type") == event_type)


def short_text(value: Any, limit: int = 140) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def render_index(output_dir: Path, max_items: int | None = None, refresh_seconds: int = 10) -> None:
    prediction_path = output_dir / "predictions.jsonl"
    rows = read_jsonl(prediction_path)
    if max_items:
        rows = rows[:max_items]

    tool_counter = Counter()
    status_counter = Counter()
    for row in rows:
        tool_counter.update(row.get("tool_call_counter") or {})
        status_counter.update([row.get("status") or "unknown"])

    row_html = []
    for index, row in enumerate(rows, start=1):
        qid = str(row.get("question_id") or "")
        stem = safe_trace_stem(qid)
        trace_html = output_dir / f"{stem}.html"
        href = f"{stem}.html" if trace_html.exists() else "#"
        status = str(row.get("status") or "")
        status_class = "ok" if status == "ok" else "bad"
        event_counter = Counter(event.get("type") for event in row.get("event_trace", []) or [])
        row_html.append(
            "<tr>"
            f"<td>{index}</td>"
            f"<td><a href='{escape(href)}'>{escape(qid)}</a></td>"
            f"<td class='{status_class}'>{escape(status)}</td>"
            f"<td>{escape(str(row.get('correct')))}</td>"
            f"<td>{escape(json.dumps(row.get('tool_call_counter') or {}, ensure_ascii=False))}</td>"
            f"<td>{escape(str(event_counter.get('format_error', 0)))}</td>"
            f"<td>{escape(str(event_counter.get('tool_error', 0)))}</td>"
            f"<td title='{escape(str(row.get('pred_answer') or ''))}'>{escape(short_text(row.get('pred_answer')))}</td>"
            f"<td title='{escape(str(row.get('gold_answer') or ''))}'>{escape(short_text(row.get('gold_answer'), 100))}</td>"
            f"<td title='{escape(str(row.get('question') or ''))}'>{escape(short_text(row.get('question'), 160))}</td>"
            "</tr>"
        )

    summary = {
        "completed": len(rows),
        "status": dict(status_counter),
        "format_errors": count_events(rows, "format_error"),
        "tool_errors": count_events(rows, "tool_error"),
        "tools": dict(tool_counter),
        "updated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    summary_json = json.dumps(summary, ensure_ascii=False, indent=2)

    html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="{int(refresh_seconds)}" />
  <title>Live Qwen SFT Trace Index</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      margin: 24px;
      color: #111827;
      background: #f8fafc;
    }}
    .summary {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .card {{
      background: white;
      border: 1px solid #e5e7eb;
      border-radius: 12px;
      padding: 14px;
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      background: white;
      border: 1px solid #e5e7eb;
    }}
    th, td {{
      border: 1px solid #e5e7eb;
      padding: 8px;
      vertical-align: top;
      font-size: 13px;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #f3f4f6;
      z-index: 1;
    }}
    a {{ color: #2563eb; }}
    .ok {{ color: #047857; font-weight: 700; }}
    .bad {{ color: #b91c1c; font-weight: 700; }}
    pre {{
      white-space: pre-wrap;
      background: #0f172a;
      color: #e5e7eb;
      border-radius: 8px;
      padding: 12px;
      margin: 0;
    }}
  </style>
</head>
<body>
  <h1>Live Qwen SFT Trace Index</h1>
  <div class="summary">
    <div class="card"><strong>Output Dir</strong><br />{escape(str(output_dir))}</div>
    <div class="card"><strong>Auto Refresh</strong><br />Every {int(refresh_seconds)} seconds</div>
    <div class="card"><strong>Prediction File</strong><br />{escape(str(prediction_path))}</div>
  </div>
  <div class="card">
    <h2>Summary</h2>
    <pre>{escape(summary_json)}</pre>
  </div>
  <h2>Completed Traces</h2>
  <table>
    <thead>
      <tr>
        <th>#</th>
        <th>question_id</th>
        <th>status</th>
        <th>exact</th>
        <th>tools</th>
        <th>format</th>
        <th>tool err</th>
        <th>pred</th>
        <th>gold</th>
        <th>question</th>
      </tr>
    </thead>
    <tbody>
      {"".join(row_html)}
    </tbody>
  </table>
</body>
</html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a live index.html for trace generation output.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--refresh-seconds", type=int, default=10)
    parser.add_argument("--max-items", type=int, default=0)
    parser.add_argument("--watch", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_items = args.max_items if args.max_items > 0 else None
    while True:
        render_index(output_dir, max_items=max_items, refresh_seconds=args.refresh_seconds)
        if not args.watch:
            break
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
