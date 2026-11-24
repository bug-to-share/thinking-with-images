from __future__ import annotations

import argparse
import html
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any, Dict, List


DEFAULT_GEMINI_DIR = (
    "./gemini_agent_native_bbox/results/"
    "cold_start_2000_seed42_with_answer"
)
DEFAULT_QWEN_DIR = (
    "./gemini_agent_native_bbox/results/"
    "qwen_sft_tool_trace_cold_start_train1791_w12_20260503_132838"
)
DEFAULT_OUTPUT_DIR = (
    "./gemini_agent_native_bbox/results/"
    "compare_gemini_qwen_cold_start"
)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_trace_rows(trace_dir: Path) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for path in sorted(trace_dir.glob("trace_*.json")):
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        qid = normalize_text(record.get("question_id"))
        if not qid:
            qid = path.stem.removeprefix("trace_")
        html_path = path.with_suffix(".html")
        rows[qid] = {
            "question_id": qid,
            "json_path": path,
            "html_path": html_path if html_path.exists() else None,
            "question": normalize_text(record.get("question")),
            "gold_answer": normalize_text(record.get("gold_answer") or record.get("gt_answer")),
            "pred_answer": normalize_text(record.get("pred_answer") or record.get("final_answer_text")),
            "correct": bool(record.get("correct")),
            "status": normalize_text(record.get("status")),
            "tool_call_counter": record.get("tool_call_counter") or {},
            "tool_calls_count": int(record.get("tool_calls_count") or 0),
            "benchmark": normalize_text(record.get("benchmark") or record.get("dataset")),
        }
    return rows


def rel(path: Path | None, base: Path) -> str:
    if path is None:
        return ""
    return os.path.relpath(path, base)


def build_rows(
    gemini_dir: Path,
    qwen_dir: Path,
    output_dir: Path,
    max_items: int,
    sample: str,
    seed: int,
) -> List[Dict[str, Any]]:
    gemini = load_trace_rows(gemini_dir)
    qwen = load_trace_rows(qwen_dir)
    rows: List[Dict[str, Any]] = []
    qids = sorted(set(gemini) & set(qwen))
    if sample == "random":
        rng = random.Random(seed)
        rng.shuffle(qids)
    if max_items > 0:
        qids = qids[:max_items]
    for qid in qids:
        g = gemini[qid]
        q = qwen[qid]
        rows.append(
            {
                "qid": qid,
                "question": q.get("question") or g.get("question"),
                "gold": q.get("gold_answer") or g.get("gold_answer"),
                "benchmark": q.get("benchmark") or g.get("benchmark"),
                "gemini_pred": g.get("pred_answer"),
                "gemini_correct": g.get("correct"),
                "gemini_status": g.get("status"),
                "gemini_tools": g.get("tool_call_counter"),
                "gemini_tool_calls": g.get("tool_calls_count"),
                "gemini_href": rel(g.get("html_path"), output_dir),
                "qwen_pred": q.get("pred_answer"),
                "qwen_correct": q.get("correct"),
                "qwen_status": q.get("status"),
                "qwen_tools": q.get("tool_call_counter"),
                "qwen_tool_calls": q.get("tool_calls_count"),
                "qwen_href": rel(q.get("html_path"), output_dir),
            }
        )
    return rows


def materialize_trace_html(rows: List[Dict[str, Any]], output_dir: Path, mode: str) -> List[Dict[str, Any]]:
    if mode == "none":
        return rows

    for side in ("gemini", "qwen"):
        side_dir = output_dir / side
        if side_dir.exists():
            shutil.rmtree(side_dir)
        side_dir.mkdir(parents=True, exist_ok=True)

    for row in rows:
        for side in ("gemini", "qwen"):
            key = f"{side}_href"
            href = row.get(key)
            if not href:
                continue
            src = (output_dir / href).resolve()
            if not src.exists():
                continue
            dst = output_dir / side / src.name
            if dst.exists() or dst.is_symlink():
                dst.unlink()
            if mode == "copy":
                shutil.copy2(src, dst)
            else:
                os.symlink(src, dst)
            row[key] = f"{side}/{src.name}"
    return rows


def render_html(rows: List[Dict[str, Any]], gemini_dir: Path, qwen_dir: Path) -> str:
    payload = json.dumps(rows, ensure_ascii=False)
    both_correct = sum(1 for row in rows if row["gemini_correct"] and row["qwen_correct"])
    gemini_only = sum(1 for row in rows if row["gemini_correct"] and not row["qwen_correct"])
    qwen_only = sum(1 for row in rows if row["qwen_correct"] and not row["gemini_correct"])
    both_wrong = sum(1 for row in rows if not row["gemini_correct"] and not row["qwen_correct"])
    return f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8" />
  <title>Gemini vs Qwen Cold-Start Trace Compare</title>
  <style>
    :root {{
      --bg: #f4f1ea;
      --ink: #1f2933;
      --muted: #5f6b7a;
      --line: #d8d1c4;
      --card: #fffaf0;
      --good: #047857;
      --bad: #b42318;
      --accent: #9a5b13;
    }}
    body {{
      margin: 0;
      background: var(--bg);
      color: var(--ink);
      font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }}
    header {{
      padding: 18px 22px 12px;
      border-bottom: 1px solid var(--line);
      background: linear-gradient(135deg, #fff8e7, #efe2ca);
      position: sticky;
      top: 0;
      z-index: 10;
    }}
    h1 {{ margin: 0 0 8px; font-size: 22px; }}
    .meta {{ color: var(--muted); font-size: 13px; }}
    .stats {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px;
      margin-top: 12px;
    }}
    .stat {{
      background: rgba(255,255,255,0.72);
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 8px 10px;
      font-size: 13px;
    }}
    .toolbar {{
      display: flex;
      gap: 10px;
      margin-top: 12px;
      flex-wrap: wrap;
      align-items: center;
    }}
    input, select {{
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 8px 10px;
      background: white;
      min-width: 180px;
    }}
    main {{
      display: grid;
      grid-template-columns: 430px minmax(0, 1fr);
      height: calc(100vh - 150px);
    }}
    aside {{
      overflow: auto;
      border-right: 1px solid var(--line);
      background: rgba(255,255,255,0.45);
    }}
    table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 12px;
    }}
    th, td {{
      border-bottom: 1px solid var(--line);
      padding: 7px 8px;
      vertical-align: top;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #efe2ca;
      z-index: 2;
      text-align: left;
    }}
    tr {{ cursor: pointer; }}
    tr:hover, tr.active {{ background: #fff4d6; }}
    .ok {{ color: var(--good); font-weight: 700; }}
    .bad {{ color: var(--bad); font-weight: 700; }}
    .qid {{ color: var(--accent); font-weight: 700; }}
    .question {{ color: var(--muted); max-width: 220px; }}
    .viewer {{
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      overflow: hidden;
    }}
    .detail {{
      padding: 12px 16px;
      border-bottom: 1px solid var(--line);
      background: var(--card);
      font-size: 13px;
    }}
    .frames {{
      display: grid;
      grid-template-columns: 1fr 1fr;
      min-height: 0;
      height: 100%;
    }}
    .panel {{
      min-width: 0;
      display: grid;
      grid-template-rows: auto 1fr;
      border-right: 1px solid var(--line);
    }}
    .panel:last-child {{ border-right: 0; }}
    .panel-title {{
      padding: 8px 10px;
      background: #2d2418;
      color: #fff7e6;
      font-weight: 700;
      display: flex;
      justify-content: space-between;
      gap: 10px;
      align-items: center;
      font-size: 13px;
    }}
    .panel-title a {{ color: #ffd47a; }}
    iframe {{
      width: 100%;
      height: 100%;
      border: 0;
      background: white;
    }}
    code {{
      background: rgba(0,0,0,0.06);
      padding: 1px 4px;
      border-radius: 4px;
    }}
  </style>
</head>
<body>
  <header>
    <h1>Gemini vs Qwen Cold-Start Trace Compare</h1>
    <div class="meta">Gemini: <code>{html.escape(str(gemini_dir))}</code><br />Qwen SFT: <code>{html.escape(str(qwen_dir))}</code></div>
    <div class="stats">
      <div class="stat">Common traces: <strong>{len(rows)}</strong></div>
      <div class="stat">Both correct: <strong>{both_correct}</strong></div>
      <div class="stat">Gemini only correct: <strong>{gemini_only}</strong></div>
      <div class="stat">Qwen only correct: <strong>{qwen_only}</strong></div>
      <div class="stat">Both wrong: <strong>{both_wrong}</strong></div>
    </div>
    <div class="toolbar">
      <input id="search" placeholder="search qid / question / answer" />
      <select id="filter">
        <option value="all">all</option>
        <option value="gemini_only">Gemini correct, Qwen wrong</option>
        <option value="qwen_only">Qwen correct, Gemini wrong</option>
        <option value="both_correct">both correct</option>
        <option value="both_wrong">both wrong</option>
      </select>
    </div>
  </header>
  <main>
    <aside>
      <table>
        <thead>
          <tr><th>#</th><th>qid</th><th>G</th><th>Q</th><th>question</th></tr>
        </thead>
        <tbody id="rows"></tbody>
      </table>
    </aside>
    <section class="viewer">
      <div class="detail" id="detail">Select a row.</div>
      <div class="frames">
        <div class="panel">
          <div class="panel-title"><span>Gemini with-answer trace</span><a id="geminiLink" href="#" target="_blank">open</a></div>
          <iframe id="geminiFrame"></iframe>
        </div>
        <div class="panel">
          <div class="panel-title"><span>Qwen SFT generated trace</span><a id="qwenLink" href="#" target="_blank">open</a></div>
          <iframe id="qwenFrame"></iframe>
        </div>
      </div>
    </section>
  </main>
  <script>
    const ROWS = {payload};
    const tbody = document.getElementById('rows');
    const search = document.getElementById('search');
    const filter = document.getElementById('filter');
    let activeQid = null;

    function cls(ok) {{ return ok ? 'ok' : 'bad'; }}
    function mark(ok) {{ return ok ? '✓' : '✗'; }}
    function esc(s) {{
      return String(s ?? '').replace(/[&<>"']/g, c => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[c]));
    }}
    function passFilter(row) {{
      const f = filter.value;
      if (f === 'gemini_only') return row.gemini_correct && !row.qwen_correct;
      if (f === 'qwen_only') return row.qwen_correct && !row.gemini_correct;
      if (f === 'both_correct') return row.gemini_correct && row.qwen_correct;
      if (f === 'both_wrong') return !row.gemini_correct && !row.qwen_correct;
      return true;
    }}
    function passSearch(row) {{
      const q = search.value.trim().toLowerCase();
      if (!q) return true;
      return [row.qid, row.question, row.gold, row.gemini_pred, row.qwen_pred, row.benchmark]
        .join(' ').toLowerCase().includes(q);
    }}
    function renderRows() {{
      const rows = ROWS.filter(r => passFilter(r) && passSearch(r));
      tbody.innerHTML = rows.map((r, i) => `
        <tr data-qid="${{esc(r.qid)}}" class="${{r.qid === activeQid ? 'active' : ''}}">
          <td>${{i + 1}}</td>
          <td><span class="qid">${{esc(r.qid)}}</span></td>
          <td class="${{cls(r.gemini_correct)}}">${{mark(r.gemini_correct)}}</td>
          <td class="${{cls(r.qwen_correct)}}">${{mark(r.qwen_correct)}}</td>
          <td class="question">${{esc(r.question).slice(0, 160)}}</td>
        </tr>`).join('');
      for (const tr of tbody.querySelectorAll('tr')) {{
        tr.addEventListener('click', () => selectRow(tr.dataset.qid));
      }}
    }}
    function selectRow(qid) {{
      const row = ROWS.find(r => r.qid === qid);
      if (!row) return;
      activeQid = qid;
      document.getElementById('detail').innerHTML = `
        <strong>${{esc(row.qid)}}</strong> · benchmark: ${{esc(row.benchmark)}}<br />
        <strong>Question:</strong> ${{esc(row.question)}}<br />
        <strong>Gold:</strong> ${{esc(row.gold)}}<br />
        <strong>Gemini:</strong> <span class="${{cls(row.gemini_correct)}}">${{esc(row.gemini_pred)}} (${{mark(row.gemini_correct)}})</span>
        · tools: <code>${{esc(JSON.stringify(row.gemini_tools))}}</code><br />
        <strong>Qwen:</strong> <span class="${{cls(row.qwen_correct)}}">${{esc(row.qwen_pred)}} (${{mark(row.qwen_correct)}})</span>
        · tools: <code>${{esc(JSON.stringify(row.qwen_tools))}}</code>`;
      document.getElementById('geminiFrame').src = row.gemini_href;
      document.getElementById('qwenFrame').src = row.qwen_href;
      document.getElementById('geminiLink').href = row.gemini_href;
      document.getElementById('qwenLink').href = row.qwen_href;
      const url = new URL(window.location);
      url.searchParams.set('qid', qid);
      history.replaceState(null, '', url);
      renderRows();
    }}
    search.addEventListener('input', renderRows);
    filter.addEventListener('change', renderRows);
    renderRows();
    const initial = new URL(window.location).searchParams.get('qid') || (ROWS[0] && ROWS[0].qid);
    if (initial) selectRow(initial);
  </script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gemini-dir", default=DEFAULT_GEMINI_DIR)
    parser.add_argument("--qwen-dir", default=DEFAULT_QWEN_DIR)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--sample", choices=("sorted", "random"), default="sorted")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--materialize",
        choices=("none", "symlink", "copy"),
        default="symlink",
        help="Make compared trace HTMLs available under output-dir so iframes work when serving output-dir directly.",
    )
    args = parser.parse_args()

    gemini_dir = Path(args.gemini_dir).resolve()
    qwen_dir = Path(args.qwen_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = build_rows(gemini_dir, qwen_dir, output_dir, args.max_items, args.sample, args.seed)
    rows = materialize_trace_html(rows, output_dir, args.materialize)
    html_text = render_html(rows, gemini_dir, qwen_dir)
    output_path = output_dir / "index.html"
    output_path.write_text(html_text, encoding="utf-8")
    summary = {
        "output": str(output_path),
        "gemini_dir": str(gemini_dir),
        "qwen_dir": str(qwen_dir),
        "common_traces": len(rows),
        "max_items": args.max_items,
        "sample": args.sample,
        "seed": args.seed,
        "materialize": args.materialize,
        "both_correct": sum(1 for row in rows if row["gemini_correct"] and row["qwen_correct"]),
        "gemini_only_correct": sum(1 for row in rows if row["gemini_correct"] and not row["qwen_correct"]),
        "qwen_only_correct": sum(1 for row in rows if row["qwen_correct"] and not row["gemini_correct"]),
        "both_wrong": sum(1 for row in rows if not row["gemini_correct"] and not row["qwen_correct"]),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
