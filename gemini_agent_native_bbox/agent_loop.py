from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from tool.common import ToolRuntimeContext

from gemini_bbox_format import extract_coordinate_texts, load_image_size, normalize_tool_arguments_for_execution
from gemini_client import DEFAULT_MODEL, GeminiClient
from gemini_tool_bridge import DEFAULT_SAM3_URL, GeminiToolBridge
from judge_client import LLMJudgeClient, judge_base_from_env
from model_canvas import CANVAS_SIZE, prepare_model_canvas
from loop_utils import (
    answers_match,
    build_question_text,
    extract_answer_text,
    extract_reason_text,
    extract_tool_call,
    file_to_inline_data_part,
    normalize_text,
    safe_trace_stem,
    strip_trailing_action_garbage,
    simplify_contents,
    validate_response_format,
)
from prompts import FORMAT_ERROR_TEXT, build_system_prompt, build_tool_feedback_text, build_user_feedback_text
from render_trace_html import render_trace_html
from sandbox_client import DEFAULT_SANDBOX_URL
from sandbox_client import runtime_index_to_python_var


DEFAULT_JSONL_PATH = './data_curation/results/slake_en_vqa_rad_open_qa_full/slake_keep_for_rl.jsonl'
DEFAULT_OUTPUT_DIR = './gemini_agent/results/slake_keep_for_rl'
DEFAULT_MAX_ITEMS = 0
DEFAULT_MAX_ROUNDS = 10
DEFAULT_MAX_TOKENS = 768


def parse_base_urls(value: Any, default: str) -> List[str]:
    if isinstance(value, str):
        candidates = value.split(',')
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = [default]
    urls = [str(item).strip().rstrip('/') for item in candidates if str(item).strip()]
    return urls or [default.rstrip('/')]


class RoundRobinURLPool:
    def __init__(self, urls: List[str]):
        if not urls:
            raise ValueError('urls must not be empty')
        self.urls = list(urls)
        self._index = 0
        self._lock = threading.Lock()

    def next(self) -> str:
        with self._lock:
            url = self.urls[self._index % len(self.urls)]
            self._index += 1
            return url


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open('r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open('a', encoding='utf-8') as file:
        file.write(json.dumps(row, ensure_ascii=False) + '\n')


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open('w', encoding='utf-8') as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + '\n')


def tool_specs() -> List[Dict[str, Any]]:
    return [
        {
            'name': 'sam3',
            'description': (
                'Use SAM3 for text-guided segmentation of one known visible target in one known region. '
                'The text should be a concrete, plain target description such as fluid around liver, dark fluid beside liver, right lung mass, or bright spot in left kidney. '
                'Do not use a vague one-word target when a location or appearance phrase would be clearer. '
                'SAM3 must always be called with text+bbox+exactly 3 positive_points+exactly 3 negative_points together. '
                'Model-facing coordinates must use Gemini native normalized format, not absolute pixels. '
                'The bbox should be as precise and tight as possible around the intended target, not a loose region that includes many unrelated structures. '
                'First localize a reliable region with python code inspection, then choose 3 positive points inside the target and 3 negative points on nearby confusing non-target structures.'
            ),
            'arguments': {
                'image_index': 'int',
                'text': 'required concrete visible target description, preferably simple location + appearance wording',
                'bbox': 'required Gemini box: [ymin, xmin, ymax, xmax], normalized to 0..1000',
                'positive_points': 'required list of exactly 3 Gemini positive points inside the target: [[y, x], ...], normalized to 0..1000',
                'negative_points': 'required list of exactly 3 Gemini negative points on nearby non-target regions: [[y, x], ...], normalized to 0..1000',
            },
        },
        {
            'name': 'python_exec',
            'description': (
                'Use a stateful Jupyter-style Python sandbox to analyze images. All currently available images are already loaded '
                'as PIL.Image variables image_0, image_1, ... . Use this tool for crop, zoom, contrast adjustment, measurement, '
                'or other image analysis. Use matplotlib plt.show() when you want returned images.'
            ),
            'arguments': {
                'code': 'python code string',
                'timeout': 'optional timeout in seconds',
            },
        },
    ]


def runtime_inventory(runtime: ToolRuntimeContext) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in runtime.image_records:
        rows.append({'image_index': int(record.index), 'path': str(record.path)})
    return rows


def build_initial_contents(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_path = str(Path(normalize_text(item.get('_model_image_path_abs') or item.get('image_path_abs'))).resolve())
    question_text = build_question_text(item)
    return [
        {
            'role': 'user',
            'parts': [
                file_to_inline_data_part(image_path),
                {'text': question_text},
            ],
        }
    ]


def content_text_message(text: str) -> Dict[str, Any]:
    return {'role': 'user', 'parts': [{'text': text}]}


def system_text_message(text: str) -> Dict[str, Any]:
    return {'role': 'system', 'parts': [{'text': text}]}


def content_text_with_images(text: str, image_paths: List[str]) -> Dict[str, Any]:
    parts: List[Dict[str, Any]] = [{'text': text}]
    for image_path in image_paths:
        parts.append(file_to_inline_data_part(image_path))
    return {'role': 'user', 'parts': parts}


def new_image_descriptions(indices: List[int]) -> List[str]:
    return [
        f'runtime image_index={idx} is available as python variable `{runtime_index_to_python_var(idx)}`'
        for idx in indices
    ]


@dataclass
class ModelTurn:
    response_text: str
    reason: str
    finish_reason: str
    raw_payload: Dict[str, Any]
    thought_text: str = ''


def build_client(args: argparse.Namespace) -> GeminiClient:
    return GeminiClient(
        api_key=args.api_key or os.getenv('GEMINI_API_KEY', ''),
        model=args.model,
        base_url=args.base_url,
        timeout=args.request_timeout,
    )


def build_judge_client(args: argparse.Namespace) -> LLMJudgeClient | None:
    if not getattr(args, 'use_llm_judge', False):
        return None
    base_url = normalize_text(getattr(args, 'judge_base_url', '')) or judge_base_from_env()
    if not base_url:
        return None
    return LLMJudgeClient(
        base_url=base_url,
        api_key=normalize_text(getattr(args, 'judge_api_key', '')) or 'EMPTY',
        timeout=float(getattr(args, 'judge_timeout', 60.0)),
    )


def maybe_transform_system_prompt(system_prompt: str, use_thoughts_reason: bool) -> str:
    if not use_thoughts_reason:
        return system_prompt
    appendix = """

Experimental thought-to-reason mode:
- Do not output `<reason>...</reason>` yourself.
- The system will derive the `<reason>` block from your Gemini thought stream.
- Your visible response must therefore contain only exactly one `<tool_call>...</tool_call>` or one `<answer>...</answer>`.
- Do not add any text before or after that action block.
""".strip()
    return f"{system_prompt}\n\n{appendix}"


def model_complete(
    client: GeminiClient,
    args: argparse.Namespace,
    contents: List[Dict[str, Any]],
    system_prompt: str,
) -> ModelTurn:
    response = client.generate(
        contents=contents,
        system_prompt=system_prompt,
        temperature=args.temperature,
        max_output_tokens=args.max_tokens,
        top_p=args.top_p,
        top_k=args.top_k,
        include_thoughts=args.use_thoughts_reason,
    )
    response_text = strip_trailing_action_garbage(normalize_text(response.text))
    thought_text = normalize_text(response.thought_text)
    if args.use_thoughts_reason and thought_text and not response_text.startswith('<reason>'):
        response_text = f'<reason>{thought_text}</reason>{response_text}'
    return ModelTurn(
        response_text=response_text,
        reason=normalize_text(extract_reason_text(response_text)),
        finish_reason=normalize_text(response.finish_reason),
        raw_payload=response.raw_payload,
        thought_text=thought_text,
    )


def compact_tool_counter(trace: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for event in trace:
        if event.get('type') == 'tool':
            counter[event['name']] += 1
    return dict(counter)


def build_failed_record(
    question_id: str,
    item: Dict[str, Any],
    args: argparse.Namespace,
    error: Exception,
    system_prompt: str = '',
) -> Dict[str, Any]:
    return {
        'question_id': question_id,
        'image_path': str(Path(normalize_text(item.get('image_path_abs'))).resolve()),
        'question': normalize_text(item.get('question')),
        'gold_answer': normalize_text(item.get('gt_answer')),
        'pred_answer': '',
        'correct': False,
        'status': 'failed',
        'event_trace': [{'type': 'exception', 'error': str(error)}],
        'messages': simplify_contents([system_text_message(system_prompt)]) if system_prompt else [],
        'tool_call_counter': {},
        'tool_calls_count': 0,
        'model': args.model,
        'system_prompt': system_prompt,
    }


def is_retryable_record(record: Dict[str, Any]) -> bool:
    status = normalize_text(record.get('status'))
    if status in {'failed', 'no_answer'}:
        return True
    if not normalize_text(record.get('pred_answer')):
        return True
    return False


def run_single_item_with_retries(
    client: GeminiClient,
    judge_client: LLMJudgeClient | None,
    args: argparse.Namespace,
    item: Dict[str, Any],
    output_dir: Path,
    system_prompt: str,
) -> Dict[str, Any]:
    max_attempts = max(1, int(getattr(args, 'item_retries', 0) or 0) + 1)
    question_id = normalize_text(item.get('question_id'))
    last_record: Dict[str, Any] | None = None
    for attempt_index in range(max_attempts):
        try:
            record = run_single_item(
                client=client,
                judge_client=judge_client,
                args=args,
                item=item,
                output_dir=output_dir,
                system_prompt=system_prompt,
            )
        except Exception as exc:
            record = build_failed_record(question_id, item, args, exc, system_prompt=system_prompt)
        record['attempt_index'] = attempt_index
        record['num_attempts'] = attempt_index + 1
        record['max_attempts'] = max_attempts
        last_record = record
        if not is_retryable_record(record):
            break
    assert last_record is not None
    return last_record


def run_single_item_task(
    args: argparse.Namespace,
    item: Dict[str, Any],
    output_dir: Path,
    system_prompt: str,
) -> Dict[str, Any]:
    client = build_client(args)
    judge_client = build_judge_client(args)
    return run_single_item_with_retries(
        client=client,
        judge_client=judge_client,
        args=args,
        item=item,
        output_dir=output_dir,
        system_prompt=system_prompt,
    )


def run_single_item(
    client: GeminiClient,
    judge_client: LLMJudgeClient | None,
    args: argparse.Namespace,
    item: Dict[str, Any],
    output_dir: Path,
    system_prompt: str,
) -> Dict[str, Any]:
    question_id = normalize_text(item.get('question_id'))
    original_image_path = str(Path(normalize_text(item.get('image_path_abs'))).resolve())
    canvas_dir = output_dir / '_model_canvas'
    canvas_path = canvas_dir / f'{safe_trace_stem(question_id)}.png'
    canvas_info = prepare_model_canvas(original_image_path, canvas_path, canvas_size=CANVAS_SIZE)
    working_item = dict(item)
    working_item['_model_image_path_abs'] = canvas_info.image_path
    working_item['_model_canvas_info'] = canvas_info.as_dict()
    working_item['_model_image_size'] = [canvas_info.canvas_size, canvas_info.canvas_size]
    original_image_size = load_image_size(canvas_info.image_path)

    runtime = ToolRuntimeContext()
    runtime.reset(image_path=canvas_info.image_path, sample_id=question_id or 'unknown')

    if not hasattr(args, 'sam3_pool'):
        args.sam3_urls = parse_base_urls(getattr(args, 'sam3_url', DEFAULT_SAM3_URL), DEFAULT_SAM3_URL)
        args.sam3_pool = RoundRobinURLPool(args.sam3_urls)

    tool_bridge = GeminiToolBridge(
        runtime=runtime,
        sam3_url=args.sam3_url,
        sandbox_url=args.sandbox_url,
        request_timeout=args.request_timeout,
    )
    contents = build_initial_contents(working_item)
    event_trace: List[Dict[str, Any]] = []
    final_answer_text: Optional[str] = None
    allowed_tool_names = {spec['name'] for spec in tool_specs()}

    for round_idx in range(args.max_rounds):
        model_turn = model_complete(client=client, args=args, contents=contents, system_prompt=system_prompt)
        response_text = model_turn.response_text
        format_error = validate_response_format(response_text)
        contents.append({'role': 'model', 'parts': [{'text': response_text}]})

        if format_error:
            event_trace.append(
                {
                    'type': 'format_error',
                    'round': round_idx,
                    'reason': model_turn.reason,
                    'response_text': response_text,
                    'finish_reason': model_turn.finish_reason,
                    'error': format_error,
                }
            )
            contents.append(content_text_message(build_user_feedback_text(FORMAT_ERROR_TEXT)))
            continue

        answer_text = extract_answer_text(response_text)
        if answer_text is not None:
            final_answer_text = answer_text
            event_trace.append(
                {
                    'type': 'answer',
                    'round': round_idx,
                    'reason': model_turn.reason,
                    'response_text': response_text,
                    'answer': answer_text,
                    'finish_reason': model_turn.finish_reason,
                }
            )
            break

        tool_call = extract_tool_call(response_text)
        if tool_call is None:
            event_trace.append(
                {
                    'type': 'format_error',
                    'round': round_idx,
                    'reason': model_turn.reason,
                    'response_text': response_text,
                    'finish_reason': model_turn.finish_reason,
                    'error': 'unable to parse tool_call',
                }
            )
            contents.append(content_text_message(build_user_feedback_text(FORMAT_ERROR_TEXT)))
            continue

        tool_name = normalize_text(tool_call.get('name'))
        if tool_name == 'sandbox':
            tool_name = 'python_exec'
        raw_arguments = tool_call.get('arguments') or {}
        coordinate_matches = extract_coordinate_texts(response_text)
        arguments = raw_arguments
        if tool_name == 'sam3':
            image_index = int(raw_arguments.get('image_index', 0) or 0)
            if image_index == 0:
                image_size = original_image_size
            else:
                image_size = runtime.get_image(image_index).size
            arguments = normalize_tool_arguments_for_execution(raw_arguments, image_size)
        if tool_name not in allowed_tool_names:
            error_text = f'<tool_response name="{tool_name}">error: unsupported tool in current configuration</tool_response>'
            contents.append(content_text_message(error_text))
            event_trace.append(
                {
                    'type': 'tool_error',
                    'round': round_idx,
                    'name': tool_name,
                    'arguments': raw_arguments,
                    'normalized_arguments': arguments,
                    'coordinate_matches': coordinate_matches,
                    'error': 'unsupported tool in current configuration',
                    'base_url': None,
                    'reason': model_turn.reason,
                    'response_text': response_text,
                    'finish_reason': model_turn.finish_reason,
                }
            )
            continue
        try:
            selected_base_url = None
            if tool_name == 'sam3':
                selected_base_url = args.sam3_pool.next()
                tool_bridge.sam3_url = selected_base_url
            result = tool_bridge.run(tool_name, arguments)
            image_paths = [runtime.get_image_path(idx) for idx in result.new_indices]
            feedback_text = build_tool_feedback_text(
                tool_name=tool_name,
                payload=result.model_payload,
                image_descriptions=new_image_descriptions(result.new_indices),
            )
            contents.append(content_text_with_images(feedback_text, image_paths))
            event_trace.append(
                {
                    'type': 'tool',
                    'round': round_idx,
                    'name': tool_name,
                    'arguments': raw_arguments,
                    'normalized_arguments': arguments,
                    'coordinate_matches': coordinate_matches,
                    'result': result.model_payload,
                    'base_url': selected_base_url,
                    'new_indices': result.new_indices,
                    'reason': model_turn.reason,
                    'response_text': response_text,
                    'finish_reason': model_turn.finish_reason,
                }
            )
        except Exception as exc:
            error_text = f'<tool_response name="{tool_name}">error: {exc}</tool_response>'
            contents.append(content_text_message(error_text))
            event_trace.append(
                {
                    'type': 'tool_error',
                    'round': round_idx,
                    'name': tool_name,
                    'arguments': raw_arguments,
                    'normalized_arguments': arguments,
                    'coordinate_matches': coordinate_matches,
                    'error': str(exc),
                    'base_url': selected_base_url,
                    'reason': model_turn.reason,
                    'response_text': response_text,
                    'finish_reason': model_turn.finish_reason,
                }
            )

    pred_answer = normalize_text(final_answer_text)
    gold_answer = normalize_text(item.get('gt_answer'))
    correct_method = 'exact_match'
    judge_payload: Dict[str, Any] | None = None
    is_correct = answers_match(pred_answer, gold_answer)
    if pred_answer and not is_correct and judge_client is not None:
        judge_result = judge_client.judge_semantic_equivalence(
            question=normalize_text(item.get('question')),
            prediction=pred_answer,
            gold=gold_answer,
        )
        is_correct = judge_result.correct
        correct_method = 'llm_judge'
        judge_payload = {
            'judge_model': judge_result.model_name,
            'judge_raw_text': judge_result.raw_text,
            'judge_base_url': judge_client.base_url,
        }
    trace_counter = compact_tool_counter(event_trace)

    return {
        'question_id': question_id,
        'image_path': original_image_path,
        'model_image_path': canvas_info.image_path,
        'question': normalize_text(item.get('question')),
        'gold_answer': gold_answer,
        'pred_answer': pred_answer,
        'correct': is_correct,
        'correctness_method': correct_method,
        'judge_result': judge_payload,
        'status': 'ok' if pred_answer else 'no_answer',
        'benchmark': normalize_text(item.get('benchmark')),
        'split': normalize_text(item.get('split')),
        'modality': normalize_text(item.get('modality')),
        'location': normalize_text(item.get('location')),
        'content_type': normalize_text(item.get('content_type')),
        'final_answer_text': final_answer_text,
        'tool_call_counter': trace_counter,
        'tool_calls_count': sum(trace_counter.values()),
        'messages': simplify_contents([system_text_message(system_prompt)] + contents),
        'event_trace': event_trace,
        'image_inventory': runtime_inventory(runtime),
        'model_canvas': canvas_info.as_dict(),
        'model': args.model,
        'system_prompt': system_prompt,
        'jsonl_path': args.jsonl_path,
        'tool_server_urls': {
            'sam3': args.sam3_url,
            'sandbox': args.sandbox_url,
        },
    }


def write_trace_artifacts(record: Dict[str, Any], output_dir: Path) -> None:
    trace_stem = safe_trace_stem(record.get('question_id'))
    json_path = output_dir / f'{trace_stem}.json'
    html_path = output_dir / f'{trace_stem}.html'
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    render_trace_html(record, html_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Gemini medical visual reasoning agent with SAM3 and python sandbox tool calls.')
    parser.add_argument('--jsonl-path', default=DEFAULT_JSONL_PATH)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--base-url', default=os.getenv('GEMINI_BASE_URL', 'https://generativelanguage.googleapis.com/v1beta'))
    parser.add_argument('--api-key', default=os.getenv('GEMINI_API_KEY', ''))
    parser.add_argument('--max-items', type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument('--item-index', type=int, default=0)
    parser.add_argument('--max-rounds', type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument('--temperature', type=float, default=0.2)
    parser.add_argument('--top-p', type=float, default=0.9)
    parser.add_argument('--top-k', type=int, default=20)
    parser.add_argument('--item-retries', type=int, default=0, help='Retry each failed/no_answer item this many times')
    parser.add_argument('--use-llm-judge', action='store_true')
    parser.add_argument('--judge-base-url', default=os.getenv('LLM_AS_A_JUDGE_BASE', ''))
    parser.add_argument('--judge-api-key', default=os.getenv('LLM_AS_A_JUDGE_API_KEY', 'EMPTY'))
    parser.add_argument('--judge-timeout', type=float, default=60.0)
    parser.add_argument('--use-thoughts-reason', action='store_true')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--request-timeout', type=float, default=240.0)
    parser.add_argument('--sam3-url', default=os.getenv('SAM3_URL', DEFAULT_SAM3_URL))
    parser.add_argument('--sandbox-url', default=os.getenv('MED_SANDBOX_URL', DEFAULT_SANDBOX_URL))
    parser.add_argument('--disable-progress', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / 'predictions.jsonl'
    summary_path = output_dir / 'summary.json'
    if record_path.exists():
        record_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    data = read_jsonl(Path(args.jsonl_path))
    if args.item_index < 0 or args.item_index >= len(data):
        raise IndexError(f'item_index out of range: {args.item_index} for dataset size {len(data)}')
    if args.max_items > 0:
        data = data[args.item_index: args.item_index + args.max_items]
    else:
        data = data[args.item_index:]

    args.sam3_urls = parse_base_urls(args.sam3_url, DEFAULT_SAM3_URL)
    args.sam3_pool = RoundRobinURLPool(args.sam3_urls)

    system_prompt = maybe_transform_system_prompt(
        build_system_prompt(tool_specs()),
        use_thoughts_reason=args.use_thoughts_reason,
    )
    if args.dry_run:
        preview = {
            'num_items': len(data),
            'first_question_id': normalize_text(data[0].get('question_id')) if data else '',
            'first_question': normalize_text(data[0].get('question')) if data else '',
            'sam3_urls': args.sam3_urls,
            'top_p': args.top_p,
            'top_k': args.top_k,
            'use_llm_judge': args.use_llm_judge,
            'judge_base_url': args.judge_base_url,
            'use_thoughts_reason': args.use_thoughts_reason,
            'system_prompt': system_prompt,
        }
        print(json.dumps(preview, ensure_ascii=False, indent=2))
        return

    total_records: List[Dict[str, Any]] = []
    if args.num_workers <= 1:
        client = build_client(args)
        judge_client = build_judge_client(args)
        for item in data:
            record = run_single_item_with_retries(
                client=client,
                judge_client=judge_client,
                args=args,
                item=item,
                output_dir=output_dir,
                system_prompt=system_prompt,
            )
            total_records.append(record)
            append_jsonl(record_path, record)
            write_trace_artifacts(record, output_dir)
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_map = {
                executor.submit(run_single_item_task, args, item, output_dir, system_prompt): item
                for item in data
            }
            for future in as_completed(future_map):
                item = future_map[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = build_failed_record(
                        normalize_text(item.get('question_id')),
                        item,
                        args,
                        exc,
                        system_prompt=system_prompt,
                    )
                total_records.append(record)
                append_jsonl(record_path, record)
                write_trace_artifacts(record, output_dir)

    total_records.sort(key=lambda row: normalize_text(row.get('question_id')))
    write_jsonl(record_path, total_records)

    attempted = len(total_records)
    correct_total = sum(1 for row in total_records if row.get('correct'))
    failed_total = sum(1 for row in total_records if row.get('status') == 'failed')
    tool_counter = Counter()
    for row in total_records:
        tool_counter.update(row.get('tool_call_counter', {}))
    summary = {
        'total_input': len(data),
        'attempted': attempted,
        'correct': correct_total,
        'failed': failed_total,
        'accuracy': round(correct_total / attempted, 6) if attempted else 0.0,
        'tool_call_counter': dict(tool_counter),
        'model': args.model,
        'base_url': args.base_url,
        'jsonl_path': args.jsonl_path,
        'item_index': args.item_index,
        'num_workers': args.num_workers,
        'max_rounds': args.max_rounds,
        'max_tokens': args.max_tokens,
        'temperature': args.temperature,
        'top_p': args.top_p,
        'top_k': args.top_k,
        'item_retries': args.item_retries,
        'use_llm_judge': args.use_llm_judge,
        'judge_base_url': args.judge_base_url,
        'use_thoughts_reason': args.use_thoughts_reason,
        'sam3_url': args.sam3_url,
        'sam3_urls': args.sam3_urls,
        'sandbox_url': args.sandbox_url,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
