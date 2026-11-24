from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

from openai import OpenAI
from tqdm.auto import tqdm

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from tool.common import ToolRuntimeContext

from loop_utils import (
    answer_letter,
    answer_text,
    build_vqa_prompt,
    extract_answer_text,
    extract_python_code,
    extract_tool_call,
    normalize_text,
    option_text_by_letter,
    parse_answer_letter,
    path_to_data_url,
    serialize_messages,
)
from prompts import (
    FORMAT_ERROR_TEXT,
    build_code_feedback_text,
    build_system_prompt,
    build_tool_feedback_text,
    build_user_prompt,
)
from sandbox_client import JupyterSandboxSession, runtime_index_to_python_var
from tool.bridge import (
    DEFAULT_BIOMEDPARSE_URL,
    DEFAULT_MEDSAM3_URL,
    DEFAULT_RETRIEVE_URL,
    ToolBridge,
)
from render_trace_html import render_trace_html


DEFAULT_VQA_PATH = './data/vqa/vqa.json'
DEFAULT_OUTPUT_DIR = './outputs/qwen35_agent/trace'
DEFAULT_MODEL = 'Qwen/Qwen3.5-9B'
DEFAULT_BASE_URL = 'http://localhost:8000/v1'
DEFAULT_MAX_ITEMS = 0
DEFAULT_MAX_TOKENS = 8192
DEFAULT_MAX_ROUNDS = 12
DEFAULT_REASONING_BACKEND = 'sglang'
DEFAULT_TOP_K = 20
SERVER_BACKED_TOOLS = {'retrieve', 'biomedparse', 'medsam3'}


def read_json(path: Path) -> Any:
    with path.open('r', encoding='utf-8') as file:
        return json.load(file)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
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


def content_text_message(text: str) -> Dict[str, Any]:
    return {'role': 'user', 'content': [{'type': 'text', 'text': text}]}


def content_text_with_images(text: str, image_data_urls: List[str]) -> Dict[str, Any]:
    content = [{'type': 'text', 'text': text}]
    for url in image_data_urls:
        content.append({'type': 'image_url', 'image_url': {'url': url}})
    return {'role': 'user', 'content': content}


def new_image_descriptions(indices: List[int]) -> List[str]:
    lines: List[str] = []
    for idx in indices:
        lines.append(f'runtime image_index={idx} is available as python variable `{runtime_index_to_python_var(idx)}`')
    return lines


def build_initial_messages(vqa_item: Dict[str, Any]) -> List[Dict[str, Any]]:
    image_path = str(Path(normalize_text(vqa_item.get('image_path'))).resolve())
    question_text = build_vqa_prompt(vqa_item)
    return [
        {
            'role': 'user',
            'content': [
                {'type': 'image_url', 'image_url': {'url': path_to_data_url(image_path)}},
                {'type': 'text', 'text': build_user_prompt(question_text)},
            ],
        }
    ]


def runtime_inventory(runtime: ToolRuntimeContext) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for record in runtime.image_records:
        rows.append(
            {
                'image_index': int(record.index),
                'path': str(record.path),
            }
        )
    return rows


def compact_tool_counter(trace: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for event in trace:
        if event.get('type') == 'tool':
            counter[event['name']] += 1
        elif event.get('type') == 'python':
            counter['python'] += 1
    return dict(counter)


def has_server_tool_event(trace: List[Dict[str, Any]]) -> bool:
    for event in trace:
        if event.get('type') == 'tool' and event.get('name') in SERVER_BACKED_TOOLS:
            return True
    return False


@dataclass
class ModelTurn:
    content: str
    reasoning: str
    action_text: str
    finish_reason: str
    raw_message: Dict[str, Any]


def message_extra_dict(message: Any) -> Dict[str, Any]:
    extra = getattr(message, 'model_extra', None)
    if isinstance(extra, dict):
        return extra
    return {}


def extract_reasoning_text(message: Any) -> str:
    direct = normalize_text(getattr(message, 'reasoning_content', None))
    if direct:
        return direct
    direct = normalize_text(getattr(message, 'reasoning', None))
    if direct:
        return direct
    extra = message_extra_dict(message)
    if extra.get('reasoning_content') is not None:
        return normalize_text(extra.get('reasoning_content'))
    if extra.get('reasoning') is not None:
        return normalize_text(extra.get('reasoning'))
    return ''


def contains_action_block(text: str) -> bool:
    if not text:
        return False
    return any(
        parser(text) is not None
        for parser in (extract_answer_text, extract_python_code, extract_tool_call)
    )


def request_extra_body(args: argparse.Namespace) -> Dict[str, Any]:
    extra: Dict[str, Any] = {}
    if args.top_k > 0:
        extra['top_k'] = args.top_k
    backend = normalize_text(args.reasoning_backend).lower()
    if backend == 'sglang':
        extra['separate_reasoning'] = True
    return extra


def model_complete(
    client: OpenAI,
    args: argparse.Namespace,
    model: str,
    base_messages: List[Dict[str, Any]],
    system_prompt: str,
    max_tokens: int,
    temperature: float,
) -> ModelTurn:
    response = client.chat.completions.create(
        model=model,
        messages=[{'role': 'system', 'content': system_prompt}] + base_messages,
        temperature=temperature,
        max_tokens=max_tokens,
        extra_body=request_extra_body(args),
    )
    choice = response.choices[0]
    message = choice.message
    content = normalize_text(message.content)
    reasoning = extract_reasoning_text(message)
    action_text = content if contains_action_block(content) else reasoning if contains_action_block(reasoning) else content
    raw_message = {
        'content': content,
        'reasoning': reasoning,
        'finish_reason': normalize_text(choice.finish_reason),
        'model_extra': message_extra_dict(message),
    }
    return ModelTurn(
        content=content,
        reasoning=reasoning,
        action_text=normalize_text(action_text),
        finish_reason=normalize_text(choice.finish_reason),
        raw_message=raw_message,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Custom thinking-with-images loop for Qwen medical VQA.')
    parser.add_argument('--vqa-path', default=DEFAULT_VQA_PATH)
    parser.add_argument('--output-dir', default=DEFAULT_OUTPUT_DIR)
    parser.add_argument('--model', default=DEFAULT_MODEL)
    parser.add_argument('--base-url', default=os.getenv('QWEN_BASE_URL', DEFAULT_BASE_URL))
    parser.add_argument('--max-items', type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument('--max-tokens', type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument('--max-rounds', type=int, default=DEFAULT_MAX_ROUNDS)
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--reasoning-backend', default=os.getenv('QWEN_REASONING_BACKEND', DEFAULT_REASONING_BACKEND), choices=['sglang', 'vllm'])
    parser.add_argument('--top-k', type=int, default=DEFAULT_TOP_K)
    parser.add_argument('--item-index', type=int, default=0)
    parser.add_argument('--retrieve-url', default=os.getenv('RETRIEVE_URL', DEFAULT_RETRIEVE_URL))
    parser.add_argument('--biomedparse-url', default=os.getenv('BIOMEDPARSE_URL', DEFAULT_BIOMEDPARSE_URL))
    parser.add_argument('--medsam3-url', default=os.getenv('MEDSAM3_URL', DEFAULT_MEDSAM3_URL))
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--require-server-tool-before-answer', dest='require_server_tool_before_answer', action='store_true')
    parser.add_argument('--allow-answer-without-server-tool', dest='require_server_tool_before_answer', action='store_false')
    parser.add_argument('--disable-progress', action='store_true')
    parser.set_defaults(require_server_tool_before_answer=True)
    return parser.parse_args()


def build_client(args: argparse.Namespace) -> OpenAI:
    return OpenAI(api_key=os.getenv('OPENAI_API_KEY', 'EMPTY'), base_url=args.base_url)


def build_failed_record(question_id: int, item: Dict[str, Any], args: argparse.Namespace, error: Exception) -> Dict[str, Any]:
    image_path = str(Path(normalize_text(item.get('image_path'))).resolve())
    gold_letter = answer_letter(item)
    return {
        'question_id': question_id,
        'image_path': image_path,
        'question': normalize_text(item.get('vqa_data', {}).get('question')),
        'source': normalize_text(item.get('source')),
        'gold_letter': gold_letter,
        'gold_text': answer_text(item),
        'pred_letter': None,
        'pred_text': '',
        'correct': False,
        'status': 'failed',
        'final_answer_text': None,
        'tool_call_counter': {},
        'tool_calls_count': 0,
        'assistant_messages': [],
        'messages': [],
        'event_trace': [
            {
                'type': 'exception',
                'error': str(error),
            }
        ],
        'reasoning_backend': args.reasoning_backend,
        'image_inventory': [{'image_index': 0, 'path': image_path}],
        'tool_server_urls': {
            'retrieve': args.retrieve_url,
            'biomedparse': args.biomedparse_url,
            'medsam3': args.medsam3_url,
        },
    }


def run_single_item_task(
    args: argparse.Namespace,
    question_id: int,
    item: Dict[str, Any],
    output_dir: Path,
    system_prompt: str,
) -> Dict[str, Any]:
    client = build_client(args)
    return run_single_item(
        client=client,
        args=args,
        question_id=question_id,
        item=item,
        output_dir=output_dir,
        system_prompt=system_prompt,
    )


def run_single_item(
    client: OpenAI,
    args: argparse.Namespace,
    question_id: int,
    item: Dict[str, Any],
    output_dir: Path,
    system_prompt: str,
) -> Dict[str, Any]:
    runtime = ToolRuntimeContext()
    image_path = str(Path(normalize_text(item.get('image_path'))).resolve())
    runtime.reset(image_path=image_path, sample_id=f'q_{question_id:06d}')

    sandbox = JupyterSandboxSession(runtime=runtime, sample_id=f'q_{question_id:06d}')
    sandbox.clear()
    sandbox.initialize_original_image()

    tool_bridge = ToolBridge(
        runtime=runtime,
        retrieve_url=args.retrieve_url,
        biomedparse_url=args.biomedparse_url,
        medsam3_url=args.medsam3_url,
    )
    messages = build_initial_messages(item)
    event_trace: List[Dict[str, Any]] = []
    assistant_messages: List[Dict[str, Any]] = []
    final_answer_text: Optional[str] = None

    for round_idx in range(args.max_rounds):
        model_turn = model_complete(
            client=client,
            args=args,
            model=args.model,
            base_messages=messages,
            system_prompt=system_prompt,
            max_tokens=args.max_tokens,
            temperature=args.temperature,
        )
        assistant_text = model_turn.action_text
        assistant_messages.append(
            {
                'role': 'assistant',
                'content': model_turn.content,
                'reasoning': model_turn.reasoning,
                'action_text': model_turn.action_text,
                'finish_reason': model_turn.finish_reason,
                'raw_message': model_turn.raw_message,
            }
        )
        history_assistant_text = model_turn.content or model_turn.action_text
        if history_assistant_text:
            messages.append({'role': 'assistant', 'content': history_assistant_text})

        answer_text_block = extract_answer_text(assistant_text)
        if answer_text_block:
            require_server_tool = bool(args.require_server_tool_before_answer)
            if require_server_tool and not has_server_tool_event(event_trace):
                messages.append(
                    content_text_message(
                        'Before the final answer, call at least one dedicated external tool such as '
                        '`biomedparse`, `medsam3`, or `retrieve`, then continue.'
                    )
                )
                event_trace.append(
                    {
                        'type': 'answer_rejected',
                        'round': round_idx,
                        'content': model_turn.content,
                        'reasoning': model_turn.reasoning,
                        'action_text': assistant_text,
                        'finish_reason': model_turn.finish_reason,
                        'reason': 'server_tool_required',
                    }
                )
                continue
            final_answer_text = answer_text_block
            event_trace.append(
                {
                    'type': 'answer',
                    'round': round_idx,
                    'content': model_turn.content,
                    'reasoning': model_turn.reasoning,
                    'action_text': assistant_text,
                    'finish_reason': model_turn.finish_reason,
                }
            )
            break

        code = extract_python_code(assistant_text)
        if code:
            exec_ret = sandbox.execute(code, timeout=60.0)
            new_indices = sandbox.register_code_output_images(exec_ret['images'], prefix='python_output')
            image_urls = [path_to_data_url(runtime.get_image_path(idx)) for idx in new_indices]
            feedback_text = build_code_feedback_text(
                stdout=exec_ret.get('stdout', ''),
                stderr=exec_ret.get('stderr', ''),
                image_descriptions=new_image_descriptions(new_indices),
            )
            messages.append(content_text_with_images(feedback_text, image_urls))
            event_trace.append(
                {
                    'type': 'python',
                    'round': round_idx,
                    'content': model_turn.content,
                    'reasoning': model_turn.reasoning,
                    'finish_reason': model_turn.finish_reason,
                    'code': code,
                    'stdout': exec_ret.get('stdout', ''),
                    'stderr': exec_ret.get('stderr', ''),
                    'new_indices': new_indices,
                }
            )
            continue

        tool_call = extract_tool_call(assistant_text)
        if tool_call:
            tool_name = normalize_text(tool_call.get('name'))
            arguments = tool_call.get('arguments') or {}
            try:
                result = tool_bridge.run(tool_name, arguments)
                sandbox.register_runtime_indices(result.new_indices)
                image_urls = [path_to_data_url(runtime.get_image_path(idx)) for idx in result.new_indices]
                feedback_text = build_tool_feedback_text(
                    tool_name=tool_name,
                    payload=result.model_payload,
                    image_descriptions=new_image_descriptions(result.new_indices),
                )
                messages.append(content_text_with_images(feedback_text, image_urls))
                event_trace.append(
                    {
                        'type': 'tool',
                        'round': round_idx,
                        'content': model_turn.content,
                        'reasoning': model_turn.reasoning,
                        'finish_reason': model_turn.finish_reason,
                        'name': tool_name,
                        'arguments': arguments,
                        'result': result.model_payload,
                        'new_indices': result.new_indices,
                    }
                )
            except Exception as exc:
                messages.append(content_text_message(f'<tool_response name="{tool_name}">error: {exc}</tool_response>'))
                event_trace.append(
                    {
                        'type': 'tool_error',
                        'round': round_idx,
                        'content': model_turn.content,
                        'reasoning': model_turn.reasoning,
                        'finish_reason': model_turn.finish_reason,
                        'name': tool_name,
                        'arguments': arguments,
                        'error': str(exc),
                    }
                )
            continue

        messages.append(content_text_message(FORMAT_ERROR_TEXT))
        event_trace.append(
            {
                'type': 'format_error',
                'round': round_idx,
                'content': model_turn.content,
                'reasoning': model_turn.reasoning,
                'action_text': assistant_text,
                'finish_reason': model_turn.finish_reason,
            }
        )

    pred_letter = parse_answer_letter(final_answer_text or '')
    gold_letter = answer_letter(item)
    is_correct = pred_letter == gold_letter
    trace_counter = compact_tool_counter(event_trace)

    return {
        'question_id': question_id,
        'image_path': image_path,
        'question': normalize_text(item.get('vqa_data', {}).get('question')),
        'source': normalize_text(item.get('source')),
        'gold_letter': gold_letter,
        'gold_text': answer_text(item),
        'pred_letter': pred_letter,
        'pred_text': option_text_by_letter(item, pred_letter),
        'correct': is_correct,
        'status': 'ok' if pred_letter else 'no_answer',
        'final_answer_text': final_answer_text,
        'tool_call_counter': trace_counter,
        'tool_calls_count': sum(trace_counter.values()),
        'assistant_messages': assistant_messages,
        'messages': serialize_messages(messages),
        'event_trace': event_trace,
        'reasoning_backend': args.reasoning_backend,
        'image_inventory': runtime_inventory(runtime),
        'tool_server_urls': {
            'retrieve': args.retrieve_url,
            'biomedparse': args.biomedparse_url,
            'medsam3': args.medsam3_url,
        },
    }


def write_trace_artifacts(record: Dict[str, Any], output_dir: Path) -> None:
    trace_stem = f'trace_q{int(record["question_id"]):06d}'
    json_path = output_dir / f'{trace_stem}.json'
    html_path = output_dir / f'{trace_stem}.html'
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding='utf-8')
    render_trace_html(record, html_path)


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

    data = read_json(Path(args.vqa_path))
    if args.item_index < 0 or args.item_index >= len(data):
        raise IndexError(f'item_index out of range: {args.item_index} for dataset size {len(data)}')
    if args.max_items > 0:
        data = data[args.item_index: args.item_index + args.max_items]
    else:
        data = data[args.item_index:]

    system_prompt = build_system_prompt(ToolBridge.tool_specs())
    indexed_data = [(args.item_index + question_id, item) for question_id, item in enumerate(data)]

    total_records: List[Dict[str, Any]] = []
    progress = tqdm(
        total=len(data),
        desc='Qwen Agent Eval',
        dynamic_ncols=True,
        disable=bool(args.disable_progress),
    )
    if args.num_workers <= 1:
        client = build_client(args)
        for global_question_id, item in indexed_data:
            record = run_single_item(
                client=client,
                args=args,
                question_id=global_question_id,
                item=item,
                output_dir=output_dir,
                system_prompt=system_prompt,
            )
            total_records.append(record)
            append_jsonl(record_path, record)
            write_trace_artifacts(record, output_dir)
            progress.update(1)
            progress.set_postfix(
                qid=global_question_id,
                status=record.get('status'),
                pred=record.get('pred_letter') or '-',
                correct=record.get('correct'),
                refresh=False,
            )
    else:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            future_map = {
                executor.submit(
                    run_single_item_task,
                    args,
                    global_question_id,
                    item,
                    output_dir,
                    system_prompt,
                ): (global_question_id, item)
                for global_question_id, item in indexed_data
            }
            for future in as_completed(future_map):
                global_question_id, item = future_map[future]
                try:
                    record = future.result()
                except Exception as exc:
                    record = build_failed_record(global_question_id, item, args, exc)
                total_records.append(record)
                append_jsonl(record_path, record)
                write_trace_artifacts(record, output_dir)
                progress.update(1)
                progress.set_postfix(
                    qid=global_question_id,
                    status=record.get('status'),
                    pred=record.get('pred_letter') or '-',
                    correct=record.get('correct'),
                    refresh=False,
                )
    progress.close()

    total_records.sort(key=lambda record: int(record['question_id']))
    write_jsonl(record_path, total_records)

    attempted = len(total_records)
    correct_total = sum(1 for record in total_records if record.get('correct'))
    failed_total = sum(1 for record in total_records if record.get('status') == 'failed')
    tool_counter = Counter()
    for record in total_records:
        tool_counter.update(record.get('tool_call_counter', {}))
    summary = {
        'total_input': len(data),
        'attempted': attempted,
        'correct': correct_total,
        'failed': failed_total,
        'accuracy': round(correct_total / attempted, 6) if attempted else 0.0,
        'tool_call_counter': dict(tool_counter),
        'model': args.model,
        'base_url': args.base_url,
        'num_workers': args.num_workers,
        'max_rounds': args.max_rounds,
        'reasoning_backend': args.reasoning_backend,
        'top_k': args.top_k,
        'vqa_path': args.vqa_path,
        'item_index': args.item_index,
        'retrieve_url': args.retrieve_url,
        'biomedparse_url': args.biomedparse_url,
        'medsam3_url': args.medsam3_url,
        'require_server_tool_before_answer': bool(args.require_server_tool_before_answer),
    }
    with summary_path.open('w', encoding='utf-8') as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
