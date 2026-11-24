from __future__ import annotations

import json
from pathlib import Path

import agent_loop as base
from prompts_with_answer import build_question_text_with_answer, build_system_prompt


def build_initial_contents_with_answer(item: dict) -> list[dict]:
    image_path = str(Path(base.normalize_text(item.get('_model_image_path_abs') or item.get('image_path_abs'))).resolve())
    question_text = build_question_text_with_answer(item)
    return [
        {
            'role': 'user',
            'parts': [
                base.file_to_inline_data_part(image_path),
                {'text': question_text},
            ],
        }
    ]


def main() -> None:
    args = base.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / 'predictions.jsonl'
    summary_path = output_dir / 'summary.json'
    if record_path.exists():
        record_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    data = base.read_jsonl(Path(args.jsonl_path))
    if args.item_index < 0 or args.item_index >= len(data):
        raise IndexError(f'item_index out of range: {args.item_index} for dataset size {len(data)}')
    if args.max_items > 0:
        data = data[args.item_index: args.item_index + args.max_items]
    else:
        data = data[args.item_index:]

    args.sam3_urls = base.parse_base_urls(args.sam3_url, base.DEFAULT_SAM3_URL)
    args.sam3_pool = base.RoundRobinURLPool(args.sam3_urls)

    original_builder = base.build_initial_contents
    try:
        base.build_initial_contents = build_initial_contents_with_answer
        system_prompt = base.maybe_transform_system_prompt(
            build_system_prompt(base.tool_specs()),
            use_thoughts_reason=args.use_thoughts_reason,
        )

        if args.dry_run:
            preview = {
                'num_items': len(data),
                'first_question_id': base.normalize_text(data[0].get('question_id')) if data else '',
                'first_question': base.normalize_text(data[0].get('question')) if data else '',
                'first_gold_answer': base.normalize_text(data[0].get('gt_answer')) if data else '',
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

        total_records = []
        if args.num_workers <= 1:
            client = base.build_client(args)
            judge_client = base.build_judge_client(args)
            for item in data:
                record = base.run_single_item_with_retries(
                    client=client,
                    judge_client=judge_client,
                    args=args,
                    item=item,
                    output_dir=output_dir,
                    system_prompt=system_prompt,
                )
                total_records.append(record)
                base.append_jsonl(record_path, record)
                base.write_trace_artifacts(record, output_dir)
        else:
            with base.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
                future_map = {
                    executor.submit(
                        base.run_single_item_task,
                        args,
                        item,
                        output_dir,
                        system_prompt,
                    ): item
                    for item in data
                }
                for future in base.as_completed(future_map):
                    item = future_map[future]
                    try:
                        record = future.result()
                    except Exception as exc:
                        record = base.build_failed_record(
                            base.normalize_text(item.get('question_id')),
                            item,
                            args,
                            exc,
                            system_prompt=system_prompt,
                        )
                    total_records.append(record)
                    base.append_jsonl(record_path, record)
                    base.write_trace_artifacts(record, output_dir)

        total_records.sort(key=lambda row: base.normalize_text(row.get('question_id')))
        base.write_jsonl(record_path, total_records)

        attempted = len(total_records)
        correct_total = sum(1 for row in total_records if row.get('correct'))
        failed_total = sum(1 for row in total_records if row.get('status') == 'failed')
        tool_counter = base.Counter()
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
            'mode': 'with_answer',
        }
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    finally:
        base.build_initial_contents = original_builder


if __name__ == '__main__':
    main()
