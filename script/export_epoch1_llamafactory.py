from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DEFAULT_VQA_PATH = "./data/vqa/vqa.json"
DEFAULT_TRACE_SPLIT_DIR = "./result/trace_split"
DEFAULT_TRACE1_DIR = "./outputs/qwen35_agent/trace_full"
DEFAULT_LLAMAFACTORY_DIR = "./LlamaFactory"
DEFAULT_OUTPUT_NAME = "med_agent_epoch1"
DEFAULT_SEED = 42
DEFAULT_TOTAL_SAMPLES = 3000
FORCE_TOOL_REMINDER_PREFIX = "Before the final answer, call at least one dedicated external tool"


THIS_DIR = Path(__file__).resolve().parent
PROJECT_DIR = THIS_DIR.parent
QWEN_AGENT_DIR = PROJECT_DIR / "qwen35_agent"
if str(QWEN_AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(QWEN_AGENT_DIR))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from prompts import build_system_prompt


def read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def build_question_text(vqa_item: Dict[str, Any]) -> str:
    question = normalize_text(vqa_item.get("vqa_data", {}).get("question"))
    options = [normalize_text(option) for option in vqa_item.get("vqa_data", {}).get("options", [])]
    option_block = "\n".join(options)
    return f"<image>\n{question}\n\nChoices:\n{option_block}"


def tool_specs() -> List[Dict[str, Any]]:
    return [
        {
            "name": "retrieve",
            "description": "Retrieve concise medical knowledge summaries from the local knowledge server.",
            "arguments": {"query": "str", "topk": "optional int"},
        },
        {
            "name": "biomedparse",
            "description": "Segment a medical target from one image and return bbox, mask image, and overlay image. The text input must be simple and short, ideally a single noun or noun phrase. Do not pass full sentences or long descriptions.",
            "arguments": {"image_index": "int", "description": "short simple noun or noun phrase string"},
        },
        {
            "name": "medsam3",
            "description": "Segment a medical target using MedSAM3 and return bbox, mask image, and overlay image. The text input must be simple and short, ideally a single noun or noun phrase. Do not pass full sentences or long descriptions.",
            "arguments": {
                "image_index": "int",
                "prompt": "short simple noun or noun phrase string",
                "threshold": "optional float",
                "nms_iou": "optional float",
            },
        },
    ]


def extract_first_user_text(trace_detail: Dict[str, Any]) -> str:
    for message in trace_detail.get("messages", []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if item.get("type") == "text":
                    text = normalize_text(item.get("text"))
                    if text:
                        return text
        elif isinstance(content, str):
            text = normalize_text(content)
            if text:
                return text
    return ""


def extract_text_from_content(content: Any) -> str:
    if isinstance(content, str):
        return normalize_text(content)
    if isinstance(content, list):
        text_blocks: List[str] = []
        for item in content:
            if item.get("type") == "text":
                text = normalize_text(item.get("text"))
                if text:
                    text_blocks.append(text)
        return "\n".join(text_blocks).strip()
    return ""


def render_reasoning_action(reasoning: str, action_text: str) -> str:
    reasoning = normalize_text(reasoning)
    action_text = normalize_text(action_text)
    if reasoning:
        return f"<reasoning>\n{reasoning}\n</reasoning>\n{action_text}".strip()
    return action_text


def render_tool_call(name: str, arguments: Dict[str, Any]) -> str:
    payload = {
        "name": normalize_text(name),
        "arguments": arguments or {},
    }
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False) + "\n</tool_call>"


CODE_BLOCK_RE = re.compile(r"<code>\s*```python\s*.*?```\s*</code>", re.S)
TOOL_BLOCK_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)
ANSWER_BLOCK_RE = re.compile(r"<answer>.*?</answer>", re.S)
TOOL_RESPONSE_OPEN_RE = re.compile(r"<tool_response(?:\s+[^>]*)?>", re.S)
TOOL_RESPONSE_CLOSE_RE = re.compile(r"</tool_response>", re.S)
CODE_RESPONSE_OPEN_RE = re.compile(r"<code_response(?:\s+[^>]*)?>", re.S)
CODE_RESPONSE_CLOSE_RE = re.compile(r"</code_response>", re.S)


def extract_code_block(text: str) -> str:
    match = CODE_BLOCK_RE.search(text)
    return match.group(0).strip() if match else ""


def extract_answer_block(text: str) -> str:
    match = ANSWER_BLOCK_RE.search(text)
    return match.group(0).strip() if match else ""


def extract_tool_call_block(text: str) -> str:
    match = TOOL_BLOCK_RE.search(text)
    if not match:
        return ""
    payload_text = normalize_text(match.group(1))
    if not payload_text:
        return ""
    try:
        payload = json.loads(payload_text)
    except Exception:
        return ""
    return render_tool_call(payload.get("name"), payload.get("arguments") or {})


def has_real_code_block(text: str) -> bool:
    return bool(extract_code_block(text))


def has_real_tool_call_block(text: str) -> bool:
    return bool(extract_tool_call_block(text))


def normalize_tool_response_text(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return "<tool_response>\n</tool_response>"
    inner_text = TOOL_RESPONSE_OPEN_RE.sub("", text)
    inner_text = TOOL_RESPONSE_CLOSE_RE.sub("", inner_text)
    inner_text = CODE_RESPONSE_OPEN_RE.sub("", inner_text)
    inner_text = CODE_RESPONSE_CLOSE_RE.sub("", inner_text)
    inner_text = normalize_text(inner_text)
    return f"<tool_response>\n{inner_text}\n</tool_response>"


def normalize_code_response_text(text: str) -> str:
    text = normalize_text(text)
    if not text:
        return "<code_response>\n</code_response>"
    inner_text = CODE_RESPONSE_OPEN_RE.sub("", text)
    inner_text = CODE_RESPONSE_CLOSE_RE.sub("", inner_text)
    inner_text = TOOL_RESPONSE_OPEN_RE.sub("", inner_text)
    inner_text = TOOL_RESPONSE_CLOSE_RE.sub("", inner_text)
    inner_text = normalize_text(inner_text)
    return f"<code_response>\n{inner_text}\n</code_response>"


def looks_like_python_feedback(text: str) -> bool:
    text = normalize_text(text)
    return "Python execution result:" in text


def render_event_action(event: Dict[str, Any]) -> str:
    event_type = normalize_text(event.get("type"))
    if event_type == "tool":
        tool_name = normalize_text(event.get("name"))
        tool_arguments = event.get("arguments") or {}
        return render_tool_call(tool_name, tool_arguments)
    raw_text = normalize_text(event.get("content")) or normalize_text(event.get("action_text"))
    if event_type == "python":
        return extract_code_block(raw_text)
    if event_type == "answer":
        return extract_answer_block(raw_text) or raw_text
    return raw_text


def load_trace_detail(trace1_dir: Path, question_id: int) -> Optional[Dict[str, Any]]:
    path = trace1_dir / f"trace_q{question_id:06d}.json"
    if not path.exists():
        return None
    return read_json(path)


def is_clean_agentic_row(row: Dict[str, Any]) -> bool:
    trace1 = row.get("trace1") or {}
    event_counter = trace1.get("event_counter") or {}
    return (
        bool(trace1.get("correct"))
        and normalize_text(trace1.get("status")) == "ok"
        and int(event_counter.get("format_error", 0)) == 0
    )


def build_direct_sample(
    row: Dict[str, Any],
    vqa_item: Dict[str, Any],
    trace_detail: Optional[Dict[str, Any]],
    system_prompt: str,
) -> Dict[str, Any]:
    prompt = build_question_text(vqa_item)
    answer = normalize_text(vqa_item["vqa_data"]["answer"]).upper()
    return {
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": f"<answer>{answer}</answer>"},
        ],
        "images": [normalize_text(vqa_item["image_path"])],
        "sample_type": "A_direct",
        "question_id": int(row["question_id"]),
    }


def collect_user_messages_after_initial(trace_detail: Dict[str, Any]) -> List[str]:
    user_messages: List[str] = []
    first_user_seen = False
    for message in trace_detail.get("messages", []):
        if message.get("role") != "user":
            continue
        content_text = extract_text_from_content(message.get("content"))
        if not content_text:
            continue
        if not first_user_seen:
            first_user_seen = True
            continue
        if content_text.startswith("Format error."):
            continue
        if content_text.startswith(FORCE_TOOL_REMINDER_PREFIX):
            continue
        user_messages.append(content_text)
    return user_messages


def inventory_map(trace_detail: Dict[str, Any]) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    for item in trace_detail.get("image_inventory", []) or []:
        try:
            image_index = int(item["image_index"])
            image_path = normalize_text(item["path"])
        except Exception:
            continue
        if image_path:
            mapping[image_index] = image_path
    return mapping


def append_image_tokens(text: str, count: int) -> str:
    if count <= 0:
        return text
    image_block = "\n".join(["<image>"] * count)
    if text:
        return f"{text}\n{image_block}"
    return image_block


def build_agentic_messages(
    trace_detail: Dict[str, Any],
    initial_prompt: str,
    initial_image_path: str,
) -> Tuple[List[Dict[str, str]], List[str]]:
    messages_out: List[Dict[str, str]] = [{"role": "user", "content": initial_prompt}]
    image_paths: List[str] = [initial_image_path]
    user_observations = collect_user_messages_after_initial(trace_detail)
    image_by_index = inventory_map(trace_detail)
    observation_index = 0

    for event in trace_detail.get("event_trace", []):
        event_type = normalize_text(event.get("type"))
        if event_type not in {"tool", "python", "answer"}:
            continue
        action_text = render_event_action(event)
        if not action_text:
            continue
        assistant_text = render_reasoning_action(
            reasoning=normalize_text(event.get("reasoning")),
            action_text=action_text,
        )
        messages_out.append({"role": "assistant", "content": assistant_text})
        if event_type != "answer":
            if observation_index < len(user_observations):
                observation_text = user_observations[observation_index]
                new_indices = event.get("new_indices", []) or []
                new_image_paths: List[str] = []
                for image_index in new_indices:
                    try:
                        image_index = int(image_index)
                    except Exception:
                        continue
                    image_path = image_by_index.get(image_index)
                    if image_path:
                        new_image_paths.append(image_path)
                observation_text = append_image_tokens(observation_text, len(new_image_paths))
                if event_type == "tool":
                    observation_text = normalize_tool_response_text(observation_text)
                elif event_type == "python":
                    observation_text = normalize_code_response_text(observation_text)
                messages_out.append({"role": "user", "content": observation_text})
                image_paths.extend(new_image_paths)
                observation_index += 1
    for idx in range(1, len(messages_out)):
        current_message = messages_out[idx]
        previous_message = messages_out[idx - 1]
        if current_message["role"] != "user":
            continue
        if previous_message["role"] != "assistant":
            continue
        current_content = current_message["content"]
        previous_content = previous_message["content"]
        if looks_like_python_feedback(current_content) or "<code_response" in current_content:
            current_message["content"] = normalize_code_response_text(current_content)
        elif "<tool_response" in current_content or has_real_tool_call_block(previous_content):
            current_message["content"] = normalize_tool_response_text(current_content)
        elif has_real_code_block(previous_content):
            current_message["content"] = normalize_code_response_text(current_content)
    return messages_out, image_paths


def build_agentic_sample(
    row: Dict[str, Any],
    vqa_item: Dict[str, Any],
    trace_detail: Dict[str, Any],
    sample_type: str,
    system_prompt: str,
) -> Dict[str, Any]:
    prompt = build_question_text(vqa_item)
    initial_image_path = normalize_text(vqa_item["image_path"])
    messages, image_paths = build_agentic_messages(
        trace_detail=trace_detail,
        initial_prompt=prompt,
        initial_image_path=initial_image_path,
    )
    if not messages:
        raise ValueError(f"empty agentic messages for qid={row['question_id']}")
    messages = [{"role": "system", "content": system_prompt}] + messages
    return {
        "messages": messages,
        "images": image_paths,
        "sample_type": sample_type,
        "question_id": int(row["question_id"]),
    }


def maybe_register_dataset(dataset_info_path: Path, dataset_name: str, file_name: str) -> None:
    dataset_info = read_json(dataset_info_path)
    dataset_info[dataset_name] = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "messages",
            "images": "images",
        },
        "tags": {
            "role_tag": "role",
            "content_tag": "content",
            "user_tag": "user",
            "assistant_tag": "assistant",
            "observation_tag": "observation",
            "function_tag": "function_call",
            "system_tag": "system",
        },
    }
    write_json(dataset_info_path, dataset_info)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export epoch1 multimodal SFT data for LlamaFactory.")
    parser.add_argument("--vqa-path", default=DEFAULT_VQA_PATH)
    parser.add_argument("--trace-split-dir", default=DEFAULT_TRACE_SPLIT_DIR)
    parser.add_argument("--trace1-dir", default=DEFAULT_TRACE1_DIR)
    parser.add_argument("--llamafactory-dir", default=DEFAULT_LLAMAFACTORY_DIR)
    parser.add_argument("--output-name", default=DEFAULT_OUTPUT_NAME)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--total-samples", type=int, default=DEFAULT_TOTAL_SAMPLES)
    parser.add_argument(
        "--a-sample-count",
        type=int,
        default=0,
        help="Total sampled count from bucket A. 0 means auto = total_samples - clean B count.",
    )
    parser.add_argument("--register-dataset-info", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(args.seed)

    vqa_data = read_json(Path(args.vqa_path))
    vqa_map = {idx: item for idx, item in enumerate(vqa_data)}

    trace_split_dir = Path(args.trace_split_dir)
    trace1_dir = Path(args.trace1_dir)
    llamafactory_dir = Path(args.llamafactory_dir)

    file_a = trace_split_dir / "direct_correct_agentic_correct.jsonl"
    file_b = trace_split_dir / "direct_wrong_agentic_correct.jsonl"

    rows_a = read_jsonl(file_a)
    rows_b = read_jsonl(file_b)

    clean_b = [row for row in rows_b if is_clean_agentic_row(row)]
    clean_a_agentic = [row for row in rows_a if is_clean_agentic_row(row)]
    system_prompt = build_system_prompt(tool_specs())

    if args.a_sample_count > 0:
        a_sample_count = min(args.a_sample_count, len(rows_a), len(clean_a_agentic) * 2)
        if a_sample_count % 2 == 1:
            a_sample_count -= 1
    else:
        a_sample_count = max(0, args.total_samples - len(clean_b))
        a_sample_count = min(a_sample_count, len(rows_a), len(clean_a_agentic) * 2)
        if a_sample_count % 2 == 1:
            a_sample_count -= 1

    a_direct_count = a_sample_count // 2
    a_agentic_count = a_sample_count // 2

    sampled_a_agentic = random.sample(clean_a_agentic, a_agentic_count)
    sampled_a_agentic_ids = {int(row["question_id"]) for row in sampled_a_agentic}

    candidate_a_direct = [row for row in rows_a if int(row["question_id"]) not in sampled_a_agentic_ids]
    sampled_a_direct = random.sample(candidate_a_direct, a_direct_count)

    dataset: List[Dict[str, Any]] = []
    stats: Dict[str, Any] = {
        "seed": args.seed,
        "bucket_a_total": len(rows_a),
        "bucket_b_total": len(rows_b),
        "bucket_a_clean_agentic": len(clean_a_agentic),
        "bucket_b_clean_agentic": len(clean_b),
        "target_total_samples": int(args.total_samples),
        "a_sample_count": a_sample_count,
        "a_direct_count": a_direct_count,
        "a_agentic_count": a_agentic_count,
        "b_agentic_count": len(clean_b),
    }

    for row in sampled_a_direct:
        qid = int(row["question_id"])
        vqa_item = vqa_map[qid]
        trace_detail = load_trace_detail(trace1_dir, qid)
        dataset.append(
            build_direct_sample(
                row=row,
                vqa_item=vqa_item,
                trace_detail=trace_detail,
                system_prompt=system_prompt,
            )
        )

    for row in sampled_a_agentic:
        qid = int(row["question_id"])
        vqa_item = vqa_map[qid]
        trace_detail = load_trace_detail(trace1_dir, qid)
        if trace_detail is None:
            continue
        dataset.append(
            build_agentic_sample(
                row=row,
                vqa_item=vqa_item,
                trace_detail=trace_detail,
                sample_type="A_agentic",
                system_prompt=system_prompt,
            )
        )

    for row in clean_b:
        qid = int(row["question_id"])
        vqa_item = vqa_map[qid]
        trace_detail = load_trace_detail(trace1_dir, qid)
        if trace_detail is None:
            continue
        dataset.append(
            build_agentic_sample(
                row=row,
                vqa_item=vqa_item,
                trace_detail=trace_detail,
                sample_type="B_agentic",
                system_prompt=system_prompt,
            )
        )

    random.shuffle(dataset)

    output_json = llamafactory_dir / "data" / f"{args.output_name}.json"
    output_summary = llamafactory_dir / "data" / f"{args.output_name}_summary.json"
    write_json(output_json, dataset)

    sample_type_counter: Dict[str, int] = {}
    for sample in dataset:
        sample_type = normalize_text(sample.get("sample_type"))
        sample_type_counter[sample_type] = sample_type_counter.get(sample_type, 0) + 1

    stats.update(
        {
            "dataset_name": args.output_name,
            "output_json": str(output_json),
            "output_summary": str(output_summary),
            "final_sample_count": len(dataset),
            "sample_type_counter": sample_type_counter,
        }
    )
    write_json(output_summary, stats)

    if args.register_dataset_info:
        maybe_register_dataset(
            dataset_info_path=llamafactory_dir / "data" / "dataset_info.json",
            dataset_name=args.output_name,
            file_name=f"{args.output_name}.json",
        )

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
