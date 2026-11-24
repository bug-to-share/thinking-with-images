from __future__ import annotations

import argparse
import base64
import json
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Iterable, List, Optional

import requests
from PIL import Image

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from loop_utils import (
    answers_match,
    build_question_text,
    extract_answer_text,
    extract_reason_text,
    extract_tool_call,
    normalize_text,
    safe_trace_stem,
    simplify_contents,
    strip_trailing_action_garbage,
    validate_response_format,
)
from judge_client import LLMJudgeClient, judge_base_from_env
from model_canvas import CANVAS_SIZE, prepare_model_canvas
from prompts import FORMAT_ERROR_TEXT, build_tool_feedback_text, build_user_feedback_text
from qwen_tool_bridge import DEFAULT_SAM3_URL, QwenToolBridge, normalize_qwen_arguments_for_execution
from render_trace_html import render_trace_html
from sandbox_client import DEFAULT_SANDBOX_URL, runtime_index_to_python_var
from tool.common import ToolRuntimeContext


DEFAULT_JSONL_PATH = "./data_curation/results/omnimedvqa_filtered/keep_for_rl_random10_seed42.jsonl"
DEFAULT_OUTPUT_DIR = "./gemini_agent_native_bbox/results/qwen_sft_real_trace_smoke"
DEFAULT_BASE_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_ADAPTER_PATH = "./LlamaFactory/saves/qwen3-vl-8b/lora/gemini_trace_qwen_sft"
DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:31001/v1"
DEFAULT_OPENAI_MODEL = "qwen3vl_trace_sft"


SYSTEM_PROMPT_QWEN = """You are a medical visual reasoning agent.

Output protocol:
- Every response must be exactly one of:
  1. `<reason>...</reason>` + one `<tool_call>...</tool_call>`
  2. `<reason>...</reason>` + one `<answer>...</answer>`
- Never output both a tool call and an answer in the same response.
- Never output multiple `<tool_call>` blocks.
- Never write free text outside the required tags.
- Never write `<tool_response>` yourself.
- After `</tool_call>` or `</answer>`, stop immediately.
- Every tool call must use this exact outer JSON shape:
  `{"name":"tool_name","arguments":{...}}`

Python tool rules:
- All available images are already loaded as PIL images: `image_0`, `image_1`, `image_2`, ...
- `image_0` is the original 1000x1000 working canvas.
- New tool-generated images will appear as later variables such as `image_1`.
- Use these variables directly. Do not reload fake file paths.
- In `python_exec`, coordinates are absolute pixel coordinates on the 1000x1000 canvas.
- If you want a new image returned to later turns, render it with matplotlib and call `plt.show()`.

SAM3 rules:
- Always call `sam3` with `text`, `bbox`, `positive_points`, and `negative_points`.
- `text` must be a concrete visible target description, not just a broad label. Prefer simple location + appearance wording such as `fluid around liver`, `dark fluid beside liver`, `right lung mass`, or `bright spot in left kidney`.
- Avoid vague one-word targets for diffuse findings. For example, use `fluid around liver` instead of only `ascites` when the image shows abdominal fluid near the liver.
- Avoid overly technical wording when a plain description is clearer. The target text should be specific enough to guide the mask, but not a long sentence.
- Use Qwen grounding coordinates in JSON arguments:
  - `bbox`: `[x1, y1, x2, y2]`, normalized to `0..1000`
  - `positive_points` / `negative_points`: `[x, y]`, normalized to `0..1000`
- The bbox must be tight around the intended visible target. Do not use a loose box that covers multiple organs or a whole body region.
- Use exactly 3 `positive_points` clearly inside the intended target.
- Use exactly 3 `negative_points` on nearby confusing regions that should be excluded, such as organ tissue, bowel gas, bone, background, or an adjacent non-target structure.
- For fluid or other fuzzy targets, put positive points inside the visible fluid/dark area and negative points on the neighboring organ tissue or non-fluid area.
- If you do not yet have a reliable bbox, positive points, and negative points, use `python_exec` first.

Tool strategy:
- Use tools only when they reduce visual uncertainty.
- After a real tool response, the next `<reason>` must first interpret what the tool returned.
- Tool outputs are evidence, not guaranteed truth. If a result seems implausible, low-confidence, or off-target, say so and do not blindly trust it.
- For SAM3, treat `top_score < 0.80` as weak evidence and `top_score < 0.70` as unreliable evidence unless the mask is plainly correct.
- For the final answer, the preceding `<reason>` should include key visual evidence, relevant medical/anatomical knowledge, and a brief distinction from a plausible alternative when applicable.

Available tools:
{"name":"python_exec","description":"Execute Python code for image inspection, cropping, visualization, contrast adjustment, measurement, and plotting. Use PIL image variables such as image_0 and matplotlib plt.show() to return new images.","arguments":{"code":"python code string"}}
{"name":"sam3","description":"Segment one known visible target in one known region using a concrete target description, tight bbox, exactly 3 positive points, and exactly 3 negative points. Use negative points to exclude nearby confusing tissue or background. The tool returns segmentation score, bbox, mask bbox, and an overlay image.","arguments":{"image_index":"int","text":"concrete visible target description, preferably simple location + appearance wording","bbox":"[x1,y1,x2,y2] normalized 0..1000","positive_points":"exactly 3 [x,y] points inside target, normalized 0..1000","negative_points":"exactly 3 [x,y] points on nearby non-target regions, normalized 0..1000"}}
""".strip()


def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def parse_base_urls(value: Any, default: str) -> List[str]:
    if isinstance(value, str):
        candidates = value.split(",")
    elif isinstance(value, (list, tuple)):
        candidates = value
    else:
        candidates = [default]
    urls = [str(item).strip().rstrip("/") for item in candidates if str(item).strip()]
    return urls or [default.rstrip("/")]


class RoundRobinURLPool:
    def __init__(self, urls: List[str]):
        if not urls:
            raise ValueError("urls must not be empty")
        self.urls = list(urls)
        self.index = 0
        self.lock = Lock()

    def next(self) -> str:
        with self.lock:
            url = self.urls[self.index % len(self.urls)]
            self.index += 1
        return url


@dataclass
class QwenModelClient:
    model: Any
    processor: Any
    device: torch.device

    @classmethod
    def load(cls, base_model: str, adapter_path: str, use_flash_attn: bool = True) -> "QwenModelClient":
        import torch
        from peft import PeftModel
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        kwargs: Dict[str, Any] = {
            "torch_dtype": torch.bfloat16,
            "device_map": "auto",
            "trust_remote_code": True,
        }
        if use_flash_attn:
            kwargs["attn_implementation"] = "flash_attention_2"
        try:
            model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, **kwargs)
        except Exception:
            kwargs.pop("attn_implementation", None)
            model = Qwen3VLForConditionalGeneration.from_pretrained(base_model, **kwargs)
        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()
        processor = AutoProcessor.from_pretrained(adapter_path, trust_remote_code=True)
        device = next(model.parameters()).device
        return cls(model=model, processor=processor, device=device)

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        prompt = self.processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        images = collect_images(messages)
        inputs = self.processor(text=[prompt], images=images, return_tensors="pt")
        inputs = inputs.to(self.device)
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": max_new_tokens,
            "do_sample": temperature > 0,
            "temperature": temperature if temperature > 0 else None,
            "top_p": top_p,
            "top_k": top_k,
            "pad_token_id": self.processor.tokenizer.eos_token_id,
        }
        generation_kwargs = {k: v for k, v in generation_kwargs.items() if v is not None}
        import torch

        with torch.inference_mode():
            output_ids = self.model.generate(**inputs, **generation_kwargs)
        generated_ids = output_ids[:, inputs["input_ids"].shape[1]:]
        text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return strip_trailing_action_garbage(text)


@dataclass
class OpenAIQwenClient:
    base_url: str
    api_key: str
    model_name: str
    timeout: float = 240.0

    def generate(
        self,
        messages: List[Dict[str, Any]],
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> str:
        payload: Dict[str, Any] = {
            "model": self.model_name,
            "messages": openai_messages_from_qwen(messages),
            "max_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        if top_k > 0:
            payload["top_k"] = top_k
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        data = response.json()
        choice = data["choices"][0]
        text = choice.get("message", {}).get("content") or choice.get("text") or ""
        return strip_trailing_action_garbage(text)


def collect_images(messages: List[Dict[str, Any]]) -> List[Image.Image]:
    images: List[Image.Image] = []
    for message in messages:
        for part in message.get("content", []) or []:
            if isinstance(part, dict) and part.get("type") == "image":
                images.append(Image.open(part["image"]).convert("RGB"))
    return images


def image_to_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime_type = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".webp":
        mime_type = "image/webp"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def openai_messages_from_qwen(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    converted: List[Dict[str, Any]] = []
    for message in messages:
        content: List[Dict[str, Any]] = []
        for part in message.get("content", []) or []:
            if part.get("type") == "text":
                content.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image":
                content.append({"type": "image_url", "image_url": {"url": image_to_data_url(part["image"])}})
        converted.append({"role": message.get("role", "user"), "content": content or [{"type": "text", "text": ""}]})
    return converted


def qwen_image_message(image_path: str, text: str) -> Dict[str, Any]:
    return {"role": "user", "content": [{"type": "image", "image": image_path}, {"type": "text", "text": text}]}


def qwen_text_message(role: str, text: str) -> Dict[str, Any]:
    return {"role": role, "content": [{"type": "text", "text": text}]}


def messages_for_log(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for message in messages:
        row = {"role": message.get("role"), "parts": []}
        for part in message.get("content", []) or []:
            if part.get("type") == "text":
                row["parts"].append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image":
                row["parts"].append({"type": "image", "path": part.get("image")})
        rows.append(row)
    return rows


def runtime_inventory(runtime: ToolRuntimeContext) -> List[Dict[str, Any]]:
    return [{"image_index": int(record.index), "path": str(record.path)} for record in runtime.image_records]


def new_image_descriptions(indices: Iterable[int]) -> List[str]:
    return [
        f"runtime image_index={idx} is available as python variable `{runtime_index_to_python_var(int(idx))}`"
        for idx in indices
    ]


def content_text_with_images(text: str, image_paths: List[str]) -> Dict[str, Any]:
    content: List[Dict[str, Any]] = [{"type": "text", "text": text}]
    for image_path in image_paths:
        content.append({"type": "image", "image": image_path})
    return {"role": "user", "content": content}


def compact_tool_counter(trace: List[Dict[str, Any]]) -> Dict[str, int]:
    counter = Counter()
    for event in trace:
        if event.get("type") == "tool":
            counter[event["name"]] += 1
    return dict(counter)


def build_judge_client(args: argparse.Namespace) -> Optional[LLMJudgeClient]:
    if not getattr(args, "use_llm_judge", False):
        return None
    base_url = normalize_text(getattr(args, "judge_base_url", "")) or judge_base_from_env()
    if not base_url:
        return None
    return LLMJudgeClient(
        base_url=base_url,
        api_key=normalize_text(getattr(args, "judge_api_key", "")) or "EMPTY",
        timeout=float(getattr(args, "judge_timeout", 60.0)),
    )


def build_failed_record(question_id: str, item: Dict[str, Any], args: argparse.Namespace, error: Exception) -> Dict[str, Any]:
    return {
        "question_id": question_id,
        "image_path": str(Path(normalize_text(item.get("image_path_abs"))).resolve()),
        "question": normalize_text(item.get("question")),
        "gold_answer": normalize_text(item.get("gt_answer")),
        "pred_answer": "",
        "correct": False,
        "status": "failed",
        "event_trace": [{"type": "exception", "error": str(error)}],
        "messages": [{"role": "system", "parts": [{"type": "text", "text": SYSTEM_PROMPT_QWEN}]}],
        "tool_call_counter": {},
        "tool_calls_count": 0,
        "model": args.model_path,
        "adapter_path": args.adapter_path,
        "system_prompt": SYSTEM_PROMPT_QWEN,
    }


def extract_coordinate_texts(text: str) -> Dict[str, List[str]]:
    number = r"-?\d+(?:\.\d+)?"
    return {
        "boxes_4d": re.findall(rf"\[\s*{number}\s*,\s*{number}\s*,\s*{number}\s*,\s*{number}\s*\]", text or ""),
        "points_2d": re.findall(rf"\[\s*{number}\s*,\s*{number}\s*\]", text or ""),
    }


def run_single_item(
    client: QwenModelClient,
    judge_client: Optional[LLMJudgeClient],
    args: argparse.Namespace,
    item: Dict[str, Any],
    output_dir: Path,
    sam3_pool: RoundRobinURLPool,
) -> Dict[str, Any]:
    question_id = normalize_text(item.get("question_id"))
    original_image_path = str(Path(normalize_text(item.get("image_path_abs"))).resolve())
    canvas_dir = output_dir / "_model_canvas"
    canvas_path = canvas_dir / f"{safe_trace_stem(question_id)}.png"
    canvas_info = prepare_model_canvas(original_image_path, canvas_path, canvas_size=CANVAS_SIZE)

    runtime = ToolRuntimeContext()
    runtime.reset(image_path=canvas_info.image_path, sample_id=question_id or "unknown")

    tool_bridge = QwenToolBridge(
        runtime=runtime,
        sam3_url=sam3_pool.next(),
        sandbox_url=args.sandbox_url,
        request_timeout=args.request_timeout,
    )

    messages: List[Dict[str, Any]] = [
        qwen_text_message("system", SYSTEM_PROMPT_QWEN),
        qwen_image_message(canvas_info.image_path, build_question_text(item)),
    ]
    event_trace: List[Dict[str, Any]] = []
    final_answer_text: Optional[str] = None
    allowed_tool_names = {"python_exec", "sam3"}

    for round_idx in range(args.max_rounds):
        response_text = client.generate(
            messages=messages,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
        )
        reason = normalize_text(extract_reason_text(response_text))
        format_error = validate_response_format(response_text)
        messages.append(qwen_text_message("assistant", response_text))

        if format_error:
            event_trace.append(
                {
                    "type": "format_error",
                    "round": round_idx,
                    "reason": reason,
                    "response_text": response_text,
                    "error": format_error,
                }
            )
            messages.append(qwen_text_message("user", build_user_feedback_text(FORMAT_ERROR_TEXT)))
            continue

        answer_text = extract_answer_text(response_text)
        if answer_text is not None:
            final_answer_text = answer_text
            event_trace.append(
                {
                    "type": "answer",
                    "round": round_idx,
                    "reason": reason,
                    "response_text": response_text,
                    "answer": answer_text,
                }
            )
            break

        tool_call = extract_tool_call(response_text)
        if tool_call is None:
            event_trace.append(
                {
                    "type": "format_error",
                    "round": round_idx,
                    "reason": reason,
                    "response_text": response_text,
                    "error": "unable to parse tool_call",
                }
            )
            messages.append(qwen_text_message("user", build_user_feedback_text(FORMAT_ERROR_TEXT)))
            continue

        tool_name = normalize_text(tool_call.get("name"))
        if tool_name == "sandbox":
            tool_name = "python_exec"
        raw_arguments = tool_call.get("arguments") or {}
        coordinate_matches = extract_coordinate_texts(response_text)

        try:
            image_index = int(raw_arguments.get("image_index", 0) or 0)
        except Exception:
            image_index = 0
        arguments = raw_arguments
        if tool_name == "sam3":
            image_size = runtime.get_image(image_index).size
            arguments = normalize_qwen_arguments_for_execution(raw_arguments, image_size)

        if tool_name not in allowed_tool_names:
            error_text = f'<tool_response name="{tool_name}">error: unsupported tool in current configuration</tool_response>'
            messages.append(qwen_text_message("user", error_text))
            event_trace.append(
                {
                    "type": "tool_error",
                    "round": round_idx,
                    "name": tool_name,
                    "arguments": raw_arguments,
                    "normalized_arguments": arguments,
                    "coordinate_matches": coordinate_matches,
                    "error": "unsupported tool in current configuration",
                    "base_url": None,
                    "reason": reason,
                    "response_text": response_text,
                }
            )
            continue

        selected_base_url = None
        try:
            if tool_name == "sam3":
                selected_base_url = sam3_pool.next()
                tool_bridge.sam3_url = selected_base_url
            result = tool_bridge.run(tool_name, arguments)
            image_paths = [runtime.get_image_path(idx) for idx in result.new_indices]
            feedback_text = build_tool_feedback_text(
                tool_name=tool_name,
                payload=result.model_payload,
                image_descriptions=new_image_descriptions(result.new_indices),
            )
            messages.append(content_text_with_images(feedback_text, image_paths))
            event_trace.append(
                {
                    "type": "tool",
                    "round": round_idx,
                    "name": tool_name,
                    "arguments": raw_arguments,
                    "normalized_arguments": arguments,
                    "coordinate_matches": coordinate_matches,
                    "result": result.model_payload,
                    "base_url": selected_base_url,
                    "new_indices": result.new_indices,
                    "reason": reason,
                    "response_text": response_text,
                }
            )
        except Exception as exc:
            error_text = f'<tool_response name="{tool_name}">error: {exc}</tool_response>'
            messages.append(qwen_text_message("user", error_text))
            event_trace.append(
                {
                    "type": "tool_error",
                    "round": round_idx,
                    "name": tool_name,
                    "arguments": raw_arguments,
                    "normalized_arguments": arguments,
                    "coordinate_matches": coordinate_matches,
                    "error": str(exc),
                    "base_url": selected_base_url,
                    "reason": reason,
                    "response_text": response_text,
                }
            )

    pred_answer = normalize_text(final_answer_text)
    gold_answer = normalize_text(item.get("gt_answer"))
    trace_counter = compact_tool_counter(event_trace)
    correct_method = "exact_match"
    judge_payload: Optional[Dict[str, Any]] = None
    is_correct = answers_match(pred_answer, gold_answer)
    if pred_answer and not is_correct and judge_client is not None:
        try:
            judge_result = judge_client.judge_semantic_equivalence(
                question=normalize_text(item.get("question")),
                prediction=pred_answer,
                gold=gold_answer,
            )
            is_correct = judge_result.correct
            correct_method = "llm_judge"
            judge_payload = {
                "judge_model": judge_result.model_name,
                "judge_raw_text": judge_result.raw_text,
                "judge_base_url": judge_client.base_url,
            }
        except Exception as exc:
            correct_method = "judge_failed"
            judge_payload = {
                "judge_error": str(exc),
                "judge_base_url": getattr(judge_client, "base_url", ""),
            }
    return {
        "question_id": question_id,
        "image_path": original_image_path,
        "model_image_path": canvas_info.image_path,
        "question": normalize_text(item.get("question")),
        "choices": item.get("choices"),
        "gold_letter": normalize_text(item.get("gold_letter")),
        "gold_answer": gold_answer,
        "pred_answer": pred_answer,
        "correct": is_correct,
        "correctness_method": correct_method,
        "judge_result": judge_payload,
        "status": "ok" if pred_answer else "no_answer",
        "benchmark": normalize_text(item.get("benchmark")),
        "split": normalize_text(item.get("split")),
        "modality": normalize_text(item.get("modality")),
        "location": normalize_text(item.get("location")),
        "content_type": normalize_text(item.get("content_type")),
        "final_answer_text": final_answer_text,
        "tool_call_counter": trace_counter,
        "tool_calls_count": sum(trace_counter.values()),
        "messages": messages_for_log(messages),
        "event_trace": event_trace,
        "image_inventory": runtime_inventory(runtime),
        "model_canvas": canvas_info.as_dict(),
        "model": args.model_path,
        "adapter_path": args.adapter_path,
        "system_prompt": SYSTEM_PROMPT_QWEN,
        "jsonl_path": args.jsonl_path,
        "tool_server_urls": {
            "sam3": args.sam3_url,
            "sandbox": args.sandbox_url,
        },
    }


def write_trace_artifacts(record: Dict[str, Any], output_dir: Path) -> None:
    trace_stem = safe_trace_stem(record.get("question_id"))
    json_path = output_dir / f"{trace_stem}.json"
    html_path = output_dir / f"{trace_stem}.html"
    json_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    render_trace_html(record, html_path)


def render_index(records: List[Dict[str, Any]], output_dir: Path) -> None:
    rows = []
    for record in records:
        stem = safe_trace_stem(record.get("question_id"))
        rows.append(
            "<tr>"
            f"<td><a href='{stem}.html'>{record.get('question_id')}</a></td>"
            f"<td>{record.get('status')}</td>"
            f"<td>{record.get('correct')}</td>"
            f"<td>{record.get('tool_calls_count')}</td>"
            f"<td>{record.get('pred_answer')}</td>"
            f"<td>{record.get('gold_answer')}</td>"
            f"<td>{record.get('question')}</td>"
            "</tr>"
        )
    html = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Qwen SFT Trace Index</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:24px;background:#f8fafc;color:#111827}
table{border-collapse:collapse;width:100%;background:white}th,td{border:1px solid #e5e7eb;padding:8px;vertical-align:top}th{background:#f3f4f6}
a{color:#2563eb}
</style></head><body>
<h1>Qwen SFT Real Trace Index</h1>
<table><thead><tr><th>question_id</th><th>status</th><th>correct</th><th>tools</th><th>pred</th><th>gold</th><th>question</th></tr></thead>
<tbody>
""" + "\n".join(rows) + """
</tbody></table></body></html>
"""
    (output_dir / "index.html").write_text(html, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run trained Qwen3-VL LoRA as a tool-using trace generator on real data.")
    parser.add_argument("--jsonl-path", default=DEFAULT_JSONL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--backend", choices=["hf", "openai"], default=os.getenv("QWEN_TRACE_BACKEND", "hf"))
    parser.add_argument("--model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--adapter-path", default=DEFAULT_ADAPTER_PATH)
    parser.add_argument("--openai-base-url", default=os.getenv("QWEN_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL))
    parser.add_argument("--openai-api-key", default=os.getenv("QWEN_OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--openai-model", default=os.getenv("QWEN_OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    parser.add_argument("--max-items", type=int, default=3)
    parser.add_argument("--item-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--max-rounds", type=int, default=6)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--sam3-url", default=os.getenv("SAM3_URL", DEFAULT_SAM3_URL))
    parser.add_argument("--sandbox-url", default=os.getenv("MED_SANDBOX_URL", DEFAULT_SANDBOX_URL))
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--judge-base-url", default=os.getenv("LLM_AS_A_JUDGE_BASE", ""))
    parser.add_argument("--judge-api-key", default=os.getenv("LLM_AS_A_JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    parser.add_argument("--disable-flash-attn", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    record_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    if record_path.exists():
        record_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    data = read_jsonl(Path(args.jsonl_path))
    if args.max_items > 0:
        data = data[args.item_index: args.item_index + args.max_items]
    else:
        data = data[args.item_index:]

    if args.backend == "openai":
        client = OpenAIQwenClient(
            base_url=args.openai_base_url,
            api_key=args.openai_api_key,
            model_name=args.openai_model,
            timeout=args.request_timeout,
        )
    else:
        client = QwenModelClient.load(
            base_model=args.model_path,
            adapter_path=args.adapter_path,
            use_flash_attn=not args.disable_flash_attn,
        )
    judge_client = build_judge_client(args)
    sam3_pool = RoundRobinURLPool(parse_base_urls(args.sam3_url, DEFAULT_SAM3_URL))

    def process_item(item: Dict[str, Any]) -> Dict[str, Any]:
        qid = normalize_text(item.get("question_id"))
        try:
            return run_single_item(
                client=client,
                judge_client=judge_client,
                args=args,
                item=item,
                output_dir=output_dir,
                sam3_pool=sam3_pool,
            )
        except Exception as exc:
            return build_failed_record(qid, item, args, exc)

    records: List[Dict[str, Any]] = []
    if args.num_workers > 1:
        with ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            futures = [executor.submit(process_item, item) for item in data]
            for future in as_completed(futures):
                record = future.result()
                records.append(record)
                append_jsonl(record_path, record)
                write_trace_artifacts(record, output_dir)
                print(json.dumps({"question_id": record.get("question_id"), "status": record.get("status"), "pred": record.get("pred_answer"), "tools": record.get("tool_call_counter")}, ensure_ascii=False), flush=True)
    else:
        for item in data:
            record = process_item(item)
            records.append(record)
            append_jsonl(record_path, record)
            write_trace_artifacts(record, output_dir)
            print(json.dumps({"question_id": record.get("question_id"), "status": record.get("status"), "pred": record.get("pred_answer"), "tools": record.get("tool_call_counter")}, ensure_ascii=False), flush=True)

    write_jsonl(record_path, records)
    render_index(records, output_dir)
    tool_counter = Counter()
    for record in records:
        tool_counter.update(record.get("tool_call_counter", {}))
    summary = {
        "attempted": len(records),
        "correct_exact": sum(1 for record in records if record.get("correct")),
        "no_answer": sum(1 for record in records if record.get("status") == "no_answer"),
        "failed": sum(1 for record in records if record.get("status") == "failed"),
        "format_errors": sum(1 for record in records for event in record.get("event_trace", []) if event.get("type") == "format_error"),
        "tool_errors": sum(1 for record in records for event in record.get("event_trace", []) if event.get("type") == "tool_error"),
        "tool_call_counter": dict(tool_counter),
        "jsonl_path": args.jsonl_path,
        "model_path": args.model_path,
        "adapter_path": args.adapter_path,
        "backend": args.backend,
        "openai_base_url": args.openai_base_url,
        "openai_model": args.openai_model,
        "output_dir": str(output_dir),
        "sam3_url": args.sam3_url,
        "sandbox_url": args.sandbox_url,
        "use_llm_judge": args.use_llm_judge,
        "judge_base_url": args.judge_base_url,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
