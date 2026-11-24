from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import requests

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from judge_client import LLMJudgeClient, judge_base_from_env
from loop_utils import answers_match, build_question_text, normalize_text, safe_trace_stem
from model_canvas import CANVAS_SIZE, prepare_model_canvas
from render_trace_html import render_trace_html


DEFAULT_JSONL_PATH = "./data_curation/results/omnimedvqa_filtered/keep_for_rl_random100_seed20260503.jsonl"
DEFAULT_OUTPUT_DIR = "./gemini_agent_native_bbox/results/qwen_sft_direct_eval"
DEFAULT_OPENAI_BASE_URL = "http://127.0.0.1:31001/v1"
DEFAULT_OPENAI_MODEL = "qwen3vl_trace_sft"


SYSTEM_PROMPT_DIRECT = """You are a medical visual question answering assistant.

Answer the question directly from the image.
If choices are provided, choose the best option and put the selected option text in the answer.
Do not call tools.
Do not mention hidden labels, reference answers, gold answers, or evaluation.
Return exactly:
<answer>your answer</answer>
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


def image_to_data_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    mime_type = "image/png"
    if suffix in {".jpg", ".jpeg"}:
        mime_type = "image/jpeg"
    elif suffix == ".webp":
        mime_type = "image/webp"
    encoded = base64.b64encode(Path(path).read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def extract_answer_text(text: str) -> str:
    start = text.find("<answer>")
    end = text.find("</answer>")
    if start >= 0 and end > start:
        return text[start + len("<answer>") : end].strip()
    return text.strip()


@dataclass
class OpenAIClient:
    base_url: str
    api_key: str
    model_name: str
    timeout: float

    def generate(self, image_path: str, question_text: str, max_tokens: int, temperature: float, top_p: float) -> str:
        payload = {
            "model": self.model_name,
            "messages": [
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT_DIRECT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                        {"type": "text", "text": question_text},
                    ],
                },
            ],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "extra_body": {"chat_template_kwargs": {"enable_thinking": False}},
        }
        headers = {"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"}
        response = requests.post(
            f"{self.base_url.rstrip('/')}/chat/completions",
            headers=headers,
            json=payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        return normalize_text(response.json()["choices"][0]["message"]["content"])


def build_judge_client(args: argparse.Namespace) -> Optional[LLMJudgeClient]:
    if not args.use_llm_judge:
        return None
    base_url = normalize_text(args.judge_base_url) or judge_base_from_env()
    if not base_url:
        return None
    return LLMJudgeClient(base_url=base_url, api_key=args.judge_api_key, timeout=args.judge_timeout)


def evaluate_one(
    item: Dict[str, Any],
    args: argparse.Namespace,
    client: OpenAIClient,
    judge_client: Optional[LLMJudgeClient],
    output_dir: Path,
) -> Dict[str, Any]:
    qid = normalize_text(item.get("question_id"))
    original_image_path = str(Path(normalize_text(item.get("image_path_abs"))).resolve())
    canvas_dir = output_dir / "_model_canvas"
    canvas_path = canvas_dir / f"{safe_trace_stem(qid)}.png"
    canvas_info = prepare_model_canvas(original_image_path, canvas_path, canvas_size=CANVAS_SIZE)
    question_text = build_question_text(item)
    response_text = client.generate(
        image_path=canvas_info.image_path,
        question_text=question_text,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
    )
    pred = extract_answer_text(response_text)
    gold = normalize_text(item.get("gt_answer"))
    is_correct = answers_match(pred, gold)
    correctness_method = "exact_match"
    judge_payload = None
    if pred and not is_correct and judge_client is not None:
        judge = judge_client.judge_semantic_equivalence(
            question=normalize_text(item.get("question")),
            prediction=pred,
            gold=gold,
        )
        is_correct = judge.correct
        correctness_method = "llm_judge"
        judge_payload = {
            "judge_model": judge.model_name,
            "judge_raw_text": judge.raw_text,
            "judge_base_url": judge_client.base_url,
        }

    return {
        "question_id": qid,
        "image_path": original_image_path,
        "model_image_path": canvas_info.image_path,
        "question": normalize_text(item.get("question")),
        "choices": item.get("choices"),
        "gold_letter": normalize_text(item.get("gold_letter")),
        "gold_answer": gold,
        "pred_answer": pred,
        "correct": is_correct,
        "correctness_method": correctness_method,
        "judge_result": judge_payload,
        "status": "ok" if pred else "no_answer",
        "response_text": response_text,
        "messages": [
            {"role": "system", "parts": [{"type": "text", "text": SYSTEM_PROMPT_DIRECT}]},
            {"role": "user", "parts": [{"type": "image", "path": canvas_info.image_path}, {"type": "text", "text": question_text}]},
            {"role": "assistant", "parts": [{"type": "text", "text": response_text}]},
        ],
        "event_trace": [
            {
                "type": "answer",
                "round": 0,
                "reason": "",
                "response_text": response_text,
                "answer": pred,
            }
        ],
        "image_inventory": [{"image_index": 0, "path": canvas_info.image_path}],
        "model_canvas": canvas_info.as_dict(),
        "model": args.openai_model,
        "backend": "direct_openai",
        "system_prompt": SYSTEM_PROMPT_DIRECT,
        "jsonl_path": args.jsonl_path,
    }


def write_trace_artifacts(record: Dict[str, Any], output_dir: Path) -> None:
    stem = safe_trace_stem(record.get("question_id"))
    (output_dir / f"{stem}.json").write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
    render_trace_html(record, output_dir / f"{stem}.html")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Direct VQA baseline for Qwen SFT model served by SGLang.")
    parser.add_argument("--jsonl-path", default=DEFAULT_JSONL_PATH)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--openai-base-url", default=os.getenv("QWEN_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL))
    parser.add_argument("--openai-api-key", default=os.getenv("QWEN_OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--openai-model", default=os.getenv("QWEN_OPENAI_MODEL", DEFAULT_OPENAI_MODEL))
    parser.add_argument("--max-items", type=int, default=100)
    parser.add_argument("--item-index", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=24)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--request-timeout", type=float, default=240.0)
    parser.add_argument("--use-llm-judge", action="store_true")
    parser.add_argument("--judge-base-url", default=os.getenv("LLM_AS_A_JUDGE_BASE", ""))
    parser.add_argument("--judge-api-key", default=os.getenv("LLM_AS_A_JUDGE_API_KEY", "EMPTY"))
    parser.add_argument("--judge-timeout", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = output_dir / "predictions.jsonl"
    summary_path = output_dir / "summary.json"
    if prediction_path.exists():
        prediction_path.unlink()
    if summary_path.exists():
        summary_path.unlink()

    data = read_jsonl(Path(args.jsonl_path))
    data = data[args.item_index : args.item_index + args.max_items] if args.max_items > 0 else data[args.item_index :]
    client = OpenAIClient(
        base_url=args.openai_base_url,
        api_key=args.openai_api_key,
        model_name=args.openai_model,
        timeout=args.request_timeout,
    )
    judge_client = build_judge_client(args)

    records: List[Dict[str, Any]] = []

    def process(item: Dict[str, Any]) -> Dict[str, Any]:
        try:
            return evaluate_one(item, args, client, judge_client, output_dir)
        except Exception as exc:
            return {
                "question_id": normalize_text(item.get("question_id")),
                "image_path": normalize_text(item.get("image_path_abs")),
                "question": normalize_text(item.get("question")),
                "gold_answer": normalize_text(item.get("gt_answer")),
                "pred_answer": "",
                "correct": False,
                "status": "failed",
                "event_trace": [{"type": "exception", "error": str(exc)}],
                "model": args.openai_model,
                "backend": "direct_openai",
            }

    with ThreadPoolExecutor(max_workers=max(1, args.num_workers)) as executor:
        futures = [executor.submit(process, item) for item in data]
        for future in as_completed(futures):
            record = future.result()
            records.append(record)
            append_jsonl(prediction_path, record)
            write_trace_artifacts(record, output_dir)
            print(
                json.dumps(
                    {
                        "question_id": record.get("question_id"),
                        "status": record.get("status"),
                        "correct": record.get("correct"),
                        "pred": record.get("pred_answer"),
                    },
                    ensure_ascii=False,
                ),
                flush=True,
            )

    summary = {
        "attempted": len(records),
        "correct": sum(1 for record in records if record.get("correct")),
        "no_answer": sum(1 for record in records if record.get("status") == "no_answer"),
        "failed": sum(1 for record in records if record.get("status") == "failed"),
        "jsonl_path": args.jsonl_path,
        "output_dir": str(output_dir),
        "backend": "direct_openai",
        "openai_base_url": args.openai_base_url,
        "openai_model": args.openai_model,
        "use_llm_judge": args.use_llm_judge,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
