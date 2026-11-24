from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from datasets import Dataset, Image, Sequence

TEXT_COT_SYSTEM_PROMPT = """You are a medical visual question answering assistant.

Inspect the attached image and reason from visible evidence and relevant medical knowledge.
Answer in the same language as the question. Your response must contain exactly two blocks:
<reason>concise step-by-step visual and medical reasoning</reason>
<answer>short final answer only</answer>

Do not add any text outside these two blocks. Do not put reasoning in the answer block.
""".strip()


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def normalize_text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def user_prompt(question: str) -> str:
    return f"<image>\nQuestion: {question}"


def convert_slake_split(
    slake_root: Path,
    split: str,
    languages: set[str],
) -> list[dict[str, Any]]:
    source_path = slake_root / f"{split}.json"
    rows = json.loads(source_path.read_text(encoding="utf-8"))
    converted: list[dict[str, Any]] = []

    for source_index, row in enumerate(rows):
        language = normalize_text(row.get("q_lang")).lower()
        if languages and language not in languages:
            continue

        image_rel = normalize_text(row.get("img_name"))
        image_path = slake_root / "imgs" / image_rel
        if not image_path.is_file():
            raise FileNotFoundError(f"SLAKE image not found: {image_path}")

        try:
            portable_image_path = image_path.relative_to(PROJECT_ROOT).as_posix()
        except ValueError:
            portable_image_path = str(image_path.resolve())

        question = normalize_text(row.get("question"))
        answer = normalize_text(row.get("answer"))
        converted.append(
            {
                "data_source": "slake_text_cot",
                "prompt": [
                    {"role": "system", "content": TEXT_COT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt(question)},
                ],
                # The Image feature keeps paths in parquet and decodes them lazily.
                "images": [portable_image_path],
                "ability": "medical_vqa",
                "reward_model": {"style": "rule", "ground_truth": answer},
                "extra_info": {
                    "index": len(converted),
                    "source_index": source_index,
                    "split": split,
                    "benchmark": "SLAKE",
                    "qid": row.get("qid"),
                    "question": question,
                    "answer": answer,
                    "q_lang": language,
                    "answer_type": row.get("answer_type"),
                    "content_type": row.get("content_type"),
                    "img_name": image_rel,
                    "image_path": portable_image_path,
                },
            }
        )

    return converted


def write_parquet(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    dataset = Dataset.from_list(rows)
    dataset = dataset.cast_column("images", Sequence(Image(decode=True)))
    dataset.to_parquet(str(output_path))


def parse_languages(raw: str) -> set[str]:
    languages = {item.strip().lower() for item in raw.split(",") if item.strip()}
    return set() if not languages or "all" in languages else languages


def split_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "size": len(rows),
        "languages": dict(sorted(Counter(row["extra_info"]["q_lang"] for row in rows).items())),
        "answer_types": dict(sorted(Counter(row["extra_info"]["answer_type"] for row in rows).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the SLAKE single-turn text-CoT GRPO dataset for VERL.")
    parser.add_argument(
        "--slake-root",
        type=Path,
        default=PROJECT_ROOT / "eval/benchmarks/SLAKE",
    )
    parser.add_argument(
        "--train-output",
        type=Path,
        default=PROJECT_ROOT / "data/med_twi/text_cot/train.parquet",
    )
    parser.add_argument(
        "--val-output",
        type=Path,
        default=PROJECT_ROOT / "eval/benchmarks/SLAKE/views/slake_validation/validation.parquet",
    )
    parser.add_argument("--languages", default="all", help="Comma-separated SLAKE languages, or `all`.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-val-samples", type=int, default=0)
    args = parser.parse_args()

    languages = parse_languages(args.languages)
    train_rows = convert_slake_split(args.slake_root, "train", languages)
    val_rows = convert_slake_split(args.slake_root, "validation", languages)

    random.Random(args.seed).shuffle(train_rows)
    for index, row in enumerate(train_rows):
        row["extra_info"]["index"] = index

    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]
    if args.max_val_samples > 0:
        val_rows = val_rows[: args.max_val_samples]

    write_parquet(train_rows, args.train_output)
    write_parquet(val_rows, args.val_output)

    summary = {
        "dataset": "SLAKE",
        "task": "single_turn_text_cot_grpo",
        "languages": "all" if not languages else sorted(languages),
        "train": split_summary(train_rows),
        "validation": split_summary(val_rows),
        "files": {
            "train": str(args.train_output),
            "validation": str(args.val_output),
        },
        "test_policy": "SLAKE test is excluded from preprocessing and reserved for final MedEvalKit evaluation.",
    }
    summary_path = args.train_output.parent / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
