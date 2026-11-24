from __future__ import annotations

import argparse
import io
import json
import random
from pathlib import Path
from typing import Any, Iterable

from datasets import Dataset, DatasetDict, concatenate_datasets, load_dataset

SYSTEM_PROMPT = """You are a medical visual reasoning assistant.

Inspect the attached image and answer the question using visual evidence and
relevant medical knowledge. Answer in the same language as the question.
""".strip()


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return " ".join(normalize_text(item) for item in value)
    return str(value).strip()


def read_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix == ".jsonl":
        rows = []
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line:
                rows.append(json.loads(line))
        return rows
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_rgb_bytes(image: Any) -> bytes:
    buffer = io.BytesIO()
    converted = image.convert("RGB") if hasattr(image, "convert") else image
    converted.save(buffer, format="JPEG")
    return buffer.getvalue()


def build_open_user_prompt(question: str) -> str:
    return "\n".join(
        [
            "<image>",
            "Question:",
            question,
            "",
            "Return the final answer inside `<answer>...</answer>`.",
        ]
    )


def answer_letter(row: dict[str, Any]) -> str:
    return normalize_text(row.get("vqa_data", {}).get("answer")).upper().replace(".", "")


def answer_text(row: dict[str, Any]) -> str:
    letter = answer_letter(row)
    options = [normalize_text(option) for option in row.get("vqa_data", {}).get("options", [])]
    index = ord(letter) - ord("A")
    if 0 <= index < len(options):
        option = options[index]
        return option.split(".", 1)[-1].strip() if "." in option else option
    return letter


def build_mcq_user_prompt(row: dict[str, Any]) -> str:
    question = normalize_text(row.get("vqa_data", {}).get("question"))
    options = [normalize_text(option) for option in row.get("vqa_data", {}).get("options", [])]
    prompt = ["<image>", "Question:", question]
    if options:
        prompt.extend(["", "Choices:"])
        prompt.extend(options)
    prompt.extend(["", "Return the final answer as `<answer>A</answer>` or the correct option letter."])
    return "\n".join(prompt)


def convert_existing_vqa_row(row: dict[str, Any], index: int) -> dict[str, Any] | None:
    image_path = Path(normalize_text(row.get("image_path")))
    if not image_path.exists():
        return None

    prompt = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_mcq_user_prompt(row)},
    ]
    options = [normalize_text(option) for option in row.get("vqa_data", {}).get("options", [])]
    ground_truth = {
        "letter": answer_letter(row),
        "text": answer_text(row),
        "options": options,
    }
    return {
        "data_source": normalize_text(row.get("source")) or "med_twi_vqa",
        "prompt": prompt,
        "images": [{"bytes": image_path.read_bytes(), "path": str(image_path)}],
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "index": index,
            "question_id": normalize_text(row.get("question_id", index)),
            "question": normalize_text(row.get("vqa_data", {}).get("question")),
            "answer_text": ground_truth["text"],
            "answer_letter": ground_truth["letter"],
            "image_path": str(image_path),
            "source": normalize_text(row.get("source")),
        },
        "agent_name": "tool_agent",
    }


def base_prompt(question: str) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_open_user_prompt(question)},
    ]


def make_example(
    *,
    source: str,
    split: str,
    question_id: str | int,
    question: str,
    answer: str,
    image_bytes: bytes,
    image_path: str,
    extra_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    info = {
        "question_id": normalize_text(question_id),
        "question": normalize_text(question),
        "answer_text": normalize_text(answer),
        "image_path": image_path,
        "source": source,
        "split": split,
    }
    if extra_info:
        info.update(extra_info)
    return {
        "data_source": source,
        "prompt": base_prompt(normalize_text(question)),
        "images": [{"bytes": image_bytes, "path": image_path}],
        "reward_model": {"style": "rule", "ground_truth": normalize_text(answer)},
        "extra_info": info,
        "agent_name": "tool_agent",
    }


def iter_slake_records(
    dataset_root: Path,
    split: str,
    languages: set[str],
) -> Iterable[dict[str, Any]]:
    slake_root = dataset_root / "SLAKE"
    rows = json.loads((slake_root / f"{split}.json").read_text(encoding="utf-8"))
    for row in rows:
        q_lang = normalize_text(row.get("q_lang")).lower()
        if languages and q_lang not in languages:
            continue
        image_rel = normalize_text(row.get("img_name"))
        image_path = slake_root / "imgs" / image_rel
        if not image_path.exists():
            continue
        yield make_example(
            source="slake",
            split=split,
            question_id=row.get("qid"),
            question=normalize_text(row.get("question")),
            answer=normalize_text(row.get("answer")),
            image_bytes=image_path.read_bytes(),
            image_path=str(image_path),
            extra_info={
                "benchmark": "SLAKE",
                "q_lang": row.get("q_lang"),
                "answer_type": row.get("answer_type"),
                "content_type": row.get("content_type"),
                "img_id": row.get("img_id"),
                "location": row.get("location"),
                "modality": row.get("modality"),
            },
        )


def iter_hf_imagefolder_records(dataset_root: Path, name: str, split: str) -> Iterable[dict[str, Any]]:
    ds = load_dataset(str(dataset_root / name / "data"), split=split)
    source_name = name.lower()
    benchmark_name = "PATH_VQA" if name.upper() == "PATH_VQA" else "VQA_RAD"
    for index, row in enumerate(ds):
        image = row["image"]
        image_bytes = ensure_rgb_bytes(image)
        image_path = f"{name}:{split}:{index}"
        yield make_example(
            source=source_name,
            split=split,
            question_id=f"{source_name}_{split}_{index}",
            question=normalize_text(row.get("question")),
            answer=normalize_text(row.get("answer")),
            image_bytes=image_bytes,
            image_path=image_path,
            extra_info={"benchmark": benchmark_name},
        )


def write_split(rows: list[dict[str, Any]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Dataset.from_list(rows).to_parquet(str(output_path))


def build_existing_vqa_dataset(args: argparse.Namespace) -> dict[str, Any]:
    input_path = Path(args.input_json)
    output_dir = Path(args.output_dir)
    eval_output_dir = Path(args.eval_output_dir) if args.eval_output_dir else output_dir

    records = read_records(input_path)
    converted = []
    skipped = 0
    for index, row in enumerate(records):
        item = convert_existing_vqa_row(row, index=index)
        if item is None:
            skipped += 1
            continue
        converted.append(item)

    if args.max_samples > 0:
        converted = converted[: args.max_samples]

    random.Random(args.seed).shuffle(converted)
    val_size = int(len(converted) * args.val_ratio) if args.val_ratio > 0 else 0
    if val_size > 0:
        val_rows = converted[:val_size]
        train_rows = converted[val_size:]
    else:
        train_rows = converted
        val_rows = []

    write_split(train_rows, output_dir / args.train_name)
    if val_rows:
        write_split(val_rows, eval_output_dir / args.val_name)

    return {
        "mode": "existing_vqa",
        "input_path": str(input_path),
        "output_dir": str(output_dir),
        "eval_output_dir": str(eval_output_dir),
        "total_input": len(records),
        "converted": len(converted),
        "skipped_missing_image": skipped,
        "train_size": len(train_rows),
        "val_size": len(val_rows),
    }


def build_benchmark_mix(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = Path(args.dataset_root)
    output_dir = Path(args.output_dir)
    eval_output_dir = Path(args.eval_output_dir) if args.eval_output_dir else output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    eval_output_dir.mkdir(parents=True, exist_ok=True)

    raw_languages = [item.strip().lower() for item in args.slake_languages.split(",") if item.strip()]
    languages = set() if (not raw_languages or "all" in raw_languages) else set(raw_languages)
    selected = {item.strip().lower() for item in args.benchmarks.split(",") if item.strip()}

    train_rows: list[dict[str, Any]] = []
    test_by_benchmark: dict[str, list[dict[str, Any]]] = {}

    if "slake" in selected:
        test_by_benchmark["slake"] = list(iter_slake_records(dataset_root, "test", languages))
        train_rows.extend(iter_slake_records(dataset_root, "train", languages))

    if "path_vqa" in selected:
        test_by_benchmark["path_vqa"] = list(iter_hf_imagefolder_records(dataset_root, "PATH_VQA", "test"))
        train_rows.extend(iter_hf_imagefolder_records(dataset_root, "PATH_VQA", "train"))

    if "vqa_rad" in selected:
        test_by_benchmark["vqa_rad"] = list(iter_hf_imagefolder_records(dataset_root, "VQA_RAD", "test"))
        train_rows.extend(iter_hf_imagefolder_records(dataset_root, "VQA_RAD", "train"))

    if args.shuffle_train:
        random.Random(args.seed).shuffle(train_rows)

    if args.max_train_samples > 0:
        train_rows = train_rows[: args.max_train_samples]

    write_split(train_rows, output_dir / args.train_name)

    combined_test_rows: list[dict[str, Any]] = []
    for key, rows in test_by_benchmark.items():
        if args.max_test_samples > 0:
            rows = rows[: args.max_test_samples]
            test_by_benchmark[key] = rows
        combined_test_rows.extend(rows)
        if args.write_split_tests:
            write_split(rows, eval_output_dir / f"test_{key}.parquet")

    if args.write_val:
        write_split(combined_test_rows, eval_output_dir / args.val_name)
    write_split(combined_test_rows, eval_output_dir / args.test_name)

    summary = {
        "mode": "benchmark_mix",
        "dataset_root": str(dataset_root),
        "output_dir": str(output_dir),
        "eval_output_dir": str(eval_output_dir),
        "benchmarks": sorted(selected),
        "slake_languages": "all" if not languages else sorted(languages),
        "train_size": len(train_rows),
        "test_size": len(combined_test_rows),
        "train_by_benchmark": {
            key: sum(1 for row in train_rows if row["data_source"] == key) for key in sorted(selected)
        },
        "test_by_benchmark": {key: len(rows) for key, rows in test_by_benchmark.items()},
        "files": {
            "train": str(output_dir / args.train_name),
            "val": str(eval_output_dir / args.val_name) if args.write_val else None,
            "test": str(eval_output_dir / args.test_name),
            **(
                {f"test_{key}": str(eval_output_dir / f"test_{key}.parquet") for key in test_by_benchmark}
                if args.write_split_tests
                else {}
            ),
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build VERL parquet datasets for medical agentic tool-use RL.")
    parser.add_argument("--input-json", default="", help="Existing project JSON/JSONL in the current medical VQA format.")
    parser.add_argument("--dataset-root", default="", help="Optional raw benchmark root.")
    parser.add_argument("--output-dir", required=True, help="Output directory for training parquet files.")
    parser.add_argument(
        "--eval-output-dir",
        default="",
        help="Output directory for validation/test parquet files; defaults to --output-dir.",
    )
    parser.add_argument("--train-name", default="train.parquet")
    parser.add_argument("--val-name", default="val.parquet")
    parser.add_argument("--test-name", default="test.parquet")
    parser.add_argument("--val-ratio", type=float, default=0.02)
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-test-samples", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--benchmarks", default="slake,path_vqa,vqa_rad", help="Comma-separated benchmark list.")
    parser.add_argument("--slake-languages", default="en", help="Comma-separated SLAKE languages to keep, or `all`.")
    parser.add_argument("--shuffle-train", action="store_true", default=True)
    parser.add_argument("--no-shuffle-train", dest="shuffle_train", action="store_false")
    parser.add_argument("--write-val", action="store_true", default=False)
    parser.add_argument("--write-split-tests", action="store_true", default=False)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.dataset_root:
        summary = build_benchmark_mix(args)
    elif args.input_json:
        summary = build_existing_vqa_dataset(args)
    else:
        raise ValueError("Provide either --dataset-root for raw benchmarks or --input-json for existing VQA JSON.")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
