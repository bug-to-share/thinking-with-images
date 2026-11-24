#!/usr/bin/env python3
import argparse
import json
from collections import Counter
from datetime import datetime
from pathlib import Path


def sample_text(sample: dict) -> str:
    messages = sample.get("messages", [])
    parts = []
    for message in messages:
        content = message.get("content", "")
        if isinstance(content, str):
            parts.append(content)
        else:
            parts.append(str(content))
    return "\n".join(parts)


def summarize(samples: list[dict]) -> dict:
    char_lengths = [len(sample_text(sample)) for sample in samples]
    image_counts = [len(sample.get("images", [])) for sample in samples]
    return {
        "num_samples": len(samples),
        "max_chars": max(char_lengths) if char_lengths else 0,
        "mean_chars": round(sum(char_lengths) / len(char_lengths), 2) if char_lengths else 0,
        "image_count_dist": dict(sorted(Counter(image_counts).items())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Filter overlong LlamaFactory ShareGPT multimodal samples.")
    parser.add_argument("--input", required=True, help="Input dataset json path.")
    parser.add_argument("--output", required=True, help="Output dataset json path.")
    parser.add_argument("--backup", default=None, help="Optional backup path for original input.")
    parser.add_argument("--stats-output", default=None, help="Optional stats json path.")
    parser.add_argument("--max-chars", type=int, default=None, help="Maximum total characters across all messages.")
    parser.add_argument("--max-images", type=int, default=None, help="Maximum number of images per sample.")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    backup_path = Path(args.backup) if args.backup else None
    stats_output_path = Path(args.stats_output) if args.stats_output else None

    samples = json.loads(input_path.read_text(encoding="utf-8"))
    before_summary = summarize(samples)

    kept = []
    removed = []
    for index, sample in enumerate(samples):
        text = sample_text(sample)
        num_chars = len(text)
        num_images = len(sample.get("images", []))

        reasons = []
        if args.max_chars is not None and num_chars > args.max_chars:
            reasons.append(f"chars>{args.max_chars}")
        if args.max_images is not None and num_images > args.max_images:
            reasons.append(f"images>{args.max_images}")

        if reasons:
            removed.append(
                {
                    "index": index,
                    "question_id": sample.get("question_id"),
                    "chars": num_chars,
                    "images": num_images,
                    "reasons": reasons,
                }
            )
        else:
            kept.append(sample)

    if backup_path:
        backup_path.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")

    output_path.write_text(json.dumps(kept, ensure_ascii=False, indent=2), encoding="utf-8")

    after_summary = summarize(kept)
    stats = {
        "timestamp": datetime.now().isoformat(),
        "input_path": str(input_path),
        "output_path": str(output_path),
        "backup_path": str(backup_path) if backup_path else None,
        "max_chars": args.max_chars,
        "max_images": args.max_images,
        "before": before_summary,
        "after": after_summary,
        "removed_count": len(removed),
        "removed_examples": removed[:200],
    }

    if stats_output_path:
        stats_output_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
