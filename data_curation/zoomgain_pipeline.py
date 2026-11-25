"""Construct and verify crop-grounded questions from a JSONL image manifest.

Run ``python -m data_curation.zoomgain_pipeline --help`` for the HTTP CLI.
Library clients implement ``complete(stage=..., prompt=..., images=...)`` and
can return JSON text or a dict. Verification clients receive only the question,
its choices when present, and the images permitted for the current condition.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import os
import random
import re
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol
from urllib.request import Request, urlopen

from PIL import Image

if __package__:
    from .bucket_by_correct_count import augment_row, bucket_name, read_jsonl, write_jsonl
else:
    from bucket_by_correct_count import augment_row, bucket_name, read_jsonl, write_jsonl


ROLLOUT_COUNT = 8
BUCKETS = ("quarantine_hard", "sft_candidate", "rl_candidate", "easy_filtered")
BUCKET_THRESHOLDS = argparse.Namespace(quarantine_max=0, sft_min=1, sft_max=3, rl_min=4, rl_max=6, easy_min=7)
QUESTION_KINDS = {"vqa", "mcq", "segment_grounded"}


class CompletionClient(Protocol):
    def complete(self, *, stage: str, prompt: str, images: Sequence[Image.Image]) -> str | dict[str, Any]: ...


@dataclass(frozen=True)
class OpenAICompatibleClient:
    base_url: str
    model: str
    api_key: str = ""
    timeout: float = 60.0
    max_tokens: int = 1024
    rollout_temperature: float = 0.7

    def complete(self, *, stage: str, prompt: str, images: Sequence[Image.Image]) -> str:
        content = [{"type": "text", "text": prompt}]
        for image in images:
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            content.append({"type": "image_url", "image_url": {"url": url}})
        payload = {
            "model": self.model, "messages": [{"role": "user", "content": content}],
            "max_tokens": self.max_tokens, "temperature": self.rollout_temperature if stage == "rollout" else 0.0,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        request = Request(self.base_url.rstrip("/") + "/chat/completions", json.dumps(payload).encode(), headers)
        with urlopen(request, timeout=self.timeout) as response:
            return json.load(response)["choices"][0]["message"]["content"]


def _json_completion(result: str | dict[str, Any]) -> dict[str, Any]:
    if isinstance(result, str):
        result = re.sub(r"^```(?:json)?\s*|\s*```$", "", result.strip(), flags=re.I)
        try:
            result = json.loads(result)
        except ValueError:
            return {}
    return result if isinstance(result, dict) else {}


def _ask(client: CompletionClient, stage: str, prompt: str, payload: dict[str, Any],
         images: Sequence[Image.Image] = ()) -> dict[str, Any]:
    return _json_completion(client.complete(
        stage=stage, prompt=prompt + "\n" + json.dumps(payload, ensure_ascii=False), images=images,
    ))


def _text(value: Any) -> str:
    return " ".join(value.split()) if isinstance(value, str) else ""


def _answer_key(text: str) -> str:
    text = _text(text).casefold().replace("&", " and ")
    return re.sub(r"^(?:a|an|the)\s+", "", _text(text).strip(" .,!?;:\"“”"))


def _same_answer(left: str, right: str) -> bool:
    return bool(_answer_key(left)) and _answer_key(left) == _answer_key(right)


def _matches(item: dict[str, Any], answer: str, equivalent: Callable[[str, str], bool]) -> bool:
    return bool(answer) and any(equivalent(answer, target) for target in [item["gt_answer"], *item["answer_aliases"]])


def _modality(row: dict[str, Any]) -> str:
    metadata = row.get("metadata") or {}
    return _text(row.get("modality") or row.get("modality_type") or metadata.get("modality")
                 or metadata.get("modality_type"))


def _image_path(row: dict[str, Any]) -> str:
    return _text(row.get("image_path_abs") or row.get("image_path_rel") or row.get("image_path"))


@dataclass(frozen=True)
class ROIAnchor:
    label: str
    bbox: tuple[float, float, float, float]
    source: str
    segment: Any = None

    def crop(self, image: Image.Image) -> Image.Image:
        x1, y1, x2, y2 = self.bbox
        width, height = image.size
        return image.crop((math.floor(x1 * width), math.floor(y1 * height),
                           math.ceil(x2 * width), math.ceil(y2 * height)))


def _unit_box(box: Any, size: tuple[int, int], coordinate_space: str, order: str) -> tuple[float, ...] | None:
    if not isinstance(box, (list, tuple)) or len(box) != 4 or any(isinstance(value, bool) for value in box):
        return None
    try:
        x1, y1, x2, y2 = map(float, box)
    except (TypeError, ValueError, OverflowError):
        return None
    if re.search(r"gemini|yxyx|ymin", order.casefold()) or "gemini" in coordinate_space:
        x1, y1, x2, y2 = y1, x1, y2, x2
    elif order.casefold() == "xywh":
        x2, y2 = x1 + x2, y1 + y2
    if "1000" in coordinate_space:
        width = height = 1000
    elif coordinate_space in {"unit", "normalized", "normalized_0_1", "relative_0_1"}:
        width = height = 1
    elif coordinate_space in {"pixels", "pixel", "absolute"}:
        width, height = size
    elif not coordinate_space:
        width, height = (1, 1) if max(x1, y1, x2, y2) <= 1 else size
    else:
        return None
    if all(math.isfinite(v) for v in (x1, y1, x2, y2)) and 0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height:
        return x1 / width, y1 / height, x2 / width, y2 / height
    return None


def _anchor_records(row: dict[str, Any]) -> Iterable[dict[str, Any]]:
    for parent in (row, row.get("metadata") or {}):
        yield parent
        defaults = {key: parent[key] for key in ("bbox_format", "coordinate_space") if key in parent}
        for field in ("annotations", "regions", "anchors", "detections"):
            annotations = parent.get(field) or []
            if isinstance(annotations, dict):
                annotations = [annotations]
            for annotation in annotations:
                if not isinstance(annotation, dict):
                    continue
                if {"bbox", "gt_bbox", "roi_bbox"} & annotation.keys():
                    yield defaults | annotation
                else:
                    for label, bbox in annotation.items():
                        if isinstance(bbox, (list, tuple)):
                            yield defaults | {"label": label, "bbox": bbox}


def _anchors(records: Iterable[dict[str, Any]], size: tuple[int, int], source: str) -> list[ROIAnchor]:
    anchors = {}
    for record in records:
        boxes = record.get("gt_bbox") or record.get("roi_bbox") or record.get("bbox")
        multi_box = isinstance(boxes, (list, tuple)) and boxes and isinstance(boxes[0], (list, tuple))
        for box in boxes if multi_box else [boxes]:
            bbox = _unit_box(box, size, _text(record.get("coordinate_space")).lower(),
                             _text(record.get("bbox_format")))
            if bbox:
                label = _text(record.get("label") or record.get("anchor") or record.get("category_name")) or "region of interest"
                segment = record.get("segment") or record.get("segmentation") or record.get("mask_path")
                key = (label.casefold(), bbox)
                if key not in anchors or (anchors[key].segment is None and segment is not None):
                    anchors[key] = ROIAnchor(label, bbox, source, segment)
    return list(anchors.values())


def propose_anchors(row: dict[str, Any], image: Image.Image, grounding_client: CompletionClient | None = None) -> list[ROIAnchor]:
    """Use annotated regions first, then fall back to a grounding backend."""
    anchors = _anchors(_anchor_records(row), image.size, "annotation")
    if anchors or grounding_client is None:
        return anchors
    proposal = _ask(grounding_client, "grounding", (
        'Propose visually identifiable regions relevant to the metadata. Return JSON {"anchors": '
        '[{"label": "...", "bbox": [x1, y1, x2, y2]}]} using xyxy coordinates normalized to [0, 1].'
    ), {"metadata": row.get("metadata", {}), "modality": _modality(row), "question": row.get("question")}, [image])
    return _anchors(
        (region | {"coordinate_space": "normalized_0_1", "bbox_format": "xyxy"}
         for region in proposal.get("anchors") or [] if isinstance(region, dict)), image.size, "grounding",
    )


def synthesize_items(client: CompletionClient, row: dict[str, Any], anchor: ROIAnchor,
                     image: Image.Image, crop: Image.Image) -> list[dict[str, Any]]:
    items = _ask(client, "synthesis", (
        'The images show the full view and then an ROI crop. Write diverse questions whose answers depend on new '
        'visual evidence in the crop. Avoid revealing the ROI label, its location, or answer in the question. '
        'Use kind vqa, mcq (a single unambiguous answer, without options), or segment_grounded (only with supplied '
        'segmentation). The question must be self-contained and must avoid referring to an invisible overlay. '
        'Return JSON {"items": [{"kind": "...", "question": "...", "answer": "...", '
        '"answer_aliases": [], "evidence": "visible detail that supports the answer"}]}.'
    ), {"metadata": row.get("metadata", {}), "modality": _modality(row), "source_question": row.get("question"),
        "source_answer": row.get("gt_answer"), "location": row.get("location"), "content_type": row.get("content_type"),
        "anchor": anchor.label, "bbox": anchor.bbox,
        "segment": anchor.segment}, [image, crop])
    seen = set()
    output = []
    for item in items.get("items") or []:
        if not isinstance(item, dict):
            continue
        kind = _text(item.get("kind")).lower().replace("-", "_")
        if kind == "segment":
            kind = "segment_grounded"
        question, answer = _text(item.get("question")), _text(item.get("answer"))
        key = (kind, _answer_key(question))
        if (kind not in QUESTION_KINDS or (kind == "segment_grounded" and anchor.segment is None)
                or not question or not answer or key in seen):
            continue
        seen.add(key)
        aliases = item.get("answer_aliases")
        output.append({
            "question_type": kind, "question": question, "gt_answer": answer,
            "answer_aliases": [_text(alias) for alias in aliases if _text(alias)] if isinstance(aliases, list) else [],
            "crop_evidence": _text(item.get("evidence")),
        })
    return output


def _predict(client: CompletionClient, item: dict[str, Any], stage: str,
             images: Sequence[Image.Image]) -> tuple[str, str | None, dict[str, Any]]:
    prompt = 'Answer the question independently. Return JSON {"answer": "short answer or option letter"}.'
    if len(images) == 2:
        prompt += " The first image is the full view and the second is a crop from that view."
    if not images:
        prompt += " No image is available. Give your best text-only answer."
    question = {"question": item["question"]}
    choices = item.get("choices") or []
    if choices:
        question["choices"] = choices
    result = _ask(client, stage, prompt, question, images)
    answer = _text(result.get("answer"))
    for choice in choices:
        if answer.strip(".() ").upper() == choice["letter"] or _same_answer(answer, choice["text"]):
            return choice["text"], choice["letter"], result
    return answer, None, result


@dataclass(frozen=True)
class CropCorrectVerifier:
    clients: Sequence[CompletionClient]
    min_gain: float = 1 / 3
    min_crop_accuracy: float = 2 / 3
    equivalent: Callable[[str, str], bool] = _same_answer

    def __post_init__(self) -> None:
        if len(self.clients) < 2 or not 0 < self.min_gain <= 1 or not 0 <= self.min_crop_accuracy <= 1:
            raise ValueError("Crop verification requires at least two voters and valid crop gain and accuracy thresholds")

    def verify(self, item: dict[str, Any], image: Image.Image, crop: Image.Image) -> dict[str, Any]:
        votes = []
        for client in self.clients:
            full, _, _ = _predict(client, item, "full_image", [image])
            cropped, _, _ = _predict(client, item, "full_image_crop", [image, crop])
            votes.append({"full_answer": full, "crop_answer": cropped,
                          "full_correct": _matches(item, full, self.equivalent),
                          "crop_correct": _matches(item, cropped, self.equivalent)})
        full = sum(vote["full_correct"] for vote in votes) / len(votes)
        cropped = sum(vote["crop_correct"] for vote in votes) / len(votes)
        gain = cropped - full
        valid = all(vote["full_answer"] and vote["crop_answer"] for vote in votes)
        return {"votes": votes, "full_accuracy": full, "crop_accuracy": cropped, "gain": gain, "valid": valid,
                "keep": valid and cropped >= self.min_crop_accuracy and gain + 1e-12 >= self.min_gain}


@dataclass(frozen=True)
class BlindPriorFilter:
    clients: Sequence[CompletionClient]
    max_accuracy: float = 0.5
    equivalent: Callable[[str, str], bool] = _same_answer

    def __post_init__(self) -> None:
        if not self.clients or not 0 <= self.max_accuracy <= 1:
            raise ValueError("Blind verification requires at least one voter and a threshold in [0, 1]")

    def verify(self, item: dict[str, Any]) -> dict[str, Any]:
        answers = [_predict(client, item, "blind", ())[0] for client in self.clients]
        accuracy = sum(_matches(item, answer, self.equivalent) for answer in answers) / len(answers)
        valid = all(answers)
        return {"answers": answers, "accuracy": accuracy, "valid": valid, "keep": valid and accuracy <= self.max_accuracy}


def _metadata_negatives(item: dict[str, Any], source: dict[str, Any], rows: Iterable[dict[str, Any]]) -> Iterable[str]:
    for parent in (source, source.get("metadata") or {}):
        negatives = parent.get("negative_answers") or []
        if isinstance(negatives, (list, tuple)):
            yield from negatives
    modality = _modality(source).casefold()
    if not modality:
        return
    for row in rows:
        if row is source or _modality(row).casefold() != modality or _image_path(row) == _image_path(source):
            continue
        metadata = row.get("metadata") or {}
        answer = _text(metadata.get("answer") or row.get("gt_answer") or metadata.get("finding") or metadata.get("label"))
        choices = row.get("choices") or []
        for choice in choices:
            if isinstance(choice, dict) and _text(choice.get("letter")).upper() == answer.upper():
                answer = _text(choice.get("text"))
                break
        if answer and (_answer_key(row.get("question") or "") != _answer_key(item["question"])):
            yield answer


@dataclass(frozen=True)
class DistractorBuilder:
    count: int = 3
    seed: int = 42
    equivalent: Callable[[str, str], bool] = _same_answer

    def __post_init__(self) -> None:
        if not 2 <= self.count <= 4:
            raise ValueError("MCQs require two to four distractors")

    def build(self, item: dict[str, Any], samples: Sequence[dict[str, Any]], source: dict[str, Any],
              metadata_rows: Iterable[dict[str, Any]]) -> dict[str, Any] | None:
        pools = {
            "rollout": iter(sample.get("pred_answer") for sample in samples if not sample.get("correct")),
            "same_modality_metadata": iter(_metadata_negatives(item, source, metadata_rows)),
        }
        negatives = []
        for _ in range(self.count):
            for origin, pool in pools.items():
                for value in pool:
                    answer = _text(value)
                    if (not answer or _matches(item, answer, self.equivalent)
                            or any(self.equivalent(answer, entry["text"]) for entry in negatives)):
                        continue
                    negatives.append({"text": answer, "source": origin})
                    break
                if len(negatives) == self.count:
                    break
            if len(negatives) == self.count:
                break
        if len(negatives) != self.count or {entry["source"] for entry in negatives} != pools.keys():
            return None
        options = [item["gt_answer"], *(negative["text"] for negative in negatives)]
        random.Random(f"{self.seed}:{item['question_id']}").shuffle(options)
        choices = [{"letter": letter, "text": option} for letter, option in zip("ABCDE", options)]
        return {"choices": choices, "num_choices": len(choices), "distractors": negatives,
                "gold_letter": choices[options.index(item["gt_answer"])]["letter"]}


def route_difficulty(correct_count: int) -> str:
    if not isinstance(correct_count, int) or isinstance(correct_count, bool) or not 0 <= correct_count <= ROLLOUT_COUNT:
        raise ValueError("correct_count must be an integer from 0 to 8")
    return bucket_name(correct_count, BUCKET_THRESHOLDS)


@dataclass
class PipelineResult:
    accepted: list[dict[str, Any]]
    rejected: list[dict[str, Any]]

    def write(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(output_dir / "results.jsonl", self.accepted)
        write_jsonl(output_dir / "rejected.jsonl", self.rejected)
        for bucket in BUCKETS:
            write_jsonl(output_dir / f"{bucket}.jsonl", (row for row in self.accepted if row["difficulty_bucket"] == bucket))


@dataclass
class ZoomGainPipeline:
    synthesis_client: CompletionClient
    rollout_client: CompletionClient
    crop_verifier: CropCorrectVerifier
    blind_filter: BlindPriorFilter
    distractor_builder: DistractorBuilder = DistractorBuilder()
    grounding_client: CompletionClient | None = None

    def _rollouts(self, item: dict[str, Any], image: Image.Image) -> list[dict[str, Any]]:
        samples = []
        for index in range(ROLLOUT_COUNT):
            answer, letter, raw = _predict(self.rollout_client, item, "rollout", [image])
            samples.append({"sample_index": index, "pred_answer": answer, "pred_letter": letter,
                            "correct": _matches(item, answer, self.crop_verifier.equivalent),
                            "status": "ok" if answer else "parse_failed", "raw_response": json.dumps(raw, ensure_ascii=False)})
        return samples

    def _verify(self, item: dict[str, Any], image: Image.Image, crop: Image.Image) -> str | None:
        item["crop_verification"] = self.crop_verifier.verify(item, image, crop)
        if not item["crop_verification"]["valid"]:
            return "invalid_crop_vote"
        if not item["crop_verification"]["keep"]:
            return "no_crop_gain"
        item["blind_verification"] = self.blind_filter.verify(item)
        if not item["blind_verification"]["valid"]:
            return "invalid_blind_vote"
        return None if item["blind_verification"]["keep"] else "blind_prior"

    def run(self, rows: Iterable[dict[str, Any]], *, image_root: Path = Path(".")) -> PipelineResult:
        manifest = list(rows)
        result = PipelineResult([], [])
        for row_index, row in enumerate(manifest):
            source_id = str(row.get("question_id") or row.get("id") or row_index)
            path = image_root / _image_path(row)
            with Image.open(path) as source_image:
                image = source_image.convert("RGB")
            anchors = propose_anchors(row, image, self.grounding_client)
            if not anchors:
                result.rejected.append({"source_question_id": source_id, "rejection_reason": "no_anchor"})
            for anchor_index, anchor in enumerate(anchors):
                crop = anchor.crop(image)
                drafts = synthesize_items(self.synthesis_client, row, anchor, image, crop)
                if not drafts:
                    result.rejected.append({"source_question_id": source_id, "anchor": anchor.label,
                                            "rejection_reason": "no_synthesized_items"})
                for item_index, draft in enumerate(drafts):
                    item = {**draft, "question_id": f"{source_id}:zoomgain:{anchor_index}:{item_index}",
                            "source_question_id": source_id, "image_path_abs": str(path.resolve()),
                            "modality_type": _modality(row), "anchor": anchor.label, "anchor_source": anchor.source,
                            "gt_bbox": list(anchor.bbox), "coordinate_space": "normalized_0_1"}
                    for field in ("dataset", "benchmark", "split", "source_group", "source_json", "image_path_rel"):
                        if field in row:
                            item[field] = row[field]
                    if anchor.segment is not None:
                        item["segment"] = anchor.segment
                    rejection = self._verify(item, image, crop)
                    if rejection is None and item["question_type"] == "mcq":
                        choices = self.distractor_builder.build(item, self._rollouts(item, image), row, manifest)
                        if choices is None:
                            rejection = "insufficient_distractors"
                        else:
                            item.update(choices)
                            item["draft_verification"] = {"crop": item.pop("crop_verification"),
                                                          "blind": item.pop("blind_verification")}
                            rejection = self._verify(item, image, crop)
                    if rejection is not None:
                        result.rejected.append({**item, "rejection_reason": rejection})
                        continue
                    samples = self._rollouts(item, image)
                    count = sum(sample["correct"] for sample in samples)
                    item.update(samples=samples, num_samples=ROLLOUT_COUNT, correct_count=count,
                                accuracy=count / ROLLOUT_COUNT, easy_threshold=7, is_easy=count >= 7,
                                parse_failed_count=sum(sample["status"] == "parse_failed" for sample in samples),
                                request_failed_count=0, decision="filter_out_easy" if count >= 7 else "keep_for_rl")
                    result.accepted.append(augment_row(item, route_difficulty(count)))
        return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="JSONL image manifest with metadata or ROI annotations")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image-root", type=Path, default=Path("."), help="Root for relative image paths")
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:31000/v1"))
    parser.add_argument("--model", required=True, help="Default synthesis, grounding and rollout model")
    parser.add_argument("--grounding-model")
    parser.add_argument("--rollout-model")
    parser.add_argument("--verifier-model", action="append", required=True, help="Repeat for at least two distinct VLMs")
    parser.add_argument("--blind-model", action="append", help="Repeat to use separate blind voters")
    parser.add_argument("--min-crop-gain", type=float, default=1 / 3)
    parser.add_argument("--min-crop-accuracy", type=float, default=2 / 3)
    parser.add_argument("--max-blind-accuracy", type=float, default=0.5)
    args = parser.parse_args(argv)
    if len(set(args.verifier_model)) != len(args.verifier_model) or len(args.verifier_model) < 2:
        parser.error("supply at least two distinct --verifier-model values")

    def client(model: str) -> OpenAICompatibleClient:
        return OpenAICompatibleClient(args.base_url, model, os.environ.get("OPENAI_API_KEY", ""))

    crop_verifier = CropCorrectVerifier([client(model) for model in args.verifier_model],
                                        args.min_crop_gain, args.min_crop_accuracy)
    blind_filter = BlindPriorFilter([client(model) for model in args.blind_model or args.verifier_model],
                                    args.max_blind_accuracy)
    pipeline = ZoomGainPipeline(client(args.model), client(args.rollout_model or args.model), crop_verifier,
                                blind_filter, grounding_client=client(args.grounding_model or args.model))
    result = pipeline.run(read_jsonl(args.input), image_root=args.image_root)
    result.write(args.output_dir)
    print(json.dumps({"accepted": len(result.accepted), "rejected": len(result.rejected), "buckets": {
        name: sum(row["difficulty_bucket"] == name for row in result.accepted) for name in BUCKETS
    }}))


if __name__ == "__main__":
    main()
