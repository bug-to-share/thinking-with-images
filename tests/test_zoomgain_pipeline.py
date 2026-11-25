import io
import json
from collections import Counter

import pytest
from PIL import Image

from data_curation import zoomgain_pipeline as zoomgain


class MockClient:
    def __init__(self, respond):
        self.respond = respond
        self.calls = []

    def complete(self, *, stage, prompt, images):
        payload = json.loads(prompt.rsplit("\n", 1)[1])
        self.calls.append((stage, payload, [image.size for image in images]))
        return self.respond(stage, payload, images)


def test_five_stages_reverify_choices_and_write_curator_schema(tmp_path):
    for name in ("source", "same_modality", "other_modality"):
        Image.new("RGB", (100, 80), (10, 20, 30)).save(tmp_path / f"{name}.png")
    manifest = [
        {"question_id": "ct1", "image_path_rel": "source.png", "modality_type": "CT", "dataset": "fixture",
         "metadata": {"reason_for_roi": "private metadata only"}, "annotations": [
             {"label": "posterior nodule", "bbox": [20, 8, 60, 32], "bbox_format": "xywh",
              "coordinate_space": "pixels", "segmentation": [[20, 8, 80, 32, 30, 40]]},
         ]},
        {"question_id": "ct2", "image_path_rel": "same_modality.png", "modality": "CT",
         "gt_answer": "lobulated", "question": "Describe another nodule"},
        {"question_id": "mri1", "image_path_rel": "other_modality.png", "modality": "MRI", "gt_answer": "stellate"},
    ]
    specs = [
        ("vqa", "Which crop feature is seen?", "fine streaks"),
        ("mcq", "How does the margin appear?", "spiculated"),
        ("segment_grounded", "Which pattern defines the outlined region?", "dotted"),
        ("vqa", "Can the full view answer this?", "coarse"),
        ("vqa", "Can the text give this away?", "bright"),
        ("mcq", "Do the choices reveal the answer?", "spiculated"),
        ("mcq", "Are any hard negatives available?", "lobulated"),
    ]
    answers = {question: answer for _, question, answer in specs}
    rollout_indices = Counter()

    def synthesize(stage, payload, images):
        assert stage == "synthesis"
        assert [image.size for image in images] == [(100, 80), (60, 32)]
        assert payload["bbox"] == [0.2, 0.1, 0.8, 0.5]
        assert payload["metadata"]["reason_for_roi"] == "private metadata only"
        return {"items": [{"kind": kind, "question": question, "answer": answer, "evidence": "visible pixels",
                           "answer_aliases": ["spiny"] if answer == "spiculated" else []}
                          for kind, question, answer in specs]}

    def responder(voter):
        def vote(stage, payload, images):
            assert set(payload) <= {"question", "choices"}
            question = payload["question"]
            choices = payload.get("choices", [])
            answer = answers[question]
            assert len(images) == {"blind": 0, "full_image": 1, "full_image_crop": 2}[stage]
            correct = (stage == "full_image_crop" or (stage == "full_image" and answer == "coarse")
                       or (stage == "blind" and (answer == "bright" or question.startswith("Do the choices") and choices
                                                or question.startswith("Which crop") and voter == 0)))
            if correct:
                option = next((choice["letter"] for choice in choices if choice["text"] == answer), None)
                return {"answer": option or answer}
            return {"answer": "smooth"}
        return vote

    def rollout(stage, payload, images):
        assert stage == "rollout" and len(images) == 1
        question = payload["question"]
        choices = payload.get("choices", [])
        key = (question, bool(choices))
        attempt = rollout_indices[key]
        rollout_indices[key] += 1
        gold = answers[question]
        if question.startswith("Are any"):
            return {"answer": gold}
        if gold == "dotted":
            return {"answer": "smooth"}
        if gold == "fine streaks":
            return {"answer": gold if attempt < 3 else "smooth"}
        if choices:
            choice = next(option for option in choices if (option["text"] == gold) == (attempt < 5))
            return f'```json\n{{"answer":"{choice["letter"]}"}}\n```'
        return {"answer": ("Spiny", "Smooth", "smooth.", "irregular")[attempt % 4]}

    voters = [MockClient(responder(0)), MockClient(responder(1))]
    synthesis = MockClient(synthesize)
    pipeline = zoomgain.ZoomGainPipeline(synthesis, MockClient(rollout), zoomgain.CropCorrectVerifier(voters),
                                         zoomgain.BlindPriorFilter(voters))
    result = pipeline.run(manifest, image_root=tmp_path)
    assert len(synthesis.calls) == 1
    vqa, mcq, segment = result.accepted
    assert (vqa["correct_count"], vqa["difficulty_bucket"]) == (3, "sft_candidate")
    assert vqa["blind_verification"]["accuracy"] == 0.5
    assert (mcq["correct_count"], mcq["difficulty_bucket"]) == (5, "rl_candidate")
    assert mcq["crop_verification"]["gain"] == 1.0
    assert len(mcq["draft_verification"]["crop"]["votes"]) == 2
    assert mcq["num_choices"] == 4 and len(mcq["samples"]) == mcq["num_samples"] == 8
    assert {entry["text"].lower() for entry in mcq["distractors"]} == {"lobulated", "smooth", "irregular"}
    assert {entry["source"] for entry in mcq["distractors"]} == {"same_modality_metadata", "rollout"}
    assert next(choice["text"] for choice in mcq["choices"] if choice["letter"] == mcq["gold_letter"]) == "spiculated"
    assert segment["correct_count"] == 0 and segment["recommended_use"] == "quarantine_hard"
    assert segment["segment"] == manifest[0]["annotations"][0]["segmentation"]
    assert Counter(row["rejection_reason"] for row in result.rejected) == {
        "no_crop_gain": 1, "blind_prior": 2, "insufficient_distractors": 1, "no_anchor": 2,
    }
    output = tmp_path / "curated"
    result.write(output)
    assert zoomgain.read_jsonl(output / "results.jsonl") == result.accepted
    assert zoomgain.read_jsonl(output / "rejected.jsonl") == result.rejected
    assert zoomgain.read_jsonl(output / "rl_candidate.jsonl") == [mcq]
    assert zoomgain.read_jsonl(output / "easy_filtered.jsonl") == []


def test_grounding_fallback_filters_invalid_boxes_and_crops_evidence():
    image = Image.new("RGB", (20, 10), "white")
    image.paste("black", (5, 2, 15, 8))
    grounding = MockClient(lambda *args: {"anchors": [
        {"label": "bad area", "bbox": [0.2, 0.2, 0.2, 0.8]},
        {"label": "bad scale", "bbox": [0, 0, 100, 100]},
        {"label": "valid", "bbox": [0.25, 0.2, 0.75, 0.8]},
        {"label": "valid", "bbox": [0.25, 0.2, 0.75, 0.8]},
    ]})
    anchor, = zoomgain.propose_anchors({}, image, grounding)
    assert anchor.source == "grounding"
    assert anchor.crop(image).size == (10, 6)
    assert anchor.crop(image).getextrema() == ((0, 0),) * 3
    assert grounding.calls[0][0] == "grounding"
    metadata = {"metadata": {"coordinate_space": "relative_0_1000", "bbox_format": "yxyx",
                             "regions": [{"bbox": [200, 250, 800, 750], "label": "annotated"}]}}
    annotated, = zoomgain.propose_anchors(metadata, image, grounding)
    assert annotated.source == "annotation" and annotated.bbox == anchor.bbox
    assert len(grounding.calls) == 1
    detected, = zoomgain.propose_anchors({"detections": [{"finding": [5, 2, 15, 8]}]}, image)
    assert detected.label == "finding" and detected.bbox == anchor.bbox


def test_invalid_judge_answers_never_count_as_negative_evidence():
    item = {"question": "Which pattern?", "gt_answer": "bright", "answer_aliases": []}
    image = Image.new("RGB", (2, 3))
    crop = Image.new("RGB", (1, 1))
    malformed = MockClient(lambda stage, *_: "not JSON" if stage in {"full_image", "blind"} else {"answer": "bright"})
    voter = MockClient(lambda stage, *_: {"answer": "bright" if stage == "full_image_crop" else "dark"})
    crop_result = zoomgain.CropCorrectVerifier([voter, malformed]).verify(item, image, crop)
    assert crop_result["gain"] == 1.0 and not crop_result["keep"]
    assert not zoomgain.BlindPriorFilter([voter, malformed]).verify(item)["keep"]


@pytest.mark.parametrize(("count", "bucket"), list(enumerate([
    "quarantine_hard", "sft_candidate", "sft_candidate", "sft_candidate", "rl_candidate", "rl_candidate",
    "rl_candidate", "easy_filtered", "easy_filtered",
])))
def test_eight_rollout_boundaries(count, bucket):
    assert zoomgain.route_difficulty(count) == bucket


def test_openai_client_uses_images_only_when_supplied(monkeypatch):
    requests = []

    def open_fake(request, timeout):
        requests.append((request, json.loads(request.data), timeout))
        return io.BytesIO(b'{"choices":[{"message":{"content":"{\\"answer\\":\\"A\\"}"}}]}')

    monkeypatch.setattr(zoomgain, "urlopen", open_fake)
    client = zoomgain.OpenAICompatibleClient("https://mock.invalid/v1/", "local", "test-token")
    assert client.complete(stage="blind", prompt="test", images=[]) == '{"answer":"A"}'
    client.complete(stage="rollout", prompt="test", images=[Image.new("RGB", (1, 1))])
    request, blind, timeout = requests[0]
    assert request.full_url == "https://mock.invalid/v1/chat/completions" and timeout == 60
    assert request.get_header("Authorization") == "Bearer test-token"
    assert blind["messages"][0]["content"] == [{"type": "text", "text": "test"}]
    rollout = requests[1][1]
    assert rollout["temperature"] > blind["temperature"] == 0
    assert rollout["messages"][0]["content"][1]["image_url"]["url"].startswith("data:image/png;base64,")


def test_cli_routes_a_manifest_through_mocked_clients(tmp_path, monkeypatch, capsys):
    class OfflineClient:
        def __init__(self, base_url, model, api_key):
            self.model = model

        def complete(self, *, stage, prompt, images):
            if stage == "synthesis":
                return {"items": [{"kind": "vqa", "question": "Which shape is visible?", "answer": "ring"}]}
            if stage == "full_image_crop" or stage == "rollout":
                return {"answer": "ring"}
            return {"answer": "line"}

    monkeypatch.setattr(zoomgain, "OpenAICompatibleClient", OfflineClient)
    Image.new("RGB", (20, 20)).save(tmp_path / "input.png")
    manifest = tmp_path / "manifest.jsonl"
    zoomgain.write_jsonl(manifest, [{"id": "fixture", "image_path": "input.png", "bbox": [0, 0, 0.5, 0.5]}])
    out = tmp_path / "output"
    zoomgain.main(["--input", str(manifest), "--image-root", str(tmp_path), "--output-dir", str(out),
                   "--model", "synthesis", "--verifier-model", "alpha", "--verifier-model", "beta"])
    summary = json.loads(capsys.readouterr().out)
    assert summary["accepted"] == 1 and summary["rejected"] == 0
    filtered, = zoomgain.read_jsonl(out / "easy_filtered.jsonl")
    assert summary["buckets"]["easy_filtered"] == 1 and filtered["correct_count"] == 8
