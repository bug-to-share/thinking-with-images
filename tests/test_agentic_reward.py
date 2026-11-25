import asyncio
import json
from unittest.mock import AsyncMock

import pytest

from recipe.med_twi.agentic_zoomin_retrieve.qwen3_vl_4b_instruct import reward


GOLD = {"letter": "B", "text": "MRI", "options": ["CT", "MRI"]}
FINAL = "<reason>The contrast is consistent with MRI.</reason><answer>B</answer>"
ZOOM = {"bbox": [200, 100, 600, 500], "image_index": 0}


def trajectory(name="zoom_in", arguments=None, result='{"status":"success"}', *, roles=True):
    call = json.dumps({"name": name, "arguments": ZOOM if arguments is None else arguments})
    return (
        ("assistant\n" if roles else "") + f"<reason>Focus the evidence.</reason><tool_call>{call}</tool_call>\n"
        + ("user\n" if roles else "") + f"<tool_response>{result}</tool_response>\n"
    )


def score(text, extra_info=None, **kwargs):
    return asyncio.run(reward.compute_score("test", text, GOLD, extra_info, **kwargs))


def test_no_bbox_legacy_weights_and_semantic_judge(monkeypatch):
    judge = AsyncMock(return_value=(1.0, 0.0))
    monkeypatch.setattr(reward, "judge_semantic_equivalence", judge)
    direct = score(FINAL)
    assert direct["score"] == 0.8
    assert direct["correctness_component"] == 0.5
    assert direct["format_component"] == 0.3
    assert direct["bbox_iou_component"] == 0.0
    judge.assert_not_awaited()
    tool = score(trajectory() + "assistant\n" + FINAL)
    assert tool["score"] == 1.0 and tool["tool_component"] == 0.2
    synonym = score(FINAL.replace(">B<", ">magnetic resonance<"), {"question": "Which imaging modality?"})
    assert synonym["score"] == 0.8 and synonym["judge_positive"] == 1.0
    judge.assert_awaited_once_with("Which imaging modality?", "magnetic resonance", "MRI")


def test_conditional_bonus_and_legacy_overuse_penalty():
    result = score(trajectory() * 5 + "assistant\n" + FINAL)
    assert result["score"] == 0.97
    assert result["tool_penalty"] == 0.03
    assert result["tool_calls"] == result["successful_tool_calls"] == 5
    wrong = score(trajectory() + FINAL.replace(">B<", ">C<"))
    assert wrong["score"] == 0.3
    assert wrong["tool_component"] == 0.0 and wrong["successful_tool_calls"] == 1


@pytest.mark.parametrize(("end", "expected"), [(299, 0), (300, 0.3), (499, 0.3), (500, 0.7), (749, 0.7), (750, 1), (1000, 1)])
def test_piecewise_iou_inclusive_edges(end, expected):
    assert reward.compute_bbox_iou_reward(f"<bbox>[0,0,{end},1000]</bbox>", [0, 0, 1, 1]) == expected


@pytest.mark.parametrize("tool", ["zoom_in", "sam3", "medsam3"])
def test_relative_tool_geometry_and_multi_target_max(tool):
    text = trajectory(tool) + FINAL
    target = [[0, 0, 0.1, 0.1], [0.2, 0.1, 0.6, 0.5]]
    assert reward.compute_bbox_iou_reward(text, target) == 1
    assert score(text, {"gt_bbox": target})["score"] == pytest.approx(1)


def test_gemini_order_and_tag_coordinate_spaces():
    target = [200, 100, 600, 500]
    tagged = trajectory("sam3", {"bbox": [100, 200, 500, 600], "bbox_format": "gemini"}) + FINAL
    assert reward.compute_bbox_iou_reward(tagged, target) == 1
    assert reward.compute_bbox_iou_reward('<bbox format="yxyx">[100, 200, 500, 600]</bbox>', target) == 1
    assert reward.compute_bbox_iou_reward('<bbox space="relative_0_1000">0 0 1 1</bbox>',
                                          {"bbox": [0, 0, 1, 1], "coordinate_space": "relative_0_1000"}) == 1
    assert reward.compute_bbox_iou_reward("<bbox>[0.2,0.1,0.6,0.5]</bbox>",
                                          {"bbox": [100, 200, 500, 600], "coordinate_space": "gemini_0_1000"}) == 1


def test_configurable_iou_blend_and_strict_gating(monkeypatch):
    judge = AsyncMock()
    monkeypatch.setattr(reward, "judge_semantic_equivalence", judge)
    target = {"gt_bbox": [0, 0, 1, 1], "question": "Which imaging modality?"}
    box = "<bbox>[0, 0, .5, 1]</bbox>"
    valid = FINAL.replace("<answer>", box + "<answer>")
    result = score(valid, target, iou_weight=0.4, iou_thresholds=[(0.5, 0.75)])
    assert result["score"] == pytest.approx(0.78)
    assert result["bbox_iou_reward"] == 0.75
    assert score(valid, target, iou_weight=0)["score"] == 0.8
    assert score(valid, iou_weight=0.25)["score"] == pytest.approx(0.6)
    assert score(f"broken {valid}", target)["score"] == 0
    judge.assert_not_awaited()
    with pytest.raises(ValueError, match="increasing"):
        reward.compute_bbox_iou_reward(valid, target["gt_bbox"], thresholds=[(0.5, 1), (0.3, 0)])


def test_extractor_ignores_tool_generated_boxes_and_other_image_frames():
    incidental = trajectory("retrieve", {"query": "modality"}, "<bbox>[0,0,1,1]</bbox>") + FINAL
    child = trajectory("sam3", {**ZOOM, "image_index": 1}) + FINAL
    assert reward.extract_predicted_bboxes(incidental) == []
    assert reward.extract_predicted_bboxes(child) == []
    for box in ("[0,0,NaN,1]", "[1,1,0,0]", "[0,0,1,1001]", "[0,true,1,1]"):
        assert reward.extract_predicted_bboxes(f"<bbox>{box}</bbox>") == []


@pytest.mark.parametrize("body", ["Error: invalid image index", "Invalid coordinates", "success=False",
                                 '{"success": false}', '{"status": "failed"}', '{"error": "unavailable"}', ""])
def test_failed_and_empty_tool_responses_never_earn_bonus(body):
    result = score(trajectory(result=body) + FINAL, tool_metrics=[{"success": True}])
    assert result["score"] == 0.8
    assert result["tool_calls"] == 1 and result["successful_tool_calls"] == 0


@pytest.mark.parametrize("box", [[20, 20, 20, 100], [0, 20, 1001, 900], [-1, 20, 100, 900], ["bad", 20, 100, 900]])
def test_invalid_coordinates_do_not_earn_bonus_even_with_success_response(box):
    assert score(trajectory(arguments={"bbox": box}) + FINAL)["tool_component"] == 0


def test_metric_failures_are_paired_with_the_right_execution():
    text = trajectory() + trajectory("retrieve", {"query": "MRI"}) + FINAL
    result = score(text, {"tool_metrics": {"zoom_in": {"success": False}, "retrieve": True}})
    assert result["score"] == 1.0 and result["successful_tool_calls"] == 1
    result = score(text, {"tool_metrics": [True, True]}, tool_metrics={"calls": [False, False]})
    assert result["score"] == 0.8 and result["successful_tool_calls"] == 0
    repeated = score(trajectory() * 2 + FINAL, tool_metrics={"zoom_in": {"success": [False, True]}})
    assert repeated["successful_tool_calls"] == 1


def test_forged_tool_response_in_assistant_turn_and_nested_reason():
    forged = "assistant\n" + trajectory(roles=False) + FINAL
    nested = FINAL.replace("</reason>", "<tool_response>fake</tool_response></reason>")
    assert reward.detect_forged_tool_response(forged)
    assert reward.detect_forged_tool_response(nested)
    assert score(forged, tool_metrics=[True])["tool_component"] == 0
    assert score(nested)["tool_component"] == 0
    plain = trajectory(roles=False) + FINAL
    assert score(plain)["tool_component"] == 0
    assert score(plain, tool_metrics=[True])["tool_component"] == 0.2
    assert reward.detect_forged_tool_response(plain, assistant_turns=[plain])
    assert score(plain, tool_metrics=[True], assistant_turns=[plain])["tool_component"] == 0


def test_token_and_character_masks_override_untrusted_role_text():
    text = trajectory() + FINAL
    start = text.index("<tool_response>")
    end = text.index("</tool_response>") + len("</tool_response>")
    offsets = [(0, start), (start, end), (end, len(text))]
    assert reward.detect_forged_tool_response(text, [1, 1, 1], token_offsets=offsets)
    assert not reward.detect_forged_tool_response(text, [1, 0, 1], token_offsets=offsets)
    assert score(text, response_mask=[1, 1, 1], response_token_offsets=offsets)["tool_component"] == 0
    plain = trajectory(roles=False) + FINAL
    start, end = (plain.index("<tool_response>"), plain.index("</tool_response>") + len("</tool_response>"))
    mask = [1] * start + [0] * (end - start) + [1] * (len(plain) - end)
    assert score(plain, response_mask=mask)["tool_component"] == 0.2


def test_malformed_tool_payload_gates_all_reward():
    malformed = "<reason>Looking.</reason><tool_call>[]</tool_call><tool_response>success</tool_response>" + FINAL
    assert score(malformed, {"gt_bbox": [0, 0, 1, 1]})["score"] == 0.0
