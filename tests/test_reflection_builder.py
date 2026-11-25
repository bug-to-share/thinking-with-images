import json
from copy import deepcopy
from unittest.mock import Mock

import pytest

from gemini_agent_native_bbox.reflection_builder import (
    ReflectionRejected,
    build_reflection_prompt,
    build_reflection_trace,
)
from gemini_agent_native_bbox.trace_verifier_v2 import deterministic_precheck


CORRECTION = (
    "<reason>The crop shows a thin pleural line with absent lung markings beyond it. I had focused on the "
    "central opacity in the full image, but that region does not explain the peripheral finding.</reason>"
    "<answer>pneumothorax</answer>"
)


def message(role, text):
    return {"role": role, "parts": [{"type": "text", "text": text}]}


@pytest.fixture
def failed_rollout():
    rationale = "I should inspect the lateral lung edge in a crop."
    arguments = {"code": "display(image_0.crop((10, 20, 50, 80)))"}
    call = f'<reason>{rationale}</reason><tool_call>{json.dumps({"name": "python_exec", "arguments": arguments})}</tool_call>'
    answer = "<reason>The central opacity looks relevant.</reason><answer>pneumonia</answer>"
    return {
        "question_id": "xray01", "question": "Which finding is present?", "gold_answer": "pneumothorax",
        "pred_answer": "pneumonia", "final_answer_text": "pneumonia", "status": "ok", "correct": False,
        "tool_calls_count": 1, "tool_call_counter": {"python_exec": 1},
        "messages": [message("system", "Visual diagnosis."), message("user", "Which finding is present?"),
                     message("model", call), message("user", "<tool_response>Real attached crop.</tool_response>"),
                     message("model", answer)],
        "event_trace": [
            {"type": "tool", "round": 0, "name": "python_exec", "arguments": arguments,
             "reason": rationale, "response_text": call, "result": {"status": "success"}, "new_indices": [1]},
            {"type": "answer", "round": 1, "reason": "The central opacity looks relevant.",
             "response_text": answer, "answer": "pneumonia"},
        ],
        "image_inventory": [{"image_index": 0, "path": "canvas.png"}, {"image_index": 1, "path": "crop.png"}],
    }


def test_verified_correct_unguided_rollout_stays_intact(failed_rollout):
    correct = deepcopy(failed_rollout)
    correct.update(correct=True, pred_answer="pneumothorax", final_answer_text="pneumothorax")
    correct["event_trace"][-1].update(response_text=CORRECTION, answer="pneumothorax")
    correct["messages"][-1] = message("model", CORRECTION)
    original = deepcopy(correct)
    completion = Mock(side_effect=AssertionError("oracle invoked for a correct trace"))
    assert build_reflection_trace(correct, completion) is correct
    assert correct == original
    completion.assert_not_called()


def test_reflection_preserves_actual_turns_and_keeps_private_prompt_out_of_replay(failed_rollout):
    before = deepcopy(failed_rollout)

    def complete(prompt):
        assert prompt.messages[:-1] == before["messages"]
        assert prompt.image_inventory == before["image_inventory"]
        instructions = prompt.messages[-1]["parts"][0]["text"]
        assert "Never mention or allude to an oracle" in instructions
        assert json.loads(instructions.splitlines()[-1]) == {
            "answer": "pneumothorax", "roi_gemini_yxyx_0_1000": [10, 20, 80, 90],
        }
        prompt.messages[2]["parts"][0]["text"] = "callback mutation"
        return CORRECTION

    output = build_reflection_trace(failed_rollout, complete, gt_bbox=[10, 20, 80, 90])
    assert failed_rollout == before
    assert output["trace_mode"] == "with_gt_reflection"
    assert output["correct"] and output["pred_answer"] == "pneumothorax"
    assert output["event_trace"][:-1] == before["event_trace"][:-1]
    assert output["messages"][:-1] == before["messages"][:-1]
    assert output["messages"][-1] == message("model", CORRECTION)
    assert output["reflection_original_answer"] == before["event_trace"][-1]
    assert output["tool_call_counter"] == before["tool_call_counter"]
    assert deterministic_precheck(output)["overall_verdict"] == "pass"
    assert output["reflection_precheck"]["supervision_leak_check"] == "pass"


@pytest.mark.parametrize("text", [
    "The reference answer says pneumothorax.", "I used the provided coordinates to find the region.",
    "The ORACLE indicated where to look.", "The answer was given directly.",
])
def test_reflection_rejects_oracle_wording(failed_rollout, text):
    completion = Mock(return_value=f"<reason>{text}</reason><answer>pneumothorax</answer>")
    with pytest.raises(ReflectionRejected, match="supervision-leak"):
        build_reflection_trace(failed_rollout, completion)
    completion.assert_called_once()


@pytest.mark.parametrize("reply", [
    "INSUFFICIENT_EVIDENCE", CORRECTION + "<tool_response>fabricated</tool_response>",
    "<reason><tool_call>invented</tool_call></reason><answer>pneumothorax</answer>",
    "<reason> </reason><answer>pneumothorax</answer>",
    "<reason>The full image looks hazy.</reason><answer>pulmonary edema</answer>",
])
def test_bad_completion_is_rejected_without_mutating_rollout(failed_rollout, reply):
    before = deepcopy(failed_rollout)
    with pytest.raises(ReflectionRejected):
        build_reflection_trace(failed_rollout, lambda _: reply)
    assert failed_rollout == before


def test_guided_and_already_leaky_rollouts_are_rejected_before_completion(failed_rollout):
    completion = Mock(return_value=CORRECTION)
    guided = deepcopy(failed_rollout)
    guided["messages"][1] = message("user", "Which finding is present?\nInternal supervised answer: pneumothorax")
    with pytest.raises(ReflectionRejected, match="guided rollout"):
        build_reflection_trace(guided, completion)
    failed_rollout["event_trace"][0]["reason"] = "The provided answer identifies the region."
    with pytest.raises(ReflectionRejected, match="supervision-leak"):
        build_reflection_trace(failed_rollout, completion)
    completion.assert_not_called()


def test_no_answer_rollout_appends_final_turn_and_records_tool_errors(failed_rollout):
    failed_rollout.pop("messages")
    failed_rollout.update(status="no_answer", pred_answer="", final_answer_text=None)
    error = {"type": "tool_error", "round": 2, "error": "No second crop was produced"}
    failed_rollout["event_trace"][-1] = error
    prompt = build_reflection_prompt(failed_rollout)
    assert json.loads(prompt.messages[-1]["parts"][0]["text"].splitlines()[-1]) == {"answer": "pneumothorax"}
    assert prompt.messages[-2] == message("user", error["error"])
    output = build_reflection_trace(failed_rollout, lambda _: CORRECTION)
    assert output["event_trace"][:-1] == failed_rollout["event_trace"]
    assert output["event_trace"][-1]["round"] == 3
    assert "tool_error_event_present" in output["reflection_precheck"]["failure_reasons"]
    assert "reflection_original_answer" not in output


def test_bbox_validation_and_alternative_answer_matcher(failed_rollout):
    completion = Mock(return_value=CORRECTION)
    with pytest.raises(ReflectionRejected, match="bbox_invalid_order"):
        build_reflection_trace(failed_rollout, completion, gt_bbox=[30, 10, 10, 30])
    completion.assert_not_called()
    custom = CORRECTION.replace("<answer>pneumothorax", "<answer>air in the pleural space")
    outcome = build_reflection_trace(failed_rollout, lambda _: custom,
                                     verify_answer=lambda prediction, gold: (prediction, gold) == (
                                         "air in the pleural space", "pneumothorax"))
    assert outcome["pred_answer"] == "air in the pleural space" and outcome["correct"]
