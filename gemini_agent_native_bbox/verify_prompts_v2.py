from __future__ import annotations

import json
from typing import Any, Dict


VERIFY_V2_OUTPUT_SCHEMA = {
    "format_check": "pass/fail",
    "answer_correctness_check": "pass/fail",
    "supervision_leak_check": "pass/fail",
    "tool_call_validity_check": "pass/fail",
    "tool_choice_rationale_check": "pass/fail",
    "tool_result_integration_check": "pass/fail",
    "tool_trust_calibration_check": "pass/fail",
    "visual_evidence_consistency_check": "pass/fail",
    "unnecessary_tool_use_check": "pass/fail",
    "tool_necessity_check": "pass/fail",
    "tool_benefit_check": "positive/neutral/negative",
    "clinical_reasoning_quality_check": "pass/fail",
    "recommended_action": "keep_tool_trace/rewrite_as_direct/discard",
    "direct_rewrite": "<reason>...</reason><answer>...</answer> or empty string",
    "overall_verdict": "pass/fail",
    "score": 0,
    "failure_reasons": [],
    "explanation": "",
}


def build_verify_v2_system_prompt() -> str:
    schema_json = json.dumps(VERIFY_V2_OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
    return f"""You are an evidence-aware verifier for medical visual reasoning traces.

The trace was generated for SFT cold-start data construction. Your job is to decide whether
the trace is safe and useful training data.

You will receive:
- a medical VQA question and ground-truth answer
- the model's predicted answer
- a trace reconstructed from real tool interactions
- optionally, the original image and tool-generated images referenced by image_index
- deterministic precheck results extracted from the raw trace

Important scope:
- If real images are provided, inspect them directly when judging whether the tool was necessary or helpful.
- If real images are not provided, the token <image> only means an image existed upstream.
- Do not invent visual findings not present in the trace, tool responses, or provided images.
- Judge whether the trace is internally consistent, evidence-grounded, and suitable for training.

Allowed tools:
- python_exec: executes Python image inspection code and may return new images.
- sam3: segments a target using text + bbox + points and may return score, bbox, mask bbox, and overlay image.

Coordinate convention:
- The model works on a 1000x1000 visual canvas.
- Gemini-style bbox is [ymin, xmin, ymax, xmax] on a 0..1000 scale.
- Gemini-style point is [y, x] on a 0..1000 scale.
- Python crop code uses PIL absolute pixel order (xmin, ymin, xmax, ymax) on the same 1000x1000 canvas.
- Do not fail a trace merely because the order differs across sam3 bbox and Python crop; this is expected.

Evaluation dimensions:
1. format_check:
- The trace should use <reason> before each <tool_call> or <answer>.
- It should not contain unrecovered malformed turns.

2. answer_correctness_check:
- The final answer should be semantically equivalent to the ground-truth answer.
- Exact wording is not required.

3. supervision_leak_check:
- The assistant must not mention that it knows the ground truth, internal supervised answer,
  reference answer, gold answer, or similar hidden supervision.

4. tool_call_validity_check:
- Tool names and arguments must be valid.
- sam3 should have target text, bbox, and at least one positive point.
- python_exec should have executable-looking code.

5. tool_choice_rationale_check:
- The reason immediately before a tool call should justify why that tool is needed now.
- For sam3, it should justify the target and ROI.

6. tool_result_integration_check:
- The next reasoning step after a tool result should explicitly interpret or use the tool result.
- It should not ignore tool output and jump to an answer without integrating evidence.

7. tool_trust_calibration_check:
- Tool outputs are evidence, not oracle truth.
- Low-confidence or spatially questionable sam3 results should be treated cautiously.
- Penalize traces that say a weak/low-score/irrelevant tool result confirms the answer.
- Reward traces that down-weight, question, retry, or cross-check weak tool results.

8. visual_evidence_consistency_check:
- The final reasoning should be consistent with the tool observations and prior visual inspections.
- Penalize contradictions, invented observations, or using a tool result for a claim it does not support.

9. unnecessary_tool_use_check:
- Penalize redundant tool calls that do not add evidence or are poorly motivated.
- Do not penalize useful multi-step inspection.

10. tool_necessity_check:
- Pass only if tool use is genuinely needed for this item.
- Tool use is usually unnecessary for broad modality recognition, simple yes/no organ presence,
  disease/category classification, cell type classification, or questions answerable from the full image and choices.
- Tool use is more justified for localized anatomy, small lesion localization, spatial relationships,
  uncertain region-of-interest inspection, or when a crop/segmentation visibly changes the evidence.

11. tool_benefit_check:
- positive: the tool result clearly improves localization, resolves ambiguity, or corrects direct visual uncertainty.
- neutral: the tool is not harmful but adds little information.
- negative: the tool result/crop/mask distracts, removes global context, is irrelevant, or leads to a worse answer.

12. recommended_action:
- keep_tool_trace: final answer is correct and tool use is necessary or clearly beneficial.
- rewrite_as_direct: final answer is correct, but the tool is unnecessary, neutral, redundant, or only ceremonial.
- discard: final answer is wrong, supervision leaks, format is unrecovered, or tool use actively creates bad reasoning.

13. direct_rewrite:
- If recommended_action is rewrite_as_direct, write a concise direct-answer training trace:
  <reason>...</reason><answer>...</answer>
- The direct rewrite must not mention tools, hidden reference answers, or ground-truth supervision.
- The answer should be semantically equivalent to the ground-truth answer.
- If recommended_action is not rewrite_as_direct, return an empty string.

14. clinical_reasoning_quality_check:
- Reasoning should be medically plausible and should explain visual findings, anatomy/pathology,
  and relevant alternatives when appropriate.

Scoring:
- score is an integer from 0 to 10.
- 9-10: strong training trace.
- 7-8: usable but imperfect.
- 4-6: weak; usually filter or manually inspect.
- 0-3: bad trace; filter.

Hard failures:
- Supervision leakage should normally make overall_verdict fail.
- Incorrect final answer should normally make overall_verdict fail.
- Unrecovered invalid format should normally make overall_verdict fail.
- Blindly trusting a clearly weak or irrelevant tool result should usually fail tool_trust_calibration_check.
- If a correct trace uses unnecessary tools, overall_verdict may pass only if recommended_action is rewrite_as_direct;
  this means the sample is usable after repair, not as a tool trace.

Output strict JSON only. No markdown. No extra text.
Use exactly these keys:
{schema_json}
""".strip()


def build_verify_v2_user_prompt(record: Dict[str, Any], trace_text: str, precheck: Dict[str, Any]) -> str:
    payload = {
        "question_id": record.get("question_id"),
        "image": "<image>",
        "question": record.get("question"),
        "ground_truth_answer": record.get("gold_answer"),
        "predicted_answer": record.get("pred_answer"),
        "original_status": record.get("status"),
        "original_correct": record.get("correct"),
        "tool_call_counter": record.get("tool_call_counter", {}),
        "tool_calls_count": record.get("tool_calls_count", 0),
        "choices": record.get("choices", []),
        "benchmark": record.get("benchmark", ""),
        "modality": record.get("modality", ""),
        "location": record.get("location", ""),
        "content_type": record.get("content_type", ""),
        "model_canvas": record.get("model_canvas", {}),
        "allowed_tools": ["python_exec", "sam3"],
        "deterministic_precheck": precheck,
        "image_inventory": record.get("image_inventory", []),
        "trace": trace_text,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
