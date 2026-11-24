from __future__ import annotations

import json
from typing import Any, Dict


VERIFY_OUTPUT_SCHEMA = {
    "format_check": "pass/fail",
    "tool_usage_check": "pass/fail",
    "tool_choice_rationale_check": "pass/fail",
    "post_tool_reason": "pass/fail",
    "clinical_correctness": "pass/fail",
    "final_answer_check": "correct/incorrect",
    "supervision_leak_check": "pass/fail",
    "tool_necessity_check": "pass/fail",
    "overall_verdict": "pass/fail",
    "score": 0,
    "explanation": "",
}


def build_verify_system_prompt() -> str:
    schema_json = json.dumps(VERIFY_OUTPUT_SCHEMA, ensure_ascii=False, indent=2)
    return f"""You are a medical visual reasoning trace verifier.

You will receive:
- an <image> placeholder token indicating an image existed upstream
- question and answer context
- one generated reasoning trace containing <reason>, <tool_call>, <tool_response>, and <answer> style steps
- optional static facts extracted from the trace structure

Your task is to strictly evaluate the quality of the trace for SFT cold-start data construction.

Important scope:
- You are NOT asked to visually inspect the real image.
- Treat <image> as a placeholder only.
- Judge the trace using question context, tool observations, structural consistency, and answer support.
- Do not hallucinate image evidence that is not present in the trace.

Project coordinate convention:
- The agent sees images in a resized/padded model working canvas.
- In the trace and <tool_response>, bbox-like fields shown to the model are Gemini-style normalized coordinates, not raw pixel coordinates.
- Gemini-style bbox format is [ymin, xmin, ymax, xmax] on a 0..1000 scale.
- Gemini-style point format is [y, x] on a 0..1000 scale.
- This applies to fields such as input_bbox, top_bbox, top_mask_bbox, detections[*].bbox, detections[*].mask_bbox, input_positive_points, and input_negative_points.
- The image_size field reports the actual image dimensions used by the backend tool, but the bbox/point fields in the trace may already have been converted back to the 0..1000 model coordinate system.
- Therefore, do NOT mark a trace as spatially invalid merely because a bbox value is larger than image_size width/height.
- Only mark coordinate usage as invalid if the reasoning itself contradicts the displayed model-coordinate convention, the bbox order is impossible, or the tool response explicitly reports an error.

Evaluation dimensions:
1. format_check
- Does the trace follow an interleaved reasoning structure?
- Are tool calls and answers well-formed?
- Are there malformed or invalid draft responses embedded in the trace?

2. tool_usage_check
- Are only allowed tools used?
- Are tool arguments plausible and consistent with the stated action?
- Are there obviously broken or nonsensical tool uses?

3. tool_choice_rationale_check
- Does the immediately preceding <reason> justify why this specific tool is needed now?
- Does it justify the region / bbox / point choice when applicable?

4. post_tool_reason
- After each <tool_response>, does the next <reason> explicitly use the returned evidence?
- Does the trace interpret tool outputs instead of ignoring them?

5. clinical_correctness
- Is the reasoning clinically and logically plausible based only on the question, tool observations, and provided trace?
- Penalize hallucinated findings, invented observations, or unsupported claims.

6. final_answer_check
- Does the final <answer> exactly match the provided ground-truth answer?

7. supervision_leak_check
- The reasoning must not reveal answer-side supervision.
- Penalize references such as “reference answer”, “provided answer”, “ground truth”, “gold answer”, “I will trust the answer”, or similar leakage.

8. tool_necessity_check
- Penalize redundant or weakly motivated tool calls.
- If a tool is used without clear information gain, mark this check as fail.

Scoring guidance:
- score is an integer from 0 to 10
- 9-10: strong SFT-quality trace
- 7-8: usable but with some issues
- 4-6: weak trace, likely needs filtering
- 0-3: poor trace, should be filtered out

Important rules:
- Base your judgment only on the provided image, question context, and trace.
- Here, “image” means only the placeholder token plus whatever the trace itself reveals about image inspection.
- Do not assume extra medical facts beyond what can be reasonably inferred.
- If the final answer is correct but the reasoning leaks supervision, the trace is still low quality.
- If static facts say there were tool errors, format errors, or exceptions, take that seriously.
- The trace format in this project uses <tool_response> for returned tool results, not <obs>.

Output strict JSON only. No markdown. No extra text.
Use exactly these keys:
{schema_json}
""".strip()


def build_verify_user_prompt(record: Dict[str, Any], trace_text: str, static_facts: Dict[str, Any]) -> str:
    payload = {
        "question_id": record.get("question_id"),
        "image": "<image>",
        "question": record.get("question"),
        "ground_truth_answer": record.get("gold_answer"),
        "predicted_answer": record.get("pred_answer"),
        "status": record.get("status"),
        "tool_call_counter": record.get("tool_call_counter", {}),
        "benchmark": record.get("benchmark", ""),
        "split": record.get("split", ""),
        "modality": record.get("modality", ""),
        "location": record.get("location", ""),
        "content_type": record.get("content_type", ""),
        "model_canvas": record.get("model_canvas", {}),
        "allowed_tools": ["python_exec", "sam3"],
        "static_trace_facts": static_facts,
        "trace": trace_text,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)
