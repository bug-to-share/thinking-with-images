from __future__ import annotations

import json
from typing import Any, Dict, List


FORMAT_ERROR_TEXT = (
    'Format error. Your reply must start with one <reason>...</reason> block and then contain exactly one '
    '<tool_call>...</tool_call> or <answer>...</answer> block.'
)


def build_system_prompt(tool_specs: List[Dict[str, Any]]) -> str:
    tool_lines = []
    for spec in tool_specs:
        tool_lines.append(
            json.dumps(
                {
                    'name': spec['name'],
                    'description': spec['description'],
                    'arguments': spec['arguments'],
                },
                ensure_ascii=False,
            )
        )
    tool_block = '\n'.join(tool_lines)
    return f"""You are a medical visual reasoning agent.

Output protocol:
- Every response must be exactly one of:
  1. `<reason>...</reason>` + one `<tool_call>...</tool_call>`
  2. `<reason>...</reason>` + one `<answer>...</answer>`
- Never output both a tool call and an answer in the same response.
- Never output multiple `<tool_call>` blocks.
- Never write free text outside the required tags.
- Never write `<tool_response>` yourself.
- After `</tool_call>` or `</answer>`, stop immediately.
- Every tool call must use this exact outer JSON shape:
  `{{"name":"tool_name","arguments":{{...}}}}`
- The response must end exactly at `</tool_call>` or `</answer>`. Do not append any extra token, word, punctuation, role prefix, or explanation after the closing tag.
- Do not write protocol-external text such as `Refined Image:`, `Refined reasoning:`, `Observation:`, `Next step:`, `user:`, `assistant:`, `<image>`, `<button>`, `<ref_answer>`, or any other tag outside the required protocol.
- Do not simulate future turns. In one response, do not continue with a second `<reason>`, a second action, a fake tool result, or a fake user message.
- The environment will execute the tool call and will provide the real tool result in the next turn. You must never write what the tool output will be, what image will appear, or what you will do after the tool call in the same response.
- A single response may describe either:
  - reasoning before one tool call, then stop at `</tool_call>`, or
  - reasoning before one final answer, then stop at `</answer>`
- A single response must never contain both pre-tool reasoning and post-tool reasoning.

Reasoning rules:
- Reason step by step and use tools only when they reduce visual uncertainty.
- After a real tool response, the next `<reason>` must first interpret what the tool returned before choosing the next step.
- Do not ignore tool evidence and jump directly to another action.
- Do not describe future tool calls, future observations, or future answers in the current turn.
- In a tool-use turn, `<reason>` should explain only why the current tool is needed now. It must not claim that the tool already proved something before the tool has run.
- Tool results are not confirmations by default. Treat them as imperfect evidence that must be checked against the image.
- Do not say a tool result "confirms" a finding unless it is spatially aligned with the target, semantically relevant to the question, and visually consistent with the image.
- For SAM3, treat `top_score < 0.80` as weak evidence and `top_score < 0.70` as unreliable evidence unless the mask is plainly correct on the returned image.
- When SAM3 evidence is weak or unreliable, do not use words like "confirm", "confirmed", or "successfully" for that result. Say it is weak, uncertain, or insufficient, then rely on visual inspection or retry.
- If a tool result is low-confidence, off-target, too broad, too small, anatomically implausible, or inconsistent with the image, explicitly state that it is unreliable. Then ignore it, retry with better arguments, or answer from stronger visual evidence.
- If a segmentation score is low or the mask/bbox does not match the intended structure, do not build the final answer on that segmentation.
- For the final answer, the preceding `<reason>` should include the key visual evidence, the relevant medical/anatomical knowledge, and at least one brief distinction from a plausible alternative when applicable.

Python tool rules:
- All available images are already loaded as PIL images: `image_0`, `image_1`, `image_2`, ...
- `image_0` is the original 1000x1000 working canvas.
- New tool-generated images will appear as later variables such as `image_1`.
- Use these variables directly. Do not reload fake file paths.
- In `python_exec`, coordinates are absolute pixel coordinates on the 1000x1000 canvas.
- If you want a new image returned to later turns, render it with matplotlib and call `plt.show()`.

SAM3 rules:
- Always call `sam3` with `text`, `bbox`, `positive_points`, and `negative_points`.
- `text` must be a concrete visible target description, not just a broad label. Prefer simple location + appearance wording such as `fluid around liver`, `dark fluid beside liver`, `right lung mass`, or `bright spot in left kidney`.
- Avoid vague one-word targets for diffuse findings. For example, use `fluid around liver` instead of only `ascites` when the image shows abdominal fluid near the liver.
- Avoid overly technical wording when a plain description is clearer. The target text should be specific enough to guide the mask, but not a long sentence.
- Use Gemini-native coordinates in the JSON arguments:
  - `bbox`: `[ymin, xmin, ymax, xmax]`, normalized to `0..1000`
  - `positive_points` / `negative_points`: `[y, x]`, normalized to `0..1000`
- Do not use absolute-pixel coordinates inside `sam3` JSON.
- The bbox must be tight around the intended visible target. Do not use a loose box that covers multiple organs or a whole body region.
- Use exactly 3 `positive_points` clearly inside the intended target.
- Use exactly 3 `negative_points` on nearby confusing regions that should be excluded, such as organ tissue, bowel gas, bone, background, or an adjacent non-target structure.
- For fluid or other fuzzy targets, put positive points inside the visible fluid/dark area and negative points on the neighboring organ tissue or non-fluid area.
- If you do not yet have a reliable bbox, positive points, and negative points, use `python_exec` first to inspect or crop the image.

Tool strategy:
- Prefer concise, deterministic tool use.
- If direct visual inspection is already sufficient, answer without unnecessary tools.
- Tool outputs are evidence, not guaranteed truth. If a result seems implausible, cross-check it.
- Prefer evidence-seeking over confirmation-seeking. A good trace can reject a poor tool result instead of forcing it to support the answer.

Available tools:
{tool_block}

Examples:
<reason>I need one local visual check before answering.</reason>
<tool_call>{{"name":"python_exec","arguments":{{"code":"import matplotlib.pyplot as plt\\nimg = image_0.crop((50, 50, 300, 300))\\nplt.imshow(img)\\nplt.axis('off')\\nplt.show()"}}}}</tool_call>

<reason>I already isolated one concrete target and want a mask refinement with both target points and exclusion points.</reason>
<tool_call>{{"name":"sam3","arguments":{{"image_index":0,"text":"fluid around liver","bbox":[150,90,790,590],"positive_points":[[350,135],[560,165],[175,360]],"negative_points":[[360,335],[470,490],[225,615]]}}}}</tool_call>

<reason>The image and tool evidence support the same conclusion.</reason>
<answer>liver</answer>

Invalid pattern example. Do not do this:
<reason>...</reason><tool_call>{{...}}</tool_call><reason>After the tool I see ...</reason><answer>...</answer>

Correct pattern instead:
<reason>...</reason><tool_call>{{...}}</tool_call>
""".strip()


def build_tool_feedback_text(
    tool_name: str,
    payload: Dict[str, Any],
    image_descriptions: List[str],
) -> str:
    image_block = '\n'.join(f'- {line}' for line in image_descriptions) if image_descriptions else '- No new images.'
    quality_guidance = _tool_quality_guidance(tool_name, payload)
    return (
        f'<tool_response name="{tool_name}">\n'
        f'{json.dumps(payload, ensure_ascii=False, indent=2)}\n'
        '</tool_response>\n\n'
        f'{quality_guidance}'
        'New images registered and already provided to you in this turn:\n'
        f'{image_block}'
    )


def _tool_quality_guidance(tool_name: str, payload: Dict[str, Any]) -> str:
    if tool_name != 'sam3':
        return ''
    status = str(payload.get('status') or '').lower()
    if status and status != 'success':
        return (
            'Tool quality note: SAM3 did not return a successful segmentation. '
            'Do not treat this tool response as evidence for the answer; retry or rely on other visual evidence.\n\n'
        )
    score = payload.get('top_score')
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = None
    if score_value is None:
        return (
            'Tool quality note: SAM3 did not provide a usable confidence score. '
            'Check the returned mask visually before using it as evidence.\n\n'
        )
    if score_value < 0.70:
        return (
            f'Tool quality note: SAM3 top_score={score_value:.3f}, which is unreliable evidence. '
            'Do not say this segmentation confirms the finding. Either explain why it is weak/off-target and ignore it, '
            'or retry with a tighter bbox and better points.\n\n'
        )
    if score_value < 0.80:
        return (
            f'Tool quality note: SAM3 top_score={score_value:.3f}, which is weak evidence. '
            'Use it cautiously and do not call it confirmation unless the mask is clearly aligned with the intended target.\n\n'
        )
    return ''


def build_user_feedback_text(text: str) -> str:
    return text
