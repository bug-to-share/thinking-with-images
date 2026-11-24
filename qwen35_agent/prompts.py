from __future__ import annotations

import json
from typing import List, Dict, Any


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
    return f"""You are a medical visual reasoning agent. Keep working until the question is solved.

You operate in a thinking-with-images loop:
- In your response, output exactly one action block:
  - `<code>```python ... ```</code>` when you need python image analysis.
  - `<tool_call>\n{{"name": "...", "arguments": {{...}}}}\n</tool_call>` when you need an external tool.
  - `<answer>...</answer>` only when you are ready to finish.

Python environment rules:
- All currently available images are already loaded into the notebook as global variables.
- The original image is `image_0`.
- If new images are produced later, they will be appended as `image_1`, `image_2`, ... and also shown to you in the conversation.
- Use `image_k` variables directly for analysis, where `k` is the runtime `image_index`.
- Use matplotlib `plt.show()` when you want the resulting image to be returned and shown in the next turn.

Tool rules:
- Only call one action per turn: either python code or one tool call.
- All bbox coordinates use Qwen grounding format `[x1, y1, x2, y2]` on a relative `0..1000` grid.
- Tool-generated images will be returned to you as real images and also registered as later python variables.
- Before the final answer, you must use at least one dedicated external tool from the available tool list.
- For localized anatomy, lesion, or fine-grained medical-image questions, inspect at least one region with a dedicated tool before answering.
- If the current tool result is consistent with your visual inspection and medical knowledge, do not keep searching for extra conflict.
- Only perform extra tool calls when the evidence is conflicting, uncertain, or still insufficient to clearly support one option.
- When evidence truly conflicts, prefer cross-checking with another tool, another prompt, or another region instead of trusting one uncertain observation.
- Stop using tools when you already have enough evidence.

Available tools:
{tool_block}

Few-shot formatting examples:

Example 1: correct `biomedparse` tool call
<tool_call>
{{"name": "biomedparse", "arguments": {{"image_index": 0, "description": "optic disc"}}}}
</tool_call>

Example 2: correct `retrieve` tool call
<tool_call>
{{"name": "retrieve", "arguments": {{"query": "optic disc retinal anatomy", "topk": 3}}}}
</tool_call>

Example 3: correct `medsam3` tool call
<tool_call>
{{"name": "medsam3", "arguments": {{"image_index": 0, "prompt": "optic disc"}}}}
</tool_call>

Example 4: correct python code action
<code>
```python
import matplotlib.pyplot as plt

img = image_0
plt.imshow(img)
plt.axis("off")
plt.show()
```
</code>

Example 5: correct final answer
<answer>B</answer>


Output contract:
- Every turn response must contain exactly one of `<code>...</code>`, `<tool_call>...</tool_call>`, or `<answer>...</answer>`.
- Final answer must be a single option letter A/B/C/D/E inside `<answer>`.
""".strip()


def build_user_prompt(question_text: str) -> str:
    return (
        f'<image>\n{question_text}\n\n'
        'Use your internal reasoning channel for step-by-step thinking. '
        'For this kind of medical VQA, try at least one external tool before answering when the question depends on localized visual evidence. '
        'If the current evidence is already consistent and sufficient, answer directly. '
        'Only continue with extra tool calls when the evidence is conflicting, uncertain, or still insufficient. '
        'If you need more visual evidence, use python or a tool. '
        'When finished, output only `<answer>X</answer>` where X is one of A/B/C/D/E.'
    )


def build_code_feedback_text(stdout: str, stderr: str, image_descriptions: List[str]) -> str:
    image_block = '\n'.join(f'- {line}' for line in image_descriptions) if image_descriptions else '- No new images.'
    return (
        'Python execution result:\n'
        f'stdout:\n```\n{stdout}\n```\n\n'
        f'stderr:\n```\n{stderr}\n```\n\n'
        f'New images registered:\n{image_block}'
    )


def build_tool_feedback_text(tool_name: str, payload: Dict[str, Any], image_descriptions: List[str]) -> str:
    image_block = '\n'.join(f'- {line}' for line in image_descriptions) if image_descriptions else '- No new images.'
    return (
        f'<tool_response name="{tool_name}">\n'
        f'{json.dumps(payload, ensure_ascii=False, indent=2)}\n'
        f'</tool_response>\n\n'
        f'New images registered:\n{image_block}'
    )


FORMAT_ERROR_TEXT = (
    'Format error. Your visible reply did not contain exactly one valid `<code>...</code>`, '
    '`<tool_call>...</tool_call>`, or `<answer>...</answer>` block. '
    'Please continue with exactly one valid action.'
)
