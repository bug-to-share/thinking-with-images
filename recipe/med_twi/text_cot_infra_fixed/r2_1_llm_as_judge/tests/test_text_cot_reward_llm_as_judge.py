from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch


REWARD_PATH = Path(__file__).parents[1] / "rewards" / "text_cot_reward_llm_as_judge.py"
SPEC = importlib.util.spec_from_file_location("text_cot_reward_llm_as_judge", REWARD_PATH)
assert SPEC is not None and SPEC.loader is not None
reward = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(reward)


class LLMAsJudgeRewardTest(unittest.IsolatedAsyncioTestCase):
    async def test_invalid_format_is_hard_gated_without_judge(self):
        judge = AsyncMock()
        with patch.object(reward, "judge_semantic_equivalence", judge):
            result = await reward.compute_score(
                "slake_text_cot",
                "The answer is MRI.",
                "MRI",
                {"question": "What modality is shown?"},
            )
        self.assertEqual(result["score"], 0.0)
        self.assertEqual(result["format_reward"], 0.0)
        judge.assert_not_awaited()

    async def test_exact_match_short_circuits_judge(self):
        judge = AsyncMock()
        with patch.object(reward, "judge_semantic_equivalence", judge):
            result = await reward.compute_score(
                "slake_text_cot",
                "<reason>Visible imaging features.</reason><answer> mri. </answer>",
                "MRI",
                {"question": "What modality is shown?"},
            )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["correctness"], 1.0)
        self.assertEqual(result["exact_match"], 1.0)
        self.assertEqual(result["judge_called"], 0.0)
        judge.assert_not_awaited()

    async def test_semantic_match_receives_full_correctness_reward(self):
        with patch.object(
            reward,
            "judge_semantic_equivalence",
            AsyncMock(return_value=(1.0, 0.0)),
        ) as judge:
            result = await reward.compute_score(
                "slake_text_cot",
                "<reason>The organ lies in the thoracic cavity.</reason><answer>thorax</answer>",
                "chest",
                {"question": "Where is the organ located?"},
            )
        self.assertEqual(result["score"], 1.0)
        self.assertEqual(result["exact_match"], 0.0)
        self.assertEqual(result["judge_positive"], 1.0)
        self.assertEqual(result["judge_called"], 1.0)
        judge.assert_awaited_once()

    async def test_semantic_mismatch_keeps_only_format_reward(self):
        with patch.object(
            reward,
            "judge_semantic_equivalence",
            AsyncMock(return_value=(0.0, 0.0)),
        ):
            result = await reward.compute_score(
                "slake_text_cot",
                "<reason>The image appears to be CT.</reason><answer>CT</answer>",
                "MRI",
                {"question": "What modality is shown?"},
            )
        self.assertAlmostEqual(result["score"], 0.3)
        self.assertEqual(result["correctness"], 0.0)
        self.assertEqual(result["judge_error"], 0.0)

    async def test_judge_failure_is_observable_and_fails_closed(self):
        with patch.object(
            reward,
            "judge_semantic_equivalence",
            AsyncMock(return_value=(0.0, 1.0)),
        ):
            result = await reward.compute_score(
                "slake_text_cot",
                "<reason>The answer is a synonym.</reason><answer>thorax</answer>",
                "chest",
                {"question": "Where is the organ located?"},
            )
        self.assertAlmostEqual(result["score"], 0.3)
        self.assertEqual(result["judge_called"], 1.0)
        self.assertEqual(result["judge_error"], 1.0)


if __name__ == "__main__":
    unittest.main()
