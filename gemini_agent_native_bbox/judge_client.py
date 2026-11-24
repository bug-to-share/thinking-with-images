from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any

import requests


def normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        return ' '.join(normalize_text(item) for item in value)
    return str(value).strip()


@dataclass
class JudgeResult:
    correct: bool
    model_name: str
    raw_text: str
    request_payload: dict[str, Any]


class LLMJudgeClient:
    def __init__(
        self,
        base_url: str,
        api_key: str = 'EMPTY',
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key or 'EMPTY'
        self.timeout = float(timeout)
        self.http = requests.Session()
        self.model_name = self._discover_model_name()

    def _headers(self) -> dict[str, str]:
        return {
            'Authorization': f'Bearer {self.api_key}',
            'Content-Type': 'application/json',
        }

    def _discover_model_name(self) -> str:
        response = self.http.get(f'{self.base_url}/models', headers=self._headers(), timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        models = payload.get('data') or []
        if not models:
            raise RuntimeError(f'No models found at judge base: {self.base_url}')
        first = models[0]
        model_name = normalize_text(first.get('id'))
        if not model_name:
            raise RuntimeError(f'Judge model id missing: {payload}')
        return model_name

    def judge_semantic_equivalence(self, question: str, prediction: str, gold: str) -> JudgeResult:
        system_prompt = (
            'You are an expert evaluator. Decide whether the model answer is semantically equivalent '
            'to the standard answer given the question. Reply with exactly one word: CORRECT or INCORRECT.'
        )
        user_prompt = (
            'I will provide a question, a standard answer, and a model answer.\n\n'
            '---\n'
            '**Example 1:**\n'
            '[Question]: Is the countertop tan or blue?\n'
            '[Standard Answer]: The countertop is tan.\n'
            '[Model Answer]: tan\n'
            '[Judgement]: CORRECT\n'
            '---\n'
            '**Example 2:**\n'
            '[Question]: Are regions of the brain infarcted?\n'
            '[Standard Answer]: yes\n'
            '[Model Answer]: no\n'
            '[Judgement]: INCORRECT\n'
            '---\n'
            f'[Question]: {normalize_text(question)}\n'
            f'[Standard Answer]: {normalize_text(gold)}\n'
            f'[Model Answer]: {normalize_text(prediction)}\n'
            '[Judgement]:'
        )
        request_payload = {
            'model': self.model_name,
            'messages': [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_prompt},
            ],
            'temperature': 0.0,
            'extra_body': {'chat_template_kwargs': {'enable_thinking': False}},
        }
        response = self.http.post(
            f'{self.base_url}/chat/completions',
            headers=self._headers(),
            json=request_payload,
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        raw_text = normalize_text(payload['choices'][0]['message']['content'])
        verdicts = re.findall(r'\b(?:INCORRECT|CORRECT)\b', raw_text, re.IGNORECASE)
        if verdicts:
            verdict = verdicts[-1].upper()
            return JudgeResult(verdict == 'CORRECT', self.model_name, raw_text, request_payload)
        raise RuntimeError(f'Unexpected judge response: {raw_text}')


def judge_base_from_env() -> str:
    return normalize_text(os.environ.get('LLM_AS_A_JUDGE_BASE'))
