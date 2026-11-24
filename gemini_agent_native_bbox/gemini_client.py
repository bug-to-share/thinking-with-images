from __future__ import annotations

import os
import time
from dataclasses import dataclass
from typing import Any, Dict, List

import requests
try:
    from google import genai as google_genai
except Exception:
    google_genai = None


DEFAULT_BASE_URL = 'https://generativelanguage.googleapis.com/v1beta'
DEFAULT_MODEL = 'gemini-2.5-pro'


@dataclass
class GeminiResponse:
    text: str
    finish_reason: str
    raw_payload: Dict[str, Any]
    thought_text: str = ''


class GeminiClient:
    def __init__(
        self,
        api_key: str | None = None,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 240.0,
    ):
        self.api_key = api_key or os.getenv('GEMINI_API_KEY', '')
        if not self.api_key:
            raise ValueError('GEMINI_API_KEY is required')
        self.model = model
        self.base_url = base_url.rstrip('/')
        self.timeout = float(timeout)
        self.http = requests.Session()
        self.sdk_client = None
        if google_genai is not None:
            self.sdk_client = google_genai.Client(
                api_key=self.api_key,
                http_options={'base_url': self.base_url},
            )

    def generate(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None = None,
        top_k: int | None = None,
        stop_sequences: List[str] | None = None,
        include_thoughts: bool = False,
    ) -> GeminiResponse:
        last_error: Exception | None = None
        methods = []
        if self.sdk_client is not None:
            methods.append(self._generate_with_sdk)
        methods.append(self._generate_with_requests)
        for attempt, method in enumerate(methods):
            try:
                return method(
                    contents,
                    system_prompt,
                    temperature,
                    max_output_tokens,
                    top_p,
                    top_k,
                    stop_sequences,
                    include_thoughts,
                )
            except Exception as exc:
                last_error = exc
                if not _is_retryable_generation_error(exc) or attempt == len(methods) - 1:
                    raise
                time.sleep(0.5)
        if last_error is not None:
            raise last_error
        raise RuntimeError('Gemini generation failed without a concrete exception')

    def _generate_with_sdk(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None,
        top_k: int | None,
        stop_sequences: List[str] | None,
        include_thoughts: bool,
    ) -> GeminiResponse:
        config = {
            'system_instruction': system_prompt,
            'temperature': temperature,
            'max_output_tokens': int(max_output_tokens),
        }
        if top_p is not None:
            config['top_p'] = float(top_p)
        if top_k is not None:
            config['top_k'] = int(top_k)
        if stop_sequences:
            config['stop_sequences'] = list(stop_sequences)
        if include_thoughts:
            thinking_config: Dict[str, Any] = {}
            thinking_config['include_thoughts'] = True
            config['thinking_config'] = thinking_config
        response = self.sdk_client.models.generate_content(
            model=self.model,
            contents=contents,
            config=config,
        )
        payload = _response_to_dict(response)
        text, thought_text, finish_reason = _extract_sdk_text_and_finish_reason(response)
        if not text.strip():
            raise RuntimeError(f'Gemini SDK returned empty text: finish_reason={finish_reason!r}')
        return GeminiResponse(
            text=text,
            finish_reason=finish_reason,
            raw_payload=payload,
            thought_text=thought_text,
        )

    def _generate_with_requests(
        self,
        contents: List[Dict[str, Any]],
        system_prompt: str,
        temperature: float,
        max_output_tokens: int,
        top_p: float | None,
        top_k: int | None,
        stop_sequences: List[str] | None,
        include_thoughts: bool,
    ) -> GeminiResponse:
        url = f'{self.base_url}/models/{self.model}:generateContent?key={self.api_key}'
        body = {
            'systemInstruction': {
                'parts': [{'text': system_prompt}],
            },
            'contents': contents,
            'generationConfig': {
                'temperature': temperature,
                'maxOutputTokens': int(max_output_tokens),
            },
        }
        if top_p is not None:
            body['generationConfig']['topP'] = float(top_p)
        if top_k is not None:
            body['generationConfig']['topK'] = int(top_k)
        if stop_sequences:
            body['generationConfig']['stopSequences'] = list(stop_sequences)
        if include_thoughts:
            thinking_config: Dict[str, Any] = {}
            thinking_config['includeThoughts'] = True
            body['generationConfig']['thinkingConfig'] = thinking_config
        response = self.http.post(url, json=body, timeout=self.timeout)
        response.raise_for_status()
        payload = response.json()
        candidates = payload.get('candidates') or []
        if not candidates:
            prompt_feedback = payload.get('promptFeedback') or {}
            raise RuntimeError(f'Gemini returned no candidates: {prompt_feedback}')
        candidate = candidates[0]
        parts = ((candidate.get('content') or {}).get('parts') or [])
        text_parts: List[str] = []
        thought_parts: List[str] = []
        for part in parts:
            if not isinstance(part, dict):
                continue
            text = part.get('text', '')
            if not text:
                continue
            if part.get('thought'):
                thought_parts.append(text)
            else:
                text_parts.append(text)
        text = ''.join(text_parts)
        thought_text = '\n'.join(thought_parts).strip()
        finish_reason = str(candidate.get('finishReason', ''))
        if not text.strip():
            raise RuntimeError(f'Gemini requests returned empty text: finish_reason={finish_reason!r}')
        return GeminiResponse(
            text=text,
            finish_reason=finish_reason,
            raw_payload=payload,
            thought_text=thought_text,
        )


def _response_to_dict(response: Any) -> Dict[str, Any]:
    for attr in ('model_dump', 'dict'):
        func = getattr(response, attr, None)
        if callable(func):
            try:
                return func(exclude_none=True)
            except TypeError:
                try:
                    return func()
                except Exception:
                    pass
    return {'response_repr': repr(response)}


def _extract_sdk_text_and_finish_reason(response: Any) -> tuple[str, str, str]:
    candidates = getattr(response, 'candidates', None) or []
    if not candidates:
        raise RuntimeError('Gemini returned no candidates')
    candidate = candidates[0]
    content = getattr(candidate, 'content', None)
    parts = getattr(content, 'parts', None) or []
    texts: List[str] = []
    thoughts: List[str] = []
    for part in parts:
        text = getattr(part, 'text', None)
        if text:
            if getattr(part, 'thought', False):
                thoughts.append(str(text))
            else:
                texts.append(str(text))
    finish_reason = getattr(candidate, 'finish_reason', '') or getattr(candidate, 'finishReason', '')
    return ''.join(texts), '\n'.join(thoughts).strip(), str(finish_reason)


def _is_retryable_generation_error(error: Exception) -> bool:
    response = getattr(error, 'response', None)
    if response is not None:
        status_code = getattr(response, 'status_code', None)
        request_url = getattr(response, 'url', '') or ''
        if ':generateContent' in request_url and status_code in {400, 429, 500, 502, 503, 504}:
            return True
    message = str(error)
    retry_markers = (
        'Gemini returned no candidates',
        'returned empty text',
        'candidates.0.content',
        'model_attributes_type',
    )
    return any(marker in message for marker in retry_markers)
