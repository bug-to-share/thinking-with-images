from __future__ import annotations

from typing import Optional, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool

from .bridge import DEFAULT_REQUEST_TIMEOUT, DEFAULT_RETRIEVE_URL
from .common import json_result, normalize_text


@register_tool('retrieve', allow_overwrite=True)
class RetrieveTool(BaseTool):
    description = 'Retrieve reusable medical knowledge from the local adaptive knowledge base.'
    parameters = {
        'type': 'object',
        'properties': {
            'query': {
                'type': 'string',
                'description': 'Concise reusable medical retrieval query in canonical terminology.'
            },
            'topk': {
                'type': 'integer',
                'description': 'Number of retrieved knowledge entries to return.',
                'minimum': 1,
                'maximum': 10,
            }
        },
        'required': ['query']
    }

    def __init__(
        self,
        cfg: Optional[dict] = None,
        server_url: str = DEFAULT_RETRIEVE_URL,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        super().__init__(cfg)
        self.server_url = server_url.rstrip('/')
        self.request_timeout = float(request_timeout)
        self.http = requests.Session()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            args = self._verify_json_format_args(params)
            query = normalize_text(args['query'])
            topk = int(args.get('topk', 3))
            response = self.http.post(
                f'{self.server_url}/retrieve',
                json={'query': query, 'topk': topk},
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            payload = response.json()
            if payload.get('status') != 'success':
                raise RuntimeError(payload.get('message', f'retrieve server failed: {payload}'))
            return json_result({
                'status': 'success',
                'tool': self.name,
                'query': query,
                'topk': topk,
                'results': payload.get('results', []),
            })
        except Exception as exc:
            return json_result({'status': 'error', 'tool': self.name, 'message': str(exc)})
