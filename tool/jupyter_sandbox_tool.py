from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

import requests
from qwen_agent.tools.base import BaseTool, register_tool

from .common import ToolRuntimeContext, json_result


DEFAULT_SANDBOX_URL = 'http://127.0.0.1:18901/jupyter_sandbox'
DEFAULT_TIMEOUT = 20.0
DEFAULT_REQUEST_TIMEOUT = 120.0


def normalize_run_url(value: Optional[str]) -> str:
    raw = (value or DEFAULT_SANDBOX_URL).strip().rstrip('/')
    if raw.endswith('/run_jupyter') or raw.endswith('/jupyter_sandbox'):
        return raw
    parsed = urlparse(raw)
    if parsed.scheme and parsed.netloc:
        return raw + '/run_jupyter'
    return DEFAULT_SANDBOX_URL


def derive_clear_url(run_url: str) -> str:
    if run_url.endswith('/run_jupyter'):
        return run_url[: -len('/run_jupyter')] + '/clear_session'
    if run_url.endswith('/jupyter_sandbox'):
        return run_url[: -len('/jupyter_sandbox')] + '/clear_session'
    return run_url.rstrip('/') + '/clear_session'


def alternate_run_url(run_url: str) -> Optional[str]:
    if run_url.endswith('/run_jupyter'):
        return run_url[: -len('/run_jupyter')] + '/jupyter_sandbox'
    if run_url.endswith('/jupyter_sandbox'):
        return run_url[: -len('/jupyter_sandbox')] + '/run_jupyter'
    return None


def build_image_init_code(image_base64: str) -> str:
    escaped_base64 = image_base64.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
    return f'''
from PIL import Image
from io import BytesIO
import base64
import numpy as np

_tool_image_base64 = """{escaped_base64}"""
_tool_image_bytes = base64.b64decode(_tool_image_base64)
image = Image.open(BytesIO(_tool_image_bytes)).convert("RGB")
current_image = image
np_image = np.array(image)
'''


@register_tool('jupyter_sandbox', allow_overwrite=True)
class JupyterSandboxTool(BaseTool):
    description = (
        'Execute stateful Python code in a multimodal Jupyter-style sandbox. '
        'Optional current image injection exposes `image`, `current_image`, and `np_image` variables.'
    )
    parameters = {
        'type': 'object',
        'properties': {
            'code': {
                'type': 'string',
                'description': 'Python code to execute inside the persistent sandbox session.'
            },
            'image_index': {
                'type': 'integer',
                'description': 'Optional 0-based image index to inject as variable `image` before executing the code.'
            },
            'timeout': {
                'type': 'number',
                'description': 'Execution timeout in seconds for the sandbox code. Default 20.'
            },
            'reset_session': {
                'type': 'boolean',
                'description': 'Whether to clear the current sandbox session before executing code.'
            }
        },
        'required': ['code']
    }

    def __init__(
        self,
        cfg: Optional[dict] = None,
        runtime: Optional[ToolRuntimeContext] = None,
        sandbox_url: Optional[str] = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        super().__init__(cfg)
        self.runtime = runtime or ToolRuntimeContext()
        self.run_url = normalize_run_url(sandbox_url)
        self.clear_url = derive_clear_url(self.run_url)
        self.request_timeout = float(request_timeout)
        self.http = requests.Session()

    def _session_id(self) -> str:
        return f'agentic_rl_{self.runtime.sample_id}'

    def _clear_session(self, session_id: str) -> Dict[str, Any]:
        response = self.http.post(
            self.clear_url,
            json={'session_id': session_id},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        return response.json()

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.http.post(url, json=payload, timeout=self.request_timeout)
        if response.status_code == 404:
            alt_url = alternate_run_url(url)
            if alt_url:
                alt_response = self.http.post(alt_url, json=payload, timeout=self.request_timeout)
                if alt_response.ok:
                    self.run_url = alt_url
                    self.clear_url = derive_clear_url(alt_url)
                    return alt_response.json()
        response.raise_for_status()
        return response.json()

    def _save_output_images(self, images: List[str]) -> List[int]:
        new_indices: List[int] = []
        for idx, image_base64 in enumerate(images or []):
            new_indices.append(self.runtime.add_image(image_base64=image_base64, prefix=f'jupyter_output_{idx}'))
        return new_indices

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            args = self._verify_json_format_args(params)
            code = str(args['code'])
            if not code.strip():
                raise ValueError('code must not be empty')

            session_id = self._session_id()
            reset_session = bool(args.get('reset_session', False))
            if reset_session:
                self._clear_session(session_id)

            image_index = args.get('image_index')
            final_code = code
            if image_index is not None:
                image_base64 = self.runtime.get_image_base64(int(image_index))
                final_code = build_image_init_code(image_base64=image_base64) + '\n' + code

            timeout = float(args.get('timeout', DEFAULT_TIMEOUT))
            payload = self._post_json(
                self.run_url,
                {
                    'session_id': session_id,
                    'code': final_code,
                    'timeout': timeout,
                },
            )
            status = payload.get('status', 'success')
            output = payload.get('output', {}) or {}
            stdout = str(output.get('stdout', ''))
            stderr = str(output.get('stderr', ''))
            result = output.get('result')
            images = output.get('images', []) or []
            new_indices = self._save_output_images(images)

            return json_result({
                'status': status,
                'tool': self.name,
                'session_id': session_id,
                'run_url': self.run_url,
                'reset_session': reset_session,
                'execution_time': payload.get('execution_time'),
                'timeout': timeout,
                'image_index': image_index,
                'result': result,
                'stdout': stdout,
                'stderr': stderr,
                'num_output_images': len(new_indices),
                'new_image_indices': new_indices,
                'image_inventory': self.runtime.dump_image_inventory(),
            })
        except Exception as exc:
            return json_result({'status': 'error', 'tool': self.name, 'message': str(exc)})


@register_tool('python_exec', allow_overwrite=True)
class PythonExecTool(JupyterSandboxTool):
    description = JupyterSandboxTool.description
    parameters = JupyterSandboxTool.parameters


@register_tool('code_sandbox', allow_overwrite=True)
class CodeSandboxTool(JupyterSandboxTool):
    description = JupyterSandboxTool.description
    parameters = JupyterSandboxTool.parameters
