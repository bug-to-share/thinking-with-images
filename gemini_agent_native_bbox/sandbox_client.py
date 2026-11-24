from __future__ import annotations

import sys
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse

import requests

THIS_DIR = Path(__file__).resolve().parent
PARENT_DIR = THIS_DIR.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from tool.common import ToolRuntimeContext


DEFAULT_SANDBOX_URL = 'http://127.0.0.1:18901/jupyter_sandbox'
DEFAULT_REQUEST_TIMEOUT = 180.0


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


def runtime_index_to_python_var(image_index: int) -> str:
    return f'image_{image_index}'


class JupyterSandboxSession:
    def __init__(
        self,
        runtime: ToolRuntimeContext,
        sample_id: Optional[str] = None,
        sandbox_url: Optional[str] = None,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        self.runtime = runtime
        self.sample_id = sample_id or f'session_{uuid.uuid4().hex[:8]}'
        self.run_url = normalize_run_url(sandbox_url)
        self.clear_url = derive_clear_url(self.run_url)
        self.request_timeout = float(request_timeout)
        self.http = requests.Session()

    @property
    def session_id(self) -> str:
        return f'gemini_agent_{self.sample_id}'

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

    def clear(self) -> None:
        self.http.post(self.clear_url, json={'session_id': self.session_id}, timeout=self.request_timeout)

    def register_runtime_indices(self, image_indices: List[int]) -> None:
        if not image_indices:
            return
        lines = [
            'from PIL import Image',
            'from io import BytesIO',
            'import base64',
        ]
        for image_index in image_indices:
            image_base64 = self.runtime.get_image_base64(image_index).replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
            lines.append(f'_img_b64_{image_index} = """{image_base64}"""')
            lines.append(
                f'{runtime_index_to_python_var(image_index)} = '
                f'Image.open(BytesIO(base64.b64decode(_img_b64_{image_index}))).convert("RGB")'
            )
        self.execute('\n'.join(lines), timeout=20.0)

    def execute(self, code: str, timeout: float = 60.0) -> Dict[str, Any]:
        payload = self._post_json(
            self.run_url,
            {
                'session_id': self.session_id,
                'code': code,
                'timeout': float(timeout),
            },
        )
        output = payload.get('output', {}) or {}
        return {
            'status': payload.get('status', 'success'),
            'execution_time': payload.get('execution_time'),
            'stdout': str(output.get('stdout', '')),
            'stderr': str(output.get('stderr', '')),
            'result': output.get('result'),
            'images': output.get('images', []) or [],
        }

    def register_code_output_images(self, images: List[str], prefix: str = 'python_output') -> List[int]:
        new_indices: List[int] = []
        for image_base64 in images:
            new_index = self.runtime.add_image(image_base64=image_base64, prefix=prefix)
            new_indices.append(new_index)
        self.register_runtime_indices(new_indices)
        return new_indices
