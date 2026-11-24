from __future__ import annotations

import json
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

try:
    from .loop_utils import base64_to_pil_image, maybe_resize_image_for_qwen, pil_to_base64
except ImportError:
    from loop_utils import base64_to_pil_image, maybe_resize_image_for_qwen, pil_to_base64


DEFAULT_SANDBOX_URL = 'http://127.0.0.1:18901/jupyter_sandbox'
DEFAULT_TIMEOUT = 60.0
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


INITIALIZATION_CODE_TEMPLATE = """
from PIL import Image
import base64
from io import BytesIO

_img_base64 = "{base64_image}"
image_0 = Image.open(BytesIO(base64.b64decode(_img_base64))).convert("RGB")
""".strip()


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
        self._registered_indices: set[int] = set()

    @property
    def session_id(self) -> str:
        return f'qwen_agent_{self.sample_id}'

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
        self._registered_indices.clear()

    def execute(self, code: str, timeout: float = DEFAULT_TIMEOUT) -> Dict[str, Any]:
        payload = self._post_json(
            self.run_url,
            {
                'session_id': self.session_id,
                'code': code,
                'timeout': float(timeout),
            },
        )
        output = payload.get('output', {}) or {}
        images = output.get('images', []) or []
        pil_images = [maybe_resize_image_for_qwen(base64_to_pil_image(image_base64)) for image_base64 in images]
        return {
            'status': payload.get('status', 'success'),
            'execution_time': payload.get('execution_time'),
            'stdout': str(output.get('stdout', '')),
            'stderr': str(output.get('stderr', '')),
            'result': output.get('result'),
            'images': pil_images,
        }

    def initialize_original_image(self) -> None:
        image_base64 = self.runtime.get_image_base64(0)
        init_code = INITIALIZATION_CODE_TEMPLATE.format(
            base64_image=image_base64.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        )
        self.execute(init_code, timeout=20.0)
        self._registered_indices.add(0)

    def _register_indices_code(self, image_indices: List[int]) -> str:
        blocks: List[str] = [
            'from PIL import Image',
            'from io import BytesIO',
            'import base64',
        ]
        for image_index in image_indices:
            if image_index == 0:
                continue
            image_base64 = self.runtime.get_image_base64(image_index).replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
            var_name = runtime_index_to_python_var(image_index)
            blocks.append(f'_img_b64_{image_index} = """{image_base64}"""')
            blocks.append(f'{var_name} = Image.open(BytesIO(base64.b64decode(_img_b64_{image_index}))).convert("RGB")')
        return '\n'.join(blocks)

    def register_runtime_indices(self, image_indices: List[int]) -> None:
        pending = [idx for idx in image_indices if idx not in self._registered_indices]
        if not pending:
            return
        if pending == [0]:
            self.initialize_original_image()
            return
        code = self._register_indices_code(pending)
        if not code.strip():
            self._registered_indices.update(pending)
            return
        self.execute(code, timeout=20.0)
        self._registered_indices.update(pending)

    def register_code_output_images(self, images: List[Any], prefix: str = 'python_output') -> List[int]:
        new_indices: List[int] = []
        for image in images:
            image_base64 = pil_to_base64(image)
            new_index = self.runtime.add_image(image_base64=image_base64, prefix=prefix)
            new_indices.append(new_index)
        self.register_runtime_indices(new_indices)
        return new_indices
