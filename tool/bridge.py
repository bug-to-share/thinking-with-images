from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List
from urllib.parse import urlparse

import requests

from .common import ToolRuntimeContext


DEFAULT_RETRIEVE_URL = 'http://127.0.0.1:6000'
DEFAULT_BIOMEDPARSE_URL = 'http://127.0.0.1:6001'
DEFAULT_MEDSAM3_URL = 'http://127.0.0.1:6002'
DEFAULT_SANDBOX_URL = 'http://127.0.0.1:18901/jupyter_sandbox'
DEFAULT_SANDBOX_EXEC_TIMEOUT = 20.0
DEFAULT_REQUEST_TIMEOUT = 180.0
BIOMEDPARSE_IMAGE_TYPES = [
    'X-Ray-Chest',
    'CT-Abdomen',
    'MRI-Abdomen',
    'MRI-Cardiac',
    'MRI-FLAIR-Brain',
    'MRI-T1-Gd-Brain',
]


def normalize_run_url(value: str | None) -> str:
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


def alternate_run_url(run_url: str) -> str | None:
    if run_url.endswith('/run_jupyter'):
        return run_url[: -len('/run_jupyter')] + '/jupyter_sandbox'
    if run_url.endswith('/jupyter_sandbox'):
        return run_url[: -len('/jupyter_sandbox')] + '/run_jupyter'
    return None


def build_python_prelude(image_base64_list: List[str], current_index: int | None = None) -> str:
    lines = [
        'import os',
        'import json',
        'import math',
        'from PIL import Image',
        'from io import BytesIO',
        'import base64',
        'try:',
        '    import numpy as np',
        'except Exception:',
        '    np = None',
        'try:',
        '    import pandas as pd',
        'except Exception:',
        '    pd = None',
        'try:',
        '    import matplotlib.pyplot as plt',
        'except Exception:',
        '    plt = None',
    ]
    for image_index, image_base64 in enumerate(image_base64_list):
        escaped_base64 = image_base64.replace('\\', '\\\\').replace('"""', '\\"\\"\\"')
        lines.extend(
            [
                f'_tool_image_base64_{image_index} = """{escaped_base64}"""',
                f'_tool_image_bytes_{image_index} = base64.b64decode(_tool_image_base64_{image_index})',
                f"image_{image_index} = Image.open(BytesIO(_tool_image_bytes_{image_index})).convert('RGB')",
            ]
        )
    if image_base64_list:
        selected = current_index if current_index is not None else 0
        selected = max(0, min(selected, len(image_base64_list) - 1))
        lines.extend(
            [
                f'current_image = image_{selected}',
                'image = current_image',
                'np_image = np.array(current_image) if np is not None else None',
            ]
        )
    return '\n'.join(lines) + '\n'


@dataclass
class ToolExecutionResult:
    name: str
    raw_payload: Dict[str, Any]
    model_payload: Dict[str, Any]
    new_indices: List[int]


class ToolBridge:
    def __init__(
        self,
        runtime: ToolRuntimeContext,
        retrieve_url: str = DEFAULT_RETRIEVE_URL,
        biomedparse_url: str = DEFAULT_BIOMEDPARSE_URL,
        medsam3_url: str = DEFAULT_MEDSAM3_URL,
        sandbox_url: str = DEFAULT_SANDBOX_URL,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        self.runtime = runtime
        self.retrieve_url = retrieve_url.rstrip('/')
        self.biomedparse_url = biomedparse_url.rstrip('/')
        self.medsam3_url = medsam3_url.rstrip('/')
        self.sandbox_run_url = normalize_run_url(sandbox_url)
        self.sandbox_clear_url = derive_clear_url(self.sandbox_run_url)
        self.request_timeout = float(request_timeout)
        self.http = requests.Session()

    @staticmethod
    def tool_specs() -> List[Dict[str, Any]]:
        image_type_text = ', '.join(BIOMEDPARSE_IMAGE_TYPES)
        return [
            {
                'name': 'python_exec',
                'description': 'Execute stateful Python code for image analysis. The code can directly use image_0, image_1, ... and should call plt.show() when it needs to return an image.',
                'arguments': {'code': 'python string'},
            },
            {
                'name': 'retrieve',
                'description': 'Retrieve concise medical knowledge summaries from the local knowledge server.',
                'arguments': {'query': 'str', 'topk': 'optional int'},
            },
            {
                'name': 'biomedparse',
                'description': 'Use BioMedParse in either segment or recognize mode. Segment mode takes a short medical entity target such as liver, lung, spleen, or optic disc. Recognize mode requires image_type from this list: ' + image_type_text + '. It returns multiple candidate structures/findings plus one overview visualization image.',
                'arguments': {
                    'image_index': 'int',
                    'mode': 'string: one of segment or recognize',
                    'description': 'optional short medical entity string for segment mode',
                    'image_type': 'optional string for recognize mode; allowed values: ' + image_type_text,
                    'p_value_threshold': 'optional float for recognize mode',
                    'visualization': 'optional string: none, mask, overlay, both',
                },
            },
            {
                'name': 'medsam3',
                'description': 'Use MedSAM3 to segment a short medical entity target from one image, such as liver, nodule, lesion, or left ventricle. It returns the highest-confidence result plus one visualization image.',
                'arguments': {'image_index': 'int', 'prompt': 'short medical entity string', 'threshold': 'optional float', 'nms_iou': 'optional float'},
            },
        ]

    def run(self, name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        if name in {'python_exec', 'code_sandbox'}:
            return self._run_code_sandbox(arguments)
        if name == 'retrieve':
            return self._run_retrieve(arguments)
        if name == 'biomedparse':
            return self._run_biomedparse(arguments)
        if name == 'medsam3':
            return self._run_medsam3(arguments)
        raise ValueError(f'unsupported tool: {name}')

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.http.post(url, json=payload, timeout=self.request_timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f'Expected JSON object from {url}, got: {type(data).__name__}')
        if data.get('status') == 'error':
            raise RuntimeError(data.get('message') or f'remote tool error from {url}')
        return data

    def _run_retrieve(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        query = str(arguments.get('query', '')).strip()
        if not query:
            raise ValueError('retrieve.query is required')
        topk = int(arguments.get('topk', 3))
        payload = self._post_json(
            f'{self.retrieve_url}/retrieve',
            {
                'query': query,
                'topk': topk,
            },
        )
        return ToolExecutionResult(
            name='retrieve',
            raw_payload=payload,
            model_payload=self._compact_payload('retrieve', payload),
            new_indices=[],
        )

    def _run_code_sandbox(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        code = str(arguments.get('code', ''))
        if not code.strip():
            raise ValueError('python_exec.code is required')

        session_id = f'agentic_rl_{self.runtime.sample_id}'
        reset_session = bool(arguments.get('reset_session', False))
        if reset_session:
            clear_response = self.http.post(
                self.sandbox_clear_url,
                json={'session_id': session_id},
                timeout=self.request_timeout,
            )
            clear_response.raise_for_status()

        image_index = arguments.get('image_index')
        current_index = int(image_index) if image_index is not None else None
        final_code = build_python_prelude(
            [record.image_base64 for record in self.runtime.image_records],
            current_index=current_index,
        ) + code

        timeout = float(arguments.get('timeout', DEFAULT_SANDBOX_EXEC_TIMEOUT))
        response = self.http.post(
            self.sandbox_run_url,
            json={
                'session_id': session_id,
                'code': final_code,
                'timeout': timeout,
            },
            timeout=self.request_timeout,
        )
        if response.status_code == 404:
            alt_url = alternate_run_url(self.sandbox_run_url)
            if alt_url:
                alt_response = self.http.post(
                    alt_url,
                    json={
                        'session_id': session_id,
                        'code': final_code,
                        'timeout': timeout,
                    },
                    timeout=self.request_timeout,
                )
                if alt_response.ok:
                    self.sandbox_run_url = alt_url
                    self.sandbox_clear_url = derive_clear_url(alt_url)
                    response = alt_response
        response.raise_for_status()
        payload = response.json()
        output = payload.get('output', {}) or {}

        new_indices: List[int] = []
        for idx, image_base64 in enumerate(output.get('images', []) or []):
            new_indices.append(self.runtime.add_image(image_base64=image_base64, prefix=f'python_exec_{idx}'))

        raw_payload = {
            'status': payload.get('status', 'success'),
            'tool': 'python_exec',
            'session_id': session_id,
            'run_url': self.sandbox_run_url,
            'reset_session': reset_session,
            'execution_time': payload.get('execution_time'),
            'timeout': timeout,
            'image_index': image_index,
            'result': output.get('result'),
            'stdout': str(output.get('stdout', '')),
            'stderr': str(output.get('stderr', '')),
            'num_output_images': len(new_indices),
            'new_image_indices': new_indices,
            'available_python_images': [f'image_{i}' for i in range(len(self.runtime.image_records))],
        }
        return ToolExecutionResult(
            name='python_exec',
            raw_payload=raw_payload,
            model_payload=self._compact_payload('python_exec', raw_payload),
            new_indices=new_indices,
        )

    def _run_biomedparse(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        image_index = int(arguments.get('image_index', 0))
        mode = str(arguments.get('mode', 'segment')).strip().lower()
        if mode not in {'segment', 'recognize'}:
            raise ValueError('biomedparse.mode must be one of: segment, recognize')
        request_payload: Dict[str, Any] = {
            'image_base64': self.runtime.get_image_base64(image_index),
        }
        if 'visualization' in arguments and arguments.get('visualization') is not None:
            request_payload['visualization'] = str(arguments['visualization']).strip()

        if mode == 'segment':
            description = str(arguments.get('description', '')).strip()
            if not description:
                raise ValueError('biomedparse.description is required for segment mode')
            request_payload['description'] = description
            if 'image_type' in arguments and arguments.get('image_type') is not None:
                request_payload['image_type'] = str(arguments['image_type']).strip()
            payload = self._post_json(f'{self.biomedparse_url}/segment', request_payload)
        else:
            image_type = str(arguments.get('image_type', '')).strip()
            if not image_type:
                raise ValueError('biomedparse.image_type is required for recognize mode')
            request_payload['image_type'] = image_type
            if 'p_value_threshold' in arguments and arguments.get('p_value_threshold') is not None:
                request_payload['p_value_threshold'] = float(arguments['p_value_threshold'])
            payload = self._post_json(f'{self.biomedparse_url}/recognize', request_payload)

        visualization_base64, visualization_kind = self._select_visualization_base64(payload)
        visualization_index = None
        if visualization_base64:
            visualization_index = self.runtime.add_image(
                image_base64=visualization_base64,
                prefix=f'biomedparse_{visualization_kind}',
            )
        if mode == 'segment':
            raw_payload = {
                'status': payload.get('status', 'success'),
                'tool': 'biomedparse',
                'mode': 'segment',
                'image_index': image_index,
                'description': request_payload.get('description'),
                'image_type': request_payload.get('image_type'),
                'bbox': self._bbox_to_xyxy_int(payload.get('bbox')),
                'foreground_pixels': payload.get('foreground_pixels'),
                'mask_mean_prob': payload.get('mask_mean_prob'),
                'p_value': payload.get('p_value'),
                'visualization_kind': visualization_kind,
                'new_image_index': visualization_index,
                'server_payload': payload,
            }
        else:
            normalized_predictions = []
            for item in payload.get('predictions', []) or []:
                normalized_predictions.append(
                    {
                        'target': item.get('target'),
                        'bbox': self._bbox_to_xyxy_int(item.get('bbox')),
                        'foreground_pixels': item.get('foreground_pixels'),
                        'mask_mean_prob': item.get('mask_mean_prob'),
                    }
                )
            raw_payload = {
                'status': payload.get('status', 'success'),
                'tool': 'biomedparse',
                'mode': 'recognize',
                'image_index': image_index,
                'image_type': request_payload.get('image_type'),
                'p_value_threshold': request_payload.get('p_value_threshold'),
                'candidate_count': payload.get('candidate_count'),
                'num_predictions': payload.get('num_predictions', len(normalized_predictions)),
                'visualization_kind': visualization_kind,
                'new_image_index': visualization_index,
                'predictions': normalized_predictions,
                'server_payload': payload,
            }
        return ToolExecutionResult(
            name='biomedparse',
            raw_payload=raw_payload,
            model_payload=self._compact_payload('biomedparse', raw_payload),
            new_indices=self._extract_new_indices(raw_payload),
        )

    def _run_medsam3(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        image_index = int(arguments.get('image_index', 0))
        prompt = str(arguments.get('prompt', '')).strip()
        if not prompt:
            raise ValueError('medsam3.prompt is required')
        request_payload: Dict[str, Any] = {
            'image_base64': self.runtime.get_image_base64(image_index),
            'prompt': prompt,
        }
        if 'threshold' in arguments and arguments.get('threshold') is not None:
            request_payload['threshold'] = float(arguments['threshold'])
        if 'nms_iou' in arguments and arguments.get('nms_iou') is not None:
            request_payload['nms_iou'] = float(arguments['nms_iou'])
        payload = self._post_json(f'{self.medsam3_url}/segment', request_payload)
        visualization_base64, visualization_kind = self._select_visualization_base64(payload)
        visualization_index = None
        if visualization_base64:
            visualization_index = self.runtime.add_image(
                image_base64=visualization_base64,
                prefix=f'medsam3_{visualization_kind}',
            )
        normalized_detections = []
        for item in payload.get('detections', []) or []:
            normalized_detections.append(
                {
                    'index': item.get('index'),
                    'score': item.get('score'),
                    'bbox': self._bbox_to_xyxy_int(item.get('bbox')),
                    'mask_bbox': self._bbox_to_xyxy_int(item.get('mask_bbox')),
                }
            )
        raw_payload = {
            'status': payload.get('status', 'success'),
            'tool': 'medsam3',
            'image_index': image_index,
            'prompt': prompt,
            'threshold': request_payload.get('threshold'),
            'nms_iou': request_payload.get('nms_iou'),
            'num_detections': payload.get('num_detections', 0),
            'top_score': payload.get('top_score'),
            'top_bbox': self._bbox_to_xyxy_int(payload.get('top_bbox')),
            'mask_bbox': self._bbox_to_xyxy_int(payload.get('mask_bbox')),
            'detections': normalized_detections,
            'visualization_kind': visualization_kind,
            'new_image_index': visualization_index,
            'message': payload.get('message'),
            'server_payload': payload,
        }
        return ToolExecutionResult(
            name='medsam3',
            raw_payload=raw_payload,
            model_payload=self._compact_payload('medsam3', raw_payload),
            new_indices=self._extract_new_indices(raw_payload),
        )


    @staticmethod
    def _extract_new_indices(payload: Dict[str, Any]) -> List[int]:
        indices: List[int] = []
        for key in ('new_image_index', 'mask_image_index'):
            value = payload.get(key)
            if value is not None:
                indices.append(int(value))
        for value in payload.get('new_image_indices', []) or []:
            indices.append(int(value))
        deduped: List[int] = []
        seen = set()
        for idx in indices:
            if idx not in seen:
                deduped.append(idx)
                seen.add(idx)
        return deduped

    @staticmethod
    def _compact_payload(name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if name == 'retrieve':
            compact_results = []
            for item in payload.get('results', []) or []:
                compact_results.append(
                    {
                        'summary': item.get('summary'),
                        'score': item.get('score'),
                        'rank': item.get('rank'),
                    }
                )
            return {
                'status': payload.get('status'),
                'query': payload.get('query'),
                'results': compact_results,
            }
        if name == 'python_exec':
            return {
                'status': payload.get('status'),
                'session_id': payload.get('session_id'),
                'execution_time': payload.get('execution_time'),
                'timeout': payload.get('timeout'),
                'image_index': payload.get('image_index'),
                'result': payload.get('result'),
                'stdout': payload.get('stdout'),
                'stderr': payload.get('stderr'),
                'num_output_images': payload.get('num_output_images'),
                'new_image_indices': payload.get('new_image_indices', []),
                'available_python_images': payload.get('available_python_images', []),
            }
        if name == 'biomedparse':
            compact = {
                'status': payload.get('status'),
                'mode': payload.get('mode'),
                'image_index': payload.get('image_index'),
                'visualization_kind': payload.get('visualization_kind'),
                'new_image_index': payload.get('new_image_index'),
            }
            if payload.get('mode') == 'segment':
                compact.update(
                    {
                        'description': payload.get('description'),
                        'image_type': payload.get('image_type'),
                        'bbox': payload.get('bbox'),
                        'foreground_pixels': payload.get('foreground_pixels'),
                        'mask_mean_prob': payload.get('mask_mean_prob'),
                        'p_value': payload.get('p_value'),
                    }
                )
            else:
                compact.update(
                    {
                        'image_type': payload.get('image_type'),
                        'p_value_threshold': payload.get('p_value_threshold'),
                        'candidate_count': payload.get('candidate_count'),
                        'num_predictions': payload.get('num_predictions'),
                        'predictions': payload.get('predictions', []),
                    }
                )
            return compact
        if name == 'medsam3':
            return {
                'status': payload.get('status'),
                'image_index': payload.get('image_index'),
                'prompt': payload.get('prompt'),
                'threshold': payload.get('threshold'),
                'nms_iou': payload.get('nms_iou'),
                'num_detections': payload.get('num_detections'),
                'top_score': payload.get('top_score'),
                'top_bbox': payload.get('top_bbox'),
                'mask_bbox': payload.get('mask_bbox'),
                'detections': payload.get('detections', []),
                'visualization_kind': payload.get('visualization_kind'),
                'new_image_index': payload.get('new_image_index'),
                'message': payload.get('message'),
            }
        return payload

    @staticmethod
    def _select_visualization_base64(payload: Dict[str, Any]) -> tuple[str | None, str]:
        overlay_base64 = payload.get('overlay_base64')
        if overlay_base64:
            return str(overlay_base64), 'overlay'
        mask_base64 = payload.get('mask_base64')
        if mask_base64:
            return str(mask_base64), 'mask'
        return None, 'none'

    @staticmethod
    def _bbox_to_xyxy_int(bbox: Any) -> list[int] | None:
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return None
        try:
            return [int(round(float(value))) for value in bbox]
        except Exception:
            return None
