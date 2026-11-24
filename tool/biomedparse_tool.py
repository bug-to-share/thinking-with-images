from __future__ import annotations

from typing import Optional, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool

from .bridge import BIOMEDPARSE_IMAGE_TYPES, DEFAULT_BIOMEDPARSE_URL, DEFAULT_REQUEST_TIMEOUT
from .common import ToolRuntimeContext, json_result, xyxy_to_qwen_bbox


@register_tool('biomedparse', allow_overwrite=True)
class BiomedParseTool(BaseTool):
    description = (
        'Use BioMedParse on one image in either segment or recognize mode. '
        'Segment mode takes a short medical entity target. '
        'Recognize mode requires one of the supported image_type values.'
    )
    parameters = {
        'type': 'object',
        'properties': {
            'image_index': {
                'type': 'integer',
                'description': '0-based image index in the current context.'
            },
            'mode': {
                'type': 'string',
                'description': 'Either `segment` or `recognize`.'
            },
            'description': {
                'type': 'string',
                'description': 'Short medical entity target to segment, such as liver, lung, spleen, optic disc, or left ventricle. Required for segment mode.'
            },
            'image_type': {
                'type': 'string',
                'enum': BIOMEDPARSE_IMAGE_TYPES,
                'description': 'Required for recognize mode and optional for segment mode. Allowed values: X-Ray-Chest, CT-Abdomen, MRI-Abdomen, MRI-Cardiac, MRI-FLAIR-Brain, MRI-T1-Gd-Brain.'
            },
            'p_value_threshold': {
                'type': 'number',
                'description': 'Optional p-value threshold used by recognize mode.'
            },
            'visualization': {
                'type': 'string',
                'description': 'Optional visualization mode: none, mask, overlay, both.'
            },
        },
        'required': ['image_index', 'mode']
    }

    def __init__(
        self,
        cfg: Optional[dict] = None,
        runtime: Optional[ToolRuntimeContext] = None,
        server_url: str = DEFAULT_BIOMEDPARSE_URL,
        request_timeout: float = DEFAULT_REQUEST_TIMEOUT,
    ):
        super().__init__(cfg)
        self.runtime = runtime or ToolRuntimeContext()
        self.server_url = server_url.rstrip('/')
        self.request_timeout = float(request_timeout)
        self.http = requests.Session()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            args = self._verify_json_format_args(params)
            image_index = int(args['image_index'])
            mode = str(args.get('mode', '')).strip().lower()
            if mode not in {'segment', 'recognize'}:
                raise ValueError('mode must be one of: segment, recognize')
            image_base64 = self.runtime.get_image_base64(image_index)
            request_payload = {
                'image_base64': image_base64,
            }
            visualization = args.get('visualization')
            if visualization is not None:
                request_payload['visualization'] = str(visualization).strip()
            if mode == 'segment':
                description = str(args.get('description', '')).strip()
                if not description:
                    raise ValueError('description is required for segment mode')
                request_payload['description'] = description
                image_type = args.get('image_type')
                if image_type is not None:
                    request_payload['image_type'] = str(image_type).strip()
                endpoint = 'segment'
            else:
                image_type = str(args.get('image_type', '')).strip()
                if not image_type:
                    raise ValueError('image_type is required for recognize mode')
                request_payload['image_type'] = image_type
                if args.get('p_value_threshold') is not None:
                    request_payload['p_value_threshold'] = float(args['p_value_threshold'])
                endpoint = 'recognize'

            response = self.http.post(
                f'{self.server_url}/{endpoint}',
                json=request_payload,
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            if result.get('status') != 'success':
                raise RuntimeError(result.get('message', f'BioMedParse server failed: {result}'))
            visualization_kind = 'none'
            new_image_index = None
            if result.get('overlay_base64'):
                visualization_kind = 'overlay'
                new_image_index = self.runtime.add_image(image_base64=result['overlay_base64'], prefix='biomedparse_overlay')
            elif result.get('mask_base64'):
                visualization_kind = 'mask'
                new_image_index = self.runtime.add_image(image_base64=result['mask_base64'], prefix='biomedparse_mask')
            image_width, image_height = self.runtime.get_image_size(image_index)
            if mode == 'segment':
                bbox = result.get('bbox')
                payload = {
                    'status': 'success',
                    'tool': self.name,
                    'mode': mode,
                    'image_index': image_index,
                    'description': request_payload.get('description'),
                    'image_type': request_payload.get('image_type'),
                    'bbox': xyxy_to_qwen_bbox(bbox, image_width, image_height) if bbox else None,
                    'foreground_pixels': result.get('foreground_pixels'),
                    'mask_mean_prob': result.get('mask_mean_prob'),
                    'p_value': result.get('p_value'),
                    'visualization_kind': visualization_kind,
                    'new_image_index': new_image_index,
                    'image_inventory': self.runtime.dump_image_inventory(),
                }
            else:
                normalized_predictions = []
                for item in result.get('predictions', []) or []:
                    bbox = item.get('bbox')
                    normalized_predictions.append(
                        {
                            'target': item.get('target'),
                            'bbox': xyxy_to_qwen_bbox(bbox, image_width, image_height) if bbox else None,
                            'foreground_pixels': item.get('foreground_pixels'),
                            'mask_mean_prob': item.get('mask_mean_prob'),
                        }
                    )
                payload = {
                    'status': 'success',
                    'tool': self.name,
                    'mode': mode,
                    'image_index': image_index,
                    'image_type': request_payload.get('image_type'),
                    'p_value_threshold': request_payload.get('p_value_threshold'),
                    'candidate_count': result.get('candidate_count'),
                    'num_predictions': result.get('num_predictions', len(normalized_predictions)),
                    'visualization_kind': visualization_kind,
                    'new_image_index': new_image_index,
                    'predictions': normalized_predictions,
                    'image_inventory': self.runtime.dump_image_inventory(),
                }
            return json_result(payload)
        except Exception as exc:
            return json_result({'status': 'error', 'tool': self.name, 'message': str(exc)})
