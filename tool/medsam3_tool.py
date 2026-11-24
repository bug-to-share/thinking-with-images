from __future__ import annotations

from typing import Optional, Union

import requests
from qwen_agent.tools.base import BaseTool, register_tool

from .bridge import DEFAULT_MEDSAM3_URL, DEFAULT_REQUEST_TIMEOUT
from .common import ToolRuntimeContext, json_result, xyxy_to_qwen_bbox


@register_tool('medsam3', allow_overwrite=True)
class MedSAM3Tool(BaseTool):
    description = 'Segment a short medical entity target using MedSAM3 and return the highest-confidence detection plus one visualization image.'
    parameters = {
        'type': 'object',
        'properties': {
            'image_index': {
                'type': 'integer',
                'description': '0-based image index in the current context.'
            },
            'prompt': {
                'type': 'string',
                'description': 'Short medical entity target to segment, such as liver, nodule, lesion, tumor, or left ventricle.'
            },
            'threshold': {
                'type': 'number',
                'description': 'Optional detection confidence threshold. Default 0.5.'
            },
            'nms_iou': {
                'type': 'number',
                'description': 'Optional NMS IoU threshold. Default 0.5.'
            }
        },
        'required': ['image_index', 'prompt']
    }

    def __init__(
        self,
        cfg: Optional[dict] = None,
        runtime: Optional[ToolRuntimeContext] = None,
        server_url: str = DEFAULT_MEDSAM3_URL,
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
            prompt = str(args['prompt']).strip()
            threshold = float(args.get('threshold', 0.5))
            nms_iou = float(args.get('nms_iou', 0.5))
            image_base64 = self.runtime.get_image_base64(image_index)
            response = self.http.post(
                f'{self.server_url}/segment',
                json={
                    'image_base64': image_base64,
                    'prompt': prompt,
                    'threshold': threshold,
                    'nms_iou': nms_iou,
                },
                timeout=self.request_timeout,
            )
            response.raise_for_status()
            result = response.json()
            if result.get('status') != 'success':
                raise RuntimeError(result.get('message', f'MedSAM3 server failed: {result}'))
            image_width, image_height = self.runtime.get_image_size(image_index)
            top_bbox = result.get('top_bbox')
            mask_bbox = result.get('mask_bbox')
            visualization_kind = 'none'
            new_image_index = None
            if result.get('overlay_base64'):
                visualization_kind = 'overlay'
                new_image_index = self.runtime.add_image(image_base64=result['overlay_base64'], prefix='medsam3_overlay')
            elif result.get('mask_base64'):
                visualization_kind = 'mask'
                new_image_index = self.runtime.add_image(image_base64=result['mask_base64'], prefix='medsam3_mask')
            return json_result({
                'status': 'success',
                'tool': self.name,
                'image_index': image_index,
                'prompt': prompt,
                'threshold': threshold,
                'nms_iou': nms_iou,
                'num_detections': result.get('num_detections', 0),
                'top_score': result.get('top_score'),
                'top_bbox': xyxy_to_qwen_bbox(top_bbox, image_width, image_height) if top_bbox else None,
                'mask_bbox': xyxy_to_qwen_bbox(mask_bbox, image_width, image_height) if mask_bbox else None,
                'visualization_kind': visualization_kind,
                'new_image_index': new_image_index,
                'image_inventory': self.runtime.dump_image_inventory(),
            })
        except Exception as exc:
            return json_result({'status': 'error', 'tool': self.name, 'message': str(exc)})
