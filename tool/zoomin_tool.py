from __future__ import annotations

from typing import Optional, Union

from qwen_agent.tools.base import BaseTool, register_tool

from .common import ToolRuntimeContext, bbox_to_xyxy, json_result, pil_to_base64, xyxy_to_qwen_bbox


@register_tool('image_zoom_in_tool', allow_overwrite=True)
class ZoomInTool(BaseTool):
    description = 'Crop and enlarge a region of interest from one image already in the tool context.'
    parameters = {
        'type': 'object',
        'properties': {
            'image_index': {
                'type': 'integer',
                'description': '0-based image index in the current context. The original image is index 0.'
            },
            'bbox': {
                'type': 'array',
                'description': 'Qwen-style bbox in [x1, y1, x2, y2] order on a 0..1000 relative coordinate grid.',
                'items': {'type': 'number'},
                'minItems': 4,
                'maxItems': 4,
            }
        },
        'required': ['image_index', 'bbox']
    }

    def __init__(self, cfg: Optional[dict] = None, runtime: Optional[ToolRuntimeContext] = None):
        super().__init__(cfg)
        self.runtime = runtime or ToolRuntimeContext()

    def call(self, params: Union[str, dict], **kwargs) -> str:
        try:
            args = self._verify_json_format_args(params)
            image_index = int(args['image_index'])
            bbox = args.get('bbox', args.get('bbox_2d'))
            if bbox is None:
                raise ValueError('bbox is required')
            bbox_mode = args.get('bbox_mode', 'qwen1000')
            with self.runtime.get_image(image_index) as img:
                x1, y1, x2, y2 = bbox_to_xyxy(bbox, img.width, img.height, bbox_mode=bbox_mode)
                crop = img.crop((x1, y1, x2, y2))
                crop_base64 = pil_to_base64(crop)
                normalized_bbox = xyxy_to_qwen_bbox([x1, y1, x2, y2], img.width, img.height)
            new_index = self.runtime.add_image(image_base64=crop_base64, prefix='zoom')
            return json_result({
                'status': 'success',
                'tool': self.name,
                'source_image_index': image_index,
                'bbox': normalized_bbox,
                'crop_size': list(crop.size),
                'new_image_index': new_index,
                'image_inventory': self.runtime.dump_image_inventory(),
            })
        except Exception as exc:
            return json_result({'status': 'error', 'tool': self.name, 'message': str(exc)})
