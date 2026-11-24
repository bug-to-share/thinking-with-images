from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

from PIL import Image


CANVAS_SIZE = 1000


@dataclass
class ModelCanvas:
    image_path: str
    canvas_size: int
    original_size: List[int]
    resized_size: List[int]
    scale: float
    pad_left: int
    pad_top: int

    def as_dict(self) -> Dict[str, Any]:
        return {
            'canvas_size': self.canvas_size,
            'original_size': self.original_size,
            'resized_size': self.resized_size,
            'scale': self.scale,
            'pad_left': self.pad_left,
            'pad_top': self.pad_top,
        }


def prepare_model_canvas(image_path: str, output_path: str | Path, canvas_size: int = CANVAS_SIZE) -> ModelCanvas:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(image_path).convert('RGB') as image:
        width, height = image.size
        scale = min(float(canvas_size) / float(width), float(canvas_size) / float(height))
        resized_width = max(1, int(round(width * scale)))
        resized_height = max(1, int(round(height * scale)))
        resized = image.resize((resized_width, resized_height), Image.Resampling.BILINEAR)
        canvas = Image.new('RGB', (canvas_size, canvas_size), (0, 0, 0))
        pad_left = (canvas_size - resized_width) // 2
        pad_top = (canvas_size - resized_height) // 2
        canvas.paste(resized, (pad_left, pad_top))
        canvas.save(output_path)
    return ModelCanvas(
        image_path=str(output_path.resolve()),
        canvas_size=canvas_size,
        original_size=[width, height],
        resized_size=[resized_width, resized_height],
        scale=scale,
        pad_left=pad_left,
        pad_top=pad_top,
    )


def map_original_xyxy_to_canvas_xyxy(bbox: List[int], canvas_info: Dict[str, Any]) -> List[int]:
    x1, y1, x2, y2 = [float(value) for value in bbox]
    scale = float(canvas_info['scale'])
    pad_left = int(canvas_info['pad_left'])
    pad_top = int(canvas_info['pad_top'])
    return [
        int(round(x1 * scale)) + pad_left,
        int(round(y1 * scale)) + pad_top,
        int(round(x2 * scale)) + pad_left,
        int(round(y2 * scale)) + pad_top,
    ]
