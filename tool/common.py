from __future__ import annotations

import base64
import json
import threading
import uuid
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


ARTIFACT_ROOT = Path('./tool/artifacts')


def normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, list):
        return ' '.join(normalize_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


@dataclass
class ImageRecord:
    index: int
    path: str
    image_base64: str


def pil_to_base64(image: Image.Image, image_format: str = 'PNG') -> str:
    buffer = BytesIO()
    image.convert('RGB').save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def base64_to_image(image_base64: str) -> Image.Image:
    payload = image_base64.split(',', 1)[1] if ',' in image_base64 else image_base64
    return Image.open(BytesIO(base64.b64decode(payload))).convert('RGB')


def image_path_to_base64(image_path: str, image_format: str = 'PNG') -> str:
    with Image.open(image_path).convert('RGB') as image:
        return pil_to_base64(image, image_format=image_format)


def save_base64_image(image_base64: str, output_path: Path) -> str:
    image = base64_to_image(image_base64)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return str(output_path.resolve())


class ToolRuntimeContext:
    def __init__(self, artifact_root: Path | str = ARTIFACT_ROOT):
        self.artifact_root = Path(artifact_root)
        self.artifact_root.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.sample_id = 'default'
        self.sample_dir = self.artifact_root / self.sample_id
        self.image_records: List[ImageRecord] = []

    def reset(self, image_path: str, sample_id: Optional[str] = None) -> None:
        with self._lock:
            self.sample_id = sample_id or f'sample_{uuid.uuid4().hex[:8]}'
            self.sample_dir = self.artifact_root / self.sample_id
            self.sample_dir.mkdir(parents=True, exist_ok=True)
            abs_path = str(Path(image_path).expanduser().resolve())
            image_base64 = image_path_to_base64(abs_path)
            self.image_records = [ImageRecord(index=0, path=abs_path, image_base64=image_base64)]

    def add_image(
        self,
        image_path: Optional[str] = None,
        image_base64: Optional[str] = None,
        prefix: str = 'image',
        suffix: str = '.png',
    ) -> int:
        with self._lock:
            if image_base64 is None and image_path is None:
                raise ValueError('either image_path or image_base64 must be provided')
            if image_base64 is None:
                abs_path = str(Path(str(image_path)).expanduser().resolve())
                image_base64 = image_path_to_base64(abs_path)
            else:
                output_path = self.make_output_path(prefix, suffix)
                abs_path = save_base64_image(image_base64, output_path)
            index = len(self.image_records)
            self.image_records.append(ImageRecord(index=index, path=abs_path, image_base64=image_base64))
            return index

    def get_image_path(self, image_index: int) -> str:
        if image_index < 0 or image_index >= len(self.image_records):
            raise IndexError(f'image_index out of range: {image_index}')
        return self.image_records[image_index].path

    def get_image_base64(self, image_index: int) -> str:
        if image_index < 0 or image_index >= len(self.image_records):
            raise IndexError(f'image_index out of range: {image_index}')
        return self.image_records[image_index].image_base64

    def get_image(self, image_index: int) -> Image.Image:
        return base64_to_image(self.get_image_base64(image_index))

    def get_image_size(self, image_index: int) -> Tuple[int, int]:
        with self.get_image(image_index) as img:
            return img.width, img.height

    def make_output_path(self, prefix: str, suffix: str = '.png') -> Path:
        unique = uuid.uuid4().hex[:10]
        return self.sample_dir / f'{prefix}_{unique}{suffix}'

    def dump_image_inventory(self) -> List[Dict[str, Any]]:
        return [{'image_index': record.index} for record in self.image_records]


def bbox_to_xyxy(bbox_2d: Sequence[float], image_width: int, image_height: int, bbox_mode: str = 'xyxy') -> List[int]:
    if len(bbox_2d) != 4:
        raise ValueError('bbox_2d must contain exactly four numbers')
    mode = (bbox_mode or 'xyxy').lower()
    values = [float(x) for x in bbox_2d]

    if mode in {'xyxy', 'qwen1000', 'normalized_1000', 'qwen'}:
        x1, y1, x2, y2 = values
        if mode in {'qwen1000', 'normalized_1000', 'qwen'}:
            x1 = x1 / 1000.0 * image_width
            y1 = y1 / 1000.0 * image_height
            x2 = x2 / 1000.0 * image_width
            y2 = y2 / 1000.0 * image_height
    elif mode == 'xywh':
        x1, y1, width, height = values
        x2 = x1 + width
        y2 = y1 + height
    elif mode == 'yxyx':
        y1, x1, y2, x2 = values
    else:
        raise ValueError(f'unsupported bbox_mode: {bbox_mode}')

    x1 = int(round(max(0, min(image_width, x1))))
    y1 = int(round(max(0, min(image_height, y1))))
    x2 = int(round(max(0, min(image_width, x2))))
    y2 = int(round(max(0, min(image_height, y2))))

    if x2 <= x1 or y2 <= y1:
        raise ValueError(f'invalid bbox after normalization: {[x1, y1, x2, y2]}')
    return [x1, y1, x2, y2]


def xyxy_to_qwen_bbox(bbox_xyxy: Sequence[float], image_width: int, image_height: int) -> List[int]:
    if len(bbox_xyxy) != 4:
        raise ValueError('bbox_xyxy must contain exactly four numbers')
    x1, y1, x2, y2 = [float(x) for x in bbox_xyxy]
    if image_width <= 0 or image_height <= 0:
        raise ValueError('image_width and image_height must be positive')
    normalized = [
        int(round(max(0.0, min(1000.0, x1 / image_width * 1000.0)))),
        int(round(max(0.0, min(1000.0, y1 / image_height * 1000.0)))),
        int(round(max(0.0, min(1000.0, x2 / image_width * 1000.0)))),
        int(round(max(0.0, min(1000.0, y2 / image_height * 1000.0)))),
    ]
    return normalized




def mask_to_bbox(mask: np.ndarray) -> Optional[List[int]]:
    if mask is None:
        return None
    binary = np.asarray(mask) > 0
    if not binary.any():
        return None
    ys, xs = np.where(binary)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def save_overlay(image_rgb: np.ndarray, binary_mask: np.ndarray, output_path: Path, color: Tuple[int, int, int] = (255, 0, 0), alpha: float = 0.45) -> str:
    image = np.asarray(image_rgb).astype(np.uint8)
    mask = np.asarray(binary_mask).astype(bool)
    overlay = image.copy().astype(np.float32)
    color_arr = np.array(color, dtype=np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + color_arr * alpha
    overlay = overlay.clip(0, 255).astype(np.uint8)
    Image.fromarray(overlay).save(output_path)
    return str(output_path.resolve())


def save_binary_mask(binary_mask: np.ndarray, output_path: Path) -> str:
    mask = (np.asarray(binary_mask).astype(np.uint8) * 255)
    Image.fromarray(mask).save(output_path)
    return str(output_path.resolve())


def json_result(payload: Dict[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False)
