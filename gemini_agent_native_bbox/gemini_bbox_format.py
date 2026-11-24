from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image


NUMBER_PATTERN = r'-?\d+(?:\.\d+)?'
BOX_TEXT_PATTERN = re.compile(rf'\[\s*{NUMBER_PATTERN}\s*,\s*{NUMBER_PATTERN}\s*,\s*{NUMBER_PATTERN}\s*,\s*{NUMBER_PATTERN}\s*\]')
POINT_TEXT_PATTERN = re.compile(rf'\[\s*{NUMBER_PATTERN}\s*,\s*{NUMBER_PATTERN}\s*\]')


def load_image_size(image_path: str | Path) -> Tuple[int, int]:
    with Image.open(image_path) as image:
        return image.size


def extract_coordinate_texts(text: str) -> Dict[str, List[str]]:
    return {
        'boxes_4d': BOX_TEXT_PATTERN.findall(text or ''),
        'points_2d': POINT_TEXT_PATTERN.findall(text or ''),
    }


def normalize_tool_arguments_for_execution(arguments: Dict[str, Any], image_size: Tuple[int, int]) -> Dict[str, Any]:
    normalized = dict(arguments)
    bbox = arguments.get('bbox')
    if isinstance(bbox, (list, tuple)) and len(bbox) == 4:
        normalized['bbox'] = gemini_box_to_xyxy(bbox, image_size)
    positive_points = arguments.get('positive_points')
    if isinstance(positive_points, (list, tuple)):
        normalized['positive_points'] = [gemini_point_to_xy(point, image_size) for point in positive_points]
    negative_points = arguments.get('negative_points')
    if isinstance(negative_points, (list, tuple)):
        normalized['negative_points'] = [gemini_point_to_xy(point, image_size) for point in negative_points]
    point = arguments.get('point')
    if isinstance(point, (list, tuple)) and len(point) == 2:
        normalized['point'] = gemini_point_to_xy(point, image_size)
    return normalized


def to_model_bbox(bbox: Any, image_size: Tuple[int, int]) -> Optional[List[int]]:
    xyxy = _coerce_quad(bbox)
    if xyxy is None:
        return None
    return xyxy_to_gemini_box(xyxy, image_size)


def to_model_points(points: Any, image_size: Tuple[int, int]) -> List[List[int]]:
    if not isinstance(points, (list, tuple)):
        return []
    rows: List[List[int]] = []
    for point in points:
        xy = _coerce_pair(point)
        if xy is None:
            continue
        rows.append(xy_to_gemini_point(xy, image_size))
    return rows


def gemini_box_to_xyxy(box: Any, image_size: Tuple[int, int]) -> List[int]:
    quad = _coerce_quad(box)
    if quad is None:
        raise ValueError('bbox must be a 4-number Gemini box [ymin, xmin, ymax, xmax]')
    width, height = image_size
    ymin, xmin, ymax, xmax = quad
    x1 = _scale_0_1000_to_px(xmin, width)
    y1 = _scale_0_1000_to_px(ymin, height)
    x2 = _scale_0_1000_to_px(xmax, width)
    y2 = _scale_0_1000_to_px(ymax, height)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    return [x1, y1, x2, y2]


def xyxy_to_gemini_box(box: Any, image_size: Tuple[int, int]) -> List[int]:
    quad = _coerce_quad(box)
    if quad is None:
        raise ValueError('box must be a 4-number xyxy box')
    width, height = image_size
    x1, y1, x2, y2 = quad
    ymin = _scale_px_to_0_1000(min(y1, y2), height)
    xmin = _scale_px_to_0_1000(min(x1, x2), width)
    ymax = _scale_px_to_0_1000(max(y1, y2), height)
    xmax = _scale_px_to_0_1000(max(x1, x2), width)
    return [ymin, xmin, ymax, xmax]


def gemini_point_to_xy(point: Any, image_size: Tuple[int, int]) -> List[int]:
    pair = _coerce_pair(point)
    if pair is None:
        raise ValueError('point must be a 2-number Gemini point [y, x]')
    width, height = image_size
    y, x = pair
    return [_scale_0_1000_to_px(x, width), _scale_0_1000_to_px(y, height)]


def xy_to_gemini_point(point: Any, image_size: Tuple[int, int]) -> List[int]:
    pair = _coerce_pair(point)
    if pair is None:
        raise ValueError('point must be a 2-number xy point [x, y]')
    width, height = image_size
    x, y = pair
    return [_scale_px_to_0_1000(y, height), _scale_px_to_0_1000(x, width)]


def _scale_0_1000_to_px(value: float, length: int) -> int:
    if length <= 1:
        return 0
    return int(round(_clamp(value, 0.0, 1000.0) / 1000.0 * length))


def _scale_px_to_0_1000(value: float, length: int) -> int:
    if length <= 0:
        return 0
    return int(round(_clamp(value, 0.0, float(length)) / float(length) * 1000.0))


def _coerce_quad(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except Exception:
        return None


def _coerce_pair(value: Any) -> Optional[List[float]]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    try:
        return [float(item) for item in value]
    except Exception:
        return None


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))
