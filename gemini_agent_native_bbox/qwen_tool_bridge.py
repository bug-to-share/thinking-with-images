from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import requests

from sandbox_client import JupyterSandboxSession
from tool.common import ToolRuntimeContext


DEFAULT_SAM3_URL = "http://127.0.0.1:6020"


@dataclass
class ToolExecutionResult:
    name: str
    model_payload: Dict[str, Any]
    new_indices: List[int]


def _scale_1000_to_px(value: Any, length: int) -> int:
    numeric = float(value)
    numeric = max(0.0, min(1000.0, numeric))
    return int(round(numeric / 1000.0 * float(length)))


def _scale_px_to_1000(value: Any, length: int) -> int:
    if length <= 0:
        return 0
    numeric = float(value)
    numeric = max(0.0, min(float(length), numeric))
    return int(round(numeric / float(length) * 1000.0))


def qwen_bbox_to_xyxy(bbox: Any, image_size: Tuple[int, int]) -> List[int]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2] normalized to 0..1000")
    width, height = image_size
    x1 = _scale_1000_to_px(bbox[0], width)
    y1 = _scale_1000_to_px(bbox[1], height)
    x2 = _scale_1000_to_px(bbox[2], width)
    y2 = _scale_1000_to_px(bbox[3], height)
    if x2 < x1:
        x1, x2 = x2, x1
    if y2 < y1:
        y1, y2 = y2, y1
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid bbox after scaling: {[x1, y1, x2, y2]}")
    return [x1, y1, x2, y2]


def xyxy_to_qwen_bbox(bbox: Any, image_size: Tuple[int, int]) -> Optional[List[int]]:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    width, height = image_size
    x1, y1, x2, y2 = bbox
    return [
        _scale_px_to_1000(min(float(x1), float(x2)), width),
        _scale_px_to_1000(min(float(y1), float(y2)), height),
        _scale_px_to_1000(max(float(x1), float(x2)), width),
        _scale_px_to_1000(max(float(y1), float(y2)), height),
    ]


def qwen_point_to_xy(point: Any, image_size: Tuple[int, int]) -> List[int]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        raise ValueError("point must be [x, y] normalized to 0..1000")
    width, height = image_size
    return [_scale_1000_to_px(point[0], width), _scale_1000_to_px(point[1], height)]


def xy_to_qwen_point(point: Any, image_size: Tuple[int, int]) -> Optional[List[int]]:
    if not isinstance(point, (list, tuple)) or len(point) != 2:
        return None
    width, height = image_size
    return [_scale_px_to_1000(point[0], width), _scale_px_to_1000(point[1], height)]


def normalize_qwen_arguments_for_execution(arguments: Dict[str, Any], image_size: Tuple[int, int]) -> Dict[str, Any]:
    normalized = dict(arguments)
    if "bbox" in normalized:
        normalized["bbox"] = qwen_bbox_to_xyxy(normalized["bbox"], image_size)
    for key in ("positive_points", "negative_points"):
        if key in normalized and isinstance(normalized[key], (list, tuple)):
            normalized[key] = [qwen_point_to_xy(point, image_size) for point in normalized[key]]
    return normalized


class QwenToolBridge:
    def __init__(
        self,
        runtime: ToolRuntimeContext,
        sam3_url: str,
        sandbox_url: str,
        request_timeout: float,
    ):
        self.runtime = runtime
        self.sam3_url = sam3_url.rstrip("/")
        self.request_timeout = float(request_timeout)
        self.http = requests.Session()
        self.sandbox = JupyterSandboxSession(
            runtime=runtime,
            sample_id=runtime.sample_id,
            sandbox_url=sandbox_url,
            request_timeout=request_timeout,
        )
        self.sandbox.register_runtime_indices([record.index for record in runtime.image_records])

    def run(self, name: str, arguments: Dict[str, Any]) -> ToolExecutionResult:
        if name in {"python_exec", "code_sandbox", "sandbox"}:
            return self._run_python_exec(arguments)
        if name == "sam3":
            return self._run_sam3(arguments)
        raise ValueError(f"unsupported tool: {name}")

    def _post_json(self, url: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        response = self.http.post(url, json=payload, timeout=self.request_timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError(f"Expected JSON object from {url}, got {type(data).__name__}")
        if data.get("status") == "error":
            raise RuntimeError(data.get("message") or f"remote tool error from {url}")
        return data

    def _run_python_exec(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        code = str(arguments.get("code", ""))
        if not code.strip():
            raise ValueError("python_exec.code is required")
        timeout = float(arguments.get("timeout", 20.0))
        payload = self.sandbox.execute(code, timeout=timeout)
        new_indices = self.sandbox.register_code_output_images(payload.get("images", []), prefix="python_exec")
        compact = {
            "status": payload.get("status", "success"),
            "session_id": self.sandbox.session_id,
            "execution_time": payload.get("execution_time"),
            "timeout": timeout,
            "result": payload.get("result"),
            "stdout": payload.get("stdout", ""),
            "stderr": payload.get("stderr", ""),
            "num_output_images": len(new_indices),
            "new_image_indices": new_indices,
            "available_python_images": [f"image_{i}" for i in range(len(self.runtime.image_records))],
        }
        return ToolExecutionResult(name="python_exec", model_payload=compact, new_indices=new_indices)

    def _run_sam3(self, arguments: Dict[str, Any]) -> ToolExecutionResult:
        image_index = int(arguments.get("image_index", 0))
        image_size = self.runtime.get_image(image_index).size
        text = str(arguments.get("text", "")).strip()
        if not text:
            raise ValueError("sam3.text is required")
        bbox = arguments.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            raise ValueError("sam3 requires a normalized internal bbox in absolute xyxy form")
        positive_points = _normalize_points(arguments.get("positive_points"))
        negative_points = _normalize_points(arguments.get("negative_points"))
        if len(positive_points) != 3:
            raise ValueError("sam3 requires exactly 3 positive_points")
        if len(negative_points) != 3:
            raise ValueError("sam3 requires exactly 3 negative_points")

        points = positive_points + negative_points
        point_labels = [1] * len(positive_points) + [0] * len(negative_points)
        request_payload: Dict[str, Any] = {
            "image_base64": self.runtime.get_image_base64(image_index),
            "text": text,
            "bbox": [int(round(float(value))) for value in bbox],
            "points": points,
            "point_labels": point_labels,
        }
        payload = self._post_json(f"{self.sam3_url}/segment", request_payload)

        new_indices: List[int] = []
        if payload.get("overlay_base64"):
            new_indices.append(self.runtime.add_image(image_base64=payload["overlay_base64"], prefix="sam3_overlay"))
            self.sandbox.register_runtime_indices(new_indices)

        detections = []
        for item in payload.get("detections", []) or []:
            detections.append(
                {
                    "bbox": xyxy_to_qwen_bbox(item.get("bbox"), image_size),
                    "mask_bbox": xyxy_to_qwen_bbox(item.get("mask_bbox"), image_size),
                    "foreground_pixels": item.get("foreground_pixels"),
                    "mask_mean_prob": item.get("mask_mean_prob"),
                }
            )

        compact = {
            "status": payload.get("status", "success"),
            "image_index": image_index,
            "text": text,
            "mode": payload.get("mode"),
            "input_bbox": xyxy_to_qwen_bbox(payload.get("input_bbox") or request_payload.get("bbox"), image_size),
            "input_positive_points": [xy_to_qwen_point(point, image_size) for point in positive_points],
            "input_negative_points": [xy_to_qwen_point(point, image_size) for point in negative_points],
            "image_size": payload.get("image_size"),
            "total_detections": payload.get("total_detections"),
            "num_detections": payload.get("num_detections"),
            "top_score": payload.get("top_score"),
            "top_bbox": xyxy_to_qwen_bbox(payload.get("top_bbox"), image_size),
            "top_mask_bbox": xyxy_to_qwen_bbox(payload.get("top_mask_bbox"), image_size),
            "detections": detections,
            "visualization_kind": "overlay" if new_indices else "none",
            "new_image_index": new_indices[0] if new_indices else None,
            "message": payload.get("message"),
        }
        return ToolExecutionResult(name="sam3", model_payload=compact, new_indices=new_indices)


def _normalize_points(value: Any) -> List[List[int]]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError("points must be a list of [x, y] pairs")
    rows: List[List[int]] = []
    for point in value:
        if not isinstance(point, (list, tuple)) or len(point) != 2:
            raise ValueError("each point must be a 2-number absolute pixel [x, y] pair")
        rows.append([int(round(float(point[0]))), int(round(float(point[1])))])
    return rows
