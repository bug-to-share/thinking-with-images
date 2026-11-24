"""Runtime adapters for medical tools with a Qwen 0--1000 coordinate contract."""
from __future__ import annotations

import asyncio
import base64
import csv
import json
import math
import os
import re
import threading
from collections import Counter
from io import BytesIO
from pathlib import Path
from typing import Any, Optional
from uuid import uuid4

import requests
from PIL import Image
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse


RELATIVE_COORDINATE_MAX = 1000
SPATIAL_RESULT_PREFIX = "__MED_TWI_SPATIAL_RESULT__"


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2)


def _to_b64(image: Image.Image) -> str:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _from_b64(value: str) -> Image.Image:
    return Image.open(BytesIO(base64.b64decode(value.split(",", 1)[-1]))).convert("RGB")


def _urls(value: Any, default: str) -> list[str]:
    values = value.split(",") if isinstance(value, str) else (list(value) if isinstance(value, (list, tuple)) else [default])
    return [str(item).strip().rstrip("/") for item in values if str(item).strip()] or [default]


def _number(value: Any, name: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must contain numeric coordinates") from exc


def _clamp_relative(value: Any) -> float:
    return max(0.0, min(float(RELATIVE_COORDINATE_MAX), _number(value, "coordinate")))


def _relative_to_pixel(value: Any, length: int, *, point: bool) -> int:
    if length <= 0:
        raise ValueError("image dimensions must be positive")
    pixel = int(round(_clamp_relative(value) * length / RELATIVE_COORDINATE_MAX))
    return min(pixel, length - 1) if point else min(pixel, length)


def _pixel_to_relative(value: Any, length: int) -> int:
    if length <= 0:
        raise ValueError("image dimensions must be positive")
    pixel = max(0.0, min(float(length), _number(value, "pixel coordinate")))
    return int(round(pixel * RELATIVE_COORDINATE_MAX / length))


def _relative_box_to_pixels(value: Any, size: tuple[int, int]) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        raise ValueError("bbox must be [x1, y1, x2, y2] in 0..1000 relative coordinates")
    width, height = size
    x1, x2 = sorted((_relative_to_pixel(value[0], width, point=False), _relative_to_pixel(value[2], width, point=False)))
    y1, y2 = sorted((_relative_to_pixel(value[1], height, point=False), _relative_to_pixel(value[3], height, point=False)))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("bbox must have positive area")
    return [x1, y1, x2, y2]


def _relative_point_to_pixels(value: Any, size: tuple[int, int]) -> list[int]:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        raise ValueError("point must be [x, y] in 0..1000 relative coordinates")
    return [_relative_to_pixel(value[0], size[0], point=True), _relative_to_pixel(value[1], size[1], point=True)]


def _pixel_box_to_relative(value: Any, size: tuple[int, int]) -> list[int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        return [
            _pixel_to_relative(value[0], size[0]),
            _pixel_to_relative(value[1], size[1]),
            _pixel_to_relative(value[2], size[0]),
            _pixel_to_relative(value[3], size[1]),
        ]
    except ValueError:
        return None


def _normalise_geometry(value: Any, size: tuple[int, int]) -> Any:
    """Convert only known xyxy geometry fields in a tool response to Qwen coordinates."""
    if isinstance(value, list):
        return [_normalise_geometry(item, size) for item in value]
    if not isinstance(value, dict):
        return value

    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"bbox", "mask_bbox", "top_bbox", "top_mask_bbox", "input_bbox"}:
            normalised = _pixel_box_to_relative(item, size)
            if normalised is not None:
                result[key] = normalised
                continue
        if key == "image_size":
            continue
        result[key] = _normalise_geometry(item, size)
    return result


def _extract_spatial_results(stdout: str) -> tuple[str, list[dict[str, Any]]]:
    clean_lines: list[str] = []
    spatial_results: list[dict[str, Any]] = []
    for line in stdout.splitlines():
        if not line.startswith(SPATIAL_RESULT_PREFIX):
            clean_lines.append(line)
            continue
        try:
            result = json.loads(line[len(SPATIAL_RESULT_PREFIX) :])
        except json.JSONDecodeError:
            clean_lines.append("Sandbox emitted an invalid structured spatial result.")
            continue
        if isinstance(result, dict):
            spatial_results.append(result)
    clean_stdout = "\n".join(clean_lines)
    if stdout.endswith("\n") and clean_stdout:
        clean_stdout += "\n"
    return clean_stdout, spatial_results


class _HTTPTool(BaseTool):
    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema):
        super().__init__(config, tool_schema)
        self.config = config
        self.timeout = float(config.get("timeout", 120))
        self.http = requests.Session()

    async def create(self, instance_id: Optional[str] = None, **kwargs):
        return instance_id or str(uuid4()), ToolResponse()

    def _post(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        response = self.http.post(url, json=payload, timeout=self.timeout)
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict):
            raise RuntimeError("tool response must be a JSON object")
        if data.get("status") == "error":
            raise RuntimeError(data.get("message", "remote tool error"))
        return data

    @staticmethod
    def _response_images(payload: dict[str, Any]) -> list[Image.Image]:
        for key in ("overlay_base64", "mask_base64"):
            if payload.get(key):
                return [_from_b64(str(payload[key]))]
        return []


class MedicalMedSAM3Tool(_HTTPTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.urls = _urls(config.get("base_urls", config.get("base_url")), "http://127.0.0.1:6002")
        self.index = 0
        self.lock = threading.Lock()

    def _next(self) -> str:
        with self.lock:
            url = self.urls[self.index % len(self.urls)]
            self.index += 1
            return url

    async def execute(self, instance_id, parameters, **kwargs):
        images = kwargs["agent_data"].image_data or []
        image_index = int(parameters.get("image_index", 0))
        prompt = str(parameters.get("prompt", "")).strip()
        if not prompt:
            return ToolResponse(text="Error: `prompt` is required."), 0.0, {}
        if image_index < 0 or image_index >= len(images):
            return ToolResponse(text=f"Error: image_index {image_index} out of range."), 0.0, {}

        request = {"image_base64": _to_b64(images[image_index]), "prompt": prompt}
        request.update({key: float(parameters[key]) for key in ("threshold", "nms_iou") if key in parameters})
        if "max_detections" in parameters:
            try:
                max_detections = int(parameters["max_detections"])
            except (TypeError, ValueError) as exc:
                raise ValueError("max_detections must be a positive integer") from exc
            if max_detections < 1:
                raise ValueError("max_detections must be a positive integer")
            # The remote service owns filtering/rendering; adapter only forwards this control.
            request["max_detections"] = max_detections
        url = self._next()
        data = await asyncio.to_thread(self._post, url + "/segment", request)
        response_images = self._response_images(data)
        evidence = _normalise_geometry({
            "status": data.get("status", "success"),
            "image_index": image_index,
            "prompt": prompt,
            "num_detections": data.get("num_detections"),
            "top_score": data.get("top_score"),
            "top_bbox": data.get("top_bbox"),
            "mask_bbox": data.get("mask_bbox"),
            "detections": data.get("detections", []),
            "visualization_returned": bool(response_images),
        }, images[image_index].size)
        evidence["coordinate_space"] = "relative_0_1000"
        return ToolResponse(text=_json(evidence), image=response_images), 0.0, {}


class MedicalSAM3Tool(_HTTPTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.urls = _urls(config.get("base_urls", config.get("base_url")), "http://127.0.0.1:6020")
        self.index = 0
        self.lock = threading.Lock()

    def _next(self) -> str:
        with self.lock:
            url = self.urls[self.index % len(self.urls)]
            self.index += 1
            return url

    async def execute(self, instance_id, parameters, **kwargs):
        images = kwargs["agent_data"].image_data or []
        image_index = int(parameters.get("image_index", 0))
        text_prompt = str(parameters.get("text", "")).strip()
        if not text_prompt:
            return ToolResponse(text="Error: `text` is required."), 0.0, {}
        if image_index < 0 or image_index >= len(images):
            return ToolResponse(text=f"Error: image_index {image_index} out of range."), 0.0, {}

        image = images[image_index]
        request: dict[str, Any] = {"image_base64": _to_b64(image), "text": text_prompt}
        if parameters.get("bbox") is not None:
            request["bbox"] = _relative_box_to_pixels(parameters["bbox"], image.size)
        positive_points = [_relative_point_to_pixels(point, image.size) for point in parameters.get("positive_points", [])]
        negative_points = [_relative_point_to_pixels(point, image.size) for point in parameters.get("negative_points", [])]
        request["points"] = positive_points + negative_points
        request["point_labels"] = [1] * len(positive_points) + [0] * len(negative_points)

        url = self._next()
        data = await asyncio.to_thread(self._post, url + "/segment", request)
        response_images = self._response_images(data)
        evidence = _normalise_geometry({
            "status": data.get("status", "success"),
            "image_index": image_index,
            "text": text_prompt,
            "mode": data.get("mode"),
            "num_detections": data.get("num_detections"),
            "top_score": data.get("top_score"),
            "top_bbox": data.get("top_bbox"),
            "top_mask_bbox": data.get("top_mask_bbox", data.get("mask_bbox")),
            "detections": data.get("detections", []),
            "visualization_returned": bool(response_images),
        }, image.size)
        evidence["coordinate_space"] = "relative_0_1000"
        return ToolResponse(text=_json(evidence), image=response_images), 0.0, {}


def _prelude(images: list[Image.Image]) -> str:
    """Inject images plus explicit coordinate bridges into sandbox source code."""
    lines = [
        "from PIL import Image, ImageDraw",
        "from io import BytesIO",
        "import base64",
        "import json",
        "import numpy as np",
        f"RELATIVE_COORDINATE_MAX = {RELATIVE_COORDINATE_MAX}",
        "_SPATIAL_RESULT_PREFIX = " + repr(SPATIAL_RESULT_PREFIX),
        "def _rel_to_px(value, length, point=False):",
        "    value = max(0.0, min(float(RELATIVE_COORDINATE_MAX), float(value)))",
        "    pixel = int(round(value * length / RELATIVE_COORDINATE_MAX))",
        "    return min(pixel, length - 1) if point else min(pixel, length)",
        "def rel_point_to_px(image, point_rel):",
        "    if len(point_rel) != 2: raise ValueError('point_rel must be [x, y]')",
        "    return [_rel_to_px(point_rel[0], image.width, True), _rel_to_px(point_rel[1], image.height, True)]",
        "def rel_box_to_px(image, box_rel):",
        "    if len(box_rel) != 4: raise ValueError('box_rel must be [x1, y1, x2, y2]')",
        "    x1, x2 = sorted((_rel_to_px(box_rel[0], image.width), _rel_to_px(box_rel[2], image.width)))",
        "    y1, y2 = sorted((_rel_to_px(box_rel[1], image.height), _rel_to_px(box_rel[3], image.height)))",
        "    if x2 <= x1 or y2 <= y1: raise ValueError('box_rel must have positive area')",
        "    return [x1, y1, x2, y2]",
        "def px_point_to_rel(image, point_px):",
        "    if len(point_px) != 2: raise ValueError('point_px must be [x, y]')",
        "    return [int(round(max(0, min(image.width - 1, float(point_px[0]))) * RELATIVE_COORDINATE_MAX / image.width)), int(round(max(0, min(image.height - 1, float(point_px[1]))) * RELATIVE_COORDINATE_MAX / image.height))]",
        "def px_box_to_rel(image, box_px):",
        "    if len(box_px) != 4: raise ValueError('box_px must be [x1, y1, x2, y2]')",
        "    return [int(round(max(0, min(image.width, float(box_px[0]))) * RELATIVE_COORDINATE_MAX / image.width)), int(round(max(0, min(image.height, float(box_px[1]))) * RELATIVE_COORDINATE_MAX / image.height)), int(round(max(0, min(image.width, float(box_px[2]))) * RELATIVE_COORDINATE_MAX / image.width)), int(round(max(0, min(image.height, float(box_px[3]))) * RELATIVE_COORDINATE_MAX / image.height))]",
        "def crop_rel(image, box_rel):",
        "    return image.crop(tuple(rel_box_to_px(image, box_rel)))",
        "def draw_box_rel(image, box_rel, color='red', width=3):",
        "    annotated = image.copy()",
        "    ImageDraw.Draw(annotated).rectangle(rel_box_to_px(annotated, box_rel), outline=color, width=width)",
        "    return annotated",
        "def report_spatial_result(image, *, label='', bbox_px=None, point_px=None, **extra):",
        "    result = {'coordinate_space': 'relative_0_1000'}",
        "    if label: result['label'] = str(label)",
        "    if bbox_px is not None: result['bbox'] = px_box_to_rel(image, bbox_px)",
        "    if point_px is not None: result['point'] = px_point_to_rel(image, point_px)",
        "    result.update(extra)",
        "    print(_SPATIAL_RESULT_PREFIX + json.dumps(result, ensure_ascii=False))",
        "    return result",
    ]
    for index, image in enumerate(images):
        lines.extend([
            f"_image_{index}_b64 = {_to_b64(image)!r}",
            f"image_{index} = Image.open(BytesIO(base64.b64decode(_image_{index}_b64))).convert('RGB')",
        ])
    return "\n".join(lines) + "\n"


class MedicalPythonExecTool(_HTTPTool):
    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.run_url = _urls(config.get("sandbox_urls", config.get("sandbox_url")), "http://127.0.0.1:18901/run_jupyter")[0]
        self.sessions: dict[str, str] = {}
        self.session_lock = threading.Lock()

    async def finalize_request(self, request_id):
        with self.session_lock:
            session_id = self.sessions.pop(str(request_id), None)
        if session_id:
            try:
                await asyncio.to_thread(self.http.post, self.run_url.rsplit("/", 1)[0] + "/clear_session", json={"session_id": session_id}, timeout=self.timeout)
            except Exception:
                pass

    async def execute(self, instance_id, parameters, **kwargs):
        agent_data = kwargs["agent_data"]
        code = str(parameters.get("code", ""))
        if not code.strip():
            return ToolResponse(text="Error: `code` is required."), 0.0, {}

        request_id = str(agent_data.request_id)
        with self.session_lock:
            session_id = self.sessions.setdefault(request_id, f"agentic_{request_id}_{uuid4().hex[:8]}")
        if parameters.get("reset_session"):
            await asyncio.to_thread(self.http.post, self.run_url.rsplit("/", 1)[0] + "/clear_session", json={"session_id": session_id}, timeout=self.timeout)

        images = agent_data.image_data or []
        timeout = float(parameters.get("timeout", self.config.get("default_timeout", 30)))
        data = await asyncio.to_thread(self._post, self.run_url, {
            "session_id": session_id,
            "code": _prelude(images) + code,
            "timeout": timeout,
        })
        output = data.get("output", {}) or {}
        response_images = [_from_b64(item) for item in output.get("images", []) or []]
        stdout, spatial_results = _extract_spatial_results(str(output.get("stdout", "")))
        evidence = {
            "status": data.get("status", "success"),
            "stdout": stdout,
            "stderr": str(output.get("stderr", "")),
            "spatial_results": spatial_results,
            "visualization_returned": bool(response_images),
            "available_python_images": [f"image_{index}" for index in range(len(images))],
            "coordinate_space": "relative_0_1000",
        }
        return ToolResponse(text=_json(evidence), image=response_images), 0.0, {}


class MedicalZoomInTool(BaseTool):
    """Crop and enlarge a model-selected region using Qwen's relative bbox contract."""

    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.min_zoom = float(config.get("min_zoom_factor", 1.0))
        self.max_zoom = float(config.get("max_zoom_factor", 4.0))
        self.max_output_side = int(config.get("max_output_side", 2048))

    async def execute(self, instance_id, parameters, **kwargs):
        del instance_id
        images = kwargs["agent_data"].image_data or []
        try:
            image_index = int(parameters.get("image_index", 0))
            if image_index < 0 or image_index >= len(images):
                raise ValueError(f"image_index {image_index} out of range (number of images: {len(images)})")
            image = images[image_index].convert("RGB")
            bbox_rel = parameters.get("bbox")
            bbox_px = _relative_box_to_pixels(bbox_rel, image.size)
            zoom_factor = max(self.min_zoom, min(self.max_zoom, float(parameters.get("zoom_factor", 2.0))))
            crop = image.crop(tuple(bbox_px))
            width = max(1, int(round(crop.width * zoom_factor)))
            height = max(1, int(round(crop.height * zoom_factor)))
            scale = min(1.0, self.max_output_side / max(width, height))
            output_size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            zoomed = crop.resize(output_size, resampling) if output_size != crop.size else crop
            evidence = {
                "status": "success",
                "image_index": image_index,
                "bbox": [float(value) for value in bbox_rel],
                "zoom_factor": zoom_factor,
                "coordinate_space": "relative_0_1000",
                "message": "A cropped and enlarged image is attached. Use it as visual evidence.",
            }
            return ToolResponse(text=_json(evidence), image=[zoomed]), 0.0, {"success": True}
        except (TypeError, ValueError, IndexError) as exc:
            return ToolResponse(text=f"Error: {exc}"), 0.0, {"success": False}


_KG_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")
_KG_STOPWORDS = {
    "a", "an", "and", "are", "be", "can", "does", "for", "how", "in", "is", "of", "on", "the", "to", "what", "where", "which", "with",
}
_KG_INDEX_CACHE: dict[str, "_SlakeKnowledgeIndex"] = {}
_KG_INDEX_LOCK = threading.Lock()


def _kg_tokens(text: str) -> list[str]:
    return [token for token in _KG_TOKEN_PATTERN.findall(text.casefold()) if token not in _KG_STOPWORDS]


class _SlakeKnowledgeIndex:
    """Small dependency-free lexical RAG index over SLAKE's English KG CSV files."""

    def __init__(self, knowledge_dir: str):
        root = Path(knowledge_dir)
        if not root.is_dir():
            raise FileNotFoundError(f"SLAKE knowledge directory not found: {root}")
        self.documents: list[dict[str, Any]] = []
        for filename in ("en_disease.csv", "en_organ.csv", "en_organ_rel.csv"):
            path = root / filename
            if not path.is_file():
                raise FileNotFoundError(f"Required SLAKE English KG file not found: {path}")
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.reader(handle, delimiter="#")
                header = next(reader, None)
                if not header or len(header) != 3 or header[1:] != ["attribute", "value"]:
                    raise ValueError(f"Unexpected header in {path}: {header!r}")
                for row in reader:
                    if len(row) != 3 or not all(item.strip() for item in row):
                        continue
                    subject, relation, value = (item.strip() for item in row)
                    text = f"{subject} {relation} {value}"
                    self.documents.append({"subject": subject, "relation": relation, "value": value, "source": filename, "text": text})
        self.term_counts = [Counter(_kg_tokens(doc["text"])) for doc in self.documents]
        self.doc_freq = Counter(token for counts in self.term_counts for token in counts)
        self.avg_len = sum(sum(counts.values()) for counts in self.term_counts) / max(1, len(self.documents))

    def search(self, query: str, top_k: int) -> list[dict[str, Any]]:
        query_tokens = _kg_tokens(query[:512])
        if not query_tokens:
            return []
        total_docs = len(self.documents)
        scored: list[tuple[float, int]] = []
        for index, (doc, counts) in enumerate(zip(self.documents, self.term_counts)):
            length = max(1, sum(counts.values()))
            score = 0.0
            for token in query_tokens:
                tf = counts.get(token, 0)
                if not tf:
                    continue
                idf = math.log(1.0 + (total_docs - self.doc_freq[token] + 0.5) / (self.doc_freq[token] + 0.5))
                score += idf * (tf * 2.2) / (tf + 1.2 * (0.25 + 0.75 * length / max(self.avg_len, 1.0)))
            if score > 0:
                phrase = " ".join(query_tokens)
                if phrase and phrase in doc["text"].casefold():
                    score += 0.5
                scored.append((score, index))
        scored.sort(key=lambda item: (-item[0], item[1]))
        return [
            {key: doc[key] for key in ("subject", "relation", "value", "source")} | {"score": round(score, 4)}
            for score, index in scored[:top_k]
            for doc in [self.documents[index]]
        ]


def _get_kg_index(knowledge_dir: str) -> _SlakeKnowledgeIndex:
    key = str(Path(knowledge_dir).resolve())
    with _KG_INDEX_LOCK:
        index = _KG_INDEX_CACHE.get(key)
        if index is None:
            index = _SlakeKnowledgeIndex(key)
            _KG_INDEX_CACHE[key] = index
        return index


class SlakeKnowledgeRetrieveTool(BaseTool):
    """Retrieve top-k facts from the SLAKE English knowledge graph."""

    def __init__(self, config, tool_schema):
        super().__init__(config, tool_schema)
        self.knowledge_dir = str(config.get("knowledge_dir") or os.environ.get("SLAKE_KG_DIR", "./eval/benchmarks/SLAKE/KG"))
        self.max_top_k = max(1, int(config.get("max_top_k", 5)))

    async def execute(self, instance_id, parameters, **kwargs):
        del instance_id, kwargs
        query = str(parameters.get("query", "")).strip()
        if not query:
            return ToolResponse(text="Error: `query` is required."), 0.0, {"success": False}
        try:
            top_k = max(1, min(self.max_top_k, int(parameters.get("top_k", 3))))
            facts = await asyncio.to_thread(_get_kg_index(self.knowledge_dir).search, query, top_k)
            payload = {
                "status": "success",
                "source": "SLAKE English knowledge graph",
                "query": query[:512],
                "facts": facts,
                "message": "Use these retrieved facts as supplementary knowledge; verify against the image and question.",
            }
            return ToolResponse(text=_json(payload)), 0.0, {"success": True, "num_facts": len(facts)}
        except (FileNotFoundError, ValueError, OSError) as exc:
            return ToolResponse(text=f"Error: {exc}"), 0.0, {"success": False}
