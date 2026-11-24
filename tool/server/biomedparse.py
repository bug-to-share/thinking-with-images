from __future__ import annotations

import argparse
import base64
import contextlib
import io
import os
import sys
import threading
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image


BIOMEDPARSE_ROOT = Path(os.environ.get('BIOMEDPARSE_ROOT', 'external/BiomedParse')).resolve()
BIOMEDPARSE_CONFIG = BIOMEDPARSE_ROOT / 'configs' / 'biomedparse_inference.yaml'

if str(BIOMEDPARSE_ROOT) not in sys.path:
    sys.path.insert(0, str(BIOMEDPARSE_ROOT))


def pil_image_from_base64(image_base64: str) -> Image.Image:
    payload = image_base64.split(',', 1)[1] if ',' in image_base64 else image_base64
    return Image.open(BytesIO(base64.b64decode(payload))).convert('RGB')


def pil_image_to_base64(image: Image.Image, image_format: str = 'PNG') -> str:
    buffer = BytesIO()
    image.save(buffer, format=image_format)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def load_input_image(payload: Dict[str, Any]) -> Image.Image:
    image_base64 = payload.get('image_base64')
    image_path = payload.get('image_path')
    if image_base64:
        return pil_image_from_base64(str(image_base64))
    if image_path:
        return Image.open(str(image_path)).convert('RGB')
    raise ValueError('One of `image_base64` or `image_path` is required.')


def mask_to_bbox(mask: np.ndarray) -> Optional[list[int]]:
    binary = np.asarray(mask) > 0
    if not binary.any():
        return None
    ys, xs = np.where(binary)
    return [int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1]


def compute_mask_mean_prob(mask_prob: np.ndarray, binary_mask: np.ndarray) -> Optional[float]:
    prob = np.asarray(mask_prob, dtype=np.float32)
    binary = np.asarray(binary_mask).astype(bool)
    if not binary.any():
        return None
    return float(prob[binary].mean())


def binary_mask_to_base64(binary_mask: np.ndarray) -> str:
    mask = np.asarray(binary_mask).astype(np.uint8) * 255
    return pil_image_to_base64(Image.fromarray(mask))


def overlay_to_base64(image_rgb: np.ndarray, binary_mask: np.ndarray, alpha: float = 0.45) -> str:
    image = np.asarray(image_rgb).astype(np.uint8)
    mask = np.asarray(binary_mask).astype(bool)
    overlay = image.copy().astype(np.float32)
    color_arr = np.array((255, 0, 0), dtype=np.float32)
    overlay[mask] = overlay[mask] * (1.0 - alpha) + color_arr * alpha
    overlay = overlay.clip(0, 255).astype(np.uint8)
    return pil_image_to_base64(Image.fromarray(overlay))


class Backend:
    def __init__(self, device: str):
        from inference_utils.inference import interactive_infer_image
        from modeling import build_model
        from modeling.BaseModel import BaseModel
        from utilities.arguments import load_opt_from_config_files
        from utilities.constants import BIOMED_CLASSES
        from utilities.distributed import init_distributed
        import torch

        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()
        self.interactive_infer_image = interactive_infer_image

        with contextlib.redirect_stdout(io.StringIO()):
            opt = load_opt_from_config_files([str(BIOMEDPARSE_CONFIG)])
            opt = init_distributed(opt)
            self.model = BaseModel(opt, build_model(opt)).from_pretrained('hf_hub:microsoft/BiomedParse').eval().to(self.device)
            with torch.no_grad():
                self.model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(BIOMED_CLASSES + ['background'], is_eval=True)

    def run(self, image: Image.Image, description: str) -> Dict[str, Any]:
        image_np = np.asarray(image)
        with self.lock:
            with contextlib.redirect_stdout(io.StringIO()):
                mask_prob = self.interactive_infer_image(self.model, Image.fromarray(image_np), description)[0]
        mask_prob = np.asarray(mask_prob, dtype=np.float32)
        binary_mask = mask_prob > 0.5
        return {
            'status': 'success',
            'description': description,
            'image_size': [image.width, image.height],
            'bbox': mask_to_bbox(binary_mask),
            'foreground_pixels': int(binary_mask.sum()),
            'mask_mean_prob': compute_mask_mean_prob(mask_prob, binary_mask),
            'mask_base64': binary_mask_to_base64(binary_mask),
            'overlay_base64': overlay_to_base64(image_np, binary_mask),
        }


def build_app(backend: Backend) -> Flask:
    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health() -> Any:
        return jsonify(
            {
                'status': 'online',
                'service': 'biomedparse',
                'device': backend.device,
                'config': str(BIOMEDPARSE_CONFIG),
            }
        )

    @app.route('/segment', methods=['POST'])
    def segment() -> Any:
        data = request.get_json(silent=True) or {}
        description = str(data.get('description', '')).strip()
        if not description:
            return jsonify({'status': 'error', 'message': 'Missing required field `description`.'}), 400
        try:
            image = load_input_image(data)
            result = backend.run(image=image, description=description)
            return jsonify(result)
        except Exception as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 500

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='BioMedParse HTTP inference server.')
    parser.add_argument('--host', default=os.environ.get('HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '6001')))
    parser.add_argument('--device', default=os.environ.get('BIOMEDPARSE_DEVICE', 'auto'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = Backend(device=args.device)
    app = build_app(backend)
    print(f'BioMedParse server listening on http://{args.host}:{args.port}', flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
