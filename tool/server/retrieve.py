from __future__ import annotations

import argparse
import json
import os
import threading
from pathlib import Path
from typing import Any, Dict, List

import faiss
import numpy as np
import open_clip
import torch
from flask import Flask, jsonify, request


DEFAULT_KNOWLEDGE_DIR = Path('./knowledge/adaptive_kb')
DEFAULT_KB_PATH = DEFAULT_KNOWLEDGE_DIR / 'knowledge_entries.jsonl'
DEFAULT_INDEX_PATH = DEFAULT_KNOWLEDGE_DIR / 'knowledge_faiss.index'
DEFAULT_MODEL_NAME = 'hf-hub:microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224'


def normalize_text(value: Any) -> str:
    if value is None:
        return ''
    if isinstance(value, list):
        return ' '.join(normalize_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False)
    return str(value).strip()


class Backend:
    def __init__(self, kb_path: Path, index_path: Path, model_name: str, device: str):
        self.kb_path = kb_path
        self.index_path = index_path
        self.model_name = model_name
        self.device = device if device != 'auto' else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.lock = threading.Lock()

        self.model, _, _ = open_clip.create_model_and_transforms(model_name, device=self.device)
        self.tokenizer = open_clip.get_tokenizer(model_name)
        self.hf_tokenizer = getattr(self.tokenizer, 'tokenizer', None)
        if self.hf_tokenizer is None:
            raise RuntimeError('BiomedCLIP tokenizer backend is missing')
        self.model.eval()

        self.entries = self._read_jsonl(self.kb_path)
        if not self.index_path.exists():
            raise FileNotFoundError(f'FAISS index not found: {self.index_path}')
        self.index = faiss.read_index(str(self.index_path))
        if self.index.ntotal != len(self.entries):
            raise ValueError(f'Index size {self.index.ntotal} does not match knowledge entries {len(self.entries)}')

    @staticmethod
    def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        with path.open('r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                rows.append(json.loads(line))
        return rows

    def _encode_queries(self, queries: List[str]) -> np.ndarray:
        with self.lock, torch.no_grad():
            encoded = self.hf_tokenizer(
                queries,
                padding='max_length',
                truncation=True,
                max_length=256,
                return_tensors='pt',
            )
            input_ids = encoded['input_ids'].to(self.device)
            features = self.model.encode_text(input_ids)
            features = features / features.norm(dim=-1, keepdim=True)
            return features.cpu().numpy().astype(np.float32)

    def _format_entry(self, entry: Dict[str, Any], score: float, rank: int) -> Dict[str, Any]:
        summary_parts = [
            normalize_text(entry.get('title')),
            normalize_text(entry.get('canonical_fact')),
            normalize_text(entry.get('evidence_summary')),
        ]
        return {
            'title': normalize_text(entry.get('title')),
            'summary': '\n'.join(part for part in summary_parts if part),
            'score': float(score),
            'rank': rank,
        }

    def search_many(self, queries: List[str], topk: int) -> List[List[Dict[str, Any]]]:
        query_embeddings = self._encode_queries(queries)
        scores, indices = self.index.search(query_embeddings, topk)
        all_results: List[List[Dict[str, Any]]] = []
        for query_idx, neighbors in enumerate(indices):
            query_results: List[Dict[str, Any]] = []
            for rank, item_idx in enumerate(neighbors, start=1):
                if item_idx < 0 or item_idx >= len(self.entries):
                    continue
                query_results.append(
                    self._format_entry(
                        entry=self.entries[int(item_idx)],
                        score=float(scores[query_idx][rank - 1]),
                        rank=rank,
                    )
                )
            all_results.append(query_results)
        return all_results


def build_app(backend: Backend) -> Flask:
    app = Flask(__name__)

    @app.route('/health', methods=['GET'])
    def health() -> Any:
        return jsonify(
            {
                'status': 'online',
                'service': 'retrieve',
                'device': backend.device,
                'indexed_items': len(backend.entries),
                'kb_path': str(backend.kb_path),
                'index_path': str(backend.index_path),
            }
        )

    @app.route('/retrieve', methods=['POST'])
    def retrieve() -> Any:
        data = request.get_json(silent=True) or {}
        topk = int(data.get('topk', 3))
        if topk <= 0:
            return jsonify({'status': 'error', 'message': '`topk` must be positive.'}), 400

        single_query = normalize_text(data.get('query'))
        query_list = data.get('queries')

        if single_query:
            queries = [single_query]
            batched = False
        elif isinstance(query_list, list) and query_list:
            queries = [normalize_text(item) for item in query_list if normalize_text(item)]
            batched = True
        else:
            return jsonify({'status': 'error', 'message': 'Provide `query` or non-empty `queries`.'}), 400

        try:
            results = backend.search_many(queries=queries, topk=topk)
            if batched:
                return jsonify({'status': 'success', 'queries': queries, 'topk': topk, 'results': results})
            return jsonify({'status': 'success', 'query': queries[0], 'topk': topk, 'results': results[0]})
        except Exception as exc:
            return jsonify({'status': 'error', 'message': str(exc)}), 500

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Adaptive KB retrieve HTTP server.')
    parser.add_argument('--host', default=os.environ.get('HOST', '127.0.0.1'))
    parser.add_argument('--port', type=int, default=int(os.environ.get('PORT', '6000')))
    parser.add_argument('--kb-path', default=os.environ.get('KB_PATH', str(DEFAULT_KB_PATH)))
    parser.add_argument('--index-path', default=os.environ.get('INDEX_PATH', str(DEFAULT_INDEX_PATH)))
    parser.add_argument('--model-name', default=os.environ.get('RETRIEVE_MODEL_NAME', DEFAULT_MODEL_NAME))
    parser.add_argument('--device', default=os.environ.get('RETRIEVE_DEVICE', 'auto'))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    backend = Backend(
        kb_path=Path(args.kb_path),
        index_path=Path(args.index_path),
        model_name=args.model_name,
        device=args.device,
    )
    app = build_app(backend)
    print(f'Retrieve server listening on http://{args.host}:{args.port}', flush=True)
    app.run(host=args.host, port=args.port, debug=False, threaded=True)


if __name__ == '__main__':
    main()
