from __future__ import annotations

import importlib.util
import os
from pathlib import Path


TARGET_PATH = Path(os.environ.get('MEDSAM3_SERVER_SCRIPT', 'external/MedSAM3/server.py')).resolve()


def load_main():
    if not TARGET_PATH.is_file():
        raise FileNotFoundError("Set MEDSAM3_SERVER_SCRIPT to the compatible MedSAM3 server entrypoint.")
    spec = importlib.util.spec_from_file_location('medsam3_project_server', TARGET_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f'Failed to load MedSAM3 server module from {TARGET_PATH}')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.main


def main() -> None:
    os.chdir(TARGET_PATH.parent)
    load_main()()


if __name__ == '__main__':
    main()
