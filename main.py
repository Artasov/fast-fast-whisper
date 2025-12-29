from __future__ import annotations

import sys
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parent / 'src'
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from fast_fast_whisper.app import (  # noqa: F401, E402
    WhisperEngine,
    WhisperModel,
    app,
    concurrency_controller,
    model_storage_candidates,
)
from fast_fast_whisper.engine import (  # noqa: F401, E402
    download_model_files,
    model_files_on_disk,
)

__all__ = [
    'app',
    'WhisperEngine',
    'WhisperModel',
    'model_storage_candidates',
    'model_files_on_disk',
    'download_model_files',
    'concurrency_controller',
]
