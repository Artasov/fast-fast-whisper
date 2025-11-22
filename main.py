from __future__ import annotations

import sys
from pathlib import Path

_SRC_PATH = Path(__file__).resolve().parent / "src"
if str(_SRC_PATH) not in sys.path:
    sys.path.insert(0, str(_SRC_PATH))

from fast_fast_whisper.app import (  # noqa: F401
    WhisperEngine,
    WhisperModel,
    app,
    concurrency_controller,
    model_storage_candidates as _model_storage_candidates,
)
from fast_fast_whisper.engine import (  # noqa: F401
    download_model_files as _download_model_files,
    model_files_on_disk as _model_files_on_disk,
)

__all__ = [
    "app",
    "WhisperEngine",
    "WhisperModel",
    "_model_storage_candidates",
    "_model_files_on_disk",
    "_download_model_files",
    "concurrency_controller",
]
