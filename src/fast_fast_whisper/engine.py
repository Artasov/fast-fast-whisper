from __future__ import annotations

import gc
import io
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import env
from .model_catalog import MODEL_REGISTRY

try:  # pragma: no cover - import guarded for optional dependency
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover - preserved from legacy behavior
    WhisperModel = None  # type: ignore

logger = logging.getLogger(__name__)


def model_storage_candidates(model_name: str) -> List[Path]:
    models_dir = Path("models")
    candidates = [models_dir / model_name]
    model_info = MODEL_REGISTRY.get(model_name)
    if model_info:
        candidates.append(models_dir / model_info.storage_dir)

    unique: List[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        marker = str(candidate.resolve())
        if marker not in seen:
            unique.append(candidate)
            seen.add(marker)
    return unique


def _directory_has_files(directory: Path) -> bool:
    if not directory.exists():
        return False
    for child in directory.rglob("*"):
        if child.is_file():
            return True
    return False


def model_files_on_disk(model_name: str) -> Tuple[bool, Optional[str]]:
    for candidate in model_storage_candidates(model_name):
        if _directory_has_files(candidate):
            return True, str(candidate)
    return False, None


def _check_cudnn_availability() -> bool:
    try:
        import ctranslate2

        contains_devices = getattr(ctranslate2, "contains_available_devices", None)
        if callable(contains_devices):
            is_available = bool(contains_devices("cuda"))
        else:
            cuda_types = ctranslate2.get_supported_compute_types("cuda")
            is_available = bool(cuda_types)

        if not is_available:
            logger.warning("CUDA/cuDNN устройства недоступны для CTranslate2")
        return is_available
    except Exception as exc:  # pragma: no cover - depends on system libs
        error_msg = str(exc).lower()
        if any(keyword in error_msg for keyword in ["cudnn", "dll", "library", "tensor", "descriptor"]):
            logger.warning("cuDNN недоступен: %s", exc)
            return False
        return True


def _check_cuda_toolkit() -> bool:
    try:
        result = subprocess.run(["nvcc", "--version"], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):  # pragma: no cover
        return False


def get_cuda_diagnostics() -> str:
    info: List[str] = []

    try:
        result = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            info.append("✓ NVIDIA driver installed")
        else:
            info.append("✗ NVIDIA driver not found")
    except Exception:  # pragma: no cover
        info.append("✗ Failed to check NVIDIA driver")

    if _check_cuda_toolkit():
        info.append("✓ CUDA Toolkit installed")
    else:
        info.append("✗ CUDA Toolkit not installed")

    try:
        import ctranslate2

        cuda_types = ctranslate2.get_supported_compute_types("cuda")
        if cuda_types:
            info.append(f"✓ ctranslate2 supports CUDA: {cuda_types}")
        else:
            info.append("✗ ctranslate2 does not support CUDA")
    except Exception as exc:  # pragma: no cover
        info.append(f"✗ ctranslate2 error: {exc}")

    return "\n".join(info)


class WhisperEngine:
    _instances: Dict[str, "WhisperEngine"] = {}

    @classmethod
    def clear_cache(cls) -> None:
        cls._instances.clear()

    def __init__(self, model_name: str, device_override: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device_override or env("WHISPER_DEVICE", "auto")
        self.compute_type = env("WHISPER_COMPUTE_TYPE", "auto")
        self.cpu_threads = int(env("WHISPER_CPU_THREADS", "0") or 0) or None

        if self.device == "auto":
            self._autodetect_device()
        elif self.device == "cuda":
            self._validate_requested_cuda()

        if env("FORCE_CPU", "").lower() in ("true", "1", "yes"):
            self.device = "cpu"
            logger.info("Forced CPU usage (FORCE_CPU=true)")

        if self.compute_type == "auto":
            self.compute_type = "float16" if self.device == "cuda" else "float32"

        logger.info(
            "Initializing model %s with parameters: device=%s, compute_type=%s",
            model_name,
            self.device,
            self.compute_type,
        )

        if WhisperModel is None:
            raise RuntimeError("faster-whisper is not installed. Ensure dependencies are installed.")

        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        self._model = self._load_model(models_dir)

    def _autodetect_device(self) -> None:
        try:
            import ctranslate2

            cuda_types = ctranslate2.get_supported_compute_types("cuda")
            if cuda_types:
                if _check_cudnn_availability():
                    self.device = "cuda"
                    logger.info("CUDA и cuDNN доступны, используем GPU (%s)", cuda_types)
                else:
                    self.device = "cpu"
                    logger.warning("CUDA доступна, но cuDNN нет — используем CPU")
            else:
                self.device = "cpu"
                logger.info("CUDA недоступна, используем CPU")
        except Exception as exc:
            logger.warning("Не удалось проверить CUDA: %s", exc)
            self.device = "cpu"

    def _validate_requested_cuda(self) -> None:
        try:
            import ctranslate2

            cuda_types = ctranslate2.get_supported_compute_types("cuda")
            if not cuda_types:
                logger.warning("CUDA недоступна, хотя явно запрошена. Переключаемся на CPU")
                self.device = "cpu"
            elif not _check_cudnn_availability():
                logger.warning("cuDNN недоступен, хотя запрошен CUDA. Переключаемся на CPU")
                self.device = "cpu"
        except Exception as exc:
            logger.warning("Ошибка при проверке CUDA (%s). Переключаемся на CPU", exc)
            self.device = "cpu"

    def _base_model_kwargs(self, models_dir: Path, *, device: Optional[str] = None, compute_type: Optional[str] = None) -> Dict[str, Any]:
        kwargs: Dict[str, Any] = {
            "model_size_or_path": self.model_name,
            "device": device or self.device,
            "compute_type": compute_type or self.compute_type,
            "download_root": str(models_dir.absolute()),
        }
        if self.cpu_threads is not None:
            kwargs["cpu_threads"] = self.cpu_threads
        return kwargs

    def _load_model(self, models_dir: Path):
        try:
            logger.info("Загружаем модель %s на устройстве %s", self.model_name, self.device)
            return WhisperModel(**self._base_model_kwargs(models_dir))
        except Exception as exc:  # pragma: no cover - зависит от окружения
            return self._recover_from_initialization_error(models_dir, exc)

    def _recover_from_initialization_error(self, models_dir: Path, exc: Exception):
        error_msg = str(exc).lower()
        is_cudnn_error = any(
            keyword in error_msg for keyword in ["cudnn", "cuda", "gpu", "tensor", "descriptor", "dll", "library"]
        )

        if is_cudnn_error and self.device == "cuda":
            logger.error("Ошибка CUDA/cuDNN: %s", exc)
            logger.error("Диагностика:\n%s", get_cuda_diagnostics())
            logger.info("Пробуем переключиться на CPU...")
            try:
                self.device = "cpu"
                self.compute_type = "float32"
                return WhisperModel(**self._base_model_kwargs(models_dir, device="cpu", compute_type="float32"))
            except Exception as cpu_exc:
                logger.error("Даже на CPU загрузить модель не удалось: %s", cpu_exc)
                raise

        logger.warning("Не удалось инициализировать модель с указанными параметрами: %s", exc)
        logger.info("Пробуем загрузить с минимальными параметрами...")
        try:
            kwargs = {
                "model_size_or_path": self.model_name,
                "download_root": str(models_dir.absolute()),
            }
            if self.cpu_threads is not None:
                kwargs["cpu_threads"] = self.cpu_threads
            return WhisperModel(**kwargs)
        except Exception as critical:
            logger.error("Критическая ошибка загрузки модели: %s", critical)
            raise

    @classmethod
    def get(cls, model_name: str, device_override: Optional[str] = None) -> "WhisperEngine":
        cache_key = f"{model_name}_{device_override or 'auto'}"
        if cache_key not in cls._instances:
            cls._instances[cache_key] = WhisperEngine(model_name, device_override)
        return cls._instances[cache_key]

    def transcribe(
        self,
        file_like: io.BufferedReader | io.BytesIO,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        translate: bool = False,
        prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.time()
        logger.info(
            "Starting recognition: device=%s, compute_type=%s, language=%s, temperature=%s, translate=%s, prompt=%s",
            self.device,
            self.compute_type,
            language,
            temperature,
            translate,
            "set" if prompt else "not set",
        )

        transcribe_kwargs: Dict[str, Any] = {
            "audio": file_like,
            "task": "translate" if translate else "transcribe",
        }
        if language is not None:
            transcribe_kwargs["language"] = language
        if temperature is not None:
            transcribe_kwargs["temperature"] = temperature
        if prompt is not None:
            transcribe_kwargs["initial_prompt"] = prompt

        try:
            segments_obj, info = self._model.transcribe(**transcribe_kwargs)
            processing_time = time.time() - started
            logger.info("Recognition completed in %.2f seconds", processing_time)
        except Exception as exc:
            logger.error("Ошибка во время распознавания: %s", exc)
            logger.error("Параметры распознавания: %s", transcribe_kwargs)
            raise

        text_parts: List[str] = []
        segments: List[Dict[str, Any]] = []
        for idx, seg in enumerate(segments_obj):
            text_parts.append(seg.text or "")
            segments.append(
                {
                    "id": idx,
                    "seek": 0,
                    "start": float(seg.start) if seg.start is not None else 0.0,
                    "end": float(seg.end) if seg.end is not None else 0.0,
                    "text": seg.text or "",
                    "tokens": [],
                    "temperature": 0.0,
                    "avg_logprob": 0.0,
                    "compression_ratio": 0.0,
                    "no_speech_prob": 0.0,
                }
            )

        result_text = ("".join(text_parts)).strip()
        result_duration = float(getattr(info, "duration", time.time() - started) or 0.0)
        detected_language = getattr(info, "language", None)

        logger.info(
            "Recognition result: language=%s, duration=%.2fs, segments=%s, text='%s%s'",
            detected_language,
            result_duration,
            len(segments),
            result_text[:100],
            "..." if len(result_text) > 100 else "",
        )

        return {
            "language": detected_language,
            "duration": result_duration,
            "text": result_text,
            "segments": segments,
        }


def download_model_files(model_name: str) -> Dict[str, Any]:
    if WhisperModel is None:
        raise RuntimeError("faster-whisper is not installed")

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)

    model_dir = models_dir / model_name
    existed_before = model_dir.exists()
    started = time.time()
    downloader = None

    try:
        downloader = WhisperModel(
            model_size_or_path=model_name,
            device="cpu",
            compute_type="float32",
            download_root=str(models_dir.absolute()),
        )
        logger.info("Model %s downloaded into %s", model_name, model_dir)
    finally:
        if downloader is not None:
            del downloader
            gc.collect()

    return {
        "model": model_name,
        "download_root": str(models_dir.absolute()),
        "model_path": str(model_dir),
        "existed_before": existed_before,
        "elapsed": time.time() - started,
    }


__all__ = [
    "WhisperEngine",
    "WhisperModel",
    "download_model_files",
    "get_cuda_diagnostics",
    "model_files_on_disk",
    "model_storage_candidates",
]
