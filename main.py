import io
import os
import time
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.responses import PlainTextResponse, JSONResponse
from pydantic import BaseModel

# Optional: lazy imports so app can start even if heavy deps are not ready
try:
    from faster_whisper import WhisperModel
except Exception:  # pragma: no cover
    WhisperModel = None  # type: ignore

import srt as srt_lib
import webvtt

# Настройка логирования
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('whisper_api.log', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Добавляем логирование при запуске
logger.info("=== Fast-Fast-Whisper API запущен ===")
print("=== Fast-Fast-Whisper API запущен ===")


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default


def _check_cudnn_availability() -> bool:
    """Проверяет доступность cuDNN для корректной работы с CUDA"""
    try:
        import ctranslate2
        # Пробуем создать простую модель для проверки cuDNN
        test_model = ctranslate2.Translator("Helsinki-NLP/opus-mt-en-ru", device="cuda")
        del test_model  # Освобождаем память
        return True
    except Exception as e:
        error_msg = str(e).lower()
        if any(keyword in error_msg for keyword in ['cudnn', 'dll', 'library', 'tensor', 'descriptor']):
            logger.warning(f"cuDNN недоступен: {e}")
            return False
        # Если ошибка не связана с cuDNN, считаем что cuDNN доступен
        return True


def _check_cuda_toolkit() -> bool:
    """Проверяет установку CUDA Toolkit"""
    try:
        import subprocess
        result = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, timeout=5)
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception):
        return False


def _get_cuda_diagnostic_info() -> str:
    """Получает диагностическую информацию о CUDA"""
    info = []
    
    # Проверяем драйвер NVIDIA
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            info.append("✓ NVIDIA драйвер установлен")
        else:
            info.append("✗ NVIDIA драйвер не найден")
    except Exception:
        info.append("✗ Не удалось проверить NVIDIA драйвер")
    
    # Проверяем CUDA Toolkit
    if _check_cuda_toolkit():
        info.append("✓ CUDA Toolkit установлен")
    else:
        info.append("✗ CUDA Toolkit не установлен")
    
    # Проверяем ctranslate2
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types('cuda')
        if cuda_types:
            info.append(f"✓ ctranslate2 поддерживает CUDA: {cuda_types}")
        else:
            info.append("✗ ctranslate2 не поддерживает CUDA")
    except Exception as e:
        info.append(f"✗ Ошибка ctranslate2: {e}")
    
    return "\n".join(info)


class ModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "system"


class ModelList(BaseModel):
    object: str = "list"
    data: List[ModelInfo]


class VerboseSegment(BaseModel):
    id: int
    seek: int
    start: float
    end: float
    text: str
    tokens: List[int] = []
    temperature: float = 0.0
    avg_logprob: float = 0.0
    compression_ratio: float = 0.0
    no_speech_prob: float = 0.0


class VerboseResponse(BaseModel):
    task: str = "transcribe"
    language: Optional[str]
    duration: Optional[float]
    text: str
    segments: List[VerboseSegment]


app = FastAPI(title="fast-fast-whisper", version="0.1.0")

# Поддерживаемые модели Whisper
SUPPORTED_MODELS = [
    "tiny", "tiny.en",
    "base", "base.en",
    "small", "small.en",
    "medium", "medium.en",
    "large", "large-v1", "large-v2", "large-v3"
]


def _validate_model(model_name: str) -> str:
    """Валидация и нормализация имени модели"""
    # Проверяем, что модель поддерживается
    if model_name not in SUPPORTED_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported model: {model_name}. Supported models: {', '.join(SUPPORTED_MODELS)}"
        )

    return model_name


class WhisperEngine:
    _instances: Dict[str, "WhisperEngine"] = {}

    @classmethod
    def clear_cache(cls) -> None:
        """Очищает кэш экземпляров для перезагрузки моделей"""
        cls._instances.clear()

    def __init__(self, model_name: str, device_override: Optional[str] = None) -> None:
        self.model_name = model_name
        self.device = device_override or _env("WHISPER_DEVICE", "auto")
        self.compute_type = _env("WHISPER_COMPUTE_TYPE", "auto")
        self.cpu_threads = int(_env("WHISPER_CPU_THREADS", "0") or 0) or None

        # Проверяем доступность CUDA и принудительно используем GPU если доступен
        if self.device == "auto":
            try:
                import ctranslate2
                cuda_compute_types = ctranslate2.get_supported_compute_types('cuda')
                if cuda_compute_types:
                    # Дополнительная проверка cuDNN перед использованием GPU
                    if _check_cudnn_availability():
                        self.device = "cuda"
                        logger.info(f"CUDA и cuDNN доступны, принудительно переключаемся на GPU. Доступные типы вычислений: {cuda_compute_types}")
                    else:
                        self.device = "cpu"
                        logger.warning("CUDA доступна, но cuDNN недоступен, используем CPU")
                        logger.warning("Для использования GPU установите CUDA Toolkit и cuDNN:")
                        logger.warning("1. Скачайте CUDA Toolkit с https://developer.nvidia.com/cuda-downloads")
                        logger.warning("2. Скачайте cuDNN с https://developer.nvidia.com/cudnn")
                        logger.warning("3. Добавьте пути к библиотекам в переменную PATH")
                else:
                    self.device = "cpu"
                    logger.info("CUDA недоступна, используем CPU")
            except Exception as e:
                logger.warning(f"Не удалось проверить доступность CUDA: {e}")
                self.device = "cpu"
        elif self.device == "cuda":
            # Если явно запрошен CUDA, проверяем его доступность
            try:
                import ctranslate2
                cuda_compute_types = ctranslate2.get_supported_compute_types('cuda')
                if not cuda_compute_types:
                    logger.warning("CUDA недоступна, но запрошена. Переключаемся на CPU")
                    self.device = "cpu"
                elif not _check_cudnn_availability():
                    logger.warning("cuDNN недоступен, но запрошен CUDA. Переключаемся на CPU")
                    self.device = "cpu"
            except Exception as e:
                logger.warning(f"Ошибка при проверке CUDA: {e}. Переключаемся на CPU")
                self.device = "cpu"
        
        # Дополнительная проверка: если установлена переменная FORCE_CPU, принудительно используем CPU
        if _env("FORCE_CPU", "").lower() in ("true", "1", "yes"):
            self.device = "cpu"
            logger.info("Принудительное использование CPU (FORCE_CPU=true)")

        # Нормализуем compute_type для ctranslate2
        if self.compute_type == "auto":
            if self.device == "cuda":
                self.compute_type = "float16"  # Оптимальный тип для GPU
            else:
                self.compute_type = "float32"  # Оптимальный тип для CPU

        logger.info(f"Инициализация модели {model_name} с параметрами: device={self.device}, compute_type={self.compute_type}")

        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not installed. Ensure dependencies are installed."
            )

        # Создаем папку models в корне проекта если её нет
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Используем локальную папку models для загрузки моделей
        try:
            logger.info(f"Загружаем модель {model_name} на устройство {self.device}")
            self._model = WhisperModel(
                model_size_or_path=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(models_dir.absolute()),
            )
            logger.info(f"Модель {model_name} успешно загружена на {self.device}")
        except Exception as e:
            # Проверяем, является ли ошибка связанной с cuDNN/CUDA
            error_msg = str(e).lower()
            is_cudnn_error = any(keyword in error_msg for keyword in [
                'cudnn', 'cuda', 'gpu', 'tensor', 'descriptor', 'dll', 'library'
            ])
            
            if is_cudnn_error and self.device == "cuda":
                logger.error(f"Ошибка CUDA/cuDNN: {e}")
                logger.error("Диагностическая информация:")
                logger.error(_get_cuda_diagnostic_info())
                logger.info("Переключаемся на CPU из-за проблем с CUDA/cuDNN...")
                try:
                    self.device = "cpu"
                    self.compute_type = "float32"
                    self._model = WhisperModel(
                        model_size_or_path=self.model_name,
                        device=self.device,
                        compute_type=self.compute_type,
                        download_root=str(models_dir.absolute()),
                    )
                    logger.info(f"Модель {model_name} успешно загружена на CPU после ошибки CUDA")
                except Exception as e3:
                    logger.error(f"Не удалось загрузить модель даже на CPU: {e3}")
                    raise
            else:
                # Если не ошибка CUDA, пробуем с минимальными параметрами
                logger.warning(f"Не удалось инициализировать с полными параметрами: {e}")
                logger.info("Пробуем с минимальными параметрами...")
                try:
                    self._model = WhisperModel(
                        model_size_or_path=self.model_name,
                        download_root=str(models_dir.absolute()),
                    )
                    logger.info(f"Модель {model_name} загружена с минимальными параметрами")
                except Exception as e2:
                    logger.error(f"Критическая ошибка при загрузке модели: {e2}")
                    raise

    @classmethod
    def get(cls, model_name: str, device_override: Optional[str] = None) -> "WhisperEngine":
        # Создаем уникальный ключ для кэша, включая устройство
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
        
        # Логируем параметры запроса
        logger.info(f"Начинаем распознавание: device={self.device}, compute_type={self.compute_type}, "
                   f"language={language}, temperature={temperature}, translate={translate}, prompt={'установлен' if prompt else 'не установлен'}")

        # Подготавливаем параметры для faster-whisper
        transcribe_kwargs = {
            "audio": file_like,
            "task": "translate" if translate else "transcribe",
        }

        # Добавляем параметры только если они не None
        if language is not None:
            transcribe_kwargs["language"] = language
        if temperature is not None:
            transcribe_kwargs["temperature"] = temperature
        if prompt is not None:
            transcribe_kwargs["initial_prompt"] = prompt

        logger.info(f"Параметры распознавания: {transcribe_kwargs}")

        try:
            segments_obj, info = self._model.transcribe(**transcribe_kwargs)
            processing_time = time.time() - started
            logger.info(f"Распознавание завершено за {processing_time:.2f} секунд")
        except Exception as e:
            logger.error(f"Ошибка во время распознавания: {e}")
            logger.error(f"Параметры распознавания: {transcribe_kwargs}")
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
        
        # Логируем результат
        logger.info(f"Результат распознавания: язык={detected_language}, длительность={result_duration:.2f}с, "
                   f"сегментов={len(segments)}, текст='{result_text[:100]}{'...' if len(result_text) > 100 else ''}'")

        return {
            "language": detected_language,
            "duration": result_duration,
            "text": result_text,
            "segments": segments,
        }


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "fast-fast-whisper: local OpenAI-compatible Whisper API"}


@app.get("/v1/models", response_model=ModelList)
def list_models() -> ModelList:
    # Возвращаем список всех поддерживаемых моделей Whisper
    created = int(time.time())
    models = []
    for model_name in SUPPORTED_MODELS:
        models.append(ModelInfo(
            id=model_name,
            created=created,
            owned_by="openai"
        ))
    return ModelList(data=models)


def _validate_file(upload: UploadFile) -> None:
    if not upload:
        raise HTTPException(status_code=400, detail="Missing file")
    # OpenAI accepts many types; we just ensure it's a binary file
    if upload.content_type and not (
            upload.content_type.startswith("audio/") or upload.content_type.startswith("video/")
    ):
        # Still allow; some clients don't set correct content_type
        pass


async def _read_upload_to_memory(upload: UploadFile) -> io.BytesIO:
    data = await upload.read()
    if not data: raise HTTPException(status_code=400, detail="Empty file")
    return io.BytesIO(data)


def _as_srt(segments: List[Dict[str, Any]]) -> str:
    items = []
    for idx, seg in enumerate(segments, start=1):
        start = seg.get("start", 0.0)
        end = seg.get("end", 0.0)
        text = (seg.get("text") or "").strip()
        items.append(
            srt_lib.Subtitle(
                index=idx,
                start=srt_lib.timedelta(seconds=float(max(0.0, start))),
                end=srt_lib.timedelta(seconds=float(max(0.0, end))),
                content=text,
            )
        )
    return srt_lib.compose(items)


def _as_vtt(segments: List[Dict[str, Any]]) -> str:
    vtt = webvtt.WebVTT()
    for seg in segments:
        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))
        text = (seg.get("text") or "").strip()

        def fmt(ts: float) -> str:
            ms = int(ts * 1000)
            hh, rem = divmod(ms, 3_600_000)
            mm, rem = divmod(rem, 60_000)
            ss, ms = divmod(rem, 1000)
            return f"{hh:02d}:{mm:02d}:{ss:02d}.{ms:03d}"

        caption = webvtt.Caption(fmt(start), fmt(end), text)
        vtt.captions.append(caption)
    buf = io.StringIO()
    vtt.write(buf)
    return buf.getvalue()


async def _handle_transcription(
        file: UploadFile,
        model: str,
        prompt: Optional[str],
        response_format: str,
        temperature: Optional[float],
        language: Optional[str],
        device: Optional[str],
        translate: bool,
):
    # Логируем входящий запрос
    logger.info(f"Получен запрос на {'перевод' if translate else 'транскрипцию'}: "
               f"модель={model}, формат={response_format}, язык={language}, "
               f"температура={temperature}, устройство={device}, файл={file.filename}, размер={file.size if hasattr(file, 'size') else 'неизвестен'}")
    
    _validate_file(file)
    # Валидируем и нормализуем имя модели
    model_name = _validate_model(model)
    
    # Валидируем параметр device
    if device is not None:
        device = device.lower().strip()
        if device not in ['cpu', 'cuda', 'gpu', 'auto']:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid device: {device}. Supported devices: cpu, cuda, gpu, auto"
            )
        # Нормализуем gpu -> cuda
        if device == 'gpu':
            device = 'cuda'

    engine = WhisperEngine.get(model_name, device_override=device)
    audio = await _read_upload_to_memory(file)

    result = engine.transcribe(
        file_like=audio,
        language=language,
        temperature=temperature,
        translate=translate,
        prompt=prompt,
    )

    text = result["text"]
    segments = result["segments"]

    # Логируем результат обработки
    logger.info(f"Обработка завершена: результат содержит {len(segments)} сегментов, "
               f"общая длина текста: {len(text)} символов")

    fmt = (response_format or "json").lower()
    if fmt == "json":
        return JSONResponse(content={"text": text})
    if fmt == "text":
        return PlainTextResponse(content=text, media_type="text/plain; charset=utf-8")
    if fmt == "srt":
        return PlainTextResponse(content=_as_srt(segments), media_type="application/x-subrip")
    if fmt == "vtt":
        return PlainTextResponse(content=_as_vtt(segments), media_type="text/vtt")
    if fmt == "verbose_json":
        payload = VerboseResponse(
            task="translate" if translate else "transcribe",
            language=result.get("language"),
            duration=result.get("duration"),
            text=text,
            segments=[VerboseSegment(**s) for s in segments],
        )
        return JSONResponse(content=payload.model_dump())

    raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")


@app.post("/v1/audio/transcriptions")
async def transcriptions(
        file: UploadFile = File(...),
        model: str = Form(...),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: Optional[float] = Form(None),
        language: Optional[str] = Form(None),
        device: Optional[str] = Form(None),
):
    return await _handle_transcription(
        file=file,
        model=model,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        language=language,
        device=device,
        translate=False,
    )


@app.post("/v1/audio/translations")
async def translations(
        file: UploadFile = File(...),
        model: str = Form(...),
        prompt: Optional[str] = Form(None),
        response_format: str = Form("json"),
        temperature: Optional[float] = Form(None),
        language: Optional[str] = Form(None),
        device: Optional[str] = Form(None),
):
    # For translations, we force translate=True regardless of language
    return await _handle_transcription(
        file=file,
        model=model,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        language=language,
        device=device,
        translate=True,
    )


@app.get("/healthz")
def healthz() -> Dict[str, str]:
    try:
        # Trigger lazy-model creation check only, without heavy load
        _ = _env("WHISPER_MODEL", "base")
        return {"status": "ok"}
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/clear_cache")
def clear_cache() -> Dict[str, str]:
    """Очищает кэш моделей для перезагрузки"""
    try:
        WhisperEngine.clear_cache()
        return {"status": "cache cleared"}
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/cuda_diagnostic")
def cuda_diagnostic() -> Dict[str, str]:
    """Возвращает диагностическую информацию о CUDA"""
    try:
        diagnostic_info = _get_cuda_diagnostic_info()
        return {
            "status": "ok",
            "diagnostic": diagnostic_info
        }
    except Exception as e:
        return {
            "status": "error",
            "diagnostic": f"Ошибка при получении диагностики: {e}"
        }
