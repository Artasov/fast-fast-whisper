import io
import os
import time
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


def _env(name: str, default: Optional[str] = None) -> Optional[str]:
    v = os.getenv(name)
    return v if v is not None else default


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

    def __init__(self, model_name: str) -> None:
        self.model_name = model_name
        self.device = _env("WHISPER_DEVICE", "auto")
        self.compute_type = _env("WHISPER_COMPUTE_TYPE", "auto")
        self.cpu_threads = int(_env("WHISPER_CPU_THREADS", "0") or 0) or None

        # Нормализуем device для ctranslate2
        if self.device == "auto":
            self.device = "cpu"  # По умолчанию используем CPU

        # Нормализуем compute_type для ctranslate2
        if self.compute_type == "auto":
            self.compute_type = "default"  # По умолчанию используем default

        if WhisperModel is None:
            raise RuntimeError(
                "faster-whisper is not installed. Ensure dependencies are installed."
            )

        # Создаем папку models в корне проекта если её нет
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)

        # Используем локальную папку models для загрузки моделей
        # Передаем только основные параметры, чтобы избежать проблем с ctranslate2
        try:
            self._model = WhisperModel(
                model_size_or_path=self.model_name,
                device=self.device,
                compute_type=self.compute_type,
                download_root=str(models_dir.absolute()),
            )
        except Exception as e:
            # Если не удается инициализировать с дополнительными параметрами,
            # попробуем с минимальными параметрами
            print(f"Warning: Failed to initialize with full parameters: {e}")
            print("Trying with minimal parameters...")
            self._model = WhisperModel(
                model_size_or_path=self.model_name,
                download_root=str(models_dir.absolute()),
            )

    @classmethod
    def get(cls, model_name: str) -> "WhisperEngine":
        if model_name not in cls._instances:
            cls._instances[model_name] = WhisperEngine(model_name)
        return cls._instances[model_name]

    def transcribe(
            self,
            file_like: io.BufferedReader | io.BytesIO,
            language: Optional[str] = None,
            temperature: Optional[float] = None,
            translate: bool = False,
            prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        started = time.time()

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

        try:
            segments_obj, info = self._model.transcribe(**transcribe_kwargs)
        except Exception as e:
            print(f"Error during transcription: {e}")
            print(f"Transcribe kwargs: {transcribe_kwargs}")
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

        return {
            "language": getattr(info, "language", None),
            "duration": float(getattr(info, "duration", time.time() - started) or 0.0),
            "text": ("".join(text_parts)).strip(),
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
        translate: bool,
):
    _validate_file(file)
    # Валидируем и нормализуем имя модели
    model_name = _validate_model(model)

    engine = WhisperEngine.get(model_name)
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
):
    return await _handle_transcription(
        file=file,
        model=model,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        language=language,
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
):
    # For translations, we force translate=True regardless of language
    return await _handle_transcription(
        file=file,
        model=model,
        prompt=prompt,
        response_format=response_format,
        temperature=temperature,
        language=language,
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
