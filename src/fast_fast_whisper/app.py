from __future__ import annotations

import asyncio
import io
import logging
import time
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from .config import env, load_settings
from .concurrency import ConcurrencyController, ConcurrencyLimits
from .engine import (
    WhisperEngine,
    WhisperModel,
    download_model_files,
    model_files_on_disk,
    model_storage_candidates,
)
from .model_catalog import MODEL_REGISTRY

logger = logging.getLogger(__name__)

settings = load_settings()
concurrency_limits = ConcurrencyLimits(
    max_concurrent_transcriptions=settings.concurrency.max_concurrent_transcriptions,
    allow_warmup_during_transcription=settings.concurrency.allow_warmup_during_transcription,
)
concurrency_controller = ConcurrencyController(concurrency_limits)


class _SimpleResponse(BaseModel):
    text: str


class _ModelActionRequest(BaseModel):
    model: str


class _WarmupRequest(_ModelActionRequest):
    device: Optional[str] = None


app = FastAPI(title="fast-fast-whisper", version="0.1.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def _normalize_device(device: Optional[str]) -> Optional[str]:
    if device is None:
        return None
    normalized = device.lower().strip()
    if normalized not in ["cpu", "cuda", "gpu", "auto"]:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid device: {device}. Supported devices: cpu, cuda, gpu, auto",
        )
    return "cuda" if normalized == "gpu" else normalized


def _validate_file(upload: UploadFile) -> None:
    if not upload:
        raise HTTPException(status_code=400, detail="Missing file")


async def _read_upload_to_memory(upload: UploadFile) -> io.BytesIO:
    data = await upload.read()
    if not data:
        raise HTTPException(status_code=400, detail="Empty file")
    return io.BytesIO(data)


def _assert_model_supported(name: str) -> str:
    if name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Model '{name}' is not available. "
                f"Supported models: {', '.join(sorted(MODEL_REGISTRY.keys()))}"
            ),
        )
    return name


def _resolve_model_name(requested_model: Optional[str]) -> str:
    configured = (env("WHISPER_MODEL") or "").strip()
    if configured:
        return _assert_model_supported(configured)
    if requested_model:
        return _assert_model_supported(requested_model)
    raise HTTPException(
        status_code=400,
        detail=(
            "Model name is required. Pass it explicitly via the 'model' form field "
            "or set the WHISPER_MODEL environment variable."
        ),
    )


def _require_model(model_name: Optional[str]) -> str:
    if not model_name:
        raise HTTPException(status_code=400, detail="Model name is required")
    normalized = model_name.strip()
    if not normalized:
        raise HTTPException(status_code=400, detail="Model name is required")
    return _assert_model_supported(normalized)


def _format_timestamp(seconds: float, *, separator: str) -> str:
    millis_total = max(0, int(round(seconds * 1000)))
    hours, remainder = divmod(millis_total, 3600_000)
    minutes, remainder = divmod(remainder, 60_000)
    secs, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{separator}{millis:03d}"


def _segments_to_srt(segments: List[Dict[str, Any]]) -> str:
    if not segments:
        return ""
    lines: List[str] = []
    for idx, segment in enumerate(segments, start=1):
        start = _format_timestamp(float(segment.get("start", 0.0)), separator=",")
        end = _format_timestamp(float(segment.get("end", 0.0)), separator=",")
        text = (segment.get("text") or "").strip()
        lines.extend([str(idx), f"{start} --> {end}", text, ""])
    return "\n".join(lines).strip() + "\n"


def _segments_to_vtt(segments: List[Dict[str, Any]]) -> str:
    lines: List[str] = ["WEBVTT", ""]
    for segment in segments:
        start = _format_timestamp(float(segment.get("start", 0.0)), separator=".")
        end = _format_timestamp(float(segment.get("end", 0.0)), separator=".")
        text = (segment.get("text") or "").strip()
        lines.extend([f"{start} --> {end}", text, ""])
    return "\n".join(lines).strip() + "\n"


async def _handle_transcription(
    *,
    file: UploadFile,
    model: Optional[str],
    prompt: Optional[str],
    response_format: str,
    temperature: Optional[float],
    language: Optional[str],
    device: Optional[str],
):
    _validate_file(file)
    model_name = _resolve_model_name(model)
    device = _normalize_device(device)
    logger.info(
        "Received request for transcription: requested_model=%s, resolved_model=%s, format=%s, language=%s, "
        "temperature=%s, device=%s, file=%s, size=%s",
        model,
        model_name,
        response_format,
        language,
        temperature,
        device,
        file.filename,
        getattr(file, "size", "unknown"),
    )

    acquired = False
    allowed, reason = await concurrency_controller.try_start_transcription()
    if not allowed:
        raise HTTPException(status_code=429, detail=reason)
    acquired = True

    try:
        engine = await asyncio.to_thread(WhisperEngine.get, model_name, device_override=device)
        audio = await _read_upload_to_memory(file)
        result = await asyncio.to_thread(
            engine.transcribe,
            file_like=audio,
            language=language,
            temperature=temperature,
            prompt=prompt,
        )
    finally:
        if acquired:
            await concurrency_controller.finish_transcription()

    text = result["text"]
    segments = result["segments"]
    logger.info("Processing completed: %s segments, text length=%s", len(segments), len(text))

    fmt = (response_format or "json").lower()
    if fmt == "json":
        return JSONResponse(content={"text": text})
    if fmt == "text":
        return PlainTextResponse(content=text, media_type="text/plain; charset=utf-8")
    if fmt == "verbose_json":
        return JSONResponse(content=result)
    if fmt == "srt":
        return PlainTextResponse(content=_segments_to_srt(segments), media_type="application/x-subrip")
    if fmt == "vtt":
        return PlainTextResponse(content=_segments_to_vtt(segments), media_type="text/vtt; charset=utf-8")

    raise HTTPException(status_code=400, detail=f"Unsupported response_format: {response_format}")


@app.get("/")
def root() -> Dict[str, str]:
    return {"message": "fast-fast-whisper: local OpenAI-compatible Whisper API"}


@app.post("/v1/audio/transcriptions")
async def transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
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
    )


@app.post("/v1/models/download")
async def download_model_endpoint(payload: _ModelActionRequest):
    model_name = _require_model(payload.model)
    logger.info("Received request to download model %s", model_name)

    try:
        result = await asyncio.to_thread(download_model_files, model_name)
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Failed to download model %s: %s", model_name, exc)
        raise HTTPException(status_code=500, detail=f"Failed to download model {model_name}: {exc}")

    status = "already_present" if result["existed_before"] else "downloaded"
    return {
        "status": status,
        "model": model_name,
        "model_path": result["model_path"],
        "download_root": result["download_root"],
        "elapsed": result["elapsed"],
    }


@app.get("/download/model/exists")
async def model_exists(model: str):
    model_name = _require_model(model)
    exists, model_path = await asyncio.to_thread(model_files_on_disk, model_name)
    return {
        "model": model_name,
        "exists": exists,
        "model_path": model_path,
    }


@app.post("/v1/models/warmup")
async def warmup_model_endpoint(payload: _WarmupRequest):
    model_name = _require_model(payload.model)
    device = _normalize_device(payload.device)
    logger.info("Received request to warmup model %s on device=%s", model_name, device or "auto")

    allowed, reason = await concurrency_controller.try_start_warmup()
    if not allowed:
        raise HTTPException(status_code=409, detail=reason)

    started = time.time()
    try:
        engine = await asyncio.to_thread(WhisperEngine.get, model_name, device_override=device)
    except Exception as exc:
        logger.error("Failed to warmup model %s: %s", model_name, exc)
        raise HTTPException(status_code=500, detail=f"Failed to warmup model {model_name}: {exc}")
    finally:
        await concurrency_controller.finish_warmup()

    return {
        "status": "ready",
        "model": model_name,
        "device": engine.device,
        "compute_type": engine.compute_type,
        "load_time": time.time() - started,
    }


@app.get("/health")
def health() -> Dict[str, str]:
    try:
        _ = env("WHISPER_MODEL", "base")
        return {"status": "ok"}
    except Exception as exc:  # pragma: no cover
        raise HTTPException(status_code=500, detail=str(exc))


__all__ = [
    "app",
    "WhisperEngine",
    "WhisperModel",
    "concurrency_controller",
    "model_storage_candidates",
]
