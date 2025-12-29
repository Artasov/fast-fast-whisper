from __future__ import annotations

import asyncio
import io
import logging
from typing import Any, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from pydantic import BaseModel

from .concurrency import ConcurrencyController, ConcurrencyLimits
from .config import env, load_settings
from .engine import WhisperEngine, WhisperModel, download_model_files, model_files_on_disk, model_storage_candidates
from .model_catalog import MODEL_REGISTRY

logger = logging.getLogger(__name__)

# Настройки
settings = load_settings()
concurrency_controller = ConcurrencyController(
    ConcurrencyLimits(
        max_concurrent_transcriptions=settings.concurrency.max_concurrent_transcriptions,
        allow_warmup_during_transcription=settings.concurrency.allow_warmup_during_transcription,
    )
)

# Таймаут для транскрипции (секунды)
TRANSCRIPTION_TIMEOUT = int(env('WHISPER_TRANSCRIPTION_TIMEOUT', '600') or 600)

app = FastAPI(title='fast-fast-whisper', version='0.1.0')
app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=True,
    allow_methods=['*'],
    allow_headers=['*'],
)


# === Вспомогательные функции ===

def _normalize_device(device: Optional[str]) -> Optional[str]:
    if not device:
        return None
    d = device.lower().strip()
    if d not in ('cpu', 'cuda', 'gpu', 'auto'):
        raise HTTPException(400, f'Неверный device: {device}')
    return 'cuda' if d == 'gpu' else d


def _resolve_model(requested: Optional[str]) -> str:
    """Определяет какую модель использовать."""
    # Приоритет: env -> запрос
    configured = (env('WHISPER_MODEL') or '').strip()
    name = configured or (requested or '').strip()
    
    if not name:
        raise HTTPException(400, 'Не указана модель. Установите WHISPER_MODEL или передайте в запросе.')
    
    if name not in MODEL_REGISTRY:
        raise HTTPException(400, f'Модель "{name}" не поддерживается. Доступные: {", ".join(MODEL_REGISTRY.keys())}')
    
    return name


def _format_time(seconds: float, sep: str) -> str:
    """Форматирует время для SRT/VTT."""
    ms = max(0, int(round(seconds * 1000)))
    h, ms = divmod(ms, 3600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f'{h:02d}:{m:02d}:{s:02d}{sep}{ms:03d}'


def _to_srt(segments: list[dict]) -> str:
    lines = []
    for i, seg in enumerate(segments, 1):
        start = _format_time(seg.get('start', 0), ',')
        end = _format_time(seg.get('end', 0), ',')
        lines.extend([str(i), f'{start} --> {end}', seg.get('text', '').strip(), ''])
    return '\n'.join(lines)


def _to_vtt(segments: list[dict]) -> str:
    lines = ['WEBVTT', '']
    for seg in segments:
        start = _format_time(seg.get('start', 0), '.')
        end = _format_time(seg.get('end', 0), '.')
        lines.extend([f'{start} --> {end}', seg.get('text', '').strip(), ''])
    return '\n'.join(lines)


# === Эндпоинты ===

@app.get('/')
def root():
    return {'message': 'fast-fast-whisper: local OpenAI-compatible Whisper API'}


@app.get('/health')
def health():
    return {'status': 'ok'}


@app.post('/v1/audio/transcriptions')
async def transcriptions(
    file: UploadFile = File(...),
    model: Optional[str] = Form(None),
    prompt: Optional[str] = Form(None),
    response_format: str = Form('json'),
    temperature: Optional[float] = Form(None),
    language: Optional[str] = Form(None),
    device: Optional[str] = Form(None),
):
    if not file:
        raise HTTPException(400, 'Файл не передан')
    
    model_name = _resolve_model(model)
    device = _normalize_device(device)
    
    logger.info('Транскрипция: model=%s, format=%s, language=%s, file=%s', model_name, response_format, language, file.filename)
    
    # Проверяем лимиты
    allowed, reason = await concurrency_controller.acquire_transcription()
    if not allowed:
        raise HTTPException(429, reason)
    
    try:
        # Читаем файл в память
        data = await file.read()
        if not data:
            raise HTTPException(400, 'Пустой файл')
        audio = io.BytesIO(data)
        
        try:
            # Получаем движок и транскрибируем с таймаутом
            async def do_transcribe():
                engine = await asyncio.to_thread(WhisperEngine.get, model_name, device)
                return await asyncio.to_thread(
                    engine.transcribe,
                    file_like=audio,
                    language=language,
                    temperature=temperature,
                    prompt=prompt,
                )
            
            result = await asyncio.wait_for(do_transcribe(), timeout=TRANSCRIPTION_TIMEOUT)
        finally:
            audio.close()
    finally:
        concurrency_controller.release_transcription()
    
    # Форматируем ответ
    text = result['text']
    segments = result['segments']
    fmt = (response_format or 'json').lower()
    
    if fmt == 'json':
        return JSONResponse({'text': text})
    if fmt == 'text':
        return PlainTextResponse(text, media_type='text/plain; charset=utf-8')
    if fmt == 'verbose_json':
        return JSONResponse(result)
    if fmt == 'srt':
        return PlainTextResponse(_to_srt(segments), media_type='application/x-subrip')
    if fmt == 'vtt':
        return PlainTextResponse(_to_vtt(segments), media_type='text/vtt; charset=utf-8')
    
    raise HTTPException(400, f'Неподдерживаемый формат: {response_format}')


class _ModelRequest(BaseModel):
    model: str


class _WarmupRequest(BaseModel):
    model: str
    device: Optional[str] = None


@app.post('/v1/models/download')
async def download_model_endpoint(payload: _ModelRequest):
    model_name = _resolve_model(payload.model)
    logger.info('Скачивание модели %s', model_name)
    
    try:
        result = await asyncio.to_thread(download_model_files, model_name)
    except Exception as e:
        logger.error('Ошибка скачивания %s: %s', model_name, e)
        raise HTTPException(500, f'Ошибка скачивания: {e}')
    
    return {
        'status': 'already_present' if result['existed_before'] else 'downloaded',
        'model': model_name,
        'model_path': result['model_path'],
        'elapsed': result['elapsed'],
    }


@app.get('/download/model/exists')
async def model_exists(model: str):
    model_name = _resolve_model(model)
    exists, path = await asyncio.to_thread(model_files_on_disk, model_name)
    return {'model': model_name, 'exists': exists, 'model_path': path}


@app.post('/v1/models/warmup')
async def warmup_model_endpoint(payload: _WarmupRequest):
    model_name = _resolve_model(payload.model)
    device = _normalize_device(payload.device)
    
    logger.info('Warmup модели %s (device=%s)', model_name, device or 'auto')
    
    allowed, reason = await concurrency_controller.acquire_warmup()
    if not allowed:
        raise HTTPException(409, reason)
    
    try:
        import time
        started = time.time()
        engine = await asyncio.to_thread(WhisperEngine.get, model_name, device)
        elapsed = time.time() - started
    except Exception as e:
        logger.error('Ошибка warmup %s: %s', model_name, e)
        raise HTTPException(500, f'Ошибка warmup: {e}')
    finally:
        await concurrency_controller.release_warmup()
    
    return {
        'status': 'ready',
        'model': model_name,
        'device': engine.device,
        'compute_type': engine.compute_type,
        'load_time': elapsed,
    }


@app.get('/v1/models')
def list_models():
    """Список доступных моделей (OpenAI-совместимый формат)."""
    model_id = env('OPENAI_MODEL_ID', 'whisper-1')
    return {
        'object': 'list',
        'data': [{'id': model_id, 'object': 'model', 'owned_by': 'local'}],
    }


__all__ = [
    'app',
    'WhisperEngine',
    'WhisperModel',
    'concurrency_controller',
    'model_storage_candidates',
]
