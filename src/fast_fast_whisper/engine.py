from __future__ import annotations

import gc
import io
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

from .config import env
from .model_catalog import MODEL_REGISTRY

try:
    from faster_whisper import WhisperModel
except Exception:
    WhisperModel = None

logger = logging.getLogger(__name__)

# Опционально torch для очистки GPU
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    torch = None


def _clear_gpu_memory() -> None:
    """Освобождает GPU память после транскрипции."""
    gc.collect()
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def _check_cuda_available() -> bool:
    """Проверяет доступность CUDA."""
    try:
        import ctranslate2
        cuda_types = ctranslate2.get_supported_compute_types('cuda')
        return bool(cuda_types)
    except Exception:
        return False


def model_files_on_disk(model_name: str) -> tuple[bool, Optional[str]]:
    """Проверяет наличие файлов модели на диске."""
    models_dir = Path('models')
    
    # Прямая папка модели
    direct = models_dir / model_name
    if direct.exists() and any(direct.rglob('*')):
        return True, str(direct)
    
    # HuggingFace cache формат
    info = MODEL_REGISTRY.get(model_name)
    if info:
        cache_dir = models_dir / info.storage_dir
        if cache_dir.exists() and any(cache_dir.rglob('*')):
            return True, str(cache_dir)
    
    return False, None


def model_storage_candidates(model_name: str) -> list[Path]:
    """Возвращает возможные пути хранения модели."""
    models_dir = Path('models')
    candidates = [models_dir / model_name]
    
    info = MODEL_REGISTRY.get(model_name)
    if info:
        candidates.append(models_dir / info.storage_dir)
    
    return list(dict.fromkeys(candidates))  # unique, preserving order


class WhisperEngine:
    """Singleton-обёртка над faster-whisper с ленивой загрузкой."""
    
    _instance: Optional['WhisperEngine'] = None
    _lock = threading.Lock()
    _transcribe_lock = threading.Lock()
    
    def __init__(self, model_name: str, device: str, compute_type: str):
        if WhisperModel is None:
            raise RuntimeError('faster-whisper не установлен')
        
        self.model_name = model_name
        self.device = device
        self.compute_type = compute_type
        self._last_used = time.time()
        
        models_dir = Path('models')
        models_dir.mkdir(exist_ok=True)
        
        logger.info('Загрузка модели %s (device=%s, compute_type=%s)', model_name, device, compute_type)
        
        try:
            self._model = WhisperModel(
                model_size_or_path=model_name,
                device=device,
                compute_type=compute_type,
                download_root=str(models_dir.absolute()),
            )
        except Exception as e:
            # Fallback на CPU при ошибках CUDA
            if device == 'cuda' and 'cuda' in str(e).lower():
                logger.warning('CUDA ошибка, переключаюсь на CPU: %s', e)
                self.device = 'cpu'
                self.compute_type = 'float32'
                self._model = WhisperModel(
                    model_size_or_path=model_name,
                    device='cpu',
                    compute_type='float32',
                    download_root=str(models_dir.absolute()),
                )
            else:
                raise
    
    @classmethod
    def get(cls, model_name: str, device_override: Optional[str] = None) -> 'WhisperEngine':
        """Получает или создаёт экземпляр движка."""
        device = device_override or env('WHISPER_DEVICE', 'auto')
        if device == 'gpu':
            device = 'cuda'
        
        # Автоопределение device
        if device == 'auto':
            device = 'cuda' if _check_cuda_available() else 'cpu'
        
        compute_type = env('WHISPER_COMPUTE_TYPE', 'auto')
        if compute_type == 'auto':
            compute_type = 'float16' if device == 'cuda' else 'float32'
        
        with cls._lock:
            # Переиспользуем если модель и device совпадают
            if cls._instance is not None:
                if cls._instance.model_name == model_name and cls._instance.device == device:
                    cls._instance._last_used = time.time()
                    return cls._instance
                # Освобождаем старую модель
                cls._unload_model()
            
            cls._instance = WhisperEngine(model_name, device, compute_type)
            return cls._instance
    
    @classmethod
    def _unload_model(cls) -> None:
        """Выгружает модель и освобождает память."""
        if cls._instance is not None:
            logger.info('Выгрузка модели %s', cls._instance.model_name)
            del cls._instance._model
            cls._instance = None
            _clear_gpu_memory()
    
    @classmethod
    def clear_cache(cls) -> None:
        """Очищает кэш моделей."""
        with cls._lock:
            cls._unload_model()
    
    def transcribe(
        self,
        file_like: io.BufferedReader | io.BytesIO,
        language: Optional[str] = None,
        temperature: Optional[float] = None,
        prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Выполняет транскрипцию аудио."""
        self._last_used = time.time()
        
        with self._transcribe_lock:
            started = time.time()
            logger.info('Начало транскрипции (device=%s, language=%s)', self.device, language)
            
            kwargs: dict[str, Any] = {'audio': file_like}
            if language:
                kwargs['language'] = language
            if temperature is not None:
                kwargs['temperature'] = temperature
            if prompt:
                kwargs['initial_prompt'] = prompt
            
            try:
                segments_gen, info = self._model.transcribe(**kwargs)
                
                # ВАЖНО: полностью consume generator чтобы не было утечек
                segments = []
                text_parts = []
                for idx, seg in enumerate(segments_gen):
                    text_parts.append(seg.text or '')
                    segments.append({
                        'id': idx,
                        'start': float(seg.start) if seg.start else 0.0,
                        'end': float(seg.end) if seg.end else 0.0,
                        'text': seg.text or '',
                    })
                
                result = {
                    'language': getattr(info, 'language', None),
                    'duration': float(getattr(info, 'duration', 0) or 0),
                    'text': ''.join(text_parts).strip(),
                    'segments': segments,
                }
                
                elapsed = time.time() - started
                logger.info(
                    'Транскрипция завершена: %d сегментов, %.2fs, язык=%s',
                    len(segments), elapsed, result['language']
                )
                
                return result
                
            finally:
                # Очистка памяти после каждой транскрипции
                _clear_gpu_memory()


def download_model_files(model_name: str) -> dict[str, Any]:
    """Скачивает файлы модели."""
    if WhisperModel is None:
        raise RuntimeError('faster-whisper не установлен')
    
    models_dir = Path('models')
    models_dir.mkdir(exist_ok=True)
    
    existed_before, _ = model_files_on_disk(model_name)
    started = time.time()
    
    # Загружаем на CPU для скачивания
    model = WhisperModel(
        model_size_or_path=model_name,
        device='cpu',
        compute_type='float32',
        download_root=str(models_dir.absolute()),
    )
    del model
    _clear_gpu_memory()
    
    return {
        'model': model_name,
        'download_root': str(models_dir.absolute()),
        'model_path': str(models_dir / model_name),
        'existed_before': existed_before,
        'elapsed': time.time() - started,
    }


__all__ = [
    'WhisperEngine',
    'WhisperModel',
    'download_model_files',
    'model_files_on_disk',
    'model_storage_candidates',
]
