from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelInfo:
    api_name: str
    storage_dir: str


# Реестр поддерживаемых моделей
MODEL_REGISTRY: dict[str, ModelInfo] = {
    'tiny': ModelInfo('tiny', 'models--Systran--faster-whisper-tiny'),
    'tiny.en': ModelInfo('tiny.en', 'models--Systran--faster-whisper-tiny.en'),
    'base': ModelInfo('base', 'models--Systran--faster-whisper-base'),
    'base.en': ModelInfo('base.en', 'models--Systran--faster-whisper-base.en'),
    'small': ModelInfo('small', 'models--Systran--faster-whisper-small'),
    'small.en': ModelInfo('small.en', 'models--Systran--faster-whisper-small.en'),
    'medium': ModelInfo('medium', 'models--Systran--faster-whisper-medium'),
    'medium.en': ModelInfo('medium.en', 'models--Systran--faster-whisper-medium.en'),
    'large': ModelInfo('large', 'models--Systran--faster-whisper-large-v3'),
    'large-v1': ModelInfo('large-v1', 'models--Systran--faster-whisper-large-v1'),
    'large-v2': ModelInfo('large-v2', 'models--Systran--faster-whisper-large-v2'),
    'large-v3': ModelInfo('large-v3', 'models--Systran--faster-whisper-large-v3'),
}


__all__ = ['ModelInfo', 'MODEL_REGISTRY']
