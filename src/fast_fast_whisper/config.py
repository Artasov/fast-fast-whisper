from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


def env(name: str, default: Optional[str] = None) -> Optional[str]:
    value = os.getenv(name)
    return value if value is not None else default


def env_int(name: str, default: int) -> int:
    raw = env(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Неверное значение переменной %s=%r (ожидалось целое). Используем %s.", name, raw, default)
        return default


def env_bool(name: str, default: bool) -> bool:
    raw = env(name)
    if raw is None:
        return default
    normalized = raw.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    logger.warning("Неверное булево значение переменной %s=%r. Используем %s.", name, raw, default)
    return default


@dataclass(frozen=True)
class ConcurrencySettings:
    max_concurrent_transcriptions: int
    allow_warmup_during_transcription: bool


@dataclass(frozen=True)
class Settings:
    concurrency: ConcurrencySettings


def load_settings() -> Settings:
    return Settings(
        concurrency=ConcurrencySettings(
            max_concurrent_transcriptions=env_int("WHISPER_MAX_CONCURRENT_TRANSCRIPTIONS", 1),
            allow_warmup_during_transcription=env_bool("WHISPER_ALLOW_WARMUP_DURING_TRANSCRIPTION", False),
        )
    )


__all__ = [
    "ConcurrencySettings",
    "Settings",
    "env",
    "env_bool",
    "env_int",
    "load_settings",
]
