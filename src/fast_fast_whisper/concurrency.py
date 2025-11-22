from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class ConcurrencyLimits:
    max_concurrent_transcriptions: int
    allow_warmup_during_transcription: bool


class ConcurrencyController:
    def __init__(self, limits: ConcurrencyLimits) -> None:
        self._limits = limits
        self._lock = asyncio.Lock()
        self._active_transcriptions = 0
        self._active_warmups = 0

    @property
    def limits(self) -> ConcurrencyLimits:
        return self._limits

    async def try_start_transcription(self) -> Tuple[bool, str]:
        async with self._lock:
            max_transcriptions = self._limits.max_concurrent_transcriptions
            if max_transcriptions > 0 and self._active_transcriptions >= max_transcriptions:
                return False, (
                    f"Сервер уже обрабатывает {self._active_transcriptions} запрос(ов) распознавания. "
                    "Дождитесь завершения и повторите попытку."
                )

            if not self._limits.allow_warmup_during_transcription and self._active_warmups > 0:
                return (
                    False,
                    "Warmup выполняется, параллельные транскрибации отключены переменной "
                    "WHISPER_ALLOW_WARMUP_DURING_TRANSCRIPTION=0.",
                )

            self._active_transcriptions += 1
            return True, ""

    async def finish_transcription(self) -> None:
        async with self._lock:
            if self._active_transcriptions > 0:
                self._active_transcriptions -= 1

    async def try_start_warmup(self) -> Tuple[bool, str]:
        async with self._lock:
            if not self._limits.allow_warmup_during_transcription and self._active_transcriptions > 0:
                return (
                    False,
                    "Выполняется транскрибация, а одновременный warmup запрещён переменной "
                    "WHISPER_ALLOW_WARMUP_DURING_TRANSCRIPTION=0.",
                )

            self._active_warmups += 1
            return True, ""

    async def finish_warmup(self) -> None:
        async with self._lock:
            if self._active_warmups > 0:
                self._active_warmups -= 1


__all__ = ["ConcurrencyController", "ConcurrencyLimits"]
