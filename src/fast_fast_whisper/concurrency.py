from __future__ import annotations

import asyncio
from dataclasses import dataclass


@dataclass(frozen=True)
class ConcurrencyLimits:
    max_concurrent_transcriptions: int
    allow_warmup_during_transcription: bool


class ConcurrencyController:
    """Простой контроллер конкурентности на базе Semaphore."""
    
    def __init__(self, limits: ConcurrencyLimits) -> None:
        self._limits = limits
        max_t = limits.max_concurrent_transcriptions
        # 0 = без лимита, используем большое число
        self._semaphore = asyncio.Semaphore(max_t if max_t > 0 else 1000)
        self._active_warmups = 0
        self._lock = asyncio.Lock()
    
    @property
    def limits(self) -> ConcurrencyLimits:
        return self._limits
    
    async def acquire_transcription(self) -> tuple[bool, str]:
        """Пытается захватить слот для транскрипции."""
        # Проверяем warmup блокировку
        if not self._limits.allow_warmup_during_transcription:
            async with self._lock:
                if self._active_warmups > 0:
                    return False, 'Warmup выполняется, транскрипции временно недоступны'
        
        # Пробуем захватить без ожидания
        if self._semaphore.locked() and self._limits.max_concurrent_transcriptions > 0:
            return False, f'Достигнут лимит ({self._limits.max_concurrent_transcriptions}) одновременных транскрипций'
        
        await self._semaphore.acquire()
        return True, ''
    
    def release_transcription(self) -> None:
        """Освобождает слот транскрипции."""
        self._semaphore.release()
    
    async def acquire_warmup(self) -> tuple[bool, str]:
        """Пытается начать warmup."""
        async with self._lock:
            if not self._limits.allow_warmup_during_transcription:
                # Проверяем что нет активных транскрипций
                # Semaphore.locked() = True когда value=0
                max_t = self._limits.max_concurrent_transcriptions or 1000
                if self._semaphore._value < max_t:
                    return False, 'Транскрипция выполняется, warmup недоступен'
            self._active_warmups += 1
            return True, ''
    
    async def release_warmup(self) -> None:
        """Завершает warmup."""
        async with self._lock:
            if self._active_warmups > 0:
                self._active_warmups -= 1


__all__ = ['ConcurrencyController', 'ConcurrencyLimits']
