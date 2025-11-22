import asyncio

from fast_fast_whisper.concurrency import ConcurrencyController, ConcurrencyLimits


def test_transcription_limit_blocks_second_request():
    controller = ConcurrencyController(ConcurrencyLimits(max_concurrent_transcriptions=1, allow_warmup_during_transcription=False))

    async def scenario():
        allowed_first, _ = await controller.try_start_transcription()
        assert allowed_first

        allowed_second, message = await controller.try_start_transcription()
        assert allowed_second is False
        assert "Сервер уже обрабатывает" in message

        await controller.finish_transcription()

    asyncio.run(scenario())


def test_warmup_and_transcription_mutex():
    controller = ConcurrencyController(ConcurrencyLimits(max_concurrent_transcriptions=1, allow_warmup_during_transcription=False))

    async def scenario():
        allowed_transcription, _ = await controller.try_start_transcription()
        assert allowed_transcription

        allowed_warmup, message = await controller.try_start_warmup()
        assert allowed_warmup is False
        assert "warmup" in message.lower()

        await controller.finish_transcription()

    asyncio.run(scenario())


def test_warmup_allowed_when_flag_enabled():
    controller = ConcurrencyController(ConcurrencyLimits(max_concurrent_transcriptions=1, allow_warmup_during_transcription=True))

    async def scenario():
        allowed_transcription, _ = await controller.try_start_transcription()
        allowed_warmup, _ = await controller.try_start_warmup()

        assert allowed_transcription
        assert allowed_warmup

        await controller.finish_transcription()
        await controller.finish_warmup()

    asyncio.run(scenario())
