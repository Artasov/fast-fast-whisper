import io
import threading
import time

from fast_fast_whisper import engine as ff_engine


class _DummyInfo:
    def __init__(self, language: str = "en", duration: float = 0.0) -> None:
        self.language = language
        self.duration = duration


class _DummySegment:
    def __init__(self, text: str) -> None:
        self.text = text
        self.start = 0.0
        self.end = 0.0


def test_get_returns_single_instance_under_concurrency(monkeypatch):
    init_counter = 0
    counter_lock = threading.Lock()

    class DummyModel:
        def __init__(self, **_: object) -> None:
            nonlocal init_counter
            with counter_lock:
                init_counter += 1

        def transcribe(self, **_: object):
            return [], _DummyInfo()

    monkeypatch.setattr(ff_engine, "WhisperModel", DummyModel)
    ff_engine.WhisperEngine.clear_cache()

    def worker():
        ff_engine.WhisperEngine.get("demo", device_override="cpu")

    threads = [threading.Thread(target=worker) for _ in range(5)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=1)

    assert init_counter == 1
    assert len(ff_engine.WhisperEngine._instances) == 1


def test_transcribe_calls_are_serialized(monkeypatch):
    entered = threading.Event()
    release_first = threading.Event()
    call_order: list[str] = []

    monkeypatch.setenv("WHISPER_DEVICE", "cpu")

    class DummyModel:
        def __init__(self, **_: object) -> None:
            pass

        def transcribe(self, **_: object):
            call_order.append("call")
            entered.set()
            release_first.wait(timeout=1)
            return [_DummySegment("hi")], _DummyInfo()

    monkeypatch.setattr(ff_engine, "WhisperModel", DummyModel)
    ff_engine.WhisperEngine.clear_cache()

    engine = ff_engine.WhisperEngine("demo", device_override="cpu")
    results: list[str] = []

    def run_transcribe():
        result = engine.transcribe(io.BytesIO(b"123"))
        results.append(result["text"])

    first = threading.Thread(target=run_transcribe)
    second = threading.Thread(target=run_transcribe)

    first.start()
    assert entered.wait(timeout=1)

    second.start()
    time.sleep(0.1)

    assert call_order == ["call"]

    release_first.set()
    first.join(timeout=1)
    second.join(timeout=1)

    assert call_order == ["call", "call"]
    assert results == ["hi", "hi"]


def test_auto_and_cuda_share_one_cached_engine(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "auto")

    class DummyModel:
        def __init__(self, **_: object) -> None:
            pass

        def transcribe(self, **_: object):
            return [_DummySegment("ok")], _DummyInfo()

    def fake_autodetect(self):
        # Force autodetect to pick CUDA without touching real hardware.
        self.device = "cuda"

    monkeypatch.setattr(ff_engine, "WhisperModel", DummyModel)
    monkeypatch.setattr(ff_engine.WhisperEngine, "_autodetect_device", fake_autodetect, raising=False)
    ff_engine.WhisperEngine.clear_cache()

    auto_engine = ff_engine.WhisperEngine.get("demo")
    cuda_engine = ff_engine.WhisperEngine.get("demo", device_override="cuda")
    gpu_engine = ff_engine.WhisperEngine.get("demo", device_override="gpu")

    assert auto_engine is cuda_engine is gpu_engine
