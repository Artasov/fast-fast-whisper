from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

import main


class DummyWhisperModel:
    def __init__(self, model_size_or_path, device="cpu", compute_type="float32", download_root=None, **kwargs):
        self.model_size_or_path = model_size_or_path
        self.device = device
        self.compute_type = compute_type
        self.download_root = download_root
        self.kwargs = kwargs

        if download_root is not None:
            root = Path(download_root)
            model_dir = root / self.model_size_or_path
            model_dir.mkdir(parents=True, exist_ok=True)

    def transcribe(self, **kwargs):
        segment = SimpleNamespace(text="stub", start=0.0, end=0.5)
        info = SimpleNamespace(duration=0.5, language=kwargs.get("language", "en"))
        return [segment], info


@pytest.fixture(autouse=True)
def stub_whisper(monkeypatch, tmp_path):
    monkeypatch.setattr(main, "WhisperModel", DummyWhisperModel)
    monkeypatch.chdir(tmp_path)
    main.WhisperEngine.clear_cache()
    yield
    main.WhisperEngine.clear_cache()


@pytest.fixture
def client():
    return TestClient(main.app)


def test_download_endpoint_creates_model_directory(client):
    response = client.post("/v1/models/download", json={"model": "tiny"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["model"] == "tiny"
    assert payload["status"] in {"downloaded", "already_present"}
    model_path = Path(payload["model_path"])

    assert model_path.exists()
    assert model_path.is_dir()


def _patch_engine(monkeypatch, expected_device, reported_device):
    engine = SimpleNamespace(device=reported_device, compute_type="float32")

    def fake_get(cls, model_name, device_override=None):
        assert model_name == "tiny"
        assert device_override == expected_device
        return engine

    monkeypatch.setattr(main.WhisperEngine, "get", classmethod(fake_get))
    return engine


def test_warmup_endpoint_initializes_model_cpu(client, monkeypatch):
    client.post("/v1/models/download", json={"model": "tiny"})
    _patch_engine(monkeypatch, expected_device="cpu", reported_device="cpu")

    response = client.post("/v1/models/warmup", json={"model": "tiny", "device": "cpu"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["device"] == "cpu"
    assert payload["model"] == "tiny"
    assert payload["load_time"] >= 0


def test_warmup_endpoint_initializes_model_gpu(client, monkeypatch):
    client.post("/v1/models/download", json={"model": "tiny"})
    _patch_engine(monkeypatch, expected_device="cuda", reported_device="cuda")

    response = client.post("/v1/models/warmup", json={"model": "tiny", "device": "gpu"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["device"] == "cuda"
    assert payload["model"] == "tiny"
    assert payload["load_time"] >= 0
