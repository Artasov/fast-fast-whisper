from pathlib import Path


import pytest
from fastapi.testclient import TestClient

import main
from fast_fast_whisper import engine as ff_engine
from fast_fast_whisper.model_catalog import ModelRegistry

REQUIRES_FASTER_WHISPER = pytest.mark.skipif(main.WhisperModel is None, reason="faster-whisper is not installed")

TEST_MODEL = ModelRegistry.LargeV3


@pytest.fixture(autouse=True)
def reset_engine_cache():
    main.WhisperEngine.clear_cache()
    yield
    main.WhisperEngine.clear_cache()


@pytest.fixture
def client():
    return TestClient(main.app)


@REQUIRES_FASTER_WHISPER
def test_download_endpoint_creates_model_directory(client):
    response = client.post("/v1/models/download", json={"model": TEST_MODEL.api_name})
    assert response.status_code == 200

    payload = response.json()
    assert payload["model"] == TEST_MODEL.api_name
    assert payload["status"] in {"downloaded", "already_present"}
    assert payload["elapsed"] > 0

    model_path = Path(payload["model_path"])
    models_root = Path("models").resolve()

    assert model_path.parent.resolve() == models_root

    downloaded_dirs = [entry for entry in models_root.iterdir() if TEST_MODEL.matches_directory(entry)]
    assert downloaded_dirs, f"model directory '{TEST_MODEL.storage_dir}' not found in {models_root}"

    downloaded_files = [p for p in downloaded_dirs[0].rglob("*") if p.is_file()]
    assert downloaded_files, "downloaded model directory must contain files"


@REQUIRES_FASTER_WHISPER
def test_warmup_endpoint_initializes_model_cpu(client):
    download_response = client.post("/v1/models/download", json={"model": TEST_MODEL.api_name})
    assert download_response.status_code == 200

    response = client.post("/v1/models/warmup", json={"model": TEST_MODEL.api_name, "device": "cpu"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["device"] == "cpu"
    assert payload["model"] == TEST_MODEL.api_name
    assert payload["load_time"] > 0

    cache_key = f"{TEST_MODEL.api_name}_cpu"
    engine = main.WhisperEngine._instances.get(cache_key)
    assert engine is not None, "warmup should cache the initialized engine"
    assert getattr(engine, "_model", None) is not None, "engine must hold a loaded Whisper model"


@REQUIRES_FASTER_WHISPER
def test_warmup_endpoint_initializes_model_gpu(client):
    client.post("/v1/models/download", json={"model": TEST_MODEL.api_name})

    response = client.post("/v1/models/warmup", json={"model": TEST_MODEL.api_name, "device": "gpu"})
    assert response.status_code == 200

    payload = response.json()
    assert payload["status"] == "ready"
    if payload["device"] != "cuda":
        pytest.skip("CUDA unavailable in test environment")
    assert payload["model"] == TEST_MODEL.api_name
    assert payload["load_time"] > 0

    cache_key = f"{TEST_MODEL.api_name}_cuda"
    engine = main.WhisperEngine._instances.get(cache_key)
    assert engine is not None, "warmup should cache the initialized GPU engine"
    assert getattr(engine, "_model", None) is not None, "engine must hold a loaded Whisper model"


def test_model_exists_endpoint_reports_missing_model(client, monkeypatch, tmp_path):
    non_existing_path = tmp_path / "models-cache" / TEST_MODEL.storage_dir

    def fake_candidates(_: str):
        return [non_existing_path]

    monkeypatch.setattr(ff_engine, "model_storage_candidates", fake_candidates, raising=False)

    response = client.get("/download/model/exists", params={"model": TEST_MODEL.api_name})
    assert response.status_code == 200

    payload = response.json()
    assert payload["model"] == TEST_MODEL.api_name
    assert payload["exists"] is False
    assert payload["model_path"] is None


def test_model_exists_endpoint_reports_present_model(client, monkeypatch, tmp_path):
    model_root = tmp_path / "models-cache" / TEST_MODEL.storage_dir
    snapshot_dir = model_root / "snapshots" / "fake"
    snapshot_dir.mkdir(parents=True)
    weights = snapshot_dir / "weights.bin"
    weights.write_bytes(b"123")

    def fake_candidates(_: str):
        return [model_root]

    monkeypatch.setattr(ff_engine, "model_storage_candidates", fake_candidates, raising=False)

    response = client.get("/download/model/exists", params={"model": TEST_MODEL.api_name})
    assert response.status_code == 200

    payload = response.json()
    assert payload["model"] == TEST_MODEL.api_name
    assert payload["exists"] is True
    assert payload["model_path"] == str(model_root)


def test_model_exists_endpoint_validates_model_name(client):
    response = client.get("/download/model/exists", params={"model": "unknown-model"})
    assert response.status_code == 400
