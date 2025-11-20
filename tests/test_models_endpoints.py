from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from model_catalog import ModelRegistry

pytestmark = pytest.mark.skipif(main.WhisperModel is None, reason="faster-whisper is not installed")

TEST_MODEL = ModelRegistry.LargeV3


@pytest.fixture(autouse=True)
def reset_engine_cache():
    main.WhisperEngine.clear_cache()
    yield
    main.WhisperEngine.clear_cache()


@pytest.fixture
def client():
    return TestClient(main.app)


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
