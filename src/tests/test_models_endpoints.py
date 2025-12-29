from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import main
from fast_fast_whisper import engine as ff_engine
from fast_fast_whisper.model_catalog import MODEL_REGISTRY

REQUIRES_FASTER_WHISPER = pytest.mark.skipif(main.WhisperModel is None, reason='faster-whisper is not installed')

TEST_MODEL_NAME = 'large-v3'
TEST_MODEL = MODEL_REGISTRY[TEST_MODEL_NAME]


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
    response = client.post('/v1/models/download', json={'model': TEST_MODEL_NAME})
    assert response.status_code == 200

    payload = response.json()
    assert payload['model'] == TEST_MODEL_NAME
    assert payload['status'] in {'downloaded', 'already_present'}
    assert payload['elapsed'] > 0

    model_path = Path(payload['model_path'])
    models_root = Path('models').resolve()

    assert model_path.parent.resolve() == models_root


@REQUIRES_FASTER_WHISPER
def test_warmup_endpoint_initializes_model_cpu(client):
    download_response = client.post('/v1/models/download', json={'model': TEST_MODEL_NAME})
    assert download_response.status_code == 200

    response = client.post('/v1/models/warmup', json={'model': TEST_MODEL_NAME, 'device': 'cpu'})
    assert response.status_code == 200

    payload = response.json()
    assert payload['status'] == 'ready'
    assert payload['device'] == 'cpu'
    assert payload['model'] == TEST_MODEL_NAME
    assert payload['load_time'] > 0

    engine = main.WhisperEngine._instance
    assert engine is not None, 'warmup should cache the initialized engine'
    assert getattr(engine, '_model', None) is not None, 'engine must hold a loaded Whisper model'


@REQUIRES_FASTER_WHISPER
def test_warmup_endpoint_initializes_model_gpu(client):
    client.post('/v1/models/download', json={'model': TEST_MODEL_NAME})

    response = client.post('/v1/models/warmup', json={'model': TEST_MODEL_NAME, 'device': 'gpu'})
    assert response.status_code == 200

    payload = response.json()
    assert payload['status'] == 'ready'
    if payload['device'] != 'cuda':
        pytest.skip('CUDA unavailable in test environment')
    assert payload['model'] == TEST_MODEL_NAME
    assert payload['load_time'] > 0


def test_model_exists_endpoint_reports_missing_model(client, monkeypatch, tmp_path):
    import sys
    app_module = sys.modules['fast_fast_whisper.app']
    
    def fake_files_on_disk(name: str):
        return False, None

    monkeypatch.setattr(app_module, 'model_files_on_disk', fake_files_on_disk)

    response = client.get('/download/model/exists', params={'model': TEST_MODEL_NAME})
    assert response.status_code == 200

    payload = response.json()
    assert payload['model'] == TEST_MODEL_NAME
    assert payload['exists'] is False
    assert payload['model_path'] is None


def test_model_exists_endpoint_reports_present_model(client, monkeypatch, tmp_path):
    import sys
    app_module = sys.modules['fast_fast_whisper.app']
    
    model_root = tmp_path / 'models-cache' / TEST_MODEL.storage_dir
    snapshot_dir = model_root / 'snapshots' / 'fake'
    snapshot_dir.mkdir(parents=True)
    weights = snapshot_dir / 'weights.bin'
    weights.write_bytes(b'123')

    def fake_files_on_disk(name: str):
        return True, str(model_root)

    monkeypatch.setattr(app_module, 'model_files_on_disk', fake_files_on_disk)

    response = client.get('/download/model/exists', params={'model': TEST_MODEL_NAME})
    assert response.status_code == 200

    payload = response.json()
    assert payload['model'] == TEST_MODEL_NAME
    assert payload['exists'] is True
    assert payload['model_path'] == str(model_root)


def test_model_exists_endpoint_validates_model_name(client):
    response = client.get('/download/model/exists', params={'model': 'unknown-model'})
    assert response.status_code == 400


def test_list_models_endpoint(client):
    response = client.get('/v1/models')
    assert response.status_code == 200
    
    payload = response.json()
    assert payload['object'] == 'list'
    assert len(payload['data']) > 0
    assert payload['data'][0]['id'] == 'whisper-1'
