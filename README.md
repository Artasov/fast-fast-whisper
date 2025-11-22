# fast-fast-whisper

A simple local OpenAI-compatible API for audio transcription based on Whisper (faster-whisper) and FastAPI.

## Quick Start

### Windows (PowerShell)

```sh
cd C:\ # or any other directory
git clone https://github.com/Artasov/fast-fast-whisper.git
cd fast-fast-whisper
.\start.bat
```

To stop the background server later, run `.\stop.bat` from the project directory.

### Linux / macOS

```sh
git clone https://github.com/Artasov/fast-fast-whisper.git
cd fast-fast-whisper
./start-unix.sh
```

Stop the server with `./stop-unix.sh` when you're done.

All helper scripts run the API on port `8868`. To override, set the environment variable `FAST_FAST_WHISPER_PORT` (or `PORT`) before launching the script. The start scripts detach the server from the console and stream logs to `fast-fast-whisper.log`.

## Manual install

Use **[Python 3.12.5](https://www.python.org/downloads/release/python-3125/)**

```sh
git clone https://github.com/Artasov/fast-fast-whisper.git
cd fast-fast-whisper
python -m venv venv
```

```sh
source ./venv/bin/activate # For Linux / macOS
```

```sh
.\venv\Scripts\Activate.ps1 # For Windows (PowerShell)
```

```sh
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8868 --reload
# or (preferred package path)
# PYTHONPATH=src uvicorn fast_fast_whisper.app:app --host 0.0.0.0 --port 8868 --reload
```

## Usage

Health check:

```sh
curl http://localhost:8868/health
```

Transcription (JSON):

```sh
curl -X POST http://localhost:8868/v1/audio/transcriptions \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

### Additional form fields

```sh
curl -X POST http://localhost:8868/v1/audio/transcriptions \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "prompt=Use Python and REST API terminology" \
  -F "response_format=json"
```

### Model management endpoints

To control model availability ahead of time the API exposes two helper endpoints.

#### Download a model

`POST /v1/models/download`

```jsonc
{
  "model": "tiny" // required, one of the Whisper model ids
}
```

- Downloads the specified model into the local `./models` directory (same storage layout as faster-whisper/HF cache).
- Response payload contains `status` (`downloaded` or `already_present`), resolved `model_path`, and elapsed time in seconds.

Example:

```sh
curl -X POST http://localhost:8868/v1/models/download \
  -H "Content-Type: application/json" \
  -d '{"model":"tiny"}'
```

#### Warm up a model

`POST /v1/models/warmup`

```jsonc
{
  "model": "tiny",     // required
  "device": "gpu"      // optional, accepts cpu/gpu/cuda/auto
}
```

- Lazily loads the model into memory on the requested device (GPU if available, otherwise CPU).
- Response payload reports `device`, `compute_type`, and `load_time`.
- Useful for health checks that need to ensure the model is ready before the first transcription request.

Example:

```sh
curl -X POST http://localhost:8868/v1/models/warmup \
  -H "Content-Type: application/json" \
  -d '{"model":"tiny","device":"gpu"}'
```

### Concurrency guard

By default the server processes only one transcription at a time to avoid GPU overload. Use the following environment variables to tune the behavior:

- `WHISPER_MAX_CONCURRENT_TRANSCRIPTIONS` — maximum number of parallel `/v1/audio/*` jobs (default: `1`, use `0` or a negative value to disable the cap).
- `WHISPER_ALLOW_WARMUP_DURING_TRANSCRIPTION` — set to `true` to allow `/v1/models/warmup` to run while a transcription is in progress (default: `false`, meaning warmup and transcription are mutually exclusive).

If a request arrives while the limits are exceeded, the API returns HTTP `429`/`409` with an explanatory message.

#### Check if a model is downloaded

`GET /download/model/exists?model=tiny`

- Returns `exists=true/false` without touching the model cache.
- `model_path` points to the directory that already contains model files when `exists=true`.
- Handy for orchestration scripts to decide whether they need to call `/v1/models/download`.

Example:

```sh
curl "http://localhost:8868/download/model/exists?model=tiny"
```

## License

MIT License — see [LICENSE](LICENSE) file.
