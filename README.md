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

The `/v1/audio/transcriptions` endpoint mirrors the OpenAI API and accepts the fields `file`, `model`, `prompt`, `response_format`, `temperature`, `language`, and `device`. The `prompt` value is passed as Whisper's `initial_prompt`, so you can guide the model toward domain-specific vocabulary or formats.

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

## License

MIT License — see [LICENSE](LICENSE) file.
