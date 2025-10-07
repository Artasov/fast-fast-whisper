# AGENTS

This mini-project provides a local HTTP API for speech recognition (Whisper) based on FastAPI. The goal is compatibility with the original OpenAI Audio API format for requests and responses, so existing clients can work without changes.

## What to implement

- OpenAI Whisper endpoints:
  - `POST /v1/audio/transcriptions` — transcription.
  - `POST /v1/audio/translations` — translation to English.
  - `GET  /v1/models` — list of available models (emulation; one entry is sufficient).
- Support form fields like OpenAI: `file`, `model`, `prompt`, `response_format`, `temperature`, `language`.
- Support response formats: `json`, `text`, `srt`, `verbose_json`, `vtt`.
- Implement lazy model loading and reuse between requests.
- Configuration through environment variables.

## Technologies and dependencies

- Python 3.12+
- FastAPI + Uvicorn
- faster-whisper (inference)
- python-multipart (file upload)

## Environment variables

- `WHISPER_MODEL` — model name/path for faster-whisper (default: `base`).
- `WHISPER_DEVICE` — device: `auto`/`cpu`/`cuda` (default: `auto`).
- `WHISPER_COMPUTE_TYPE` — compute type: `auto`/`int8`/`float16` etc. (default: `auto`).
- `WHISPER_CPU_THREADS` — CPU threads (integer, default empty — auto).
- `OPENAI_MODEL_ID` — `id` field value in `/v1/models` (default: `whisper-1`).

## Example usage

1. Install dependencies (via `pip`):
   - `pip install -e .` or `pip install fastapi uvicorn[standard] python-multipart faster-whisper srt webvtt-py`
2. Start server:
   - `uvicorn main:app --host 0.0.0.0 --port 8000`

## Request examples

Transcription (JSON):

```
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "model=whisper-1" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

Translation to English (SRT):

```
curl -X POST http://localhost:8000/v1/audio/translations \
  -H "Content-Type: multipart/form-data" \
  -F "model=whisper-1" \
  -F "file=@sample.mp3" \
  -F "response_format=srt"
```

## Compatibility notes

- The `model` field is accepted for compatibility, but the local model from `WHISPER_MODEL` is actually used.
- For `verbose_json`, fields compatible in structure are returned, but values of some metrics (e.g., tokens, log probability) are not calculated and filled with zeros.

## Style and structure

- Minimal changes, readability and simple architecture.
- Everything in the project is concentrated in `main.py` for simplified startup.