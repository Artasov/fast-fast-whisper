# fast-fast-whisper

Local OpenAI-compatible API for audio transcription/translation using Whisper (faster-whisper) built with FastAPI.

## Features

- OpenAI Audio API compatible endpoints:
  - `POST /v1/audio/transcriptions`
  - `POST /v1/audio/translations`
  - `GET /v1/models`
- Support for all Whisper models: `tiny`, `base`, `small`, `medium`, `large`, `large-v1`, `large-v2`, `large-v3`
- Support for `response_format`: `json`, `text`, `srt`, `verbose_json`, `vtt`.
- Lazy model creation and reuse.
- Automatic model loading to `models/` folder in project root.
- Configuration through environment variables.

## Installation

Poetry is recommended:

```bash
poetry install
```

Alternatively via `pip`:

```bash
pip install fastapi uvicorn[standard] python-multipart faster-whisper srt webvtt-py
```

## Running

Via Poetry (recommended):

```bash
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Or directly:

```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

Health check:

```bash
curl http://localhost:8000/healthz
```

## Environment Variables

- `WHISPER_DEVICE` (`auto`/`cpu`/`cuda`, default: `auto`)
- `WHISPER_COMPUTE_TYPE` (e.g., `auto`/`int8`/`float16`, default: `auto`)
- `WHISPER_CPU_THREADS` (integer, default: auto)

**Note:** The model is now specified in each request via the `model` parameter, not through environment variables.

## Examples

### List available models

```bash
curl http://localhost:8000/v1/models
```

### Transcription with different models

`tiny` `base` `small` `medium` `large-v2` `large-v3` `distil-large-v3` `Systran/faster-whisper-large-v3` `Systran/faster-whisper-medium`

**Fast model (tiny):**

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=tiny" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

**Balanced model (base):**

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

**Accurate model (large-v3):**

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=large-v3" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

### Translation to English

```bash
curl -X POST http://localhost:8000/v1/audio/translations \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "response_format=srt"
```

### Different response formats

**JSON (default):**

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=large-v3" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=tyne" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

**Text:**

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "response_format=text"
```

**SRT subtitles:**

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "response_format=srt"
```

**VTT subtitles:**

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "response_format=vtt"
```

**Detailed JSON with segments:**

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "response_format=verbose_json"
```

## Compatibility

- Request and response format is fully compatible with OpenAI Audio API
- All Whisper models are supported: `tiny`, `base`, `small`, `medium`, `large`, `large-v1`, `large-v2`, `large-v3`
- Models are automatically loaded to `models/` folder on first use
- In `verbose_json` token/log-likelihood metrics are filled with zeros for structure compatibility
- Both short names (`tiny`, `base`) and full names (`whisper-tiny`, `whisper-base`) are supported

## License

MIT License - see [LICENSE](LICENSE) file for details.

## Development

The project uses Poetry for dependency management:

```bash
# Install dependencies
poetry install

# Run in development mode
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Check imports
poetry run python scripts/check_import.py
```

## Contributing

1. Fork the repository
2. Create a branch for your feature (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add some amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Support

If you have questions or issues, create an issue in the repository.
