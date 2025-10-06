# fast-fast-whisper

Локальный OpenAI‑совместимый API для транскрибирования аудио на базе Whisper (faster-whisper) и FastAPI.

## Быстрый старт

```bash
git clone https://github.com/your-org/fast-fast-whisper.git
cd fast-fast-whisper
poetry install
poetry run uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Проверка:

```bash
curl http://localhost:8000/health
```

Транскрипция (JSON):

```bash
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

## License

MIT License — см. файл [LICENSE](LICENSE).
