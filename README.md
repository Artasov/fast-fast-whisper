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

### Дополнительные поля формы

Эндпоинт `/v1/audio/transcriptions` полностью повторяет форму OpenAI и принимает поля `file`, `model`, `prompt`, `response_format`, `temperature`, `language`, `device`. Поле `prompt` передаётся в Whisper как `initial_prompt`, поэтому с его помощью можно подсказать модели специфичные термины или формат.

```sh
curl -X POST http://localhost:8868/v1/audio/transcriptions \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "prompt=Используй термины Python и REST API" \
  -F "response_format=json"
```

## License

MIT License — see [LICENSE](LICENSE) file.
