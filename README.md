# fast-fast-whisper

A simple local OpenAI-compatible API for audio transcription based on Whisper (faster-whisper) and FastAPI.

## Quick Start

### In `powershell`:

```sh
git clone https://github.com/Artasov/fast-fast-whisper.git
cd fast-fast-whisper
.\start.bat

```

## Manual install

Use **[Python 3.12.5](https://www.python.org/downloads/release/python-3125/)**

```sh
git clone https://github.com/Artasov/fast-fast-whisper.git
cd fast-fast-whisper
python -m venv venv
```

```sh
source ./venv/Scripts/activate # For Linux
```

```sh
./venv/Scripts/activate # For Windows
```

```sh
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

## Usage

Health check:

```sh
curl http://localhost:8000/health
```

Transcription (JSON):

```sh
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "model=base" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

## License

MIT License — see [LICENSE](LICENSE) file.
