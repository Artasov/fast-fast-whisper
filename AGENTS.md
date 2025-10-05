# AGENTS

Этот мини‑проект поднимает локальный HTTP API для распознавания речи (Whisper) на базе FastAPI. Цель — совместимость по формату запросов и ответов с оригинальным OpenAI Audio API, чтобы существующие клиенты могли работать без изменений.

## Что реализовать

- Эндпоинты OpenAI Whisper:
  - `POST /v1/audio/transcriptions` — транскрибирование.
  - `POST /v1/audio/translations` — перевод в английский.
  - `GET  /v1/models` — список доступных моделей (эмуляция; достаточно одной записи).
- Поддержать форм‑поля, как у OpenAI: `file`, `model`, `prompt`, `response_format`, `temperature`, `language`.
- Поддержать форматы ответов: `json`, `text`, `srt`, `verbose_json`, `vtt`.
- Реализовать ленивую загрузку модели и повторное использование между запросами.
- Конфигурацию через переменные окружения.

## Технологии и зависимости

- Python 3.12+
- FastAPI + Uvicorn
- faster-whisper (инференс)
- python-multipart (загрузка файлов)

## Переменные окружения

- `WHISPER_MODEL` — имя/путь модели для faster-whisper (по умолчанию: `base`).
- `WHISPER_DEVICE` — устройство: `auto`/`cpu`/`cuda` (по умолчанию: `auto`).
- `WHISPER_COMPUTE_TYPE` — тип вычислений: `auto`/`int8`/`float16` и т.п. (по умолчанию: `auto`).
- `WHISPER_CPU_THREADS` — потоки CPU (целое, по умолчанию пусто — авто).
- `OPENAI_MODEL_ID` — значение поля `id` в `/v1/models` (по умолчанию: `whisper-1`).

## Пример запуска

1. Установите зависимости (через `pip`):
   - `pip install -e .` либо `pip install fastapi uvicorn[standard] python-multipart faster-whisper srt webvtt-py`
2. Запустите сервер:
   - `uvicorn main:app --host 0.0.0.0 --port 8000`

## Примеры запросов

Транскрипция (JSON):

```
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -H "Content-Type: multipart/form-data" \
  -F "model=whisper-1" \
  -F "file=@sample.mp3" \
  -F "response_format=json"
```

Перевод в английский (SRT):

```
curl -X POST http://localhost:8000/v1/audio/translations \
  -H "Content-Type: multipart/form-data" \
  -F "model=whisper-1" \
  -F "file=@sample.mp3" \
  -F "response_format=srt"
```

## Замечания по совместимости

- Поле `model` принимается для совместимости, но фактически используется локальная модель из `WHISPER_MODEL`.
- Для `verbose_json` возвращаются поля, совместимые по структуре, но значения некоторых метрик (например, токены, лог‑правдоподобие) не рассчитываются и заполняются нулями.

## Стиль и структура

- Минимальные изменения, читаемость и простая архитектура.
- В проекте всё сосредоточено в `main.py` для упрощения запуска.
