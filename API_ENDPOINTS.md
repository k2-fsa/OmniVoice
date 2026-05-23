# OmniVoice API — Эндпоинты

## Запуск сервера

```bash
pip install omnivoice[api]

omnivoice-api \
  --model k2-fsa/OmniVoice \
  --host 0.0.0.0 \
  --port 8000 \
  --library-dir /srv/voices \
  --api-key mysecretkey

# Swagger UI (интерактивная документация)
# http://localhost:8000/docs
```

---

## Аутентификация

Если сервер запущен с `--api-key`, все запросы должны содержать заголовок:

```
X-Api-Key: mysecretkey
```

Без ключа — возвращает `401 Unauthorized`.

---

## Система

### GET /health
Проверка работоспособности сервера.

**Ответ:**
```json
{
  "status": "ok",
  "sample_rate": 24000,
  "device": "cuda:0",
  "voices_count": 5
}
```

---

### GET /info
Подробная информация о модели и библиотеке голосов.

**Ответ:**
```json
{
  "model": "k2-fsa/OmniVoice",
  "sample_rate": 24000,
  "device": "cuda:0",
  "supported_languages": 617,
  "library_dir": "/srv/voices",
  "voices": ["Alice", "Bob", "Narrator RU"]
}
```

---

## Голоса

### GET /voices
Список всех сохранённых голосов с метаданными.

**Заголовки:** `X-Api-Key`

**Ответ:**
```json
{
  "voices": [
    {
      "name": "Alice",
      "safe_name": "Alice",
      "ref_text": "Hello, my name is Alice.",
      "ref_rms": 0.082
    },
    {
      "name": "Narrator RU",
      "safe_name": "Narrator_RU",
      "ref_text": "Добрый день, дорогие слушатели.",
      "ref_rms": 0.091
    }
  ],
  "count": 2
}
```

---

### GET /voices/{name}
Метаданные одного голоса по имени.

**Параметры пути:** `name` — имя голоса (точное совпадение)

**Заголовки:** `X-Api-Key`

**Ответ `200`:**
```json
{
  "name": "Alice",
  "safe_name": "Alice",
  "ref_text": "Hello, my name is Alice.",
  "ref_rms": 0.082
}
```

**Ответ `404`:**
```json
{ "detail": "Voice 'Alice' not found." }
```

---

### POST /voices/clone
Клонировать голос из загруженного аудиофайла и сохранить в библиотеку.

**Заголовки:** `X-Api-Key`, `Content-Type: multipart/form-data`

**Тело (form-data):**

| Поле | Тип | Обязательно | Описание |
|------|-----|:-----------:|----------|
| `name` | string | ✅ | Имя голоса (уникальное) |
| `audio` | file | ✅ | Аудиофайл WAV / MP3 / FLAC (рекомендуется 3–10 сек) |
| `ref_text` | string | — | Транскрипция аудио. Если не указана — авто-транскрипция через Whisper |
| `preprocess` | bool | — | Обрезать тишину и нормализовать громкость (по умолч. `true`) |

**Ответ `201`:**
```json
{
  "name": "Alice",
  "safe_name": "Alice",
  "ref_text": "Hello, my name is Alice.",
  "ref_rms": 0.082
}
```

**Ответ `409`** — голос с таким именем уже существует.

**Пример:**
```bash
curl -X POST http://localhost:8000/voices/clone \
  -H "X-Api-Key: mysecretkey" \
  -F "name=Alice" \
  -F "audio=@speaker.wav" \
  -F "ref_text=Hello my name is Alice" \
  -F "preprocess=true"
```

---

### DELETE /voices/{name}
Удалить голос из библиотеки.

**Заголовки:** `X-Api-Key`

**Ответ `200`:**
```json
{ "deleted": true, "name": "Alice" }
```

**Ответ `404`:**
```json
{ "detail": "Voice 'Alice' not found." }
```

**Пример:**
```bash
curl -X DELETE http://localhost:8000/voices/Alice \
  -H "X-Api-Key: mysecretkey"
```

---

### PATCH /voices/{name}
Переименовать голос.

**Заголовки:** `X-Api-Key`, `Content-Type: application/json`

**Тело:**
```json
{ "new_name": "Alice UK" }
```

**Ответ `200`** — метаданные с новым именем.

**Ответ `404`** — голос не найден.

**Ответ `409`** — голос с новым именем уже существует.

**Пример:**
```bash
curl -X PATCH http://localhost:8000/voices/Alice \
  -H "X-Api-Key: mysecretkey" \
  -H "Content-Type: application/json" \
  -d '{"new_name": "Alice UK"}'
```

---

## Генерация

Общие параметры тела запроса для всех эндпоинтов генерации:

| Поле | Тип | По умолч. | Описание |
|------|-----|:---------:|----------|
| `text` | string | — | Текст для синтеза (обязательно, макс. 10 000 символов) |
| `voice` | string | `null` | Имя голоса из библиотеки. `null` — без голоса (авто-режим) |
| `language` | string | `null` | Язык: `"Russian"`, `"English"`, `"zh"` и т.д. `null` — авто-определение |
| `instruct` | string | `null` | Инструкция для дизайна голоса: `"male, british accent"` |
| `speed` | float | `null` | Скорость речи: `0.3`–`3.0`. `null` — авто |
| `duration` | float | `null` | Фиксированная длина в секундах (переопределяет `speed`) |
| `num_step` | int | `32` | Шаги диффузии: `4`–`64`. Меньше = быстрее, больше = качественнее |
| `guidance_scale` | float | `2.0` | Сила CFG guidance: `0.0`–`4.0` |
| `denoise` | bool | `true` | Шумоподавление |
| `postprocess_output` | bool | `true` | Удалять длинные паузы из результата |
| `audio_chunk_duration` | float | `15.0` | Секунд на чанк при длинных текстах |
| `audio_chunk_threshold` | float | `30.0` | Порог (сек) включения чанкинга |

---

### POST /generate
Синхронная генерация — возвращает полный WAV файл.

Подходит для коротких текстов или когда нужен файл целиком перед воспроизведением.

**Заголовки:** `X-Api-Key`, `Content-Type: application/json`

**Ответ `200`:** бинарный WAV файл (`audio/wav`)

**Заголовки ответа:**
- `X-Sample-Rate` — частота дискретизации (например `24000`)
- `X-Audio-Duration` — длина аудио в секундах (например `3.520`)
- `Content-Disposition` — `attachment; filename="output.wav"`

**Пример:**
```bash
curl -X POST http://localhost:8000/generate \
  -H "X-Api-Key: mysecretkey" \
  -H "Content-Type: application/json" \
  -d '{
    "text": "Привет! Это тест голосового синтеза.",
    "voice": "Alice",
    "language": "Russian",
    "speed": 1.0,
    "num_step": 32
  }' \
  --output output.wav
```

**Python:**
```python
import requests

resp = requests.post(
    "http://localhost:8000/generate",
    headers={"X-Api-Key": "mysecretkey"},
    json={
        "text": "Hello world",
        "voice": "Alice",
        "language": "English",
    },
)
with open("output.wav", "wb") as f:
    f.write(resp.content)
```

---

### POST /generate/stream
Потоковая генерация — возвращает аудио по мере готовности чанками.

Используйте для длинных текстов или когда нужно начать воспроизведение не дожидаясь окончания генерации.

**Заголовки:** `X-Api-Key`, `Content-Type: application/json`

**Ответ `200`:** `application/octet-stream` с `Transfer-Encoding: chunked`

**Заголовки ответа:**
- `X-Sample-Rate` — частота дискретизации (например `24000`)
- `X-Voice` — имя использованного голоса

**Формат бинарного потока:**

Поток состоит из фреймов. Каждый фрейм:
```
[4 байта uint32-LE: длина PCM данных] [N байт: PCM int16-LE моно]
```
Последний фрейм — маркер конца потока:
```
[4 байта: 0x00000000]
```

Каждый фрейм содержит **только новый** аудио-сегмент с момента последнего фрейма.
Для получения полного аудио — конкатенируйте все PCM-данные всех фреймов.

**Python-клиент:**
```python
import struct, requests, numpy as np, soundfile as sf

resp = requests.post(
    "http://localhost:8000/generate/stream",
    headers={"X-Api-Key": "mysecretkey"},
    json={"text": "Длинный текст для стриминга...", "voice": "Alice"},
    stream=True,
)

sr = int(resp.headers["X-Sample-Rate"])
pcm_parts = []
buf = b""

for raw_chunk in resp.iter_content(chunk_size=8192):
    buf += raw_chunk
    while len(buf) >= 4:
        frame_len = struct.unpack("<I", buf[:4])[0]
        if frame_len == 0:             # конец потока
            break
        if len(buf) < 4 + frame_len:  # ждём ещё данных
            break
        pcm = np.frombuffer(buf[4:4 + frame_len], dtype=np.int16)
        pcm_parts.append(pcm)
        buf = buf[4 + frame_len:]
        # Можно воспроизводить pcm прямо здесь (sounddevice.play)

audio = np.concatenate(pcm_parts).astype(np.float32) / 32767.0
sf.write("output.wav", audio, sr)
```

---

### WS /ws/generate
WebSocket стриминг — двунаправленный канал для real-time воспроизведения в браузере или приложении.

**URL:** `ws://localhost:8000/ws/generate`

#### Клиент → Сервер (один раз, JSON):
```json
{
  "api_key": "mysecretkey",
  "text": "Текст для синтеза",
  "voice": "Alice",
  "language": "Russian",
  "speed": 1.0,
  "num_step": 32,
  "guidance_scale": 2.0
}
```
Все поля из таблицы параметров генерации поддерживаются. `api_key` — только для WebSocket (заголовок здесь недоступен).

#### Сервер → Клиент (поток сообщений):

**Бинарные фреймы** — сырой PCM int16-LE моно, только новый сегмент:
```
<binary: PCM int16-LE>
```

**Текстовые фреймы** (JSON) — между бинарными чанками:
```json
{ "type": "status", "message": "Generating... chunk 2/5", "chunk": 2, "total": 5 }
```

Финальное сообщение:
```json
{ "type": "done", "duration": 12.48, "sample_rate": 24000 }
```

Сообщение об ошибке:
```json
{ "type": "error", "message": "Voice 'Alice' not found in library." }
```

**JavaScript-клиент (браузер):**
```javascript
const ws = new WebSocket("ws://localhost:8000/ws/generate");
const audioContext = new AudioContext({ sampleRate: 24000 });
const pcmChunks = [];

ws.onopen = () => {
    ws.send(JSON.stringify({
        api_key: "mysecretkey",
        text: "Hello! This is real-time streaming TTS.",
        voice: "Alice",
        language: "English",
    }));
};

ws.onmessage = async (e) => {
    if (e.data instanceof Blob) {
        // Новый аудио-чанк
        const buf = await e.data.arrayBuffer();
        const pcm = new Int16Array(buf);
        const float32 = new Float32Array(pcm.length);
        for (let i = 0; i < pcm.length; i++) float32[i] = pcm[i] / 32767;

        const audioBuf = audioContext.createBuffer(1, float32.length, 24000);
        audioBuf.getChannelData(0).set(float32);

        const source = audioContext.createBufferSource();
        source.buffer = audioBuf;
        source.connect(audioContext.destination);
        source.start();

        pcmChunks.push(pcm);
    } else {
        const msg = JSON.parse(e.data);
        if (msg.type === "status") {
            console.log(msg.message);  // "Generating... chunk 1/3"
        } else if (msg.type === "done") {
            console.log(`Done! Duration: ${msg.duration}s`);
            ws.close();
        } else if (msg.type === "error") {
            console.error("Error:", msg.message);
            ws.close();
        }
    }
};
```

**Python asyncio-клиент:**
```python
import asyncio, json
import numpy as np
import websockets

async def stream_tts():
    async with websockets.connect("ws://localhost:8000/ws/generate") as ws:
        await ws.send(json.dumps({
            "api_key": "mysecretkey",
            "text": "Длинный текст для WebSocket стриминга...",
            "voice": "Alice",
            "language": "Russian",
        }))

        pcm_parts = []
        async for message in ws:
            if isinstance(message, bytes):
                pcm = np.frombuffer(message, dtype=np.int16)
                pcm_parts.append(pcm)
                print(f"  Received {len(pcm)/24000:.2f}s of audio")
            else:
                msg = json.loads(message)
                print(f"  Status: {msg}")
                if msg["type"] == "done":
                    break

        audio = np.concatenate(pcm_parts).astype(np.float32) / 32767.0
        import soundfile as sf
        sf.write("output.wav", audio, 24000)
        print("Saved output.wav")

asyncio.run(stream_tts())
```

---

## Коды ответов

| Код | Значение |
|-----|----------|
| `200` | Успех |
| `201` | Голос создан |
| `401` | Неверный или отсутствующий API ключ |
| `404` | Голос не найден |
| `409` | Конфликт — голос с таким именем уже существует |
| `422` | Ошибка валидации параметров |
| `500` | Ошибка сервера (генерация или клонирование) |
| `503` | Модель ещё не загружена |

---

## Docker

```dockerfile
FROM python:3.11-slim
RUN pip install omnivoice[api]
ENV OMNIVOICE_MODEL=k2-fsa/OmniVoice
ENV OMNIVOICE_LIBRARY_DIR=/data/voices
ENV OMNIVOICE_API_KEY=changeme
EXPOSE 8000
CMD ["omnivoice-api", "--host", "0.0.0.0", "--port", "8000"]
```

```bash
docker build -t omnivoice-api .
docker run -p 8000:8000 \
  -v /srv/voices:/data/voices \
  --gpus all \
  omnivoice-api
```
