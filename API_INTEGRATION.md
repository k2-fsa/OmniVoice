# OmniVoice — Полная инструкция по API интеграции

## 0. Установка и импорты

```python
# Установка (если ещё не сделано)
# pip install omnivoice

import numpy as np
import soundfile as sf
import torch

from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.voice_library import VoiceLibrary
```

---

## 1. Загрузка модели (один раз при старте)

```python
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice",       # или локальный путь
    device_map="cuda",         # "cuda" / "cpu" / "mps" (Apple)
    dtype=torch.float16,       # float16 для GPU, float32 для CPU
    load_asr=True,             # Whisper для авто-транскрипции референса
    asr_model_name="openai/whisper-large-v3-turbo",
)

SAMPLE_RATE = model.sampling_rate  # 24000 Гц
```

---

## 2. VoiceLibrary — управление голосами

```python
# Хранилище голосов (по умолчанию ~/.omnivoice/voices/)
lib = VoiceLibrary()

# Или своя папка:
lib = VoiceLibrary("/srv/myapp/voices")
```

### 2.1 Получить список доступных голосов

```python
# Просто список имён
names = lib.names()
print(names)
# ['Alice', 'Bob', 'Narrator RU']

# Полные метаданные (имя, ref_text, ref_rms)
voices = lib.list_voices()
for v in voices:
    print(f"  {v['name']:20s}  ref: {v['ref_text'][:50]}")
# Alice                 ref: Hello, my name is Alice and I work at...
# Bob                   ref: Good morning everyone, let me start...

# Проверить существование голоса
if lib.exists("Alice"):
    print("есть")
```

### 2.2 Добавить новый голос (клонировать и сохранить)

```python
# Вариант А — из файла (авто-транскрипция через Whisper)
prompt = model.create_voice_clone_prompt("speaker.wav")
lib.save("Alice", prompt)

# Вариант Б — из файла с готовым текстом (быстрее, не нужен Whisper)
prompt = model.create_voice_clone_prompt(
    ref_audio="speaker.wav",
    ref_text="Hello, my name is Alice.",
    preprocess_prompt=True,   # обрезка тишины, нормализация (рекомендуется)
)
lib.save("Alice", prompt)

# Вариант В — из numpy-массива в памяти
import soundfile as sf
waveform, sr = sf.read("speaker.wav", always_2d=True)   # (channels, T)
waveform = waveform.T                                    # -> (1, T)
prompt = model.create_voice_clone_prompt(
    ref_audio=(torch.from_numpy(waveform).float(), sr)
)
lib.save("Alice", prompt)
```

### 2.3 Загрузить голос для использования

```python
# Загрузка по имени — мгновенно, без аудиофайла
prompt = lib.load("Alice")
# prompt.ref_audio_tokens  — тензор (8, T) на CPU
# prompt.ref_text           — транскрипция референса
# prompt.ref_rms            — громкость для нормализации
```

### 2.4 Переименовать / удалить голос

```python
lib.rename("Alice", "Alice (UK)")   # переименовать
lib.delete("Alice (UK)")            # удалить → True/False
```

---

## 3. Потоковая генерация (streaming)

`generate_streaming` — это **Python-генератор**. На каждой итерации отдаёт `(audio_array, status)`:

- **Короткий текст** (< ~30 сек аудио): один `yield` по завершении
- **Длинный текст**: несколько `yield` — после каждого чанка, **накопительно**

### 3.1 Базовый пример

```python
prompt = lib.load("Alice")

for audio, status in model.generate_streaming(
    text="Привет! Это потоковая генерация речи с выбранным голосом.",
    language="Russian",            # или None для авто-определения
    voice_clone_prompt=prompt,
):
    print(f"[{status}]  audio shape: {audio.shape}")
    # Последний yield — финальный результат
    final_audio = audio

# Сохранить результат
sf.write("output.wav", final_audio, SAMPLE_RATE)
```

### 3.2 Воспроизведение в реальном времени (по чанкам)

```python
import sounddevice as sd   # pip install sounddevice

prompt = lib.load("Alice")
prev_len = 0

for audio, status in model.generate_streaming(
    text="Длинный текст, который займёт несколько секунд генерации...",
    voice_clone_prompt=prompt,
    generation_config=OmniVoiceGenerationConfig(
        num_step=32,
        audio_chunk_duration=15.0,   # секунд на чанк
        audio_chunk_threshold=30.0,  # порог включения чанкинга
    ),
):
    # Воспроизводим только НОВУЮ часть аудио
    new_part = audio[prev_len:]
    if len(new_part) > 0:
        sd.play(new_part, SAMPLE_RATE, blocking=False)
    prev_len = len(audio)
    print(f"Status: {status}")
```

### 3.3 Стриминг в FastAPI (SSE — Server-Sent Events)

```python
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
import io, struct, json

app = FastAPI()

@app.get("/tts/stream")
async def tts_stream(
    text: str,
    voice: str = "Alice",
    language: str = None,
    speed: float = 1.0,
):
    """Стриминговый TTS — отдаёт WAV-чанки по мере генерации."""

    def _generate():
        try:
            prompt = lib.load(voice)
        except KeyError:
            yield b""   # голос не найден
            return

        for audio, status in model.generate_streaming(
            text=text,
            language=language or None,
            voice_clone_prompt=prompt,
            speed=speed,
        ):
            # Конвертируем float32 → int16 WAV-байты
            pcm = (audio * 32767).astype(np.int16).tobytes()
            # Шлём: 4 байта длины + PCM данные
            yield struct.pack("<I", len(pcm)) + pcm

    return StreamingResponse(
        _generate(),
        media_type="application/octet-stream",
        headers={"X-Sample-Rate": str(SAMPLE_RATE)},
    )


@app.get("/voices")
def list_voices():
    """Список доступных голосов."""
    return {
        "voices": lib.list_voices(),
        "count": len(lib.names()),
    }


@app.post("/voices/{name}")
async def add_voice(name: str, ref_audio_path: str, ref_text: str = None):
    """Добавить голос в библиотеку."""
    prompt = model.create_voice_clone_prompt(
        ref_audio=ref_audio_path,
        ref_text=ref_text or None,
    )
    lib.save(name, prompt)
    return {"status": "saved", "name": name}


@app.delete("/voices/{name}")
def delete_voice(name: str):
    ok = lib.delete(name)
    return {"deleted": ok, "name": name}
```

### 3.4 WebSocket — push аудио браузеру

```python
from fastapi import WebSocket
import asyncio

@app.websocket("/tts/ws")
async def tts_websocket(ws: WebSocket):
    await ws.accept()

    data = await ws.receive_json()
    text   = data["text"]
    voice  = data.get("voice", "Alice")
    lang   = data.get("language", None)

    try:
        prompt = lib.load(voice)
    except KeyError:
        await ws.send_json({"error": f"Voice '{voice}' not found"})
        await ws.close()
        return

    loop = asyncio.get_event_loop()

    def _run():
        for audio, status in model.generate_streaming(
            text=text,
            language=lang,
            voice_clone_prompt=prompt,
        ):
            pcm = (audio * 32767).astype(np.int16).tobytes()
            is_last = status == "Done."
            # Блокирующий send в синхронном генераторе — шедулируем через loop
            asyncio.run_coroutine_threadsafe(
                ws.send_bytes(pcm), loop
            ).result()
            if is_last:
                asyncio.run_coroutine_threadsafe(
                    ws.send_json({"status": "done"}), loop
                ).result()

    await loop.run_in_executor(None, _run)
```

---

## 4. Параметры генерации

```python
cfg = OmniVoiceGenerationConfig(
    num_step=32,              # шаги диффузии: 4–64 (меньше=быстрее, больше=качественнее)
    guidance_scale=2.0,       # сила CFG guidance (0.0–4.0)
    speed=1.0,                # скорость речи (0.5–1.5)
    denoise=True,             # шумоподавление (рекомендуется)
    postprocess_output=True,  # удалять длинные паузы из результата
    preprocess_prompt=True,   # предобработка референса
    audio_chunk_duration=15.0,    # секунд на чанк при длинных текстах
    audio_chunk_threshold=30.0,   # порог включения чанкинга
)

for audio, status in model.generate_streaming(
    text="...",
    voice_clone_prompt=prompt,
    language="Russian",       # "English", "Chinese", "German", None...
    speed=1.1,                # или через generation_config
    duration=5.0,             # зафиксировать длину в секундах (переопределяет speed)
    generation_config=cfg,
):
    ...
```

---

## 5. Полный рабочий скрипт

```python
"""example_streaming.py — полный пример с нуля."""

import numpy as np
import soundfile as sf
import torch
from omnivoice import OmniVoice, OmniVoiceGenerationConfig
from omnivoice.utils.voice_library import VoiceLibrary

# 1. Загрузка
model = OmniVoice.from_pretrained(
    "k2-fsa/OmniVoice", device_map="cuda", dtype=torch.float16, load_asr=True
)
lib = VoiceLibrary()

# 2. Добавить голос (один раз)
if not lib.exists("Demo Voice"):
    prompt = model.create_voice_clone_prompt("reference.wav")
    lib.save("Demo Voice", prompt)
    print("Voice saved!")

# 3. Список доступных голосов
print("Available voices:", lib.names())

# 4. Потоковая генерация
prompt = lib.load("Demo Voice")

final_audio = None
for audio, status in model.generate_streaming(
    text=(
        "Это демонстрация потоковой генерации речи. "
        "Аудио обновляется по мере готовности каждого фрагмента текста."
    ),
    language="Russian",
    voice_clone_prompt=prompt,
    generation_config=OmniVoiceGenerationConfig(num_step=32, speed=1.0),
):
    print(f"  [{status}] got {len(audio)/24000:.2f}s of audio")
    final_audio = audio

sf.write("output.wav", final_audio, model.sampling_rate)
print("Saved output.wav")
```

---

## Шпаргалка

| Задача                  | Метод                                                                 |
|-------------------------|-----------------------------------------------------------------------|
| Список голосов          | `lib.names()`                                                         |
| Полные метаданные       | `lib.list_voices()`                                                   |
| Проверить наличие       | `lib.exists("Alice")`                                                 |
| Загрузить голос         | `lib.load("Alice")`                                                   |
| Добавить голос          | `lib.save("Alice", model.create_voice_clone_prompt("ref.wav"))`       |
| Переименовать           | `lib.rename("Alice", "Alice UK")`                                     |
| Удалить                 | `lib.delete("Alice")`                                                 |
| Потоковая генерация     | `for audio, status in model.generate_streaming(text, voice_clone_prompt=prompt)` |
| Обычная генерация       | `audio_list = model.generate(text, voice_clone_prompt=prompt)`        |

---

## Структура файлов библиотеки голосов

```
~/.omnivoice/voices/          # (или ваша папка)
    Alice.json                # метаданные: name, ref_text, ref_rms
    Alice.pt                  # тензор ref_audio_tokens (8, T)
    Bob.json
    Bob.pt
    Narrator_RU.json
    Narrator_RU.pt
```

Файлы `.pt` можно переносить между машинами — модель и версия PyTorch должны совпадать.
