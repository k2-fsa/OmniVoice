"""Generate installer/icon.ico — a waveform on a purple gradient.

Run once at build time (needs Pillow); the resulting .ico is committed so
normal rebuilds don't depend on it:
    .venv/Scripts/python.exe installer/make_icon.py
"""

import math
from pathlib import Path

from PIL import Image, ImageDraw

HERE = Path(__file__).parent
SIZE = 256
TOP = (30, 30, 46)  # #1e1e2e (control-panel BG)
BOTTOM = (98, 78, 158)  # deep purple
WAVE = (203, 166, 247)  # #cba6f7 (control-panel accent)


def make_base(size: int = SIZE) -> Image.Image:
    img = Image.new("RGBA", (size, size))
    d = ImageDraw.Draw(img)

    # Vertical gradient inside a rounded square.
    for y in range(size):
        t = y / (size - 1)
        color = tuple(int(a + (b - a) * t) for a, b in zip(TOP, BOTTOM))
        d.line([(0, y), (size, y)], fill=color + (255,))
    mask = Image.new("L", (size, size), 0)
    ImageDraw.Draw(mask).rounded_rectangle(
        [0, 0, size - 1, size - 1], radius=size // 5, fill=255
    )
    img.putalpha(mask)

    # Waveform: symmetric vertical bars with a speech-like envelope.
    d = ImageDraw.Draw(img)
    n = 11
    bar_w = size // 24
    gap = (size - n * bar_w) // (n + 1)
    mid = size // 2
    for i in range(n):
        env = math.sin(math.pi * (i + 0.5) / n)  # bell overall
        wig = 0.55 + 0.45 * math.sin(i * 2.1 + 0.7)  # per-bar variety
        h = int(size * 0.33 * env * wig) + size // 20
        x = gap + i * (bar_w + gap)
        d.rounded_rectangle(
            [x, mid - h, x + bar_w, mid + h], radius=bar_w // 2, fill=WAVE + (255,)
        )
    return img


if __name__ == "__main__":
    base = make_base()
    out = HERE / "icon.ico"
    base.save(
        out,
        sizes=[
            (16, 16),
            (24, 24),
            (32, 32),
            (48, 48),
            (64, 64),
            (128, 128),
            (256, 256),
        ],
    )
    print(f"wrote {out}")
