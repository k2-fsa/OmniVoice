"""Generate the Phase B ASMR listening set against the running server (:9192).

Matrix per character: normal / instruct whisper / whisper + close DSP /
whisper + drift DSP, in EN and JA. A voice with a real whisper recording adds
that clip as its own condition. Output: demo_out/asmr/<char>_<lang>_<cond>.wav

    .venv/Scripts/python.exe tests/asmr_samples.py
"""

from __future__ import annotations

import json
import os
import sys
import time
import urllib.request

import soundfile as sf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "server"))
from asmr_fx import asmr_pipeline  # noqa: E402

BASE = os.environ.get("OMNIVOICE_URL", "http://127.0.0.1:9192")
OUT = os.path.join(ROOT, "demo_out", "asmr")
os.makedirs(OUT, exist_ok=True)

TEXT = {
    "en": "Hey... it's just the two of us now. Close your eyes, I'll stay right here until you fall asleep.",
    "ja": "ねえ…今はふたりきりだよ。目を閉じて。眠るまで、ずっとそばにいるからね。",
}

CHARS = {
    # edit to your own clips under ref/ or ref-output/
    "voice_a": {"ref": "ref/default_ref.wav", "whisper_ref": None},
    "voice_b": {
        "ref": "ref-output/voice_b.wav",
        "whisper_ref": "ref-output/voice_b_asmr.wav",
    },
}

# name -> (ref key, instruct)
CONDS = [
    ("normal", "ref", None),
    ("whisper", "ref", "whisper"),
    ("whisper_female", "ref", "female, whisper"),
    ("whisper_lowpitch", "ref", "whisper, low pitch"),
    ("whisperref", "whisper_ref", None),  # only voices with a real whisper clip
    ("whisperref_whisper", "whisper_ref", "whisper"),
]


def synth(text, ref, instruct, lang):
    body = {"text": text, "ref_audio": ref, "language": lang, "guidance_scale": 2.0}
    if instruct:
        body["instruct"] = instruct
    req = urllib.request.Request(
        BASE + "/synthesize",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=300) as r:
        j = json.load(r)
    with urllib.request.urlopen(BASE + j["audio_url"], timeout=120) as r:
        return r.read(), j


def main():
    rows = []
    for cid, c in CHARS.items():
        for lang, text in TEXT.items():
            for cond, refkey, instruct in CONDS:
                ref = c.get(refkey)
                if not ref:
                    continue
                name = f"{cid}_{lang}_{cond}"
                dst = os.path.join(OUT, name + ".wav")
                t0 = time.time()
                try:
                    wav, meta = synth(text, ref, instruct, lang)
                except Exception as e:  # keep going, report
                    rows.append((name, "FAIL " + str(e)[:80]))
                    print("FAIL", name, e)
                    continue
                open(dst, "wb").write(wav)
                rows.append(
                    (
                        name,
                        f"{meta['duration']:.1f}s rtf={meta['rtf']:.2f} {time.time() - t0:.1f}s",
                    )
                )
                print("ok", name, rows[-1][1])
                # DSP variants on the plain whisper takes
                if cond in ("whisper", "whisperref"):
                    x, sr = sf.read(dst, dtype="float32")
                    for preset in ("close", "drift"):
                        y, _ = asmr_pipeline(x, sr, preset)
                        d2 = os.path.join(OUT, f"{name}_fx-{preset}.wav")
                        sf.write(d2, y, sr)
                        rows.append((os.path.basename(d2), "dsp"))
    with open(os.path.join(OUT, "INDEX.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            "Phase B ASMR listening set\n\nText EN: %s\nText JA: %s\n\n"
            % (TEXT["en"], TEXT["ja"])
        )
        for n, info in rows:
            fh.write(f"{n:48s} {info}\n")
    print("\nwrote", len(rows), "files ->", OUT)


if __name__ == "__main__":
    main()
