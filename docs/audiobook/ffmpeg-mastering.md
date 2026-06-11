# FFmpeg Mastering

OmniVoice uses FFmpeg as an optional local mastering engine.

## Commands

Concatenate segments and normalize stream shape:

```powershell
ffmpeg -n -f concat -safe 0 -i concat.txt -ar 44100 -ac 1 -c:a pcm_s16le chapter_raw.wav
```

Remaster a chapter or full book:

```powershell
ffmpeg -n -i chapter_raw.wav -af "silenceremove=start_periods=1:start_duration=0.2:start_threshold=-50dB,atempo=1.0,loudnorm=I=-20:TP=-3:LRA=11,alimiter=limit=0.707946" chapter_master.wav
```

Inspect metadata:

```powershell
ffprobe -v error -show_format -show_streams -of json chapter_master.wav
```

Create a QC report from a generated checkpoint:

```powershell
omnivoice-audiobook-qc --plan checkpoint.json --output qc_report.json
```

## Defaults

- Target loudness: `-20 LUFS`
- True peak: `-3 dB`
- Loudness range: `11`
- Stream normalization: mono, 44.1 kHz, PCM WAV
- Remaster output: forced to mono, 44.1 kHz by default
- `dynaudnorm` and `acompressor`: opt-in
- `silenceremove`: edge cleanup only by default
- Existing outputs are preserved by default. Use CLI `--overwrite` only when replacing an output is intentional.

## Safety

Generated chunks are never overwritten during mastering. Concatenated and
remastered outputs are separate files.

If `ffmpeg` or `ffprobe` is missing, the tool fails with a clear local error.

Do not claim audiobook readiness unless the QC report has `gate_status=pass`.
