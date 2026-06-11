# Audiobook JSON Contract

The audiobook plan is the stable handoff between DOCX ingestion, narration
generation, cache, assembly, and QC.

## Top-Level Shape

```json
{
  "project": {
    "title": "string",
    "author": "string",
    "language": "pt-BR",
    "genre": "technical",
    "source_docx_hash": "sha256",
    "created_at": "ISO-8601"
  },
  "voice_profile": {
    "mode": "design",
    "default_voice": "narrator",
    "speed": 0.92,
    "style": "technical_clear",
    "pronunciation_notes": []
  },
  "chapters": [],
  "qc_targets": {},
  "settings": {}
}
```

## Segment Shape

```json
{
  "id": "seg_001_000001",
  "text": "Narration text.",
  "text_hash": "sha256",
  "speaker": "narrator",
  "pause_after_ms": 750,
  "speed": 0.92,
  "tone": "neutral",
  "chapter_id": "ch_001",
  "status": "pending",
  "source_paragraph_index": null
}
```

## Rules

- Segment IDs must be deterministic by chapter and order.
- `text_hash` is SHA-256 of the segment text.
- `source_docx_hash` is SHA-256 of the original DOCX bytes.
- Provider-specific metadata belongs under `settings`, never inside raw text.
- Real manuscripts and generated audio stay out of Git.
