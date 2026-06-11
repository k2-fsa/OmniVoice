# Large DOCX Files up to 500 Pages

The pipeline is designed for resumable work, not one giant request.

## Units

- DOCX: local source of truth.
- Chunk: provider planning unit, default target 1,800 words.
- Segment: TTS generation unit, usually 400-900 characters.
- Chapter: review and assembly unit.
- Full book: final optional merge.

## Modes

- Manual: user generates one segment or block at a time.
- Continuous: generate next pending segment until paused or failed.
- Chapter: generate and assemble a single chapter.
- Full book: assemble all completed chapters into one output file.

## Resume

`AudiobookGenerationJob` tracks segment state and checkpoint helpers persist the
plan as JSON. Pending and failed segments remain resumable without regenerating
completed audio.

Useful commands:

```powershell
omnivoice-audiobook-workflow status --plan audiobook_plan.json
omnivoice-audiobook-workflow next --plan audiobook_plan.json
omnivoice-audiobook-workflow mark-generated --plan audiobook_plan.json --segment-id seg_001_000001 --audio-path audio/seg.wav --output checkpoint.json
omnivoice-audiobook-workflow mark-failed --plan checkpoint.json --segment-id seg_001_000002 --error "provider failed" --output checkpoint.json
```

`next` redacts manuscript text by default. Add `--include-text` only for local
operator handoff to a TTS renderer.

## Expected Scale

A 500-page manuscript can produce thousands of segments. The system should be
operated through chunk/chapter progress rather than a single monolithic job.

The harness includes a synthetic 500-paragraph scale fixture to assert bounded
chunk size, deterministic chunk IDs, ordered paragraph ranges, short continuity
summaries, and provider payloads that contain only the selected chunk rather
than the whole manuscript.
