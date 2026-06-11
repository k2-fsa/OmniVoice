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

## Expected Scale

A 500-page manuscript can produce thousands of segments. The system should be
operated through chunk/chapter progress rather than a single monolithic job.
