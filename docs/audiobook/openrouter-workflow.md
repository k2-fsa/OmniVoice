# OpenRouter Audiobook Workflow

OpenRouter is an explicit online mode. It is not used by the offline DOCX
planner.

## Flow

1. Read DOCX locally.
2. Split the manuscript into semantic chunks with `chunk_docx_document`.
3. Preview each chunk before provider submission when requested.
4. Send one approved chunk at a time to OpenRouter.
5. Validate the structured JSON response.
6. Merge returned chapters and segments into the local audiobook plan with `omnivoice-audiobook-workflow merge-openrouter`.
7. Generate audio locally segment by segment.

## Consent Gate

The CLI requires `--confirm-online-provider` for real provider calls. Without it,
the adapter fails before HTTP transport.

## Operational Commands

Preview one chunk without provider traffic:

```powershell
omnivoice-openrouter-audiobook-chunk --docx book.docx --output preview.json --model <model> --preview-only
```

Preview output redacts manuscript text by default. Use `--include-text` only for
local review artifacts that will not be committed or shared.

Structure one approved chunk:

```powershell
omnivoice-openrouter-audiobook-chunk --docx book.docx --output chunk-0001.json --model <model> --chunk-index 0 --confirm-online-provider
```

Merge chunk results into an audiobook plan:

```powershell
omnivoice-audiobook-workflow merge-openrouter --docx book.docx --result chunk-0001.json --output audiobook_plan.json --title "Livro" --model <model>
```

## Request Contract

- Endpoint: `https://openrouter.ai/api/v1/chat/completions`
- Auth: `OPENROUTER_API_KEY`
- Model: CLI `--model` or `OPENROUTER_MODEL`
- Output: `response_format.type=json_schema`
- Provider privacy settings:
  - `provider.require_parameters=true`
  - `provider.data_collection=deny`
  - `provider.zdr=true`

## Large Books

For DOCX files up to 500 pages, never send the entire manuscript. Use chunks of
roughly 1,800 target words and 2,400 maximum words by default. Each chunk carries
only a short previous summary to preserve local continuity.

## Failure Handling

- Missing consent: fail before HTTP.
- Missing API key: fail before HTTP.
- Unsupported model: fail before chunk submission.
- Invalid JSON/schema: fail the chunk and keep the local plan unchanged.
- Transient provider errors: retry only a small bounded number of times.
- Provider HTTP error bodies are redacted in local exceptions to avoid leaking manuscript text into logs.
