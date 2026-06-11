# OpenRouter Audiobook Workflow

OpenRouter is an explicit online mode. It is not used by the offline DOCX
planner.

## Flow

1. Read DOCX locally.
2. Split the manuscript into semantic chunks with `chunk_docx_document`.
3. Preview each chunk before provider submission when requested.
4. Send one approved chunk at a time to OpenRouter.
5. Validate the structured JSON response.
6. Merge returned chapters and segments into the local audiobook plan.
7. Generate audio locally segment by segment.

## Consent Gate

The CLI requires `--confirm-online-provider` for real provider calls. Without it,
the adapter fails before HTTP transport.

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
