# API Key Handling

The frontend may collect a provider key, but it must not become the long-term
owner of that secret. After saving, the UI shows only status and non-secret
metadata.

## Required Behavior

- Hide the key after save.
- Never render the stored key.
- Never store the key in browser storage.
- Never call OpenRouter directly from browser code.
- Use a loopback-only local backend for provider operations.
- Exclude the key from backups.

## Fallback

If no secure local secret store exists, the application should require
`OPENROUTER_API_KEY` through the runtime environment or allow session-only use.
