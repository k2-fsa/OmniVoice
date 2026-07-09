# DOCX Ingestion

The local ingestion path reads `.docx` files with the Python standard library:
ZIP extraction plus WordprocessingML parsing from `word/document.xml`.

## Extracted Fields

- paragraph index;
- paragraph text;
- paragraph style when present;
- source file SHA-256.

## Chapter Detection

A paragraph is treated as a chapter heading when:

- Word style resembles `Heading`, `Title`, or `Titulo`; or
- the text is short, alphabetic, and does not end like a normal sentence.

Ambiguous structure should remain visible in the generated plan instead of being
silently rewritten.

## Non-Goals

- No editorial rewriting.
- No external LLM call.
- No provider upload.
- No destructive manuscript cleanup.
