# Contributing to OmniVoice

Thank you for your interest in contributing to OmniVoice! We use pre-commit hooks to ensure code quality and consistency. Before contributing, please follow these guidelines to enable and use the pre-commit hooks.

## Setup

```bash
pip install pre-commit
pre-commit install
```

This enables automatic code style checks (linting, formatting, trailing whitespace, etc.) on every `git commit`.

Before committing, run:

```bash
pre-commit run --all-files
```

This will auto-fix any style issues. Then stage and commit as usual:

```bash
git add .
git commit -m "your message"
```
