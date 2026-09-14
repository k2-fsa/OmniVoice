#!/usr/bin/env python3
"""Generate a Colombian Spanish synthetic corpus from the live books D1 database."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

import soundfile as sf
import torch

from omnivoice import OmniVoice, VoiceClonePrompt

LOGGER = logging.getLogger(__name__)
QUERY = """
SELECT id, text
FROM page_variants
WHERE language = 'es'
  AND framework = 'atal'
  AND text IS NOT NULL
  AND length(trim(text)) > 0
  AND id > ?
ORDER BY id
LIMIT ?
"""
TRANSIENT_HTTP_STATUSES = {429, 500, 502, 503, 504}
WHITESPACE = re.compile(r"\s+")


class D1Error(RuntimeError):
    """Raised when Cloudflare D1 rejects or cannot complete a query."""


@dataclass(frozen=True)
class Sample:
    source_id: int
    sample_id: str
    text: str

    @property
    def relative_audio_path(self) -> Path:
        return Path("audio") / f"{self.sample_id}.flac"


def load_env(path: Path) -> None:
    """Load simple KEY=VALUE entries without overwriting the process environment."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        os.environ.setdefault(key.strip(), value)


def required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class D1Client:
    """Minimal read-only Cloudflare D1 REST client with transient retries."""

    def __init__(
        self,
        account_id: str,
        database_id: str,
        api_token: str,
        *,
        timeout: float = 60.0,
    ) -> None:
        self.url = (
            f"https://api.cloudflare.com/client/v4/accounts/{account_id}"
            f"/d1/database/{database_id}/query"
        )
        self.api_token = api_token
        self.timeout = timeout

    def query(self, sql: str, params: list[Any]) -> list[dict[str, Any]]:
        body = json.dumps({"sql": sql, "params": params}).encode("utf-8")
        last_error: Exception | None = None

        for attempt in range(4):
            request = urllib.request.Request(
                self.url,
                data=body,
                headers={
                    "Authorization": f"Bearer {self.api_token}",
                    "Content-Type": "application/json",
                    "User-Agent": "omnivoice-colombian-spanish-generator",
                },
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    payload = json.load(response)
            except urllib.error.HTTPError as error:
                response_body = error.read().decode("utf-8", errors="replace")[:500]
                last_error = D1Error(f"D1 HTTP {error.code}: {response_body}")
                if error.code not in TRANSIENT_HTTP_STATUSES:
                    raise last_error from error
            except (urllib.error.URLError, TimeoutError) as error:
                last_error = error
            else:
                if not payload.get("success"):
                    raise D1Error(f"D1 query failed: {payload.get('errors')}")
                results = payload.get("result")
                if not isinstance(results, list) or not results:
                    raise D1Error("D1 response did not contain a query result")
                rows = results[0].get("results", [])
                if not isinstance(rows, list):
                    raise D1Error("D1 query result did not contain rows")
                return rows

            if attempt < 3:
                time.sleep(0.5 * (2**attempt))

        raise D1Error("D1 request failed after four attempts") from last_error


def iter_spanish_rows(client: D1Client, page_size: int) -> Iterator[dict[str, Any]]:
    """Fetch every eligible row using stable primary-key pagination."""
    last_id = 0
    while True:
        rows = client.query(QUERY, [last_id, page_size])
        if not rows:
            return
        for row in rows:
            yield row
        next_id = int(rows[-1]["id"])
        if next_id <= last_id:
            raise D1Error("D1 pagination did not advance")
        last_id = next_id
        if len(rows) < page_size:
            return


def normalize_transcript(text: str) -> str:
    """Collapse paragraph and other whitespace to the spaces spoken by the model."""
    return WHITESPACE.sub(" ", text).strip()


def load_samples(
    client: D1Client,
    *,
    page_size: int,
    limit: int | None,
) -> tuple[list[Sample], int]:
    """Normalize then deduplicate transcripts, retaining the first source row."""
    samples: list[Sample] = []
    seen: set[str] = set()
    duplicate_count = 0

    for row in iter_spanish_rows(client, page_size):
        text = normalize_transcript(str(row["text"]))
        if not text:
            continue
        if text in seen:
            duplicate_count += 1
            continue
        seen.add(text)
        sample_id = hashlib.sha256(text.encode("utf-8")).hexdigest()
        samples.append(Sample(int(row["id"]), sample_id, text))
        if limit is not None and len(samples) >= limit:
            break

    return samples, duplicate_count


def valid_flac(path: Path, sample_rate: int) -> bool:
    if not path.is_file():
        return False
    try:
        info = sf.info(path)
    except RuntimeError:
        return False
    return info.format == "FLAC" and info.samplerate == sample_rate and info.frames > 0


def write_flac_atomic(path: Path, audio: Any, sample_rate: int) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        sf.write(temporary, audio, sample_rate, format="FLAC")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def write_manifest_atomic(output_dir: Path, samples: list[Sample]) -> Path:
    manifest_path = output_dir / "manifest.jsonl"
    temporary = output_dir / f".manifest.jsonl.{os.getpid()}.tmp"
    try:
        with temporary.open("w", encoding="utf-8") as stream:
            for sample in samples:
                record = {
                    "id": sample.sample_id,
                    "audio_path": sample.relative_audio_path.as_posix(),
                    "text": sample.text,
                    "language_id": "es",
                }
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, manifest_path)
    finally:
        temporary.unlink(missing_ok=True)
    return manifest_path


def batched(samples: list[Sample], batch_size: int) -> Iterator[list[Sample]]:
    for offset in range(0, len(samples), batch_size):
        yield samples[offset : offset + batch_size]


def generate(args: argparse.Namespace) -> None:
    load_env(args.env_file)
    client = D1Client(
        required_env("CLOUDFLARE_ACCOUNT_ID"),
        required_env("CLOUDFLARE_D1_BOOKS_DATABASE_ID"),
        required_env("CLOUDFLARE_API_TOKEN"),
        timeout=args.d1_timeout,
    )
    samples, duplicate_count = load_samples(
        client,
        page_size=args.page_size,
        limit=args.limit,
    )
    if not samples:
        raise RuntimeError("The D1 query returned no Spanish transcripts")
    LOGGER.info(
        "Loaded %d unique transcripts (%d normalized duplicates skipped)",
        len(samples),
        duplicate_count,
    )

    audio_dir = args.output_dir / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    model = OmniVoice.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=torch.float16,
    )
    prompt = VoiceClonePrompt.load(args.prompt)
    sample_rate = model.sampling_rate
    if sample_rate != 24_000:
        raise RuntimeError(
            f"OmniVoice reported {sample_rate} Hz; this corpus requires 24000 Hz"
        )

    pending = [
        sample
        for sample in samples
        if args.overwrite
        or not valid_flac(args.output_dir / sample.relative_audio_path, sample_rate)
    ]
    pending.sort(key=lambda sample: (-len(sample.text), sample.source_id))
    LOGGER.info(
        "Generating %d transcripts in length-grouped batches of up to %d",
        len(pending),
        args.batch_size,
    )

    torch.manual_seed(args.seed)
    completed = 0
    for batch in batched(pending, args.batch_size):
        texts = [sample.text for sample in batch]
        audios = model.generate(
            text=texts,
            language="es",
            voice_clone_prompt=prompt,
            audio_chunk_duration=args.audio_chunk_duration,
            audio_chunk_threshold=args.audio_chunk_threshold,
        )
        if len(audios) != len(batch):
            raise RuntimeError(
                f"OmniVoice returned {len(audios)} audios for a batch of {len(batch)}"
            )
        for sample, audio in zip(batch, audios):
            write_flac_atomic(
                args.output_dir / sample.relative_audio_path,
                audio,
                sample_rate,
            )
        completed += len(batch)
        LOGGER.info("Generated %d/%d queued transcripts", completed, len(pending))

    invalid = [
        sample.relative_audio_path.as_posix()
        for sample in samples
        if not valid_flac(args.output_dir / sample.relative_audio_path, sample_rate)
    ]
    if invalid:
        raise RuntimeError(f"Refusing to write manifest; invalid audio: {invalid[:3]}")

    manifest = write_manifest_atomic(args.output_dir, samples)
    LOGGER.info("Wrote %d entries to %s", len(samples), manifest)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than zero")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env-file", type=Path, default=Path(".env"))
    parser.add_argument("--model", default="k2-fsa/OmniVoice")
    parser.add_argument(
        "--prompt", type=Path, default=Path("voices/colombian_spanish.pt")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("results/colombian_es_synthetic")
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=positive_int, default=4)
    parser.add_argument("--page-size", type=positive_int, default=1000)
    parser.add_argument("--limit", type=positive_int)
    parser.add_argument("--d1-timeout", type=float, default=60.0)
    parser.add_argument("--audio-chunk-duration", type=float, default=15.0)
    parser.add_argument("--audio-chunk-threshold", type=float, default=30.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = build_parser().parse_args()
    if args.d1_timeout <= 0:
        raise SystemExit("--d1-timeout must be greater than zero")
    if args.audio_chunk_duration <= 0:
        raise SystemExit("--audio-chunk-duration must be greater than zero")
    if args.audio_chunk_threshold <= 0:
        raise SystemExit("--audio-chunk-threshold must be greater than zero")
    generate(args)


if __name__ == "__main__":
    main()
