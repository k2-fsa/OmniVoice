from __future__ import annotations

import argparse
import json
from pathlib import Path

from omnivoice.audiobook.generation import load_generation_checkpoint
from omnivoice.audiobook.qc import inspect_audiobook_plan_audio


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Create an audiobook QC report from a generated plan.")
    parser.add_argument("--plan", required=True, help="Audiobook plan/checkpoint JSON.")
    parser.add_argument("--output", required=True, help="QC report JSON path.")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    plan = load_generation_checkpoint(Path(args.plan))
    report = inspect_audiobook_plan_audio(plan)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote audiobook QC report: {output}")
    if report.gate_status != "pass":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
