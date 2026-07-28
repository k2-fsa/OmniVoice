#!/usr/bin/env python3
"""Measure cold/warm BamiBERT and deterministic normalization costs."""

import argparse
import json
import os
import time

from omnivoice.text_normalization import (
    DEFAULT_MODEL_PATH,
    get_bamibert_detector,
    normalize_candidates,
)
from omnivoice.text_normalization.detector import _reset_detector_cache

CASES = {
    "no-number": "Tôi thích đọc sách vào buổi sáng.",
    "short-heavy": "Tôi có 25 quyển sách và đã đọc 2 quyển.",
    "multi-entity": (
        "Hẹn lúc 08:30 ngày 27/07/2026, phí là 250.000 đồng "
        "và gọi số 0912 345 678."
    ),
    "long": (
        "Thông báo cuộc họp bắt đầu lúc 08:30 ngày 27/07/2026. "
        "Phí tham dự là 250.000 đồng cho mỗi người; vui lòng gọi "
        "0912 345 678 và cung cấp mã sinh viên 3036123456 khi đăng ký."
    ),
}


def _rss_mib() -> float | None:
    try:
        with open("/proc/self/status", encoding="utf-8") as status:
            for line in status:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024
    except OSError:
        return None
    return None


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--repeat", type=int, default=3)
    args = parser.parse_args(argv)
    if args.repeat < 1:
        parser.error("--repeat must be positive")

    rss_before = _rss_mib()
    _reset_detector_cache()
    started = time.perf_counter()
    detector = get_bamibert_detector(args.model_path, args.device)
    load_seconds = time.perf_counter() - started
    rss_loaded = _rss_mib()

    measurements = []
    for name, text in CASES.items():
        for iteration in range(args.repeat):
            total_started = time.perf_counter()
            prediction_started = time.perf_counter()
            candidates = detector(text)
            prediction_seconds = time.perf_counter() - prediction_started
            deterministic_started = time.perf_counter()
            result = normalize_candidates(text, candidates)
            deterministic_seconds = time.perf_counter() - deterministic_started
            measurements.append(
                {
                    "case": name,
                    "iteration": iteration + 1,
                    "characters": len(text),
                    "candidates": len(candidates),
                    "prediction_seconds": prediction_seconds,
                    "deterministic_seconds": deterministic_seconds,
                    "total_seconds": time.perf_counter() - total_started,
                    "changed": result.text != text,
                    "rss_mib": _rss_mib(),
                }
            )

    payload = {
        "pid": os.getpid(),
        "model_path": args.model_path,
        "device": args.device,
        "cold_load_seconds": load_seconds,
        "rss_before_mib": rss_before,
        "rss_after_load_mib": rss_loaded,
        "measurements": measurements,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
