"""Batch worker executed by the disposable baseline environment."""

import argparse
import json
import sys
import time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("vietnormalizer", "soe_vinorm"))
    args = parser.parse_args()
    if args.method == "vietnormalizer":
        from vietnormalizer import VietnameseNormalizer, __version__

        normalizer = VietnameseNormalizer(enable_transliteration=False)
        metadata = {"version": __version__, "transliteration": False}
    else:
        import onnxruntime
        from soe_vinorm import SoeNormalizer, __version__

        normalizer = SoeNormalizer()
        metadata = {
            "version": __version__,
            "onnxruntime": onnxruntime.__version__,
            "model": "vinhdq842/soe-vinorm@cb9705b",
            "device": "CPU",
        }
    for line in sys.stdin:
        request = json.loads(line)
        started = time.perf_counter()
        try:
            output = normalizer.normalize(request["text"])
            response = {
                "id": request["id"],
                "output_text": output,
                "supported": True,
                "status": "success",
                "error_type": None,
                "error_message": None,
                "latency_ms": (time.perf_counter() - started) * 1000,
                "metadata": metadata,
            }
        except Exception as error:
            response = {
                "id": request["id"],
                "output_text": "",
                "supported": False,
                "status": "runtime_error",
                "error_type": type(error).__name__,
                "error_message": str(error),
                "latency_ms": (time.perf_counter() - started) * 1000,
                "metadata": metadata,
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
