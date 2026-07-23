"""JSON-lines runner executed inside a disposable external-package venv."""

import argparse
import json
import sys


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("method", choices=("vietnormalizer", "soe_vinorm"))
    args = parser.parse_args()
    if args.method == "vietnormalizer":
        from vietnormalizer import VietnameseNormalizer, __version__

        normalizer = VietnameseNormalizer(enable_transliteration=False)
        call = normalizer.normalize
    else:
        from soe_vinorm import SoeNormalizer, __version__

        normalizer = SoeNormalizer()
        call = normalizer.normalize
    for line in sys.stdin:
        request = json.loads(line)
        try:
            output = call(request["text"])
            response = {
                "id": request["id"],
                "output_text": output,
                "available": True,
                "changed": output != request["text"],
                "uncertain": False,
                "error": None,
                "metadata": {"version": __version__},
            }
        except Exception as error:
            response = {
                "id": request["id"],
                "output_text": "",
                "available": False,
                "changed": False,
                "uncertain": True,
                "error": f"{type(error).__name__}: {error}",
                "metadata": {"version": __version__},
            }
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
