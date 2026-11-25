"""Read JSONL cutoff snapshots and return learned source IDs and first-check decisions."""

import argparse
import json
from pathlib import Path
import sys
from contextlib import nullcontext
from .inference import load_model, predict


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--input", default="-")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args()
    model, threshold = load_model(args.model, args.device)
    source = (
        nullcontext(sys.stdin)
        if args.input == "-"
        else open(args.input, encoding="utf-8")
    )
    with source as stream:
        for line in stream:
            if line.strip():
                try:
                    row = json.loads(line)
                except ValueError:
                    row = None
                print(json.dumps(predict(model, row, threshold), ensure_ascii=False))


if __name__ == "__main__":
    main()
