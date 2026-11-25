"""Route a single bilingual packet or a JSONL stream with the calibrated study."""

import argparse
import json
import sys
from pathlib import Path

import torch
from threadpoolctl import threadpool_limits

from data import ROOT, Packet
from evaluation import LANGS
from study import SEEDS, packet_from_dict, predictor


def route(predict, record):
    try:
        packet = packet_from_dict(record)
    except ValueError:
        # A missing or malformed record has no safe candidate-to-line binding.
        packet = Packet(("", ""), ("", ""), "")
    return predict(packet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jsonl", action="store_true", help="Read one packet per line.")
    parser.add_argument("--candidate", action="append", metavar="DOMAIN:TAG")
    parser.add_argument("--line", action="append", help="One received memo line.")
    parser.add_argument("--lang", choices=LANGS, default="en")
    parser.add_argument("--method", help="Default is the validation-selected method.")
    parser.add_argument("--gate", choices=("utility", "safety"), default="utility")
    parser.add_argument("--seed", type=int, choices=SEEDS, default=42)
    parser.add_argument("--lock", type=Path, default=ROOT / "run" / "validation.json")
    args = parser.parse_args()
    if args.jsonl and (args.candidate or args.line):
        parser.error("Use either JSONL or individual candidate and line arguments.")
    if not args.jsonl and not (args.candidate or args.line):
        parser.error("Supply a packet with --candidate and --line, or use --jsonl.")

    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        predict = predictor(args)
        if args.jsonl:
            for line in sys.stdin:
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except ValueError:
                    record = None
                print(json.dumps(route(predict, record), ensure_ascii=False))
        else:
            candidates = []
            for item in args.candidate or []:
                domain, _, tag = item.partition(":")
                candidates.append({"domain": domain, "tag": tag})
            record = {"candidates": candidates, "memo": "\n".join(args.line or [])}
            print(json.dumps(route(predict, record), ensure_ascii=False))


if __name__ == "__main__":
    main()
