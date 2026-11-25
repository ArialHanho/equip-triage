"""Train on local state-labelled memos or predict using a saved Transformer."""

import argparse
from contextlib import nullcontext
import json
import math
from pathlib import Path
import sys

import torch
from torch.nn import functional as F

from .encoding import STATES
from .inference import load_checkpoint, predict, save_checkpoint
from .model import CandidateStateTransformer, ModelConfig


def train(args):
    if (
        min(args.epochs, args.batch_size) < 1
        or not math.isfinite(args.lr)
        or args.lr <= 0
    ):
        raise ValueError("Use positive epochs, batch size and learning rate")
    if args.data.resolve() == args.out.resolve():
        raise ValueError("Use a checkpoint path different from the training input")
    torch.manual_seed(args.seed)
    model = CandidateStateTransformer(ModelConfig(threshold=args.threshold))
    records, targets = [], []
    with args.data.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                labels = row.get("states") if isinstance(row, dict) else None
                if (
                    not isinstance(labels, list)
                    or len(labels) != 2
                    or any(s not in STATES for s in labels)
                ):
                    raise ValueError(
                        "states must contain two labels in candidate order"
                    )
                # Validate here so a bad training row can be located.
                model.encode([row])
            except ValueError as exc:
                raise ValueError(f"Training line {number}: {exc}") from exc
            records.append(row)
            targets.append([STATES.index(label) for label in labels])
    if not records:
        raise ValueError("Training data is empty")
    batch = model.encode(records)
    target = torch.tensor(targets)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    generator = torch.Generator().manual_seed(args.seed + 1)
    for _ in range(args.epochs):
        for indices in torch.randperm(len(records), generator=generator).split(
            args.batch_size
        ):
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch.select(indices))
            loss = F.cross_entropy(
                logits.reshape(-1, len(STATES)), target[indices].reshape(-1)
            )
            loss.backward()
            optimizer.step()
    model.eval()
    with torch.inference_mode():
        total = 0.0
        for indices in torch.arange(len(records)).split(args.batch_size):
            logits = model(batch.select(indices))
            total += float(
                F.cross_entropy(
                    logits.flatten(0, 1), target[indices].flatten(), reduction="sum"
                )
            )
    save_checkpoint(model, args.out)
    print(
        json.dumps(
            {
                "model": type(model).__name__,
                "records": len(records),
                "seed": args.seed,
                "epochs": args.epochs,
                "training_state_loss": total / (2 * len(records)),
                "parameters": sum(p.numel() for p in model.parameters()),
                "checkpoint": str(args.out),
            },
            ensure_ascii=False,
        )
    )


def infer(args):
    model = load_checkpoint(args.model)
    source = (
        nullcontext(sys.stdin)
        if args.input == "-"
        else Path(args.input).open(encoding="utf-8")
    )
    with source as stream:
        for line in stream:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except ValueError:
                record = None
            print(json.dumps(predict(model, record), ensure_ascii=False))


def main():
    parser = argparse.ArgumentParser(prog="equip-triage", description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    fit = commands.add_parser(
        "train", help="Train from scratch on a local labelled JSONL"
    )
    fit.add_argument("--data", required=True, type=Path)
    fit.add_argument("--out", required=True, type=Path)
    fit.add_argument("--epochs", type=int, default=40)
    fit.add_argument("--batch-size", type=int, default=18)
    fit.add_argument("--lr", type=float, default=0.003)
    fit.add_argument("--seed", type=int, default=42)
    fit.add_argument("--threshold", type=float, default=0.5)
    infer_parser = commands.add_parser(
        "predict", help="Read JSONL and run a saved model"
    )
    infer_parser.add_argument("--model", required=True, type=Path)
    infer_parser.add_argument(
        "--input", default="-", help="Path to JSONL or - for stdin"
    )
    args = parser.parse_args()
    torch.set_num_threads(1)
    try:
        if args.command == "train":
            train(args)
        else:
            infer(args)
    except (OSError, ValueError) as exc:
        parser.exit(2, f"{parser.prog}: {exc}\n")
