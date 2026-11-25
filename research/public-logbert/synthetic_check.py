"""Deterministic CPU check that resolving a hidden event can require a later event."""
import argparse
import itertools
import json
import random

import torch
from torch.nn import functional as F

from bert_mlm import EventMLM, MASK, train_epoch

# Positions 2 and 4 are the hidden target and its right-context cue respectively.
TARGET = 2
CUE = 4


def make_groups(seed):
    # Each nuisance triple owns a quartet. Siblings share every visible event except the cue.
    keys = list(itertools.product(range(11, 19), repeat=3))
    random.Random(seed).shuffle(keys)

    def expand(part):
        return torch.tensor([
            [left_a, left_b, 3 + mode, right_noise, 7 + mode]
            for left_a, left_b, right_noise in part for mode in range(4)
        ])

    train_keys, test_keys = keys[:224], keys[224:256]
    assert set(train_keys).isdisjoint(test_keys)
    return expand(train_keys), expand(test_keys)


@torch.inference_mode()
def measure(model, examples, hide_cue=False):
    model.eval()
    inputs = examples.clone()
    expected = inputs[:, TARGET].clone()
    inputs[:, TARGET] = MASK
    if hide_cue:
        inputs[:, CUE] = MASK
    assert not (inputs == expected[:, None]).any()
    logits = model(inputs)[:, TARGET]
    correct = int(logits.argmax(dim=-1).eq(expected).sum())
    return {
        "correct": correct,
        "accuracy": correct / len(expected),
        "mean_nll_nats": float(F.cross_entropy(logits, expected)),
    }


def run(seed=42):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    train, test = make_groups(seed)
    generator = torch.Generator().manual_seed(seed + 1)
    order = torch.Generator().manual_seed(seed + 2)
    model = EventMLM(vocab_size=19, max_len=5, width=32, heads=4, layers=2)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    before = measure(model, test)
    epochs = 16
    for _ in range(epochs):
        batches = train[torch.randperm(len(train), generator=order)].split(32)
        last = train_epoch(model, batches, optimizer, generator)
    after = measure(model, test)
    removed = measure(model, test, hide_cue=True)
    if not (after["accuracy"] >= 0.9 and after["mean_nll_nats"] < before["mean_nll_nats"]
            and removed["accuracy"] <= 0.25):
        raise RuntimeError(f"Right-context check failed: {before}, {after}, {removed}")
    return {
        "seed": seed, "torch": str(torch.__version__), "device": "cpu",
        "train_groups": len(train) // 4, "test_groups": len(test) // 4,
        "train_sequences": len(train), "test_sequences": len(test),
        "parameters": sum(p.numel() for p in model.parameters()),
        "epochs": epochs, "last_training_epoch": last,
        "initial": before, "trained": after, "trained_cue_hidden": removed,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=42)
    print(json.dumps(run(parser.parse_args().seed), indent=2))
