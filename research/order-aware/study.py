"""Matched-pair study of synthetic equipment event order. Run with --help."""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import random
import statistics
import sys
import time

import torch
from torch import nn
from torch.nn import functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "public-logbert"))
from bert_mlm import EventMLM, corrupt_for_training, leave_one_out_nll, masked_loss, pad_batch

# IDs 0..2 are PAD/MASK/UNK as in EventMLM. All other IDs are in the fixed schema.
EVENTS = ["PAD", "MASK", "UNK", "START", "CLAMP", "ZERO", "SCAN", "RELEASE",
          "POLL", "WAIT", "SAMPLE", "ACK", "END"]
SIZES = (1600, 400, 400)
DATA_SEED = 20261125
EPOCHS = 10


def generate():
    rng = random.Random(DATA_SEED)
    seen = set()
    partitions = []
    for count in SIZES:
        pairs = []
        while len(pairs) < count:
            sequence = [3]
            anchors = []
            for event in (4, 5, 6, 7):
                sequence.extend(rng.choices(range(8, 12), k=rng.randrange(1, 5)))
                anchors.append(len(sequence))
                sequence.append(event)
            sequence.append(12)
            normal = tuple(sequence)
            # Reverse exactly one of three adjacent milestone pairs. Leave all
            # intervening observations in place. Reject duplicate model inputs.
            kind = len(pairs) % 3
            a, b = anchors[kind:kind + 2]
            sequence[a], sequence[b] = sequence[b], sequence[a]
            altered = tuple(sequence)
            if normal in seen or altered in seen:
                continue
            seen.update((normal, altered))
            pairs.append((normal, altered, kind))
        partitions.append(pairs)
    return partitions


class Counts:
    def __init__(self, rows):
        self.profiles = Counter(tuple(sorted(x)) for x in rows)
        self.total = len(rows)

    def __call__(self, x):
        return -math.log((1 + self.profiles[tuple(sorted(x))]) / (self.total + 1))


class Trigram:
    def __init__(self, rows):
        self.counts = defaultdict(Counter)
        for row in rows:
            for i, token in enumerate(row):
                for k in range(min(i, 2) + 1):
                    self.counts[row[i-k:i]][token] += 1

    def __call__(self, x):
        losses = []
        for i, token in enumerate(x):
            for k in range(min(i, 2), -1, -1):
                counts = self.counts.get(x[i-k:i])
                if counts:
                    break
            losses.append(-math.log((counts[token]+1)/(sum(counts.values())+len(EVENTS)-3)))
        return statistics.mean(losses)


class Precedence:
    """Learn directed pairs that always hold among events present once per session."""
    def __init__(self, rows):
        milestones = set.intersection(*[{t for t,n in Counter(r).items() if n == 1} for r in rows])
        self.rules = [(a,b) for a in milestones for b in milestones if a != b
                      and all(r.index(a) < r.index(b) for r in rows)]

    def __call__(self, x):
        return sum(x.index(a) > x.index(b) for a,b in self.rules if a in x and b in x)


class OrderModel(EventMLM):
    def __init__(self):
        super().__init__(len(EVENTS), 32)
        self.head = nn.Linear(32, 1)

    def order_logits(self, ids):
        valid = ids.ne(0)
        states = self.input_norm(self.tokens(ids) + self.positions(torch.arange(ids.shape[1])))
        for block in self.blocks:
            states = block(states, valid)
        # START is already in every trace. Its contextual vector pools both sides.
        return self.head(states[:, 0]).squeeze(1)


def train(pairs, seed, joint):
    torch.manual_seed(seed)
    model = OrderModel() if joint else EventMLM(len(EVENTS), 32)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    masks = torch.Generator().manual_seed(seed)
    order = list(range(len(pairs)))
    shuffle = random.Random(seed)
    for _ in range(EPOCHS):
        shuffle.shuffle(order)
        for offset in range(0, len(order), 64):
            selected = [pairs[i] for i in order[offset:offset+64]]
            normals = [p[0] for p in selected]
            hidden, targets = corrupt_for_training(pad_batch(normals), len(EVENTS), masks)
            loss = masked_loss(model(hidden), targets)
            if joint:
                corrupted = [p[1] for p in selected]
                logits = model.order_logits(pad_batch(normals + corrupted))
                labels = torch.cat((torch.zeros(len(normals)), torch.ones(len(corrupted))))
                loss = loss + F.binary_cross_entropy_with_logits(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    return model.eval()


@torch.inference_mode()
def score_model(model, rows, joint):
    values = []
    for i in range(0, len(rows), 32):
        batch = pad_batch(rows[i:i+32])
        scores = model.order_logits(batch) if joint else leave_one_out_nll(model, batch).nanmean(dim=1)
        values.extend(scores.tolist())
    return values


def operating_point(validation):
    return sorted(validation)[math.ceil(.99*len(validation))-1]


def summarize(pairs, scores, threshold):
    normal, corrupt = scores[::2], scores[1::2]
    delta = [c-n for n,c in zip(normal, corrupt)]
    return dict(pairs=len(delta), wins=sum(d>0 for d in delta), ties=sum(d==0 for d in delta),
                losses=sum(d<0 for d in delta), median_score_difference=statistics.median(delta),
                tp=sum(c>threshold for c in corrupt), fp=sum(n>threshold for n in normal),
                both_classified_correctly=sum(n<=threshold<c for n,c in zip(normal, corrupt)),
                by_swap={str(k):dict(total=sum(p[2]==k for p in pairs),
                                     flagged=sum(p[2]==k and c>threshold for p,c in zip(pairs,corrupt)))
                         for k in range(3)})


def benchmark(seeds, pair_path):
    torch.set_num_threads(1)
    train_pairs, validation, test = generate()
    normals = [p[0] for p in train_pairs]
    calibration = [p[0] for p in validation]
    test_rows = [row for p in test for row in p[:2]]
    report = dict(data_seed=DATA_SEED, sizes=SIZES, epochs=EPOCHS, torch=torch.__version__,
                  platform=platform.platform(), cpu_threads=1, runs=[])
    details = []
    methods = [(cls.__name__, None, cls) for cls in (Counts, Trigram, Precedence)]
    methods.extend((name, seed, None) for seed in seeds for name in ("MLM", "OrderHead"))
    for name, seed, cls in methods:
        start = time.process_time()
        if cls:
            model = cls(normals)
            score = lambda rows: [model(r) for r in rows]
        else:
            model = train(train_pairs, seed, name == "OrderHead")
            score = lambda rows: score_model(model, rows, name == "OrderHead")
        fitted = time.process_time()-start
        validation_scores = score(calibration)
        threshold = operating_point(validation_scores)
        start = time.process_time()
        scores = score(test_rows)
        scored = time.process_time()-start
        summary = summarize(test, scores, threshold)
        summary.update(method=name, seed=seed, threshold=threshold,
                       validation_fp=sum(s>threshold for s in validation_scores),
                       training_cpu_seconds=fitted, test_cpu_seconds=scored,
                       mean_test_cpu_ms_per_trace=scored*1000/len(test_rows))
        report['runs'].append(summary)
        for i, ((normal, _, kind), a, b) in enumerate(zip(test, scores[::2], scores[1::2])):
            details.append([name,seed,i,kind,hashlib.sha256(bytes(normal)).hexdigest(),a,b,b-a])
    if pair_path:
        with pair_path.open('w', newline='') as f:
            writer=csv.writer(f, lineterminator='\n')
            writer.writerow(['method','seed','pair_index','swap','normal_sha256','normal_score','corrupt_score','difference'])
            writer.writerows(details)
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--pairs", type=Path, help="Write individual test-pair scores to CSV")
    parser.add_argument("--seeds", nargs="+", type=int, default=[41,42,43])
    args = parser.parse_args()
    args.out.write_text(json.dumps(benchmark(args.seeds,args.pairs),indent=2)+"\n")
