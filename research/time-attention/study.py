"""Matched synthetic timing and order faults on unseen permutations of four stages."""

import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import accumulate, permutations
import json
import math
from pathlib import Path
import random
import platform
import statistics
import time

import torch
from torch import nn
from torch.nn import functional as F

# 0 padding, 1 START, 2/3 synthetic recipe profiles, 4..7 stage markers,
# 8 READ, 9 ACK, 10 CLOSE, 11 END. Time units are invented ticks.
MODES = (
    "no_time",
    "input_time",
    "position_bias",
    "input_and_position",
    "time_bias",
    "input_and_bias",
)


@dataclass(frozen=True)
class Trace:
    events: tuple
    times: tuple


@dataclass(frozen=True)
class Pair:
    normal: Trace
    fault: Trace
    family: str
    kind: str


def make_pair(order, serial, rng):
    profile = serial % 2
    events = [1, 2 + profile]
    gaps = [0, rng.randint(1, 3)]
    reads, required_acks = {}, []
    for stage in order:
        events.extend([stage, 8])
        reads[stage] = len(events) - 1
        long_stage = (stage - 4) % 2 == profile
        gaps.extend(
            [rng.randint(1, 3), rng.randint(9, 14) if long_stage else rng.randint(2, 6)]
        )
        required = (stage < 6) == bool(profile)
        if required or rng.random() < 0.5:
            if required:
                required_acks.append(len(events))
            events.append(9)
            gaps.append(rng.randint(1, 3))
        events.append(10)
        gaps.append(rng.randint(1, 3))
    events.append(11)
    gaps.append(rng.randint(1, 3))
    normal = Trace(tuple(events), tuple(accumulate(gaps)))
    if (serial // 2) % 2 == 0:
        # Exchange a long and a short read interval; endpoint and all counts match.
        a = reads[rng.choice([4, 6])]
        b = reads[rng.choice([5, 7])]
        gaps[a], gaps[b] = gaps[b], gaps[a]
        kind = "timing"
    else:
        a = rng.choice(required_acks)
        events[a], events[a + 1] = events[a + 1], events[a]
        kind = "order"
    fault = Trace(tuple(events), tuple(accumulate(gaps)))
    return Pair(normal, fault, "".join(str(s - 4) for s in order), kind)


def data():
    rng = random.Random(20261125)
    families = list(permutations(range(4, 8)))
    rng.shuffle(families)
    partitions = [families[:16], families[16:20], families[20:]]
    return [
        [make_pair(family, i, rng) for family in part for i in range(64)]
        for part in partitions
    ]


def intervals(trace):
    return [0] + [b - a for a, b in zip(trace.times, trace.times[1:])]


def audit(parts):
    seen_families, seen_events = set(), set()
    result = []
    for pairs in parts:
        families = {p.family for p in pairs}
        sequences = {t.events for p in pairs for t in (p.normal, p.fault)}
        assert families.isdisjoint(seen_families) and sequences.isdisjoint(seen_events)
        seen_families.update(families)
        seen_events.update(sequences)
        for p in pairs:
            a, b = p.normal, p.fault
            assert Counter(a.events) == Counter(b.events) and a.times[-1] == b.times[-1]
            assert sorted(intervals(a)) == sorted(intervals(b))
            assert all(gap > 0 for gap in intervals(a)[1:] + intervals(b)[1:])
            if p.kind == "timing":
                assert a.events == b.events and a.times != b.times
            else:
                assert a.times == b.times and a.events != b.events
        result.append(
            dict(
                pairs=len(pairs),
                families=sorted(families),
                faults=dict(Counter(p.kind for p in pairs)),
            )
        )
    return result


class CountProfile:
    """Score a multiset by its frequency among training normals."""

    def __init__(self, pairs):
        self.counts = Counter(tuple(sorted(p.normal.events)) for p in pairs)
        self.total = len(pairs)

    def score(self, trace):
        frequency = self.counts[tuple(sorted(trace.events))]
        return -math.log((frequency + 1) / (self.total + 1))


class Trigram:
    """Mean add-one log loss, backing off when a context is unseen."""

    def __init__(self, pairs):
        self.counts = defaultdict(Counter)
        self.vocabulary = {e for p in pairs for e in p.normal.events}
        for p in pairs:
            events = p.normal.events
            for i, event in enumerate(events):
                for width in range(min(2, i) + 1):
                    self.counts[events[i - width : i]][event] += 1

    def score(self, trace):
        losses = []
        for i, event in enumerate(trace.events):
            for width in range(min(2, i), -1, -1):
                counts = self.counts.get(trace.events[i - width : i])
                if counts:
                    break
            losses.append(
                -math.log((counts[event] + 1) / (counts.total() + len(self.vocabulary)))
            )
        return statistics.mean(losses)


class Precedence:
    """Count violations of orders that hold for unique milestones in every normal."""

    def __init__(self, pairs):
        rows = [p.normal.events for p in pairs]
        milestones = set.intersection(
            *[{e for e, n in Counter(row).items() if n == 1} for row in rows]
        )
        self.rules = [
            (a, b)
            for a, b in permutations(sorted(milestones), 2)
            if all(row.index(a) < row.index(b) for row in rows)
        ]

    def score(self, trace):
        return sum(trace.events.index(a) > trace.events.index(b) for a, b in self.rules)


class ConditionalRule:
    """Fit normal stage durations and permitted ACK/READ/CLOSE subsequences."""

    def __init__(self, pairs):
        self.patterns = defaultdict(set)
        durations = defaultdict(list)
        for pair in pairs:
            for key, seq, gap in self.stages(pair.normal):
                self.patterns[key].add(seq)
                durations[key].append(gap)
        self.bounds = {k: (min(v), max(v)) for k, v in durations.items()}

    @staticmethod
    def stages(trace):
        points = [i for i, e in enumerate(trace.events) if 4 <= e <= 7] + [
            len(trace.events) - 1
        ]
        for start, end in zip(points, points[1:]):
            yield (
                (trace.events[1], trace.events[start]),
                trace.events[start + 1 : end],
                trace.times[start + 1] - trace.times[start],
            )

    def score(self, trace):
        maximum = 0
        for key, seq, gap in self.stages(trace):
            low, high = self.bounds[key]
            maximum = max(
                maximum, int(seq not in self.patterns[key]), low - gap, gap - high
            )
        return maximum


def tensors(traces):
    ids = nn.utils.rnn.pad_sequence(
        [torch.tensor(t.events) for t in traces], batch_first=True
    )
    stamps = nn.utils.rnn.pad_sequence(
        [torch.tensor(t.times) for t in traces], batch_first=True
    )
    return ids, stamps


class Block(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(width, 3 * width)
        self.merge = nn.Linear(width, width)
        self.norm1 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        self.ff = nn.Sequential(
            nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width)
        )
        self.bias = nn.Embedding(17, heads)
        nn.init.zeros_(self.bias.weight)

    def forward(self, x, valid, buckets):
        batch, length, width = x.shape
        q, k, v = (
            self.qkv(x)
            .reshape(batch, length, 3, self.heads, width // self.heads)
            .permute(2, 0, 3, 1, 4)
        )
        logits = (q @ k.transpose(-2, -1)) / math.sqrt(width // self.heads)
        if buckets is not None:
            logits = logits + self.bias(buckets).permute(0, 3, 1, 2)
        logits = logits.masked_fill(~valid[:, None, None, :], float("-inf"))
        output = (logits.softmax(-1) @ v).transpose(1, 2).reshape(batch, length, width)
        x = self.norm1(x + self.merge(output))
        return self.norm2(x + self.ff(x))


class Encoder(nn.Module):
    def __init__(self, mode):
        super().__init__()
        self.mode = mode
        self.event = nn.Embedding(12, 32, padding_idx=0)
        self.position = nn.Embedding(32, 32)
        self.input_time = nn.Linear(2, 32, bias=False)
        self.blocks = nn.ModuleList([Block(32, 4), Block(32, 4)])
        self.out = nn.Linear(32, 1)

    def forward(self, ids, stamps):
        pos = torch.arange(ids.size(1))
        x = self.event(ids) + self.position(pos)
        if self.mode in ("input_time", "input_and_position", "input_and_bias"):
            gaps = torch.diff(stamps, dim=1, prepend=stamps[:, :1])
            elapsed = stamps - stamps[:, :1]
            features = torch.stack([gaps, elapsed], dim=-1).clamp_min(0).float().log1p()
            x = x + self.input_time(features)
        buckets = None
        if self.mode in (
            "time_bias",
            "input_and_bias",
            "position_bias",
            "input_and_position",
        ):
            coordinate = (
                pos[None, :]
                if self.mode in ("position_bias", "input_and_position")
                else stamps
            )
            delta = coordinate[:, :, None] - coordinate[:, None, :]
            magnitude = (
                torch.floor(torch.log2(delta.abs().float().clamp_min(1))).long() + 1
            )
            buckets = 8 + delta.sign() * magnitude.clamp(max=8)
        for block in self.blocks:
            x = block(x, ids.ne(0), buckets)
        return self.out(x[:, 0]).squeeze(-1)


def fit(mode, pairs, seed):
    torch.manual_seed(seed)
    model = Encoder(mode)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002)
    ids, stamps = tensors([t for p in pairs for t in (p.normal, p.fault)])
    labels = torch.arange(len(ids)).remainder(2).float()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(10):
        for ix in torch.randperm(len(ids), generator=generator).split(64):
            optimizer.zero_grad()
            loss = F.binary_cross_entropy_with_logits(
                model(ids[ix], stamps[ix]), labels[ix]
            )
            loss.backward()
            optimizer.step()
    model.eval()

    @torch.inference_mode()
    def score(traces):
        ids, stamps = tensors(traces)
        return torch.cat(
            [model(a, b) for a, b in zip(ids.split(64), stamps.split(64))]
        ).tolist()

    return score


def metrics(pairs, values, threshold):
    normal, bad = values[::2], values[1::2]

    def counts(indices):
        return dict(
            pairs=len(indices),
            tp=sum(bad[i] > threshold for i in indices),
            fp=sum(normal[i] > threshold for i in indices),
            wins=sum(bad[i] > normal[i] for i in indices),
            ties=sum(bad[i] == normal[i] for i in indices),
        )

    return {
        key: counts(
            [i for i, p in enumerate(pairs) if key in ("all", p.kind, p.family)]
        )
        for key in ["all", "timing", "order"] + sorted({p.family for p in pairs})
    }


def run():
    torch.set_num_threads(1)
    train, validation, test = data()
    result = dict(
        torch=torch.__version__,
        threads=1,
        python=platform.python_version(),
        data_seed=20261125,
        epochs=10,
        batch_size=64,
        learning_rate=0.002,
        validation_normal_quantile=0.99,
        splits=audit([train, validation, test]),
        runs=[],
    )
    baselines = {
        "counts": CountProfile,
        "trigram": Trigram,
        "precedence": Precedence,
        "conditional_rule": ConditionalRule,
    }
    for mode in (*baselines, *MODES):
        for seed in [None] if mode in baselines else [41, 42, 43]:
            start = time.process_time()
            if mode in baselines:
                rule = baselines[mode](train)
                score = lambda traces: [rule.score(t) for t in traces]
            else:
                score = fit(mode, train, seed)
            fit_cpu = time.process_time() - start
            normal_val = score([p.normal for p in validation])
            threshold = sorted(normal_val)[math.ceil(0.99 * len(normal_val)) - 1]
            start = time.process_time()
            scores = score([t for p in test for t in (p.normal, p.fault)])
            scoring_cpu = time.process_time() - start
            measured = metrics(test, scores, threshold)
            result["runs"].append(
                dict(
                    method=mode,
                    seed=seed,
                    threshold=threshold,
                    validation_fp=sum(s > threshold for s in normal_val),
                    test=measured,
                    fit_cpu_s=fit_cpu,
                    score_cpu_ms_per_trace=scoring_cpu * 1000 / len(scores),
                )
            )
            print(mode, seed, measured["all"], flush=True)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(run(), indent=2) + "\n")
