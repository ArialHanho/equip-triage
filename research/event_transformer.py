"""AI-authored reference: synthetic causal event prediction, no pretrained weights.

For target event t, input is [BOS, events[:t]]. Attention cannot see later
positions; the current target is never an input at its prediction position.
Only normal TRAIN scenarios fit models. VALIDATION picks anomaly thresholds.
TEST scenarios are held out by template, not randomly split windows.
"""
import argparse
from collections import Counter, defaultdict
import csv
import json
import math
from pathlib import Path
import random
import time

import torch
from torch import nn
import torch.nn.functional as F

VOCAB = ('BOS', 'START', 'CHECK', 'LOAD', 'MEASURE', 'RETEST', 'UNLOAD', 'END', 'WAIT')
TOKEN = {name: i for i, name in enumerate(VOCAB)}
ROOT = Path(__file__).resolve().parent
# Different complete templates in each split. No window crosses a scenario.
TEMPLATES = {
    'train': [('START', 'CHECK', 'LOAD', 'MEASURE', 'UNLOAD', 'END'),
              ('START', 'LOAD', 'CHECK', 'MEASURE', 'RETEST', 'MEASURE', 'UNLOAD', 'END')],
    'validation': [('START', 'CHECK', 'LOAD', 'MEASURE', 'RETEST', 'MEASURE', 'UNLOAD', 'END')],
    'test': [('START', 'LOAD', 'CHECK', 'MEASURE', 'UNLOAD', 'END')],
}


def scenarios(seed=42):
    rng = random.Random(seed)
    rows = []
    for split, templates in TEMPLATES.items():
        for template_id, template in enumerate(templates):
            # Each split also has a disjoint number of consecutive measurements.
            counts = {'train': range(1, 7), 'validation': range(7, 10), 'test': range(10, 13)}[split]
            for repeats in counts:
                normal = []
                for event in template:
                    normal.extend([event] * (repeats if event == 'MEASURE' else 1))
                cases = [('normal', normal)]
                if split != 'train':
                    first = normal.index('MEASURE')
                    missing = normal[:first] + normal[first + repeats:]
                    corrupt = list(normal)
                    corrupt[first] = 'END'
                    novel = list(normal)
                    novel[first] = 'WAIT'
                    cases += [('missing_measure', missing), ('early_end', corrupt), ('unexpected_wait', novel)]
                for label, events in cases:
                    rows.append({'split': split, 'scenario': f'{split}-template{template_id}-repeat{repeats}-{label}',
                                 'template': f'{split}-template{template_id}', 'label': label, 'events': events})
    rng.shuffle(rows)
    # A full normal sequence is not duplicated across splits.
    signatures = defaultdict(set)
    for row in rows:
        signature = tuple(row['events'])
        signatures[signature].add(row['split'])
    assert all(len(splits) == 1 for splits in signatures.values())
    return rows


class Attention(nn.Module):
    def __init__(self, width, heads):
        super().__init__()
        assert width % heads == 0
        self.heads, self.depth = heads, width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.out = nn.Linear(width, width)

    def forward(self, x):
        batch, length, width = x.shape
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q, k, v = [z.reshape(batch, length, self.heads, self.depth).transpose(1, 2) for z in (q, k, v)]
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.depth)
        future = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
        scores = scores.masked_fill(future, float('-inf'))
        weights = scores.softmax(dim=-1)
        result = (weights @ v).transpose(1, 2).contiguous().reshape(batch, length, width)
        return self.out(result)


class TinyTransformer(nn.Module):
    def __init__(self, width=32, heads=4, max_length=96):
        super().__init__()
        self.embedding = nn.Embedding(len(VOCAB), width)
        position = torch.arange(max_length).unsqueeze(1)
        scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        pe = torch.zeros(max_length, width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(position * scale), torch.cos(position * scale)
        self.register_buffer('position', pe)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.attention = Attention(width, heads)
        self.ff = nn.Sequential(nn.Linear(width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.output = nn.Linear(width, len(VOCAB))

    def forward(self, ids):
        x = self.embedding(ids) + self.position[:ids.shape[1]]
        x = x + self.attention(self.norm1(x))  # residual 1
        x = x + self.ff(self.norm2(x))        # residual 2
        return self.output(x)


def batch(rows):
    length = max(len(row['events']) for row in rows)
    inputs = torch.zeros(len(rows), length, dtype=torch.long)
    targets = torch.full_like(inputs, -100)  # ignored padding in masked loss
    for i, row in enumerate(rows):
        ids = [TOKEN[e] for e in row['events']]
        inputs[i, :len(ids)] = torch.tensor([TOKEN['BOS']] + ids[:-1])
        targets[i, :len(ids)] = torch.tensor(ids)
    return inputs, targets


class NGram:
    def __init__(self, rows, order=3):
        self.order = order
        self.counts = defaultdict(Counter)
        for row in rows:
            history = [TOKEN['BOS']]
            for event in row['events']:
                token = TOKEN[event]
                for size in range(order):
                    context = tuple(history[-size:]) if size else ()
                    self.counts[context][token] += 1
                history.append(token)

    def losses(self, events):
        history, values = [TOKEN['BOS']], []
        for event in events:
            token = TOKEN[event]
            for size in range(min(self.order - 1, len(history)), -1, -1):
                context = tuple(history[-size:]) if size else ()
                counts = self.counts.get(context)
                if counts:
                    break
            # Add-one smoothing, including unseen event IDs.
            values.append(-math.log((counts[token] + 1) / (sum(counts.values()) + len(VOCAB))))
            history.append(token)
        return values


def transformer_losses(model, events):
    x, y = batch([{'events': events}])
    with torch.no_grad():
        return F.cross_entropy(model(x).reshape(-1, len(VOCAB)), y.reshape(-1), reduction='none').tolist()


def metrics(rows, score, threshold):
    tp = fp = tn = fn = 0
    details = []
    for row in rows:
        value = score(row['events'])
        predicted, actual = value > threshold, row['label'] != 'normal'
        tp += predicted and actual
        fp += predicted and not actual
        tn += not predicted and not actual
        fn += not predicted and actual
        details.append({'scenario': row['scenario'], 'label': row['label'], 'score': round(value, 6), 'flagged': predicted})
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'anomaly_f1': round(2 * tp / max(1, 2 * tp + fp + fn), 6), 'scenarios': details}


def evaluate(rows, losses):
    # Maximum observed event surprise; online score at t needs no event after t.
    score = lambda events: max(losses(events))
    validation = [r for r in rows if r['split'] == 'validation']
    test = [r for r in rows if r['split'] == 'test']
    candidates = sorted({score(r['events']) for r in validation})
    threshold = max(candidates, key=lambda t: (metrics(validation, score, t)['anomaly_f1'], t))
    return {'validation_threshold': threshold, 'validation': metrics(validation, score, threshold),
            'test': metrics(test, score, threshold)}


def run(seed=42, epochs=120):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rows = scenarios(seed)
    train = [r for r in rows if r['split'] == 'train']
    model = TinyTransformer()  # Entirely random initialization, no download path.
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    x, target = batch(train)
    started = time.perf_counter()
    for _ in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(x)
        # Causal prediction + ignored padding. No bidirectional masked-token objective.
        loss = F.cross_entropy(logits.reshape(-1, len(VOCAB)), target.reshape(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
    elapsed = time.perf_counter() - started
    model.eval()
    baseline = NGram(train)
    result = {'scope': 'synthetic template-disjoint pilot; not production evidence',
              'seed': seed, 'epochs': epochs, 'torch': torch.__version__, 'device': 'cpu', 'threads': 1,
              'parameters': sum(p.numel() for p in model.parameters()), 'training_seconds': round(elapsed, 4),
              'training_final_loss': round(loss.item(), 6),
              'split_counts': {s: sum(r['split'] == s for r in rows) for s in TEMPLATES},
              'ngram': evaluate(rows, baseline.losses),
              'transformer': evaluate(rows, lambda events: transformer_losses(model, events))}
    return rows, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=120)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--export-csv', type=Path)
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error('--epochs must be positive')
    rows, result = run(args.seed, args.epochs)
    if args.export_csv:
        with args.export_csv.open('w', newline='', encoding='utf-8') as file:
            writer = csv.DictWriter(file, fieldnames=['split', 'scenario', 'template', 'label', 'events'], lineterminator='\n')
            writer.writeheader()
            writer.writerows({**row, 'events': ' '.join(row['events'])} for row in rows)
    payload = json.dumps(result, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.write_text(payload, encoding='utf-8')
    else:
        print(payload, end='')


if __name__ == '__main__':
    main()
