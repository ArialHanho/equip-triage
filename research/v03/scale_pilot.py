"""30k synthetic session reference, randomly initialized causal model only.

AI-authored reference. Families, virtual equipment, conditions are split-disjoint.
Draws are individually seeded, but share a hand-authored simulator: not independent
real-world equipment evidence. Targets and thresholds never fit on test sessions.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path
import random
import sys
import time

import torch
from torch import nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from event_transformer import Attention

EVENTS = ('BOS', 'START', 'MODE_A', 'MODE_B', 'LOAD', 'MEASURE', 'CHECK', 'WAIT',
          'UNLOAD', 'RELEASE_A', 'RELEASE_B', 'END', 'STEP')
IDS = {e: i for i, e in enumerate(EVENTS)}
SIZES = {'train': 24000, 'validation': 3000, 'test': 3000}
FAMILIES = {'train': range(24), 'validation': range(24, 27), 'test': range(27, 30)}
ANOMALIES = ('wrong_release', 'missing_unload', 'sequence_gap', 'delay', 'duplicate', 'clock_reverse')


def generate(seed=42, per_family=1000):
    """5% eval anomalies, normal-only train. No actual workplace material."""
    for split, families in FAMILIES.items():
        for family in families:
            for serial in range(per_family):
                rng = random.Random(seed * 10**8 + family * 10000 + serial)
                mode = rng.randrange(2)
                equipment = f'SYN-{split}-{family:02}-{serial % 10:02}'
                condition = f'COND-{split}-{family:02}-{serial % 5}'
                # Different timing domains by split; known interval travels with condition.
                base_interval = {'train': 0.5, 'validation': 0.8, 'test': 1.1}[split]
                interval = base_interval + rng.random() * 0.2
                tokens = ['START', 'MODE_A' if mode == 0 else 'MODE_B', 'LOAD']
                # A family's 5-bit CHECK schedule is unique and held out as a whole.
                for stage in range(5):
                    tokens.append('STEP')
                    for _ in range(rng.randint(1, 3)):
                        tokens.append('MEASURE')
                        if rng.random() < 0.22:
                            tokens.append('WAIT')  # legitimate jitter / normal variation
                    if family & (1 << stage):
                        tokens.append('CHECK')
                tokens += ['UNLOAD', 'RELEASE_A' if mode == 0 else 'RELEASE_B', 'END']
                abnormal = split != 'train' and serial < int(per_family * 0.05)
                label = ANOMALIES[serial % len(ANOMALIES)] if abnormal else 'normal'
                if label == 'wrong_release':
                    tokens[-2] = 'RELEASE_B' if mode == 0 else 'RELEASE_A'
                elif label == 'missing_unload':
                    tokens.remove('UNLOAD')
                records, stamp = [], 0.0
                for i, token in enumerate(tokens):
                    stamp += interval * rng.uniform(0.5, 1.5)
                    records.append({'id': token, 'sequence': i + 1, 'time': round(stamp, 6)})
                index = rng.randrange(4, len(records) - 3)
                if label == 'sequence_gap':
                    for record in records[index:]:
                        record['sequence'] += 1
                elif label == 'delay':
                    for record in records[index:]:
                        record['time'] += 12 * interval
                elif label == 'duplicate':
                    records.insert(index, dict(records[index-1]))
                elif label == 'clock_reverse':
                    records[index]['time'] = records[index-1]['time'] - interval
                yield {'split': split, 'family': f'FAMILY-{family:02}', 'equipment': equipment,
                       'condition': condition, 'session': f'{split}-{family:02}-{serial:04}',
                       'label': label, 'interval': interval, 'records': records}


def tokens(row):
    return [IDS[r['id']] for r in row['records']]


def audit(rows):
    summary = {}
    partitions = {s: [r for r in rows if r['split'] == s] for s in FAMILIES}
    for key in ('family', 'equipment', 'condition', 'session'):
        sets = {s: {r[key] for r in rs} for s, rs in partitions.items()}
        assert not (sets['train'] & sets['validation'] or sets['train'] & sets['test'] or sets['validation'] & sets['test'])
    full = Counter()
    token_signatures = Counter()
    cross = defaultdict(set)
    for r in rows:
        # Excludes unique identifiers: duplicate audit is over actual numeric event payload.
        signature = hashlib.sha256(json.dumps(r['records'], sort_keys=True).encode()).hexdigest()
        full[signature] += 1
        signature = tuple(tokens(r))
        token_signatures[signature] += 1
        cross[signature].add(r['split'])
    for split, rs in partitions.items():
        summary[split] = {'sessions': len(rs), 'event_rows': sum(len(r['records']) for r in rs),
                          'families': len({r['family'] for r in rs}),
                          'equipment': len({r['equipment'] for r in rs}),
                          'conditions': len({r['condition'] for r in rs}),
                          'labels': dict(Counter(r['label'] for r in rs)),
                          'token_only_extra_duplicates_within_split': len(rs)-len({tuple(tokens(r)) for r in rs}),
                          'min_events': min(len(r['records']) for r in rs),
                          'max_events': max(len(r['records']) for r in rs)}
    assert all(r['label'] == 'normal' for r in partitions['train'])
    assert sum(v-1 for v in full.values()) == 0
    # Token collisions are reported, not disguised by identifiers or timestamps.
    assert all(len(v) == 1 for v in cross.values()), 'token signatures cross split boundaries'
    summary['duplicate_audit'] = {'full_numeric_payload_extra_duplicates': sum(v-1 for v in full.values()),
                                  'token_only_extra_duplicates': sum(v-1 for v in token_signatures.values()),
                                  'token_only_signatures_crossing_splits': sum(len(v)>1 for v in cross.values())}
    return summary


def numeric_score(row):
    maximum = 0.0
    for a, b in zip(row['records'], row['records'][1:]):
        if b['sequence'] != a['sequence'] + 1 or b['time'] <= a['time']:
            return 20.0
        maximum = max(maximum, (b['time'] - a['time']) / row['interval'])
    return 20.0 if maximum > 3 else 0.0


def state_score(row):
    """Strong known-grammar baseline; remembers mode across arbitrary waits.

    Flags a violation when observed. END signals a completed session; no future
    event is used. This baseline knows the simulator grammar by construction.
    """
    score = numeric_score(row)
    mode, phase, measurements = None, 'start', 0
    for record in row['records']:
        token = record['id']
        bad = False
        if phase == 'start':
            bad, phase = token != 'START', 'mode'
        elif phase == 'mode':
            bad = token not in ('MODE_A', 'MODE_B')
            mode, phase = token, 'load'
        elif phase == 'load':
            bad, phase = token != 'LOAD', 'measure'
        elif phase == 'measure':
            if token == 'MEASURE':
                measurements += 1
            elif token == 'UNLOAD':
                bad, phase = measurements < 1, 'release'
            elif token not in ('WAIT', 'CHECK', 'STEP'):
                bad = True
        elif phase == 'release':
            expected = 'RELEASE_A' if mode == 'MODE_A' else 'RELEASE_B'
            bad, phase = token != expected, 'end'
        elif phase == 'end':
            bad, phase = token != 'END', 'closed'
        else:
            bad = True
        if bad:
            score = 20.0
    return score


class Trigram:
    def __init__(self, train):
        self.counts = defaultdict(Counter)
        for row in train:
            history = [0]
            for token in tokens(row):
                for size in range(3):
                    self.counts[tuple(history[-size:]) if size else ()][token] += 1
                history.append(token)

    def score(self, row):
        history, maximum = [0], 0.0
        for token in tokens(row):
            for size in range(min(2, len(history)), -1, -1):
                counts = self.counts.get(tuple(history[-size:]) if size else ())
                if counts:
                    break
            probability = (counts[token]+1) / (sum(counts.values())+len(EVENTS))
            maximum = max(maximum, -math.log(probability))
            history.append(token)
        return maximum


class Model(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.embedding = nn.Embedding(len(EVENTS), width)
        pos = torch.arange(96).unsqueeze(1)
        scale = torch.exp(torch.arange(0, width, 2) * (-math.log(10000.0) / width))
        pe = torch.zeros(96, width)
        pe[:, 0::2], pe[:, 1::2] = torch.sin(pos*scale), torch.cos(pos*scale)
        self.register_buffer('position', pe)
        self.attention = Attention(width, 4)
        self.norm1, self.norm2 = nn.LayerNorm(width), nn.LayerNorm(width)
        self.ff = nn.Sequential(nn.Linear(width, 64), nn.GELU(), nn.Linear(64, width))
        self.out = nn.Linear(width, len(EVENTS))

    def forward(self, x):
        x = self.embedding(x) + self.position[:x.shape[1]]
        x = x + self.attention(self.norm1(x))
        return self.out(x + self.ff(self.norm2(x)))


def batch(rows):
    ids = [tokens(r) for r in rows]
    x = torch.zeros(len(ids), max(map(len, ids)), dtype=torch.long)
    y = torch.full_like(x, -100)
    for i, values in enumerate(ids):
        x[i, :len(values)] = torch.tensor([0]+values[:-1])
        y[i, :len(values)] = torch.tensor(values)
    return x, y


def neural_scores(model, rows, size=128):
    values = []
    model.eval()
    with torch.no_grad():
        for offset in range(0, len(rows), size):
            x, y = batch(rows[offset:offset+size])
            loss = F.cross_entropy(model(x).transpose(1, 2), y, ignore_index=-100, reduction='none')
            values.extend(loss.max(dim=1).values.tolist())
    return values


def operating_metrics(labels, scores, threshold):
    tp = fp = tn = fn = 0
    for actual, score in zip(labels, scores):
        flag = score > threshold
        tp += actual and flag
        fp += not actual and flag
        tn += not actual and not flag
        fn += actual and not flag
    return {'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
            'precision': tp / max(1, tp+fp), 'recall': tp / max(1, tp+fn),
            'false_positive_rate': fp / max(1, fp+tn)}


def average_precision(labels, scores):
    """Step-integrated PR-AUC (AP); equal score ties evaluated as a group."""
    groups = defaultdict(lambda: [0, 0])
    for label, score in zip(labels, scores):
        groups[score][int(label)] += 1
    positives = sum(labels)
    tp = fp = 0
    area = 0.0
    for score in sorted(groups, reverse=True):
        negative, positive = groups[score]
        tp += positive
        fp += negative
        area += (positive / max(1, positives)) * tp / (tp+fp)
    return min(1.0, max(0.0, area))  # floating-point roundoff at perfect AP


def evaluate(validation, test, val_scores, test_scores):
    val_labels = [r['label'] != 'normal' for r in validation]
    test_labels = [r['label'] != 'normal' for r in test]
    # Normal-only validation quantile: strict > handles ties conservatively.
    normal = sorted(s for r, s in zip(validation, val_scores) if r['label'] == 'normal')
    threshold = normal[math.ceil(0.99 * len(normal))-1]
    return {'threshold_source': 'validation normal only; target FPR <=1%', 'threshold': threshold,
            'validation': {'pr_auc_average_precision': average_precision(val_labels, val_scores),
                           **operating_metrics(val_labels, val_scores, threshold)},
            'test': {'pr_auc_average_precision': average_precision(test_labels, test_scores),
                     **operating_metrics(test_labels, test_scores, threshold)},
            'test_recall_by_anomaly': {label: sum(s>threshold for r,s in zip(test,test_scores) if r['label']==label) /
                                      max(1,sum(r['label']==label for r in test)) for label in ANOMALIES}}


def run(seed=42, epochs=3, per_family=1000, export=None):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    start = time.perf_counter()
    rows = list(generate(seed, per_family))
    counts = audit(rows)
    print(json.dumps({'dataset_audit': counts}), flush=True)
    if export:
        with export.open('w', encoding='utf-8') as file:
            for row in rows:
                file.write(json.dumps(row, separators=(',',':'))+'\n')
    train = [r for r in rows if r['split']=='train']
    validation = [r for r in rows if r['split']=='validation']
    test = [r for r in rows if r['split']=='test']
    ngram = Trigram(train)
    model = Model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    rng = random.Random(seed)
    losses = []
    training_start = time.perf_counter()
    for epoch in range(epochs):
        rng.shuffle(train)
        total = count = 0
        model.train()
        for offset in range(0, len(train), 128):
            chunk = train[offset:offset+128]
            x, y = batch(chunk)
            optimizer.zero_grad()
            loss = F.cross_entropy(model(x).transpose(1,2), y, ignore_index=-100)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += loss.item()*len(chunk)
            count += len(chunk)
        losses.append(total/count)
        print(json.dumps({'epoch':epoch+1, 'train_loss':losses[-1]}), flush=True)
    training_seconds = time.perf_counter()-training_start
    ng_val, ng_test = [ngram.score(r) for r in validation], [ngram.score(r) for r in test]
    tr_val, tr_test = neural_scores(model,validation), neural_scores(model,test)
    num_val, num_test = [numeric_score(r) for r in validation], [numeric_score(r) for r in test]
    hybrid = lambda raw, numeric: [max(a,b) for a,b in zip(raw,numeric)]
    result = {'scope':'synthetic simulator only; not real equipment performance',
              'seed':seed, 'epochs':epochs, 'torch':torch.__version__, 'device':'cpu', 'threads':1,
              'parameters':sum(p.numel() for p in model.parameters()), 'audit':counts,
              'train_loss':losses, 'training_seconds':training_seconds,
              'ngram_event_only':evaluate(validation,test,ng_val,ng_test),
              'transformer_event_only':evaluate(validation,test,tr_val,tr_test),
              'ngram_with_numeric_rules':evaluate(validation,test,hybrid(ng_val,num_val),hybrid(ng_test,num_test)),
              'transformer_with_numeric_rules':evaluate(validation,test,hybrid(tr_val,num_val),hybrid(tr_test,num_test)),
              'state_machine_with_numeric_rules':evaluate(validation,test,[state_score(r) for r in validation],[state_score(r) for r in test]),
              'elapsed_seconds':time.perf_counter()-start}
    return result


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--epochs',type=int,default=3)
    parser.add_argument('--per-family',type=int,default=1000)
    parser.add_argument('--export-jsonl',type=Path)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.epochs<1 or args.per_family<100:
        parser.error('epochs>=1 and per-family>=100 required')
    result=run(args.seed,args.epochs,args.per_family,args.export_jsonl)
    args.out.write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    print('Saved '+str(args.out),flush=True)
