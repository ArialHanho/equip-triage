"""Separate generator-specific memory diagnostic, not the main anomaly pilot.
The target follows a shared suffix of LOAD, MEASURE, UNLOAD IDs. CHECK versus RETEST
at the start selects the final token; a trigram cannot observe this mode.
Random initialization; no pretrained models; complete scenarios split by length.
"""
import argparse
import json
from pathlib import Path
import time

import torch
import torch.nn.functional as F

from event_transformer import NGram, TinyTransformer, TOKEN, VOCAB, batch


def make_rows():
    rows = []
    lengths = {'train': [3, 4, 6, 7, 9, 10], 'validation': [5, 8], 'test': [11, 12]}
    for split, counts in lengths.items():
        for count in counts:
            for mode, target in [('CHECK', 'CHECK'), ('RETEST', 'RETEST')]:
                rows.append({'split': split, 'scenario': f'{split}-{mode}-{count}',
                             'events': ['START', mode] + ['MEASURE'] * count + ['LOAD', 'MEASURE', 'UNLOAD', target, 'END'],
                             'target_index': 5 + count})
    assert len({tuple(r['events']) for r in rows}) == len(rows)
    return rows


def run(seed=42, epochs=240):
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rows = make_rows()
    train = [r for r in rows if r['split'] == 'train']
    model = TinyTransformer()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    x, y = batch(train)
    start = time.perf_counter()
    for _ in range(epochs):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x).reshape(-1, len(VOCAB)), y.reshape(-1), ignore_index=-100)
        loss.backward()
        optimizer.step()
    model.eval()
    ngram = NGram(train)
    results = []
    with torch.no_grad():
        for row in rows:
            if row['split'] == 'train':
                continue
            index = row['target_index']
            prefix = row['events'][:index]
            # Only prefix is ever fed to either model at this target.
            tensor = torch.tensor([[TOKEN['BOS']] + [TOKEN[e] for e in prefix]])
            probability = model(tensor)[0, -1].softmax(-1)
            prediction = VOCAB[int(probability.argmax())]
            context = tuple(TOKEN[e] for e in prefix[-2:])
            counts = ngram.counts[context]
            total = sum(counts.values()) + len(VOCAB)
            probs = [(counts[i] + 1) / total for i in range(len(VOCAB))]
            ng_prediction = VOCAB[max(range(len(VOCAB)), key=lambda i: probs[i])]
            target = row['events'][index]
            results.append({'split': row['split'], 'scenario': row['scenario'], 'prefix': prefix,
                            'target': target, 'transformer_prediction': prediction,
                            'ngram_prediction': ng_prediction,
                            'transformer_target_probability': round(float(probability[TOKEN[target]]), 6),
                            'ngram_target_probability': round(probs[TOKEN[target]], 6)})
    summary = {}
    for split in ('validation', 'test'):
        subset = [r for r in results if r['split'] == split]
        summary[split] = {name + '_correct': sum(r[name + '_prediction'] == r['target'] for r in subset)
                          for name in ('ngram', 'transformer')}
        summary[split]['n'] = len(subset)
    return {'scope': 'hand-designed mode memory diagnostic, not real equipment performance',
            'seed': seed, 'epochs': epochs, 'torch': torch.__version__, 'device': 'cpu',
            'training_seconds': round(time.perf_counter() - start, 4),
            'summary': summary, 'predictions': results}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    payload = json.dumps(run(args.seed), ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.write_text(payload, encoding='utf-8')
    else:
        print(payload, end='')
