"""Offline, group-isolated HDFS_v1 subset pilot; no pretrained model or weights.

Raw LogHub files stay in a temporary external cache. Only this code and aggregate
results are suitable for the Equip Triage repository. Original research/academic
terms: https://github.com/logpai/loghub/blob/master/LICENSE
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import heapq
import io
import json
import math
from pathlib import Path
import random
import tempfile
import time
import urllib.request
import zipfile

import torch
from torch.nn import functional as F

from bert_mlm import EventMLM, MASK, PAD, UNK, leave_one_out_nll, pad_batch, train_epoch

URL = 'https://zenodo.org/records/8196385/files/HDFS_v1.zip?download=1'
EXPECTED_MD5 = '76a24b4d9a6164d543fb275f89773260'
ARCHIVE_BYTES = 186645559
TRACE_MEMBER = 'preprocessed/Event_traces.csv'
LABEL_MEMBER = 'preprocessed/anomaly_label.csv'


def digest_file(path, algorithm):
    digest = hashlib.new(algorithm)
    with path.open('rb') as file:
        for block in iter(lambda: file.read(1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def obtain_archive(path, download=False):
    repo = Path(__file__).resolve().parents[2]
    if path.resolve().is_relative_to(repo):
        raise ValueError('Keep raw datasets outside the repository.')
    if not path.exists():
        if not download:
            raise FileNotFoundError('Pass --download or supply the verified official archive via --archive.')
        path.parent.mkdir(parents=True, exist_ok=True)
        request = urllib.request.Request(URL, headers={'User-Agent': 'EquipTriage-research/1.0'})
        partial = path.with_suffix('.partial')
        with urllib.request.urlopen(request, timeout=60) as response, partial.open('wb') as file:
            count = 0
            while chunk := response.read(1024*1024):
                count += len(chunk)
                if count > ARCHIVE_BYTES:
                    raise ValueError('Unexpected oversized official archive; inspect source metadata.')
                file.write(chunk)
        partial.replace(path)
    if path.stat().st_size != ARCHIVE_BYTES or digest_file(path, 'md5') != EXPECTED_MD5:
        raise ValueError('Official archive size or published MD5 mismatch.')
    return {'url': URL, 'bytes': path.stat().st_size, 'md5': EXPECTED_MD5,
            'sha256': digest_file(path, 'sha256'), 'trace_member': TRACE_MEMBER,
            'label_member': LABEL_MEMBER}


def prefix_signature(events):
    return hashlib.sha256(','.join(events).encode()).hexdigest()


def partition(signature, seed):
    # Entire model-visible sequences (including distinct block IDs with the same
    # sequence) stay together. Split before vocabulary fitting/window creation.
    bucket = int(hashlib.sha256(f'{seed}:{signature}'.encode()).hexdigest()[:16], 16) % 1000
    return 'train' if bucket < 700 else 'validation' if bucket < 850 else 'test'


def prepare(path, seed, max_len, caps):
    heaps = {s: [] for s in caps}
    seen_blocks = set()
    signatures = defaultdict(set)
    source_counts = Counter()
    label_counts = {s: Counter() for s in caps}
    source_prefix_counts = Counter()
    truncated = 0
    with zipfile.ZipFile(path) as archive:
        with archive.open(LABEL_MEMBER) as file:
            reader = csv.DictReader(io.TextIOWrapper(file, encoding='utf-8'))
            labels = {}
            for row in reader:
                if row['BlockId'] in labels:
                    raise ValueError('Duplicate block IDs in official labels.')
                if row['Label'] not in ('Normal', 'Anomaly'):
                    raise ValueError('Unrecognized official label.')
                labels[row['BlockId']] = row['Label'] == 'Anomaly'
        with archive.open(TRACE_MEMBER) as file:
            reader = csv.DictReader(io.TextIOWrapper(file, encoding='utf-8'))
            for row in reader:
                block = row['BlockId']
                if block in seen_blocks or block not in labels:
                    raise ValueError('Nonunique or unlabeled block trace.')
                seen_blocks.add(block)
                events = row['Features'].strip('[]').split(',')
                if not events or any(not event.startswith('E') or not event[1:].isdigit() for event in events):
                    raise ValueError('Unexpected event trace format.')
                source_counts['blocks'] += 1
                source_counts['events'] += len(events)
                source_counts['anomaly' if labels[block] else 'normal'] += 1
                truncated += len(events) > max_len
                events = events[:max_len]
                signature = prefix_signature(events)
                split = partition(signature, seed)
                signatures[split].add(signature)
                source_prefix_counts[signature] += 1
                label_counts[split]['anomaly' if labels[block] else 'normal'] += 1
                if split == 'train' and labels[block]:
                    continue  # train only normal; validation/test retain observed labels
                priority = int(hashlib.sha256(f'{seed}:{block}'.encode()).hexdigest(), 16)
                record = {'events': events, 'anomaly': labels[block], 'block': block,
                          'signature': signature, 'original_length': len(row['Features'].strip('[]').split(','))}
                item = (-priority, block, record)
                heap = heaps[split]
                if len(heap) < caps[split]:
                    heapq.heappush(heap, item)
                elif priority < -heap[0][0]:
                    heapq.heapreplace(heap, item)
    if len(labels) != len(seen_blocks):
        raise ValueError('Trace/label coverage mismatch.')
    assert not (signatures['train'] & signatures['validation'] or signatures['train'] & signatures['test'] or
                signatures['validation'] & signatures['test'])
    rows = {s: [item[2] for item in sorted(heap, reverse=True)] for s, heap in heaps.items()}
    vocabulary = {event: index+3 for index, event in enumerate(sorted({e for row in rows['train'] for e in row['events']}))}
    encoded_seen = set()
    encoded_exclusions = {}
    for split in ('train', 'validation', 'test'):
        selected, split_encoded = [], set()
        excluded = 0
        for row in rows[split]:
            row['ids'] = [vocabulary.get(event, UNK) for event in row['events']]
            signature = tuple(row['ids'])
            if signature in encoded_seen:
                excluded += 1
                continue
            selected.append(row)
            split_encoded.add(signature)
        encoded_seen.update(split_encoded)
        rows[split] = selected
        encoded_exclusions[split] = excluded
    audit = {'source': dict(source_counts), 'source_prefix_duplicates_extra': sum(n-1 for n in source_prefix_counts.values()),
             'source_partition_label_counts': {s: dict(c) for s,c in label_counts.items()},
             'source_blocks_truncated_at_max_len': truncated, 'max_len': max_len,
             'split_rule': 'seeded SHA256 of first max_len event IDs; 70/15/15 hash buckets',
             'subset_rule': 'lowest seeded block-ID hashes per partition; normal-only train',
             'encoded_cross_split_collision_exclusions': encoded_exclusions, 'train_event_vocabulary_size': len(vocabulary),
             'selected': {s: {'blocks': len(rs), 'anomaly': sum(r['anomaly'] for r in rs),
                              'normal': sum(not r['anomaly'] for r in rs),
                              'original_events': sum(r['original_length'] for r in rs),
                              'visible_events': sum(len(r['ids']) for r in rs),
                              'truncated_blocks': sum(r['original_length']>max_len for r in rs),
                              'unique_visible_sequences': len({tuple(r['ids']) for r in rs}),
                              'unknown_events': sum(i==UNK for r in rs for i in r['ids'])} for s,rs in rows.items()}}
    for a,b in [('train','validation'),('train','test'),('validation','test')]:
        assert not ({r['block'] for r in rows[a]} & {r['block'] for r in rows[b]})
        assert not ({tuple(r['ids']) for r in rows[a]} & {tuple(r['ids']) for r in rows[b]})
    return rows, vocabulary, audit


class NGram:
    def __init__(self, rows, vocab_size, order):
        self.counts = defaultdict(Counter)
        self.vocab_size, self.order = vocab_size, order
        for row in rows:
            history = [-1]
            for token in row['ids']:
                for size in range(order):
                    self.counts[tuple(history[-size:]) if size else ()][token] += 1
                history.append(token)

    def score(self, row):
        history, losses = [-1], []
        for token in row['ids']:
            for size in range(min(self.order-1,len(history)), -1, -1):
                counts = self.counts.get(tuple(history[-size:]) if size else ())
                if counts:
                    break
            losses.append(-math.log((counts[token]+1)/(sum(counts.values())+self.vocab_size)))
            history.append(token)
        return sum(losses)/len(losses)


def pr_ap(labels, scores):
    groups = defaultdict(lambda:[0,0])
    for label,score in zip(labels,scores):
        groups[score][int(label)] += 1
    tp=fp=0
    positives=sum(labels)
    area=0.0
    for score in sorted(groups,reverse=True):
        negative,positive=groups[score]
        tp+=positive
        fp+=negative
        area+=(positive/max(1,positives))*tp/(tp+fp)
    return min(1.0,max(0.0,area))


def evaluate(validation,test,val_scores,test_scores):
    normal=sorted(score for row,score in zip(validation,val_scores) if not row['anomaly'])
    if not normal:
        raise ValueError('No validation normal blocks.')
    threshold=normal[math.ceil(0.99*len(normal))-1]
    result={'threshold':threshold,'threshold_rule':'validation normal 99th percentile; strict score>threshold'}
    for name,rows,scores in [('validation',validation,val_scores),('test',test,test_scores)]:
        labels=[r['anomaly'] for r in rows]
        tp=fp=tn=fn=0
        for actual,score in zip(labels,scores):
            predicted=score>threshold
            tp+=actual and predicted
            fp+=not actual and predicted
            tn+=not actual and not predicted
            fn+=actual and not predicted
        result[name]={'average_precision':pr_ap(labels,scores),'tp':tp,'fp':fp,'tn':tn,'fn':fn,
                      'precision':tp/max(1,tp+fp),'recall':tp/max(1,tp+fn),'fpr':fp/max(1,fp+tn)}
    return result


def model_scores(model,rows):
    scores=[]
    for offset in range(0,len(rows),32):
        ids=pad_batch([row['ids'] for row in rows[offset:offset+32]])
        losses=leave_one_out_nll(model,ids,chunk_size=128)
        scores.extend(torch.nanmean(losses,dim=1).tolist())
    return scores


def shifted_causal_inputs(ids):
    # The same reserved unused ID is BOS for this independently trained model.
    # It is never an output target in the causal training objective.
    return torch.cat((torch.full_like(ids[:, :1], MASK), ids[:, :-1]), dim=1)


def causal_fit_and_score(train, validation, test, vocab_size, max_len, seed, epochs):
    torch.manual_seed(seed)
    model = EventMLM(vocab_size, max_len)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
    rng = random.Random(seed)
    rows = list(train)
    training = []
    for _ in range(epochs):
        rng.shuffle(rows)
        model.train()
        total = count = 0
        for offset in range(0,len(rows),64):
            ids = pad_batch([r['ids'] for r in rows[offset:offset+64]])
            logits = model(shifted_causal_inputs(ids), causal=True)
            loss = F.cross_entropy(logits.flatten(0, 1), ids.flatten(), ignore_index=PAD)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            valid = int(ids.ne(PAD).sum())
            count += valid
            total += valid * loss.item()
        training.append({'mean_loss':total/count,'selected_tokens':count})
        print(json.dumps({'causal_epoch':len(training),**training[-1]}),flush=True)

    @torch.inference_mode()
    def scores(rows):
        model.eval()
        output = []
        for offset in range(0,len(rows),64):
            ids = pad_batch([r['ids'] for r in rows[offset:offset+64]])
            logits = model(shifted_causal_inputs(ids), causal=True)
            losses = F.cross_entropy(logits.transpose(1, 2), ids, ignore_index=PAD, reduction='none')
            output.extend((losses.sum(dim=1) / ids.ne(PAD).sum(dim=1)).tolist())
        return output
    return evaluate(validation, test, scores(validation), scores(test)), training


def count_vectors(rows, vocab_size):
    vectors = []
    for row in rows:
        counts = Counter(row['ids'])
        vectors.append([counts.get(i, 0) for i in range(UNK, vocab_size)])
    return torch.tensor(vectors, dtype=torch.float32)


def bag_l1_scores(train, rows, vocab_size):
    """Distance to the closest distinct normal count vector; event order is absent."""
    reference = count_vectors(train, vocab_size).unique(dim=0)
    queries = count_vectors(rows, vocab_size)
    return [float(x) for batch in queries.split(128)
            for x in torch.cdist(batch, reference, p=1).amin(dim=1)]


def shuffled(rows, seed):
    """Deterministically permute each complete visible sequence, preserving counts."""
    result = []
    for row in rows:
        ids = list(row['ids'])
        random.Random(f"{seed}:{row['block']}:order-control").shuffle(ids)
        result.append({**row, 'ids': ids})
    return result


def run(args):
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    started=time.perf_counter()
    source=obtain_archive(args.archive,args.download)
    rows,vocabulary,audit=prepare(args.archive,args.seed,args.max_len,
                                  {'train':args.train_blocks,'validation':args.eval_blocks,'test':args.eval_blocks})
    print(json.dumps({'audit':audit}),flush=True)
    model=EventMLM(len(vocabulary)+3,args.max_len)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001)
    generator=torch.Generator().manual_seed(args.seed)
    rng=random.Random(args.seed)
    training=[]
    train=list(rows['train'])
    for epoch in range(args.epochs):
        rng.shuffle(train)
        batches=(pad_batch([r['ids'] for r in train[i:i+64]]) for i in range(0,len(train),64))
        record=train_epoch(model,batches,optimizer,generator)
        training.append(record)
        print(json.dumps({'epoch':epoch+1,**record}),flush=True)
    val_scores=model_scores(model,rows['validation'])
    test_scores=model_scores(model,rows['test'])
    results={'mlm':evaluate(rows['validation'],rows['test'],val_scores,test_scores)}
    for name,order in [('unigram',1),('trigram',3)]:
        baseline=NGram(train,len(vocabulary)+1,order)  # ordinary IDs + UNK output classes
        results[name]=evaluate(rows['validation'],rows['test'],[baseline.score(r) for r in rows['validation']],
                               [baseline.score(r) for r in rows['test']])
    bag_test = bag_l1_scores(train, rows['test'], len(vocabulary)+3)
    results['bag_l1'] = evaluate(
        rows['validation'], rows['test'],
        bag_l1_scores(train, rows['validation'], len(vocabulary)+3), bag_test)
    count_match = {
        'normal_exact_match': sum(not r['anomaly'] and score == 0 for r, score in zip(rows['test'], bag_test)),
        'anomaly_exact_match': sum(r['anomaly'] and score == 0 for r, score in zip(rows['test'], bag_test)),
    }
    # Apply the same perturbation to validation and test, retuning only on the
    # perturbed validation normals. Training remains ordered; this is a
    # sensitivity probe, not a separately trained shuffled baseline.
    results['mlm_inference_shuffled'] = evaluate(
        rows['validation'], rows['test'],
        model_scores(model, shuffled(rows['validation'], args.seed)),
        model_scores(model, shuffled(rows['test'], args.seed)))
    results['causal_transformer'], causal_training = causal_fit_and_score(
        rows['train'], rows['validation'], rows['test'], len(vocabulary)+3,
        args.max_len, args.seed, args.epochs)
    return {'scope':'limited HDFS_v1 public IT-log subset; offline bidirectional MLM, not industrial validation',
            'model':'randomly initialized two-block BERT-style event encoder; MLM only, not full original LogBERT',
            'source_archive':source,'audit':audit,'seed':args.seed,'epochs':args.epochs,
            'torch':torch.__version__,'device':'cpu','threads':1,
            'parameters':sum(p.numel() for p in model.parameters()),
            'baseline_protocol': {'causal_transformer':'same architecture and size with causal mask, separate random-init next-event training and mean next-event NLL', 'bag_l1':'nearest normal L1 event-count vector; includes unknown count and implicitly length',
                                  'mlm_inference_shuffled':'same ordered-trained model, permuted validation and test inputs, distinct validation-only threshold'},
            'training':training,'causal_training':causal_training,'score':'mean leave-one-out masked event NLL; completed first-max_len-event prefixes',
            'test_count_vector_match_to_normal_train':count_match,'results':results,'elapsed_seconds':time.perf_counter()-started}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive',type=Path,default=Path(tempfile.gettempdir())/'equip-public-logbert'/'HDFS_v1.zip')
    parser.add_argument('--download',action='store_true')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--max-len',type=int,default=64)
    parser.add_argument('--epochs',type=int,default=5)
    parser.add_argument('--train-blocks',type=int,default=4000)
    parser.add_argument('--eval-blocks',type=int,default=1000)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if min(args.epochs,args.max_len,args.train_blocks,args.eval_blocks)<1:
        parser.error('positive dimensions required')
    payload=run(args)
    args.out.write_text(json.dumps(payload,indent=2)+'\n',encoding='utf-8')
    print('Saved aggregate results: '+str(args.out),flush=True)
