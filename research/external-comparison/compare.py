"""Offline paired equipment-order comparison. Only --fetch contacts publishers."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import tempfile
import urllib.request
import sys
import time

import torch
from torch.nn import functional as F
from transformers import (AutoTokenizer, AutoModelForSequenceClassification, AutoConfig,
                          AutoModelForCausalLM)

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "order-aware"))
from study import Counts, Precedence, Trigram, EVENTS, operating_point, score_model, train

MODELS = {
    'tiny': ('google/bert_uncased_L-2_H-128_A-2', '30b0a37ccaaa32f332884b96992754e246e48c5f'),
    'mini': ('google/bert_uncased_L-4_H-256_A-4', '387825ce42dbb39b87911cdf8e383ee3b25184f8'),
    'smol': ('HuggingFaceTB/SmolLM2-135M-Instruct', '12fd25f77366fa6b3b4b768ec3050bf629380bac')
}
METHODS = ['counts','trigram','precedence','mlm','order','milestone_order',
           'tiny_pretrained','tiny_random','mini_pretrained','mini_random','smol','tiny_milestones','mini_milestones']
SEED = 42
ROOT = Path(__file__).resolve().parent


def generate():
    rng = random.Random(241125)
    seen = set()
    splits = []
    for split, count in [('train',384), ('validation',128), ('test',192)]:
        pairs = []
        while len(pairs) < count:
            i = len(pairs)
            layout = [4,5,6,7] if i % 2 == 0 else [5,4,6,7]
            norm, positions = [3], []
            for event in layout:
                norm.extend(rng.choices(range(8,12), k=rng.randint(1,4)))
                positions.append(len(norm))
                norm.append(event)
            norm.append(12)
            held = split == 'test' and i % 4 >= 2
            a,b = positions[2:] if held else positions[1:3]
            bad = norm.copy()
            bad[a],bad[b] = bad[b],bad[a]
            normal,corrupt = tuple(norm),tuple(bad)
            if normal in seen or corrupt in seen:
                continue
            seen.update([normal,corrupt])
            assert Counter(normal) == Counter(corrupt)
            pairs.append((normal,corrupt,'held_release' if held else 'seen_scan'))
        splits.append(pairs)
    return splits


def text_trace(row):
    return ' '.join(EVENTS[i].lower() for i in row)


def external_path(key, cache, fetch):
    name, rev = MODELS[key]
    folder = cache/name/rev
    manifest = json.loads((ROOT/'sources.json').read_text())[name]
    if fetch:
        folder.mkdir(parents=True, exist_ok=True)
        metadata_url = f'https://huggingface.co/api/models/{name}/revision/{rev}'
        with urllib.request.urlopen(metadata_url, timeout=30) as response:
            published = json.load(response)
        if published['sha'] != rev:
            raise ValueError('Unexpected model revision')
        allowed = {'config.json', 'vocab.txt', 'tokenizer.json', 'tokenizer_config.json',
                   'special_tokens_map.json', 'merges.txt', 'vocab.json', manifest['file']}
        for filename in sorted(allowed & {f['rfilename'] for f in published['siblings']}):
            target = folder/filename
            if target.is_file():
                continue
            with tempfile.NamedTemporaryFile(dir=folder, delete=False) as temp:
                partial = Path(temp.name)
                try:
                    url = f'https://huggingface.co/{name}/resolve/{rev}/{filename}'
                    with urllib.request.urlopen(url, timeout=120) as response:
                        shutil.copyfileobj(response, temp)
                except Exception:
                    partial.unlink(missing_ok=True)
                    raise
            partial.replace(target)
    with (folder/manifest['file']).open('rb') as weights:
        if hashlib.file_digest(weights, 'sha256').hexdigest() != manifest['sha256']:
            raise ValueError(f'Checkpoint checksum mismatch in {folder}')
    return str(folder)


def encode(tok, rows):
    return tok([text_trace(x) for x in rows],padding=True,return_tensors='pt')


def fit_bert(path, training, pretrained):
    tok = AutoTokenizer.from_pretrained(path,local_files_only=True)
    cfg = AutoConfig.from_pretrained(path,local_files_only=True,num_labels=2)
    if pretrained:
        model = AutoModelForSequenceClassification.from_pretrained(path,config=cfg,local_files_only=True)
    else:
        model = AutoModelForSequenceClassification.from_config(cfg)
    # Reset the classifier identically within each pretrained/random ablation.
    with torch.no_grad():
        gen=torch.Generator().manual_seed(SEED)
        model.classifier.weight.normal_(0,cfg.initializer_range,generator=gen)
        model.classifier.bias.zero_()
    opt=torch.optim.AdamW(model.parameters(),lr=0.0001)
    rng=random.Random(SEED)
    indices=list(range(len(training)))
    model.train()
    for _ in range(10):
        rng.shuffle(indices)
        for i in range(0,len(indices),64):
            pairs=[training[j] for j in indices[i:i+64]]
            rows=[p[c] for c in (0,1) for p in pairs]
            labels=torch.tensor([c for c in (0,1) for p in pairs])
            logits=model(**encode(tok,rows)).logits
            opt.zero_grad()
            F.cross_entropy(logits,labels).backward()
            opt.step()
    model.eval()
    def scores(rows):
        output=[]
        with torch.inference_mode():
            for i in range(0,len(rows),16):
                logits=model(**encode(tok,rows[i:i+16])).logits
                output.extend((logits[:,1]-logits[:,0]).tolist())
        return output
    return scores,sum(p.numel() for p in model.parameters())


def frozen_llm(path, training):
    tok=AutoTokenizer.from_pretrained(path,local_files_only=True)
    tok.pad_token=tok.eos_token
    model=AutoModelForCausalLM.from_pretrained(path,local_files_only=True,dtype=torch.float32).eval()
    # Train-only examples in balanced ABBA order; no hidden grammar instructions.
    prefix='Classify the equipment event order. A means normal; B means faulty. Reply with one letter.\n'
    for i, label in [(0,0),(0,1),(1,1),(1,0)]:
        prefix+=text_trace(training[i][label])+' => '+('B' if label else 'A')+'\n'
    label_ids=[tok.encode(x,add_special_tokens=False) for x in ['A','B']]
    assert all(len(s)==1 for s in label_ids)
    def scores(rows):
        output=[]
        with torch.inference_mode():
            for i in range(0,len(rows),8):
                dialogs=[[{'role':'user','content':prefix+text_trace(row)+' =>'}] for row in rows[i:i+8]]
                texts=[tok.apply_chat_template(d,tokenize=False,add_generation_prompt=True) for d in dialogs]
                batch=tok(texts,padding=True,add_special_tokens=False,return_tensors='pt')
                # Select the final actual token of each right-padded prompt. No generation.
                last=batch.attention_mask.sum(1)-1
                states=model.model(**batch,use_cache=False).last_hidden_state
                logits=model.lm_head(states[torch.arange(len(texts)),last])
                output.extend((logits[:,label_ids[1][0]]-logits[:,label_ids[0][0]]).tolist())
        return output
    return scores,sum(p.numel() for p in model.parameters())


def fit(method, training, cache, fetch):
    normal=[p[0] for p in training]
    if method in ('counts','trigram','precedence'):
        model={'counts':Counts,'trigram':Trigram,'precedence':Precedence}[method](normal)
        return lambda rows:[model(r) for r in rows],0
    if method in ('mlm','order','milestone_order'):
        # Learn which events occur exactly once in every normal training session.
        # Retain START, END and recurring milestones while removing variable telemetry.
        keep=set.intersection(*[{t for t,n in Counter(r).items() if n==1} for r in normal])
        def project(row):
            return tuple(t for t in row if t in keep) if method=='milestone_order' else row
        train_pairs=[(project(a),project(b),k) for a,b,k in training]
        model=train(train_pairs,SEED,method!='mlm')
        return lambda rows:score_model(model,[project(r) for r in rows],method!='mlm'),sum(p.numel() for p in model.parameters())
    key=method.split('_')[0]
    path=external_path(key,cache,fetch)
    if key == 'smol':
        return frozen_llm(path,training)
    if method.endswith('_milestones'):
        keep=set.intersection(*[{t for t,n in Counter(r).items() if n==1} for r in normal])
        project=lambda row:tuple(t for t in row if t in keep)
        projected=[(project(a),project(b),k) for a,b,k in training]
        scorer,params=fit_bert(path,projected,True)
        return lambda rows:scorer([project(row) for row in rows]),params
    return fit_bert(path,training,method.endswith('_pretrained'))


def evaluate(pairs, values, threshold):
    groups={}
    for name in ['all','seen_scan','held_release','normal_clamp_first','normal_zero_first']:
        indices=[i for i,p in enumerate(pairs) if name=='all' or name==p[2] or
                 (name=='normal_clamp_first' and p[0].index(4)<p[0].index(5)) or
                 (name=='normal_zero_first' and p[0].index(5)<p[0].index(4))]
        normal=[values[2*i] for i in indices]
        bad=[values[2*i+1] for i in indices]
        ranked=Counter(bad)
        negatives=Counter(normal)
        tp=fp=0
        ap=0.
        for s in sorted(set(ranked)|set(negatives),reverse=True):
            tp+=ranked[s]
            fp+=negatives[s]
            ap+=ranked[s]/len(indices)*tp/(tp+fp)
        groups[name]=dict(pairs=len(indices),tp=sum(v>threshold for v in bad),fp=sum(v>threshold for v in normal),
                          pair_wins=sum(b>a for a,b in zip(normal,bad)),pair_ties=sum(b==a for a,b in zip(normal,bad)),
                          average_precision=min(ap,1.))
    return groups


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--method',choices=METHODS,required=True)
    parser.add_argument('--cache',type=Path,default=Path('/tmp/equip-external'))
    parser.add_argument('--fetch',action='store_true')
    parser.add_argument('--seed',type=int,default=42,help='Training initialization and shuffle seed; data remain fixed')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.cache.resolve().is_relative_to(ROOT.parents[1]):
        parser.error('Keep downloaded weights outside the repository')
    global SEED
    SEED=args.seed
    torch.set_num_threads(1)
    torch.manual_seed(SEED)
    training,val,pairs=generate()
    start=time.process_time()
    scorer,params=fit(args.method,training,args.cache,args.fetch)
    fit_cpu=time.process_time()-start
    calibration=scorer([p[0] for p in val])
    threshold=operating_point(calibration)
    start=time.process_time()
    values=scorer([r for p in pairs for r in p[:2]])
    cpu=time.process_time()-start
    result=dict(method=args.method,seed=SEED,split_sizes=[len(training),len(val),len(pairs)],
                split_digest=hashlib.sha256(repr([training,val,pairs]).encode()).hexdigest(),
                parameters=params,threshold=threshold,validation_fp=sum(s>threshold for s in calibration),
                fit_and_load_cpu_seconds=fit_cpu,test_scoring_cpu_seconds=cpu,cpu_traces_per_second=2*len(pairs)/cpu,
                metrics=evaluate(pairs,values,threshold))
    args.out.write_text(json.dumps(result,indent=2)+'\n')
    print(args.method,result['metrics']['all'],flush=True)


if __name__=='__main__':
    main()
