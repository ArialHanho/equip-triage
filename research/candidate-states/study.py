"""Reproduce the candidate-state benchmark or route bilingual packets locally."""

import argparse
import csv
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time
import numpy as np
import sklearn
import tokenizers
import torch
from threadpoolctl import threadpool_limits
from data import ACTIONS, DOMAINS, ROOT, STATES, Packet, file_hash, generate, summary
from encoder import MODES, encode, fit_net, train_tokenizer
from evaluation import LANGS, evaluate, gates, resample
from linear import packet_models, candidate_models
from routing import Output, prepare, rule_output, state_output

SEEDS = (41, 42, 43)
PACKET_CHAR = ("char_direct_raw", "char_direct", "char_mentions", "char_mentions_bound")
CANDIDATE_CHAR = ("char_states_global", "char_states_bound", "char_states_bound_wta")
DISPLAY = {
    "en": (
        "Check measurement system",
        "Check data collection",
        "Check recipe setup",
        "Check equipment",
        "Hold for review",
    ),
    "ko": (
        "측정계 점검",
        "데이터 수집 점검",
        "레시피 설정 점검",
        "장비 구동부 점검",
        "판정 보류",
    ),
}


def dumps(obj):
    return json.dumps(obj, ensure_ascii=False, sort_keys=True, indent=2) + "\n"


def verify_sources():
    manifest = json.loads((ROOT / "manifest.json").read_text())
    for name, expected in manifest["files"].items():
        if file_hash(name) != expected:
            raise ValueError(f"Frozen file changed: {name}")
    files = (
        "manifest.json",
        "study.py",
        "evaluation.py",
        "encoder.py",
        "linear.py",
        "routing.py",
        "../public-logbert/bert_mlm.py",
    )
    return manifest, {f: file_hash(f) for f in files}


def load_split(name, manifest):
    rows = generate(name)
    if summary(rows) != manifest["splits"][name]:
        raise ValueError(f"Split mismatch: {name}")
    return rows, [prepare(r.packet) for r in rows]


def fit_group(group, train, rows, tok, seed):
    if group == "rule":
        return {"scoped_rule": rule_output}, None
    if group in ("char_packet", "char_candidate"):
        scorer = (packet_models if group == "char_packet" else candidate_models)(
            rows, train
        )
        names = PACKET_CHAR if group == "char_packet" else CANDIDATE_CHAR
        return {name: lambda batch, n=name: scorer(batch, n) for name in names}, None
    scorer, active = fit_net(rows, train, tok, group, seed)
    modes = ["bert_" + group]
    if group == "flat":
        modes.insert(0, "bert_flat_raw")
    if group == "bound_query":
        modes.append("bert_bound_query_wta")
    return {name: lambda batch, n=name: scorer(batch, n) for name in modes}, active


def fit_all(train, tr, val, vr, tok):
    jobs = [(g, None) for g in ("rule", "char_packet", "char_candidate")]
    jobs.extend((g, s) for s in SEEDS for g in MODES)
    scores = []
    configs = []
    ranking = {}
    for group, seed in jobs:
        start = time.process_time()
        fitted, active = fit_group(group, train, tr, tok, seed)
        fit_cost = time.process_time() - start
        for name, score in fitted.items():
            operating = gates(val, score(vr))
            configs.append(
                dict(
                    method=name,
                    group=group,
                    seed=seed,
                    active_parameters=active,
                    fit_cpu_s=fit_cost,
                    **operating,
                )
            )
            scores.append(score)
            ranking.setdefault(name, []).append(operating["validation_f1"])
        print(f"Fitted {group} (seed {seed}).", flush=True)
    return (
        scores,
        configs,
        max(ranking, key=lambda name: statistics.mean(ranking[name])),
    )


def latency(score, examples, rows):
    groups = {}
    for i, r in enumerate(examples):
        groups.setdefault((r.family, r.lang), []).append(i)
    ix = [g[q * len(g) // 4] for q in range(4) for g in groups.values()]
    for i in ix[:4]:
        score([prepare(examples[i].packet)])
    ms = []
    for i in ix:
        start = time.perf_counter_ns()
        score([prepare(examples[i].packet)])
        ms.append((time.perf_counter_ns() - start) / 1e6)
    return dict(
        n=len(ms), p50_ms=float(np.median(ms)), p95_ms=float(np.percentile(ms, 95))
    )


def write_csv(path, items):
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(items[0]))
        writer.writeheader()
        writer.writerows(items)


def benchmark(out):
    manifest, source = verify_sources()
    if out.exists() and any(out.iterdir()):
        raise ValueError("Choose a fresh output directory.")
    out.mkdir(exist_ok=True, parents=True)
    train, tr = load_split("train", manifest)
    val, vr = load_split("validation", manifest)
    tok = train_tokenizer(tr)
    scores, configs, selected = fit_all(train, tr, val, vr, tok)
    lock = dict(
        sources=source,
        selected=selected,
        configs=configs,
        vocab=tok.get_vocab_size(),
        utc=datetime.now(timezone.utc).isoformat(),
        python=platform.python_version(),
        torch=torch.__version__,
        sklearn=sklearn.__version__,
        numpy=np.__version__,
        tokenizers=tokenizers.__version__,
        threads=1,
    )
    lock_path = out / "validation.json"
    lock_path.write_text(dumps(lock), encoding="utf-8")
    # The official evaluation first materializes the challenge here.
    challenge, cr = load_split("challenge", manifest)
    for e, row in zip(challenge, cr):
        if (e.mode != "incomplete") != row.complete:
            raise ValueError(f"Binding audit failed: {e.case}, {e.pattern}")
    oracle = state_output(cr, np.eye(3)[[e.states for e in challenge]])
    if not all(ACTIONS[a] == e.gold for a, e in zip(oracle.action, challenge)):
        raise ValueError("Oracle disagrees with gold.")
    results = []
    table = []
    oracle_spec = dict(method="oracle_states", seed=None, utility=0.0, safety=0.0)
    for score, c in [*zip(scores, configs), (None, oracle_spec)]:
        outp = score(cr) if score else oracle
        measured = evaluate(challenge, cr, outp, c)
        cpu = latency(score, challenge, cr) if score else None
        run = dict(method=c["method"], seed=c["seed"], cpu=cpu, **measured)
        results.append(run)
        for op, v in measured["operating_points"].items():
            for l in LANGS:
                m = {k: val for k, val in v["metrics"][l].items() if k != "confusion"}
                pair = v["pairs"]["one_vs_two/" + l]
                table.append(
                    dict(
                        method=c["method"],
                        seed=c["seed"],
                        gate=op,
                        cutoff=v["threshold"],
                        lang=l,
                        **m,
                        pair_correct=pair["correct"],
                        pair_total=pair["n"],
                        p50_ms=cpu["p50_ms"] if cpu else None,
                        p95_ms=cpu["p95_ms"] if cpu else None,
                    )
                )
    ci = resample(results)
    result = dict(
        validation_sha256=hashlib.sha256(lock_path.read_bytes()).hexdigest(),
        selected=selected,
        splits=manifest["splits"],
        unique_normalized_challenge=len(set(cr)),
        runs=results,
        intervals=ci,
    )
    (out / "aggregates.json.gz").write_bytes(
        gzip.compress(
            json.dumps(result, ensure_ascii=False, separators=(",", ":")).encode(),
            compresslevel=9,
            mtime=0,
        )
    )
    write_csv(out / "metrics.csv", table)
    write_csv(
        out / "contrasts.csv",
        [
            {k: v for k, v in r.items() if k != "ci95"}
            | dict(low=r["ci95"][0], high=r["ci95"][1])
            for r in ci["contrasts"]
        ],
    )
    print(
        f"Evaluated {len(results)} method/seed configurations on {len(challenge)} packets."
    )


def packet_from_dict(obj):
    if not isinstance(obj, dict):
        raise ValueError("Expected a packet object.")
    candidates = obj.get("candidates")
    if not isinstance(candidates, list) or len(candidates) != 2:
        raise ValueError("Supply exactly two candidate records.")
    if not all(isinstance(item, dict) for item in candidates):
        raise ValueError("Candidate records must be objects.")
    memo = obj.get("memo")
    if not isinstance(memo, str):
        raise ValueError("memo must be a string.")

    def field(item, key):
        val = item.get(key)
        if val is None:
            return ""
        if not isinstance(val, str):
            raise ValueError(f"{key} must be a string or null.")
        return val

    return Packet(
        tuple(field(c, "domain") for c in candidates),
        tuple(field(c, "tag") for c in candidates),
        memo,
    )


def predictor(args):
    manifest, source = verify_sources()
    lock = json.loads(args.lock.read_text())
    if lock["sources"] != source:
        raise ValueError("Prediction code or data differs from the calibration lock.")
    name = args.method or lock["selected"]
    seed = args.seed if name.startswith("bert_") else None
    c = next(
        (v for v in lock["configs"] if (v["method"], v["seed"]) == (name, seed)), None
    )
    if not c:
        raise ValueError(f"No calibrated method {name} at seed {seed}.")
    train, tr = load_split("train", manifest)
    tok = train_tokenizer(tr)
    scores, _ = fit_group(c["group"], train, tr, tok, seed)

    def predict(packet):
        row = prepare(packet)
        cutoff = c[args.gate]
        # All user-facing routes require complete inputs, including raw research baselines.
        if not row.complete:
            return dict(
                method=name,
                action="hold",
                display=DISPLAY[args.lang][4],
                reason="incomplete_or_ambiguous_binding",
                candidate_states=None,
            )
        try:
            encode(tok, row)
        except ValueError:
            return dict(
                method=name,
                action="hold",
                display=DISPLAY[args.lang][4],
                reason="line_too_long",
                candidate_states=None,
            )
        output = scores[name]([row])
        uncertain = output.confidence[0] < cutoff
        raw = int(output.action[0])
        final = 4 if uncertain else raw
        states = (
            None
            if output.states is None
            else [
                dict(domain=d, state=STATES[s])
                for d, s in zip(packet.domains, output.states[0])
            ]
        )
        return dict(
            method=name,
            seed=seed,
            action=ACTIONS[final],
            display=DISPLAY[args.lang][final],
            reason="confidence_gate" if uncertain else "routing_policy",
            raw_action=ACTIONS[raw],
            candidate_states=states,
            gate=args.gate,
            threshold=cutoff,
            confidence=float(output.confidence[0]),
        )

    return predict


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)
    bench = sub.add_parser("benchmark")
    bench.add_argument("--out", required=True, type=Path)
    demo = sub.add_parser("predict")
    demo.add_argument("--lock", type=Path, default=ROOT / "run" / "validation.json")
    demo.add_argument("--candidate", action="append", metavar="DOMAIN:TAG")
    demo.add_argument(
        "--line", action="append", help="One memo clause; repeat for the second line."
    )
    demo.add_argument(
        "--jsonl",
        action="store_true",
        help="Read packets from stdin; fit the chosen scorer once.",
    )
    demo.add_argument("--lang", choices=LANGS, default="en")
    demo.add_argument("--method")
    demo.add_argument("--seed", type=int, choices=SEEDS, default=42)
    demo.add_argument("--gate", choices=("utility", "safety"), default="utility")
    args = p.parse_args()
    torch.set_num_threads(1)
    with threadpool_limits(limits=1):
        if args.command == "benchmark":
            benchmark(args.out)
            return
        if args.jsonl:
            if args.candidate or args.line:
                p.error("Use --jsonl or individual packet arguments.")
            predict = predictor(args)
            for line in sys.stdin:
                if line.strip():
                    print(
                        json.dumps(
                            predict(packet_from_dict(json.loads(line))),
                            ensure_ascii=False,
                        )
                    )
        else:
            if len(args.candidate or []) != 2 or not args.line:
                p.error("Supply two --candidate records and at least one --line.")
            checks = [s.partition(":") for s in args.candidate]
            if not all(sep for _, sep, _ in checks):
                p.error("Use DOMAIN:TAG for each candidate.")
            packet = Packet(
                tuple(c[0] for c in checks),
                tuple(c[2] for c in checks),
                "\n".join(args.line),
            )
            print(json.dumps(predictor(args)(packet), ensure_ascii=False))


if __name__ == "__main__":
    main()
