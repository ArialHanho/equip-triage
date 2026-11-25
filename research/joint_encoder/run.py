"""Prepare a locked comparison, fit/resume named arms, then evaluate one complete seed."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
from tempfile import NamedTemporaryFile
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import joblib
import sklearn
import torch
from torch.nn import functional as F

from equip_triage.joint import JointEncoder, batch, load_model, prepare
from equip_triage.joint.model import PUBLIC_REV
from equip_triage.joint.sources import VERSION
from .audit import inspect
from .baselines import CharEvents, rule_outputs
from .data import generate, manifest
from .metrics import breakdown, calibrate, family_contrast
from .public_base import FILES, assets, file_sha

ROOT = Path(__file__).resolve().parents[2]
BASES = ("rule", "char_joint", "char_log", "char_memo")
NATIVE = ("pooled", "parallel", "unbound", "log_only", "memo_only", "direct", "joint")
PRIMARY = ("rule", "char_joint", "pooled", "direct", "public")
SEEDS = (17, 41, 73)


def read(p):
    return json.loads(p.read_text(encoding="utf-8"))


def write(p, val):
    p.parent.mkdir(parents=True, exist_ok=True)
    store_bytes(
        p,
        (
            json.dumps(
                val, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False
            )
            + "\n"
        ).encode(),
    )


def store_bytes(p, data):
    with NamedTemporaryFile(dir=p.parent, delete=False) as f:
        tmp = Path(f.name)
        f.write(data)
    try:
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def save(p, obj):
    p.parent.mkdir(parents=True, exist_ok=True)
    with NamedTemporaryFile(dir=p.parent, delete=False) as f:
        tmp = Path(f.name)
    try:
        torch.save(obj, tmp)
        os.replace(tmp, p)
    finally:
        tmp.unlink(missing_ok=True)


def fingerprints():
    paths = [
        ROOT / "equip_triage/model.py",
        ROOT / "equip_triage/encoding.py",
        *sorted((ROOT / "equip_triage/joint").glob("*.py")),
        *sorted((ROOT / "research/joint_encoder").glob("*.py")),
        Path(__file__).with_name("protocol.md"),
    ]
    return {str(p.relative_to(ROOT)): file_sha(p) for p in paths}


def splits(profile):
    return ("pilot_train", "pilot") if profile == "pilot" else ("train", "val", "test")


def lock(args):
    parts = splits(args.profile)
    obj = {
        "format": VERSION,
        "profile": args.profile,
        "sources": fingerprints(),
        "data": {part: manifest(part) for part in parts},
        "audit": inspect(parts),
    }
    path = args.work / "manifest.json"
    if path.exists():
        if read(path) != obj:
            raise ValueError(
                "Existing source/data manifest differs; select a fresh workspace"
            )
    else:
        write(path, obj)
    print(
        "Locked", args.profile, {s: obj["data"][s]["rows"] for s in parts}, flush=True
    )


def check(args):
    path = args.work / "manifest.json"
    obj = read(path)
    if obj["profile"] != args.profile or obj["sources"] != fingerprints():
        raise ValueError(
            "Profile or locked sources differ. Use the matching checkout and directory."
        )
    return file_sha(path)


def set_device(device):
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA unavailable. Connect a GPU or explicitly use the CPU pilot."
        )
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return torch.device(device)


def data(split):
    rows = list(generate(split))
    return rows, [prepare(r.packet) for r in rows]


def targets(cases, snaps, device):
    states, target, pl, pm = [], [], [], []
    n = max(1, max(len(s.units) for s in snaps))
    for e, s in zip(cases, snaps):
        domains = [d for _, d in s.registry]
        states.append(e.states)
        target.append(domains.index(e.gold_action) if e.gold_action in domains else 2)
        ids = {rid: i for i, u in enumerate(s.units) for rid in u.ids}
        pl.append([ids.get(r, n) for r in e.logs])
        pm.append([ids.get(r, n) for r in e.memos])
    return [
        torch.tensor(x, dtype=torch.long, device=device)
        for x in (states, target, pl, pm)
    ]


def loss_for(out, t, mode):
    states, action, log, memo = t
    direct = F.cross_entropy(out["direct"], action)
    if mode == "direct":
        return direct
    loss = F.cross_entropy(out["state"].flatten(0, 1), states.flatten()) + 0.3 * direct
    if mode != "memo_only":
        loss = loss + 0.35 * F.cross_entropy(out["log"].flatten(0, 1), log.flatten())
    if mode != "log_only":
        loss = loss + 0.35 * F.cross_entropy(out["memo"].flatten(0, 1), memo.flatten())
    return loss


@torch.inference_mode()
def outputs(model, snapshots, device, size=8):
    model.eval()
    result = []
    for start in range(0, len(snapshots), size):
        sub = snapshots[start : start + size]
        out = model(batch(sub).to(device))
        ps = out["state"].softmax(-1).cpu().numpy()
        direct = out["direct"].softmax(-1).cpu().numpy()
        pointers = [out[key].argmax(-1).cpu().tolist() for key in ("log", "memo")]
        for i, s in enumerate(sub):
            refs = []
            for j in range(2):
                refs.append(
                    tuple(
                        s.units[p[i][j]].ids if p[i][j] < len(s.units) else ()
                        for p in pointers
                    )
                )
            result.append({"state": ps[i], "direct": direct[i], "refs": refs})
    return result


def cal(profile, cases, snaps, p, direct=False):
    if profile == "main":
        return calibrate(cases, snaps, p, direct)
    return {"threshold": 0.5, "validation": []}


def fit(args):
    h = check(args)
    if args.profile == "main" and args.device != "cuda":
        raise ValueError(
            "Main fitting requires CUDA; use --profile pilot for the CPU smoke."
        )
    device = set_device(args.device)
    arm, seed = args.arm, args.seed
    folder = args.work / "models" / f"{arm}-{seed}"
    folder.mkdir(parents=True, exist_ok=True)
    spec = {
        "format": VERSION,
        "source_lock": h,
        "arm": arm,
        "seed": seed,
        "profile": args.profile,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "epochs": 2 if args.profile == "pilot" else 3,
        "logical_batch": 16,
        "microbatch": args.microbatch,
        "precision": "fp32",
        "width": 32,
    }
    if (
        args.microbatch < 1
        or 16 % args.microbatch
        or args.save_every < 1
        or (args.max_steps is not None and args.max_steps < 1)
    ):
        raise ValueError("Microbatch must divide 16. Step limits must be positive.")
    done = folder / "fit.json"
    if done.exists():
        prev = read(done)
        if (
            prev["spec"] != spec
            or file_sha(folder / "model.pt") != prev["weights_sha256"]
        ):
            raise ValueError("Prior fit metadata or checkpoint differs")
        print("Already complete", arm, seed, flush=True)
        return
    train, val, *_ = splits(args.profile)
    rows, snaps = data(train)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    public = arm == "public"
    cfg = {
        "mode": "joint" if public else arm,
        "width": 32,
        "backbone": "distilmbert" if public else "bytes",
    }
    base = assets(args.public_assets) if public else None
    model = JointEncoder(**cfg, assets=base).to(device)
    if public:
        token_dir = folder / "tokenizer"
        token_dir.mkdir(exist_ok=True)
        for name in FILES:
            if name != "model.safetensors":
                shutil.copyfile(base / name, token_dir / name)
        backbone = list(model.rows.encoder.parameters())
        keys = {id(p) for p in backbone}
        groups = [
            {"params": backbone, "lr": 0.00002},
            {
                "params": [p for p in model.parameters() if id(p) not in keys],
                "lr": 0.002,
            },
        ]
    else:
        groups = model.parameters()
    opt = torch.optim.AdamW(groups, lr=0.002)
    prog = {
        "epoch": 0,
        "next_batch": 0,
        "steps": 0,
        "examples": 0,
        "history": [],
        "epoch_sum": 0.0,
        "seconds": 0.0,
    }
    cp = folder / "resume.pt"
    if cp.exists():
        saved = torch.load(cp, map_location="cpu", weights_only=True)
        if saved["spec"] != spec:
            raise ValueError("Resume configuration differs")
        model.load_state_dict(saved["model"])
        opt.load_state_dict(saved["optimizer"])
        prog = saved["progress"]
        torch.set_rng_state(saved["rng"])
        if device.type == "cuda":
            torch.cuda.set_rng_state(saved["cuda_rng"], device)
        del saved
    if args.max_steps is not None and prog["steps"] >= args.max_steps:
        print("Already at requested pause", prog["steps"], flush=True)
        return
    start = time.perf_counter()

    def persist():
        nonlocal start
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        prog["seconds"] += time.perf_counter() - start
        save(
            cp,
            {
                "spec": spec,
                "model": model.state_dict(),
                "optimizer": opt.state_dict(),
                "progress": prog,
                "rng": torch.get_rng_state(),
                "cuda_rng": torch.cuda.get_rng_state(device)
                if device.type == "cuda"
                else None,
            },
        )
        start = time.perf_counter()

    for epoch in range(prog["epoch"], spec["epochs"]):
        model.train()
        indices = torch.randperm(
            len(rows), generator=torch.Generator().manual_seed(seed + 1009 * epoch)
        )
        batches = indices.split(16)
        for j in range(prog["next_batch"], len(batches)):
            ids = batches[j].tolist()
            opt.zero_grad(set_to_none=True)
            for k in range(0, len(ids), args.microbatch):
                part = ids[k : k + args.microbatch]
                ss = [snaps[n] for n in part]
                rr = [rows[n] for n in part]
                out = model(batch(ss).to(device))
                loss = loss_for(out, targets(rr, ss, device), cfg["mode"])
                (loss * (len(part) / len(ids))).backward()
                prog["epoch_sum"] += float(loss.detach()) * len(part)
            opt.step()
            prog["steps"] += 1
            prog["examples"] += len(ids)
            prog["next_batch"] = j + 1
            if args.max_steps is not None and prog["steps"] >= args.max_steps:
                persist()
                print("Paused after", prog["steps"], "steps", flush=True)
                return
            if prog["steps"] % args.save_every == 0:
                persist()
        prog["history"].append(prog["epoch_sum"] / len(rows))
        prog.update(epoch=epoch + 1, next_batch=0, epoch_sum=0.0)
        persist()
    r, s = data(val) if args.profile == "main" else (None, None)
    calibration = (
        cal("main", r, s, outputs(model, s, device, args.microbatch), arm == "direct")
        if r
        else {"threshold": 0.5, "validation": []}
    )
    product = folder / "model.pt"
    save(
        product,
        {
            "format": VERSION,
            "config": cfg,
            "model": model.state_dict(),
            "threshold": calibration["threshold"],
            "public_revision": PUBLIC_REV if public else None,
        },
    )
    write(
        done,
        {
            "spec": spec,
            "progress": prog,
            "calibration": calibration,
            "weights_sha256": file_sha(product),
            "parameters": sum(p.numel() for p in model.parameters()),
            "python": platform.python_version(),
            "finished_utc": datetime.now(timezone.utc).isoformat(),
        },
    )
    print(
        "Fitted",
        arm,
        seed,
        "steps",
        prog["steps"],
        "seconds",
        round(prog["seconds"], 2),
        flush=True,
    )


def fit_baselines(args):
    h = check(args)
    folder = args.work / "baselines"
    folder.mkdir(exist_ok=True)
    done = folder / "fit.json"
    if done.exists():
        if read(done)["source_lock"] != h:
            raise ValueError("Baseline source lock differs")
        return
    train, val, *_ = splits(args.profile)
    tr, ts = data(train)
    vr, vs = data(val) if args.profile == "main" else ([], [])
    frozen = {}
    for arm in BASES:
        t = time.perf_counter()
        if arm == "rule":
            p = [rule_outputs(s) for s in vs]
            digest = None
        else:
            model = CharEvents(arm.split("_")[1]).fit(ts, tr)
            path = folder / (arm + ".joblib")
            joblib.dump(model, path, compress=3)
            digest = file_sha(path)
            p = model.outputs(vs) if vs else []
        frozen[arm] = {
            "calibration": cal(args.profile, vr, vs, p),
            "model_sha256": digest,
            "fit_validation_seconds": time.perf_counter() - t,
        }
    write(done, {"source_lock": h, "sklearn": sklearn.__version__, "arms": frozen})


def evaluate(args):
    h = check(args)
    if args.profile == "main" and args.device != "cuda":
        raise ValueError("Main evaluation requires the Colab CUDA run.")
    device = set_device(args.device)
    final = args.work / f"evaluation-{args.seed}.json"
    if final.exists():
        raise ValueError(
            "Evaluation exists. Keep the frozen output or use a fresh replication."
        )
    names = (*NATIVE, "public") if args.profile == "main" else NATIVE
    bas = read(args.work / "baselines/fit.json")
    if bas["source_lock"] != h:
        raise ValueError("Baseline calibration lock differs")
    fits = {}
    for arm in names:
        p = args.work / "models" / f"{arm}-{args.seed}"
        f = read(p / "fit.json")
        if (
            f["spec"]["source_lock"] != h
            or f["spec"]["device"] != str(device)
            or file_sha(p / "model.pt") != f["weights_sha256"]
        ):
            raise ValueError("A calibration or checkpoint differs")
        fits[arm] = (p, f)
    calibration = {a: v["calibration"] for a, v in bas["arms"].items()}
    calibration.update({a: f["calibration"] for a, (p, f) in fits.items()})
    selectable = (
        PRIMARY
        if args.profile == "main"
        else tuple(a for a in PRIMARY if a != "public")
    )

    def val_f1(a):
        curve = calibration[a]["validation"]
        selected = calibration[a]["threshold"]
        return next(
            (float(item[2]) for item in curve if item[0] == selected),
            calibration[a].get("hold_all_f1", 0.0),
        )

    # Never select the pilot comparator on pilot scores.
    comparator = (
        sorted(selectable, key=lambda a: (-val_f1(a), a))[0]
        if args.profile == "main"
        else "char_joint"
    )
    write(
        args.work / f"calibrated-{args.seed}.json",
        {"source_lock": h, "comparator": comparator, "arms": calibration},
    )
    cases, snaps = data(splits(args.profile)[-1])
    result, times = {}, {}
    for arm in (*BASES, *names):
        start = time.perf_counter()
        if arm == "rule":
            pred = [rule_outputs(s) for s in snaps]
        elif arm in BASES:
            path = args.work / "baselines" / (arm + ".joblib")
            if file_sha(path) != bas["arms"][arm]["model_sha256"]:
                raise ValueError("Baseline artifact changed")
            pred = joblib.load(path).outputs(snaps)
        else:
            model, _ = load_model(fits[arm][0] / "model.pt", device)
            pred = outputs(model, snaps, device, 8)
            del model
        times[arm] = time.perf_counter() - start
        t = calibration[arm]["threshold"]
        fixed = breakdown(cases, snaps, pred, 0.5, arm == "direct")
        selected = (
            breakdown(cases, snaps, pred, t, arm == "direct") if t != 0.5 else fixed
        )
        if arm == "direct":
            for reports in (fixed, selected):
                for row in reports.values():
                    for key in (
                        "state_cm",
                        "state_macro_f1",
                        "source_hits",
                        "null_hits",
                        "both_sources",
                    ):
                        row.pop(key, None)
        result[arm] = {"fixed": fixed, "selected": selected, "threshold": t}
    pair = {
        a: family_contrast(result["joint"]["selected"], result[a]["selected"])
        for a in result
        if a != "joint"
    }
    tested = pair[comparator]
    write(
        final,
        {
            "profile": args.profile,
            "split": splits(args.profile)[-1],
            "seed": args.seed,
            "source_lock": h,
            "device": str(device),
            "results": result,
            "load_and_score_seconds": times,
            "contrasts": pair,
            "comparator": comparator,
            "primary_evaluated": args.profile == "main",
            "point_pass": all(z["point_pass"] for z in tested.values())
            if args.profile == "main"
            else None,
            "interval_pass": all(z["interval_pass"] for z in tested.values())
            if args.profile == "main"
            else None,
        },
    )
    print("Evaluation", args.profile, args.seed, final.name, flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--profile", choices=("pilot", "main"), required=True)
    parser.add_argument(
        "--public-assets",
        type=Path,
        default=Path("artifacts/public-distilmbert") / PUBLIC_REV,
    )
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("prepare", "fetch-public", "fit-baselines"):
        commands.add_parser(name)
    train = commands.add_parser("fit")
    train.add_argument("--arm", choices=(*NATIVE, "public"), required=True)
    train.add_argument("--microbatch", type=int, default=4)
    train.add_argument("--save-every", type=int, default=100)
    train.add_argument("--max-steps", type=int)
    ev = commands.add_parser("evaluate")
    for cmd in (train, ev):
        cmd.add_argument("--device", choices=("cpu", "cuda"), required=True)
        cmd.add_argument("--seed", type=int, choices=SEEDS, default=17)
    args = parser.parse_args()
    if args.command == "fetch-public":
        print("Verified", assets(args.public_assets, download=True), flush=True)
    else:
        {
            "prepare": lock,
            "fit": fit,
            "fit-baselines": fit_baselines,
            "evaluate": evaluate,
        }[args.command](args)


if __name__ == "__main__":
    main()
