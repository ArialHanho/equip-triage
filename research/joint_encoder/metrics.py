"""Packet- and family-level evaluation of selective decisions and source pointers."""

from collections import Counter, defaultdict
import numpy as np
from equip_triage.encoding import DOMAINS
from equip_triage.joint.inference import route

ACTIONS = (*DOMAINS, "hold")
GATES = (0.0, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 1.0)


def mf(matrix):
    cm = np.asarray(matrix)
    den = cm.sum(0) + cm.sum(1)
    return float(
        np.divide(2 * cm.diagonal(), den, out=np.zeros(len(den)), where=den != 0).mean()
    )


def choices(s, out, threshold, direct):
    ds = [v for _, v in s.registry]
    ps = out["state"]
    states = ps.argmax(-1).tolist()
    if direct:
        p = out["direct"]
        chosen = (*ds, "hold")[int(p.argmax())]
        raw = (
            chosen if threshold is not None and float(p.max()) >= threshold else "hold"
        )
    else:
        raw, _ = route(states, ds, float(ps.max(-1).min()), threshold, ())
    return raw, "hold" if s.issues else raw


def breakdown(cases, snaps, outputs, t, direct=False):
    groups = defaultdict(list)
    for e, s, p in zip(cases, snaps, outputs):
        raw, dec = choices(s, p, t, direct)
        z = (e, s, p, raw, dec)
        groups[("language", e.language)].append(z)
        groups[("family", e.language, e.family)].append(z)
        groups[("slice", e.language, e.variant)].append(z)
    return {"/".join(key): score(v) for key, v in sorted(groups.items())}


def score(records):
    cm = np.zeros((5, 5), dtype=int)
    sc = np.zeros((3, 3), dtype=int)
    rc = np.zeros((5, 5), dtype=int)
    known, correct, nulls, null_ok = Counter(), Counter(), Counter(), Counter()
    joint_known, joint_ok = 0, 0
    panels = defaultdict(dict)
    for e, s, out, raw, dec in records:
        a, y, v = ACTIONS.index(e.gold_action), ACTIONS.index(dec), ACTIONS.index(raw)
        cm[a, y] += 1
        rc[a, v] += 1
        panels[e.family, e.panel][e.variant] = dec == e.gold_action
        for j, state in enumerate(e.states):
            sc[state, int(out["state"][j].argmax())] += 1
            both = []
            for k, source in enumerate(("log", "memo")):
                gold = (e.logs, e.memos)[k][j]
                picked = out["refs"][j][k]
                if gold is None:
                    nulls[source] += 1
                    null_ok[source] += int(not picked)
                else:
                    known[source] += 1
                    hit = gold in picked
                    correct[source] += int(hit)
                    both.append(hit)
            if len(both) == 2:
                joint_known += 1
                joint_ok += int(all(both))
    paired = {}
    for variant in (
        "log_flip",
        "memo_flip",
        "channel_switch",
        "sequence_swap",
        "two_continue",
    ):
        tests = [
            m["single"] and m[variant]
            for m in panels.values()
            if "single" in m and variant in m
        ]
        paired[variant] = [sum(tests), len(tests)]
    routes = int(cm[:, :4].sum())
    ok = int(cm.diagonal()[:4].sum())
    gh = int(cm[4].sum())
    false = int(cm[4, :4].sum())
    return {
        "n": len(records),
        "action_cm": cm.tolist(),
        "raw_cm": rc.tolist(),
        "state_cm": sc.tolist(),
        "macro_f1": mf(cm),
        "raw_macro_f1": mf(rc),
        "state_macro_f1": mf(sc),
        "correct_checks": ok,
        "routes": routes,
        "coverage": routes / len(records),
        "accepted_error": (routes - ok) / routes if routes else None,
        "gold_hold": gh,
        "false_releases": false,
        "false_release_rate": false / gh if gh else None,
        "raw_false_releases": int(rc[4, :4].sum()),
        "wrong_domain": routes - ok - false,
        "actionable_held": int(cm[:4, 4].sum()),
        "source_hits": {key: [correct[key], known[key]] for key in ("log", "memo")},
        "null_hits": {key: [null_ok[key], nulls[key]] for key in ("log", "memo")},
        "both_sources": [joint_ok, joint_known],
        "paired": paired,
    }


def calibrate(cases, snaps, outputs, direct=False):
    scored = []
    for t in GATES:
        m = [
            score(
                [
                    (e, s, p, *choices(s, p, t, direct))
                    for e, s, p in zip(cases, snaps, outputs)
                    if e.language == l
                ]
            )
            for l in ("en", "ko")
        ]
        risk = max(z["false_release_rate"] for z in m)
        scored.append((t, risk, sum(z["macro_f1"] for z in m) / 2))
    eligible = [z for z in scored if z[1] <= 0.05]
    # Threshold 1 could still release confident rules. If none qualify, explicitly hold all.
    if not eligible:
        return {
            "threshold": None,
            "validation": scored,
            "hold_all_f1": sum(
                mf(
                    [
                        [
                            0,
                            0,
                            0,
                            0,
                            sum(e.language == l and e.gold_action == a for e in cases),
                        ]
                        for a in ACTIONS
                    ]
                )
                for l in ("en", "ko")
            )
            / 2,
        }
    best = max(eligible, key=lambda z: (z[2], -z[1], z[0]))
    return {"threshold": best[0], "validation": scored}


def family_contrast(joint, comparator, seed=812):
    out = {}
    for lang in ("en", "ko"):
        keys = sorted(k for k in joint if k.startswith("family/" + lang + "/"))
        rng = np.random.default_rng(seed)
        j = np.asarray([joint[k]["action_cm"] for k in keys])
        c = np.asarray([comparator[k]["action_cm"] for k in keys])

        def diff(a, b):
            return (
                mf(a) - mf(b),
                a[4, :4].sum() / a[4].sum() - b[4, :4].sum() / b[4].sum(),
            )

        point = diff(j.sum(0), c.sum(0))
        draws = np.array(
            [
                diff(np.einsum("i,ijk->jk", n, j), np.einsum("i,ijk->jk", n, c))
                for n in rng.multinomial(len(keys), [1 / len(keys)] * len(keys), 2000)
            ]
        )
        ci = np.quantile(draws, (0.025, 0.975), axis=0).T.tolist()
        out[lang] = {
            "delta_f1": float(point[0]),
            "delta_false_release_rate": float(point[1]),
            "ci_f1": ci[0],
            "ci_false_releases": ci[1],
            "families": len(keys),
            "point_pass": bool(point[0] > 0 and point[1] <= 0),
            "interval_pass": bool(ci[0][0] > 0 and ci[1][1] <= 0),
        }
    return out
