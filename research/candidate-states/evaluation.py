"""Operating points, exact matched decisions and paired family intervals."""

from collections import Counter, defaultdict
import numpy as np
from data import ACTIONS

LANGS = ("en", "ko")
GRID = (0.0, 0.25, 0.40, 0.55, 0.70, 0.85, 0.95, 1.01)


def decisions(output, t):
    return np.where(output.confidence < t, 4, output.action)


def confusion(rows, pred):
    gold = np.array([ACTIONS.index(r.gold) for r in rows])
    return np.bincount(5 * gold + pred, minlength=25).reshape(5, 5)


def macro(cm):
    denom = cm.sum(-1) + cm.sum(-2)
    return np.divide(
        2 * np.diagonal(cm, axis1=-2, axis2=-1),
        denom,
        out=np.zeros_like(denom, dtype=float),
        where=denom > 0,
    ).mean(-1)


def summary(cm):
    n = int(cm.sum())
    gold_hold = int(cm[4].sum())
    routed = int(cm[:, :4].sum())
    unsafe = int(cm[4, :4].sum())
    correct = int(np.trace(cm[:4, :4]))
    return dict(
        n=n,
        macro_f1=float(macro(cm)),
        gold_hold=gold_hold,
        unsafe=unsafe,
        unsafe_rate=unsafe / gold_hold if gold_hold else None,
        correct_check=correct,
        wrong_check=routed - unsafe - correct,
        actionable_held=int(cm[:4, 4].sum()),
        routed=routed,
        coverage=routed / n,
        accepted_risk=(routed - correct) / routed if routed else None,
    )


def langs(rows, pred, with_matrix=False):
    result = {}
    for lang in LANGS:
        ix = [i for i, r in enumerate(rows) if r.lang == lang]
        cm = confusion([rows[i] for i in ix], pred[ix])
        result[lang] = summary(cm)
        if with_matrix:
            result[lang]["confusion"] = cm.tolist()
    return result


def gates(rows, out):
    candidates = []
    safe = []
    for t in GRID:
        by = langs(rows, decisions(out, t))
        ms = list(by.values())
        tail = (-sum(x["unsafe"] for x in ms), -sum(x["wrong_check"] for x in ms), t)
        candidates.append((np.mean([x["macro_f1"] for x in ms]), *tail))
        if all(x["unsafe_rate"] <= 0.1 for x in ms):
            safe.append((sum(x["correct_check"] for x in ms), *tail))
    return dict(
        utility=float(max(candidates)[-1]),
        safety=float(max(safe)[-1]),
        validation_f1=float(max(candidates)[0]),
    )


def contrasts(rows):
    groups = defaultdict(dict)
    for i, r in enumerate(rows):
        groups[r.case, r.lang][r.mode, r.view, r.pattern] = i
    pairs = defaultdict(list)
    for (case, lang), lookup in groups.items():
        a, b = ("pc", "pu") if case.endswith("/0") else ("cp", "up")

        def add(name, *keys):
            indices = tuple(lookup[k] for k in keys)
            pairs[name, lang].append(indices)

        for mode, view in sorted({(m, v) for m, v, _ in lookup if m != "incomplete"}):

            def ix(p):
                return mode, view, p

            add("one_vs_two", ix(a), ix("pp"))
            add("both_single_directions_and_two", ix("pc"), ix("cp"), ix("pp"))
            add("latest_cleared", ix(a), ix("cc"))
            add("missing_outcome", ix(a), ix(b))
        for p in ("pc", "cp"):
            add(
                "positional_slot_swap",
                ("ordinal", "base", p),
                ("ordinal", "swap_slots", p),
            )
            add(
                "registered_slot_swap",
                ("new_tag", "base", p),
                ("new_tag", "swap_slots", p),
            )
            add("new_registered_tag", ("new_tag", "base", p), ("common", "base", p))
            for m in ("new_tag", "ordinal"):
                add("line_order", (m, "base", p), (m, "swap_lines", p))
    return pairs


def pair_counts(rows, pred):
    ok = pred == np.array([ACTIONS.index(r.gold) for r in rows])
    return {
        f"{name}/{lang}": dict(
            correct=int(ok[np.array(items)].all(1).sum()), n=len(items)
        )
        for (name, lang), items in contrasts(rows).items()
    }


def evaluate(rows, prepared, out, c):
    all_gates = {"raw": 0.0, "utility": c["utility"], "safety": c["safety"]}
    report = {"operating_points": {}, "families": {}}
    valid = [i for i, r in enumerate(prepared) if r.complete]
    fams = defaultdict(list)
    for i, r in enumerate(rows):
        fams[r.family].append(i)
    for gate, t in all_gates.items():
        p = decisions(out, t)
        record = dict(
            threshold=t,
            metrics=langs(rows, p, True),
            valid=langs([rows[i] for i in valid], p[valid]),
            pairs=pair_counts(rows, p),
        )
        report["operating_points"][gate] = record
        for f, indices in fams.items():
            sub = [rows[i] for i in indices]
            pred = p[indices]
            pc = pair_counts(sub, pred)
            for lang in LANGS:
                ix = [j for j, r in enumerate(sub) if r.lang == lang]
                cm = confusion([sub[j] for j in ix], pred[ix])
                report["families"].setdefault(f, {}).setdefault(lang, {})[gate] = dict(
                    cm=cm.tolist(), pair=pc["one_vs_two/" + lang]
                )
    p = decisions(out, c["utility"])
    slices = defaultdict(list)
    for i, r in enumerate(rows):
        slices[r.pattern, r.lang].append(i)
    report["by_pattern"] = {
        f"{kind}/{lang}": dict(
            n=len(ix), correct=sum(int(p[i] == ACTIONS.index(rows[i].gold)) for i in ix)
        )
        for (kind, lang), ix in slices.items()
    }
    if out.states is not None:
        report["valid_states"] = {}
        for lang in LANGS:
            ix = [i for i in valid if rows[i].lang == lang]
            gold = np.asarray([rows[i].states for i in ix])
            states = out.states[ix]
            report["valid_states"][lang] = dict(
                n=len(ix),
                joint_correct=int((gold == states).all(1).sum()),
                candidate_correct=int((gold == states).sum()),
                confusion=np.bincount((3 * gold + states).ravel(), minlength=9)
                .reshape(3, 3)
                .tolist(),
            )
    report["risk_curve"] = {str(t): langs(rows, decisions(out, t)) for t in GRID}
    return report


def resample(runs):
    fams = sorted(runs[0]["families"])
    n = len(fams)
    rng = np.random.default_rng(1783)
    weights = np.vstack(
        [np.ones(n, dtype=int), rng.multinomial(n, np.full(n, 1 / n), size=2000)]
    )
    dist = defaultdict(list)
    for r in runs:
        for lang in LANGS:
            for gate in ("raw", "utility", "safety"):
                parts = [r["families"][f][lang][gate] for f in fams]
                cms = np.einsum(
                    "rf,fij->rij", weights, np.array([x["cm"] for x in parts])
                )
                pair = weights @ np.array(
                    [[x["pair"]["correct"], x["pair"]["n"]] for x in parts]
                )
                values = {
                    "pair": pair[:, 0] / pair[:, 1],
                    "f1": macro(cms),
                    "unsafe_rate": cms[:, 4, :4].sum(1) / cms[:, 4, :].sum(1),
                    "coverage": cms[:, :, :4].sum((1, 2)) / cms.sum((1, 2)),
                }
                for metric, vals in values.items():
                    dist[r["method"], lang, gate, metric].append(vals)
    means = {k: np.mean(v, axis=0) for k, v in dist.items()}

    def ci(a):
        return dict(
            estimate=float(a[0]), ci95=np.quantile(a[1:], [0.025, 0.975]).tolist()
        )

    absolute = {":".join(k): ci(v) for k, v in means.items() if k[2] != "raw"}
    contrasts_out = []
    comparisons = [
        ("char_direct_raw", "char_direct"),
        ("bert_flat_raw", "bert_flat"),
        ("char_mentions", "char_mentions_bound"),
        ("char_mentions_bound", "char_states_bound"),
        ("char_states_global", "char_states_bound"),
        ("bert_flat", "bert_local_direct"),
        ("bert_local_direct", "bert_cls_states"),
        ("bert_cls_states", "bert_bound_query"),
        ("bert_all_mean", "bert_bound_mean"),
        ("bert_all_query", "bert_bound_query"),
        ("bert_bound_mean", "bert_bound_query"),
        ("char_states_bound", "bert_bound_query"),
        ("scoped_rule", "bert_bound_query"),
        ("char_states_bound_wta", "char_states_bound"),
        ("bert_bound_query_wta", "bert_bound_query"),
    ]
    for a, b in comparisons:
        for lang in LANGS:
            for gate in ("utility", "raw") if a.endswith("wta") else ("utility",):
                for met in ("f1", "pair"):
                    x = means[b, lang, gate, met] - means[a, lang, gate, met]
                    contrasts_out.append(
                        dict(a=a, b=b, lang=lang, gate=gate, metric=met, **ci(x))
                    )
    return dict(
        families=fams,
        replicates=2000,
        bootstrap_seed=1783,
        absolute=absolute,
        contrasts=contrasts_out,
    )
