"""Inspect excluded observations and counterfactual controls before optimization."""

from collections import defaultdict
from copy import deepcopy
from .data import generate
from equip_triage.joint import prepare


def visible(s, modality=None):
    return (
        tuple(d for _, d in s.registry),
        tuple(
            (u.modality, u.slot, u.text, u.age, u.order)
            for u in s.units
            if modality is None or u.modality == modality
        ),
    )


def inspect(splits):
    out, seen_assets, seen_notes = {}, set(), set()
    for part in splits:
        groups = defaultdict(dict)
        assets, lines = set(), set()
        total = 0
        for e in generate(part):
            s = prepare(e.packet)
            assets.add(s.equipment)
            lines.update(u.text for u in s.units if u.modality == "memo")
            if e.variant in ("single", "log_flip", "memo_flip", "sequence_swap"):
                groups[e.panel, e.language][e.variant] = (s, e.gold_action)
            if e.variant == "single":
                p = deepcopy(e.packet)
                p["logs"].reverse()
                p["notes"].reverse()
                for r in p["notes"]:
                    if r["available"] > p["as_of"] or r["kind"] == "resolution":
                        r["text"] = "deliberately changed excluded evidence"
                p["logs"] = [
                    r
                    for r in p["logs"]
                    if r["equipment"] == p["equipment"]
                    and r["lot"] == p["lot"]
                    and r["observed"] <= p["as_of"]
                ]
                if prepare(p) != s:
                    raise ValueError(
                        "Presentation order, wrong lot or a future/repair record changed the snapshot"
                    )
            total += 1
        if (assets & seen_assets) or (lines & seen_notes):
            raise ValueError(
                "A virtual equipment or complete memo sentence crosses partitions"
            )
        seen_assets.update(assets)
        seen_notes.update(lines)
        n = 0
        for qs in groups.values():
            base, target = qs["single"]
            for kind, identical in (
                ("log_flip", "memo"),
                ("memo_flip", "log"),
                ("sequence_swap", "memo"),
            ):
                other, gold = qs[kind]
                assert target != gold and visible(base, identical) == visible(
                    other, identical
                )

            def bag(s):
                return sorted(
                    (u.slot, u.channel, u.band) for u in s.units if u.modality == "log"
                )

            assert bag(base) == bag(qs["sequence_swap"][0]) and visible(
                base, "log"
            ) != visible(qs["sequence_swap"][0], "log")
            n += 1
        out[part] = {
            "rows": total,
            "bilingual_language_panels": n,
            "fixed_other_modality_pairs": 2 * n,
            "order_pairs_with_same_log_multiset": n,
            "order_and_exclusion_invariance_checks": n,
            "equipment": len(assets),
            "distinct_memo_sentences": len(lines),
        }
    return out
