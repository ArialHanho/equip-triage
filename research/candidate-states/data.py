"""Original synthetic, paired bilingual packets. Generation uses no external corpus."""

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
from hashlib import sha256
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DOMAINS = ("measurement", "data", "setup", "equipment")
ACTIONS = (*DOMAINS, "hold")
STATES = ("continues", "cleared", "unknown")
SPLITS = ("train", "validation", "challenge")
# Each family sees all domains in two pairs. Across families, all six domain pairs occur.
MATCHINGS = (((0, 1), (2, 3)), ((0, 2), (1, 3)), ((0, 3), (1, 2)))
TAGS = {
    "common": ("AM17", "BD26", "CE35", "DF44"),
    "train": ("EK13", "FM22", "GN31", "HP40"),
    "validation": ("JQ14", "KR23", "LS32", "MT41"),
    "challenge": ("NU15", "PV24", "QW33", "RX42"),
}
ORDINAL = {
    "en": ("first listed check", "second listed check"),
    "ko": ("앞 점검 항목", "뒤 점검 항목"),
}
# Gold actions are specified independently of the executable decoder.
PATTERNS = {
    "pc": 0,
    "cp": 1,
    "pp": None,
    "cc": None,
    "pu": None,
    "up": None,
    "uu": None,
}
STATUS = {"p": 0, "c": 1, "u": 2}
VIEWS = (
    ("common", "base"),
    ("new_tag", "base"),
    ("new_tag", "swap_slots"),
    ("new_tag", "swap_lines"),
    ("ordinal", "base"),
    ("ordinal", "swap_slots"),
    ("ordinal", "swap_lines"),
)


@dataclass(frozen=True)
class Packet:
    domains: tuple
    tags: tuple
    memo: str


@dataclass(frozen=True)
class Example:
    family: str
    case: str
    lang: str
    mode: str
    view: str
    pattern: str
    packet: Packet
    states: tuple
    gold: str


def wordings(split):
    seen = set()
    for line in (ROOT / "wordings.txt").read_text(encoding="utf-8").splitlines():
        if not line or line.startswith("#"):
            continue
        partition, family, *pairs = line.split("|")
        if family in seen or len(pairs) != 3:
            raise ValueError("Duplicate or malformed scenario family.")
        seen.add(family)
        if partition == split:
            templates = [pair.split(" ~ ") for pair in pairs]
            assert all(
                len(pair) == 2 and all(t.count("{r}") == 1 for t in pair)
                for pair in templates
            )
            yield family, templates


def generate(split):
    rows = []
    for fi, (family, templates) in enumerate(wordings(split)):
        for pair_num, pair in enumerate(MATCHINGS[fi % len(MATCHINGS)]):
            domains = tuple(DOMAINS[i] for i in pair)
            for li, lang in enumerate(("en", "ko")):
                case = f"{family}/{pair_num}"
                for mode, view in VIEWS:
                    bank = TAGS["common" if mode == "common" else split]
                    tags = tuple(bank[(i + fi) % len(bank)] for i in pair)
                    refs = ORDINAL[lang] if mode == "ordinal" else tags
                    reverse = view == "swap_slots"
                    listed = domains[::-1] if reverse else domains
                    listed_tags = tags[::-1] if reverse else tags
                    for pattern, active in PATTERNS.items():
                        codes = tuple(STATUS[s] for s in pattern)
                        lines = [
                            templates[s][li].format(r=ref)
                            for s, ref in zip(codes, refs)
                        ]
                        if view == "swap_lines":
                            lines.reverse()
                        packet = Packet(listed, listed_tags, "\n".join(lines))
                        states = codes[::-1] if reverse and mode != "ordinal" else codes
                        meaning = listed if mode == "ordinal" else domains
                        gold = "hold" if active is None else meaning[active]
                        rows.append(
                            Example(
                                family,
                                case,
                                lang,
                                mode,
                                view,
                                pattern,
                                packet,
                                states,
                                gold,
                            )
                        )
                # Each integrity failure has its own observable cause. The first
                # continuing clause is insufficient when the second result is missing.
                bank = TAGS[split]
                tags = tuple(bank[(i + fi) % len(bank)] for i in pair)
                first = templates[0][li].format(r=tags[0])
                second = templates[1][li]
                for kind, ids, other, states in (
                    ("missing_line", domains, "", (0, 2)),
                    (
                        "missing_record",
                        (domains[0], ""),
                        second.format(r=tags[1]),
                        (0, 2),
                    ),
                    ("unlisted", domains, second.format(r="ZZ99"), (0, 2)),
                    ("duplicate_reference", domains, second.format(r=tags[0]), (2, 2)),
                ):
                    rows.append(
                        Example(
                            family,
                            case,
                            lang,
                            "incomplete",
                            "base",
                            kind,
                            Packet(ids, tags, first + ("\n" + other if other else "")),
                            states,
                            "hold",
                        )
                    )
    return rows


def fingerprint(rows):
    h = sha256()
    for row in rows:
        h.update(
            json.dumps(
                asdict(row), ensure_ascii=False, sort_keys=True, separators=(",", ":")
            ).encode()
        )
        h.update(b"\n")
    return h.hexdigest()


def summary(rows):
    return dict(
        rows=len(rows),
        families=len({r.family for r in rows}),
        cases=len({r.case for r in rows}),
        labels=dict(Counter(r.gold for r in rows)),
        sha256=fingerprint(rows),
    )


def file_hash(name):
    return sha256((ROOT / name).read_bytes()).hexdigest()


def make_manifest():
    parts = {s: generate(s) for s in SPLITS}
    seen_families = set()
    seen_inputs = set()
    seen_memos = set()
    for rows in parts.values():
        families = {r.family for r in rows}
        inputs = {r.packet for r in rows}
        memos = {r.packet.memo for r in rows}
        if families & seen_families or inputs & seen_inputs or memos & seen_memos:
            raise ValueError("Cross-split scenario, raw packet or exact memo leakage.")
        seen_families.update(families)
        seen_inputs.update(inputs)
        seen_memos.update(memos)
        assert len(inputs) == len(rows), "Duplicate raw packet within a split."
    return dict(
        files={n: file_hash(n) for n in ("data.py", "wordings.txt", "PROTOCOL.md")},
        splits={s: summary(rows) for s, rows in parts.items()},
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    current = make_manifest()
    if args.verify and current != json.loads((ROOT / "manifest.json").read_text()):
        raise SystemExit("Manifest mismatch.")
    print(json.dumps(current, indent=2))
