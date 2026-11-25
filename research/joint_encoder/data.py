"""Entirely synthetic, paired equipment snapshots; labels are stored outside public packets."""

from dataclasses import dataclass
import hashlib
import itertools
import json
import random

from equip_triage.encoding import DOMAINS
from equip_triage.joint.sources import action, prepare

FAMILIES = {
    "train": [
        ("bench", "At {c}, now {s}. Earlier {p}.", "{c}에서 현재 {s}. 앞선 값은 {p}."),
        (
            "second",
            "Previous {p}. Second check on {c}: {s}.",
            "직전 {p}. {c} 재확인 값은 {s}.",
        ),
        (
            "load",
            "Under load {c} reads {s}; old entry {p}.",
            "부하에서 {c} 실측 {s}. 지난 기록 {p}.",
        ),
        (
            "restart",
            "Before start {p}. Since restart {c} shows {s}.",
            "시작 전 {p}. 재가동 후 {c}에서 {s}.",
        ),
        (
            "portable",
            "Portable gauge at {c}: {s}. Notebook had {p}.",
            "휴대계로 {c} 실측 {s}. 수첩에는 {p}.",
        ),
        (
            "shift",
            "Last shift {p}. On this shift {c} reads {s}.",
            "전 교대 {p}. 이번 교대 {c}에서 {s}.",
        ),
        (
            "warm",
            "Once warm, {c} is {s}. Cold check was {p}.",
            "예열 후 {c} 실측 {s}. 냉간에는 {p}.",
        ),
        (
            "cycle",
            "Initial pass {p}. On repeat {c} is {s}.",
            "초회 {p}. 반복 측정 시 {c} 결과 {s}.",
        ),
    ],
    "pilot": [
        (
            "probe",
            "Earlier note {p}. Hand probe on {c} reads {s}.",
            "이전 메모 {p}. 손으로 잰 {c} 값은 {s}.",
        ),
        (
            "display",
            "Display at {c} gives {s}. The logbook had {p}.",
            "{c} 표시값 {s}. 대장에는 {p}.",
        ),
    ],
    "val": [
        (
            "manual",
            "The sheet says {p}. Manual reading at {c}: {s}.",
            "시트상 {p}. {c} 수동 판독은 {s}.",
        ),
        (
            "steady",
            "At steady load {c} is {s}. Before settling: {p}.",
            "정상 부하에서 {c}는 {s}. 안정 전에는 {p}.",
        ),
        (
            "receipt",
            "First receipt {p}. The local gauge at {c} gives {s}.",
            "첫 기록 {p}. {c} 현장 계기로 {s}.",
        ),
        (
            "release",
            "At release {c} reads {s}. Before release {p}.",
            "해제 시 {c} 값 {s}. 해제 전 {p}.",
        ),
    ],
    "test": [
        (
            "plate",
            "Plate value {p}; a fresh reading on {c} gives {s}.",
            "명판 값 {p}. {c}에서 새로 읽은 값 {s}.",
        ),
        (
            "spare",
            "Using {c} on the live unit: {s}. The spare was {p}.",
            "가동기 {c}에서 {s}. 예비기는 {p}.",
        ),
        (
            "draft",
            "Discard draft {p}. Signed observation from {c}: {s}.",
            "초안 {p} 폐기. 서명한 {c} 관측은 {s}.",
        ),
        (
            "standby",
            "Running at {c}, {s}. On standby the entry was {p}.",
            "운전 중 {c}에서 {s}. 대기 때 기록은 {p}.",
        ),
        (
            "replica",
            "The replica was {p}. At {c} on the unit: {s}.",
            "복제품은 {p}. 본체의 {c} 값은 {s}.",
        ),
        (
            "pickup",
            "Measured {s} at {c}. The reference sheet gave {p}.",
            "실측은 {s}, 측정 위치 {c}. 기준표에는 {p}.",
        ),
    ],
}
STATUS = (
    (
        ("above limit", "상한 초과"),
        ("high", "높음"),
        ("out of band", "범위 밖"),
        ("not in band", "대역 이탈"),
    ),
    (
        ("within limit", "범위 안"),
        ("in band", "대역 내"),
        ("normal", "정상"),
        ("not high", "높지 않음"),
    ),
    (
        ("unreadable", "판독 불가"),
        ("undetermined", "미확정"),
        ("unsure", "불명"),
        ("no reading", "판독 없음"),
    ),
)
CHANNEL = (
    (("port A", "A 포트"), ("port B", "B 포트")),
    (("channel A", "A 채널"), ("channel B", "B 채널")),
)
VARIANTS = (
    "single",
    "other_single",
    "alternate_channel",
    "old_note_noise",
    "prior_band_noise",
    "log_flip",
    "memo_flip",
    "channel_switch",
    "sequence_swap",
    "two_continue",
    "both_cleared",
    "uncertain_note",
    "stale_note",
    "missing_note",
    "missing_log",
    "gap",
    "log_conflict",
    "note_conflict",
)
SIZE = {"pilot_train": 3, "pilot": 4, "train": 24, "val": 8, "test": 12}
GROUPS = {"pilot_train": FAMILIES["train"][:4], **FAMILIES}


@dataclass
class Example:
    packet: dict
    family: str
    panel: str
    variant: str
    language: str
    states: tuple
    logs: tuple
    memos: tuple
    gold_action: str

    def envelope(self):
        return {
            "packet": self.packet,
            "labels": {
                "states": self.states,
                "log": self.logs,
                "memo": self.memos,
                "action": self.gold_action,
            },
            "group": [self.family, self.panel, self.variant, self.language],
        }


def encoded(x):
    return (
        json.dumps(x, sort_keys=True, ensure_ascii=False, separators=(",", ":")) + "\n"
    ).encode()


def panel(split, family, fi, rep):
    seed = int.from_bytes(
        hashlib.sha256(f"joint-v1/{split}/{family[0]}/{rep}".encode()).digest()[:8],
        "big",
    )
    rng = random.Random(seed)
    tags = [f"K{rng.randrange(10, 99)}", f"P{rng.randrange(10, 99)}"]
    # All four action domains cycle through each construction independently of the variants.
    active = (fi + rep) % 2
    winner = (fi + rep) % len(DOMAINS)
    domains = [None, None]
    domains[active] = DOMAINS[winner]
    domains[1 - active] = DOMAINS[(winner + 1 + rep % 3) % 4]
    start = 100000 + 100 * (seed % 2000000)
    asof, first = start + 18, 1 + seed % 17
    scope = {"equipment": f"R{seed:016x}", "lot": f"L{seed:x}"}
    refs = rng.sample(range(1000, 9999), 22)
    ref = lambda n: f"S{refs[n]}"
    lanes = [rng.randrange(2), rng.randrange(2)]
    current = [0 if j == active else 1 for j in range(2)]
    sy = [rng.randrange(4) for _ in range(2)]
    old_sy = [rng.randrange(4) for _ in range(2)]
    old_status = [rng.randrange(2) for _ in range(2)]
    aliases = [rng.randrange(2) for _ in range(2)]
    lane_order = list(itertools.product(range(2), range(2)))
    rng.shuffle(lane_order)
    orders = [lane_order, list(reversed(lane_order))]
    seed_order = rng.randrange(2**32)

    def text(slot, lang, selected, value, previous):
        c = CHANNEL[aliases[slot]][selected][lang]
        s = STATUS[value][sy[slot]][lang]
        prior = STATUS[previous][old_sy[slot]][lang]
        return family[1 + lang].format(c=c, s=s, p=prior)

    variants = VARIANTS + (("both_disagree", "late_arrival") if split == "test" else ())
    for variant in variants:
        for lang, language in enumerate(("en", "ko")):
            logs, notes = [], []
            for j in range(2):
                logs.append(
                    dict(
                        scope,
                        id=ref(j),
                        seq=first + j,
                        observed=start + j,
                        available=start + j + 1,
                        kind="register",
                        tag=tags[j],
                        domain=domains[j],
                    )
                )
            latest, report = current.copy(), current.copy()
            selected = lanes.copy()
            if variant == "other_single":
                latest = [1 - v for v in latest]
                report = latest.copy()
            if variant == "both_disagree":
                latest = [1 - v for v in latest]
            if variant == "alternate_channel":
                selected = [1 - v for v in selected]
            if variant == "two_continue":
                latest[1 - active] = report[1 - active] = 0
            if variant == "both_cleared":
                latest[active] = report[active] = 1
            if variant in ("log_flip", "sequence_swap"):
                latest[active] = 1 - latest[active]
            if variant == "memo_flip":
                report[active] = 1 - report[active]
            if variant == "channel_switch":
                selected[active] = 1 - selected[active]
            if variant == "uncertain_note":
                report[active] = 2
            readings = {}
            gold_log, gold_memo = [None, None], [None, None]
            for stage, order in enumerate(orders):
                for pos, (j, ch) in enumerate(order):
                    idx = 2 + 4 * stage + pos
                    if stage == 0:
                        reading = (1 - current[j]) if ch == lanes[j] else current[j]
                        if variant == "prior_band_noise":
                            reading = 1 - reading
                        if (
                            variant == "sequence_swap"
                            and j == active
                            and ch == lanes[j]
                        ):
                            reading = current[j]
                    else:
                        reading = latest[j] if ch == lanes[j] else 1 - current[j]
                        if variant == "alternate_channel":
                            reading = 1 - reading
                        readings[j, ch] = reading
                    if variant == "missing_log" and j == active and ch == selected[j]:
                        continue
                    if variant == "gap" and stage == 0 and pos == 0:
                        continue
                    row = dict(
                        scope,
                        id=ref(idx),
                        seq=first + idx,
                        observed=start + idx,
                        available=start + idx + 1,
                        kind="sample",
                        tag=tags[j],
                        channel=("a", "b")[ch],
                        reading=("above", "within")[reading],
                    )
                    logs.append(row)
                    if stage == 1 and ch == selected[j]:
                        gold_log[j] = row["id"]
                        if variant == "log_conflict" and j == active:
                            logs.append(
                                dict(
                                    row,
                                    id=ref(16),
                                    reading=("within", "above")[reading],
                                )
                            )
                            gold_log[j] = None
            for j in range(2):
                for stage in range(2):
                    if variant == "missing_note" and j == active:
                        continue
                    t = asof - (3 if stage else 21)
                    if variant == "late_arrival" and j == active and not stage:
                        t = asof - 51
                    if variant == "stale_note" and j == active:
                        t = asof - (42 if stage else 51)
                    rid = ref(10 + 2 * j + stage)
                    value = report[j] if stage else old_status[j]
                    if variant == "old_note_noise" and not stage:
                        value = 1 - value
                    chosen = selected[j] if stage else 1 - lanes[j]
                    sentence = text(
                        j,
                        lang,
                        chosen,
                        value,
                        old_status[j] if stage else 1 - old_status[j],
                    )
                    row = dict(
                        scope,
                        id=rid,
                        observed=t,
                        available=t + 1,
                        kind="observation",
                        tag=tags[j],
                        text=sentence,
                    )
                    if variant == "late_arrival" and j == active and stage:
                        row["available"] = asof + 2
                    notes.append(row)
                    if stage:
                        gold_memo[j] = rid
                        if variant == "note_conflict" and j == active:
                            notes.append(
                                dict(
                                    row,
                                    id=ref(17),
                                    text=text(
                                        j, lang, chosen, 1 - current[j], old_status[j]
                                    ),
                                )
                            )
                            gold_memo[j] = None
            states = [2, 2]
            # The reported comparison state is solely the two independent observations.
            for j in range(2):
                states[j] = (
                    report[j]
                    if report[j] < 2 and report[j] == readings[j, selected[j]]
                    else 2
                )
            if variant in (
                "missing_log",
                "missing_note",
                "log_conflict",
                "note_conflict",
                "stale_note",
                "late_arrival",
            ):
                states[active] = 2
            if variant in (
                "missing_note",
                "stale_note",
                "note_conflict",
                "late_arrival",
            ):
                gold_log[active] = None
            if variant in ("stale_note", "late_arrival"):
                gold_memo[active] = None
            # Negative controls are inaccessible at cutoff or belong to another equipment/lot.
            logs.append(
                dict(
                    scope,
                    id=ref(18),
                    kind="service",
                    seq=first + 10,
                    observed=asof + 5,
                    available=asof + 7,
                )
            )
            logs.append(dict(logs[0], id=ref(19), lot=scope["lot"] + "X"))
            notes.append(
                dict(
                    scope,
                    id=ref(20),
                    kind="observation",
                    tag=tags[active],
                    observed=asof - 1,
                    available=asof + 5,
                    text="late reading" if not lang else "늦게 도착한 판독",
                )
            )
            notes.append(
                dict(
                    scope,
                    id=ref(21),
                    kind="resolution",
                    tag=tags[active],
                    observed=asof - 2,
                    available=asof - 1,
                    text="service record" if not lang else "정비 조치 기록",
                )
            )
            rnd = random.Random(seed_order)
            rnd.shuffle(logs)
            rnd.shuffle(notes)
            packet = dict(
                scope,
                as_of=asof,
                first_sequence=first,
                through_sequence=first + 9,
                logs=logs,
                notes=notes,
            )
            invalid = variant in (
                "missing_log",
                "missing_note",
                "stale_note",
                "gap",
                "log_conflict",
                "note_conflict",
                "late_arrival",
            )
            gold = "hold" if invalid else action(states, domains)
            yield Example(
                packet,
                family[0],
                f"{fi}-{rep}",
                variant,
                language,
                tuple(states),
                tuple(gold_log),
                tuple(gold_memo),
                gold,
            )


def generate(split):
    for fi, family in enumerate(GROUPS[split]):
        for rep in range(SIZE[split]):
            yield from panel(split, family, fi, rep)


def manifest(split):
    full, obs = hashlib.sha256(), hashlib.sha256()
    groups, assets, cases, features = set(), set(), set(), set()
    from collections import Counter

    labels = Counter()
    for r in generate(split):
        full.update(encoded(r.envelope()))
        obs.update(encoded(r.packet))
        s = prepare(r.packet)
        if any(u.age < 0 for u in s.units) or len(s.registry) != 2:
            raise ValueError("Generator produced invalid scope")
        visible = (
            tuple(d for _, d in s.registry),
            tuple((u.slot, u.modality, u.text, u.age, u.order) for u in s.units),
            s.issues,
        )
        features.add(hashlib.sha256(encoded(visible)).digest())
        groups.add(r.family)
        cases.add((r.family, r.panel))
        assets.add((s.equipment, s.lot))
        labels[(r.language, r.gold_action)] += 1
    return {
        "rows": sum(labels.values()),
        "families": sorted(groups),
        "panels": len(cases),
        "equipment_lot_groups": len(assets),
        "distinct_visible_packets": len(features),
        "labels": {"/".join(k): v for k, v in sorted(labels.items())},
        "packet_sha256": obs.hexdigest(),
        "labeled_sha256": full.hexdigest(),
    }
