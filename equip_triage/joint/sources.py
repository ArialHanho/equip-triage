"""Validate a cutoff snapshot and retain source IDs outside the neural feature stream."""

from collections import defaultdict
from dataclasses import dataclass
import unicodedata

from ..encoding import DOMAINS

VERSION = "equip-triage/joint-v1"
MAX_BYTES = 160
MAX_UNITS = 32
MAX_AGE = 30
CHANNELS = ("a", "b")
BANDS = ("above", "within")


class SourceError(ValueError):
    pass


def integer(x):
    if type(x) is not int or x < 0:
        raise SourceError("invalid_integer")
    return x


def identifier(x):
    if not isinstance(x, str) or not x.strip() or len(x) > 128:
        raise SourceError("invalid_identifier")
    return x


def clean(x):
    if not isinstance(x, str):
        raise SourceError("invalid_text")
    return " ".join(unicodedata.normalize("NFKC", x).split())


@dataclass(frozen=True)
class Unit:
    modality: str
    slot: int
    text: str
    age: int
    order: int
    ids: tuple
    observed: int
    sequence: int | None
    channel: str = ""
    band: str = ""


@dataclass
class Snapshot:
    equipment: str
    lot: str
    cutoff: int
    registry: tuple
    units: tuple
    issues: tuple
    gaps: tuple
    conflicts: tuple
    registrations: tuple

    def rows(self, slot, modality):
        return [u for u in self.units if u.slot == slot and u.modality == modality]


def prepare(packet):
    if not isinstance(packet, dict):
        raise SourceError("invalid_packet")
    try:
        eq, lot = identifier(packet["equipment"]), identifier(packet["lot"])
        now = integer(packet["as_of"])
        first, end = (
            integer(packet["first_sequence"]),
            integer(packet["through_sequence"]),
        )
        if first < 1 or first > end:
            raise SourceError("invalid_window")
        logs, notes = packet["logs"], packet["notes"]
        if not isinstance(logs, list) or not isinstance(notes, list):
            raise SourceError("invalid_sources")
        return _prepare(eq, lot, now, first, end, logs, notes)
    except (KeyError, UnicodeError) as exc:
        raise SourceError("invalid_record") from exc


def _prepare(eq, lot, now, first, end, logs, notes):
    issues, conflicts = set(), []
    used_ids = {}

    def visible(records):
        for r in records:
            if not isinstance(r, dict):
                raise SourceError("invalid_source_row")
            if (r.get("equipment"), r.get("lot")) != (eq, lot):
                continue
            obs, avail = integer(r["observed"]), integer(r["available"])
            if obs > now or avail > now:
                continue
            if avail < obs:
                issues.add("availability_precedes_observation")
            yield r, obs

    def register_id(r, identity):
        rid = identifier(r["id"])
        if used_ids.setdefault(rid, identity) != identity:
            issues.add("conflicting_source_id")
            conflicts.append((rid,))
        return rid

    events = defaultdict(lambda: defaultdict(list))
    for r, t in visible(logs):
        seq = integer(r["seq"])
        if not first <= seq <= end:
            continue
        kind = r.get("kind")
        if kind in ("service", "resolution"):
            continue
        if kind not in ("register", "sample", "marker"):
            issues.add("unknown_log_kind")
            continue
        tag = identifier(r["tag"]) if kind != "marker" else ""
        domain = r["domain"] if kind == "register" else ""
        lane = r["channel"] if kind == "sample" else ""
        band = r["reading"] if kind == "sample" else ""
        if kind == "register" and domain not in DOMAINS:
            raise SourceError("invalid_domain")
        if kind == "sample" and (lane not in CHANNELS or band not in BANDS):
            raise SourceError("invalid_measurement")
        key = (kind, tag, domain, lane, band, t)
        events[seq][key].append(register_id(r, (seq, *key)))

    present = sorted(events)
    bounds = [first - 1, *present, end + 1]
    gaps = tuple((a + 1, b - 1) for a, b in zip(bounds, bounds[1:]) if b > a + 1)
    if gaps:
        issues.add("sequence_gap")
    registry, registration_ids, log_records = [], [], []
    latest_time = -1
    for seq in present:
        choices = events[seq]
        if len(choices) > 1:
            issues.add("sequence_conflict")
            conflicts.append(tuple(sorted(i for ids in choices.values() for i in ids)))
        times = [key[-1] for key in choices]
        if min(times) < latest_time:
            issues.add("clock_sequence_conflict")
        latest_time = max(latest_time, *times)
        for (kind, tag, domain, lane, band, t), ids in sorted(choices.items()):
            ids = tuple(sorted(set(ids)))
            if kind == "register":
                if (tag, domain) not in registry:
                    registry.append((tag, domain))
                registration_ids.extend(ids)
            elif kind == "sample":
                log_records.append((tag, lane, band, t, seq, ids))
    if (
        len(registry) != 2
        or len({t for t, _ in registry}) != 2
        or len({d for _, d in registry}) != 2
    ):
        issues.add("incomplete_registry")
    slot_by_tag = {tag: j for j, (tag, _) in enumerate(registry)}
    units = []
    for tag, lane, band, t, seq, ids in log_records:
        if tag not in slot_by_tag:
            issues.add("unregistered_source")
            continue
        units.append(
            Unit(
                "log",
                slot_by_tag[tag],
                f"{lane} {band}",
                now - t,
                seq - first,
                ids,
                t,
                seq,
                lane,
                band,
            )
        )

    note_keys = defaultdict(lambda: defaultdict(list))
    for r, t in visible(notes):
        if r.get("kind") in ("service", "resolution"):
            continue
        if r.get("kind") != "observation":
            issues.add("unknown_note_kind")
            continue
        tag, text = identifier(r["tag"]), clean(r["text"])
        rid = register_id(r, ("note", tag, t, text))
        if tag not in slot_by_tag:
            issues.add("unregistered_source")
            continue
        if not text or len(text.encode("utf-8")) > MAX_BYTES:
            issues.add("unusable_note")
            continue
        note_keys[(slot_by_tag[tag], t)][text].append(rid)
    order = defaultdict(int)
    for (slot, t), variants in sorted(note_keys.items()):
        if len(variants) > 1:
            issues.add("note_conflict")
            conflicts.append(tuple(sorted(i for ids in variants.values() for i in ids)))
        for text, ids in sorted(variants.items()):
            units.append(
                Unit(
                    "memo",
                    slot,
                    text,
                    now - t,
                    order[slot],
                    tuple(sorted(set(ids))),
                    t,
                    None,
                )
            )
        order[slot] += 1
    for j in range(len(registry)):
        for mod in ("log", "memo"):
            rows = [u for u in units if u.slot == j and u.modality == mod]
            if not rows:
                issues.add(f"missing_{mod}")
            elif min(u.age for u in rows) > MAX_AGE:
                issues.add(f"stale_{mod}")
        for lane in CHANNELS:
            if not any(
                u.modality == "log"
                and u.slot == j
                and u.channel == lane
                and u.age <= MAX_AGE
                for u in units
            ):
                issues.add("incomplete_channel_window")
    if len(units) > MAX_UNITS:
        issues.add("window_too_large")
    return Snapshot(
        eq,
        lot,
        now,
        tuple(registry),
        tuple(units),
        tuple(sorted(issues)),
        gaps,
        tuple(conflicts),
        tuple(registration_ids),
    )


def action(states, domains):
    if len(states) == 2 and sorted(states) == [0, 1]:
        return domains[states.index(0)]
    return "hold"
