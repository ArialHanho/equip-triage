"""Fixed bilingual scope rules and train-only character/event logistic baselines."""

import re
import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from equip_triage.joint.sources import BANDS, MAX_AGE
from .data import STATUS

terms = {
    s.casefold(): y
    for y, choices in enumerate(STATUS)
    for term in choices
    for s in term
}
STATUS_RE = re.compile(
    "|".join(re.escape(s) for s in sorted(terms, key=len, reverse=True))
)
CHANNEL_RE = re.compile(r"(?:port|channel)\s+([ab])\b|\b([ab])\s*(?:포트|채널)", re.I)


def current_note(snap, j):
    notes = snap.rows(j, "memo")
    if not notes:
        return None
    t = max(u.observed for u in notes)
    latest = [u for u in notes if u.observed == t]
    return latest[0] if len(latest) == 1 and latest[0].age <= MAX_AGE else None


def parse(note):
    if note is None:
        return None, 2
    text = note.text.casefold()
    ref = CHANNEL_RE.search(text)
    if not ref:
        return None, 2
    lane = ref.group(1) or ref.group(2)
    # Prefer the current reading within the channel's clause. Status atoms include negation.
    start = max(text.rfind(ch, 0, ref.start()) for ch in (".", ";")) + 1
    next_breaks = [k for ch in (".", ";") if (k := text.find(ch, ref.end())) >= 0]
    end = min(next_breaks, default=len(text))
    terms_after = list(STATUS_RE.finditer(text, ref.end(), end))
    terms_before = list(STATUS_RE.finditer(text, start, ref.start()))
    found = (
        terms_after[0] if terms_after else terms_before[-1] if terms_before else None
    )
    return lane, terms[found.group()] if found else 2


def reading(snap, j, lane):
    samples = [u for u in snap.rows(j, "log") if u.channel == lane]
    if not samples:
        return None
    seq = max(u.sequence for u in samples)
    recent = [u for u in samples if u.sequence == seq]
    return recent[0] if len(recent) == 1 and recent[0].age <= MAX_AGE else None


def refs(snap, j, view):
    note = current_note(snap, j) if view != "log" else None
    lane, status = parse(note)
    obs = reading(snap, j, lane) if view != "memo" else None
    return note, obs, status


def rule_outputs(snap):
    result, indices = [], []
    for j in range(2):
        note, obs, y = refs(snap, j, "joint")
        state = y if obs and y < 2 and BANDS[y] == obs.band else 2
        result.append(np.eye(3)[state])
        indices.append((obs.ids if obs else (), note.ids if note else ()))
    return {"state": np.array(result), "direct": None, "refs": indices}


def fields(snap, j, view):
    note = current_note(snap, j) if view != "log" else None
    all_notes = snap.rows(j, "memo") if view != "log" else []
    text = (
        " latest "
        + (note.text.casefold() if note else "")
        + " previous "
        + " / ".join(u.text.casefold() for u in all_notes if u != note)
    )
    features = {f"domain={snap.registry[j][1]}": 1}
    gates = []
    if view != "memo":
        for lane in ("a", "b"):
            latest = reading(snap, j, lane)
            band = latest.band if latest else "missing"
            gates.append(band)
            past = sorted(
                [u for u in snap.rows(j, "log") if u.channel == lane],
                key=lambda u: (u.sequence, u.band),
            )
            features[f"{lane}:series=" + "/".join(u.band for u in past)] = 1
            features[f"{lane}:latest={band}"] = 1
    return text, features, gates


class CharEvents:
    def __init__(self, view):
        self.view = view
        self.char = TfidfVectorizer(
            analyzer="char",
            ngram_range=(2, 5),
            min_df=2,
            sublinear_tf=True,
            dtype=np.float64,
        )
        self.events = DictVectorizer()
        self.lr = LogisticRegression(C=4, max_iter=400, random_state=17)

    def matrix(self, snapshots, fit):
        rows = [fields(s, j, self.view) for s in snapshots for j in range(2)]
        texts, features, gates = zip(*rows)
        ev = (
            self.events.fit_transform(features)
            if fit
            else self.events.transform(features)
        )
        if self.view == "log":
            return ev
        tx = self.char.fit_transform(texts) if fit else self.char.transform(texts)
        blocks = [ev, tx]
        if self.view == "joint":
            # Trainable text × observed event-band interactions; no gold selected channel.
            bands = np.array(gates)
            for j in range(2):
                for value in (*BANDS, "missing"):
                    blocks.append(tx.multiply((bands[:, j] == value)[:, None]))
        return hstack(blocks, format="csr")

    def fit(self, snapshots, cases):
        self.lr.fit(
            self.matrix(snapshots, True),
            np.array([r.states for r in cases]).reshape(-1),
        )
        return self

    def outputs(self, snapshots):
        pr = self.lr.predict_proba(self.matrix(snapshots, False)).reshape(-1, 2, 3)
        out = []
        for s, p in zip(snapshots, pr):
            ids = []
            for j in range(2):
                note, obs, _ = refs(s, j, self.view)
                ids.append((obs.ids if obs else (), note.ids if note else ()))
            out.append({"state": p, "direct": None, "refs": ids})
        return out
