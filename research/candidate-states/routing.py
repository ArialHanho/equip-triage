"""Observable registry binding, status interpretation, and routing policies."""

from dataclasses import dataclass
import re
import numpy as np
from data import ACTIONS, DOMAINS, ORDINAL, Packet

CONTINUES, CLEARED, UNKNOWN = range(3)
HOLD = 4
MISSING_DOMAIN = 4
NO_BINDING = -1


@dataclass(frozen=True)
class Evidence:
    domains: tuple
    lines: tuple
    binding: tuple
    complete: bool


def prepare(packet: Packet):
    """Bind an observed name/ordinal to a slot. Never consult a family or label."""
    domains = tuple(
        DOMAINS.index(x) if x in DOMAINS else MISSING_DOMAIN for x in packet.domains
    )
    tags = tuple(t.strip().casefold() for t in packet.tags)
    records_ok = (
        len(set(domains)) == 2
        and MISSING_DOMAIN not in domains
        and len(set(tags)) == 2
        and all(re.fullmatch(r"[a-z][a-z0-9_-]{0,31}", t) for t in tags)
    )
    raw = [line.strip().casefold() for line in packet.memo.splitlines() if line.strip()]
    lines, binding = [], []
    for line in (raw + ["", ""])[:2]:
        matches = []
        scrubbed = line
        for slot, tag in enumerate(tags):
            names = [n[slot] for n in ORDINAL.values()]
            if tag:
                names.append(tag)
            patterns = []
            for name in names:
                pattern = re.escape(name)
                if name.isascii():
                    pattern = rf"(?<![a-z0-9_]){pattern}(?![a-z0-9_])"
                patterns.append(pattern)
            expression = "|".join(patterns)
            if re.search(expression, line):
                matches.append(slot)
                scrubbed = re.sub(expression, " @ref ", scrubbed)
        binding.append(matches[0] if len(matches) == 1 else NO_BINDING)
        lines.append(" ".join(scrubbed.split()))
    complete = bool(records_ok and len(raw) == 2 and sorted(binding) == [0, 1])
    return Evidence(domains, tuple(lines), tuple(binding), complete)


def decode_states(domains, states, active_prob, *, exact=True):
    if UNKNOWN in states:
        return HOLD
    active = [i for i, s in enumerate(states) if s == CONTINUES]
    if not active or (exact and len(active) != 1):
        return HOLD
    return domains[max(active, key=lambda i: active_prob[i])]


@dataclass
class Output:
    action: np.ndarray
    confidence: np.ndarray
    states: np.ndarray | None = None


def state_output(rows, probs, *, exact=True):
    states = probs.argmax(2)
    actions = [
        decode_states(r.domains, s, p[:, CONTINUES], exact=exact)
        if r.complete
        else HOLD
        for r, s, p in zip(rows, states, probs)
    ]
    return Output(np.asarray(actions), probs.max(2).min(1), states)


def direct_output(rows, probs, guarded=True):
    scores = probs.copy()
    if guarded:
        for i, row in enumerate(rows):
            if not row.complete:
                scores[i] = [0, 0, 0, 0, 1]
            else:
                for j in range(4):
                    if j not in row.domains:
                        scores[i, j] = 0
    return Output(scores.argmax(1), scores.max(1))


# Match longer polarity constructions as a unit; when two observations are given,
# the last recognized state determines the current state. Unreadable evidence vetoes.
UNKNOWN_CUE = re.compile(
    r"unknown|unconfirmed|inconclusive|blank|unreadable|no (?:post[- ]\w+ )?(?:reading|result|observation)|without (?:a |the )?(?:reading|result)|before .*(?:captured|verdict)|not (?:been )?(?:checked|recorded)|skipped|blocked observation|cannot be (?:determined|verified)|is missing|누락|판독할 수 없|판정할 수 없|알 수 없|기록.{0,4}없|기록하지 않|얻지 못|남기지 못|비어 있|생략|미확인|대기 중|확인하지 못|관찰할 수 없"
)
POS_CUE = r"did not (?:recover|clear|resolve|disappear)|not resolved|out(?:side| of) (?:the )?(?:tolerance|range)|same fault|failed?|failure|faulty|unstable|drift reproduced|reproduced the deviation|불량|불합격|불안정|재발|해소되지 않|사라지지 않|회복되지 않|범위.{0,3}(?:초과|벗어)|오류.{0,2}지속|편차.{0,4}재현됐"
CLEAR_CUE = r"no (?:remaining )?(?:fault|drift)|no longer reproduced|did not reproduce|ruled out|returned to normal|normal|back (?:inside|in) (?:tolerance|range)|stable|passed?|clear(?:ed)?|recovered|disappeared|정상|합격|다시 안정|안정 상태|해소됐|회복됐|회복 후|재현되지 않|잔류 이상은 없|배제됐|사라졌|범위 안"
CUES = re.compile(f"(?P<continues>{POS_CUE})|(?P<cleared>{CLEAR_CUE})")


def status_rule(line):
    if UNKNOWN_CUE.search(line):
        return UNKNOWN
    matches = list(CUES.finditer(line))
    if not matches:
        return UNKNOWN
    return CONTINUES if matches[-1].lastgroup == "continues" else CLEARED


def rule_output(rows):
    states = []
    for row in rows:
        result = [UNKNOWN, UNKNOWN]
        if row.complete:
            for slot, line in zip(row.binding, row.lines):
                result[slot] = status_rule(line)
        states.append(result)
    return state_output(rows, np.eye(3)[states])
