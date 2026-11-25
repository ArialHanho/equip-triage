"""Encode a visible two-check registry and two referenced memo lines."""

from dataclasses import dataclass
import re
import unicodedata

import torch
from torch.nn.utils.rnn import pad_sequence


DOMAINS = ("measurement", "data", "setup", "equipment")
STATES = ("continues", "cleared", "unknown")
PAD, CLS, SEP = 0, 1, 2
DOMAIN_START, SLOT_START, REF, BYTE_START = 3, 7, 9, 10
VOCAB_SIZE = BYTE_START + 256
PREFIX_LENGTH = 4
ORDINALS = (
    ("first listed check", "앞 점검 항목"),
    ("second listed check", "뒤 점검 항목"),
)


class PacketError(ValueError):
    """A packet is missing information required for a candidate decision."""


@dataclass(frozen=True)
class MemoBatch:
    ids: torch.Tensor  # [batch, 2 lines, padded length]
    domains: torch.Tensor  # [batch, 2 candidates], domain token IDs
    binding: torch.Tensor  # [batch, 2 lines], candidate index named by each line

    def to(self, device):
        return MemoBatch(
            self.ids.to(device), self.domains.to(device), self.binding.to(device)
        )

    def select(self, indices):
        return MemoBatch(
            self.ids[indices], self.domains[indices], self.binding[indices]
        )


def _normalize(text):
    return unicodedata.normalize("NFKC", text).casefold()


def _encode_packet(record, limit):
    if not isinstance(record, dict):
        raise PacketError("invalid_record")
    items = record.get("candidates")
    if not isinstance(items, list) or len(items) != 2:
        raise PacketError("incomplete_registry")
    tags, domains = [], []
    for item in items:
        if not isinstance(item, dict) or not isinstance(item.get("tag"), str):
            raise PacketError("incomplete_registry")
        tag = _normalize(item["tag"].strip())
        if item.get("domain") not in DOMAINS or not re.fullmatch(
            r"[a-z][a-z0-9_-]{0,31}", tag
        ):
            raise PacketError("invalid_candidate")
        tags.append(tag)
        domains.append(DOMAIN_START + DOMAINS.index(item["domain"]))
    if tags[0] == tags[1] or domains[0] == domains[1]:
        raise PacketError("ambiguous_registry")

    memo = record.get("memo")
    if not isinstance(memo, str):
        raise PacketError("missing_memo")
    lines = [" ".join(_normalize(s).split()) for s in memo.splitlines() if s.strip()]
    if len(lines) != 2:
        raise PacketError("expected_two_lines")
    # Tag boundaries use ASCII because a Korean particle can directly follow a tag.
    aliases = {}
    for slot, tag in enumerate(tags):
        for alias in (tag, *ORDINALS[slot]):
            aliases[alias] = slot
    parts = []
    for alias in sorted(aliases, key=len, reverse=True):
        part = re.escape(alias)
        if alias.isascii():
            part = rf"(?<![a-z0-9_-]){part}(?![a-z0-9_-])"
        parts.append(part)
    ref_pattern = re.compile("|".join(parts))
    clauses, binding = [], []
    for line in lines:
        matches = list(ref_pattern.finditer(line))
        named = {aliases[m.group()] for m in matches}
        if len(named) != 1:
            raise PacketError("unresolved_reference")
        slot = named.pop()
        if not any(c.isalnum() for c in ref_pattern.sub(" ", line)):
            raise PacketError("missing_observation")
        try:
            line.encode("utf-8")
        except UnicodeEncodeError as exc:
            raise PacketError("invalid_text") from exc
        body, start = [], 0
        for match in matches:
            body.extend(
                BYTE_START + b for b in line[start : match.start()].encode("utf-8")
            )
            body.append(REF)
            start = match.end()
        body.extend(BYTE_START + b for b in line[start:].encode("utf-8"))
        if len(body) > limit:
            raise PacketError("line_too_long")
        binding.append(slot)
        clauses.append(torch.tensor([CLS, *domains, SLOT_START + slot, *body, SEP]))
    if binding[0] == binding[1]:
        raise PacketError("non_unique_binding")
    return clauses, domains, binding


def encode_packets(records, max_evidence_tokens=160):
    """Build a CPU tensor batch. Raise PacketError for an incomplete packet."""
    if max_evidence_tokens < 1:
        raise ValueError("max_evidence_tokens must be positive")
    rows = [_encode_packet(r, max_evidence_tokens) for r in records]
    if not rows:
        raise ValueError("Supply at least one packet")
    clauses, domains, binding = zip(*rows)
    ids = pad_sequence([line for pair in clauses for line in pair], batch_first=True)
    return MemoBatch(
        ids.reshape(len(rows), 2, -1), torch.tensor(domains), torch.tensor(binding)
    )
