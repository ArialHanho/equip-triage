"""The clause encoder and candidate-query readout, implemented in native PyTorch."""

from dataclasses import dataclass
import math

import torch
from torch import nn

from .encoding import (
    DOMAIN_START,
    DOMAINS,
    PAD,
    SEP,
    PREFIX_LENGTH,
    VOCAB_SIZE,
    MemoBatch,
    encode_packets,
)


@dataclass(frozen=True)
class ModelConfig:
    width: int = 32
    heads: int = 4
    layers: int = 2
    max_evidence_tokens: int = 160
    threshold: float = 0.5

    def __post_init__(self):
        dims = (self.width, self.heads, self.layers, self.max_evidence_tokens)
        if any(type(n) is not int or n < 1 for n in dims) or self.width % self.heads:
            raise ValueError(
                "Use positive integer dimensions and a width divisible by heads"
            )
        if not 0 <= self.threshold <= 1:
            raise ValueError("threshold must be between zero and one")


class EncoderBlock(nn.Module):
    """Bidirectional attention, followed by a GELU FFN, with post-norm residuals."""

    def __init__(self, width, heads):
        super().__init__()
        self.heads = heads
        self.head_width = width // heads
        self.qkv = nn.Linear(width, 3 * width)
        self.merge = nn.Linear(width, width)
        self.attention_norm = nn.LayerNorm(width)
        self.feed_forward = nn.Sequential(
            nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width)
        )
        self.output_norm = nn.LayerNorm(width)

    def forward(self, states, valid):
        rows, length, width = states.shape
        qkv = self.qkv(states).reshape(rows, length, 3, self.heads, self.head_width)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        attention = scores.masked_fill(~valid[:, None, None, :], float("-inf")).softmax(
            -1
        )
        mixed = (attention @ value).transpose(1, 2).reshape(rows, length, width)
        hidden = self.attention_norm(states + self.merge(mixed))
        return self.output_norm(hidden + self.feed_forward(hidden))


class CandidateStateTransformer(nn.Module):
    """Return three state logits per registered candidate, in registry order."""

    def __init__(self, config=None):
        super().__init__()
        self.config = config if config is not None else ModelConfig()
        width = self.config.width
        self.token = nn.Embedding(VOCAB_SIZE, width, padding_idx=PAD)
        self.position = nn.Embedding(
            self.config.max_evidence_tokens + PREFIX_LENGTH + 1, width
        )
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(
            EncoderBlock(width, self.config.heads) for _ in range(self.config.layers)
        )
        self.slot = nn.Embedding(2, width)
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.merge = nn.Linear(width, width)
        self.output_norm = nn.LayerNorm(width)
        self.state = nn.Linear(width, 3)

    def encode(self, records):
        """Bind raw records and move the padded batch to this model's device."""
        return encode_packets(records, self.config.max_evidence_tokens).to(
            self.token.weight.device
        )

    def _split_heads(self, x):
        heads = self.config.heads
        return x.reshape(
            x.size(0), x.size(1), heads, self.config.width // heads
        ).transpose(1, 2)

    def forward(self, batch: MemoBatch, *, return_attention=False):
        ids, domains, binding = batch.ids, batch.domains, batch.binding
        if (
            ids.ndim != 3
            or ids.size(1) != 2
            or ids.shape[:2] != domains.shape
            or domains.shape != binding.shape
        ):
            raise ValueError("Expected ids [B,2,L] and domains/binding [B,2]")
        size, _, length = ids.shape
        if not size or not PREFIX_LENGTH + 2 <= length <= self.position.num_embeddings:
            raise ValueError("Invalid batch size or padded line length")
        if any(t.dtype != torch.long for t in (ids, domains, binding)):
            raise ValueError(
                "Use integer token, domain and binding tensors with torch.long dtype"
            )
        if (
            not (
                (domains >= DOMAIN_START) & (domains < DOMAIN_START + len(DOMAINS))
            ).all()
            or (domains[:, 0] == domains[:, 1]).any()
        ):
            raise ValueError("A packet needs two distinct domain token IDs")
        if not torch.equal(
            binding.sort(dim=1).values,
            torch.arange(2, device=ids.device).expand(size, -1),
        ):
            raise ValueError("Every candidate must be bound to exactly one line")
        positions = torch.arange(length, device=ids.device)
        flat_ids = ids.reshape(2 * size, length)
        valid = flat_ids.ne(PAD)
        if not valid.any(dim=1).all():
            raise ValueError("Each line needs at least one nonpadding token")
        hidden = self.input_norm(self.token(flat_ids) + self.position(positions))
        for block in self.blocks:
            hidden = block(hidden, valid)

        memory = hidden.reshape(size, 2 * length, self.config.width)
        slots = torch.arange(2, device=ids.device)
        candidates = self.token(domains) + self.slot(slots)
        q = self._split_heads(self.query(candidates))
        k = self._split_heads(self.key(memory))
        v = self._split_heads(self.value(memory))
        evidence = (positions >= PREFIX_LENGTH) & ids.ne(PAD) & ids.ne(SEP)
        matches = binding[:, None, :, None].eq(slots[None, :, None, None])
        mask = (evidence[:, None, :, :] & matches).reshape(size, 1, 2, 2 * length)
        scores = q @ k.transpose(-2, -1) / math.sqrt(q.size(-1))
        weights = (
            scores.masked_fill(~mask, torch.finfo(scores.dtype).min).softmax(-1) * mask
        )
        pooled = (weights @ v).transpose(1, 2).reshape(size, 2, self.config.width)
        logits = self.state(self.output_norm(candidates + self.merge(pooled)))
        return (logits, weights) if return_attention else logits
