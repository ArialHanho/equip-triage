"""Scratch BERT-family controls and shared, candidate-conditioned tensor readouts."""

from dataclasses import dataclass
from pathlib import Path
import math
import sys
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from data import ACTIONS
from routing import direct_output, state_output

# Public, directly implemented QKV/residual/LayerNorm block from this repository.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "public-logbert"))
from bert_mlm import EncoderBlock

WIDTH, HEADS, MAX_WORDS = 32, 4, 120
RESERVED = [
    "[PAD]",
    "[CLS]",
    "[SEP]",
    "[MEAS]",
    "[DATA]",
    "[SETUP]",
    "[EQUIP]",
    "[MISSING]",
    "[UNBOUND]",
    "[FIRST]",
    "[SECOND]",
]
PAD, CLS, SEP = 0, 1, 2
MODES = (
    "flat",
    "local_direct",
    "cls_states",
    "all_mean",
    "all_query",
    "bound_mean",
    "bound_query",
)


def train_tokenizer(rows):
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trainer = trainers.BpeTrainer(
        vocab_size=768,
        min_frequency=2,
        special_tokens=RESERVED,
        initial_alphabet=sorted(pre_tokenizers.ByteLevel.alphabet()),
        show_progress=False,
    )
    tok.train_from_iterator(
        sorted({s for row in rows for s in row.lines if s}), trainer=trainer
    )
    return tok


@dataclass
class Encoded:
    clauses: list
    flat: list
    domains: tuple
    binding: tuple


def encode(tok, row):
    words = [tok.encode(line, add_special_tokens=False).ids for line in row.lines]
    if any(len(word) > MAX_WORDS for word in words):
        raise ValueError("A memo line exceeds the tokenizer limit.")
    domain_ids = tuple(3 + d for d in row.domains)
    flat = [CLS, *domain_ids]
    clauses = []
    for slot, tokens in zip(row.binding, words):
        segment = [9 + slot, *tokens, SEP]
        clauses.append([CLS, *domain_ids, *segment])
        flat.extend(segment)
    return Encoded(clauses, flat, domain_ids, row.binding)


def pack(batch, flat):
    if flat:
        ids = nn.utils.rnn.pad_sequence(
            [torch.as_tensor(e.flat) for e in batch], batch_first=True
        )
    else:
        clauses = [torch.as_tensor(s) for e in batch for s in e.clauses]
        ids = nn.utils.rnn.pad_sequence(clauses, batch_first=True).reshape(
            len(batch), 2, -1
        )
    return (
        ids,
        torch.tensor([e.domains for e in batch]),
        torch.tensor([e.binding for e in batch]),
    )


class CandidateEncoder(nn.Module):
    def __init__(self, vocab, mode):
        super().__init__()
        self.mode = mode
        self.token = nn.Embedding(vocab, WIDTH, padding_idx=PAD)
        self.position = nn.Embedding(2 * MAX_WORDS + 8, WIDTH)
        self.input_norm = nn.LayerNorm(WIDTH)
        self.blocks = nn.ModuleList([EncoderBlock(WIDTH, HEADS) for _ in range(2)])
        self.slot = nn.Embedding(2, WIDTH)
        self.query = nn.Linear(WIDTH, WIDTH)
        self.key = nn.Linear(WIDTH, WIDTH)
        self.value = nn.Linear(WIDTH, WIDTH)
        self.merge = nn.Linear(WIDTH, WIDTH)
        self.output_norm = nn.LayerNorm(WIDTH)
        self.state = nn.Linear(WIDTH, 3)
        self.direct = nn.Linear(WIDTH, len(ACTIONS))
        self.local_direct = nn.Linear(2 * WIDTH, len(ACTIONS))
        self.cls_states = nn.Linear(2 * WIDTH, 6)

    def contextualize(self, ids):
        valid = ids.ne(PAD)
        h = self.input_norm(
            self.token(ids)
            + self.position(torch.arange(ids.size(-1), device=ids.device))
        )
        for block in self.blocks:
            h = block(h, valid)
        return h

    def head_split(self, t):
        return t.reshape(t.shape[0], t.shape[1], HEADS, WIDTH // HEADS).transpose(1, 2)

    def forward(self, ids, domains, binding):
        if self.mode == "flat":
            return self.direct(self.contextualize(ids)[:, 0])
        batch, clauses, length = ids.shape
        # The two clauses are separate batch entries; their token encoders cannot mix evidence.
        h = self.contextualize(ids.reshape(batch * clauses, length)).reshape(
            batch, clauses, length, WIDTH
        )
        cls = h[:, :, 0].reshape(batch, 2 * WIDTH)
        if self.mode == "local_direct":
            return self.local_direct(cls)
        if self.mode == "cls_states":
            return self.cls_states(cls).reshape(batch, 2, 3)
        flat = h.reshape(batch, 2 * length, WIDTH)
        slots = torch.arange(2, device=ids.device)
        candidate = self.token(domains) + self.slot(slots)
        # Eligible evidence excludes both header prefixes, separators and padding.
        real = (
            (torch.arange(length, device=ids.device) >= 4) & ids.ne(PAD) & ids.ne(SEP)
        )
        eligible = real[:, None, :, :].expand(-1, 2, -1, -1)
        if self.mode.startswith("bound"):
            eligible = eligible & binding[:, None, :, None].eq(
                slots[None, :, None, None]
            )
        eligible = eligible.reshape(batch, 1, 2, 2 * length)
        if self.mode.endswith("query"):
            q = self.head_split(self.query(candidate))
            k = self.head_split(self.key(flat))
            logits = (q @ k.transpose(-1, -2)) / math.sqrt(WIDTH // HEADS)
            weights = logits.masked_fill(~eligible, -1e9).softmax(-1) * eligible
        else:
            weights = eligible.to(flat.dtype) / eligible.sum(
                -1, keepdim=True
            ).clamp_min(1)
        values = self.head_split(self.value(flat))
        pooled = (weights @ values).transpose(1, 2).reshape(batch, 2, WIDTH)
        return self.state(self.output_norm(candidate + self.merge(pooled)))


def fit_net(rows, examples, tok, mode, seed):
    torch.manual_seed(seed)
    model = CandidateEncoder(tok.get_vocab_size(), mode)
    flat = mode == "flat"
    direct = mode in ("flat", "local_direct")
    loss_targets = torch.tensor(
        [ACTIONS.index(r.gold) if direct else r.states for r in examples]
    )
    inputs = [encode(tok, row) for row in rows]
    rng = torch.Generator().manual_seed(seed + 1000)
    opt = torch.optim.AdamW(model.parameters(), lr=0.002)
    for _ in range(12):
        for ix in torch.randperm(len(rows), generator=rng).split(64):
            logits = model(*pack([inputs[i] for i in ix.tolist()], flat))
            loss = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)), loss_targets[ix].reshape(-1)
            )
            opt.zero_grad()
            loss.backward()
            opt.step()
    active = sum(p.numel() for p in model.parameters() if p.grad is not None)
    model.eval()

    @torch.inference_mode()
    def predict(batch, name):
        encoded = [encode(tok, r) for r in batch]
        scores = torch.cat(
            [
                model(*pack(encoded[i : i + 64], flat)).softmax(-1)
                for i in range(0, len(batch), 64)
            ]
        ).numpy()
        if direct:
            return direct_output(batch, scores, guarded=not name.endswith("raw"))
        return state_output(batch, scores, exact=not name.endswith("wta"))

    return predict, active
