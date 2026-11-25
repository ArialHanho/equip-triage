"""A randomly initialized, bidirectional masked encoder for discrete log events."""
import math

import torch
from torch import nn
from torch.nn import functional as F

PAD, MASK, UNK = 0, 1, 2
IGNORE = -100


def pad_batch(sequences):
    """Return a right-padded CPU LongTensor. Each sequence has at least one event."""
    rows = [torch.as_tensor(seq, dtype=torch.long, device="cpu") for seq in sequences]
    if not rows or any(row.ndim != 1 or row.numel() == 0 for row in rows):
        raise ValueError("A batch must contain one or more nonempty sequences.")
    return nn.utils.rnn.pad_sequence(rows, batch_first=True, padding_value=PAD)


class EncoderBlock(nn.Module):
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

    def forward(self, states, valid, causal=False):
        batch, length, width = states.shape
        qkv = self.qkv(states).reshape(batch, length, 3, self.heads, self.head_width)
        query, key, value = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        # Every real query can attend to real keys on both sides.
        scores = scores.masked_fill(~valid[:, None, None, :], float("-inf"))
        if causal:
            future = torch.ones((length, length), dtype=torch.bool, device=states.device).triu(diagonal=1)
            scores = scores.masked_fill(future, float("-inf"))
        weights = scores.softmax(dim=-1)
        attended = (weights @ value).transpose(1, 2).reshape(batch, length, width)
        states = self.attention_norm(states + self.merge(attended))
        return self.output_norm(states + self.feed_forward(states))


class EventMLM(nn.Module):
    def __init__(self, vocab_size, max_len, width=32, heads=4, layers=2):
        super().__init__()
        if min(width, heads, max_len, layers) < 1 or width % heads or vocab_size <= UNK + 1:
            raise ValueError("Invalid dimensions or no ordinary event classes.")
        self.vocab_size = vocab_size
        self.max_len = max_len
        self.tokens = nn.Embedding(vocab_size, width, padding_idx=PAD)
        self.positions = nn.Embedding(max_len, width)
        self.input_norm = nn.LayerNorm(width)
        self.blocks = nn.ModuleList(EncoderBlock(width, heads) for _ in range(layers))
        self.readout = nn.Linear(width, vocab_size)

    def forward(self, ids, *, causal=False):
        if ids.ndim != 2 or ids.shape[1] > self.max_len:
            raise ValueError("Expected [batch, length] within the configured length limit.")
        valid = ids.ne(PAD)
        if not valid.any(dim=1).all():
            raise ValueError("Each sequence must include at least one nonpadding token.")
        positions = torch.arange(ids.shape[1], device=ids.device)
        states = self.input_norm(self.tokens(ids) + self.positions(positions))
        for block in self.blocks:
            states = block(states, valid, causal=causal)
        return self.readout(states)


def corrupt_for_training(ids, vocab_size, generator):
    """Independently select 15% of ordinary events, with 80/10/10 corruption."""
    selected = ids.gt(UNK) & (torch.rand(ids.shape, generator=generator) < 0.15)
    labels = ids.masked_fill(~selected, IGNORE)
    branch = torch.rand(ids.shape, generator=generator)
    replace_random = selected & (branch >= 0.8) & (branch < 0.9)
    corrupted = ids.masked_fill(selected & (branch < 0.8), MASK)
    # A uniform random event may coincide with the original, as in BERT.
    alternatives = torch.randint(UNK + 1, vocab_size, ids.shape, generator=generator)
    corrupted[replace_random] = alternatives[replace_random]
    return corrupted, labels


def masked_loss(logits, labels):
    """Average cross entropy over selected targets. Empty masks yield a zero gradient."""
    count = labels.ne(IGNORE).sum()
    total = F.cross_entropy(
        logits.flatten(0, 1), labels.flatten(), ignore_index=IGNORE, reduction="sum"
    )
    return total / count.clamp_min(1)


def train_epoch(model, batches, optimizer, generator):
    """One epoch over padded CPU batches using fresh masks. Return token-weighted loss."""
    model.train()
    device = next(model.parameters()).device
    total_loss = 0.0
    selected_tokens = 0
    for ids in batches:
        corrupted, labels = corrupt_for_training(ids, model.vocab_size, generator)
        count = labels.ne(IGNORE).sum().item()
        if count == 0:
            continue
        optimizer.zero_grad(set_to_none=True)
        logits = model(corrupted.to(device))
        loss = masked_loss(logits, labels.to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * count
        selected_tokens += count
    if selected_tokens == 0:
        raise ValueError("No ordinary tokens were selected during this epoch.")
    return {"mean_loss": total_loss / selected_tokens, "selected_tokens": selected_tokens}


@torch.inference_mode()
def leave_one_out_nll(model, ids, chunk_size=128):
    """Mask each event separately and score its original ID. Return CPU nats and NaN pads."""
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive.")
    if (ids == MASK).any():
        raise ValueError("Scoring requires original event IDs.")
    model.eval()
    device = next(model.parameters()).device
    scores = torch.full(ids.shape, float("nan"), device="cpu")
    indices = ids.ne(PAD).nonzero()
    for chunk in indices.split(chunk_size):
        if not len(chunk):
            continue
        rows, columns = chunk.T
        corrupted = ids.index_select(0, rows)
        batch = torch.arange(len(rows), device=ids.device)
        targets = corrupted[batch, columns].clone()
        corrupted[batch, columns] = MASK
        logits = model(corrupted.to(device))[batch.to(device), columns.to(device)]
        losses = F.cross_entropy(logits, targets.to(device), reduction="none")
        scores[rows.cpu(), columns.cpu()] = losses.cpu()
    return scores
