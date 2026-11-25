"""Shared state and source heads for two registered checks across logs and memos."""

from dataclasses import dataclass
import math
import torch
from torch import nn

from ..encoding import DOMAINS
from ..model import EncoderBlock
from .sources import MAX_BYTES, MAX_UNITS

PUBLIC_ID = "distilbert/distilbert-base-multilingual-cased"
PUBLIC_REV = "45c032ab32cc946ad88a166f7cb282f58c753c2e"
MODES = ("joint", "parallel", "unbound", "pooled", "log_only", "memo_only", "direct")


@dataclass
class Batch:
    texts: tuple
    row_ids: torch.Tensor
    modalities: torch.Tensor
    owners: torch.Tensor
    ages: torch.Tensor
    order: torch.Tensor
    domains: torch.Tensor

    def to(self, device):
        return Batch(
            self.texts,
            *(
                v.to(device)
                for v in (
                    self.row_ids,
                    self.modalities,
                    self.owners,
                    self.ages,
                    self.order,
                    self.domains,
                )
            ),
        )


def batch(snapshots):
    if not snapshots or any(
        len(s.registry) != 2 or len(s.units) > MAX_UNITS for s in snapshots
    ):
        raise ValueError("Supply in-window snapshots with two registered checks")
    width = max(1, max(len(s.units) for s in snapshots))
    arrays = [
        torch.full((len(snapshots), width), v, dtype=torch.long)
        for v in (0, 0, -1, 0, 0)
    ]
    row_ids, kinds, owners, ages, order = arrays
    text_ids = {"": 0}
    domains = []
    for i, snap in enumerate(snapshots):
        domains.append([DOMAINS.index(d) for _, d in snap.registry])
        for j, u in enumerate(snap.units):
            row_ids[i, j] = text_ids.setdefault(u.text, len(text_ids))
            kinds[i, j] = 1 if u.modality == "log" else 2
            owners[i, j], ages[i, j], order[i, j] = (
                u.slot,
                min(u.age, 60),
                min(u.order, 31),
            )
    return Batch(tuple(text_ids), *arrays, torch.tensor(domains))


class ByteRows(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.tokens = nn.Embedding(258, width, padding_idx=0)
        self.positions = nn.Embedding(MAX_BYTES + 1, width)
        self.norm = nn.LayerNorm(width)
        self.layer = EncoderBlock(width, 4)

    def forward(self, texts):
        device = self.tokens.weight.device
        ids = [[1, *(n + 2 for n in s.encode("utf-8"))] for s in texts]
        n = max(map(len, ids))
        if n > MAX_BYTES + 1:
            raise ValueError("Evidence exceeded the common raw byte window")
        t = torch.tensor([s + [0] * (n - len(s)) for s in ids], device=device)
        h = self.norm(self.tokens(t) + self.positions(torch.arange(n, device=device)))
        return self.layer(h, t.ne(0))[:, 0]


class MultilingualRows(nn.Module):
    """Public WordPiece row encoder. The common input window always preserves complete rows."""

    def __init__(self, width, assets, *, initialize_public):
        super().__init__()
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            assets, local_files_only=True, trust_remote_code=False
        )
        cfg = AutoConfig.from_pretrained(
            assets, local_files_only=True, trust_remote_code=False
        )
        if cfg.model_type != "distilbert" or cfg.dim != 768 or cfg.n_layers != 6:
            raise ValueError(
                "Expected the pinned multilingual DistilBERT configuration"
            )
        cfg.dropout = cfg.attention_dropout = 0.0
        if initialize_public:
            self.encoder = AutoModel.from_pretrained(
                assets,
                config=cfg,
                local_files_only=True,
                use_safetensors=True,
                trust_remote_code=False,
            )
        else:
            self.encoder = AutoModel.from_config(cfg, trust_remote_code=False)
        self.project = nn.Linear(cfg.dim, width)

    def forward(self, texts):
        # Special tokens are added by the tokenizer only at the row boundaries.
        t = self.tokenizer(
            list(texts), return_tensors="pt", padding=True, truncation=False
        )
        if t["input_ids"].size(1) > MAX_BYTES + 2:
            raise ValueError(
                "Public tokenizer exceeded the common complete-row contract"
            )
        t = t.to(self.project.weight.device)
        return self.project(self.encoder(**t).last_hidden_state[:, 0])


class Reader(nn.Module):
    def __init__(self, width):
        super().__init__()
        self.q = nn.Linear(width, width)
        self.k = nn.Linear(width, width)
        self.v = nn.Linear(width, width)
        self.null = nn.Parameter(torch.zeros(width))

    def forward(self, query, rows, eligible, pooled):
        null = self.null.expand(rows.size(0), 1, -1)
        values = torch.cat((rows, null), dim=1)
        mask = torch.cat((eligible, torch.ones_like(eligible[..., :1])), -1)
        scores = (
            self.q(query) @ self.k(values).transpose(-1, -2) / math.sqrt(rows.shape[-1])
        )
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
        if pooled:
            # The readout uses ordinary means. Pointer supervision is identical across controls.
            chosen = torch.cat((eligible, ~eligible.any(-1, keepdim=True)), -1)
            weights = chosen.to(rows.dtype) / chosen.sum(-1, keepdim=True)
        else:
            weights = scores.softmax(-1)
        return weights @ self.v(values), scores


class JointEncoder(nn.Module):
    """Encode a cutoff snapshot and return candidate embeddings, state/route and source logits."""

    def __init__(
        self,
        mode="joint",
        width=32,
        backbone="bytes",
        assets=None,
        *,
        initialize_public=True,
    ):
        super().__init__()
        if (
            mode not in MODES
            or width < 4
            or width % 4
            or backbone not in ("bytes", "distilmbert")
        ):
            raise ValueError("Unsupported encoder configuration")
        self.mode, self.width, self.backbone = mode, width, backbone
        self.rows = (
            ByteRows(width)
            if backbone == "bytes"
            else MultilingualRows(width, assets, initialize_public=initialize_public)
        )
        self.modality, self.owner = nn.Embedding(3, width), nn.Embedding(3, width)
        self.age, self.order = nn.Embedding(61, width), nn.Embedding(32, width)
        self.domain = nn.Embedding(len(DOMAINS), width)
        self.row_norm = nn.LayerNorm(width)
        self.memo_reader, self.log_reader = Reader(width), Reader(width)
        self.fuse = nn.Sequential(
            nn.Linear(3 * width, 2 * width),
            nn.GELU(),
            nn.Linear(2 * width, width),
            nn.LayerNorm(width),
        )
        self.states = nn.Linear(width, 3)
        self.route_slot, self.route_hold = nn.Linear(width, 1), nn.Linear(width, 1)

    def forward(self, b):
        slots = torch.arange(2, device=b.domains.device)
        q = self.domain(b.domains) + self.owner(slots)
        rows = self.row_norm(
            self.rows(b.texts)[b.row_ids]
            + self.modality(b.modalities)
            + self.owner(b.owners.clamp_min(0))
            + self.age(b.ages)
            + self.order(b.order)
        )
        owner_ok = b.owners[:, None, :].eq(slots[None, :, None])
        if self.mode == "unbound":
            owner_ok = torch.ones_like(owner_ok)
        log_mask = (
            owner_ok & b.modalities[:, None, :].eq(1) & (self.mode != "memo_only")
        )
        note_mask = (
            owner_ok & b.modalities[:, None, :].eq(2) & (self.mode != "log_only")
        )
        is_mean = self.mode in ("pooled", "direct")
        note, note_scores = self.memo_reader(q, rows, note_mask, is_mean)
        conditioned = self.mode in ("joint", "unbound")
        log, log_scores = self.log_reader(
            q + note if conditioned else q, rows, log_mask, is_mean
        )
        candidate = self.fuse(torch.cat((q, note, log), -1))
        route = torch.cat(
            (
                self.route_slot(candidate).squeeze(-1),
                self.route_hold(candidate.mean(1)),
            ),
            dim=-1,
        )
        return {
            "candidate": candidate,
            "state": self.states(candidate),
            "direct": route,
            "log": log_scores,
            "memo": note_scores,
        }
