"""Persist a local model and decode its two neural state predictions."""

from dataclasses import asdict
import os
from pathlib import Path
from tempfile import NamedTemporaryFile

import torch

from .encoding import PacketError, STATES
from .model import CandidateStateTransformer, ModelConfig


FORMAT = "equip-triage/candidate-state-transformer-v1"
ENCODING = "nfkc-casefold-utf8-bytes-ref-v1"


def save_checkpoint(model, path):
    """Atomically save only model configuration, encoding version and tensor weights."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = None
    try:
        with NamedTemporaryFile(
            dir=path.parent, prefix=".memo-", delete=False
        ) as stream:
            temp = stream.name
            torch.save(
                {
                    "format": FORMAT,
                    "encoding": ENCODING,
                    "config": asdict(model.config),
                    "state_dict": model.state_dict(),
                },
                stream,
            )
        os.replace(temp, path)
    finally:
        if temp is not None:
            Path(temp).unlink(missing_ok=True)


def load_checkpoint(path, *, device="cpu"):
    """Load a locally trained v1 checkpoint in evaluation mode."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if (
        not isinstance(checkpoint, dict)
        or checkpoint.get("format") != FORMAT
        or checkpoint.get("encoding") != ENCODING
    ):
        raise ValueError("Unsupported model checkpoint or token encoding")
    model = CandidateStateTransformer(ModelConfig(**checkpoint["config"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval()


@torch.inference_mode()
def predict(model, record):
    """Report model states and evidence-line numbers, then apply the exact-count gate."""
    result = {
        "model": "CandidateStateTransformer",
        "action": "hold",
        "states": [],
        "confidence": None,
        "threshold": model.config.threshold,
    }
    try:
        batch = model.encode([record])
    except PacketError as exc:
        return {**result, "reason": str(exc)}
    probabilities = model(batch)[0].softmax(-1)
    confidence, ids = probabilities.max(-1)
    labels = [STATES[i] for i in ids.tolist()]
    binding = batch.binding[0].tolist()
    result["states"] = [
        {
            "domain": c["domain"],
            "tag": c["tag"],
            "state": labels[i],
            "score": float(confidence[i]),
            "evidence_line": 1 + binding.index(i),
        }
        for i, c in enumerate(record["candidates"])
    ]
    result["confidence"] = float(confidence.min())
    if "unknown" in labels:
        result["reason"] = "unknown_state"
    elif labels.count("continues") != 1:
        result["reason"] = "no_single_continuing_check"
    elif result["confidence"] < model.config.threshold:
        result["reason"] = "below_threshold"
    else:
        result["reason"] = "one_continues_other_cleared"
        result["action"] = record["candidates"][labels.index("continues")]["domain"]
    return result
