"""Offline inference, source provenance and independent structural release policy."""

from pathlib import Path
import torch
from ..encoding import STATES
from .sources import MAX_UNITS, VERSION, SourceError, action, prepare
from .model import JointEncoder, PUBLIC_REV, batch


def route(states, domains, confidence, threshold, issues):
    tentative = action(states, domains)
    if issues:
        return "hold", "source_guard"
    if threshold is None:
        return "hold", "release_disabled"
    if tentative == "hold":
        return "hold", "state_gate"
    if confidence < threshold:
        return "hold", "below_threshold"
    return tentative, "eligible"


def load_model(path, device="cpu"):
    path = Path(path)
    obj = torch.load(path, map_location="cpu", weights_only=True)
    if obj.get("format") != VERSION:
        raise ValueError("Incompatible joint checkpoint")
    config = obj["config"]
    if config["backbone"] == "distilmbert" and obj.get("public_revision") != PUBLIC_REV:
        raise ValueError("Unexpected public base revision")
    model = JointEncoder(
        **config, assets=path.parent / "tokenizer", initialize_public=False
    )
    model.load_state_dict(obj["model"], strict=True)
    return model.to(device).eval(), obj["threshold"]


@torch.inference_mode()
def predict(model, packet, threshold=0.5):
    if threshold is not None and not 0 <= threshold <= 1:
        raise ValueError("Use a threshold from zero to one")
    try:
        s = prepare(packet)
    except SourceError as error:
        return {
            "format": VERSION,
            "action": "hold",
            "reason": "source_guard",
            "issues": [str(error)],
            "candidates": [],
        }
    report = {
        "format": VERSION,
        "equipment": s.equipment,
        "lot": s.lot,
        "as_of": s.cutoff,
        "issues": list(s.issues),
        "sequence_gaps": list(s.gaps),
        "conflicting_source_ids": list(s.conflicts),
        "registration_source_ids": list(s.registrations),
    }
    if "incomplete_registry" in s.issues or len(s.units) > MAX_UNITS:
        return {
            **report,
            "action": "hold",
            "reason": "source_guard",
            "candidates": [
                {
                    "tag": tag,
                    "domain": domain,
                    "state": "unknown",
                    "evidence": {"log": [], "memo": []},
                }
                for tag, domain in s.registry
            ],
        }
    b = batch([s]).to(next(model.parameters()).device)
    out = model(b)
    if model.mode == "direct":
        pr = out["direct"][0].softmax(-1)
        domains = [d for _, d in s.registry]
        choice = (*domains, "hold")[int(pr.argmax())]
        raw = (
            choice if threshold is not None and float(pr.max()) >= threshold else "hold"
        )
        return {
            **report,
            "action": "hold" if s.issues else raw,
            "reason": "source_guard" if s.issues else "direct_control",
            "raw_direct_action": raw,
            "threshold": threshold,
            "candidates": [
                {
                    "tag": tag,
                    "domain": domain,
                    "state": None,
                    "evidence": {"log": [], "memo": []},
                }
                for tag, domain in s.registry
            ],
        }
    probs = out["state"][0].softmax(-1)
    confidence, states = probs.max(-1)
    labels, domains = states.tolist(), [d for _, d in s.registry]
    selected, reason = route(
        labels, domains, float(confidence.min()), threshold, s.issues
    )
    candidates = []
    for j, (tag, domain) in enumerate(s.registry):
        evidence = {}
        for source in ("log", "memo"):
            i = int(out[source][0, j].argmax())
            evidence[source] = list(s.units[i].ids) if i < len(s.units) else []
        candidates.append(
            {
                "tag": tag,
                "domain": domain,
                "state": STATES[labels[j]],
                "probability": float(confidence[j]),
                "evidence": evidence,
            }
        )
    direct = int(out["direct"][0].argmax())
    return {
        **report,
        "action": selected,
        "reason": reason,
        "threshold": threshold,
        "raw_state_action": action(labels, domains),
        "raw_direct_action": (domains + ["hold"])[direct],
        "candidates": candidates,
    }
