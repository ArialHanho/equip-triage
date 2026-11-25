# Project index

**Goal:** Review equipment events and operator notes for first-check triage.

**Current state:** The CSV viewer shows event sequences and memo text together and applies rules. The standalone PyTorch model reads two registered checks and two memo lines. It does not consume the CSV event sequence.

[Run the viewer and model](README.md) · [한국어](README.ko.md) · [Python API and input schema](docs/MEMO_MODEL.md) · [Frozen evaluation evidence](docs/EVALUATION.md)

**Open question:** Validate whether event-derived candidates and independently reviewed notes can improve first-check selection, including how disagreements should be held rather than forced into a route.
