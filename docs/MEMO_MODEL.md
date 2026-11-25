# The candidate-state Transformer API

`equip_triage` is a standalone, installable Python package. Its runtime dependencies are PyTorch and the standard library. The working checkout supports `python -m equip_triage` directly. To install the package into an environment that already has PyTorch, setuptools and wheel, run `python -m pip install --no-index --no-deps --no-build-isolation .`. The installed `equip-triage` command accepts the same arguments. The code was exercised on Python 3.12 and CPU PyTorch 2.9.1.

## Train and read a saved model

```sh
python -m equip_triage train --data examples/memo-train.jsonl --out artifacts/memo.pt
python -m equip_triage predict --model artifacts/memo.pt --input examples/memo-input.jsonl
```

Training fits the encoder and state head together using cross-entropy on both candidate labels. Default settings are seed 42, 40 epochs, batch size 18 and AdamW at 0.003. Use `train --help` for overrides. Training holds the input file in memory. A checkpoint contains a format tag, the byte-encoding version, the complete model configuration (including its threshold), and its PyTorch state dictionary. Saving replaces the destination atomically. Inference loads these tensors in evaluation mode; it performs no fit or download. Weights are produced locally.

Each nonblank JSONL line is one packet. A training row adds exactly two `states` labels in candidate order. The four domain IDs are `measurement`, `data`, `setup` and `equipment`. The allowed states are `continues`, `cleared` and `unknown`. For example:

```json
{"candidates":[{"domain":"data","tag":"L7"},{"domain":"measurement","tag":"P3"}],"memo":"P3 passed again.\nL7 failed again.","states":["continues","cleared"]}
```

Prediction reads the same record without the labels. With `--input -`, JSONL comes from stdin. Each nonblank input line yields a JSON decision. An incomplete registry, missing line, reference without any observation text, ambiguous reference or oversized line yields `hold` with an empty state list and an explanatory reason. Valid packets return each candidate's domain, tag, neural state, state score and one-based evidence-line number. The confidence for the pair is the lower state score. A route requires exactly one continuing candidate and one cleared candidate, plus confidence at least the saved threshold. A score is the maximum of the three softmax components. The example threshold of 0.50 has no field calibration.

## Use the tensors

```python
from equip_triage import CandidateStateTransformer, ModelConfig, save_checkpoint, load_checkpoint, predict
import torch

packet = {
    "candidates": [{"domain": "data", "tag": "L7"}, {"domain": "measurement", "tag": "P3"}],
    "memo": "P3 passed again.\nL7 failed again.",
}
model = CandidateStateTransformer(ModelConfig(width=32, heads=4, layers=2))
batch = model.encode([packet])
logits, weights = model(batch, return_attention=True)
assert logits.shape == (1, 2, 3)
assert weights.shape == (1, 4, 2, 2 * batch.ids.shape[-1])

# One optimizer step with the two candidate labels from this illustration.
optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
target = torch.tensor([[0, 1]])  # continues, cleared
loss = torch.nn.functional.cross_entropy(logits.reshape(-1, 3), target.reshape(-1))
optimizer.zero_grad()
loss.backward()
optimizer.step()
save_checkpoint(model, "artifacts/api-example.pt")
restored = load_checkpoint("artifacts/api-example.pt")
decision = predict(restored, packet)
```

`CandidateStateTransformer.forward(batch)` returns raw logits. Requesting attention also returns the **candidate pooling weights**. For an installed model on another device, `load_checkpoint(path, device=...)` or `model.to(device)` works with `model.encode(...)`, which places inputs on the model's device. The CLI uses CPU. The low-level `encode_packets(records)` returns a CPU `MemoBatch`; use `.to(device)` to move it. `model.encode` raises `PacketError` for unsupported records, while `predict` converts that exception into a hold. The package accepts exactly two candidate records and two nonblank memo lines.

## Input and architecture

The visible registry binds each line to its named tag or one of the recognized order references (`first listed check`, `second listed check`, `앞 점검 항목`, `뒤 점검 항목`). Each line must name a different candidate. Text is normalized with Unicode NFKC, case folding and whitespace normalization. Every reference becomes one `REF` token and the remainder becomes UTF-8 byte IDs. All 256 bytes and ten reserved IDs have fixed meanings. The default limit is 160 evidence tokens per line, measured after reference replacement. Headers add four tokens and SEP adds one. A line can use Korean particles immediately after a tag. Binding reads references only; the Transformer determines the status of each check.

| Quantity | Default shape | Meaning |
| --- | --- | --- |
| Input | `[B,2,L]` | CLS, both domain IDs, the line's bound slot, memo evidence and SEP; right-padded |
| Encoder states | `[2B,L,32]` | Two shared-weight blocks with four heads, eight features per head, and a 32 → 128 → 32 GELU FFN |
| Candidate queries | `[B,4,2,8]` | Projection of the domain-token embedding plus learned candidate slot |
| Keys and values | `[B,4,2L,8]` | Projections of the two separately contextualized lines |
| Eligibility mask | `[B,1,2,2L]` | Each candidate can attend only to memo tokens on its bound line. All heads share the mask |
| Pooling weights | `[B,4,2,2L]` | Masked scaled dot products after softmax |
| State logits | `[B,2,3]` | One linear head shared across candidates after the residual addition and layer normalization |

The encoder uses the standard bidirectional attention and residual design. The task-specific part is the observable evidence binding, the two candidate queries with a shared three-state readout, and the count-based decoder. The package extracts the bound-query path from this repository’s frozen memo implementation. It replaces the fitted BPE tokenizer with fixed UTF-8 byte encoding, removes the comparison heads and adds a versioned model API and checkpoint. Research checkpoints and scores belong to their original tokenizer and protocol. The frozen test gave the query arm 0/112 primary pairs in each language for every reported seed. Five examples sharing the training grammar verify the software workflow; performance on new operators' memos remains unmeasured. [Frozen encoder](../research/candidate-states/encoder.py) · [Original Q/K/V block](../research/public-logbert/bert_mlm.py) · [Study report](../research/candidate-states/REPORT.md)

The editable figure is original artwork for this package, with an English page and a Korean page. Open the source in diagrams.net, select a page, edit its native text, tokens, layers, or connected arrows, or export the edited page as an SVG. The arrangement of tokens, embeddings and heads draws on BERT Figure 1. Here the left panel fits labelled candidate states from scratch and the right runs a saved checkpoint. The package implements supervised state training. [Editable source](assets/memo-transformer.drawio) · [English SVG](assets/memo-transformer-en.svg) · [Korean SVG](assets/memo-transformer-ko.svg) · [Devlin et al., Figure 1](https://aclanthology.org/N19-1423.pdf#page=3)
