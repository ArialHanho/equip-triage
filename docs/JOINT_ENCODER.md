# Joint log and memo encoder · joint-v1

`equip_triage.joint` returns two candidate state distributions, learned source pointers and a first-check decision for an equipment/lot cutoff. The CPU pilot trains locally from random initialization. The larger comparison has a separate CUDA notebook and checkpoint directory. [Synthetic example packets](../examples/joint-observations.jsonl) · [Colab notebook](../notebooks/equip-triage-joint-colab.ipynb) · [Protocol and measurements](../research/joint_encoder/REPORT.md)

## Run the native model

Use Python 3.11+ and a PyTorch build appropriate for the device. The package inference path needs PyTorch. The study driver also needs the Python numerical stack used for the baselines.

```sh
python -m pip install .
python -m pip install scikit-learn==1.8.0
python -m research.joint_encoder.run --profile pilot --work artifacts/joint-pilot prepare
python -m research.joint_encoder.run --profile pilot --work artifacts/joint-pilot fit --arm joint --device cpu --seed 17 --microbatch 4
python -m equip_triage.joint --model artifacts/joint-pilot/models/joint-17/model.pt --input examples/joint-observations.jsonl
```

The four public inputs are synthetic training examples. The first two show the same packet in English and Korean. One demonstrates a duplicate-sequence conflict. The fourth was chosen from training to show a returned check. The recorded CPU checkpoint returned `hold, hold, hold, data`. These examples verify the interface.

```python
import json
from pathlib import Path
from equip_triage.joint import batch, load_model, predict, prepare

packet = json.loads(Path("examples/joint-observations.jsonl").read_text().splitlines()[0])
model, threshold = load_model("artifacts/joint-pilot/models/joint-17/model.pt")
snapshot = prepare(packet)
tensors = model(batch([snapshot]).to(next(model.parameters()).device))
embedding = tensors["candidate"]  # [1, 2, 32]
answer = predict(model, packet, threshold)
print(answer["action"], answer["candidates"][0]["evidence"])
```

## Input window

| Field | Meaning |
|---|---|
| `equipment`, `lot` | Exact scope for the decision. |
| `as_of` | Integer observation/availability cutoff, in seconds on the caller's common clock. |
| `first_sequence`, `through_sequence` | Inclusive expected sequence range for this packet. |
| `logs[]` | Records have `id, equipment, lot, seq, observed, available, kind`. A `register` row provides `tag, domain`. A `sample` row provides `tag, channel, reading`. A `marker` contributes sequence occupancy. |
| `notes[]` | An `observation` line carries `id, equipment, lot, tag, observed, available, text`. The tag is supplied by the caller. |
| Observed vocabulary | Two registered tags map to different domains from `measurement, data, setup, equipment`. Channels are `a, b`, and readings are `above, within`. These describe observations. |

A component is visible only when its observation and availability times are both at or before the decision. The log sequence must also fall in the requested range. Service and resolution entries are excluded. The same timestamp convention applies to both modalities. The ingestion layer has to supply these fields. The older six-column replay CSV lacks availability times, registered candidate channels and reading bands. Its fields do not determine a joint packet.

The window retains source IDs for audit. Equivalent observations at the same sequence retain their IDs together. Contradictory payloads at the same sequence and contradictory same-time notes retain their source IDs and trigger a hold. Source-level checks include gaps including the boundary, clock reversal, ambiguous registration, incomplete or stale observations, and missing channel coverage. The freshness limit is 30 seconds in this synthetic contract. There are at most 32 input units and 160 normalized UTF-8 bytes per note. An oversized window or a rejected note is flagged. These values require task-specific validation for another installation.

## Learned model and policy

Each row supplies its full normalized text or the canonical channel/reading pair, `modality`, anonymous candidate slot, capped age and within-stream order. The two registered domains seed candidate queries. Equipment IDs, source IDs, case IDs and adjudicated targets are excluded from embeddings. A single 32-wide bidirectional block reuses the project's four-head Q/K/V implementation to encode each evidence row. The CLS representation is combined with source metadata. For each registered check, its query reads eligible memo lines. The resulting vector conditions the second reader over eligible log rows. A shared head maps the candidate, memo and log vectors to three state logits. Separate supervised source pointers allow a null result. The model has 44,453 parameters. [Row and candidate encoder](../equip_triage/joint/model.py) · [Editable overview](assets/joint-overview.drawio) · [Editable encoder detail](assets/joint-encoder.drawio)

The API supplies `raw_state_action`, the auxiliary `raw_direct_action`, the operational `action`, and `reason`. Each candidate has `state`, `probability` and `evidence.log` / `evidence.memo` containing the original IDs. Pointers are predictions, so they can cite the wrong row or return an empty list. Duplicate-sequence and note conflicts also return a separate conflict trace. The final policy requires exactly one continuing candidate, a cleared counterpart, sufficient confidence and a clean source window. A semantic disagreement between a note and a valid log reading must be interpreted by the state model. A source-format guard cannot resolve its meaning. [Input preparation](../equip_triage/joint/sources.py) · [Output policy](../equip_triage/joint/inference.py)

## GPU training and restart

Open the self-contained notebook in Google Colab, select a GPU runtime, then run all its cells. It prints the actual GPU, validates 6,912 training, 1,152 validation and 2,880 final records against canonical hashes, then fits the native encoder, matched controls and the multilingual pretrained comparator. The final evaluator needs every arm's sealed validation receipt. The notebook explicitly stops on missing CUDA. A previously completed T4 run belongs to the older memo-only model. [Joint Colab notebook](../notebooks/equip-triage-joint-colab.ipynb)

The equivalent commands from a checked-out project with CUDA PyTorch are below. Choose a private persistent `WORK` directory to survive runtime replacement. The notebook uses already-mounted Drive when available and otherwise uses ephemeral VM storage. It does not authenticate an account or publish a model.

```sh
python -m pip install scikit-learn==1.8.0 transformers==4.57.3 'safetensors>=0.5,<1'
WORK=artifacts/joint-v1-colab-main
python -m research.joint_encoder.run --profile main --work "$WORK" prepare
python -m research.joint_encoder.run --profile main --work "$WORK" fit --arm joint --device cuda --seed 17 --microbatch 4
python -m research.joint_encoder.run --profile main --work "$WORK" fit-baselines
for arm in pooled parallel unbound log_only memo_only direct; do
  python -m research.joint_encoder.run --profile main --work "$WORK" fit --arm "$arm" --device cuda --seed 17 --microbatch 4
done
python -m research.joint_encoder.run --profile main --work "$WORK" fetch-public
python -m research.joint_encoder.run --profile main --work "$WORK" fit --arm public --device cuda --seed 17 --microbatch 4
python -m research.joint_encoder.run --profile main --work "$WORK" evaluate --device cuda --seed 17
```

Use the same fit command and save directory after a pause. The atomic `models/<arm>-<seed>/resume.pt` carries weights, AdamW moments, RNG states, completed update count, epoch, next batch and source/device configuration. By default it saves after 100 updates and at epoch boundaries. `--max-steps N` pauses at an absolute completed step. `model.pt` is the smaller inference checkpoint with format `equip-triage/joint-v1`, trained tensors, architecture config and validation threshold. The public arm also needs its adjacent `tokenizer/` directory. Retain private checkpoints locally. The multi-arm study produces `manifest.json`, frozen fit/calibration receipts and `evaluation-17.json`. For the two prespecified replications run the same eight neural arms and evaluation with seed 41 or 73. [Frozen protocol](../research/joint_encoder/protocol.md)

The public comparison fine-tunes the Apache-2.0 multilingual DistilBERT at revision `45c032ab32cc946ad88a166f7cb282f58c753c2e`. The published model card includes both English and Korean. Each of the five original files is checked against a SHA-256. Its WordPiece encoder reads the same complete normalized units and shares the new state, evidence and release path. Its pretrained row encoder, tokenizer, parameter count and cost differ from the scratch system. [Pinned public model](https://huggingface.co/distilbert/distilbert-base-multilingual-cased/tree/45c032ab32cc946ad88a166f7cb282f58c753c2e) · [Verified file hashes](../research/joint_encoder/public_base.py)
