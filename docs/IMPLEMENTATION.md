# Implementation map

The inspection replay app and sequence research solve different problems. Follow the source from a raw row to an audit trail, or from discrete event IDs to anomaly scores.

| Component | Implemented in this repository | Verification |
| --- | --- | --- |
| Log replay | [app.py](../app.py) preserves raw rows, normalizes UTC, reconstructs equipment/lot streams, detects gaps/conflicts and exposes a local read-only view. | Run the local replay and query `/report.json` to inspect reconstructed evidence. |
| Encoder backbone | [bert_mlm.py](../research/public-logbert/bert_mlm.py) builds input embeddings, two residual blocks, Q/K/V projections, head splitting and merging, padding-key masks, scaled attention, FFN, layer normalization and event logits. | The public masked-event diagnostic shows when right context changes the prediction. |
| Training and scoring | Random initialization. Given event IDs, sample supervised positions, corrupt inputs, compute cross-entropy there, and optimize with AdamW. At evaluation, hide each event separately, then average the negative log-probability of the observed events. | [Controlled masked-event probe](../research/public-logbert/SYNTHETIC_MLM.md) isolates the effect of a later cue. |
| Public benchmark | [hdfs_benchmark.py](../research/public-logbert/hdfs_benchmark.py) checks the pinned archive, joins the supplied trace/label tables, isolates visible signatures across splits, fits normal-only models, and calibrates alert thresholds on validation normals. | The runner emits the split audit, thresholds, scores and comparison results. |

## Tensor flow

For the public benchmark, the position limit is 64 and the hidden width is 32. Let *B* be batch size and *T* the padded sequence length.

```text
event IDs [B,T] + position indices
  -> learned embeddings [B,T,32]
  -> projected Q, K, V [B,4,T,8] each
  -> QKᵀ / √8 [B,4,T,T], mask padding keys, softmax
  -> weighted V, concatenate heads [B,T,32]
  -> residual + LayerNorm -> FFN (32 → 128 → 32) -> residual + LayerNorm
  -> second block -> masked-event logits [B,T,V]
```

PyTorch supplies tensors, linear/embedding/normalization primitives, loss, automatic differentiation and the optimizer. The code does not use `BertModel`, `TransformerEncoder` or a packaged attention module. This is a compact BERT-style event encoder, not an exact reproduction of the original NLP BERT or the complete LogBERT method.

The upstream HDFS maintainers provided event templates and block labels. This repository does **not** claim to learn an HDFS raw-log parser. See the [contribution guide](../CONTRIBUTIONS.md) for the distinction between reference work and independent contributions.
