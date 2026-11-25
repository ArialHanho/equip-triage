![Equip Triage: inspection log replay and from-scratch sequence models](docs/assets/banner.svg)

[![Python 3.11+](https://img.shields.io/badge/Python-3.11%2B-3776AB)](#quickstart)
[![License: MIT](https://img.shields.io/badge/License-MIT-2CA58D)](LICENSE)
[![Open 30K benchmark in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/ArialHanho/equip-triage/blob/c08386f3c4d0c7425a5dd8e7e60b259aa575628e/notebooks/equip-triage-v03-30k.ipynb)

Replay inspection logs without losing the original evidence. Then test whether **an encoder built from scratch** can recognize unusual event context that simpler methods miss.

## What belongs in software, and what belongs in a model?

| Layer | Responsibility | Status |
| --- | --- | --- |
| **Log replay** | Parse CSV, preserve source rows, canonicalize timestamps to UTC, isolate equipment/lot streams, and flag duplicates, gaps, or conflicts. | Runnable local app using Python and SQLite. |
| **Sequence baselines** | Learn normal event order with a trigram and a hand-built causal Transformer. Compare both with an explicit state machine and shared timing/sequence rules. | Evaluated on 30,000 synthetic sessions. |
| **BERT-style encoder** | Implement token and position embeddings, multi-head Q/K/V attention, feed-forward layers, residuals, normalization, and a masked-event head. Train from random initialization. | [Backbone and masked-event diagnostic](research/public-logbert/SYNTHETIC_MLM.md) implemented and tested. Evaluated on public HDFS traces; see the comparison below. |

The causal model predicts the **next event from its prefix**. The BERT-style objective reconstructs a **masked event using both sides** of a completed sequence. The latter is for retrospective triage, not live alarms or online early detection. Event IDs are the tokens; this is not a Korean language model. Neither research model is wired to the replay app. No pretrained checkpoints are required.

## Quickstart

From the repository root, use Python 3.11+; the replay app has no external dependencies.

```sh
python app.py --serve
```

Open [localhost:8765](http://127.0.0.1:8765/). Stop with Ctrl+C. The bundled CSV is synthetic. Rows for the same lot ID on different machines stay separate; select a flagged stream to inspect the original record.

```sh
python app.py --json
python app.py --input examples/events.csv --serve --port 8766
```

The CSV header is `equipment,lot,sequence,time,code,memo`. Use UTF-8, an offset-aware timestamp (or `Z`), and sequence numbers starting at 1 for each equipment/lot pair. The app replays a file into an in-memory database; it is not a live collector, permanent historian, or equipment controller.

## Order-aware upgrade: when counts miss the fault

When an inspection tool records all expected events but performs them out of order, event counts cannot distinguish the fault. In 400 matched test pairs, the upgraded encoder adds a 33-parameter order head to the original two-layer masked-event model. Both models start from scratch. It scores a completed sequence in one encoder pass rather than separately masking and rescoring every event.

| Method | Reversals caught / 400 | False alarms / 400 normal |
| --- | ---: | ---: |
| Event counts | 0 | 0 |
| Trigram | 303 | 1 |
| Original masked-event model, 3 initializations | 88–234 | 3–4 |
| Masked-event model + order head, 3 initializations | 399–400 | 4–6 |
| Learned precedence rule | 400 | 0 |

The upgrade improved detection versus the original model at thresholds fixed on validation normals. The simpler rule won on the synthetic fixed-order workflow. Use verified rules for known interlocks; test learned scoring where valid order varies. These are generated equipment traces, not HDFS or production results.

[Why we built it](docs/WHY_ORDER.md) · [Full results and reproduction](research/order-aware/README.md) · [Implementation](research/order-aware/study.py)

## Measured: 30,000 public HDFS blocks

The same split and validation-only threshold rule were used for every method. The train/validation/test caps were 24,000 / 3,000 / 3,000. The table below is seed 42.

| Method | Test AP ↑ | Caught ↑ | False alarms ↓ |
| --- | ---: | ---: | ---: |
| Nearest normal event counts | 0.9247 | 62/62 | 3/2,938 |
| Bidirectional masked encoder | 0.6517 | 60/62 | 53/2,938 |
| Causal Transformer (same size) | 0.5726 | 41/62 | 43/2,938 |
| Trigram | 0.3485 | 32/62 | 54/2,938 |
| Unigram | 0.3480 | 35/62 | 53/2,938 |

**Nearest normal event counts had the highest AP in the main run.** The masked encoder reached AP 0.6517 versus 0.9247 for the counter of event IDs. Event counts also won the second split (AP 0.9757). The masked encoder beat the same-size causal model in the main run but lost in the second (AP 0.5047 vs. 0.7682). None of the test anomalies in either split shared an exact count vector with a normal training block.

[Full metrics, order-shuffle control, second split and commands](research/public-logbert/RESEARCH.md) · [What was implemented and how it was tested](docs/IMPLEMENTATION.md)

This is Hadoop IT telemetry supplied as preprocessed event traces, not a semiconductor field evaluation. [Data provenance](docs/EVALUATION.md) also explains why SECOM is a separate tabular task.

## Measured: 30,000 synthetic equipment sessions

24,000 normal train sessions, 3,000 validation, 3,000 test; 766,290 event rows total. The test set has 150 injected anomalies and 2,850 normal sessions. Fix each threshold on validation normals (target false-alarm rate ≤1%), then apply it unchanged to test.

| Method | Test AP ↑ | Anomalies caught ↑ | False alarms ↓ |
| --- | ---: | ---: | ---: |
| Trigram, events only | 0.2924 | 38/150 | 0/2,850 |
| Causal Transformer, events only | 0.5063 | 70/150 | 98/2,850 |
| Trigram + shared numeric rules | 0.8315 | 123/150 | 0/2,850 |
| Causal Transformer + shared numeric rules | 1.0000 | 150/150 | 98/2,850 |
| State machine + numeric rules | 1.0000 | 150/150 | 0/2,850 |

These are the [committed Linux CPU results](research/v03/results-30k.json). The Colab CPU run reproduced the AP and recall shown here with 97 Transformer false alarms instead of 98. **The state machine performed best at the chosen operating point.** It knows the generator grammar; the Transformer ranked this synthetic test perfectly with numeric rules but still fired on normal sessions at the threshold chosen on validation. That is not evidence of a production yield gain or field accuracy. See the [experiment protocol](research/v03/RESEARCH.md).

## Measured: can the encoder use right context?

A two-block, 26,867-parameter BERT-style encoder recovered 128/128 masked targets on held-out synthetic sequences when it could see the later cue, versus 32/128 when the cue was masked too. The cue is sufficient by design, so a lookup table could also solve this probe. This checks a property of the implementation, not anomaly-detection performance.

[Encoder code](research/public-logbert/bert_mlm.py) · [Method and reproduction](research/public-logbert/SYNTHETIC_MLM.md) · [Results](research/public-logbert/synthetic-results.json)

## Repository map

- [Project index](INDEX.md) · [Original causal and memo baselines](research/RESEARCH.md) · [Contribution guide](CONTRIBUTIONS.md)
- [30K experiment code](research/v03/scale_pilot.py) · [Notebook](notebooks/equip-triage-v03-30k.ipynb) · [Results](research/v03/results-30k.json)

MIT applies to repository code and original synthetic fixtures, not third-party datasets. No proprietary manufacturing data, live equipment integration, or production validation is included.
