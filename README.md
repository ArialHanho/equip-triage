**English** · [한국어](README.ko.md)

# Equip Triage

![Equip Triage](docs/assets/banner.svg)

**Offline inspection-log review and from-scratch Transformer research for equipment event sequences.**

Replay an equipment CSV by machine and lot. Flag duplicate, late, missing, and conflicting records. Show the original rows behind a review or hold so an operator can inspect the equipment record before drawing a process conclusion.

[Quickstart](#quickstart) · [Model architecture](#model-architecture) · [Benchmarks](#benchmarks) · [Research roadmap](#research-roadmap)

![Two separate paths: equipment CSV to evidence review; event IDs to a two-block encoder with masked-event and order heads. External models run only as comparisons.](docs/assets/architecture-en.svg)

## Quickstart

From the repository root, run with Python 3.11+; the replay app requires no external dependencies.

```sh
python app.py --serve
python app.py --json
```

Open the [English dashboard](http://127.0.0.1:8765/) or [한국어](http://127.0.0.1:8765/?lang=ko). Select **SIM-A / LOT-001** to inspect why the app returns **Hold**: sequence 6 is missing and two records disagree at sequence 7. The input is a [synthetic fixture](examples/events.csv). For another file, use `python app.py --input path/to/events.csv --serve` with UTF-8 and columns `equipment,lot,sequence,time,code,memo`. Timestamps need a UTC offset and each equipment/lot stream starts at sequence 1.

The app is an offline Python/SQLite replay, not a live collector or controller. The UI and evidence review do not invoke a research model.

## Model architecture

| Component | Role |
| --- | --- |
| [Event encoder](research/public-logbert/bert_mlm.py) | Discrete event IDs plus learned position embeddings; two Transformer blocks by default, width 32 and four attention heads. The repository implements Q/K/V projection, scaled dot-product attention with padding masking, and residual/LayerNorm around attention and the GELU feed-forward block using PyTorch primitives. |
| [Masked-event head](research/public-logbert/bert_mlm.py) | Predict a hidden event from both sides of a completed sequence; score anomalies by mean leave-one-out prediction loss. Train the reference encoder from random initialization. |
| [Order head](research/order-aware/study.py) | Add a 33-parameter linear scorer to the contextual `START` vector. Train with masked-event loss on normal sequences and binary classification on matched normal/reordered pairs; score a sequence in one encoder pass. |
| [External models](research/external-comparison/README.md) | Google BERT-Tiny/Mini act as comparison encoders; frozen SmolLM2-135M-Instruct is a four-shot decoder-only comparison. The project encoder requires neither model nor a pretrained checkpoint. |

The order head targets **illegal order with unchanged event counts**. The bidirectional model looks at completed sequences for retrospective review; the separate causal baseline predicts the next event from its prefix. Event IDs are not natural-language tokens. The dashboard supports Korean text, but the sequence encoder is not a Korean language model or an LLM.

## Benchmarks

![External benchmark: detection by fault type, false alerts, and batch CPU throughput. With the milestone filter, the custom model, BERT-Mini, and learned precedence each detect 192/192 faults with zero false alerts.](docs/assets/external-benchmark.png)

| Evaluation | Result and scope |
| --- | --- |
| [400 synthetic order-fault pairs](research/order-aware/README.md) | Event counts detect 0/400. The original masked model detects 88–234/400 across three seeds; adding the order head yields 399–400/400 with 4–6 false alerts on 400 normals. The learned precedence rule yields 400/400 and zero false alerts on this fixed grammar. |
| [192 synthetic pairs: two valid workflows](research/external-comparison/README.md) | Raw traces: custom order head 3/192, pretrained BERT-Mini 81/192 and frozen SmolLM2 3/192 (seed 42 comparison). A normal-train-only milestone filter brings the custom encoder and BERT-Mini to 192/192 with zero false alerts, matching the faster precedence rule. Both valid workflows are already present during training. Only one fault type is withheld, not a normal recipe. |
| [30,000 public Hadoop HDFS blocks](research/public-logbert/RESEARCH.md) | In the main split, the count baseline reaches AP 0.9247 versus 0.6517 for the masked encoder; counts also lead on the second split. HDFS is authentic IT telemetry, not factory equipment data. |
| [30,000 synthetic equipment sessions](research/v03/RESEARCH.md) | The state machine plus numeric rules detects 150/150 injected faults with 0/2,850 false alerts; the causal Transformer plus the same numeric rules detects 150/150 with 98 false alerts in the Linux CPU run. |

[Vector chart](docs/assets/external-benchmark.svg) · [Metrics CSV](research/external-comparison/metrics.csv) · [30K synthetic notebook](notebooks/equip-triage-v03-30k.ipynb) · [Data provenance](docs/EVALUATION.md)

**Finding:** The choice of event representation can matter more than the model. Neither the order head nor the milestone filter shows an advantage over a simple rule on the tested grammar. No industrial detection or yield claim follows from these experiments.

## Research roadmap

1. **Hold out entire normal recipes.** Exclude at least one legal recipe family from training and threshold calibration. Pair legal and invalid traces with the same counts, and include a case where `ACK` is optional in one recipe but required in another. Freeze the threshold on validation normals; report recall, false alerts by recipe, and CPU scoring cost. Compare count, trigram, learned precedence, and the same-size encoder on the same splits.
2. **Isolate the architecture change.** Relative position in attention is [prior work](https://arxiv.org/abs/1803.02155); the question here is whether event timing helps equipment triage. Compare standard QKV with a small relative event-distance/time-gap bias, keeping training data and budget fixed. Compare against an input-only time-gap version and recipe-conditioned precedence. An architectural contribution requires measurable improvement on unseen legal recipes at a comparable false-alert rate. Treat this as a proposal, not an implemented model.
3. **Add industrial evidence separately.** Evaluate an offline sensor replay candidate from the [PHM Society 2018 ion-mill challenge](https://phmsociety.org/conference/annual-conference-of-the-phm-society/annual-conference-of-the-prognostics-and-health-management-society-2018-b/phm-data-challenge-6/) once access and reuse terms are verified. Compare a maintenance window with same-tool, same-recipe reference windows. Its labels mark maintenance shutdowns, not the onset of faults or illegal recipe order. Keep sensor-replay results separate from the order benchmark.

[Implementation notes](docs/IMPLEMENTATION.md) · [Contribution guide](CONTRIBUTIONS.md) · [License](LICENSE)

MIT covers repository code and original synthetic fixtures, not third-party data or checkpoints.
