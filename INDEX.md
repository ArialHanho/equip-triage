# Project index

**Goal:** Build a transparent inspection-log replay tool and evaluate hand-implemented sequence models against simple baselines.

**Current evidence:** Read-only CSV replay is runnable. The causal model and three comparison methods have synthetic results in [v0.3](research/v03/RESEARCH.md); these do not validate a real factory. The separate [masked encoder probe](research/public-logbert/SYNTHETIC_MLM.md) confirms right-context use on a designed synthetic task only.

**Public evidence:** See [30K HDFS results](research/public-logbert/RESEARCH.md) for controls, two split seeds, thresholds and failure cases. See the [implementation map](docs/IMPLEMENTATION.md) for the exact code and test boundaries.

**Next question:** Can any sequence model detect an order-only deviation when counts are identical? Build an order-controlled diagnostic and later test against authorized equipment logs by machine and time. See [sources and evaluation criteria](docs/EVALUATION.md).

**Start here:** [README](README.md) · [Baseline research](research/RESEARCH.md) · [Contribution guide](CONTRIBUTIONS.md) · [Release checklist](RELEASE_CHECKLIST.md).
