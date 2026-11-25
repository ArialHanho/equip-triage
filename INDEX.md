# Project index

**Goal:** Build a transparent inspection-log replay tool and evaluate hand-implemented sequence models against simple baselines.

**Current evidence:** Read-only CSV replay is runnable. The causal model and three comparison methods have synthetic results in [v0.3](research/v03/RESEARCH.md); these do not validate a real factory. The separate [masked encoder probe](research/public-logbert/SYNTHETIC_MLM.md) confirms right-context use on a designed synthetic task only.

**Next question:** On held-out public log sequences, does bidirectional context add anything beyond unseen tokens, counts, length, and short prefixes? See the [source and evaluation register](docs/EVALUATION.md).

**Start here:** [README](README.md) · [Baseline research](research/RESEARCH.md) · [Contribution guide](CONTRIBUTIONS.md) · [Release checklist](RELEASE_CHECKLIST.md).
