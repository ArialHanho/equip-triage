# Project index

**Goal:** Build a transparent inspection-log replay tool and evaluate hand-implemented sequence models against simple baselines.

**Current evidence:** Read-only CSV replay is runnable. The causal model and three comparison methods have synthetic results in [v0.3](research/v03/RESEARCH.md); these do not validate a real factory. The separate [masked encoder probe](research/public-logbert/SYNTHETIC_MLM.md) confirms right-context use on a designed synthetic task only.

**Public evidence:** See [30K HDFS results](research/public-logbert/RESEARCH.md) for controls, two split seeds, thresholds and failure cases. See the [implementation map](docs/IMPLEMENTATION.md) for the exact code and test boundaries.

**Order-aware upgrade:** [Matched synthetic reversal benchmark](research/order-aware/README.md): the new head beats the original MLM at the fixed threshold; learned precedence wins the benchmark overall. [Why it exists](docs/WHY_ORDER.md).

**External comparison:** [Matched Google BERT and local LLM runs](research/external-comparison/README.md). Train-only event projection repaired most missed order faults on generated traces, but learned precedence rules match the best result in this controlled grammar.

**Next question:** How does the order head handle legal alternative paths and reversal patterns excluded from training? Then validate on authorized equipment logs with machine/time splits and operator-reviewed alerts.

**Start here:** [README](README.md) · [Baseline research](research/RESEARCH.md) · [Contribution guide](CONTRIBUTIONS.md) · [Release checklist](RELEASE_CHECKLIST.md).
