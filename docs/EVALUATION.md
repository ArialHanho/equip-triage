# Evaluation archive

The [preserved snapshot](https://github.com/ArialHanho/equip-triage/blob/5b026353b9153db79f9e08d6365754c0a22a1d99/README.md) contains the former app documentation, notebooks, benchmark source, plots and metrics. It remains in the same repository's Git history. The current README and package document the runnable release; removing older files from the main branch does not change those experiments or their results.

| Evidence | What it shows |
| --- | --- |
| [Candidate states](https://github.com/ArialHanho/equip-triage/blob/5b026353b9153db79f9e08d6365754c0a22a1d99/research/candidate-states/REPORT.md) | On held-out synthetic wording families, scoped rules got 70/112 English and 98/112 Korean one-versus-two pairs right. The query encoder got 0/112 in both languages under that study protocol. The archived arm used a different tokenizer, so it is not a test of the included demo weights. |
| [Event order](https://github.com/ArialHanho/equip-triage/blob/5b026353b9153db79f9e08d6365754c0a22a1d99/research/order-aware/README.md) and [external models](https://github.com/ArialHanho/equip-triage/blob/5b026353b9153db79f9e08d6365754c0a22a1d99/research/external-comparison/README.md) | Controlled sequence comparisons; precedence rules matched or exceeded the best results in these generated cases. |
| [Public HDFS logs](https://github.com/ArialHanho/equip-triage/blob/5b026353b9153db79f9e08d6365754c0a22a1d99/research/public-logbert/RESEARCH.md) | An actual IT log benchmark, not semiconductor equipment validation. |
| [Synthetic equipment sessions](https://github.com/ArialHanho/equip-triage/blob/5b026353b9153db79f9e08d6365754c0a22a1d99/research/v03/RESEARCH.md) | Early end-to-end simulation and error-injection measurements, not shop-floor performance. |

The current CSV viewer and native memo Transformer are separate execution paths. The package demo does not establish performance on unfamiliar worker notes, and none of the archived experiments validate a joint field decision pipeline.
