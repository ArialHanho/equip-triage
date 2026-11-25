# Public-data evaluation criteria

**Research question:** Can a randomly initialized bidirectional encoder exploit event order beyond what a simpler detector learns from event IDs, counts, or lengths?

| Source | What it supports | What it does not support |
| --- | --- | --- |
| [LogHub HDFS v1](https://github.com/logpai/loghub/tree/master/HDFS), [archive and terms](https://zenodo.org/records/8196385) | Block-level anomaly scoring on Hadoop traces. Supplied traces can test sequence modeling without pretending to parse raw logs. | Semiconductor-field accuracy, operator memo routing, early warning, or conclusions about production yield. |
| [UCI SECOM](https://archive.ics.uci.edu/dataset/179/secom), [raw archive](https://archive.ics.uci.edu/static/public/179/secom.zip) | An independent tabular pass/fail experiment on semiconductor measurements. The downloaded matrix contains 1,567 × 590 measurements; labels are separate (104 failures). | Event modeling, repair timelines, or a join to HDFS. The raw width differs from the catalog's 591-feature field. |

[Executed protocol, metrics, and reproducer](../research/public-logbert/RESEARCH.md).

## Acceptance criteria

1. Pin the data source and record checksums, terms, preprocessing, sample counts, and exclusions. Do not commit the upstream archive.
2. Split complete block IDs before creating windows or masks; fit token vocabulary and baselines on normal train blocks only. Report exact sequence overlap and use a signature-disjoint stress evaluation where feasible.
3. Use identical held-out data for a trigram, an order-insensitive token novelty/count/length baseline, and the masked-event model. Track whether shuffling event order disproportionately degrades the masked model. Novel IDs and count changes are known HDFS shortcuts.
4. Publish test average precision, alert counts, recall and false-alarm rate at thresholds frozen on normal validation data. Include the prevalence, seed, truncation policy and test coverage. Full-context scores are retrospective.

## Primary references

Reviewed 2026-11-25.

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762): scaled dot-product multi-head attention.
- [BERT](https://arxiv.org/abs/1810.04805): bidirectional encoding with masked prediction. The proposed model uses a small event vocabulary, not original BERT weights or a reproduction of its full training recipe.
- [LogBERT](https://arxiv.org/abs/2103.04475): a related log anomaly detector with multiple self-supervised objectives; masked-event prediction here is not a full LogBERT reproduction.
- [LogAnomaly](https://www.ijcai.org/proceedings/2019/658): prior work studying sequential and quantitative anomalies.
- [Landauer et al., FSE 2024](https://doi.org/10.1145/3660768): an audit of benchmark shortcuts, motivating explicit order-insensitive controls.
