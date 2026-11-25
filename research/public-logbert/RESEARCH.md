# HDFS v1: does event order beat event counts?

## Dataset and split

The [LogHub HDFS v1 dataset](https://github.com/logpai/loghub/tree/master/HDFS) is a Hadoop benchmark run. The [official Zenodo archive](https://zenodo.org/records/8196385) contains maintainer-provided event traces and anomaly labels. The runner validates the upstream archive size and MD5, records SHA-256, and reads the two preprocessed CSV members directly. It scanned 575,061 blocks and 11,175,629 events; it did not train on the full archive. Labels derive from the upstream process, not semiconductor outcomes. The third-party dataset follows [LogHub's separate use terms](https://github.com/logpai/loghub/blob/master/LICENSE); no original data are distributed here.

The main experiment (seed 42) selected 24,000 normal training blocks, 3,000 validation blocks and 3,000 test blocks. The seed 7 check reruns the entire protocol with a different split and initialization; it is not an independent corpus or an initialization-only sensitivity test. A complete block belongs to a single partition. We hash the first 64 event IDs to keep identical visible sequences together, then select blocks by a deterministic block-ID hash. The model vocabulary is fitted only on selected normal training traces. The runner checks for collisions after unknown-token encoding and removes cross-split collisions before scoring. Counts, exclusions, and class balance are in the result JSONs.

This is a **signature-disjoint stress test**, not a chronological or machine-held-out split. Several blocks can repeat the same signature within each partition, so the count of blocks is not a count of unique workflows or independent operating conditions. The upstream parser already processed the corpus; this benchmark does not validate raw log parsing. Equal caps do not imply equal unique-signature counts or anomaly prevalence across seeds.

## Models and controls

- **Nearest normal event counts:** L1 distance to the closest normal-training count vector, including unknown-event counts. This method captures sequence length and event composition without order.
- **Unigram / trigram:** Add-one-smoothed mean event surprise; trigram backs off as required and sees at most two preceding IDs.
- **Causal Transformer:** Same size and primitives as the bidirectional model, with a future-key mask. The shifted input starts with a reserved ID; at each position the model predicts the next original event. Use mean next-event NLL. Its training objective supervises more targets per batch than MLM, so this is architecture-matched, not compute- or target-matched.
- **Bidirectional BERT-style encoder:** Randomly initialized, two blocks, four heads, width 32. Corrupt 15% of ordinary event positions with BERT's 80/10/10 recipe and optimize only selected labels. At evaluation, mask one event at a time and average its NLL over the trace. The model sees later events; this is retrospective triage.
- **Shuffled-order sensitivity:** Use the same trained bidirectional model, permute event order in validation and test while preserving counts, and recalibrate only on shuffled validation normals. Shuffling creates out-of-distribution input. A drop shows sensitivity to order, not that order generalizes to real equipment or provides an advantage over event counts.

All methods score the same selected validation and test blocks. Fit each alert threshold at the validation-normal 99th percentile, then use strict `score > threshold` on test. Ties can place validation FPR below 1%, and test FPR can exceed 1%. Average precision (AP) is threshold-free; alert counts, precision, recall and FPR come from the fixed operating point.

## Results

### Seed 42

train: 24,000 blocks, 0 anomalies, 1,993 unique encoded signatures, validation: 3,000 blocks, 53 anomalies, 290 unique encoded signatures, test: 3,000 blocks, 62 anomalies, 375 unique encoded signatures. End-to-end CPU runtime: 94.1 s. Both neural models had 28,690 parameters.

| Method | Test AP ↑ | Caught ↑ | False alarms ↓ | Precision | Recall | Test FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nearest normal event counts | 0.9247 | 62/62 | 3/2,938 | 95.38% | 100.00% | 0.10% |
| Bidirectional masked encoder | 0.6517 | 60/62 | 53/2,938 | 53.10% | 96.77% | 1.80% |
| Causal Transformer (same size) | 0.5726 | 41/62 | 43/2,938 | 48.81% | 66.13% | 1.46% |
| Trigram | 0.3485 | 32/62 | 54/2,938 | 37.21% | 51.61% | 1.84% |
| Unigram | 0.3480 | 35/62 | 53/2,938 | 39.77% | 56.45% | 1.80% |
| Masked encoder, shuffled inputs | 0.0373 | 0/62 | 24/2,938 | 0.00% | 0.00% | 0.82% |

Highest AP: **Nearest normal event counts**. See the [full results](hdfs-results-seed42.json) for validation FPR, thresholds, split audits, and training loss.

### Seed 7

train: 24,000 blocks, 0 anomalies, 1,750 unique encoded signatures, validation: 3,000 blocks, 271 anomalies, 566 unique encoded signatures, test: 3,000 blocks, 310 anomalies, 647 unique encoded signatures. End-to-end CPU runtime: 97.0 s. Both neural models had 28,690 parameters.

| Method | Test AP ↑ | Caught ↑ | False alarms ↓ | Precision | Recall | Test FPR |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Nearest normal event counts | 0.9757 | 310/310 | 15/2,690 | 95.38% | 100.00% | 0.56% |
| Bidirectional masked encoder | 0.5047 | 73/310 | 32/2,690 | 69.52% | 23.55% | 1.19% |
| Causal Transformer (same size) | 0.7682 | 90/310 | 32/2,690 | 73.77% | 29.03% | 1.19% |
| Trigram | 0.3364 | 57/310 | 34/2,690 | 62.64% | 18.39% | 1.26% |
| Unigram | 0.3900 | 71/310 | 48/2,690 | 59.66% | 22.90% | 1.78% |
| Masked encoder, shuffled inputs | 0.0708 | 0/310 | 13/2,690 | 0.00% | 0.00% | 0.48% |

Highest AP: **Nearest normal event counts**. See the [full results](hdfs-results-seed7.json) for validation FPR, thresholds, split audits, and training loss.


## Why the simplest method won

Not one test anomaly shared an exact event-count vector with a normal training block: **0/62** at seed 42 and **0/310** at seed 7. In contrast, **2,935/2,938** and **2,675/2,690** test normals had matches. This is a benchmark shortcut on these splits: order sensitivity can be real and still not be the most useful signal. Seed 42 test anomaly prevalence was 2.07%; seed 7 was 10.33%, so do not read an AP difference across the two seeds as a controlled model improvement.

## Reproduce

From the repository root with Python 3.12 and PyTorch 2.9.1 CPU:

```sh
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python -B -m unittest discover -s research/public-logbert -v
python -B research/public-logbert/hdfs_benchmark.py --download --seed 42 --epochs 5 --train-blocks 24000 --eval-blocks 3000 --out hdfs-42-local.json
python -B research/public-logbert/hdfs_benchmark.py --seed 7 --epochs 5 --train-blocks 24000 --eval-blocks 3000 --out hdfs-7-local.json
```

The download caches the archive outside the repository. For an existing official ZIP, use `--archive PATH` instead. These runs were on cloud Linux CPU; the public-data experiment was not verified in Google Colab. The README Colab badge refers to the separate synthetic causal experiment.

## Limits and next check

This is not a field test or a reproduction of all LogBERT objectives. Deploying an encoder is not justified by a synthetic or HDFS score. A stronger next check isolates **order-only** failures with matching event counts and records where the encoder loses, then repeats on authorized equipment logs with an independent machine/time split, measured alert cost, and known sensor/recipe changes. Compare latency as well as AP: leave-one-out masked scoring requires a separate forward input for every scored position, unlike next-event scoring.

[Implementation](../../docs/IMPLEMENTATION.md) · [Primary sources and protocol](../../docs/EVALUATION.md)
