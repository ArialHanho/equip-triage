# Where should event time enter attention?

This offline experiment asks whether a signed time-difference bias inside self-attention improves detection beyond giving the same timestamps to the encoder input. The test uses **four unseen permutations of familiar stages**. The vocabulary, two synthetic recipe profiles and per-stage timing ranges are shared across partitions. The experiment measures transfer to a new ordering of known stages.

## Reproduce

From the repository root, in a Python environment with CPU PyTorch installed, run the following command. The recorded run uses Python 3.12.14, PyTorch 2.9.1+cpu and one PyTorch compute thread. Installing that build in a fresh environment uses `python -m pip install --index-url https://download.pytorch.org/whl/cpu torch==2.9.1`.

```sh
python research/time-attention/study.py --out time-attention-local.json
```

The command generates the traces in memory, audits the pair and split constraints, fits four rules and 18 randomly initialized encoders, then writes aggregate results. It requires no external data or model checkpoints. [Source](study.py) · [Recorded run](results.json)

## Paired traces and procedure families

A trace starts with a START token and a recipe-profile token and ends with END. Each of four stages emits a stage marker, READ, an optional or mandatory ACK, and CLOSE. A family is the complete permutation of those four stages. Each family supplies 64 matched pairs, balanced across both recipe profiles and both fault types.

For example, in invented profile A, stages 0 and 2 take 9–14 ticks from the stage marker to READ. Stages 1 and 3 take 2–6 ticks. ACK is mandatory at stages 2 and 3 and appears with probability 0.5 at stages 0 and 1. Profile B reverses both assignments. All other intervals take 1–3 ticks. These numbers are synthetic construction parameters.

| Paired fault | Change to the normal | Quantities identical within the pair |
| --- | --- | --- |
| Timing | Swap the read intervals of one long-wait and one short-wait stage. | Profile, full event sequence, multiset of intervals, total duration |
| Order | Swap a required ACK with the immediately following CLOSE. | Profile, event counts, full timestamp vector, total duration |

The seeded generator partitions all 24 families into 16 train, four validation and four test families. That yields 1,024 train pairs, 256 validation pairs and 256 test pairs. Both members stay together. The audit also checks that no event sequence appears in different partitions. Family IDs and fault types are evaluation labels. The models see the profile token, stage markers and event codes. Time-enabled variants also see the timestamp sequence. Repeated random initializations use this one fixed split.

## Encoders and controls

All encoders use learned event and absolute position embeddings, two bidirectional blocks, width 32, four heads, and a linear fault classifier on START. Each block explicitly computes Q, K and V, masks padding keys, aggregates values, and applies residual connections, layer normalization and a GELU feed-forward layer. The classifier uses a completed trace. All encoder variants train on the same labeled normal/fault pairs with binary cross-entropy, ten epochs, batch size 64 and AdamW at 0.002. The shared parameters and batch order start identically within each seed.

- **Input time** adds a linear projection of `[log1p(t_i − t_previous), log1p(t_i − t_start)]` to the token embedding. The first gap is zero.
- **Relative time bias** changes each attention score to `dot(q_i, k_j) / sqrt(8) + b[head, bucket(t_i − t_j)]`. Zero has its own bin. Nonzero differences use the sign and the magnitude `min(8, 1 + floor(log2(abs(difference))))`. Each block starts its 17 × 4 table at zero and learns it with the classifier. With input time enabled, this adds 136 trainable coefficients across the two blocks.
- **Position control** uses the same learned table and signed buckets on the token-index difference. Combining it with input time matches the active parameter count of the full time model. Additional arms disable both time paths, use only position bias, or use only time bias. Every encoder allocates the same modules. Parameters receiving gradients range from 18,529 in the untimed encoder to 18,729 when both paths are used.
- **Rule controls** fit training normals. Count profiles score multiset rarity. A smoothed trigram with context backoff scores sequence likelihood. Global precedence counts inversions of uniquely occurring events with invariant order. The conditional rule uses known stage boundaries and the marker-to-read interval. For each profile and stage, it learns the observed read-time range and the set of legal event subsequences. Its score is the maximum range violation or unseen-pattern indicator.

## Operating point and measured cost

For each method and seed, the cutoff is the 254th smallest of 256 validation-normal scores (nearest-rank 99th percentile). An alert requires a score strictly above the cutoff. Ties can yield zero validation alerts. The cutoff is frozen before test scoring. A pair win means the corrupted trace scores *strictly* above its matched normal independent of the threshold. Ties are recorded separately.

| Method | Timing TP / 128 | Order TP / 128 | False alerts / 256 | Timing pair wins / 128 | CPU ms/trace |
| --- | ---: | ---: | ---: | ---: | ---: |
| Count profile | 0 | 0 | 0 | 0 | 0.0007 |
| Trigram | 4 | 128 | 9 | 0 | 0.0417 |
| Global precedence | 0 | 0 | 0 | 0 | 0.0020 |
| Conditional stage rule | 128 | 128 | 0 | 128 | 0.0032 |
| No time | 0 / 0 / 0 | 123 / 128 / 128 | 0 / 0 / 0 | 0 / 0 / 0 | 0.0506 / 0.0533 / 0.0523 |
| Input time | 1 / 5 / 4 | 128 / 128 / 123 | 1 / 11 / 10 | 65 / 56 / 53 | 0.0551 / 0.0478 / 0.0525 |
| Position bias only | 0 / 0 / 0 | 128 / 128 / 128 | 0 / 0 / 0 | 0 / 0 / 0 | 0.0567 / 0.0513 / 0.0531 |
| Input time + position bias | 0 / 0 / 0 | 128 / 128 / 122 | 2 / 2 / 7 | 65 / 53 / 60 | 0.0553 / 0.0536 / 0.0547 |
| Time bias only | 0 / 1 / 3 | 128 / 128 / 128 | 0 / 3 / 6 | 46 / 45 / 69 | 0.0583 / 0.0594 / 0.0598 |
| Input time + time bias | 0 / 3 / 1 | 127 / 128 / 121 | 0 / 5 / 2 | 63 / 57 / 64 | 0.0613 / 0.0607 / 0.0612 |

Slashes give seeds **41 / 42 / 43** on the same 256 test pairs. Each rule has a single deterministic result. A timed model can hit a timing fault at the cutoff while still ranking it below its matched normal. The trigram illustrates another pitfall: its four timing detections have four corresponding false alerts on the exact same event sequences, with all 128 timing pairs tied.

At seed 42 the input-only encoder detects 5/128 timing faults and raises 11/256 false alerts. Adding the relative time bias yields 3/128 and 5/256. The three seeds consistently exchange detections for fewer alerts. Timing-pair wins are 56 versus 57 at seed 42, with no consistent gain across seeds. At equal active parameter count, the input-plus-position control detects zero timing faults across the three seeds. The conditional rule detects both fault types perfectly. Its stage features directly match the generator’s constraints and the two read-duration bands are disjoint.

The CPU column reports process time per trace over one batched scoring pass on 512 traces, on an Intel Xeon Platinum 8573C. Encoder batches contain up to 64 traces. The measurement includes Python scoring and tensor preparation. It measures throughput on this machine and is sensitive to runtime conditions. Per-family true positives, false positives, pair wins, ties, fitted thresholds and fitting times are available for every method. [Raw aggregates](results.json)

This experiment evaluates synthetic timing and order faults. Four held-out permutations reuse the same stage definitions. The experiment provides no estimate for a new stage definition, a new plant, a production fault rate or streaming detection latency. The public HDFS benchmark and the replay app use separate workflows.

Relative-position attention and time-aware log detection have precedents. The contribution here is the paired, controlled comparison of two ways to supply timing on held-out synthetic procedure families. [Shaw et al., Self-Attention with Relative Position Representations (2018)](https://aclanthology.org/N18-2074/) · [Gong et al., LogETA (2024)](https://doi.org/10.1016/j.future.2024.03.028)
