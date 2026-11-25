# Detecting a reversed equipment operation

**Why add an order objective?** In the project's earlier 30K-block HDFS study, an event-count baseline won on both reported splits. That result left an engineering question open. If an equipment session contains every expected event exactly the expected number of times, can a detector still recognize that two operations occurred in the wrong order? This experiment isolates that question using synthetic pairs. The HDFS result supplies the motivation. All scores below come from the synthetic experiment. [Earlier project revision](https://github.com/ArialHanho/equip-triage/tree/c0c9f89)

The new head flagged 399–400 of 400 reversed sessions across three initializations. The existing masked language model (MLM) flagged 88–234 at the same validation false-alarm target. A rule fitted only to normal training sessions flagged all 400 reversals with zero false alarms. For a process with invariant milestones like these, the rule gave the best operating point and lower CPU cost.

## What changed

The existing two-layer, four-head, width-32 bidirectional encoder learns to reconstruct masked event IDs on normal sessions. Its anomaly score averages the negative log probability of each event, hiding that event in a separate forward pass. The upgraded model keeps the same initial backbone parameters and MLM loss. A linear head with 33 additional parameters reads the contextual vector at START and estimates whether a session contains a swap. Both methods score a completed session.

Each training batch supplies 64 normal sessions for MLM. The upgrade additionally sees those 64 unmasked sessions and their 64 corrupted partners. Its objective adds binary cross entropy on their order labels to the MLM loss with weight one. The inference score is the order head's logit. Comparisons therefore measure the entire change, including the extra supervision and new scoring rule. They cannot attribute a gain solely to the encoder representation. Each initialization uses identical normal-session order, masking stream, 10 epochs and AdamW learning rate 0.001 in both models.

ALBERT's sentence-order prediction task uses swapped segments to train a coherence objective. Here the swap exchanges two equipment milestones within one trace. Landauer and colleagues audited public log benchmarks and found that many detectable anomalies do not require sequential information. Matching event counts within each pair makes an order-insensitive detector's limit directly testable. [Lan et al., ALBERT](https://arxiv.org/abs/1909.11942) · [Landauer et al., FSE 2024](https://doi.org/10.1145/3660768)

## Matched sessions and controls

Normal sessions contain START, CLAMP, ZERO, SCAN, RELEASE and END in that order. Each milestone is preceded by one to four randomly sampled observations from POLL, WAIT, SAMPLE and ACK. A corrupted partner swaps CLAMP/ZERO, ZERO/SCAN or SCAN/RELEASE while leaving all other positions untouched. The two traces have equal length and identical event counts. The three swap types occur 134, 133 and 133 times in test. A single seeded generator makes 1,600 training pairs, 400 validation pairs and 400 test pairs. Both partners stay together and every full trace is unique across the experiment.

The count score measures rarity of the sorted event multiset in normal training data. Trigram scores mean next-event surprise with add-one smoothing and backoff. The learned precedence baseline discovers events appearing exactly once in every normal training session, retains directed pairs whose order always holds and counts violated pairs. It receives no milestone names, generator rule or corrupted training examples. All methods fit on the same normal sessions. Only the order-head method receives the derived swap examples.

For each method, the 99th percentile of the 400 validation **normal** scores fixes the threshold. A test score must strictly exceed it to raise an alert. A pairwise win means the corrupted trace scored higher than its own normal partner. It does not require crossing the alert threshold. Pairwise deltas are meaningful within a method because the methods use different score units.

| Method | Seed | Pair wins / 400 | Corrupt flagged / 400 | Normal flagged / 400 | Both partners classified correctly / 400 | Fit CPU s | Score CPU ms/trace |
|---|---:|---:|---:|---:|---:|---:|---:|
| Count profile | – | 0 (400 ties) | 0 | 0 | 0 | 0.002 | 0.0008 |
| Trigram | – | 356 | 303 | 1 | 302 | 0.016 | 0.0334 |
| Learned precedence | – | 400 | 400 | 0 | 400 | 0.011 | 0.0042 |
| MLM | 41 | 399 | 234 | 3 | 231 | 3.689 | 0.7746 |
| MLM + order head | 41 | 399 | 399 | 4 | 395 | 7.716 | 0.0607 |
| MLM | 42 | 400 | 215 | 4 | 211 | 2.667 | 0.7501 |
| MLM + order head | 42 | 400 | 400 | 6 | 394 | 7.291 | 0.0576 |
| MLM | 43 | 400 | 88 | 4 | 84 | 2.588 | 0.7221 |
| MLM + order head | 43 | 400 | 399 | 4 | 395 | 7.657 | 0.0579 |

The original MLM ranked almost every pair correctly. Its normal scores across different sessions still overlapped the corrupted scores enough to miss many anomalies at the fixed cutoff. The order head improved detection at that cutoff and had the same pair-win counts as MLM at each seed. At seed 42 it caught all 400 reversals but flagged six normal sessions, a test false-positive rate of 1.5% after calibration at 1%. The precedence rule's perfect result follows from a generator with a stable total order. Within these traces it is cheaper and has fewer false alerts.

CPU costs use process CPU time and PyTorch 2.9.1+cpu with one thread. Training includes model construction and optimizer setup. Scoring covers all 800 test traces after validation scoring, with neural batch size 32 and masked copies evaluated in chunks of 128. The head costs extra to fit but needs one encoder pass per scoring batch. Timings are from one environment and describe batch throughput. Single-request latency and equipment deployment remain unmeasured.

## Reproduce

From the repository root with Python 3.12 and PyTorch 2.9.1 installed, run:

```sh
python -B research/order-aware/study.py --out results-local.json --pairs pair-scores-local.csv
```

The runner imports the released EventMLM backbone from the sibling public-logbert folder. It generates all inputs in memory and initializes both models from scratch. Model seeds default to 41, 42 and 43. The data seed is 20261125 for every run. The CSV contains one normal score, corrupted score and difference per test pair, method and seed. The SHA256 column identifies the normal sequence generated by the script. The JSON contains method-specific thresholds, median score differences, per-swap detection counts and CPU costs. Generate the full 3,600-row pair-level CSV with the command above. Two independent executions in the reference CPU environment produced the same file (SHA-256: `415f860802f7d55f0f8f94a1339bc26bdf6edbf77085e30d52af96dad32f1d23`); other platforms can differ numerically. [Recorded aggregate results](results.json).

The three initializations share the same 400 test pairs. The experiment measures variation across training seeds on a single grammar and uses the same swap mechanism during training and evaluation. Valid alternative workflows, unseen kinds of corruption, HDFS and physical equipment have not been evaluated with this head. This AI-assisted reference implementation is available under the [project license](../../LICENSE).
