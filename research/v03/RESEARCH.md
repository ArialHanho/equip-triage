# v0.3: 30,000 synthetic equipment sessions

This benchmark extends the original causal pilot. It trains without pretrained weights and uses no production or company data.

| Split | Sessions | Event rows | Normal | Anomalous | Families / equipment IDs / conditions |
| --- | ---: | ---: | ---: | ---: | ---: |
| Train | 24,000 | 608,178 | 24,000 | 0 | 24 / 240 / 120 |
| Validation | 3,000 | 77,601 | 2,850 | 150 | 3 / 30 / 15 |
| Test | 3,000 | 80,511 | 2,850 | 150 | 3 / 30 / 15 |

Anomalies include wrong conditional release, missing unload, sequence gaps, long delays, repeated sequence numbers and backward timestamps. Normal traces vary measurement repetitions, legal waiting, check placement and timing jitter. Families share one authored grammar, not independent field distributions. There are zero exact payload duplicates, zero event-ID signatures shared across splits and 559 repeated signatures within splits. Validation and test each use just three synthetic families. The 5% anomaly rate is injected, not an industry estimate.

## Methods and operating point

- The 9,389-parameter causal Transformer implements multi-head Q/K/V attention, positional signals, residuals and future masking. Seed 42; 3 epochs; batch 128; CPU with one thread.
- The trigram sees at most two preceding tokens and uses backoff and add-one smoothing. Learned methods train on normal sessions only. The strongest state machine is given the synthetic workflow grammar.
- Event scores use maximum surprise over the completed session. Methods marked “numeric” share checks for delays and sequence integrity. A complete-session summary does not measure early-warning latency.
- Average precision uses a step integral with tied scores grouped. The fixed decision is `score > threshold`, where the threshold is the 99th percentile of normal validation scores. Do not retune it on test.

## Results

| Method | Test AP | Test recall | Validation false-alarm rate | Test false-alarm rate |
| --- | ---: | ---: | ---: | ---: |
| Trigram | 0.2924 | 25.33% | 0.00% | 0.00% |
| Causal Transformer | 0.5063 | 46.67% | 0.98% | 3.44% |
| Trigram + numeric rules | 0.8315 | 82.00% | 0.00% | 0.00% |
| Causal Transformer + numeric rules | 1.0000 | 100.00% | 0.98% | 3.44% |
| State machine + numeric rules | 1.0000 | 100.00% | 0.00% | 0.00% |

[Full Linux CPU results](results-30k.json). The independently run Colab CPU notebook reproduced the AP and recall with 97 false alarms for each Transformer variant, versus 98 on Linux. The state machine wins at the selected operating point; perfect ranking does not imply that a fixed threshold transfers. One seed and an authored grammar do not establish field performance.

## Reproduce

Python 3.12 recommended. From the repository root:

```sh
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cpu
python research/v03/scale_pilot.py --seed 42 --epochs 3 --out results-local.json --export-jsonl sessions-local.jsonl
python -m unittest discover -s research/v03 -p test_scale.py -v
```

The optional generated JSONL is not committed. Run the [standalone notebook on Colab](https://colab.research.google.com/github/ArialHanho/equip-triage/blob/c08386f3c4d0c7425a5dd8e7e60b259aa575628e/notebooks/equip-triage-v03-30k.ipynb). Additional seeds, independently designed families, and authorized domain data remain open checks.
