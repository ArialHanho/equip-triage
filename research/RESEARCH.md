# Original pilots: causal events and memo routing

## Causal model

The 9,129-parameter reference in [event_transformer.py](event_transformer.py) implements four-head scaled dot-product attention, sinusoidal positions, residuals and a feed-forward network using PyTorch primitives. Weights are randomly initialized. There is no `nn.Transformer` or `MultiheadAttention` wrapper. At each position, the target is the next event; future keys are masked and padded targets do not enter the cross-entropy. This is not the bidirectional masked-event task.

The first pilot used 36 synthetic scenarios (12 normal train, 12 validation, 12 test). Each evaluation split contains three normal and nine anomalous scenarios; measurement repetition ranges are 1–6, 7–9, and 10–12 respectively. Both the trigram and Transformer achieved 9 TP / 0 FP / 3 TN / 0 FN. A small, separate long-context probe used identical recent suffixes with different earlier modes: the trigram got 2/4 test cases and the Transformer 4/4. Neither experiment establishes model superiority or field accuracy. See [scenarios](event_scenarios.csv), [first results](event-results.json), and [context predictions](long-context-results.json).

From the repository root (Python 3.12 recommended):

```sh
python -m pip install -r research/requirements-events.txt
python research/event_transformer.py --seed 42 --epochs 120 --out events-local.json
python research/long_context.py --seed 42 --out context-local.json
python -m unittest discover -s research -p test_events.py -v
```

[Self-contained original pilot notebook](../notebooks/equip-triage-scratch.ipynb). The separate [30K benchmark](v03/RESEARCH.md) scales this causal research, not the original memo task.

## Korean memo baselines

[study.py](study.py) uses an authored [synthetic Korean corpus](memo_corpus.tsv) (24 train / 12 validation / 16 test) with four known categories and unknown material. Keyword rules scored 0.3500 macro-F1; character TF-IDF with logistic regression scored 0.4917. See [pilot results](pilot-results.json) for routing and abstention details. This task is distinct from event sequence modeling.

```sh
python -m pip install -r research/requirements.txt
python research/study.py --out memo-local.json
python -m unittest discover -s research -p test_study.py -v
```

Neither research experiment is integrated with the replay app.
