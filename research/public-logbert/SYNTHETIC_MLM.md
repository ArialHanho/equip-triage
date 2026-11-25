# Can the encoder recover an event from its right context?

After 16 epochs the encoder recovered all 128 hidden middle events in a held-out set of generated sequences. Hiding the later cue as well reduced the score to 32 of 128. This test checks whether information flows backward through the encoder during offline reconstruction. It does not estimate anomaly detection accuracy in HDFS or a factory.

## Reproduce the check

Run from `research/public-logbert` with Python 3.12 and PyTorch installed. The recorded run used PyTorch 2.9.1+cpu on a single CPU thread. Each invocation initializes new parameters and generates its inputs in memory.

```sh
python -B synthetic_check.py --seed 42
```

## Task and controls

Four sequences share the same two prefix events and the same fourth event. Within each quartet, the third event has four possible identities and the fifth event uniquely identifies it. The cue and the target use disjoint sets of IDs. All three remaining events come from a third set of eight IDs. For example, the model can use a final 7 to recover a hidden 3, even though 3 appears nowhere in the masked input.

The generator shuffles the 512 possible ordered triples of context IDs with seed 42. It assigns 224 triples to training and 32 different triples to testing, then generates all four modes for every chosen triple. This gives 896 training and 128 test sequences, with zero shared triples across the split. Each quartet contains four different answers to the same prefix. Hiding both the target and the final cue makes all four inputs identical, so any deterministic predictor can get at most one answer per quartet.

Training uses all event positions as potential targets. The evaluation always hides the middle event. Both the initial and trained evaluations use the same 128 inputs, without supplying the original target ID anywhere in an input. Top-1 prediction considers the full 19-class output.

| Model and visible context | Correct / 128 | Accuracy | Mean target NLL, nats |
| --- | ---: | ---: | ---: |
| Random initialization, later cue visible | 0 | 0.0% | 2.830448 |
| After training, later cue visible | 128 | 100.0% | 0.011756 |
| Same trained model, later cue hidden | 32 | 25.0% | 1.444201 |

The initial score is an observation for one seed. The 25% bound follows from the balanced quartet design. A four-entry lookup table given the cue would also solve this task. The results establish that the implementation learned this specific dependency.

## Architecture and scoring

The 26,867-parameter model embeds event IDs and learned absolute positions into 32 dimensions. Each of its two blocks forms query, key and value tensors for four heads, scales the dot products by the square root of the head width, masks padding keys, then applies softmax across all real keys. Each block has an attention residual, a feed-forward residual and layer normalization after each addition. A linear head projects each position to event logits.

Each training pass independently selects ordinary event positions with probability 0.15. Of the selected positions, the sampler masks 80%, substitutes a uniformly sampled ordinary event for 10%, and retains 10%. The cross-entropy loss uses the original IDs only at selected positions. AdamW runs for 16 fixed epochs with batches of 32 and learning rate 0.003. The synthetic data checks only the masked language modeling objective. The architecture and corruption scheme draw on BERT. [Devlin et al., 2019, §3.1](https://aclanthology.org/N19-1423.pdf)

For a completed log sequence, `leave_one_out_nll` masks each event in a separate copy and returns its negative log probability in nats. The model has access to events on both sides, so the score applies to retrospective triage. The original project’s causal next-event predictor remains a separate model. The published LogBERT method additionally optimizes a hypersphere objective. Comparisons with that method would require matching its objectives and evaluation protocol. [Guo et al., 2021](https://arxiv.org/abs/2103.04475)

Seven automated checks cover right-context sensitivity, invariance to right padding, sampled corruption frequencies, gradients only at selected targets, zero loss and gradient for empty masks, masking of each scored event including unknown events, and quartet separation across the split.
