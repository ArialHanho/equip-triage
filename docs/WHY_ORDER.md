# Why test event order?

Equip Triage started with a practical question: when an inspection result looks wrong, can we reconstruct what the equipment did and identify which records to inspect first? The replay software preserves and groups the recorded evidence. It does not decide that a product passed, that a tool failed, or that a specific event caused a problem.

A sequence model solves a narrower problem. Consider two illustrative completed sessions with the same event IDs and counts:

```text
Expected:    LOAD → CONTACT_WARNING → RETEST → RELEASE
Order fault: LOAD → CONTACT_WARNING → RELEASE → RETEST
```

An event-count detector assigns both the same count vector. When a validated procedure explicitly says that retest must precede release, a deterministic check is the best first choice: it is inexpensive and points to an exact violated condition. Learned scoring is worth evaluating when the expected ordering varies with known context or is not encoded in a rule. Even then, a high score means “inspect these records,” not “hold or release the lot automatically.”

## What changed after HDFS

In the [public HDFS study](../research/public-logbert/RESEARCH.md), nearest-normal event counts outperformed both Transformer objectives in two held-out splits. In neither split did a test anomaly have exactly the same event counts as a normal training block. That result makes HDFS a useful baseline check, but not a sufficient test of whether learning order helps. An independent [audit of common log datasets](https://doi.org/10.1145/3660768) identifies the same class of evaluation problem.

The [order-aware experiment](../research/order-aware/README.md) compares verified normal traces with reordered counterparts. Both retain the same event inventory. The original model reconstructs masked event IDs; an order-sensitive variant explicitly learns to distinguish intact and reordered traces. [ALBERT's sentence-order objective](https://arxiv.org/abs/1909.11942) is a related idea, not a claim that this is a new general architecture or a reproduction of ALBERT's NLP pipeline.

## What a useful result would show

1. **Does order add signal?** Report matched-pair ranking on held-out traces. An order-insensitive score must tie within every pair.
2. **Does it help at a usable operating point?** Freeze thresholds using validation normals and report test recall, false alarms, and the count of each. A higher AP that only comes with an unacceptable false-alarm rate is not a deployment case.
3. **What does it cost?** Report CPU scoring time and model size alongside performance. A single pooled score and repeated token masking have different inference costs.
4. **What could explain the result without a Transformer?** Compare short-context and explicit precedence methods. If a simple rule wins, say so and keep the learned model optional.

This distinction is the contribution: a reproducible comparison that tells reviewers **when** a model is worth using. A generated order-fault test validates sensitivity to its own fault classes. It cannot establish production accuracy, a yield improvement, or transfer to an unseen tool or recipe.
