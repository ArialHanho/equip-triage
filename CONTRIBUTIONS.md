# Contribution guide

Keep changes reproducible and document the evaluation conditions. The encoder uses PyTorch tensor operations, layers, autograd, and the optimizer. It does not claim a new Transformer architecture.

Useful follow-up contributions:

1. Reimplement and explain the Q/K/V projections, head reshaping, and attention masks; add shape and invariance tests.
2. Check causal masking by perturbing future tokens. Check bidirectional masking by making a masked target depend on information to its right.
3. Verify the shifted loss and the masked-token loss on a small example with padding.
4. Add an independently designed event family and freeze the split and operating point before examining results.
5. Compare longer n-grams, count-based baselines and order ablations across multiple seeds. Record failures as well as gains.

Use reviewable commits and record exactly what you implemented or validated. Do not include private equipment logs, recipes, or employer code.
