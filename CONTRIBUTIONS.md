# Contribution guide

The published implementation, synthetic generators, benchmark harness, and tests began as AI-assisted reference work. Do not attribute that work to someone who did not implement or verify it. PyTorch provides tensor primitives, layers, autograd, and the optimizer; the sequence architecture and experiments are not inventions of those underlying methods.

Useful follow-up contributions:

1. Reimplement and explain the Q/K/V projections, head reshaping, and attention masks; add shape and invariance tests.
2. Check causal masking by perturbing future tokens. Check bidirectional masking by making a masked target depend on information to its right.
3. Verify the shifted loss and the masked-token loss on a small example with padding.
4. Add an independently designed event family and freeze the split and operating point before examining results.
5. Compare longer n-grams, count-based baselines and order ablations across multiple seeds. Record failures as well as gains.

Use reviewable commits and record exactly what you implemented or validated. Do not include private equipment logs, recipes, or employer code.
