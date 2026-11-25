# What distinguishes one continuing check from two?

A model must route the single-cause memo correctly and hold its two-cause counterpart. On a fresh synthetic challenge a shared character classifier over each candidate’s evidence gets 35/112 English and 83/112 Korean pairs right. The scoped bilingual rule gets 70/112 and 98/112. A compact candidate-query BERT learns the training patterns but fails to recover a single challenge pair at any of three seeds. A count of the continuing states improves decisions with frozen character scores. The attention hypothesis receives no support in this experiment.

## Evaluation population

Each language has 848 challenge packets in eight held-out wording families, comprising 224 actionable cases and 624 gold holds. Each family supplies 14 primary comparisons per language. The two members differ in the current state of one check. The comparison counts only when the correct check is selected in the single-cause member and its matched two-cause member is held. A stricter triple requires correct decisions on both possible single-cause configurations and the shared two-cause packet.

Raw packets are distinct within each split, and wording families, exact memos and raw packets are disjoint across splits. Reversals, translations and aliases are dependent views. There are 1,024 distinct normalized evidence inputs among the 1,696 raw challenge rows. The alias matcher deliberately makes different spellings of registered tags equivalent. The counts describe a controlled test of two named checks, not thousands of independent operators or plants.

## Results at the validation-selected utility gate

| Method | One/two pairs EN / 112 | One/two pairs KO / 112 | Macro-F1 EN / KO | CPU p50 ms |
| --- | ---: | ---: | ---: | ---: |
| Scoped bilingual rule | 70 | 98 | 0.8028 / 0.9423 | 0.060 |
| Guarded character direct | 0 | 0 | 0.1696 / 0.1696 | 0.614 |
| Character mention heads | 0 | 8 | 0.2142 / 0.2722 | 1.013 |
| Mention states + observed binding | 0 | 14 | 0.2387 / 0.3472 | 0.970 |
| Shared character states, both lines | 56 | 84 | 0.6314 / 0.8771 | 1.385 |
| Shared character states, bound line | 35 | 83 | 0.5102 / 0.8579 | 0.889 |
| Flat scratch BERT, direct | 0 | 0 | 0.1696 / 0.1696 | 0.827 |
| Local CLS, slot-specific state heads | 0 | 0 | 0.1696 / 0.1693 | 0.911 |
| Bound uniform pooling | 0 | 0 | 0.1696 / 0.1696 | 1.081 |
| Bound candidate-query BERT | 0 | 0 | 0.1696 / 0.1696 | 1.122 |
| Gold states + exact count | 112 | 112 | 1.0000 / 1.0000 | — |

Neural F1 entries average the three fixed seeds; the cohort size in each run is 848 per language. The abbreviated BERT rows in the table have identical pair counts at all three seeds. The run files retain each seed, latency and gate. The oracle gets every decision right by construction and uses unavailable gold states. [All metrics](run/metrics.csv) · [Paired uncertainty estimates](run/contrasts.csv)

The bound character model was selected on validation mean macro-F1 (0.722, compared with 0.717 for the scoped rule). The scoped rule scores higher on the new challenge in both languages. The released default reproduces the validation decision. With only eight challenge families, a few new forms of negation, final-state ordering or vocabulary can change rankings.

## Which change mattered?

- **Candidate-level state reading.** The mention-order linear control with observed binding gets 0/112 English and 14/112 Korean pairs right. Moving to a shared character classifier on each candidate’s bound line gives gains of 31.2 percentage points in English (paired 95% interval 6.1 to 62.5) and 61.6 points in Korean (31.2 to 89.3). This comparison changes target indexing, parameter sharing and feature organization together. It does not isolate a tensor operation.
- **Evidence scope.** Giving the linear candidate classifier the other line as a separate feature channel gets 56/112 and 84/112 pairs right. Removing that channel changes success by −18.8 points in English (interval −50.0 to 0.0) and −0.9 points in Korean (−24.1 to 21.4). Hard restriction to one line does not improve over the richer character control on this challenge.
- **Decision policy.** Before applying any confidence threshold, the exact-count policy gets 37/112 and 86/112 pairs right with the bound character scores. Winner-take-all gets 0/112 and 5/112 on the **same** scores. The paired gains are 33.0 points in English (7.8 to 62.5) and 72.3 in Korean (44.6 to 100). With a gate, even winner-take-all can sometimes decline a two-cause case, so the zero-threshold comparison is the cleaner decoder ablation.
- **Learned pooling.** Switching bound uniform weights to bound query attention gives a zero-point utility pair difference in both languages. All three seeds choose a hold-all utility threshold for both models. Before thresholding, neither method recovers any challenge pair either. This floor result cannot establish equivalence between general pooling mechanisms. At seed 42, the query arm has 62,435 active parameters and the uniform arm 60,323.
- **A training diagnostic.** A separate seed-42 refit of the fixed query model recovers 168/168 unthresholded pairs per language on training and 0/56 per language on validation. On complete packets it gets 4,704/4,704 candidate states on training and 532/1,568 on validation. Training contains only 96 distinct nonempty normalized lines. The model can fit those lines; the diagnostic is consistent with poor transfer to new phrasing. It cannot identify the sole cause of failure.

Intervals use 2,000 paired bootstrap draws of whole families. Metrics are recomputed from pooled counts with the same draw for both methods, then averaged over the fixed seeds. They condition on this generator, its lexicon, the calibration split and the seeds. They do not quantify uncertainty from external authorship or field deployment. [Frozen protocol](PROTOCOL.md)

## Errors behind the aggregate score

At the bound character model’s utility cutoff of 0.40, it routes 114/848 English packets (13.4%). Among them, 70 reach the correct check, 28 reach the wrong actionable check and 16 should have been held. The false-release rate is 16/624 (2.6%); the error rate among its routed English cases is 44/114 (38.6%). In Korean it routes 161/848 (19.0%), all correct in this sample. It holds 126/224 actionable English cases and 63/224 actionable Korean cases. The separately chosen safety cutoff is also 0.40 for this method. A 10% validation constraint on false release does not bound the risk of a wrong actionable check.

Across structurally complete packets alone the selected model releases 16/560 English and 0/560 Korean gold holds. It correctly holds all 64 structurally incomplete cases per language; these cases cover missing lines, incomplete records, unlisted references and duplicated references. It also holds all 112/112 per language where **both** outcomes are unavailable despite complete, uniquely matched records. In contrast, when just one outcome is unavailable, its English correct counts are 99/112 and 109/112 depending on the slot. A successful integrity guard cannot replace semantic state reading.

The scoped rule routes 140/848 English and 196/848 Korean packets with zero observed false releases or wrong checks. Its remaining 84 and 28 actionable cases are held. The bound-query encoder’s utility gate routes no packets, so an error rate among accepted packets is undefined. At seed 42 it labels every candidate in the 784 complete Korean challenge packets as unknown before the gate.

| Matched intervention at the utility gate | Bound character EN / KO | Scoped rule EN / KO | Comparisons per language |
| --- | ---: | ---: | ---: |
| One continuing vs two | 35 / 83 | 70 / 98 | 112 |
| All three outcomes correct | 28 / 72 | 70 / 98 | 112 |
| Latest reading clears | 35 / 83 | 70 / 98 | 112 |
| Latest reading unavailable | 35 / 83 | 70 / 98 | 112 |
| Reversing slots changes referent | 10 / 22 | 20 / 28 | 32 |
| Reversing slots preserves named check | 8 / 21 | 20 / 28 | 32 |
| Reversing the two memo lines | 20 / 46 | 40 / 56 | 64 |
| A previously unseen registered tag | 10 / 23 | 20 / 28 | 32 |

The bound character model succeeds on all 14 English primary pairs in the index-pulse and seal-change families and none in five other English families. Its Korean result also falls to 0/14 on the margin-review family. These differences explain the wide family intervals. The fresh challenge uses a visible registry and explicit line breaks. Historical scores from other memo protocols are not valid before/after comparisons.

## Scope of the contribution

This is an implementation and a falsifiable comparison of candidate states, evidence scope and hard routing logic. Target-conditioned attention, learned query pooling, concept prediction and selective prediction all have established precedents. The results favor useful structure and scoped lexical evidence under this generator. They provide no evidence of a general attention improvement, independently authored language coverage, or factory performance. No production system, customer data, HDFS experiment or external checkpoint participates in the new study. [Original sources and architecture](README.md)
