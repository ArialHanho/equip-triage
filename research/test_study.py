import unittest

import numpy as np

from study import ABSTAIN, classify, evaluate, load, rule


class ResearchContract(unittest.TestCase):
    def test_group_and_counts(self):
        splits = load()
        self.assertEqual([len(splits[k]) for k in ("train", "validation", "test")], [24, 12, 16])
        seen = {k: {r["group"] for r in rows} for k, rows in splits.items()}
        self.assertFalse(seen["train"] & seen["test"])
        self.assertFalse(seen["validation"] & seen["test"])

    def test_abstention(self):
        self.assertEqual(classify(np.array([[0, 0, 0, 0], [2, 2, 1, 0], [3, 2, 1, 0]]), 2),
                         [ABSTAIN, ABSTAIN, "data"])

    def test_validation_only_threshold(self):
        data = load()
        r1 = evaluate(rule, data, [1, 2, 3], False)
        data["test"] = list(reversed(data["test"]))
        r2 = evaluate(rule, data, [1, 2, 3], False)
        self.assertEqual(r1["validation_cutoff"], r2["validation_cutoff"])
        self.assertEqual(r1["test"], r2["test"])
