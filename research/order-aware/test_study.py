import unittest
from collections import Counter

import torch

from study import Counts, EPOCHS, EVENTS, EventMLM, OrderModel, Precedence, generate, operating_point, pad_batch


class OrderContracts(unittest.TestCase):
    def test_matched_pairs_and_split_isolation(self):
        seen = set()
        for partition in generate():
            for normal, corrupt, kind in partition:
                self.assertEqual(Counter(normal), Counter(corrupt))
                self.assertEqual([n for n, c in zip(normal, corrupt) if n != c], [4+kind, 5+kind])
                for trace in (normal, corrupt):
                    self.assertNotIn(trace, seen)
                    seen.add(trace)

    def test_same_initial_backbone_and_padding_invariance(self):
        torch.manual_seed(42)
        original = EventMLM(len(EVENTS), 32)
        torch.manual_seed(42)
        upgraded = OrderModel()
        for name, value in original.state_dict().items():
            torch.testing.assert_close(value, upgraded.state_dict()[name], rtol=0, atol=0)
        short = [3, 4, 8, 5, 6, 7, 12]
        mixed = pad_batch([short, short[:-1]+[9, 10, 12]])
        torch.testing.assert_close(upgraded.order_logits(mixed)[0],
                                   upgraded.order_logits(pad_batch([short]))[0])

    def test_rules_learn_reversed_training_grammar(self):
        normal = (3, 5, 8, 4, 12)
        swapped = (3, 4, 8, 5, 12)
        counts = Counts([normal])
        self.assertEqual(counts(normal), counts(swapped))
        self.assertGreater(Precedence([normal])(swapped), Precedence([normal])(normal))
        self.assertGreater(Precedence([swapped])(normal), Precedence([swapped])(swapped))

    def test_calibration_ties(self):
        scores = [0.0]*396 + [2.0]*4
        threshold = operating_point(scores)
        self.assertEqual(sum(s>threshold for s in scores), 4)


if __name__ == '__main__':
    unittest.main()
