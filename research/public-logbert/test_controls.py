import unittest
import torch
from bert_mlm import EventMLM

from hdfs_benchmark import bag_l1_scores, shuffled, shifted_causal_inputs


class ControlTests(unittest.TestCase):
    def test_bag_is_order_invariant_and_penalizes_extra_events(self):
        train = [{'ids': [3, 4, 3]}]
        query = [{'ids': [4, 3, 3]}, {'ids': [3, 3, 4, 4]}, {'ids': [3, 3, 2]}]
        self.assertEqual(bag_l1_scores(train, query, 5), [0., 1., 2.])

    def test_shuffle_preserves_bag_and_does_not_mutate_source(self):
        train = [{'block': 'demo', 'ids': [3, 4, 3, 5, 6]}]
        permuted = shuffled(train, 42)
        self.assertCountEqual(permuted[0]['ids'], train[0]['ids'])
        self.assertEqual(permuted, shuffled(train, 42))
        self.assertEqual(train[0]['ids'], [3, 4, 3, 5, 6])

    def test_causal_prediction_does_not_see_future_events(self):
        torch.manual_seed(14)
        model = EventMLM(10, 8)
        original = torch.tensor([[3, 4, 5, 6]])
        changed = torch.tensor([[3, 4, 7, 8]])
        a = model(shifted_causal_inputs(original), causal=True)
        b = model(shifted_causal_inputs(changed), causal=True)
        torch.testing.assert_close(a[:, :3], b[:, :3])
        self.assertFalse(torch.allclose(a[:, 3], b[:, 3]))
