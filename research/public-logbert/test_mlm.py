"""Behavioral tests for the bidirectional encoder, MLM loss, and held-out scoring."""
import unittest

import torch
from torch.nn import functional as F

from bert_mlm import (
    EventMLM, IGNORE, MASK, PAD, UNK, corrupt_for_training, leave_one_out_nll,
    masked_loss, pad_batch,
)
from synthetic_check import CUE, TARGET, make_groups


class MaskedModelTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(42)
        self.model = EventMLM(23, 8, width=16, heads=4, layers=2).eval()

    def test_right_context_changes_prediction(self):
        ids = torch.tensor([[4, MASK, 5], [4, MASK, 6]])
        logits = self.model(ids)[:, 1]
        self.assertGreater((logits[0] - logits[1]).abs().max().item(), 1e-5)

    def test_padding_does_not_change_real_logits(self):
        short = torch.tensor([[3, MASK, 4]])
        mixed = pad_batch([[3, MASK, 4], [7, 8, 9, 10, 11]])
        torch.testing.assert_close(self.model(short)[0], self.model(mixed)[0, :3])

    def test_training_corruption_and_ignored_positions(self):
        gen = torch.Generator().manual_seed(99)
        ids = torch.tensor([[PAD, UNK, 3, 4, 5, 6, 7, 8]]).repeat(20000, 1)
        corrupted, labels = corrupt_for_training(ids, 103, gen)
        selected = labels != IGNORE
        self.assertTrue(torch.equal(corrupted[~selected], ids[~selected]))
        self.assertTrue(torch.equal(labels[selected], ids[selected]))
        self.assertFalse(selected[:, :2].any())
        self.assertAlmostEqual(selected.sum().item() / (ids > UNK).sum().item(), 0.15, delta=0.005)
        hidden = corrupted[selected].eq(MASK).float().mean().item()
        unchanged = corrupted[selected].eq(ids[selected]).float().mean().item()
        self.assertAlmostEqual(hidden, 0.80, delta=0.015)
        # 100 ordinary types make a random replacement match the original 1% of the time.
        self.assertAlmostEqual(unchanged, 0.101, delta=0.01)
        self.assertFalse(torch.isin(corrupted[selected], torch.tensor([PAD, UNK])).any())

    def test_only_selected_tokens_contribute_to_loss(self):
        logits = torch.randn(2, 3, 12, requires_grad=True)
        labels = torch.tensor([[3, IGNORE, IGNORE], [IGNORE, 7, IGNORE]])
        selected = labels != IGNORE
        expected = F.cross_entropy(logits[selected], labels[selected])
        actual = masked_loss(logits, labels)
        torch.testing.assert_close(actual, expected)
        actual.backward()
        self.assertTrue((logits.grad[~selected] == 0).all())
        self.assertGreater(logits.grad[selected].abs().sum().item(), 0)

    def test_empty_selection_has_zero_gradient(self):
        logits = torch.randn(1, 2, 9, requires_grad=True)
        loss = masked_loss(logits, torch.full((1, 2), IGNORE))
        self.assertEqual(loss.item(), 0)
        loss.backward()
        self.assertEqual(logits.grad.abs().sum().item(), 0)

    def test_score_hides_each_target_and_skips_pads(self):
        ids = pad_batch([[5, 7, UNK], [6, 9]])
        seen = []
        hook = self.model.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone()))
        try:
            scores = leave_one_out_nll(self.model, ids, chunk_size=2)
        finally:
            hook.remove()
        copies = torch.cat(seen)
        for copy, (row, pos) in zip(copies, ids.ne(PAD).nonzero(), strict=True):
            expected = ids[row].clone()
            expected[pos] = MASK
            self.assertTrue(torch.equal(copy, expected))
            logit = self.model(expected.unsqueeze(0))[0, pos]
            torch.testing.assert_close(scores[row, pos], -logit.log_softmax(-1)[ids[row, pos]])
        self.assertTrue(torch.isnan(scores[ids == PAD]).all())
        self.assertTrue(torch.isfinite(scores[ids != PAD]).all())

    def test_groups_are_disjoint_and_only_cue_identifies_target(self):
        train, test = make_groups(42)
        def group_keys(rows):
            return {tuple(row) for row in rows[:, [0, 1, 3]].tolist()}
        self.assertTrue(group_keys(train).isdisjoint(group_keys(test)))
        for quartet in test.split(4):
            self.assertEqual(quartet[:, TARGET].unique().numel(), 4)
            self.assertEqual(quartet[:, CUE].unique().numel(), 4)
            self.assertEqual(len(group_keys(quartet)), 1)


if __name__ == "__main__":
    unittest.main()
