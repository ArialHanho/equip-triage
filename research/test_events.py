"""Meaningful leakage and loss-contract checks for the reference experiment."""
import unittest

import torch
import torch.nn.functional as F

from event_transformer import NGram, TinyTransformer, VOCAB, batch, scenarios


class EventContracts(unittest.TestCase):
    def test_future_cannot_change_prefix_logits(self):
        torch.manual_seed(7)
        model = TinyTransformer().eval()
        a = torch.tensor([[0, 1, 2, 3, 4, 6]])
        b = torch.tensor([[0, 1, 2, 8, 7, 5]])
        with torch.no_grad():
            torch.testing.assert_close(model(a)[:, :3], model(b)[:, :3], atol=1e-6, rtol=1e-6)

    def test_padding_does_not_change_masked_loss(self):
        model = TinyTransformer().eval()
        rows = scenarios()
        row = min(rows, key=lambda r: len(r['events']))
        x, y = batch([row])
        with torch.no_grad():
            expected = F.cross_entropy(model(x).reshape(-1, len(VOCAB)), y.reshape(-1))
            padded_x = F.pad(x, (0, 4))
            padded_y = F.pad(y, (0, 4), value=-100)
            actual = F.cross_entropy(model(padded_x).reshape(-1, len(VOCAB)), padded_y.reshape(-1), ignore_index=-100)
        torch.testing.assert_close(actual, expected)

    def test_scenario_splits_and_shift(self):
        rows = scenarios()
        sets = {s: {r['template'] for r in rows if r['split'] == s} for s in ('train', 'validation', 'test')}
        self.assertFalse(sets['train'] & sets['test'])
        self.assertFalse(sets['validation'] & sets['test'])
        self.assertTrue(all(r['label'] == 'normal' for r in rows if r['split'] == 'train'))
        x, y = batch([rows[0]])
        self.assertEqual(x[0, 0].item(), 0)
        self.assertTrue(torch.equal(x[0, 1:], y[0, :-1]))

    def test_memory_diagnostic_has_shared_suffix_and_disjoint_sequences(self):
        from long_context import make_rows
        rows = make_rows()
        self.assertEqual(len({tuple(r['events']) for r in rows}), len(rows))
        for row in rows:
            index = row['target_index']
            self.assertEqual(row['events'][index-3:index], ['LOAD', 'MEASURE', 'UNLOAD'])
            self.assertEqual(row['events'][index], row['events'][1])

    def test_ngram_sees_only_prefix(self):
        rows = scenarios()
        model = NGram([r for r in rows if r['split'] == 'train'])
        a = ['START', 'LOAD', 'CHECK', 'MEASURE', 'END']
        b = ['START', 'LOAD', 'CHECK', 'WAIT', 'RETEST']
        self.assertEqual(model.losses(a)[:3], model.losses(b)[:3])


if __name__ == '__main__':
    torch.set_num_threads(1)
    unittest.main()
