import unittest

import torch

from scale_pilot import Model, audit, average_precision, batch, evaluate, generate, numeric_score, state_score


class ScaleContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.rows = list(generate(per_family=200))

    def test_split_payload_audit_and_prevalence(self):
        result = audit(self.rows)
        self.assertEqual([result[s]['sessions'] for s in ('train','validation','test')], [4800,600,600])
        self.assertEqual(result['validation']['labels']['normal'],570)
        self.assertEqual(result['test']['labels']['normal'],570)
        self.assertEqual(result['duplicate_audit']['token_only_signatures_crossing_splits'],0)
        self.assertEqual(result['duplicate_audit']['full_numeric_payload_extra_duplicates'],0)

    def test_state_machine_knows_generator_grammar(self):
        for row in self.rows:
            self.assertEqual(state_score(row)>0,row['label']!='normal',row['session'])
            if row['label']=='normal':
                self.assertEqual(numeric_score(row),0)

    def test_average_precision_ties(self):
        self.assertEqual(average_precision([True,False],[1.0,1.0]),0.5)
        self.assertEqual(average_precision([True,False],[2.0,1.0]),1.0)
        self.assertEqual(average_precision([True,False],[1.0,2.0]),0.5)

    def test_threshold_independent_of_test_scores(self):
        validation=[r for r in self.rows if r['split']=='validation']
        test=[r for r in self.rows if r['split']=='test']
        scores=[state_score(r) for r in validation]
        a=evaluate(validation,test,scores,[state_score(r) for r in test])
        b=evaluate(validation,test,scores,[1000.0]*len(test))
        self.assertEqual(a['threshold'],b['threshold'])
        self.assertLessEqual(a['validation']['false_positive_rate'],0.01)

    def test_causal_mask_and_shift(self):
        model=Model().eval()
        a=torch.tensor([[0,1,2,3,4,5]])
        b=torch.tensor([[0,1,2,7,8,9]])
        with torch.no_grad():
            torch.testing.assert_close(model(a)[:,:3],model(b)[:,:3],atol=1e-6,rtol=1e-6)
        x,y=batch([self.rows[0]])
        self.assertTrue(torch.equal(x[:,1:],y[:,:-1]))


if __name__=='__main__':
    unittest.main()
