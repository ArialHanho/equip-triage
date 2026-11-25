"""Synthetic archive fixtures only: never copies public log records into tests."""
import csv
import io
from pathlib import Path
import tempfile
import unittest
import zipfile

from hdfs_benchmark import LABEL_MEMBER, TRACE_MEMBER, evaluate, obtain_archive, partition, prefix_signature, prepare, pr_ap


class PublicBenchmarkContracts(unittest.TestCase):
    def test_grouping_and_train_only_vocabulary(self):
        traces, labels = io.StringIO(), io.StringIO()
        trace_writer=csv.writer(traces)
        label_writer=csv.writer(labels)
        trace_writer.writerow(['BlockId','Features'])
        label_writer.writerow(['BlockId','Label'])
        for group in range(128):
            events=['E1']+['E3' if group & (1<<bit) else 'E4' for bit in range(7)]+['E2']
            for copy in range(2):
                name=f'SYNTHETIC_BLOCK_{group}_{copy}'
                trace_writer.writerow([name,'['+','.join(events)+']'])
                label_writer.writerow([name,'Anomaly' if group%11==0 else 'Normal'])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'synthetic-fixture.zip'
            with zipfile.ZipFile(path,'w') as archive:
                archive.writestr(TRACE_MEMBER,traces.getvalue())
                archive.writestr(LABEL_MEMBER,labels.getvalue())
            rows,vocabulary,audit=prepare(path,42,64,{'train':200,'validation':100,'test':100})
        self.assertEqual(audit['source']['blocks'],256)
        self.assertTrue(all(not row['anomaly'] for row in rows['train']))
        self.assertEqual(set(vocabulary),{'E1','E2','E3','E4'})
        self.assertTrue(all(rows[split] for split in rows))
        for a,b in [('train','validation'),('train','test'),('validation','test')]:
            self.assertFalse({tuple(r['ids']) for r in rows[a]} & {tuple(r['ids']) for r in rows[b]})
        for split in rows:
            self.assertTrue(all(partition(prefix_signature(r['events']),42)==split for r in rows[split]))

    def test_corrupt_archive_is_rejected_without_download(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'bad.zip'
            path.write_bytes(b'not the official dataset')
            with self.assertRaisesRegex(ValueError,'MD5 mismatch'):
                obtain_archive(path)

    def test_ap_handles_score_ties(self):
        self.assertEqual(pr_ap([True,False],[1,1]),0.5)
        self.assertEqual(pr_ap([True,False],[2,1]),1.0)

    def test_threshold_never_uses_test(self):
        validation=[{'anomaly':False}]*100+[{'anomaly':True}]
        test=[{'anomaly':False},{'anomaly':True}]
        scores=list(range(100))+[200]
        a=evaluate(validation,test,scores,[1,2])
        b=evaluate(validation,test,scores,[10000,10001])
        self.assertEqual(a['threshold'],b['threshold'])
        self.assertLessEqual(a['validation']['fpr'],0.01)


if __name__=='__main__':
    unittest.main()
