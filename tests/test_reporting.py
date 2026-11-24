"""Synthetic, offline regression tests for reporting/export; NOT evidence of model performance."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd

TOOLS=Path(__file__).resolve().parents[1]/'tools'
sys.path.insert(0,str(TOOLS))
from generate_report import create_report, _read_task
from export_project import make_repo_zip, make_report_zip, make_selected_models_backup

class ReportingTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)/'MixBhasha_FakeExperiment'
        self.task='cyberbullying7'
        self.y=np.tile(np.arange(7,dtype=int),5)
        d=self.root/'models'/self.task
        d.mkdir(parents=True)
        (self.root/'data').mkdir()
        (self.root/'results').mkdir()
        pd.DataFrame({'text':[f'synthetic {i}' for i in range(len(self.y))],'label':self.y}).to_csv(
            self.root/'data'/f'{self.task}_val.csv',index=False)
        (self.root/'results'/'data_audit.json').write_text(json.dumps({self.task:{'split_sha256':'test123',
            'splits':{'train':{'input_rows':999,'kept':900},'val':{'input_rows':35,'kept':35}}}}))
        for name,acc in [('tfidf_svm',.85),('hing_roberta_mixed',.75)]:
            folder=d/name
            folder.mkdir()
            rng=np.random.default_rng(12 if name=='tfidf_svm' else 24)
            p=np.zeros((len(self.y),7))
            for i,c in enumerate(self.y):
                winner=c if rng.random()<acc else (c+1)%7
                p[i,:]=.02
                p[i,winner]=.88
            p=p/p.sum(axis=1,keepdims=True)
            np.savez_compressed(folder/'validation_probabilities.npz',labels=self.y,probs=p)
            (folder/'validated.json').write_text(json.dumps({'meta':{'split_sha256':'test123','seed':42},
                                  'metrics':{'threshold':.5}}))
            (folder/'model.joblib').write_bytes(b'FAKE')

    def test_report_generates_comparisons_and_all_model_charts(self):
        out=self.root/'reports'/'public_validation'
        summary=create_report(self.root,out,tasks=[self.task],bootstrap=30)
        self.assertIn(self.task,summary)
        self.assertEqual(len(summary[self.task]['models']),2)
        for f in ('REPORT.html','REPORT.md', 'report_summary.json',
                  'charts/cyberbullying7/model_metrics_comparison.png',
                  'charts/cyberbullying7/class_f1_heatmap.png',
                  'charts/cyberbullying7/tfidf_svm_confusion.png',
                  'charts/cyberbullying7/hing_roberta_mixed_pr_per_class.png',
                  'cyberbullying7/paired_bootstrap_top2.json'):
            self.assertTrue((out/f).exists(),f)
        board=pd.read_csv(out/self.task/'leaderboard.csv')
        self.assertAlmostEqual(board.loc[0,'macro_f1'],summary[self.task]['best_validation_macro_f1'])
        self.assertNotIn('synthetic 0',(out/'REPORT.html').read_text())

    def test_split_sha_mismatch_blocks_comparison(self):
        folder=self.root/'models'/self.task/'tfidf_svm'/'validated.json'
        s=json.loads(folder.read_text());s['meta']['split_sha256']='different';folder.write_text(json.dumps(s))
        with self.assertRaisesRegex(RuntimeError,'Split fingerprint mismatch'):
            _read_task(self.root,self.task)

    def test_public_zip_keeps_code_but_not_private_artifacts(self):
        src=Path(__file__).resolve().parents[1]
        destination=self.root/'public.zip'
        make_repo_zip(src,destination)
        import zipfile
        with zipfile.ZipFile(destination) as z:
            names=z.namelist()
        self.assertIn('MixBhasha/tools/generate_report.py',names)
        self.assertIn('MixBhasha/README.md',names)
        self.assertFalse(any('/data/' in n or '/models/' in n or '/private_analysis/' in n for n in names))

    def test_private_selected_models_archive(self):
        destination=Path(self.tmp.name)/'private_models.zip'
        make_selected_models_backup(self.root,destination,[f'{self.task}:tfidf_svm'])
        import zipfile
        with zipfile.ZipFile(destination) as z:
            names=z.namelist()
        self.assertTrue(any('tfidf_svm/model.joblib' in n for n in names))
        self.assertFalse(any('hing_roberta_mixed/model.joblib' in n for n in names))

if __name__=='__main__':
    unittest.main()
