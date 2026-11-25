from pathlib import Path
import sys,tempfile,unittest,json
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np,pandas as pd
import mixbhasha_core as m

class V5RegressionTests(unittest.TestCase):
    @staticmethod
    def mc():
        orig=['Not-Cyberbullying','Age','Gender','Religion','Mockery','Abusive','Offensive']
        return pd.DataFrame({'tweets':[f'this is record {i} example' for i in range(350)],
                             'cyberbullying_type':orig*50})

    def test_mc_seven_categories_preserved(self):
        z=m._mc_all(self.mc(),'cyberbullying7')
        self.assertEqual(z.label.value_counts().sort_index().to_dict(),{i:50 for i in range(7)})
        self.assertEqual(len(z),350)
        binary=m._mc_all(self.mc(),'cyberbullying_binary')
        self.assertEqual(binary.label.value_counts().to_dict(),{1:300,0:50})

    def test_raw_sentiment_schema(self):
        with tempfile.TemporaryDirectory() as td:
            f=Path(td)/'raw.tsv'
            f.write_text('uid\ttext\tlabel\n1\tKal class hai but movie achhi thi\t2\n2\tHaan bilkul\t1\n3\tvery disappointing day\t0\n',encoding='utf8')
            rows=m._sent_raw_tsv(f)
            self.assertEqual(rows.label.tolist(),[2,1,0])
            self.assertIn('Kal class',rows.text.iloc[0])

    def test_numeric_label_mapping_refuses_guess(self):
        d=self.mc();d['cyberbullying_type']=(np.arange(len(d))%7).astype(str)
        with self.assertRaises(ValueError):m._mc_all(d,'cyberbullying7')
        mp={str(i):v for i,v in enumerate(m.LABELS['cyberbullying7'])}
        out=m._mc_all(d,'cyberbullying7',label_map=mp)
        self.assertEqual(len(out),350)

    def test_holdout_label_does_not_filter_training(self):
        pool=m._mc_all(self.mc().iloc[:280],'cyberbullying7')
        shared=pd.DataFrame({'text':['this is record 7 example']+[f'test record x{i} new' for i in range(20)],
                             'label':[0]+[i%7 for i in range(20)]})
        frames1,log1=m.strict_split_source(pool,shared,'cyberbullying7')
        swapped=shared.copy();swapped.loc[0,'label']=5
        frames2,log2=m.strict_split_source(pool,swapped,'cyberbullying7')
        pd.testing.assert_frame_equal(frames1['train'],frames2['train'])
        pd.testing.assert_frame_equal(frames1['val'],frames2['val'])
        self.assertFalse(log1['test_policy']['source_holdout_labels_used_to_filter_training'])
        self.assertNotIn('this is record 7 example',frames1['train'].text.tolist())
        self.assertNotIn('this is record 7 example',frames1['val'].text.tolist())

    def test_train_conflicts_logged(self):
        z=m._mc_all(self.mc(),'cyberbullying_binary')
        dup=pd.concat([z,pd.DataFrame({'text':[z.text.iloc[0]],'label':[1]})],ignore_index=True)
        clean,audit=m._dedupe_train_only(dup,'cyberbullying_binary')
        self.assertGreaterEqual(audit['conflicting_rows_removed'],2)
        self.assertNotIn(z.text.iloc[0],clean.text.tolist())

    def test_load_prepared_can_exclude_holdout_splits(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);(root/'data').mkdir()
            pd.DataFrame({'text':['training example'],'label':[0]}).to_csv(
                root/'data'/'cyberbullying7_train.csv',index=False)
            for split in ('val','test'):
                (root/'data'/f'cyberbullying7_{split}.csv').write_text(
                    'text,label\nholdout example,not-a-label\n',encoding='utf8')
            loaded=m.load_prepared(root,['cyberbullying7'],splits=('train',))
            self.assertEqual(list(loaded['cyberbullying7']),['train'])
            self.assertEqual(loaded['cyberbullying7']['train'].label.tolist(),[0])

    def test_prepare_refuses_to_replace_data_when_model_fingerprint_differs(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td)
            frames={
                'train':self.mc(),
                'val':self.mc().iloc[:14].reset_index(drop=True),
                'test':self.mc().iloc[14:28].reset_index(drop=True)}
            model_dir=root/'models'/'cyberbullying7'/'tfidf_svm'
            model_dir.mkdir(parents=True)
            (model_dir/'meta.json').write_text(json.dumps({
                'task':'cyberbullying7','split_sha256':'a'*64}),encoding='utf8')
            with patch.object(m,'load_task',return_value=(
                    frames,'synthetic regression fixture',{})):
                with self.assertRaisesRegex(RuntimeError,'model metadata'):
                    m.prepare(root,tasks=['cyberbullying7'])
            self.assertFalse((root/'data'/'cyberbullying7_train.csv').exists())

    def test_ensemble_validation_only(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);m.output_paths(root)
            def df(n,offset):
                return pd.DataFrame({'text':[f"{'awful boring sad' if i%2==0 else 'lovely happy amazing'} example{offset+i} nice" for i in range(n)],
                                     'label':[i%2 for i in range(n)]})
            frame={'train':df(80,0),'val':df(30,80),'test':df(30,110)}
            a=m.fit_one(root,frame,'cyberbullying_binary','tfidf_word',epochs_lstm=1,epochs_hf=1)
            b=m.fit_one(root,frame,'cyberbullying_binary','tfidf_svm',epochs_lstm=1,epochs_hf=1)
            self.assertIn('macro_f1',b['metrics'])
            self.assertEqual(m.fit_one(root,frame,'cyberbullying_binary','tfidf_word',epochs_lstm=1,epochs_hf=1),a)
            result=m.paired_bootstrap_validation(root,'cyberbullying_binary','tfidf_word','tfidf_svm',repeats=10)
            self.assertEqual(result['samples'],30)
            proposal=m.propose_validation_ensemble(root,'cyberbullying_binary',min_gain=-.1)
            self.assertEqual(proposal['status'],'accepted')
            lock=m.select_and_lock(root,{'cyberbullying_binary':frame},['cyberbullying_binary'])
            self.assertIn(lock['cyberbullying_binary']['model'],['tfidf_word','tfidf_svm','ensemble2'])
            result=m.test_locked(root,frame,'cyberbullying_binary')
            self.assertEqual(result['metrics']['n'],30)
            output=m.predict_text(root,'cyberbullying_binary','very awful amazing')
            self.assertIn('probabilities',output)

if __name__=='__main__':unittest.main(verbosity=2)
