"""MixBhasha v5: benchmark-aware code-mixed classification experiments."""
from __future__ import annotations
import hashlib, html, json, os, random, re, unicodedata, warnings, gc, inspect
from collections import Counter
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (accuracy_score, f1_score, precision_score, recall_score,
    balanced_accuracy_score, average_precision_score, confusion_matrix,
    classification_report, brier_score_loss)

SEED=42
LABELS={'sentiment':['negative','neutral','positive'],
        'sentiment_raw':['negative','neutral','positive'],
        'cyberbullying7':['not_cyberbullying','age','gender','religion','mockery','abusive','offensive'],
        'cyberbullying_binary':['not_cyberbullying','cyberbullying_category'],
        'offense_hasoc':['not_hate_offensive','hate_or_offensive'],
        'hate':['non_hate','hate']}
SENT_URLS={
 'train':'https://raw.githubusercontent.com/singhnivedita/SemEval2020-Task9/master/Fully%20Processed%20Datasets/FinalTrainingOnly.tsv',
 'test':'https://raw.githubusercontent.com/singhnivedita/SemEval2020-Task9/master/Fully%20Processed%20Datasets/ValidationOnly.tsv'}
RAW_SENT_URLS={
 'train':'https://raw.githubusercontent.com/keshav22bansal/BAKSA_IITK/master/data/hinglish/train.txt',
 'test':'https://raw.githubusercontent.com/keshav22bansal/BAKSA_IITK/master/data/hinglish/test.txt'}
MC_REPO='sahinurlaskar/MC-Hinglish1.0'
MC_FILES={'train':'Train_data_MC-Hinglish1.0.csv','test':'Test_data_MC-Hinglish1.0.csv'}
PRISM_REPO='pankajbiswas6/prism-hinglish-hate-speech'
TRANSFORMERS={
 'distilbert':'distilbert/distilbert-base-uncased',
 'muril':'google/muril-base-cased',
 'hingbert':'l3cube-pune/hing-bert',
 'xlmr':'FacebookAI/xlm-roberta-base',
 'hing_roberta_mixed':'l3cube-pune/hing-roberta-mixed',
 'twitter_xlmr':'cardiffnlp/twitter-xlm-roberta-base',
 'twitter_xlmr_sentiment':'cardiffnlp/twitter-xlm-roberta-base-sentiment'}
MODEL_NAMES=['tfidf_word','tfidf_char','tfidf_word_char','tfidf_normalized','tfidf_svm',
             'bilstm_last','bilstm_meanmax','bilstm_attention',*TRANSFORMERS.keys()]
URL_RX=re.compile(r'(?:https?://|www\.)\S+',re.I)
MENTION_RX=re.compile(r'(?<!\w)@[\w]+',re.UNICODE)
SPACE_RX=re.compile(r'\s+')
TOKEN_RX=re.compile(r'(?u)\w+|[^\w\s]')
ALIASES={'bohot':'bahut','bohut':'bahut','bhot':'bahut','bht':'bahut',
         'achha':'acha','accha':'acha','achaa':'acha','nhi':'nahi','nahin':'nahi',
         'kyaaa':'kya','plzz':'please'}

def set_seed(seed=SEED):
    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
    if torch.cuda.is_available():torch.cuda.manual_seed_all(seed)

def clean(text):
    text=unicodedata.normalize('NFC',html.unescape(str(text)))
    text=URL_RX.sub(' HTTPURL ',text)
    text=MENTION_RX.sub(' @USER ',text)
    return SPACE_RX.sub(' ',text).strip()

def normalize_roman(text):
    return re.sub(r'(?u)\b\w+\b',lambda m:ALIASES.get(m.group(0).lower(),m.group(0)),str(text)).lower()

def output_paths(root):
    root=Path(root)
    for name in ['data','models','results','private_analysis']:(root/name).mkdir(parents=True,exist_ok=True)
    return root

def download_once(url,dest):
    from urllib.request import urlopen,Request
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if not dest.exists() or dest.stat().st_size<100:
        try:
            with urlopen(Request(url,headers={'User-Agent':'MixBhasha-v3-academic'}),timeout=120) as resp:
                b=resp.read()
            if len(b)<100:raise ValueError('file unexpectedly short')
            tmp=dest.with_name(dest.name+'.partial')
            tmp.write_bytes(b)
            tmp.replace(dest)
        except Exception as ex:raise RuntimeError(f'Cannot fetch {url}. Enable internet or supply a local copy.') from ex
    return dest

def _sent_tsv(path):
    df=pd.read_csv(path,sep='\t',header=None,names=['uid','text','label'],
                   dtype=str,on_bad_lines='warn')
    df=df[df.label.isin(['0','1','2'])].copy()
    df['label']=df.label.astype(int)
    return df[['text','label']]

def _sent_raw_tsv(path):
    """Team-released SentiMix raw Hinglish text with header: uid, text, label.

    This is a public team copy, NOT a claim of official organizer gold test release.
    """
    df=pd.read_csv(path,sep='\t',dtype=str,on_bad_lines='error')
    cols={str(x).strip().lower():x for x in df.columns}
    if not {'text','label'}.issubset(cols):
        raise ValueError(f'Raw Hinglish mirror changed schema; required text,label: {list(df.columns)}')
    out=df[[cols['text'],cols['label']]].copy()
    out.columns=['text','label']
    bad=set(out.label.dropna().astype(str)) - {'0','1','2'}
    if bad:raise ValueError(f'Unexpected raw sentiment labels {sorted(bad)}; verify label codebook')
    out=out.dropna(subset=['text','label'])
    out['label']=out.label.astype(int)
    return out


def _col(df,options):
    canon={re.sub(r'[^a-z0-9]','',str(c).lower()):c for c in df.columns}
    for item in options:
        v=canon.get(re.sub(r'[^a-z0-9]','',item.lower()))
        if v is not None:return v
    raise ValueError(f'Cannot identify one of {options}. CSV columns: {list(df.columns)}. '
                     'Set CSV column headers to text,label manually; no guessed labels are allowed.')

def _canonical_label(value):
    """Standardize explicit strings, never infer what an integer class ID means."""
    return re.sub(r'[^a-z0-9]+','',str(value).casefold())

MC_KNOWN_CATEGORIES={
    'notcyberbullying','notbullying','noncyberbullying',
    'offensive','abusive','age','gender','religion','mockery'
}

def _offense_mc(df, label_map=None, text_column=None, label_column=None):
    """Restricted MC-Hinglish subset; numeric IDs require a dataset-documented map.

    `label_map` should map source IDs/names to actual original category strings,
    NOT directly to 0/1. E.g., {'0':'Not-Cyberbullying','4':'Offensive'} only
    when verified from the particular source release. Unknown categories excluded.
    """
    tcol=text_column if text_column is not None else _col(df, [
        'text','tweet','tweets','tweet text','comment','comments','sentence',
        'post','post text','content','message','hinglish text'])
    if tcol not in df.columns:
        raise ValueError(f'MC text_column {tcol!r} not in CSV columns {list(df.columns)}')
    opts=['label','labels','class','class id','class_id','category','type','tag','cyberbullying','target']
    candidates=[label_column] if label_column else [
        df.columns[i] for i,c in enumerate(df.columns)
        if _canonical_label(c) in {_canonical_label(x) for x in opts}]
    if not candidates:raise ValueError(f'No MC label column found. Columns={list(df.columns)}')
    positive={'offensive','abusive'}
    negative={'notcyberbullying','notbullying','noncyberbullying'}
    normalized_map={_canonical_label(k):_canonical_label(v) for k,v in (label_map or {}).items()}
    invalid=set(normalized_map.values())-MC_KNOWN_CATEGORIES
    if invalid:raise ValueError(f'Unknown mapped MC categories {sorted(invalid)}; use documented category names')
    scored=[]
    for c in candidates:
        if c not in df.columns:raise ValueError(f'MC label_column {c!r} absent. Columns={list(df.columns)}')
        raw=df[c].fillna('').map(_canonical_label)
        numeric=(raw.str.fullmatch(r'\d+').fillna(False)).any()
        if numeric and not normalized_map:
            raise ValueError(f'MC column {c!r} contains numeric class IDs. '
                             'Provide MC_LABEL_MAP based on the dataset codebook; do not guess IDs.')
        if normalized_map:
            unknown_ids=sorted(set(raw[raw.str.fullmatch(r'\d+').fillna(False)])-set(normalized_map))
            if unknown_ids:raise ValueError(f'Unmapped numeric MC IDs in {c!r}: {unknown_ids}. '
                                               'Provide a complete verified MC_LABEL_MAP.')
        mapped=raw.map(lambda x:normalized_map.get(x,x))
        n=int(mapped.isin(positive|negative).sum())
        scored.append((n,c,mapped))
    scored.sort(key=lambda z:z[0],reverse=True)
    hits,lcol,lbl=scored[0]
    if hits==0:raise ValueError(f'No recognized MC categories. Values={df[lcol].value_counts(dropna=False).head(12).to_dict()}')
    if len(scored)>1 and scored[1][0]==hits and label_column is None:
        raise ValueError(f'Ambiguous label columns {lcol!r}, {scored[1][1]!r}; set MC_LABEL_COLUMN explicitly')
    chosen=lbl.isin(positive|negative)
    out=pd.DataFrame({'text':df.loc[chosen,tcol].values,
        'label':lbl.loc[chosen].isin(positive).astype(int).values})
    class_counts=lbl.value_counts(dropna=False).to_dict()
    out.attrs['source_mapping']={
        'text_column':str(tcol),'label_column':str(lcol),
        'raw_rows':int(len(df)),'included_rows':int(len(out)),
        'excluded_rows':int((~chosen).sum()),
        'source_categories':{str(k):int(v) for k,v in class_counts.items()},
        'positive_source_categories':['Offensive','Abusive'],
        'negative_source_categories':['Not-Cyberbullying'],
        'scope':'restricted binary target, not all offensive/hateful/cyberbullying content'}
    return out

def _offense_hasoc(df):
    tcol=_col(df,['text','tweet','tweets','comment','post','content'])
    lcol=_col(df,['label','class','category','task1','task_1','tag'])
    lbl=df[lcol].fillna('').astype(str).str.upper().str.strip()
    allowed={'HOF':1,'NOT':0}
    bad=sorted(set(lbl)-set(allowed))
    if bad:raise ValueError(f'Unexpected HASOC labels {bad}; expected exact HOF,NOT. '
                             'Do not silently convert HATE to OFFENSIVE.')
    return pd.DataFrame({'text':df[tcol], 'label':lbl.map(allowed).astype(int)})

def _normalize(df,task):
    df=df.copy().dropna(subset=['text','label'])
    df['text']=df.text.astype(str).map(clean)
    df=df[df.text.str.len().between(2,4000)].copy()
    df['label']=pd.to_numeric(df.label,errors='raise').astype(int)
    if not set(df.label.unique()).issubset(set(range(len(LABELS[task])))):
        raise ValueError(f'Invalid label in {task}: {df.label.unique().tolist()}')
    return df.reset_index(drop=True)

def _dedupe(frames,task,require_all_splits=True):
    """Quarantine contradictions globally; protect holdout before validation/train.

    All records with conflicting labels for the same normalized text are removed,
    not assigned to the arbitrary first-seen class. No raw user text is logged.
    Holdout precedence: test > val > train for identically labelled duplicates.
    """
    order=[split for split in ['test','val','train'] if split in frames]
    normalized={}; report={}
    for split in order:
        original=frames[split]
        df=_normalize(original,task)
        df['_key']=df.text.str.casefold().str.strip()
        normalized[split]=df
        report[split]={
            'input_rows':int(len(original)),
            'post_normalization_rows':int(len(df)),
            'invalid_or_empty_rows_removed':int(len(original)-len(df)),
            'within_split_conflicting_text_keys':int(df.groupby('_key').label.nunique().gt(1).sum())}
    all_pairs=pd.concat([normalized[k][['_key','label']] for k in order],ignore_index=True)
    contradictory=set(all_pairs.groupby('_key').label.nunique().loc[lambda z:z>1].index)
    # Hash-only examples allow human audit without exporting raw text.
    digest=[hashlib.sha256(k.encode('utf-8')).hexdigest()[:16] for k in sorted(contradictory)]
    global_report={'contradictory_text_keys':len(contradictory),
                   'contradictory_rows_removed':int(all_pairs['_key'].isin(contradictory).sum()),
                   'conflicting_key_hash_samples':digest[:20],
                   'policy':'quarantine every occurrence of a text with conflicting labels'}
    seen=set();out={}
    for split in order:
        df=normalized[split]
        conflicts=df['_key'].isin(contradictory)
        good=df.loc[~conflicts].copy()
        same=good['_key'].duplicated(keep='first')
        good=good.loc[~same].copy()
        across=good['_key'].isin(seen)
        keep=good.loc[~across].copy()
        seen.update(keep['_key'].tolist())
        kept=keep.drop(columns='_key').reset_index(drop=True)
        r=report[split]
        r.update({'conflicting_rows_removed':int(conflicts.sum()),
                  'same_label_duplicates_removed':int(same.sum()),
                  'cross_split_overlap_rows_removed':int(across.sum()),
                  'kept':int(len(kept)),
                  'removed_total':int(r['input_rows']-len(kept)),
                  'class_counts':{str(k):int(v) for k,v in kept.label.value_counts().sort_index().items()}})
        if require_all_splits and (len(kept)<15 or kept.label.nunique()!=len(LABELS[task])):
            raise ValueError(f'{task}/{split}: too small or missing classes after audited dedup: {r}')
        if len(kept)==0:raise ValueError(f'{task}/{split}: no usable data left after dedup')
        out[split]=kept
    report['_global']=global_report
    return out,report

def _mc_all(df, task, label_map=None, text_column=None, label_column=None):
    """Use *all seven* original categories. No undocumented numeric label guesses."""
    if task not in ('cyberbullying7','cyberbullying_binary'):
        raise ValueError('MC full-label loader requires cyberbullying task')
    tcol=text_column or _col(df,['tweets','tweet','text','comment','post','content','message'])
    lcol=label_column or _col(df,['cyberbullying_type','category','label','class','type'])
    raw=df[lcol].fillna('').map(_canonical_label)
    normalized_map={_canonical_label(k):_canonical_label(v) for k,v in (label_map or {}).items()}
    if raw.str.fullmatch(r'\d+').fillna(False).any() and not normalized_map:
        raise ValueError('Numeric MC labels need an explicit *source-category* mapping; refusing to guess')
    unknown_ids=set(raw[raw.str.fullmatch(r'\d+').fillna(False)])-set(normalized_map)
    if unknown_ids:raise ValueError(f'Unmapped MC numeric IDs: {sorted(unknown_ids)}')
    cat=raw.map(lambda x:normalized_map.get(x,x))
    canon=list(LABELS['cyberbullying7'])
    canon_lookup={_canonical_label(name):i for i,name in enumerate(canon)}
    unexpected=sorted(set(cat)-set(canon_lookup))
    if unexpected:raise ValueError(f'Unexpected MC category: {unexpected[:20]}. Update only with verified source taxonomy.')
    labels=cat.map(canon_lookup).astype(int)
    if task=='cyberbullying_binary':labels=(labels!=0).astype(int)
    out=pd.DataFrame({'text':df[tcol].values,'label':labels.values})
    out.attrs['source_mapping']={'raw_rows':int(len(df)), 'included_rows':int(len(out)),
        'categories':{str(k):int(v) for k,v in cat.value_counts().items()},
        'task_label_scope':('seven mutually exclusive annotated MC cyberbullying categories' if task=='cyberbullying7' else
                          'derived binary: any of the six harmful categories vs not-cyberbullying'),
        'note':'Harmful-category is NOT equivalent to all offensive language or hate speech'}
    return out


def _key_series(df):
    return df.text.astype(str).str.casefold().str.strip()


def _dedupe_train_only(pool,task,excluded_texts=()):
    """Train-label-only cleaning. Holdout labels are NEVER consulted to curate train."""
    x=_normalize(pool,task)
    x['_key']=_key_series(x)
    contradictory=set(x.groupby('_key').label.nunique().loc[lambda n:n>1].index)
    mask_conflict=x._key.isin(contradictory)
    x1=x.loc[~mask_conflict].copy()
    mask_repeat=x1._key.duplicated()
    x2=x1.loc[~mask_repeat].copy()
    forbid=set(excluded_texts)
    mask_overlap=x2._key.isin(forbid)
    clean=x2.loc[~mask_overlap].drop(columns=['_key']).reset_index(drop=True)
    audit={'input_rows':len(pool),'invalid_rows_removed':int(len(pool)-len(x)),
           'conflicting_keys_quarantined':len(contradictory),'conflicting_rows_removed':int(mask_conflict.sum()),
           'duplicate_same_label_removed':int(mask_repeat.sum()),
           'overlap_with_reserved_text_removed':int(mask_overlap.sum()),
           'remaining_rows':int(len(clean)),
           'conflicting_key_sha256_prefixes':[hashlib.sha256(k.encode('utf8')).hexdigest()[:16] for k in sorted(contradictory)[:20]]}
    return clean,audit


def strict_split_source(pool, holdout, task, seed=SEED, validation_fraction=.12):
    """Test labels read for reporting only, never used to drop/retain training data."""
    reserved=_normalize(holdout,task)
    # IMPORTANT: no operations involving reserved['label'] decide train membership.
    reserved_keys=set(_key_series(reserved))
    cleaned,audit=_dedupe_train_only(pool,task,reserved_keys)
    count=cleaned.label.value_counts().reindex(range(len(LABELS[task])),fill_value=0)
    if (count<3).any():raise ValueError(f'{task} has insufficient examples after cleaning: {count.to_dict()}')
    tr,va=train_test_split(cleaned,test_size=validation_fraction,stratify=cleaned.label,random_state=seed)
    out={'train':tr.reset_index(drop=True),'val':va.reset_index(drop=True),'test':reserved.reset_index(drop=True)}
    tests=_key_series(reserved)
    audit['test_policy']={'rows_before':len(holdout),'rows_after_text_filter':len(reserved),
                          'same_text_duplicate_rows':int(tests.duplicated().sum()),
                          'source_holdout_labels_used_to_filter_training':False,
                          'holdout_label_cleanup':'No label-dependent row removals',
                          'previously_exposed_v4_test_for_sentiment':task=='sentiment',
                          'sentiMix_related_potential_overlap':task in ('sentiment','sentiment_raw')}
    groups=[set(_key_series(out[k])) for k in ('train','val','test')]
    assert not (groups[0]&groups[1] or groups[0]&groups[2] or groups[1]&groups[2])
    return out,audit


def strict_presplit_source(original,task):
    """Use existing PRISM splits, clean train labels only; remove text-key overlaps without test labels."""
    reserved=_normalize(original['test'],task)
    val=_normalize(original['val'],task)
    testkeys=set(_key_series(reserved))
    val=val.loc[~_key_series(val).isin(testkeys)].reset_index(drop=True)
    forbidden=testkeys|set(_key_series(val))
    tr,audit=_dedupe_train_only(original['train'],task,forbidden)
    out={'train':tr,'val':val,'test':reserved}
    return out,{'train_cleaning':audit, 'policy':'preserve source partitions; no test-label curation'}


def load_task(task,root,offense_source='mc',hasoc_train='',hasoc_test='',
              mc_label_map=None,mc_text_column=None,mc_label_column=None):
    root=output_paths(root)
    if task=='sentiment':
        pool=_sent_tsv(download_once(SENT_URLS['train'],root/'data'/'sentiment_train_source.tsv'))
        test=_sent_tsv(download_once(SENT_URLS['test'],root/'data'/'sentiment_dev_source.tsv'))
        source='Third-party processed SentiMix: published dev previously exposed by v4, not official test'
    elif task=='sentiment_raw':
        pool=_sent_raw_tsv(download_once(RAW_SENT_URLS['train'],root/'data'/'raw_sentimix_train_team_mirror.tsv'))
        test=_sent_raw_tsv(download_once(RAW_SENT_URLS['test'],root/'data'/'raw_sentimix_test_team_mirror.tsv'))
        source='BAKSA public team mirror of SentiMix Hinglish raw text; NOT organizer-verified official leaderboard test'
    elif task in ('cyberbullying7','cyberbullying_binary'):
        from huggingface_hub import hf_hub_download
        paths={k:hf_hub_download(repo_id=MC_REPO,filename=v,repo_type='dataset',
                local_dir=str(root/'data'/'mc_source')) for k,v in MC_FILES.items()}
        mapped={k:_mc_all(pd.read_csv(path),task,mc_label_map,mc_text_column,mc_label_column)
                for k,path in paths.items()}
        pool,test=mapped['train'],mapped['test']
        source='MC-Hinglish1.0 all 7 categories; derived binary supported but separately named'
    elif task=='offense_hasoc':
        if not (hasoc_train and hasoc_test and Path(hasoc_train).is_file() and Path(hasoc_test).is_file()):
            raise FileNotFoundError('Authorized HASOC 2021 code-mix CSVs must be supplied with HOF/NOT labels')
        pool=_offense_hasoc(pd.read_csv(hasoc_train))
        test=_offense_hasoc(pd.read_csv(hasoc_test))
        source='HASOC 2021 authorized Subtask 2; context-sensitive HOF/NOT'
    elif task=='hate':
        from datasets import load_dataset
        ds=load_dataset(PRISM_REPO)
        if not {'train','val','test'}.issubset(set(ds)):
            raise ValueError('PRISM requires original train/val/test splits')
        original={k:ds[k].to_pandas()[['text','label','lang']] for k in ('train','val','test')}
        # this small optional task retains only text,label; language analysis can run directly on original.
        frames,audit=strict_presplit_source(original,task)
        return frames,'PRISM hate != generic offensive-language moderation',audit
    else:raise ValueError(f'Unsupported task {task}')
    frames,audit=strict_split_source(pool,test,task)
    if task in ('cyberbullying7','cyberbullying_binary'):
        audit['source_categories']=mapped['train'].attrs['source_mapping']
    return frames,source,audit

def prepare(root,tasks=('sentiment','cyberbullying7'),offense_source='mc',hasoc_train='',hasoc_test='',
            mc_label_map=None,mc_text_column=None,mc_label_column=None):
    root=output_paths(root);all_data={};report={}
    for task in tasks:
        frames,source,audit=load_task(task,root,offense_source,hasoc_train,hasoc_test,
                                     mc_label_map,mc_text_column,mc_label_column)
        # Do not overwrite frozen splits if a previous experiment used a different source.
        existing={split:root/'data'/f'{task}_{split}.csv' for split in ['train','val','test']}
        if all(p.exists() for p in existing.values()):
            prev=load_prepared(root,[task])[task]
            if fingerprint(prev)!=fingerprint(frames):
                if any((root/'models'/task).rglob('validated.json')):
                    raise RuntimeError(f'{task}: prepared dataset differs from a trained experiment. '
                        'Start with a NEW ROOT; refusing to mix old models and new splits.')
        for split,d in frames.items():d.to_csv(existing[split],index=False)
        all_data[task]=frames
        report[task]={'source':source,'split_sha256':fingerprint(frames),'splits':audit}
    audit_file=root/'results'/'data_audit.json'
    previous=json.loads(audit_file.read_text(encoding='utf8')) if audit_file.exists() else {}
    previous.update(report)
    audit_file.write_text(json.dumps(previous,indent=2,ensure_ascii=False),encoding='utf8')
    return all_data,report

def load_prepared(root,tasks):
    root=Path(root)
    data={}
    for task in tasks:
        data[task]={s:pd.read_csv(root/'data'/f'{task}_{s}.csv').fillna({'text':''}) for s in ['train','val','test']}
        for s in data[task]:data[task][s]['label']=data[task][s]['label'].astype(int)
    return data

def fingerprint(frames):
    h=hashlib.sha256()
    for name in ['train','val','test']:
        d=frames[name]
        h.update(name.encode())
        for row in d[['text','label']].itertuples(index=False,name=None):
            h.update((str(row[0])+'\t'+str(row[1])+'\n').encode('utf-8'))
    return h.hexdigest()

def feature_model(name):
    word=lambda pre=None:TfidfVectorizer(analyzer='word',ngram_range=(1,2),min_df=2,
                                      max_features=45000,sublinear_tf=True,preprocessor=pre)
    char=lambda pre=None:TfidfVectorizer(analyzer='char',ngram_range=(3,5),min_df=2,
                                      max_features=50000,sublinear_tf=True,preprocessor=pre)
    if name=='tfidf_svm':
        v=FeatureUnion([('word',word()),('char',char())])
        classifier=CalibratedClassifierCV(estimator=LinearSVC(C=1.0,class_weight='balanced'),method='sigmoid',cv=3)
        return Pipeline([('vectorizer',v),('classifier',classifier)])
    if name=='tfidf_word':v=word()
    elif name=='tfidf_char':v=char()
    elif name=='tfidf_word_char':v=FeatureUnion([('word',word()),('char',char())])
    elif name=='tfidf_normalized':v=FeatureUnion([('word',word(normalize_roman)),('char',char(normalize_roman))])
    else:raise ValueError(name)
    return Pipeline([('vectorizer',v),('classifier',LogisticRegression(
        max_iter=1500,class_weight='balanced',solver='lbfgs'))])

def tokens(text):return TOKEN_RX.findall(str(text).lower())
def vocab_build(texts,max_vocab=35000,min_freq=2):
    c=Counter(t for x in texts for t in tokens(x))
    d={'<PAD>':0,'<UNK>':1}
    for k,n in c.most_common():
        if len(d)>=max_vocab:break
        if n>=min_freq:d[k]=len(d)
    return d

def encode(text,vocab,max_len=110):return [vocab.get(t,1) for t in tokens(text)[:max_len]] or [1]

class Words(Dataset):
    def __init__(self,df,vocab):
        self.x=[encode(t,vocab) for t in df.text];self.y=list(map(int,df.label))
    def __len__(self):return len(self.x)
    def __getitem__(self,i):return self.x[i],self.y[i]

def batch_words(rows):
    ids,y=zip(*rows)
    lens=torch.tensor([len(s) for s in ids])
    x=nn.utils.rnn.pad_sequence([torch.tensor(s,dtype=torch.long) for s in ids],batch_first=True,padding_value=0)
    return x,lens,torch.tensor(y,dtype=torch.long)

class BiLSTM(nn.Module):
    def __init__(self,n_vocab,n_labels,pool='attention',emb=128,hidden=96):
        super().__init__()
        if pool not in ['last','meanmax','attention']:raise ValueError(pool)
        self.pool=pool
        self.emb=nn.Embedding(n_vocab,emb,padding_idx=0)
        self.rnn=nn.LSTM(emb,hidden,batch_first=True,bidirectional=True)
        if pool=='attention':self.att=nn.Linear(hidden*2,1)
        dims=hidden*4 if pool=='meanmax' else hidden*2
        self.head=nn.Sequential(nn.Dropout(.35),nn.Linear(dims,64),nn.ReLU(),nn.Dropout(.2),nn.Linear(64,n_labels))
    def forward(self,ids,lens):
        x=self.emb(ids)
        packed=nn.utils.rnn.pack_padded_sequence(x,lens.cpu(),batch_first=True,enforce_sorted=False)
        out,(h,_) = self.rnn(packed)
        if self.pool=='last':feat=torch.cat([h[-2],h[-1]],dim=-1)
        else:
            out,_=nn.utils.rnn.pad_packed_sequence(out,batch_first=True,total_length=ids.size(1))
            mask=torch.arange(ids.size(1),device=ids.device)[None,:]<lens.to(ids.device)[:,None]
            if self.pool=='attention':
                scores=self.att(out).squeeze(-1).masked_fill(~mask,-1e4)
                weights=torch.softmax(scores,dim=1)
                feat=(out*weights.unsqueeze(-1)).sum(1)
            else:
                mean=(out*mask.unsqueeze(-1)).sum(1)/lens.to(ids.device).clamp_min(1).unsqueeze(1)
                mx=out.masked_fill(~mask.unsqueeze(-1),-1e4).max(dim=1).values
                feat=torch.cat([mean,mx],dim=-1)
        return self.head(feat)

@torch.no_grad()
def neural_predict(model,df,vocab,device,batch_size=128):
    loader=DataLoader(Words(df,vocab),batch_size=batch_size,collate_fn=batch_words,shuffle=False)
    model.eval();parts=[]
    for x,l,_ in loader:
        y=model(x.to(device),l)
        parts.append(torch.softmax(y,dim=-1).cpu().numpy())
    return np.concatenate(parts,axis=0)

def fit_bilstm(train,val,task,model_name,path,device,epochs=8,batch_size=64,seed=SEED):
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    pool=model_name[len('bilstm_'):]
    vocab_file=path/'vocab.json';checkpoint=path/'train_state.pt';best_file=path/'best.pt'
    if vocab_file.exists():vocab=json.loads(vocab_file.read_text(encoding='utf8'))
    else:
        vocab=vocab_build(train.text)
        vocab_file.write_text(json.dumps(vocab,ensure_ascii=False),encoding='utf8')
    net=BiLSTM(len(vocab),len(LABELS[task]),pool=pool).to(device)
    count=train.label.value_counts().reindex(range(len(LABELS[task])),fill_value=1).to_numpy()
    weights=np.clip(len(train)/(len(LABELS[task])*count),.5,4.)
    loss_fn=nn.CrossEntropyLoss(weight=torch.tensor(weights,dtype=torch.float32,device=device))
    opt=torch.optim.AdamW(net.parameters(),lr=.001,weight_decay=.01)
    start,best,bad=0,-1,0
    if checkpoint.exists():
        state=torch.load(checkpoint,map_location=device,weights_only=False)
        net.load_state_dict(state['net']);opt.load_state_dict(state['opt'])
        start,best,bad=state['epoch'],state['best'],state['bad']
        if 'torch_rng' in state:torch.set_rng_state(state['torch_rng'].cpu())
        if torch.cuda.is_available() and 'cuda_rng' in state:torch.cuda.set_rng_state_all(state['cuda_rng'])
        print(f'Resuming {task}/{model_name} at epoch {start+1}')
    loader=DataLoader(Words(train,vocab),batch_size=batch_size,shuffle=True,collate_fn=batch_words)
    for epoch in range(start+1,epochs+1):
        net.train();total=0
        for x,l,y in loader:
            opt.zero_grad(set_to_none=True)
            logits=net(x.to(device),l);loss=loss_fn(logits,y.to(device))
            loss.backward();nn.utils.clip_grad_norm_(net.parameters(),1.);opt.step();total+=loss.item()*len(y)
        prob=neural_predict(net,val,vocab,device)
        f1=f1_score(val.label,prob.argmax(1),average='macro',zero_division=0)
        if f1>best+1e-5:
            best=f1;bad=0;torch.save(net.state_dict(),best_file)
        else:bad+=1
        state={'net':net.state_dict(),'opt':opt.state_dict(),'epoch':epoch,'best':best,'bad':bad,
               'torch_rng':torch.get_rng_state()}
        if torch.cuda.is_available():state['cuda_rng']=torch.cuda.get_rng_state_all()
        torch.save(state,checkpoint)
        print(f'{task}/{model_name}: epoch={epoch} loss={total/len(train):.4f} val_argmax_f1={f1:.4f}')
        if bad>=2:break
    if not best_file.exists():raise RuntimeError('BiLSTM best checkpoint missing')
    net.load_state_dict(torch.load(best_file,map_location=device,weights_only=True))
    return net,vocab

def _hf_imports():
    try:
        from transformers import AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding
        from transformers.trainer_utils import get_last_checkpoint
        return AutoTokenizer,AutoModelForSequenceClassification,Trainer,TrainingArguments,DataCollatorWithPadding,get_last_checkpoint
    except ImportError as e:raise ImportError('Install requirements-train.txt to run transformer training') from e

class TokenData(Dataset):
    """Tokenize by individual example; avoids full-DataFrame token-id RAM spike."""
    def __init__(self,df,tokenizer,max_len=128):
        self.texts=df.text.astype(str).tolist()
        self.y=df.label.astype(int).tolist()
        self.tokenizer=tokenizer
        self.max_len=max_len
    def __len__(self):return len(self.y)
    def __getitem__(self,i):
        enc=self.tokenizer(self.texts[i],truncation=True,max_length=self.max_len)
        return {**enc,'labels':self.y[i]}

def softmax(logits):
    z=np.asarray(logits[0] if isinstance(logits,tuple) else logits,dtype=float)
    z=z-z.max(axis=1,keepdims=True);z=np.exp(z)
    return z/z.sum(axis=1,keepdims=True)

def hf_predict(model,tokenizer,df,device,batch_size=16,max_len=128):
    # Avoid Trainer API/incompatible checkpoint usage during inference; direct torch inference.
    model.eval();prob=[]
    with torch.no_grad():
        for lo in range(0,len(df),batch_size):
            texts=df.text.iloc[lo:lo+batch_size].astype(str).tolist()
            inp=tokenizer(texts,truncation=True,max_length=max_len,padding=True,return_tensors='pt')
            logits=model(**{k:v.to(device) for k,v in inp.items()}).logits
            prob.append(torch.softmax(logits,dim=1).cpu().numpy())
    return np.vstack(prob)

def fit_transformer(train,val,task,name,path,device,epochs=3,batch=4,max_len=128,seed=SEED):
    AutoTok,AutoModel,Trainer,Args,Collator,last_ckpt=_hf_imports()
    path=Path(path);path.mkdir(parents=True,exist_ok=True)
    final=path/'final';run=path/'checkpoints'
    base=TRANSFORMERS[name]
    if (final/'config.json').exists():
        print('Loading existing finished transformer:',task,name)
        return AutoModel.from_pretrained(str(final)).to(device),AutoTok.from_pretrained(str(final))
    tokenizer=AutoTok.from_pretrained(base,use_fast=True)
    model=AutoModel.from_pretrained(base,num_labels=len(LABELS[task]),ignore_mismatched_sizes=True,
             id2label={i:v for i,v in enumerate(LABELS[task])},
             label2id={v:i for i,v in enumerate(LABELS[task])})
    kwargs=dict(output_dir=str(run),num_train_epochs=epochs,learning_rate=2e-5,
                per_device_train_batch_size=batch,per_device_eval_batch_size=batch*2,
                gradient_accumulation_steps=4,weight_decay=.01,
                save_strategy='epoch',load_best_model_at_end=True,metric_for_best_model='macro_f1',
                greater_is_better=True,save_total_limit=2,report_to='none',seed=seed,
                fp16=(device.type=='cuda'),logging_strategy='epoch',dataloader_num_workers=0)
    fields=inspect.signature(Args.__init__).parameters
    kwargs['eval_strategy' if 'eval_strategy' in fields else 'evaluation_strategy']='epoch'
    args=Args(**kwargs)
    def compute_metrics(ev):
        pred=ev.predictions[0] if isinstance(ev.predictions,tuple) else ev.predictions
        return {'macro_f1':f1_score(ev.label_ids,np.asarray(pred).argmax(1),average='macro',zero_division=0)}
    trainer=Trainer(model=model,args=args,
          train_dataset=TokenData(train,tokenizer,max_len),eval_dataset=TokenData(val,tokenizer,max_len),
          data_collator=Collator(tokenizer=tokenizer),compute_metrics=compute_metrics)
    resume=last_ckpt(str(run)) if run.is_dir() else None
    trainer.train(resume_from_checkpoint=resume)
    final.mkdir(parents=True,exist_ok=True)
    trainer.save_model(str(final));tokenizer.save_pretrained(str(final))
    del trainer;gc.collect()
    return model.to(device),tokenizer

def best_threshold(y,prob):
    y=np.asarray(y,dtype=int);pos=np.asarray(prob)[:,1]
    candidates=np.round(np.linspace(.10,.90,81),2)
    s=[f1_score(y,(pos>=t).astype(int),average='macro',zero_division=0) for t in candidates]
    rank=sorted(range(len(candidates)),key=lambda i:(-s[i],abs(candidates[i]-.5)))
    return float(candidates[rank[0]])

def classify(probs,threshold=.5):
    p=np.asarray(probs)
    return (p[:,1]>=threshold).astype(int) if p.shape[1]==2 else p.argmax(1)

def scores(y,probs,task,threshold=.5):
    true=np.asarray(y,dtype=int);p=np.asarray(probs,dtype=float);pred=classify(p,threshold)
    vals={'n':int(len(true)),'accuracy':float(accuracy_score(true,pred)),
          'macro_f1':float(f1_score(true,pred,average='macro',zero_division=0)),
          'weighted_f1':float(f1_score(true,pred,average='weighted',zero_division=0)),
          'macro_precision':float(precision_score(true,pred,average='macro',zero_division=0)),
          'macro_recall':float(recall_score(true,pred,average='macro',zero_division=0)),
          'balanced_accuracy':float(balanced_accuracy_score(true,pred)),
          'threshold':float(threshold)}
    if p.shape[1]==2:
        vals.update({'positive_precision':float(precision_score(true,pred,zero_division=0)),
                     'positive_recall':float(recall_score(true,pred,zero_division=0)),
                     'pr_auc_average_precision':float(average_precision_score(true,p[:,1])) if len(set(true))==2 else None,
                     'brier':float(brier_score_loss(true,p[:,1]))})
    return vals

def model_meta(task,name,data_sig,seed,epochs_lstm,epochs_hf,max_len):
    return {'task':task,'model':name,'split_sha256':data_sig,'seed':seed,
            'epochs_lstm':epochs_lstm,'epochs_hf':epochs_hf,'max_len':max_len,
            'architecture_version':'v5.0'}

def _check_existing(path,meta):
    p=Path(path)/'meta.json'
    if p.exists() and json.loads(p.read_text(encoding='utf8'))!=meta:
        raise RuntimeError(f'Config/data changed for {path}. Choose a NEW ROOT for a new study, '
                           'not old checkpoints from a different dataset or configuration.')
    path.mkdir(parents=True,exist_ok=True)
    if not p.exists():p.write_text(json.dumps(meta,indent=2),encoding='utf8')

def _device():return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_model(root,task,name,device=None):
    root=Path(root);device=device or _device();path=root/'models'/task/name
    if name=='ensemble2':
        obj=json.loads((path/'ensemble.json').read_text(encoding='utf8'))
        members=[load_model(root,task,m,device) for m in obj['members']]
        return ('ensemble',members,obj['weights'])
    if name.startswith('tfidf'):
        return ('classical',joblib.load(path/'model.joblib'),None)
    if name.startswith('bilstm'):
        vocab=json.loads((path/'vocab.json').read_text(encoding='utf8'))
        model=BiLSTM(len(vocab),len(LABELS[task]),pool=name.split('_',1)[1]).to(device)
        model.load_state_dict(torch.load(path/'best.pt',map_location=device,weights_only=True))
        return ('neural',model,vocab)
    # Deployment needs only Auto classes, not Trainer or Accelerate.
    from transformers import AutoTokenizer as AutoTok, AutoModelForSequenceClassification as AutoModel
    model=AutoModel.from_pretrained(str(path/'final')).to(device)
    tok=AutoTok.from_pretrained(str(path/'final'))
    return ('transformer',model,tok)

def predict_loaded(loaded,df,device=None,max_len=128):
    device=device or _device();typ,model,aux=loaded
    if typ=='ensemble':
        return sum(float(w)*predict_loaded(member,df,device,max_len=max_len) for w,member in zip(aux,model))
    if typ=='classical':return model.predict_proba(df.text.astype(str))
    if typ=='neural':return neural_predict(model,df,aux,device)
    return hf_predict(model,aux,df,device,max_len=max_len)

def fit_one(root,frames,task,name,seed=SEED,epochs_lstm=8,epochs_hf=3,max_len=128):
    assert name in MODEL_NAMES,f'Unknown model: {name}'
    root=output_paths(root);sig=fingerprint(frames);path=root/'models'/task/name
    meta=model_meta(task,name,sig,seed,epochs_lstm,epochs_hf,max_len)
    _check_existing(path,meta)
    done=path/'validated.json';device=_device()
    if done.exists():
        record=json.loads(done.read_text(encoding='utf8'))
        print(f'SKIP completed {task}/{name} (val macro-F1={record["metrics"]["macro_f1"]:.4f})')
        return record
    set_seed(seed)
    train,val=frames['train'],frames['val']
    if name.startswith('tfidf'):
        model=feature_model(name);model.fit(train.text.astype(str),train.label)
        joblib.dump(model,path/'model.joblib',compress=3)
        loaded=('classical',model,None)
    elif name.startswith('bilstm'):
        net,vocab=fit_bilstm(train,val,task,name,path,device,epochs_lstm,seed=seed)
        loaded=('neural',net,vocab)
    else:
        model,tokenizer=fit_transformer(train,val,task,name,path,device,epochs_hf,max_len=max_len,seed=seed)
        loaded=('transformer',model,tokenizer)
    prob=predict_loaded(loaded,val,device,max_len)
    threshold=best_threshold(val.label,prob) if len(LABELS[task])==2 else .5
    metrics=scores(val.label,prob,task,threshold)
    np.savez_compressed(path/'validation_probabilities.npz',probs=prob,labels=val.label.to_numpy(dtype=int))
    # Review criterion is validation-derived quantile, NOT claimed calibrated probability.
    review_cutoff=float(np.quantile(np.max(prob,axis=1),.20))
    record={'meta':meta,'metrics':metrics,'review_below':review_cutoff,'source_split':'val'}
    done.write_text(json.dumps(record,indent=2),encoding='utf8')
    print(f'COMPLETED {task}/{name}: validation macro-F1={metrics["macro_f1"]:.4f}, threshold={threshold:.2f}')
    del loaded;gc.collect()
    if torch.cuda.is_available():torch.cuda.empty_cache()
    return record

def scoreboard(root,task):
    root=Path(root);rows=[]
    for name in [*MODEL_NAMES,'ensemble2']:
        p=root/'models'/task/name/'validated.json'
        if p.exists():
            r=json.loads(p.read_text());rows.append({'model':name,**r['metrics']})
    if not rows:return pd.DataFrame()
    out=pd.DataFrame(rows).sort_values(['macro_f1','model'],ascending=[False,True]).reset_index(drop=True)
    out.to_csv(root/'results'/f'{task}_validation_scores.csv',index=False)
    return out

def propose_validation_ensemble(root,task,min_gain=.01):
    """Validation-only 2-model mixture; require >1pp macro-F1 improvement to avoid noise chasing.

    Call before locking; the test is neither loaded nor read here.
    """
    root=Path(root)
    tab=scoreboard(root,task)
    singles=tab[tab.model!='ensemble2']
    if len(singles)<2:return {'status':'not_enough_completed_models'}
    names=singles.model.iloc[:2].tolist()
    loaded=[np.load(root/'models'/task/n/'validation_probabilities.npz') for n in names]
    p1,p2=(a['probs'] for a in loaded)
    y1,y2=(a['labels'] for a in loaded)
    if not np.array_equal(y1,y2):raise RuntimeError('Validation probability label sequences mismatch')
    vals=[]
    for w in np.linspace(.2,.8,7):
        mix=w*p1+(1-w)*p2
        threshold=best_threshold(y1,mix) if mix.shape[1]==2 else .5
        metric=scores(y1,mix,task,threshold)
        vals.append((metric['macro_f1'],float(w),float(threshold),metric,mix))
    chosen=max(vals,key=lambda z:z[0])
    gain=chosen[0]-float(singles.iloc[0].macro_f1)
    if gain<min_gain:
        return {'status':'rejected','members':names,'best_gain':round(float(gain),5),'required_gain':min_gain}
    path=root/'models'/task/'ensemble2'
    if (root/'results'/'locked_validation_winners.json').exists():
        locked=json.loads((root/'results'/'locked_validation_winners.json').read_text())
        if task in locked:raise RuntimeError('Refusing to create validation ensemble after task lock')
    path.mkdir(parents=True,exist_ok=True)
    sig=json.loads((root/'models'/task/names[0]/'validated.json').read_text())['meta']['split_sha256']
    for name in names:
        meta=json.loads((root/'models'/task/name/'validated.json').read_text())['meta']
        if meta['split_sha256']!=sig:raise RuntimeError('Members trained on different splits')
    path.joinpath('ensemble.json').write_text(json.dumps({'members':names,'weights':[chosen[1],1-chosen[1]]},indent=2))
    meta={'task':task,'model':'ensemble2','split_sha256':sig,'max_len':128,
          'architecture_version':'v5.0','ensemble_members':names}
    result={'meta':meta,'metrics':chosen[3],
            'review_below':float(np.quantile(chosen[4].max(axis=1),.20)), 'source_split':'val'}
    path.joinpath('validated.json').write_text(json.dumps(result,indent=2))
    np.savez_compressed(path/'validation_probabilities.npz',probs=chosen[4],labels=y1)
    return {'status':'accepted','members':names,'gain':float(gain),'weights':[chosen[1],1-chosen[1]],'metrics':chosen[3]}

def select_and_lock(root,data,tasks,allow_existing=True):
    root=Path(root);lock_file=root/'results'/'locked_validation_winners.json'
    if lock_file.exists():
        old=json.loads(lock_file.read_text(encoding='utf8'))
    else:old={}
    for task in tasks:
        if task in old:
            print(f'{task} ALREADY LOCKED -> {old[task]["model"]}. No reselection permitted.')
            continue
        tab=scoreboard(root,task)
        if tab.empty:raise RuntimeError(f'No models trained for {task}')
        winner=tab.iloc[0]['model'];record=json.loads((root/'models'/task/winner/'validated.json').read_text())
        if record['meta']['split_sha256']!=fingerprint(data[task]):raise RuntimeError('Dataset fingerprint mismatch')
        old[task]={'model':winner,'threshold':record['metrics']['threshold'],
                   'review_below':record['review_below'], 'split_sha256':record['meta']['split_sha256'],
                   'selected_on':'validation macro-F1; holdout unseen', 'candidate_count':len(tab)}
    lock_file.write_text(json.dumps(old,indent=2),encoding='utf8')
    return old

def test_locked(root,frames,task,save_private_errors=True):
    root=Path(root);f=root/'results'/'locked_validation_winners.json'
    if not f.exists():raise RuntimeError('Select & lock winner BEFORE testing')
    r=json.loads(f.read_text(encoding='utf8'))
    if task not in r:raise RuntimeError(f'{task} not locked')
    locked=r[task]
    if locked['split_sha256']!=fingerprint(frames):raise RuntimeError('Test data signature differs from locked study')
    path=root/'results'/f'{task}_locked_test.json'
    if path.exists():
        print(f'{task} holdout already evaluated: leaving existing result untouched')
        return json.loads(path.read_text())
    test=frames['test'];device=_device()
    metadata=json.loads((root/'models'/task/locked['model']/'validated.json').read_text())['meta']
    prob=predict_loaded(load_model(root,task,locked['model'],device),test,device,max_len=metadata['max_len'])
    y=np.asarray(test.label)
    preds=classify(prob,locked['threshold'])
    report={'task':task,'winner':locked['model'], 'source':('previously_exposed_v4_internal_holdout' if task=='sentiment' else
                      'team_mirror_sentiMix_holdout_not_official_test' if task=='sentiment_raw' else 'source_supplied_holdout'),
            'metrics':scores(y,prob,task,locked['threshold']),
            'confusion':confusion_matrix(y,preds,labels=range(len(LABELS[task]))).tolist(),
            'per_class':classification_report(y,preds,labels=range(len(LABELS[task])),
                          target_names=LABELS[task],zero_division=0,output_dict=True)}
    err=test.loc[preds!=y,['text','label']].copy()
    err['pred']=preds[preds!=y]
    err['id']=err.text.map(lambda x:hashlib.sha256(str(x).encode()).hexdigest()[:14])
    if save_private_errors:err.to_csv(root/'private_analysis'/f'{task}_test_errors_PRIVATE.csv',index=False)
    # Never include raw training/test tweets in the public metric JSON.
    path.write_text(json.dumps(report,indent=2),encoding='utf8')
    return report

def synthetic_perturb(text):
    alias={'bahut':'bohot','bohot':'bahut','acha':'accha','accha':'acha',
           'nahi':'nhi','nhi':'nahi','yaar':'yr','kyun':'kyu','pyar':'pyaar'}
    return re.sub(r'(?ui)\b\w+\b',lambda m:alias.get(m.group(0).lower(),m.group(0)),str(text))

def script_bucket(text):
    # Script categories are NOT language IDs; Latin-only includes both English and Roman Hinglish.
    t=str(text)
    deva=bool(re.search(r'[\u0900-\u097F]',t))
    latin=bool(re.search(r'[A-Za-z]',t))
    return ('mixed_devanagari_latin' if deva and latin else
            'devanagari_only' if deva else 'latin_script' if latin else 'other')

def robustness_locked(root,frames,task):
    root=Path(root);lock=json.loads((root/'results'/'locked_validation_winners.json').read_text())[task]
    if fingerprint(frames)!=lock['split_sha256']:raise ValueError('Fingerprint mismatch')
    test=frames['test'];device=_device();loader=load_model(root,task,lock['model'],device)
    metadata=json.loads((root/'models'/task/lock['model']/'validated.json').read_text())['meta']
    max_len=metadata['max_len']
    altered=test.copy();altered['text']=altered.text.map(synthetic_perturb)
    changed=(altered.text!=test.text)
    result={'task':task,'method':'synthetic spelling variants; NOT a human-verified robustness benchmark',
            'n_changed':int(changed.sum())}
    if changed.sum()>10:
        a=predict_loaded(loader,test.loc[changed].reset_index(drop=True),device,max_len=max_len)
        b=predict_loaded(loader,altered.loc[changed].reset_index(drop=True),device,max_len=max_len)
        y=test.loc[changed].label
        result['original']=scores(y,a,task,lock['threshold'])
        result['perturbed']=scores(y,b,task,lock['threshold'])
        result['macro_f1_delta']=result['perturbed']['macro_f1']-result['original']['macro_f1']
    result['script_slices']={}
    for sc in sorted(set(test.text.map(script_bucket))):
        subset=test[test.text.map(script_bucket)==sc]
        if len(subset)<20 or subset.label.nunique()<2:continue
        px=predict_loaded(loader,subset,device,max_len=max_len)
        result['script_slices'][sc]=scores(subset.label,px,task,lock['threshold'])
    result['script_slices_note']='Script buckets are NOT validated language labels; Latin includes English/Hinglish.'
    if task=='hate' and 'lang' in test.columns:
        result['language_slices']={}
        for lang,g in test.groupby('lang'):
            if len(g)<10:continue
            p=predict_loaded(loader,g,device,max_len=max_len)
            result['language_slices'][str(lang)]=scores(g.label,p,task,lock['threshold'])
    (root/'results'/f'{task}_robustness.json').write_text(json.dumps(result,indent=2),encoding='utf8')
    return result

def predict_text(root,task,text,loaded=None,device=None):
    root=Path(root);device=device or _device()
    lock_file=root/'results'/'locked_validation_winners.json'
    locks=json.loads(lock_file.read_text(encoding='utf8'))
    if task not in locks:raise RuntimeError(f'{task} has no locked trained model')
    spec=locks[task];loaded=loaded or load_model(root,task,spec['model'],device)
    dummy=pd.DataFrame({'text':[clean(text)],'label':[0]})
    metadata=json.loads((root/'models'/task/spec['model']/'validated.json').read_text())['meta']
    p=predict_loaded(loaded,dummy,device,max_len=metadata['max_len'])[0]
    idx=int(classify(p[None,:],spec['threshold'])[0])
    return {'task':task,'model':spec['model'],'label':LABELS[task][idx],
            'probabilities':{name:round(float(v),4) for name,v in zip(LABELS[task],p)},
            'uncalibrated_max_probability':round(float(np.max(p)),4),
            'review_required':bool(np.max(p)<spec['review_below']),
            'review_threshold_validation_derived':round(float(spec['review_below']),4),
            'positive_decision_threshold':spec['threshold'] if len(p)==2 else None,
            'calibrated':False,'research_only':True}


def paired_bootstrap_validation(root,task,model_a,model_b,repeats=1000,seed=SEED):
    """Paired bootstrap of *validation* macro-F1 difference B - A; not a test-set significance claim."""
    root=Path(root)
    vals=[]
    for name in (model_a,model_b):
        p=root/'models'/task/name/'validation_probabilities.npz'
        if not p.exists():raise FileNotFoundError(p)
        arr=np.load(p)
        params=json.loads((root/'models'/task/name/'validated.json').read_text())['metrics']
        vals.append((arr['probs'],arr['labels'],float(params['threshold'])))
    (pa,ya,ta),(pb,yb,tb)=vals
    if not np.array_equal(ya,yb):raise ValueError('Validation samples differ; comparison invalid')
    a=classify(pa,ta);b=classify(pb,tb);rng=np.random.default_rng(seed)
    n=len(ya);d=[]
    for _ in range(repeats):
        indices=rng.integers(0,n,n)
        d.append(f1_score(ya[indices],b[indices],average='macro',zero_division=0)-
                 f1_score(ya[indices],a[indices],average='macro',zero_division=0))
    point=float(f1_score(ya,b,average='macro')-f1_score(ya,a,average='macro'))
    return {'task':task,'model_a':model_a,'model_b':model_b,
            'delta_b_minus_a':point,'bootstrap_95_pct_interval':list(map(float,np.quantile(d,[.025,.975]))),
            'samples':n,'bootstrap_repeats':repeats,'scope':'same validation examples; exploratory'}


def evaluate_external_challenge(root,task,path):
    """Independently annotated challenge CSV with exact, task-specific human labels.

    Reject any text seen in *any* experiment split. Does not edit/select models.
    """
    root=Path(root)
    csv=Path(path)
    df=pd.read_csv(csv)
    if not {'text','label'}.issubset(df.columns):
        raise ValueError('Challenge CSV needs text,label. Label must be an exact task label name.')
    if df.text.isna().any() or df.label.isna().any():raise ValueError('Challenge labels cannot be blank')
    df['text']=df.text.astype(str).map(clean)
    df['label']=df.label.astype(str).str.strip()
    classes=LABELS[task]
    if not set(df.label).issubset(set(classes)):
        raise ValueError(f'Challenge labels must be one of: {classes}. Got: {sorted(set(df.label)-set(classes))}')
    if len(df)<20:raise ValueError('Challenge set too small; require >=20 human-labeled examples')
    keys=_key_series(df)
    if keys.duplicated().any():raise ValueError('Challenge set contains exact duplicate texts')
    seen=set()
    for split in ('train','val','test'):
        dftrain=pd.read_csv(root/'data'/f'{task}_{split}.csv')
        seen.update(_key_series(dftrain))
    overlap=set(keys)&seen
    if overlap:raise ValueError(f'{len(overlap)} challenge texts overlap training/validation/test; refusing contaminated evaluation')
    locked=json.loads((root/'results'/'locked_validation_winners.json').read_text())
    if task not in locked:raise RuntimeError('Task must be locked before independent evaluation')
    info=locked[task]
    metadata=json.loads((root/'models'/task/info['model']/'validated.json').read_text())['meta']
    tmp=df[['text']].copy();tmp['label']=df.label.map({name:i for i,name in enumerate(classes)})
    probs=predict_loaded(load_model(root,task,info['model']),tmp,max_len=metadata['max_len'])
    result={'source':'human_annotated_independent_challenge',
            'task':task,'model':info['model'],
            'metrics':scores(tmp.label,probs,task,info['threshold']),
            'per_class':classification_report(tmp.label,classify(probs,info['threshold']),
                        labels=range(len(classes)),target_names=classes,output_dict=True,zero_division=0),
            'scope':'Human annotation quality and independence must be documented separately'}
    out=root/'results'/f'{task}_external_challenge.json'
    out.write_text(json.dumps(result,indent=2,ensure_ascii=False),encoding='utf8')
    return result
