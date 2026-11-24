#!/usr/bin/env python3
"""Build publication-style VALIDATION diagnostics from existing MixBhasha v5 outputs.

No training, network, heavyweight model loading, or holdout re-evaluation.
Raw text is not included in generated public reports.
"""
from __future__ import annotations
import argparse
import html
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, confusion_matrix,
    f1_score, precision_score, recall_score, precision_recall_curve,
    roc_curve, average_precision_score, roc_auc_score
)

LABELS = {
    'sentiment_raw': ['negative', 'neutral', 'positive'],
    'sentiment': ['negative', 'neutral', 'positive'],
    'cyberbullying7': ['not_cyberbullying', 'age', 'gender', 'religion', 'mockery', 'abusive', 'offensive'],
    'cyberbullying_binary': ['not_cyberbullying', 'cyberbullying_category'],
    'offense_hasoc': ['not_hate_offensive', 'hate_or_offensive'],
    'hate': ['non_hate', 'hate'],
}
DPI = 160


def _json(p: Path):
    return json.loads(p.read_text(encoding='utf-8'))


def _save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def _barh(names, values, title, x_label, dest, max_x=None):
    fig, ax = plt.subplots(figsize=(11, max(3.8, len(names) * .58 + 1.6)))
    idx = np.arange(len(names))
    ax.barh(idx, values)
    ax.set_yticks(idx, names)
    ax.invert_yaxis()
    ax.set_xlabel(x_label)
    ax.set_title(title, pad=14)
    ax.grid(axis='x', alpha=.2)
    ax.set_axisbelow(True)
    maxval = max(max(values) if len(values) else 0, .001)
    ax.set_xlim(0, max_x if max_x is not None else maxval * 1.18)
    for i, v in enumerate(values):
        ax.text(float(v) + (maxval * .012), i, f'{v:.3f}', va='center', fontsize=9)
    _save(fig, dest)


def _matrix(matrix, row_names, col_names, title, dest, fmt='{:.2f}', vmax=None):
    arr = np.asarray(matrix, dtype=float)
    fig, ax = plt.subplots(figsize=(max(7.5, len(col_names) * 1.3), max(5, len(row_names) * .68 + 1.5)))
    im = ax.imshow(arr, vmin=0, vmax=vmax if vmax is not None else max(float(arr.max(initial=1.)), 1e-6), cmap='Blues', aspect='auto')
    ax.set_xticks(np.arange(len(col_names)), col_names, rotation=40, ha='right')
    ax.set_yticks(np.arange(len(row_names)), row_names)
    ax.set_title(title, pad=13)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            val = arr[i, j]
            ax.text(j, i, fmt.format(val), ha='center', va='center', fontsize=8,
                    color='white' if val > (.58 * (im.norm.vmax or 1.)) else 'black')
    fig.colorbar(im, ax=ax, shrink=.80)
    _save(fig, dest)


def _assert_data_integrity(root, task, model, meta, labels_array):
    # Existing model metadata must match the saved task dataset; do not use test labels to select models.
    audit_file = root / 'results' / 'data_audit.json'
    if audit_file.exists():
        all_audits = _json(audit_file)
        expected = all_audits.get(task, {}).get('split_sha256')
        observed = meta.get('split_sha256')
        if expected and expected != observed:
            raise RuntimeError(f'Split fingerprint mismatch for {task}/{model}; do not combine these studies')
    split_file = root / 'data' / f'{task}_val.csv'
    if split_file.exists():
        saved = pd.read_csv(split_file)
        if len(saved) != len(labels_array) or not np.array_equal(saved['label'].astype(int).to_numpy(), labels_array):
            raise RuntimeError(f'Validation row-order/labels mismatch: {task}/{model}')


def _read_task(root, task):
    model_root = root / 'models' / task
    if not model_root.is_dir():
        return []
    candidates = []
    for model_dir in sorted(model_root.iterdir()):
        if not model_dir.is_dir():
            continue
        record = model_dir / 'validated.json'
        pred_path = model_dir / 'validation_probabilities.npz'
        if not record.exists() or not pred_path.exists():
            continue
        saved = _json(record)
        with np.load(pred_path, allow_pickle=False) as packed:
            y = np.asarray(packed['labels'], dtype=int)
            probs = np.asarray(packed['probs'], dtype=float)
        if probs.ndim != 2 or probs.shape != (len(y), len(LABELS[task])):
            raise ValueError(f'Invalid probability shape for {task}/{model_dir.name}: {probs.shape}')
        if not np.isfinite(probs).all() or np.any(probs < -1e-6):
            raise ValueError(f'Nonfinite/negative probabilities for {task}/{model_dir.name}')
        if not np.allclose(probs.sum(axis=1), 1.0, atol=2e-3):
            raise ValueError(f'Probability row sums invalid for {task}/{model_dir.name}')
        if not np.isin(y, np.arange(probs.shape[1])).all():
            raise ValueError(f'Invalid class IDs for {task}/{model_dir.name}')
        _assert_data_integrity(root, task, model_dir.name, saved['meta'], y)
        candidates.append(dict(name=model_dir.name, dir=model_dir, y=y, prob=probs,
                               metrics=saved.get('metrics', {}), meta=saved.get('meta', {})))
    if candidates:
        y0 = candidates[0]['y']
        sig0 = candidates[0]['meta'].get('split_sha256')
        for c in candidates[1:]:
            if not np.array_equal(c['y'], y0) or c['meta'].get('split_sha256') != sig0:
                raise RuntimeError(f'Different validation examples in {task}; refusing model comparison')
    return candidates


def _predictions(y, prob, threshold=0.5):
    return (prob[:, 1] >= threshold).astype(int) if prob.shape[1] == 2 else prob.argmax(axis=1)


def _confidence_diagnostics(y, prob):
    pred = prob.argmax(axis=1)
    conf = prob.max(axis=1)
    correct = (pred == y).astype(float)
    rows = []
    for k in range(10):
        lo, hi = k / 10, (k + 1) / 10
        mask = (conf >= lo) & ((conf < hi) if k < 9 else (conf <= hi))
        rows.append({'bin': k, 'n': int(mask.sum()), 'mean_confidence': float(conf[mask].mean()) if mask.any() else np.nan,
                     'accuracy': float(correct[mask].mean()) if mask.any() else np.nan})
    ece = float(sum((r['n'] / len(y)) * abs(r['accuracy'] - r['mean_confidence'])
                    for r in rows if r['n'] > 0))
    y_onehot = np.eye(prob.shape[1])[y]
    multiclass_brier = float(np.mean(np.sum((prob - y_onehot)**2, axis=1)))
    return pd.DataFrame(rows), ece, multiclass_brier


def _plot_reliability(bins, ece, name, path):
    fig, ax = plt.subplots(figsize=(6.5, 5))
    ax.plot([0, 1], [0, 1], linestyle='--', label='Perfect confidence calibration')
    nz = bins[bins['n'] > 0]
    ax.plot(nz.mean_confidence, nz.accuracy, marker='o', label='Validation empirical accuracy')
    for row in nz.itertuples(index=False):
        ax.annotate(str(row.n), (row.mean_confidence, row.accuracy), xytext=(3, 5), textcoords='offset points', fontsize=7)
    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Predicted maximum class probability',
           ylabel='Observed accuracy', title=f'{name}: confidence diagnostics (ECE={ece:.3f})')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(alpha=.25)
    _save(fig, path)


def _plot_ranking_curves(y, prob, class_names, task, model, base):
    onehot = np.eye(len(class_names))[y]
    for kind in ('pr', 'roc'):
        fig, ax = plt.subplots(figsize=(7.5, 5.6))
        any_plotted = False
        for ix, label in enumerate(class_names):
            binary = onehot[:, ix]
            if len(np.unique(binary)) < 2:
                continue
            if kind == 'pr':
                prec, rec, _ = precision_recall_curve(binary, prob[:, ix])
                score = average_precision_score(binary, prob[:, ix])
                ax.plot(rec, prec, label=f'{label} (AP={score:.3f})')
            else:
                fpr, tpr, _ = roc_curve(binary, prob[:, ix])
                score = roc_auc_score(binary, prob[:, ix])
                ax.plot(fpr, tpr, label=f'{label} (AUC={score:.3f})')
            any_plotted = True
        if kind == 'roc':
            ax.plot([0,1], [0,1], linestyle='--', alpha=.45)
        ax.set(xlim=(0,1), ylim=(0,1), xlabel='Recall' if kind == 'pr' else 'False positive rate',
               ylabel='Precision' if kind == 'pr' else 'True positive rate',
               title=f'{task} — {model} — one-vs-rest {kind.upper()} (validation)')
        ax.legend(fontsize=8, loc='best')
        ax.grid(alpha=.2)
        if any_plotted:
            _save(fig, base / f'{model}_{kind}_per_class.png')
        else:
            plt.close(fig)


def _training_curves(model_dir: Path, name: str, target: Path):
    # HF Trainer states are kept in checkpoints; scratch BiLSTM v5 only saves weights/epochs, not epoch logs.
    files = list((model_dir / 'checkpoints').rglob('trainer_state.json')) if (model_dir / 'checkpoints').is_dir() else []
    if not files:
        return False
    state = max(files, key=lambda x: x.stat().st_mtime)
    items = _json(state).get('log_history', [])
    tr = [(x.get('epoch', x.get('step')), x['loss']) for x in items if 'loss' in x and x.get('epoch',x.get('step')) is not None]
    ev = [(x.get('epoch', x.get('step')), x['eval_loss']) for x in items if 'eval_loss' in x and x.get('epoch', x.get('step')) is not None]
    f1 = [(x.get('epoch', x.get('step')), x['eval_macro_f1']) for x in items if 'eval_macro_f1' in x and x.get('epoch', x.get('step')) is not None]
    if tr or ev:
        fig,ax = plt.subplots(figsize=(7.5,4.6))
        for pts, lab in ((tr, 'Training loss'), (ev, 'Validation loss')):
            if pts:
                xs, ys = zip(*pts); ax.plot(xs,ys,marker='o' if len(pts)<10 else None,label=lab)
        ax.set(xlabel='Epoch', ylabel='Loss', title=f'{name} — saved Trainer history')
        ax.grid(alpha=.2);ax.legend(); _save(fig,target/f'{name}_learning_curve_loss.png')
    if f1:
        fig,ax = plt.subplots(figsize=(7.5,4.6)); xs,ys=zip(*f1)
        ax.plot(xs,ys,marker='o');ax.set(xlabel='Epoch',ylabel='Validation macro-F1',title=f'{name} — saved Trainer F1')
        ax.grid(alpha=.2);_save(fig,target/f'{name}_learning_curve_macro_f1.png')
    return bool(tr or ev or f1)


def _bootstrap_two(a, b, repetitions=700, seed=42):
    y = a['y'];n = len(y);rng = np.random.default_rng(seed)
    p_a = _predictions(y, a['prob'], a['metrics'].get('threshold', 0.5))
    p_b = _predictions(y, b['prob'], b['metrics'].get('threshold', 0.5))
    observed = f1_score(y, p_a, average='macro',zero_division=0)-f1_score(y,p_b,average='macro',zero_division=0)
    deltas = np.empty(repetitions)
    for k in range(repetitions):
        idx = rng.integers(0,n,n)
        deltas[k]=f1_score(y[idx],p_a[idx],average='macro',zero_division=0)-f1_score(y[idx],p_b[idx],average='macro',zero_division=0)
    return float(observed),np.quantile(deltas,[.025,.975]).tolist(),deltas


def _manual_review_aggregates(root, task, dest, charts=None):
    """Aggregate selected manual-review CSVs without exporting private text or row-level predictions.

    These samples are purposively selected from model errors; never infer dataset-wide rates.
    """
    candidates = [root/'private_analysis'/f'{task}_error_review_completed.csv',
                  root/'private_analysis'/f'{task}_error_review.csv']
    p = next((c for c in candidates if c.exists()), None)
    if p is None:
        return None
    df = pd.read_csv(p)
    summary = {}
    for col in ('error_reason', 'original_label_questionable'):
        if col not in df:
            continue
        series = df[col].fillna('unreviewed').astype(str).str.strip().replace('', 'unreviewed')
        counts = series.value_counts()
        counts.rename_axis('category').reset_index(name='count').to_csv(dest/f'manual_{col}_counts.csv',index=False)
        summary[col] = {str(k): int(v) for k,v in counts.items()}
        if charts is not None:
            _barh(counts.index.astype(str).tolist(), counts.to_numpy(dtype=float),
                  f'{task} — manually reviewed {col.replace("_", " ")}', 'Reviewed mistakes (selected sample)',
                  charts/f'manual_{col}_counts.png')
    if 'actual_label' in df and 'error_reason' in df:
        tab = pd.crosstab(df['actual_label'], df['error_reason'])
        tab.to_csv(dest/'manual_error_reason_by_original_class.csv')
        if charts is not None:
            _matrix(tab.to_numpy(), list(tab.index), list(tab.columns),
                    f'{task} — error reasons by original category (selected mistakes)',
                    charts/'manual_error_reasons_by_class.png', fmt='{:.0f}')
    if 'actual_label' in df and 'original_label_questionable' in df:
        tab = pd.crosstab(df['actual_label'], df['original_label_questionable'])
        tab.to_csv(dest/'manual_label_audit_by_original_class.csv')
        if charts is not None:
            _matrix(tab.to_numpy(), list(tab.index), list(tab.columns),
                    f'{task} — original-label review flags (selected mistakes)',
                    charts/'manual_label_audit_by_class.png',fmt='{:.0f}')
    if summary:
        (dest/'manual_review_aggregate.json').write_text(json.dumps({'reviewed_sample_size':int(len(df)),
          'note':'Purposively sampled misclassifications. NOT representative of source dataset. Review flags are not verified relabels.',
          **summary},indent=2),encoding='utf8')
    return summary


def _audit_aggregates(root, dest):
    p=root/'results'/'data_audit.json'
    if not p.exists():return
    audit=_json(p); rows=[]
    for task, record in audit.items():
        for split,entry in record.get('splits', {}).items():
            if isinstance(entry,dict) and split in ('train','val','test'):
                row={'task':task,'split':split}
                for k in ('input_rows','post_normalization_rows','kept','removed_total','conflicting_rows_removed','cross_split_overlap_rows_removed'):
                    if k in entry:row[k]=entry[k]
                for cat,n in entry.get('class_counts', {}).items():row[f'class_{cat}']=n
                rows.append(row)
    if rows:pd.DataFrame(rows).to_csv(dest/'data_audit_aggregate.csv',index=False)


def create_report(root: Path, out: Path, tasks: list[str]|None=None, bootstrap=700):
    root=Path(root).expanduser().resolve()
    out=Path(out).expanduser().resolve()
    out.mkdir(parents=True,exist_ok=True)
    (out/'charts').mkdir(exist_ok=True)
    _audit_aggregates(root,out)
    tasks=tasks or [t for t in LABELS if (root/'models'/t).exists()]
    summary={}
    for task in tasks:
        if task not in LABELS:
            warnings.warn(f'Unknown task {task}; skipping')
            continue
        models=_read_task(root,task)
        if not models:
            print(f'NO VALIDATION ARTIFACTS: {task}');continue
        labels=LABELS[task]
        tdir=out/task;tdir.mkdir(exist_ok=True)
        charts=out/'charts'/task; charts.mkdir(parents=True,exist_ok=True)
        y=models[0]['y'];n=len(y)
        counts=np.bincount(y,minlength=len(labels))
        pd.DataFrame({'label':labels,'validation_support':counts}).to_csv(tdir/'class_support.csv',index=False)
        _barh(labels,counts.astype(float),f'{task} — validation class support','Samples',charts/'class_balance.png')
        # Distribution across train/validation/test from saved split labels; this only counts classes.
        split_rows=[]
        for split in ('train','val','test'):
            csv_file=root/'data'/f'{task}_{split}.csv'
            if csv_file.exists():
                split_labels=pd.read_csv(csv_file,usecols=['label'])['label'].astype(int).to_numpy()
                for i,lab in enumerate(labels):
                    split_rows.append({'split':split,'label':lab,'count':int(np.sum(split_labels==i))})
        if split_rows:
            split_df=pd.DataFrame(split_rows)
            split_df.to_csv(tdir/'split_class_distribution.csv',index=False)
            pivot=split_df.pivot(index='label',columns='split',values='count').reindex(index=labels)
            fig,ax=plt.subplots(figsize=(10,max(4.3,.6*len(labels)+2)))
            x=np.arange(len(labels));splits=list(pivot.columns);width=.75/len(splits)
            for i,split in enumerate(splits):
                ax.barh(x+(i-(len(splits)-1)/2)*width,pivot[split].values,width,label=split)
            ax.set_yticks(x,labels);ax.invert_yaxis();ax.set_xlabel('Number of examples')
            ax.set_title(f'{task} — label coverage by split');ax.legend();ax.grid(axis='x',alpha=.2)
            _save(fig,charts/'split_class_distribution.png')

        board=[];classes=[];predictions=[];all_cal=[]
        for m in models:
            name=m['name'];p=m['prob'];pred=_predictions(y, p, m['metrics'].get('threshold', 0.5))
            # Size of inference-ready artifacts only: exclude ephemeral trainer checkpoints.
            model_bytes = sum(f.stat().st_size for f in m['dir'].rglob('*')
                              if f.is_file() and 'checkpoints' not in f.relative_to(m['dir']).parts
                              and f.name not in {'train_state.pt', 'optimizer.pt', 'scheduler.pt'})
            board.append({'model':name,'n':n,'inference_artifact_mib':round(model_bytes/1024**2,2),
                'max_len':m['meta'].get('max_len'),'accuracy':accuracy_score(y,pred),
                'macro_f1':f1_score(y,pred,average='macro',zero_division=0),
                'weighted_f1':f1_score(y,pred,average='weighted',zero_division=0),
                'macro_precision':precision_score(y,pred,average='macro',zero_division=0),
                'macro_recall':recall_score(y,pred,average='macro',zero_division=0),
                'balanced_accuracy':balanced_accuracy_score(y,pred),
                'model_training_seed':m['meta'].get('seed'),
                'split_sha256':m['meta'].get('split_sha256')})
            cm=confusion_matrix(y,pred,labels=np.arange(len(labels)))
            pd.DataFrame(cm,index=labels,columns=labels).to_csv(tdir/f'{name}_confusion.csv',index_label='actual_label')
            _matrix(cm,labels,labels,f'{task} — {name} — confusion counts',charts/f'{name}_confusion.png',fmt='{:.0f}')
            cmnorm=cm/np.maximum(1,cm.sum(axis=1,keepdims=True))
            _matrix(cmnorm,labels,labels,f'{task} — {name} — recall-normalized confusion',charts/f'{name}_confusion_normalized.png',fmt='{:.2f}',vmax=1.)
            class_rows=[]
            for c, label in enumerate(labels):
                binary=(y==c); c_pred=(pred==c)
                cl={'model':name,'class_label':label,'support':int(binary.sum()),
                    'precision':precision_score(binary,c_pred,zero_division=0),
                    'recall':recall_score(binary,c_pred,zero_division=0),
                    'f1':f1_score(binary,c_pred,zero_division=0),
                    'average_precision_ovr':average_precision_score(binary,p[:,c]) if binary.any() else np.nan}
                class_rows.append(cl); classes.append(cl)
            per_class=pd.DataFrame(class_rows)
            per_class.to_csv(tdir/f'{name}_per_class.csv',index=False)
            _barh(labels,per_class.f1.to_numpy(),f'{task} — {name} — per-class F1','F1',charts/f'{name}_per_class_f1.png',max_x=1.05)
            _plot_ranking_curves(y,p,labels,task,name,charts)
            bins,ece,brier=_confidence_diagnostics(y,p)
            bins.to_csv(tdir/f'{name}_reliability_bins.csv',index=False)
            all_cal.append({'model':name,'ece_10_bins':ece,'multiclass_brier':brier,
                            'caution':'Validation confidence diagnostic; only calibrated SVM has fitting-time calibration'})
            _plot_reliability(bins,ece,name,charts/f'{name}_reliability.png')
            predictions.append(pred)
            _training_curves(m['dir'],name,charts)
        board=pd.DataFrame(board).sort_values(['macro_f1','model'],ascending=[False,True]).reset_index(drop=True)
        board.to_csv(tdir/'leaderboard.csv',index=False)
        pd.DataFrame(classes).to_csv(tdir/'all_models_per_class.csv',index=False)
        pd.DataFrame(all_cal).to_csv(tdir/'confidence_diagnostics.csv',index=False)
        _barh(board.model.tolist(),board.macro_f1.to_numpy(),f'{task} — macro-F1 ranking (validation)','Macro-F1',charts/'model_ranking_macro_f1.png',max_x=1.05)
        # Four metrics in the same chart (same scale).
        compare=['macro_f1','accuracy','weighted_f1','balanced_accuracy']
        fig,ax=plt.subplots(figsize=(12,max(4.3,.75*len(board)+1.6)))
        idx=np.arange(len(board));width=.2
        for i,k in enumerate(compare):
            ax.barh(idx+(i-1.5)*width,board[k].values,height=width,label=k)
        ax.set_yticks(idx,board.model);ax.invert_yaxis();ax.set_xlim(0,1.07)
        ax.set_xlabel('Validation score');ax.set_title(f'{task} — model metrics comparison');ax.legend(ncol=2)
        ax.grid(axis='x',alpha=.2);ax.set_axisbelow(True)
        _save(fig,charts/'model_metrics_comparison.png')
        f1table=pd.DataFrame(classes).pivot(index='model',columns='class_label',values='f1').reindex(index=board.model,columns=labels)
        f1table.to_csv(tdir/'class_f1_comparison.csv')
        _matrix(f1table.to_numpy(),f1table.index.tolist(),f1table.columns.tolist(),
                f'{task} — F1 per class and model',charts/'class_f1_heatmap.png',fmt='{:.2f}',vmax=1.)
        # Percentage of same predictions; NOT proof of correctness or independent annotator agreement.
        pnames=[m['name'] for m in models]
        agreement=np.array([[(a==b).mean() for b in predictions] for a in predictions])
        pd.DataFrame(agreement,index=pnames,columns=pnames).to_csv(tdir/'model_prediction_agreement.csv')
        _matrix(agreement,pnames,pnames,f'{task} — validation prediction agreement',charts/'model_agreement.png',vmax=1.)
        bootstrap_info=None
        if len(models)>=2 and bootstrap>0:
            lookup={m['name']:m for m in models}
            top1,top2=board.model.iloc[0],board.model.iloc[1]
            delta,ci,draws=_bootstrap_two(lookup[top1],lookup[top2],repetitions=bootstrap)
            fig,ax=plt.subplots(figsize=(7.5,4.5));ax.hist(draws,bins=35)
            ax.axvline(0,ls='--',lw=1,label='Equal macro-F1')
            ax.axvline(delta,ls='-',lw=1,label='Observed delta')
            ax.set(xlabel=f'Macro-F1 ({top1} − {top2})',ylabel='Bootstrap count',title=f'{task} — paired validation uncertainty')
            ax.legend();_save(fig,charts/'paired_bootstrap_top2.png')
            bootstrap_info={'comparison':f'{top1} minus {top2}', 'delta_macro_f1':delta,'bootstrap_95_percent_interval':ci,
                            'resamples':bootstrap,'scope':'Exploratory, paired validation sample; not across random training seeds.'}
            (tdir/'paired_bootstrap_top2.json').write_text(json.dumps(bootstrap_info,indent=2),encoding='utf8')
        manual=_manual_review_aggregates(root,task,tdir,charts)
        summary[task]={'models':[str(s) for s in board.model],'best_model':str(board.model.iloc[0]),
                       'best_validation_macro_f1':float(board.macro_f1.iloc[0]),'n_validation':n,
                       'bootstrap_top2':bootstrap_info,'manual_review_counts_if_available':manual}
        print(f'CREATED {task}: {len(models)} model(s), {n} validation rows -> {tdir}')
    (out/'report_summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
    md=['# MixBhasha — Validation Experiment Report','',
        '> Generated from saved model outputs. No training or held-out test evaluation was performed.',
        '> Task-specific datasets and holdouts are not directly cross-comparable.','',
        '> Review samples are selected from model mistakes and **do not estimate dataset-wide label error rates**.','']
    for task,result in summary.items():
        md += [f'## {task}','',f"- Validation samples: {result['n_validation']}",
               f"- Observed leading model: **{result['best_model']}**",
               f"- Macro-F1: **{result['best_validation_macro_f1']:.4f}**",'',
               f'![Ranking](charts/{task}/model_ranking_macro_f1.png)','',
               f'![Class F1](charts/{task}/class_f1_heatmap.png)','',
               f'![Model comparison](charts/{task}/model_metrics_comparison.png)','']
        if result.get('bootstrap_top2'):
            d=result['bootstrap_top2'];md += [f"- Paired exploratory bootstrap, {d['comparison']}: {d['delta_macro_f1']:.4f}, 95% interval {d['bootstrap_95_percent_interval']}",'']
        # Link CSV tables instead of embedding social media source examples.
        md += [f'- [Leaderboard CSV]({task}/leaderboard.csv)',f'- [Class F1 table]({task}/class_f1_comparison.csv)','']
    md += ['## Methodological notes','','- Each candidate uses its saved **validation** probability vectors and the same class order.',
           '- Model comparison refuses mismatched split fingerprints or validation label rows.',
           '- Precision-recall/ROC curves are **one-versus-rest**, not threshold-tuned model rankings.',
           '- Confidence reliability diagrams are diagnostic; do not claim neural probabilities are calibrated.',
           '- Training curves are drawn only when original Trainer logs exist. Bi-LSTM epoch history is unavailable if it was never persisted.',
           '- No source texts, private annotation rows, raw training/test CSVs, or model weights are included in this report.',
           '- Source code, datasets and pretrained model licenses must be checked independently before public release.','']
    (out/'REPORT.md').write_text('\n'.join(md),encoding='utf8')
    # Self-contained browser-friendly HTML with tables and relative PNG links (zip preserves paths).
    parts=['<!doctype html><html lang="en"><head><meta charset="utf-8"><title>MixBhasha Validation Report</title>',
           '<style>body{max-width:1200px;margin:2rem auto;font:16px/1.55 system-ui,sans-serif;color:#15233b;padding:0 1rem}h1,h2{color:#154785}img{max-width:100%;display:block;margin:1.5rem auto;border:1px solid #dde5ef;border-radius:9px}table{border-collapse:collapse;width:100%;font-size:.88rem}td,th{padding:.55rem;border-bottom:1px solid #ddd;text-align:right}th:first-child,td:first-child{text-align:left}.note{background:#f3f7ff;padding:1rem;border-radius:8px}</style></head><body>',
           '<h1>MixBhasha — Validation Experiment Report</h1><p class="note">Derived from saved validation predictions only. No new test data were accessed. Do not equate the different task metrics or claim calibrated confidence without calibration checks.</p>']
    for task in summary:
        tab=pd.read_csv(out/task/'leaderboard.csv')
        columns=['model','n','accuracy','macro_f1','weighted_f1','balanced_accuracy']
        pretty=tab[columns].copy()
        for col in columns[2:]:pretty[col]=pretty[col].map(lambda x:f'{x:.4f}')
        parts.append('<h2>'+html.escape(task)+'</h2>')
        parts.append(pretty.to_html(index=False,escape=True,border=0))
        for fig in ('model_ranking_macro_f1.png','class_f1_heatmap.png','model_metrics_comparison.png','model_agreement.png'):
            parts.append(f'<img loading="lazy" src="charts/{html.escape(task)}/{fig}" alt="{html.escape(task)} {fig}">')
        parts.append(f'<p>Additional per-model confusion matrices, precision–recall curves, ROC, confidence diagnostics, class tables and saved training histories are in <code>charts/{html.escape(task)}/</code>.</p>')
    parts.append('<p class="note">Validation-only model selection; test results (if previously generated) are separate and not re-evaluated. Reviewed errors are selected and not representative.</p></body></html>')
    (out/'REPORT.html').write_text('\n'.join(parts),encoding='utf8')
    print('REPORT FINISHED:',out)
    return summary


if __name__=='__main__':
    parser=argparse.ArgumentParser(description='Generate safe aggregate MixBhasha v5 validation graphs, CSVs and reports')
    parser.add_argument('--root',required=True,help='Drive MixBhasha_V5_seed42 experiment directory')
    parser.add_argument('--output',help='Default: ROOT/reports/public_validation')
    parser.add_argument('--tasks',nargs='*',choices=sorted(LABELS),help='Default: every trained task')
    parser.add_argument('--bootstrap',type=int,default=700)
    args=parser.parse_args()
    create_report(Path(args.root),Path(args.output) if args.output else Path(args.root)/'reports'/'public_validation',args.tasks,args.bootstrap)
