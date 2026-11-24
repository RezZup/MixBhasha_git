#!/usr/bin/env python3
"""Export a safe GitHub source ZIP, a reports ZIP, and optional PRIVATE checkpoint backup.

Never upload raw social-media text, personal annotations, or massive model weights to GitHub by accident.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

SKIP_DIRS = {'.git', '__pycache__', '.ipynb_checkpoints', '.venv', 'venv', 'node_modules', '.pytest_cache', '.mypy_cache'}
SKIP_EXT = {'.pyc', '.pyo', '.pyd', '.key', '.pem', '.env'}


def _safe_code_files(src):
    for p in sorted(src.rglob('*')):
        if not p.is_file():continue
        rel=p.relative_to(src)
        if any(part in SKIP_DIRS for part in rel.parts):continue
        if any(part in {'data','models','private_analysis','logs','results','reports'} for part in rel.parts):continue
        if p.name in {'.env', '.env.local', 'credentials.json', 'token.json', '.git-credentials', 'RELEASE_NOTES.txt'}:continue
        if p.suffix.lower() in SKIP_EXT or p.suffix.lower() in {'.zip','.pt','.pth','.bin','.safetensors','.joblib','.npz'}:continue
        if p.stat().st_size > 10_000_000:continue
        yield p,rel


def _add_bytes(z, name, data):
    z.writestr(name,data)


def make_repo_zip(src, out, report_dir=None):
    src=Path(src).expanduser().resolve();out=Path(out).expanduser().resolve()
    if not (src/'src'/'mixbhasha_core.py').exists():raise FileNotFoundError(f'Expected v5 source tree at {src}')
    out.parent.mkdir(parents=True,exist_ok=True)
    n=0
    with zipfile.ZipFile(out,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for f,rel in _safe_code_files(src):
            name='MixBhasha/'+rel.as_posix()
            if f.suffix.lower()=='.ipynb':
                # Strip any execution outputs, including possible copies of raw social media rows.
                doc=json.loads(f.read_text(encoding='utf8'))
                for cell in doc.get('cells',[]):
                    if cell.get('cell_type')=='code':
                        cell['outputs']=[]
                        cell['execution_count']=None
                _add_bytes(z,name,json.dumps(doc,ensure_ascii=False,indent=1).encode('utf8'))
            else:
                z.write(f,name)
            n+=1
        if report_dir and Path(report_dir).exists():
            report_dir=Path(report_dir)
            for f in sorted(report_dir.rglob('*')):
                if not f.is_file():continue
                rel=f.relative_to(report_dir)
                # ONLY generated aggregate reporting artefacts. No raw inputs or annotation rows.
                if f.suffix.lower() not in {'.png','.csv','.json','.md','.html'}:continue
                if 'PRIVATE' in f.name.upper() or any(x in f.name for x in ('errors', 'challenge_predictions')):continue
                z.write(f,'MixBhasha/reports/public_validation/'+rel.as_posix())
                n+=1
        _add_bytes(z,'MixBhasha/RELEASE_NOTES.txt',(
            'Generated source-only GitHub distribution.\n'
            'No full datasets, model checkpoints, private review examples, credentials or raw predictions included.\n'
            'The validation reports are *not* held-out leaderboard claims.\n'
            'Run locally or in Colab to generate your own weights.\n'
            'Upstream dataset/model licenses are NOT replaced by this source-code license.\n'
        ).encode('utf8'))
    print(f'GITHUB-READY ZIP ({n} files): {out} ({out.stat().st_size/1024**2:.2f} MiB)')
    return out


def make_report_zip(report_dir, out):
    report_dir=Path(report_dir).expanduser().resolve();out=Path(out).expanduser().resolve()
    if not (report_dir/'REPORT.html').exists():raise FileNotFoundError('Run generate_report.py first')
    out.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED,compresslevel=6,allowZip64=True) as z:
        for f in report_dir.rglob('*'):
            if f.is_file():z.write(f,Path('MixBhasha_Validation_Report')/f.relative_to(report_dir))
    print(f'REPORT ZIP: {out} ({out.stat().st_size/1024**2:.2f} MiB)')
    return out


def _safe_big_zip(path, members, root, require_free_space=True):
    root=Path(root).resolve();path=Path(path).resolve()
    total=sum(f.stat().st_size for f in members)
    if require_free_space:
        available=shutil.disk_usage(path.parent).free
        # Model weights are already compressed, plan for ZIP roughly equal to source size.
        if available < total*1.05+50*1024*1024:
            raise OSError(f'Insufficient free space. Input {total/1024**3:.2f} GiB, available {available/1024**3:.2f} GiB. '
                          'Choose fewer models, another destination, or copy to a larger storage location.')
    print(f'Preparing PRIVATE archive of {len(members)} files; input {total/1024**3:.2f} GiB')
    with zipfile.ZipFile(path,'w',allowZip64=True) as z:
        for i,f in enumerate(members,1):
            # Store model files with no extra compression to save Colab time; ZIP supports >4GB.
            z.write(f,root.name+'/'+f.relative_to(root).as_posix(),compress_type=zipfile.ZIP_STORED)
            if i%50==0:print(f'  archived {i}/{len(members)} files')
    print(f'PRIVATE ARCHIVE (do NOT add to GitHub): {path} ({path.stat().st_size/1024**3:.2f} GiB)')
    return path


def make_selected_models_backup(root,out,models):
    root=Path(root).expanduser().resolve();out=Path(out).expanduser().resolve();out.parent.mkdir(parents=True,exist_ok=True)
    paths=[]
    for spec in models:
        if ':' not in spec:raise ValueError(f'Model spec must be task:model, got {spec}')
        task,name=spec.split(':',1)
        p=root/'models'/task/name
        if not p.is_dir():raise FileNotFoundError(f'Missing model {p}')
        paths.extend(f for f in p.rglob('*') if f.is_file() and 'checkpoints' not in f.relative_to(p).parts)
    for rel in ('results/locked_validation_winners.json','results/data_audit.json'):
        p=root/rel
        if p.exists():paths.append(p)
    return _safe_big_zip(out,sorted(set(paths)),root)


def make_full_private_backup(root,out,include_intermediate_checkpoints=False):
    root=Path(root).expanduser().resolve();out=Path(out).expanduser().resolve();out.parent.mkdir(parents=True,exist_ok=True)
    members=[]
    for f in root.rglob('*'):
        if not f.is_file():continue
        if f.resolve()==out:continue
        rel=f.relative_to(root)
        if any(x in SKIP_DIRS for x in rel.parts):continue
        if not include_intermediate_checkpoints and 'checkpoints' in rel.parts:continue
        if f.name.endswith('.partial'):continue
        if f.suffix.lower()=='.zip':continue # avoids nested zip runaway
        members.append(f)
    return _safe_big_zip(out,members,root)


if __name__=='__main__':
    p=argparse.ArgumentParser(description='Create public GitHub source/report ZIPs, optional large PRIVATE checkpoint backups')
    p.add_argument('--root',required=True,help='Existing MixBhasha_V5_seed42 experiment folder')
    p.add_argument('--source',required=True,help='Unzipped MixBhasha project source directory')
    p.add_argument('--destination',required=True,help='Output directory on Drive')
    p.add_argument('--models',nargs='*',default=[],help='Optional PRIVATE checkpoint selection e.g. sentiment_raw:hing_roberta_mixed cyberbullying7:tfidf_svm')
    p.add_argument('--full-private-backup',action='store_true',help='Private FULL archive of data/models/results (large). Never push to GitHub')
    p.add_argument('--include-intermediate-checkpoints',action='store_true',help='Very large, only for private full backup')
    args=p.parse_args()
    root=Path(args.root);dest=Path(args.destination);dest.mkdir(parents=True,exist_ok=True)
    reports=root/'reports'/'public_validation'
    make_repo_zip(args.source,dest/'MixBhasha_GitHub_Ready.zip',reports if reports.exists() else None)
    if (reports/'REPORT.html').exists():make_report_zip(reports,dest/'MixBhasha_All_Graphs_And_Metrics.zip')
    if args.models:make_selected_models_backup(root,dest/'MixBhasha_SELECTED_MODELS_PRIVATE.zip',args.models)
    if args.full_private_backup:
        make_full_private_backup(root,dest/'MixBhasha_FULL_BACKUP_PRIVATE.zip',args.include_intermediate_checkpoints)
