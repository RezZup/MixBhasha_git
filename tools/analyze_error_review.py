#!/usr/bin/env python3
"""Build public AGGREGATE cyberbullying error-audit reports from a PRIVATE review CSV.

Only exports counts and plots. Does not export source texts, names, notes,
individual predictions, or other row-level annotation fields.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

CLASSES = ["abusive", "mockery", "offensive"]
REASONS = ["category_confusion", "slang", "sarcasm", "context", "other", "spelling"]
FLAGS = ["yes", "unclear", "no"]
MODEL_FIELDS = {
    "HingRoBERTa": "hing_roberta_mixed_correct",
    "Bi-LSTM Attention": "bilstm_attention_correct",
}

def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    fig.savefig(path.with_suffix(".svg"), bbox_inches="tight")
    plt.close(fig)


def run(input_csv: Path, output_dir: Path) -> dict:
    input_csv, output_dir = Path(input_csv), Path(output_dir)
    df = pd.read_csv(input_csv)
    required = {"actual_label", "error_reason", "original_label_questionable", "tfidf_svm_pred", "tfidf_svm_correct", *MODEL_FIELDS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required CSV columns: {sorted(missing)}")
    if df.empty: raise ValueError("Input is empty")
    if df["actual_label"].isna().any(): raise ValueError("Missing original class")
    if not df["actual_label"].isin(CLASSES).all(): raise ValueError("Unexpected class; update CLASSES for your task")
    if not df["original_label_questionable"].isin(FLAGS).all(): raise ValueError("Unexpected annotation audit flag")
    if not df["error_reason"].isin(REASONS).all(): raise ValueError("Unexpected error reason")
    for col in ["tfidf_svm_correct", *MODEL_FIELDS.values()]:
        if df[col].dtype != bool:
            # Be strict: strings "False" have truth-value True in Python.
            normalized=df[col].astype(str).str.strip().str.lower()
            if not normalized.isin(["true", "false"]).all():
                raise ValueError(f"Bad boolean data in {col}")
            df[col]=normalized.eq("true")
    if df["tfidf_svm_correct"].any():
        raise ValueError("Expected a sample restricted to TF-IDF SVM misclassifications")

    output_dir.mkdir(parents=True, exist_ok=True)
    reason_table = pd.crosstab(df.error_reason, df.actual_label).reindex(index=REASONS, columns=CLASSES, fill_value=0)
    flag_table = pd.crosstab(df.original_label_questionable, df.actual_label).reindex(index=FLAGS, columns=CLASSES, fill_value=0)
    confusion = pd.crosstab(df.actual_label, df.tfidf_svm_pred).reindex(index=CLASSES, fill_value=0)
    reason_table.to_csv(output_dir/'error_reason_by_class.csv')
    flag_table.to_csv(output_dir/'label_flags_by_class.csv')
    confusion.to_csv(output_dir/'svm_error_predicted_labels.csv')

    n = len(df)
    rcorrect = df[MODEL_FIELDS['HingRoBERTa']]
    bcorrect = df[MODEL_FIELDS['Bi-LSTM Attention']]
    rescued_either = int((rcorrect | bcorrect).sum())
    both = int((rcorrect & bcorrect).sum())
    summary = {
        "scope": "Purposively sampled TF-IDF SVM validation misclassifications; not a random dataset sample",
        "n_reviewed": int(n),
        "per_original_class": {k: int(v) for k,v in df.actual_label.value_counts().sort_index().items()},
        "error_reason_counts": {k: int(v) for k,v in df.error_reason.value_counts().reindex(REASONS, fill_value=0).items()},
        "suspected_original_label_counts": {k: int(v) for k,v in df.original_label_questionable.value_counts().reindex(FLAGS, fill_value=0).items()},
        "other_models_correct_on_svm_errors": {
            "hing_roberta_mixed": int(rcorrect.sum()),
            "bilstm_attention": int(bcorrect.sum()),
            "both": both,
            "either": rescued_either,
            "neither": int(n-rescued_either),
        },
        "cautions": [
            "Suspected-label flags are not adjudicated truths or dataset-wide label-noise estimates.",
            "Evaluation examples are validation cases used for exploratory error diagnosis only.",
            "Do not train on/relabel these examples while continuing to claim comparable validation scores.",
            "Model-rescue counts do not imply a realizable ensemble gain.",
        ],
    }
    (output_dir/'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')

    fig, ax = plt.subplots(figsize=(10,5))
    reason_table.plot.barh(stacked=True, ax=ax)
    ax.set_title(f"Why SVM missed these cases (selected validation errors, n={n})")
    ax.set_xlabel("Reviewed errors")
    ax.set_ylabel("Assigned error reason")
    ax.invert_yaxis()
    ax.legend(title="Original label", loc='best')
    save(fig, output_dir/'error_reasons_by_class.png')

    fig, ax = plt.subplots(figsize=(9,4.6))
    flag_table.plot.barh(stacked=True, ax=ax)
    ax.set_title(f"Original label audit: suspected status (selected errors, n={n})")
    ax.set_xlabel("Reviewed errors")
    ax.set_ylabel("Questionable? (subjective review)")
    ax.invert_yaxis()
    save(fig, output_dir/'label_audit_by_class.png')

    fig, ax = plt.subplots(figsize=(11,4.8))
    matrix=confusion.to_numpy()
    im=ax.imshow(matrix, aspect='auto', cmap='Blues')
    ax.set_xticks(range(len(confusion.columns)), confusion.columns, rotation=40, ha='right')
    ax.set_yticks(range(len(confusion.index)), confusion.index)
    ax.set_xlabel("SVM predicted label")
    ax.set_ylabel("Original label")
    ax.set_title("SVM predicted categories on selected mistakes only (not a full confusion matrix)")
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j,i,int(matrix[i,j]),ha='center',va='center',color='white' if matrix[i,j]>matrix.max()/2 else 'black')
    fig.colorbar(im,ax=ax,shrink=.84,label="Count")
    save(fig, output_dir/'svm_error_confusions.png')

    fig, ax = plt.subplots(figsize=(9,4.6))
    quantities={
        "HingRoBERTa correct":int(rcorrect.sum()),
        "Bi-LSTM correct":int(bcorrect.sum()),
        "Both correct":both,
        "Either correct":rescued_either,
        "Neither correct":int(n-rescued_either),
    }
    bars=ax.barh(list(quantities),list(quantities.values()))
    ax.bar_label(bars,padding=4)
    ax.set_xlim(0, max(quantities.values())*1.2+1)
    ax.set_title("Alternative-model recovery on selected SVM errors")
    ax.set_xlabel("Cases (not generalizable to full validation)")
    ax.invert_yaxis()
    save(fig, output_dir/'alternative_model_recovery.png')

    fmt=lambda dic: "\n".join(f"| `{k}` | {v} | {v/n*100:.1f}% |" for k,v in dic.items())
    report=f"""# MixBhasha v5 — Cyberbullying error audit

**Scope:** {n} **selected TF-IDF SVM validation mistakes**, not a random selection from all validation posts. Original classes in this review are {', '.join(CLASSES)}. This is **exploratory qualitative review**, not a new held-out evaluation or annotation-quality prevalence estimate.

## What was marked

| Reviewer-assigned reason | Count | Share within selected mistakes |
|---|---:|---:|
{fmt(summary['error_reason_counts'])}

![Reasons by class](error_reasons_by_class.png)

## Original annotation audit (not verified relabeling)

| Subjective flag | Count | Share within selected mistakes |
|---|---:|---:|
{fmt(summary['suspected_original_label_counts'])}

![Annotation audit](label_audit_by_class.png)

The 'yes' and 'unclear' flags indicate a need for **independent annotation review**, not established errors. The percentage **must not** be extrapolated to the full dataset because sampling was conditioned on SVM mistakes from three difficult classes.

## Alternative-model agreement

On these {n} examples, HingRoBERTa matches the existing label on **{int(rcorrect.sum())}** and Bi-LSTM Attention on **{int(bcorrect.sum())}**. Both match on **{both}**, at least one on **{rescued_either}**, neither on **{int(n-rescued_either)}**. An oracle could pick the correct model in some instances, but **this does not establish that a practical ensemble will improve validation performance**.

![Model agreement](alternative_model_recovery.png)

![SVM predictions among its errors](svm_error_confusions.png)

## Interpretive hypotheses to test

- **Category boundaries:** Offensive, abusive and mockery labels overlap semantically. Build a written annotation rubric with targeted-attack, profanity-use, sarcasm and identity-attack examples.
- **Code-mixed slang:** Slang may express aggression, teasing or informal frustration depending on target and context. More lexical features alone may reinforce shortcuts.
- **Shortcut patterns:** Inspect cases where innocuous identity terms trigger age, religion or gender predictions; no rate can be estimated from this review alone.
- **Out-of-domain/noise:** Inspect unusual language, fragments, and context-free posts in the **training split** before making data-cleaning changes.

## Next steps without validation contamination

1. Freeze this validation split; preserve all original labels and the saved model outputs.
2. Write annotation rules; conduct blinded independent double review on a **random sample of validation records** for a more representative label-quality estimate, *without changing the benchmark scores*.
3. Audit the **training split** for near-duplicates and ambiguous labels. Any manual corrections must have provenance and versioned training-only CSVs.
4. Tune SVM via stratified cross-validation **only on training data**; retain one declared validation comparison. Evaluate hard-case recall and per-class F1, not only global macro-F1.
5. Compare fixed-weight or training-calibrated ensemble candidates; a validation-only apparent improvement requires later confirmation on a genuinely independent set.

**Data release:** Raw social posts, reviewer notes, personal identifiers, and the review CSV are omitted from this public report. Check source dataset terms before distributing any text or checkpoints.
"""
    (output_dir/'cyberbullying7_error_audit.md').write_text(report, encoding='utf-8')
    print(f"Created public aggregate report in {output_dir}")
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--input',required=True,help='PRIVATE manually completed review CSV')
    parser.add_argument('--output',required=True,help='PUBLIC output folder for aggregate charts and counts')
    args=parser.parse_args()
    run(Path(args.input),Path(args.output))
