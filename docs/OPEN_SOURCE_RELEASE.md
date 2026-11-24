# MixBhasha: publishing safely

**Status:** Training code and evaluation utilities are ready for public source release. Model weights, data-sharing rights and reproducible external benchmark performance are separate considerations.

## What is safe for a public GitHub repository?

- Source code, importable modules, app entry point, unexecuted notebook, tests, documentation, requirements, model/data cards, aggregate metrics and charts produced by `tools/generate_report.py`.
- No secrets, authentication tokens, private Drive links, private annotation spreadsheets, original posts, raw data CSVs, raw prediction CSVs, or saved model checkpoints. Confirm any example prompts are appropriate to share.
- If model checkpoints are subsequently published, verify *each* pretrained model's and dataset's license and any restrictions regarding derivative weights before distribution. Do not assume this project's MIT license grants rights over upstream datasets or models.

## Validation evidence currently reported by the user

The values below are **user-reported from Colab logs** and should be replaced by report-generator output from stored probability vectors. These are **validation scores**, not external test scores:

| Task | Leading model | Validation macro-F1 | Validation examples |
| --- | --- | ---: | ---: |
| `sentiment_raw` | HingRoBERTa-Mixed | 0.6684 | 1382 |
| `cyberbullying7` | TF-IDF + calibrated LinearSVC | 0.8127 | 879 |

### Cyberbullying targeted review (selected errors, not population prevalence)

For a **purposively selected** sample of 60 mistakes (20 each in abusive, mockery and offensive), the user recorded:

| Potential failure reason | Count |
| --- | ---: |
| Category confusion | 27 |
| Slang | 19 |
| Sarcasm | 7 |
| Context | 4 |
| Other | 2 |
| Spelling | 1 |

Reviewers considered the source label *questionable* in 27 cases, unclear in 21, and not questionable in 12. This does not establish a 45% overall labeling-error rate: selection was conditioned on model mistakes, and the review is not independent ground truth.

## Generate figures & validation tables from saved experiments

```bash
pip install -r requirements-report.txt
python tools/generate_report.py --root /path/to/MixBhasha_V5_seed42
```

Produces per-model confusion matrices (raw & row-normalized), PR and ROC curves, class-F1 charts, leaderboard charts, reliability diagnostics, model agreement heatmaps, paired bootstrap comparisons, aggregate manual-review counts (if a private CSV exists), HTML report and CSV tables. **Does not access or re-evaluate the held-out test.**

Trainer learning curves appear only if Hugging Face's saved `trainer_state.json` is present. The original v5 scratch Bi-LSTM does not save every epoch into a structured history: no curve is manufactured when logs are absent.

## Create a GitHub ZIP and optionally a private checkpoint backup

```bash
python tools/export_project.py \
    --source . \
    --root /path/to/MixBhasha_V5_seed42 \
    --destination /path/to/exports
```

The GitHub ZIP includes code + **aggregate** report only. A separate report ZIP holds all graphs/CSV tables. To back up selected trained weights **privately**, append:

```bash
--models sentiment_raw:hing_roberta_mixed cyberbullying7:tfidf_svm
```

Or add `--full-private-backup` for a much larger private experiment archive (large free-space required). Intermediate trainer checkpoints are omitted by default; append `--include-intermediate-checkpoints` only if exact interruption-resumption state is required. Full backup contains raw data, training data and private reviews; keep it private.

## GitHub push after extracting the public ZIP

```bash
git init
git add .
git commit -m "Release MixBhasha reproducible NLP toolkit"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/MixBhasha.git
git push -u origin main
```

Create the remote repository first. Only push code you own/have rights to redistribute. When publishing figures from experiments, record the dataset source, label scope, split fingerprint and that these are *validation* scores.

## What to improve next

1. Analyze **offensive / abusive / mockery** confusions; label taxonomy may itself be ambiguous. Do not auto-correct validation labels.
2. Tune LinearSVC on **training-only** cross-validation and re-evaluate validation once per prespecified study.
3. Test a separately calibrated (training-only) SVM + HingRoBERTa ensemble on validation only.
4. Lock model selection before the reserved cyberbullying test. Preserve previously exposed sentiment test results as exploratory historical results.
5. Create an independent, human-annotated Hinglish challenge benchmark with a documented, consistent taxonomy.
