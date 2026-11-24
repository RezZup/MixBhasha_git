# MixBhasha — Code-Mixed Sentiment, Cyberbullying & Dataset Audit Toolkit

> Includes MixBhasha v5 training engine plus **reproducible experiment reporting and safe GitHub/Drive export**. See [Open-source release guide](docs/OPEN_SOURCE_RELEASE.md).

For the complete figure generator run `python tools/generate_report.py --root /path/to/MixBhasha_V5_seed42`. It reads saved validation probability vectors and writes per-model confusion matrices, per-class metrics, ROC/PR, confidence diagnostics, bootstrap comparisons, leaderboard figures, Markdown and HTML reports. No retraining or unseen test evaluation is performed.

For a public release ZIP and independent graph ZIP run `python tools/export_project.py --source . --root /path/to/MixBhasha_V5_seed42 --destination /path/to/exports`. Private checkpoint exports are opt-in and **must not** be pushed to GitHub.


**A reproducible experiment framework for Hinglish sentiment and category-aware cyberbullying.** Python + PyTorch + Hugging Face Transformers + scikit-learn + Streamlit. Kaggle and Google Colab notebook included.

This release contains **source code and user-reported v5 aggregate validation measurements**, but **not trained weights, raw social-media records, or proprietary/private datasets**. The measured values below were obtained in the existing experiment `MixBhasha_V5_seed42` (seed 42); they were not independently reproduced by the source ZIP itself. Run the experiment notebook to reproduce them.

## Current v5 results (validation, seed 42)

**Seven-category cyberbullying** on the same **879 validation examples**:

| Model | Accuracy | Macro-F1 |
|---|---:|---:|
| TF-IDF + calibrated Linear SVM | 0.8203 | **0.8127** |
| HingRoBERTa-Mixed | 0.8089 | 0.7958 |
| TF-IDF word + char | 0.7873 | 0.7804 |
| Twitter-XLM-R | 0.7725 | 0.7599 |
| Bi-LSTM Attention | 0.7190 | 0.7118 |
| MuRIL | 0.6018 | 0.5390 |

**Raw-text sentiment** (`sentiment_raw`, separately defined source/split; 1,382 validation examples):

| Model | Macro-F1 |
|---|---:|
| HingRoBERTa-Mixed | **0.6684** |
| Twitter-XLM-R | 0.6420 |
| MuRIL | 0.6398 |
| TF-IDF word + char | 0.5956 |
| TF-IDF SVM | 0.5881 |

These are **validation scores**, not blind final test scores or published benchmark state of the art. Different task scores **must not be directly compared**. The original `sentiment_raw` holdout had been evaluated earlier during development (macro-F1 ~0.6641 for the selected HingRoBERTa), so it is **not a fresh blind evaluation** for later sentiment improvements.

**Error review (cyberbullying):** We manually audited **60 selected SVM validation mistakes** (20 each from offensive, abusive and mockery). Reviewer-assigned reasons: **27 category confusion**, **19 slang**, 7 sarcasm, 4 context, 2 other, 1 spelling. Original annotation flags: **27 suspected questionable**, 21 unclear, 12 no. These are **not confirmed annotation errors**, and the rates cannot be extrapolated beyond this deliberately difficult subset. HingRoBERTa matched the original label on 20/60, Bi-LSTM on 19/60; at least one model matched on 30/60. This does not imply an ensemble can automatically recover those cases. See the [public error audit, charts and safe aggregate data](docs/audits/cyberbullying7/cyberbullying7_error_audit.md). Reproduce it using `python tools/analyze_error_review.py --input /path/to/PRIVATE_review.csv --output docs/audits/cyberbullying7`. **Raw posts and reviewer notes are excluded from GitHub.**

### Reproduce and visualize results

Open [`notebooks/MixBhasha_Export_And_Analytics_Colab.ipynb`](notebooks/MixBhasha_Export_And_Analytics_Colab.ipynb), set `ROOT` to your existing saved experiment, and run all cells. This builds raw + normalized confusion matrices, per-class precision/recall/F1, macro comparison tables, ROC/PR, reliability diagnostics, model agreement, paired bootstrap intervals, manual review aggregate charts (when the private review file exists), and HTML/Markdown reports. No model training or test-label access is required.

The source ZIP intentionally excludes generated checkpoints; they require Google Drive/private object storage and are controlled by upstream model/data licensing. Raw model outputs and private review CSVs belong outside the public repository.

## Research question
Can code-mixed pretraining and category-aware data labeling produce reliable gains over word/character TF-IDF for Hinglish text, especially neutral sentiment, sarcasm, negation and mixed scripts, without creating test contamination or hiding uncertainty?

## Benchmarks and label scope

| Task key | Source | Labels | Interpretation |
|---|---|---|---|
| `sentiment` | Processed derivative of SentiMix 2020 | negative, neutral, positive | Same source family as v4, but v5 cleaning/holdout membership differs; NOT directly score-comparable or official SemEval test |
| `sentiment_raw` | BAKSA public team mirror of SentiMix with original-style raw text | negative, neutral, positive | Useful alternative preserving noisy script/spelling; provenance NOT official organizer gold, and test may overlap other SentiMix releases |
| `cyberbullying7` | MC-Hinglish1.0 | not_cyberbullying, age, gender, religion, mockery, abusive, offensive | Original seven mutually exclusive annotated classes |
| `cyberbullying_binary` | MC-Hinglish1.0 | not_cyberbullying vs cyberbullying_category | Derived binary where all six harmful categories are 1; not a universal abusive language detector |
| `offense_hasoc` | Authorized HASOC 2021 Subtask 2 code-mixed | not_hate_offensive vs hate_or_offensive | HOF/NOT includes contextual social-media exchanges; separate data, do not merge with MC |
| `hate` | PRISM | non_hate vs hate | Only hate-speech detection, NOT general offense |

### 2026 research priorities

1. **HingRoBERTa-Mixed** (`l3cube-pune/hing-roberta-mixed`): task-specific code-mix pretraining; Roman + Devanagari.
2. **Twitter-XLM-R** (`cardiffnlp/twitter-xlm-roberta-base`): multilingual Twitter-domain pretraining.
3. **Twitter-XLM-R sentiment-transfer** (`cardiffnlp/twitter-xlm-roberta-base-sentiment`): pretrained 3-way sentiment model; treat as a transfer-learning reference, do not confuse with training from a generic pretrained encoder.
4. **Word+character TF-IDF + calibrated LinearSVM**: inexpensive strong baseline alongside TF-IDF logistic regression.
5. **MuRIL / HingBERT / XLM-R**: additional ablations; train only if resources allow.
6. **Bi-LSTM Attention**: necessary course-aligned architecture comparison; do not presume it will outperform a transformer or TF-IDF.

For deep-learning comparisons, use the same data splits, class order, validation selection method and maximum sequence length. A separate experiment root is required for new seeds or changed hyperparameters. Train 1 transformer per free GPU session.

## First run — Colab or Kaggle

1. Open `notebooks/MixBhasha_V5_Train_Colab_Kaggle.ipynb` in a GPU-enabled runtime.
2. Enable internet for initial downloads. The notebook installs Python dependencies without upgrading the environment's CUDA PyTorch build.
3. Leave `SEED=42`, set `RUN_TASKS=['sentiment_raw']` for the alternative raw-text source (or `['sentiment']` for another processed-SentiMix experiment (not identical cleaned splits)), `RUN_MODELS=['tfidf_word_char','tfidf_svm','hing_roberta_mixed']`, `FINALIZE=False`.
4. Run all cells; validation scores and checkpoints are saved under `ROOT=.../MixBhasha_V5_seed42`.
5. In later sessions set `RUN_MODELS=['twitter_xlmr']`, `['muril']`, or other candidates, keeping the **same ROOT and seed**. Run `RUN_TASKS=['cyberbullying7']` and `['cyberbullying_binary']` separately.
6. Decide your intended comparison set **before** viewing holdout metrics. Optionally set `BUILD_BLEND=True` (validation only). Set `FINALIZE=True` only once when candidate selection is complete. Existing locks cannot be reselected.
7. Export `MixBhasha_V5_metrics.zip`; the heavier model checkpoint folder is *not* bundled in that metrics archive.

**Persistent checkpoints:** Colab mounts Google Drive by default. Kaggle `/kaggle/working` is ephemeral across new sessions: save notebook versions/outputs or restore an attached Kaggle dataset. Simply re-running a fresh Kaggle kernel without previous outputs does NOT resume.

## Architecture, data audit and strict integrity

- `strict_split_source` quarantines contradictory and duplicate labels **only inside source training data**, then removes training rows matching reserved test **text keys only**. Heldout labels are not referenced to decide which training data to keep.
- Source-provided holdout rows are preserved except invalid text or invalid label input, rather than selectively removing difficult/contradictory labels.
- Training and validation are stratified and disjoint after cleaning; test is not used for training, threshold tuning or model selection.
- SHA-256 dataset fingerprint is saved. Re-using mismatched trained outputs raises an error.
- All results are separated by task and per-model checkpoint. Validation candidate scores saved in `validated.json`; validation probability vectors are saved in `validation_probabilities.npz` for bootstrap and controlled blending.
- `propose_validation_ensemble` uses **only** validation predictions, and requires >=1 percentage-point improvement over the best single model to create an ensemble candidate. Do not treat small improvements as independently verified.
- For **binary** classification, threshold is tuned on validation macro-F1; always inspect positive precision, positive recall and PR-AUC. Raw probabilities are not calibrated for transformer/BiLSTM; the LinearSVM uses cross-validated sigmoid calibration on training data only.
- `test_locked` only evaluates after validation lock and refuses to replace existing test JSON.

## Historical v4 numbers (NOT v5 targets achieved)

| v4 measurement | Value |
|---|---:|
| Sentiment validation TF-IDF word+char macro-F1 | 0.62269 |
| Sentiment validation MuRIL macro-F1 | 0.62193 |
| Sentiment validation Bi-LSTM Attention macro-F1 | 0.58000 |
| Sentiment historical internal holdout TF-IDF macro-F1 | 0.70192 |
| Restricted MC binary validation Bi-LSTM Attention macro-F1 | 0.94027 |

**Important:** v4 sentiment published-dev internal holdout has already been inspected. Testing v5 on it is an **exploratory historical comparison**, not fresh blind generalization evidence. The old restricted offense task excludes four MC categories and is **not** directly comparable with v5 full MC binary or 7-class tasks.

## Challenge benchmark (human annotation)

Make an **independent** human-annotated CSV, columns `text,label`, with one exact task label name per text. Aim for 200–300 original examples, balanced across difficulty types (sarcasm, negation, Roman spellings, code-mixing, context/quotation, genuine neutral posts). Have 2 reviewers annotate independently, adjudicate disagreements and report agreement.

Set `EXTERNAL_CHALLENGE_CSV='/content/annotated_sentiment.csv'` after locking a task. The evaluator refuses records overlapping ANY of that experiment's training/validation/test splits, exact duplicate rows, missing labels or unexpected labels. Store raw social media texts privately; observe licensing and platform policies.

## Streamlit demo

```bash
pip install -r requirements-app.txt
export MIXBHASHA_ROOT=/path/to/MixBhasha_V5_seed42
streamlit run app.py
```

On Windows PowerShell:

```powershell
$env:MIXBHASHA_ROOT='D:\Models\MixBhasha_V5_seed42'
streamlit run app.py
```

The app loads **only locked models** and shows label scores, a relative uncertainty-based review flag and limitations. It cannot evaluate an untrained model. It is **not** an automated punitive moderation tool.

## Offline validation before cloud training

```bash
python -m unittest discover -s tests -p 'test_v5.py' -v
```

Tests use synthetic toy text and class labels and establish functionality, not accuracy. Full Hugging Face downloads, transformer training and Streamlit deployment require Colab/Kaggle execution.

## Sources / credit / access

- SentiMix 2020 task paper: https://aclanthology.org/2020.semeval-1.100/ ; processed derivative: https://github.com/singhnivedita/SemEval2020-Task9 ; raw-text team mirror (not official organizer gold): https://github.com/keshav22bansal/BAKSA_IITK
- MC-Hinglish1.0 authors/dataset: https://huggingface.co/datasets/sahinurlaskar/MC-Hinglish1.0 ; article: https://www.scielo.org.mx/scielo.php?pid=S1405-55462024000301091&script=sci_arttext
- HASOC 2021 (password protected; access through organizer): https://hasocfire.github.io/hasoc/2021/dataset.html
- L3Cube HingCorpus / HingRoBERTa: https://github.com/l3cube-pune/code-mixed-nlp ; paper: https://aclanthology.org/2022.wildre-1.2/
- Twitter-XLM-R: https://huggingface.co/cardiffnlp/twitter-xlm-roberta-base
- GLUECoS external benchmark (setup can be complex, its public test gold labels are not standard labeled CSVs): https://github.com/microsoft/GLUECoS

Dataset/code licenses differ: MC-Hinglish1.0 and L3Cube corpora have non-commercial/share-alike restrictions. Do not redistribute corpora, models or users' text without verifying the relevant licenses and authorization.

**Controlled comparisons:** Both v5 sentiment variants use different cleaning/split membership from v4 (and `sentiment_raw` also uses a different raw-text team mirror). A v5 score above 0.7019 is NOT by itself proof of improvement over v4. A new held-out human-annotated set or a common fixed benchmark is required.
