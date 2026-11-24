# MixBhasha v5 research protocol (preregister before training)

## Questions
1. Does HingRoBERTa-Mixed materially beat TF-IDF word+character and MuRIL on the same SentiMix derivative validation split?
2. Is neutral recall improved without making negative/positive F1 worse?
3. Does a social-media pretrained Twitter-XLM-R outperform generic multilingual encoders on Roman-script Hinglish?
4. Can full MC-Hinglish7 produce acceptable per-class F1 instead of benefiting from an artificially narrow 3-category subset?
5. Do measured gains persist on independently annotated hard cases and under spelling/script perturbations?

## Experimental protocol
- Fixed dataset provenance and label mappings in data_audit.json; training-only label conflict cleaning.
- Freeze train/val/test text keys and seed=42; repeat best candidates with seeds 43/44 using separate roots (same fixed split).
- All classification metrics: macro-F1 (primary), class precision/recall/F1, accuracy, balanced accuracy; binary additionally AP/PR-AUC and positive recall/precision.
- Train-only calibrated SVM; no calibration claims for raw neural probabilities. Validation-only threshold tuning for binary tasks; report threshold.
- Record tokenizer/truncation policy, epoch count, class weights and device. Keep batch size and effective gradient accumulation consistent when possible.
- Pairwise bootstrap 95% CI on validation macro-F1 delta. Then evaluate one frozen winner per task once on source holdout.
- SentiMix dev was exposed by v4; label it exploratory regardless of any v5 improvement. Use a fresh external annotation set for a proper new validation of transfer.
- Do not conflate sentiment, cyberbullying, generic offense and hate speech. Keep each as a separate task.

## Compute phases
A: Raw sentiment (`sentiment_raw`) and separately controlled processed sentiment (`sentiment`): tfidf_word_char, tfidf_svm, hing_roberta_mixed.
B: Sentiment: twitter_xlmr, muril. Optional XLM-R, HingBERT, Bi-LSTM Attention.
C: Cyberbullying7: tfidf_word_char, tfidf_svm, hing_roberta_mixed.
D: Cyberbullying binary: tfidf_svm, hing_roberta_mixed or twitter_xlmr.
E: Optional HASOC HOF/NOT, only with original authorized files; evaluate conversation context if available.
F: Validation-only two-model ensemble accepted only >=1 percentage point macro-F1 gain, and only before lock.
G: Independent challenge evaluation and stability across seeds after method choices have been frozen.

## What would count as a defensible improvement?
- Macro-F1 improvement over the strongest baseline with a positive paired interval or stable multi-seed evidence on *identical samples*.
- Neutral-class recall improvement with acceptable trade-offs (record all class F1 values).
- Improved behavior on independently labeled sarcasm/negation/script examples; exclude duplicates against all splits.
- Transparent computational cost, failure modes, bias risks, and review/abstention handling.

**Not sufficient:** adding more model names; publishing a bare 95% accuracy without class scope; comparing v5 MC seven-way F1 to v4 restricted binary 94%; test tuning after seeing held-out scores; calling previously exposed dev an unseen test.
