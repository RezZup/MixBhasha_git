# Troubleshooting

- **`ModuleNotFoundError: transformers/datasets`**: Run the notebook pip installation cell in the same runtime, restart the notebook kernel only if imports still reference old packages, and run cells in order.
- **CUDA out-of-memory**: Reduce transformer `batch` from 8 to 4 or 2, use gradient accumulation and lower `max_len` to 96. Do not change this mid-checkpoint; start a new experiment root when changing run metadata. Free GPU sessions are finite.
- **Hugging Face network/permission error**: Enable Kaggle Internet. MC-Hinglish1.0 public dataset should be accessed with `huggingface_hub`. HASOC organizer downloads require authorization and a key; do not bypass access controls.
- **Invalid MC label values**: The loader accepts only original seven categories or an explicit verified numeric-to-category mapping; it refuses to infer numeric meaning.
- **`ValueError: prepared dataset differs`**: You've changed dataset source or preprocessing after training. Start a new root rather than overwriting old checkpoints.
- **`Missing locked_validation_winners.json` in app**: Train models, then explicitly finalize one task. Point MIXBHASHA_ROOT to the output directory containing both `models/` and `results/`.
- **Large transformer classifier-head warning**: Normal when initializing new classification head from masked language model or mismatched 3-class head. Train the new head; do not report results before fine-tuning.
- **High v5 sentiment test F1**: Historical v4 published dev is not unseen. Interpret as exploratory, and evaluate independently labeled new examples.
- **Ensemble not accepted**: Validation improvement < 0.01 macro-F1. The best standalone candidate stays selected.
- **On Kaggle session restart**: Restore saved `/kaggle/working/MixBhasha_V5_seed42` from a previous run or attach as a Kaggle dataset. Simply using the same code does not restore checkpoints.
