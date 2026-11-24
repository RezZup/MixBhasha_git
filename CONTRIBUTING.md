# Contributing to MixBhasha

Thank you for improving code-mixed NLP tools. Contributions should include a concise description, unit tests, and clear provenance for any datasets.

1. Propose a small issue describing the bug/feature and expected behavior.
2. Add or update tests before proposing fixes to the data loader or evaluation.
3. Never include raw user comments, private human annotations, credentials, unlicensed datasets or model weights in a public pull request.
4. Keep `train`, `val`, and `test` disjoint; do not tune on the held-out test set.
5. Report metrics by **task and dataset split**. Do not compare metrics from different label taxonomies as though equivalent.
6. Use reproducible seeds and note GPU/runtime/library versions when a result depends on them.
7. Follow respectful and safety-conscious moderation practices; present uncertainty and human review clearly.

Run tests with `python -m unittest discover -s tests -v` and reporting with `python tools/generate_report.py --help`.
