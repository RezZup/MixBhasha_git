from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import audit_training_data as audit


class TrainingDataAuditTests(unittest.TestCase):
    def test_audit_uses_text_only_for_holdouts_and_keeps_raw_text_private(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            data_dir = root / "data"
            results_dir = root / "results"
            data_dir.mkdir()
            results_dir.mkdir()

            private_text = "PRIVATE_RAW_TRAINING_TEXT_X"
            train = pd.DataFrame(
                {
                    "text": [
                        "routine sample text class zero",
                        "age mention in a long enough example",
                        "gender mention in a long enough example",
                        "religion mention in a long enough example",
                        "lmao nooooo!!! that is teasing",
                        "abusive example with enough content",
                        "offensive example with enough content",
                        private_text,
                        private_text,
                    ],
                    "label": [0, 1, 2, 3, 4, 5, 6, 4, 6],
                }
            )
            train.to_csv(data_dir / "cyberbullying7_train.csv", index=False)
            pd.DataFrame({"text": [private_text], "label": [0]}).to_csv(
                data_dir / "cyberbullying7_val.csv", index=False
            )
            # The intentionally invalid test label proves this file's labels are not parsed.
            (data_dir / "cyberbullying7_test.csv").write_text(
                "text,label\n" + private_text + ",not-a-label\n",
                encoding="utf-8",
            )
            (results_dir / "data_audit.json").write_text(
                json.dumps(
                    {
                        "cyberbullying7": {
                            "source": "fixture",
                            "split_sha256": "a" * 64,
                        }
                    }
                ),
                encoding="utf-8",
            )

            summary = audit.run(root)

            aggregate_dir = root / "reports" / "training_audit"
            private_csv = root / "private_analysis" / "suspicious_training_examples.csv"
            self.assertFalse(summary["data_leakage"]["test_labels_inspected"])
            self.assertEqual(
                summary["duplicates"]["conflicting_training_labels"][0]["match_type"],
                "exact",
            )
            self.assertTrue((aggregate_dir / "training_data_audit.md").is_file())
            self.assertTrue((aggregate_dir / "class_distribution.png").is_file())
            self.assertTrue(private_csv.is_file())
            self.assertNotIn(
                private_text,
                (aggregate_dir / "training_data_audit.json").read_text(encoding="utf-8"),
            )
            self.assertIn(
                private_text,
                private_csv.read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
