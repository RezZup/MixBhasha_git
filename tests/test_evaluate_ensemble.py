from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import mixbhasha_core as core
import evaluate_ensemble as ensemble


class EnsembleEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.labels = np.arange(len(core.LABELS["cyberbullying7"]), dtype=int)
        self.validation = pd.DataFrame(
            {
                "text": [f"validation row {index}" for index in self.labels],
                "label": self.labels,
            }
        )
        self.probabilities = np.eye(len(self.labels), dtype=float) * 0.9 + (
            np.ones((len(self.labels), len(self.labels))) * (0.1 / len(self.labels))
        )

    def test_split_fingerprints_must_match_audit_and_both_models(self):
        fingerprint = "a" * 64
        metadata = {
            "tfidf_svm": {"split_sha256": fingerprint.upper()},
            "hing_roberta_mixed": {"split_sha256": fingerprint},
        }
        self.assertEqual(
            ensemble.verify_split_fingerprints(fingerprint, metadata),
            fingerprint,
        )
        metadata["hing_roberta_mixed"]["split_sha256"] = "b" * 64
        with self.assertRaisesRegex(RuntimeError, "fingerprint mismatch"):
            ensemble.verify_split_fingerprints(fingerprint, metadata)

    def test_prediction_alignment_checks_labels_and_split_fingerprint(self):
        fingerprint = "c" * 64
        result = ensemble.verify_prediction_alignment(
            "tfidf_svm",
            self.probabilities,
            self.labels,
            self.validation,
            fingerprint,
            fingerprint,
        )
        self.assertTrue(result["saved_labels_match_validation_in_order"])
        self.assertTrue(result["model_split_fingerprint_matches_current_experiment"])

        with self.assertRaisesRegex(RuntimeError, "labels do not match"):
            ensemble.verify_prediction_alignment(
                "tfidf_svm",
                self.probabilities,
                self.labels[::-1],
                self.validation,
                fingerprint,
                fingerprint,
            )
        with self.assertRaisesRegex(RuntimeError, "split fingerprint"):
            ensemble.verify_prediction_alignment(
                "tfidf_svm",
                self.probabilities[::-1],
                self.labels,
                self.validation,
                fingerprint,
                "d" * 64,
            )

    def test_probability_shape_and_values_are_validated(self):
        with self.assertRaisesRegex(ValueError, "probability shape"):
            ensemble.validate_probabilities(
                self.probabilities[:, :2],
                len(self.labels),
                "fixture",
            )
        invalid = self.probabilities.copy()
        invalid[0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, "non-finite"):
            ensemble.validate_probabilities(invalid, len(self.labels), "fixture")

    def test_svm_and_transformer_class_order_must_match_core_mapping(self):
        class FakeSvm:
            classes_ = np.arange(len(ensemble.CLASSES))

        self.assertEqual(
            ensemble.verify_model_class_order(
                ensemble.SVM_NAME,
                ("classical", FakeSvm(), None),
            ),
            ensemble.CLASSES,
        )
        FakeSvm.classes_ = np.asarray([1, 0, 2, 3, 4, 5, 6])
        with self.assertRaisesRegex(ValueError, "class IDs"):
            ensemble.verify_model_class_order(
                ensemble.SVM_NAME,
                ("classical", FakeSvm(), None),
            )

        class FakeTransformer:
            config = type(
                "FakeConfig",
                (),
                {
                    "num_labels": len(ensemble.CLASSES),
                    "id2label": {
                        index: label for index, label in enumerate(ensemble.CLASSES)
                    },
                    "label2id": {
                        label: index for index, label in enumerate(ensemble.CLASSES)
                    },
                },
            )()

        self.assertEqual(
            ensemble.verify_model_class_order(
                ensemble.ROBERTA_NAME,
                ("transformer", FakeTransformer(), None),
            ),
            ensemble.CLASSES,
        )
        FakeTransformer.config.id2label[0], FakeTransformer.config.id2label[1] = (
            FakeTransformer.config.id2label[1],
            FakeTransformer.config.id2label[0],
        )
        with self.assertRaisesRegex(ValueError, "class order"):
            ensemble.verify_model_class_order(
                ensemble.ROBERTA_NAME,
                ("transformer", FakeTransformer(), None),
            )

    def test_fixed_ensembles_use_multiclass_argmax_and_svm_tie_break(self):
        svm = np.zeros((2, len(ensemble.CLASSES)), dtype=float)
        roberta = np.zeros_like(svm)
        svm[0, 0], svm[0, 1] = 0.7, 0.3
        roberta[0, 0], roberta[0, 1] = 0.3, 0.7
        svm[1, 1], svm[1, 2] = 0.7, 0.3
        roberta[1, 1], roberta[1, 2] = 0.3, 0.7

        probabilities, predictions, definitions = ensemble.build_candidate_predictions(
            svm,
            roberta,
        )
        self.assertEqual(
            predictions["ensemble_equal_50_50"].tolist(),
            [0, 1],
        )
        self.assertEqual(
            predictions["ensemble_svm_75_roberta_25"].tolist(),
            [0, 1],
        )
        self.assertEqual(
            predictions["ensemble_hard_vote_svm_tiebreak"].tolist(),
            [0, 1],
        )
        self.assertTrue(
            np.allclose(
                probabilities["ensemble_equal_50_50"],
                0.5 * svm + 0.5 * roberta,
            )
        )
        self.assertEqual(definitions["ensemble_svm_75_roberta_25"]["weights"]["tfidf_svm"], 0.75)
        self.assertEqual(
            set(predictions),
            set(ensemble.CANDIDATE_NAMES),
        )

    def test_agreement_and_per_class_recovery_are_counted_both_directions(self):
        labels = np.arange(len(ensemble.CLASSES), dtype=int)
        svm_predictions = labels.copy()
        roberta_predictions = labels.copy()
        svm_predictions[4] = 3
        roberta_predictions[1] = 0

        def probabilities_for(predicted):
            values = np.full((len(predicted), len(ensemble.CLASSES)), 0.01 / 6)
            values[np.arange(len(predicted)), predicted] = 0.99
            return values

        _, predictions, _ = ensemble.build_candidate_predictions(
            probabilities_for(svm_predictions),
            probabilities_for(roberta_predictions),
        )
        agreement = ensemble.build_model_agreement(labels, predictions)
        self.assertEqual(agreement["correctness_partition"]["both_correct"], 5)
        self.assertEqual(
            agreement["correctness_partition"]["svm_correct_hing_roberta_incorrect"],
            1,
        )
        self.assertEqual(
            agreement["correctness_partition"]["hing_roberta_correct_svm_incorrect"],
            1,
        )
        self.assertEqual(
            agreement["per_class_error_recovery_vs_svm"]["hing_roberta_mixed"][4][
                "errors_recovered"
            ],
            1,
        )
        self.assertEqual(
            agreement["per_class_svm_recovery_from_hing_roberta_errors"][1][
                "errors_recovered"
            ],
            1,
        )
        self.assertTrue(agreement["hard_vote_matches_original_svm_predictions"])

    def test_paired_bootstrap_is_reproducible_and_hard_vote_matches_svm(self):
        result, first = ensemble.paired_bootstrap_macro_f1(
            self.labels,
            self.labels,
            self.labels,
            repeats=100,
            seed=ensemble.SEED,
        )
        _, second = ensemble.paired_bootstrap_macro_f1(
            self.labels,
            self.labels,
            self.labels,
            repeats=100,
            seed=ensemble.SEED,
        )
        self.assertTrue(np.array_equal(first, second))
        self.assertEqual(result["confidence_interval_95_percentile"], [0.0, 0.0])
        self.assertTrue(result["interval_crosses_zero"])
        self.assertFalse(result["statistical_significance_claimed"])

    def test_metrics_and_confusion_matrix_use_all_seven_classes(self):
        metrics, per_class, confusion = ensemble.evaluate_predictions(
            self.labels,
            self.labels,
        )
        self.assertEqual(metrics["accuracy"], 1.0)
        self.assertEqual(metrics["macro_f1"], 1.0)
        self.assertEqual(per_class["label"].tolist(), ensemble.CLASSES)
        self.assertEqual(confusion.shape, (7, 7))
        self.assertTrue(np.array_equal(confusion, np.eye(7, dtype=int)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
