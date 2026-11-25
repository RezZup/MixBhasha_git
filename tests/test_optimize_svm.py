from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "tools"))
import mixbhasha_core as core
import optimize_svm as optimization


class SvmOptimizationTests(unittest.TestCase):
    def setUp(self):
        self.train = pd.DataFrame(
            {
                "text": [
                    f"training sample class{label} ordinary phrase index{index}"
                    for label in range(7)
                    for index in range(8)
                ],
                "label": [label for label in range(7) for _ in range(8)],
            }
        )

    def test_cv_receives_only_training_rows(self):
        captured = {}

        class FakeGridSearch:
            def __init__(self, estimator, param_grid, scoring, cv, **kwargs):
                captured["estimator"] = estimator
                captured["cv"] = cv
                captured["verbose"] = kwargs.get("verbose")
                self.param_grid = param_grid

            def fit(self, x, y):
                captured["x"] = np.asarray(x).tolist()
                captured["y"] = np.asarray(y).tolist()
                self.cv_results_ = {
                    "params": [
                        {
                            "estimator__svm__C": 0.5,
                            "estimator__svm__class_weight": None,
                        }
                    ],
                    "mean_test_score": np.asarray([0.6]),
                    "std_test_score": np.asarray([0.01]),
                    "split0_test_score": np.asarray([0.59]),
                    "split1_test_score": np.asarray([0.61]),
                }
                return self

        with patch.object(optimization, "GridSearchCV", FakeGridSearch):
            results = optimization.cross_validate_training(
                self.train,
                "word_1_2",
                c_values=(0.5,),
                class_weights=(None,),
                cv_folds=2,
            )

        self.assertEqual(captured["x"], self.train.text.tolist())
        self.assertEqual(captured["y"], self.train.label.tolist())
        self.assertEqual(captured["verbose"], 2)
        self.assertFalse(any("VALIDATION_ONLY" in text for text in captured["x"]))
        self.assertFalse(any("TEST_ONLY" in text for text in captured["x"]))
        self.assertEqual(captured["cv"].random_state, optimization.SEED)
        self.assertEqual(len(results), 1)

    def test_class_ids_reuse_core_label_mapping(self):
        self.assertEqual(optimization.CLASSES, core.LABELS["cyberbullying7"])
        for label_id, label_name in enumerate(core.LABELS["cyberbullying7"]):
            self.assertEqual(optimization.label_name(label_id), label_name)
        with self.assertRaises(ValueError):
            optimization.label_name(7)

    def test_tfidf_and_calibration_are_nested_inside_cv_estimator(self):
        estimator = optimization.make_calibrated_estimator(
            "word_char_baseline",
            c_value=1.0,
            class_weight="balanced",
        )
        self.assertIsInstance(estimator, CalibratedClassifierCV)
        self.assertIsInstance(estimator.estimator, Pipeline)
        self.assertIn("features", estimator.estimator.named_steps)
        self.assertIn("svm", estimator.estimator.named_steps)
        self.assertEqual(estimator.cv, optimization.CALIBRATION_FOLDS)
        vectorizers = estimator.estimator.named_steps["features"].transformer_list
        self.assertTrue(all(not hasattr(vectorizer, "vocabulary_") for _, vectorizer in vectorizers))

    def _create_experiment_root(self, root: Path) -> tuple[pd.DataFrame, pd.DataFrame, list[Path]]:
        data_dir = root / "data"
        result_dir = root / "results"
        model_dir = root / "models" / optimization.TASK / "tfidf_svm"
        data_dir.mkdir(parents=True)
        result_dir.mkdir()
        model_dir.mkdir(parents=True)
        validation = pd.DataFrame(
            {
                "text": [
                    f"validation sample class{label} ordinary phrase holdout{index}"
                    for label in range(7)
                    for index in range(3)
                ],
                "label": [label for label in range(7) for _ in range(3)],
            }
        )
        self.train.to_csv(data_dir / "cyberbullying7_train.csv", index=False)
        validation.to_csv(data_dir / "cyberbullying7_val.csv", index=False)
        (result_dir / "data_audit.json").write_text(
            json.dumps(
                {
                    optimization.TASK: {
                        "source": "synthetic regression fixture",
                        "split_sha256": "a" * 64,
                    }
                }
            ),
            encoding="utf-8",
        )

        baseline = core.feature_model("tfidf_svm")
        baseline.fit(self.train["text"], self.train["label"])
        baseline_probabilities = baseline.predict_proba(validation["text"])
        baseline_metrics, _, _, _ = optimization.evaluate_probabilities(
            validation["label"].to_numpy(dtype=int),
            baseline_probabilities,
        )
        model_path = model_dir / "model.joblib"
        validation_path = model_dir / "validation_probabilities.npz"
        metadata_path = model_dir / "validated.json"
        joblib.dump(baseline, model_path, compress=3)
        np.savez_compressed(
            validation_path,
            probs=baseline_probabilities,
            labels=validation["label"].to_numpy(dtype=int),
        )
        metadata_path.write_text(
            json.dumps(
                {
                    "meta": {
                        "task": optimization.TASK,
                        "model": "tfidf_svm",
                        "seed": optimization.SEED,
                        "split_sha256": "a" * 64,
                    },
                    "metrics": baseline_metrics,
                }
            ),
            encoding="utf-8",
        )
        # A deliberately malformed test file must never be opened by this experiment.
        (data_dir / "cyberbullying7_test.csv").write_text(
            "text,label\nTEST_ONLY_SECRET,not-an-integer\n",
            encoding="utf-8",
        )
        return validation, baseline_metrics, [model_path, validation_path, metadata_path]

    def test_run_preserves_baseline_and_reproduces_aligned_outputs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            validation, _, baseline_paths = self._create_experiment_root(root)
            signatures_before = {
                path.name: optimization._hash_file(path) for path in baseline_paths
            }
            kwargs = {
                "c_values": (0.5,),
                "class_weights": (None,),
                "feature_names": ("word_1_2",),
                "cv_folds": 2,
                "bootstrap_repeats": 100,
            }
            report = optimization.run_experiment(root, **kwargs)
            output_dir = root / optimization.DEFAULT_OUTPUT_RELATIVE
            first_cv = (output_dir / "cv_results.csv").read_bytes()
            first_report = (output_dir / "optimization_report.json").read_bytes()

            self.assertFalse(report["sample_counts"]["test_rows_loaded"])
            self.assertFalse(report["sample_counts"]["test_labels_inspected"])
            self.assertTrue(report["baseline_artifact_preservation"]["unchanged"])
            self.assertEqual(
                signatures_before,
                {path.name: optimization._hash_file(path) for path in baseline_paths},
            )
            predictions = pd.read_csv(output_dir / "validation_predictions.csv")
            self.assertEqual(
                predictions["validation_row_index"].tolist(),
                list(range(len(validation))),
            )
            self.assertEqual(
                predictions["true_label_id"].tolist(),
                validation["label"].tolist(),
            )
            with np.load(output_dir / "optimized_validation_predictions.npz") as packed:
                self.assertEqual(
                    packed["labels"].tolist(),
                    validation["label"].tolist(),
                )

            with patch.object(
                optimization,
                "cross_validate_training",
                side_effect=AssertionError("cached experiment should not repeat CV"),
            ):
                cached_report = optimization.run_experiment(root, **kwargs)
            self.assertEqual(
                cached_report["reproducibility"]["stored_project_split_sha256"],
                report["reproducibility"]["stored_project_split_sha256"],
            )

            optimization.run_experiment(root, overwrite=True, **kwargs)
            self.assertEqual(first_cv, (output_dir / "cv_results.csv").read_bytes())
            self.assertEqual(
                first_report,
                (output_dir / "optimization_report.json").read_bytes(),
            )


if __name__ == "__main__":
    unittest.main(verbosity=2)
