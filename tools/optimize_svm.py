#!/usr/bin/env python3
"""Run a leakage-controlled TF-IDF + SVM search for cyberbullying7."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import mixbhasha_core as core

TASK = "cyberbullying7"
SEED = 42
CLASSES = list(core.LABELS[TASK])
C_VALUES = (0.1, 0.5, 1.0, 2.0, 5.0, 10.0)
CLASS_WEIGHTS = (None, "balanced")
CALIBRATION_FOLDS = 3
DEFAULT_CV_FOLDS = 5
DEFAULT_BOOTSTRAP_REPEATS = 5000
DEFAULT_ROOT = Path("/content/drive/MyDrive/MixBhasha_V5_seed42")
DEFAULT_OUTPUT_RELATIVE = Path("experiments") / "cyberbullying7_svm_optimization"

FEATURE_VARIANTS = {
    "word_1_2": {
        "description": "Word TF-IDF, ngrams (1, 2); existing core word settings.",
    },
    "char_2_5": {
        "description": "Character TF-IDF, ngrams (2, 5); existing core char settings.",
    },
    "word_char_baseline": {
        "description": "Existing core word (1, 2) + character (3, 5) FeatureUnion.",
    },
    "char_3_5": {
        "description": "Character TF-IDF, ngrams (3, 5); existing core char settings.",
    },
}

FIGURE_NAMES = (
    "cv_macro_f1_by_c",
    "cv_macro_f1_by_feature",
    "validation_metrics_comparison",
    "original_svm_confusion_matrix",
    "optimized_svm_confusion_matrix",
    "per_class_f1_comparison",
    "per_class_recall_comparison",
    "cv_hyperparameter_heatmap",
    "paired_bootstrap_difference",
)
OUTPUT_FILENAMES = {
    "cv_results.csv",
    "best_parameters.json",
    "validation_comparison.csv",
    "per_class_comparison.csv",
    "error_transition_summary.json",
    "optimization_report.md",
    "optimization_report.json",
    "optimized_svm.joblib",
    "optimized_validation_predictions.npz",
    "validation_predictions.csv",
    *(f"{name}.{extension}" for name in FIGURE_NAMES for extension in ("png", "svg")),
}
REUSE_REQUIRED_OUTPUTS = {
    "cv_results.csv",
    "best_parameters.json",
    "validation_comparison.csv",
    "per_class_comparison.csv",
    "error_transition_summary.json",
    "optimization_report.md",
    "optimization_report.json",
    "optimized_svm.joblib",
    "optimized_validation_predictions.npz",
    "validation_predictions.csv",
}
PRIVATE_TRANSITIONS_NAME = "cyberbullying7_svm_optimization_error_transitions_PRIVATE.csv"
METRIC_NAMES = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "balanced_accuracy",
)


def _read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _add_hash_part(digest: Any, value: str) -> None:
    encoded = value.encode("utf-8")
    digest.update(len(encoded).to_bytes(8, "big"))
    digest.update(encoded)


def train_validation_fingerprint(train: pd.DataFrame, validation: pd.DataFrame) -> str:
    """Hash the current train/validation rows without reading test data or labels."""
    digest = hashlib.sha256()
    for split_name, frame in (("train", train), ("val", validation)):
        _add_hash_part(digest, split_name)
        for text, label in frame[["text", "label"]].itertuples(index=False, name=None):
            _add_hash_part(digest, str(text))
            _add_hash_part(digest, str(int(label)))
    return digest.hexdigest()


def label_name(label_id: int) -> str:
    if not 0 <= int(label_id) < len(CLASSES):
        raise ValueError(f"Invalid {TASK} label ID {label_id}; expected 0..{len(CLASSES) - 1}.")
    return CLASSES[int(label_id)]


def _validate_frame(frame: pd.DataFrame, split_name: str) -> pd.DataFrame:
    if not {"text", "label"}.issubset(frame.columns):
        raise ValueError(f"Prepared {split_name} split must contain 'text' and 'label' columns.")
    if frame.empty:
        raise ValueError(f"Prepared {split_name} split is empty.")
    if frame["text"].isna().any() or frame["label"].isna().any():
        raise ValueError(f"Prepared {split_name} split contains missing text or labels.")
    checked = frame[["text", "label"]].copy()
    checked["text"] = checked["text"].astype(str)
    checked["label"] = pd.to_numeric(checked["label"], errors="raise").astype(int)
    invalid = sorted(set(checked["label"]) - set(range(len(CLASSES))))
    if invalid:
        raise ValueError(
            f"Unexpected {split_name} label IDs {invalid}; use the existing core mapping "
            f"{CLASSES!r}."
        )
    missing = sorted(set(range(len(CLASSES))) - set(checked["label"]))
    if missing:
        raise ValueError(
            f"{split_name} split is missing classes "
            f"{[CLASSES[index] for index in missing]}; refusing incomplete seven-class metrics."
        )
    return checked.reset_index(drop=True)


def load_experiment_data(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, str, str]:
    """Load only the existing train and validation CSVs; never read the test split."""
    root = Path(root)
    for split in ("train", "val"):
        path = root / "data" / f"{TASK}_{split}.csv"
        if not path.is_file():
            raise FileNotFoundError(
                f"Missing prepared {split} split: {path}. Run the existing V5 "
                "cyberbullying7 preparation once; this experiment does not download "
                "or regenerate data."
            )
    loaded = core.load_prepared(root, [TASK], splits=("train", "val"))[TASK]
    train = _validate_frame(loaded["train"], "training")
    validation = _validate_frame(loaded["val"], "validation")

    audit_path = root / "results" / "data_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Missing {audit_path}; the V5 data-preparation cell must save the "
            "existing split fingerprint first."
        )
    audit = _read_json(audit_path)
    task_audit = audit.get(TASK)
    if not isinstance(task_audit, dict):
        raise ValueError(f"{audit_path} has no {TASK!r} entry.")
    split_fingerprint = task_audit.get("split_sha256")
    if not isinstance(split_fingerprint, str) or len(split_fingerprint) != 64:
        raise ValueError(f"{audit_path} does not contain a valid {TASK} split_sha256.")
    if any(character not in "0123456789abcdefABCDEF" for character in split_fingerprint):
        raise ValueError(f"{audit_path} contains a malformed {TASK} split_sha256.")
    return train, validation, split_fingerprint, str(task_audit.get("source", "not recorded"))


def make_feature_transformer(feature_variant: str):
    if feature_variant not in FEATURE_VARIANTS:
        raise ValueError(f"Unknown feature variant {feature_variant!r}.")
    if feature_variant == "word_1_2":
        return core.feature_model("tfidf_word").named_steps["vectorizer"]
    if feature_variant == "char_2_5":
        transformer = core.feature_model("tfidf_char").named_steps["vectorizer"]
        transformer.set_params(ngram_range=(2, 5))
        return transformer
    if feature_variant == "word_char_baseline":
        return core.feature_model("tfidf_svm").named_steps["vectorizer"]
    return core.feature_model("tfidf_char").named_steps["vectorizer"]


def make_calibrated_estimator(
    feature_variant: str,
    c_value: float = 1.0,
    class_weight: str | None = "balanced",
    calibration_folds: int = CALIBRATION_FOLDS,
) -> CalibratedClassifierCV:
    """Put TF-IDF inside each calibration fold, itself nested within outer CV."""
    feature_and_svm = Pipeline(
        [
            ("features", make_feature_transformer(feature_variant)),
            (
                "svm",
                LinearSVC(
                    C=float(c_value),
                    class_weight=class_weight,
                    random_state=SEED,
                    max_iter=5000,
                ),
            ),
        ]
    )
    return CalibratedClassifierCV(
        estimator=feature_and_svm,
        method="sigmoid",
        cv=calibration_folds,
        ensemble=True,
    )


def cross_validate_training(
    train: pd.DataFrame,
    feature_variant: str,
    c_values: tuple[float, ...] = C_VALUES,
    class_weights: tuple[str | None, ...] = CLASS_WEIGHTS,
    cv_folds: int = DEFAULT_CV_FOLDS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Search one feature family using only the supplied training rows."""
    checked = _validate_frame(train, "training")
    if cv_folds < 2:
        raise ValueError("Stratified CV requires at least two folds.")
    if not c_values or any(float(value) <= 0 for value in c_values):
        raise ValueError("Every SVM C value must be positive.")
    if not class_weights or any(value not in (None, "balanced") for value in class_weights):
        raise ValueError("class_weight values must be None or 'balanced'.")
    class_counts = checked["label"].value_counts()
    minimum_required = max(cv_folds, CALIBRATION_FOLDS + 1)
    if int(class_counts.min()) < minimum_required:
        raise ValueError(
            f"Each class needs at least {minimum_required} training examples for "
            f"{cv_folds}-fold outer CV and {CALIBRATION_FOLDS}-fold calibration."
        )

    estimator = make_calibrated_estimator(feature_variant)
    search = GridSearchCV(
        estimator=estimator,
        param_grid={
            "estimator__svm__C": [float(value) for value in c_values],
            "estimator__svm__class_weight": list(class_weights),
        },
        scoring="f1_macro",
        cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed),
        refit=False,
        n_jobs=1,
        verbose=2,
        return_train_score=False,
        error_score="raise",
    )
    x_train = checked["text"].to_numpy(dtype=str)
    y_train = checked["label"].to_numpy(dtype=int)
    search.fit(x_train, y_train)

    results = search.cv_results_
    rows = []
    for index, parameters in enumerate(results["params"]):
        row = {
            "feature_variant": feature_variant,
            "C": float(parameters["estimator__svm__C"]),
            "class_weight": (
                "None"
                if parameters["estimator__svm__class_weight"] is None
                else parameters["estimator__svm__class_weight"]
            ),
            "mean_test_macro_f1": float(results["mean_test_score"][index]),
            "std_test_macro_f1": float(results["std_test_score"][index]),
        }
        for fold in range(cv_folds):
            row[f"fold_{fold + 1}_macro_f1"] = float(
                results[f"split{fold}_test_score"][index]
            )
        rows.append(row)
    return pd.DataFrame(rows)


def _class_weight_order(value: str | None) -> int:
    return 0 if _class_weight_value(value) is None else 1


def _class_weight_value(value: str | None) -> str | None:
    if value is None or pd.isna(value) or value == "None":
        return None
    return str(value)


def rank_cv_results(results: pd.DataFrame, feature_names: tuple[str, ...]) -> pd.DataFrame:
    ranked = results.copy()
    feature_order = {name: index for index, name in enumerate(feature_names)}
    ranked["_weight_order"] = ranked["class_weight"].map(_class_weight_order)
    ranked["_feature_order"] = ranked["feature_variant"].map(feature_order)
    ranked = ranked.sort_values(
        [
            "mean_test_macro_f1",
            "std_test_macro_f1",
            "_feature_order",
            "C",
            "_weight_order",
        ],
        ascending=[False, True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked["global_cv_rank"] = np.arange(1, len(ranked) + 1, dtype=int)
    ranked["feature_cv_rank"] = (
        ranked.groupby("feature_variant")["mean_test_macro_f1"]
        .rank(method="min", ascending=False)
        .astype(int)
    )
    return ranked.drop(columns=["_weight_order", "_feature_order"])


def _validate_probabilities(probabilities: np.ndarray, expected_rows: int) -> np.ndarray:
    probabilities = np.asarray(probabilities, dtype=float)
    expected_shape = (expected_rows, len(CLASSES))
    if probabilities.shape != expected_shape:
        raise ValueError(
            f"Expected probability shape {expected_shape}; received {probabilities.shape}."
        )
    if not np.isfinite(probabilities).all() or np.any(probabilities < -1e-7):
        raise ValueError("Predicted probabilities contain non-finite or negative values.")
    if not np.allclose(probabilities.sum(axis=1), 1.0, rtol=0.0, atol=2e-3):
        raise ValueError("Predicted probability rows do not sum to one.")
    return probabilities


def predict_probabilities(estimator, texts: pd.Series | np.ndarray) -> np.ndarray:
    if not hasattr(estimator, "predict_proba"):
        raise TypeError("The calibrated SVM artifact must implement predict_proba().")
    classes = np.asarray(getattr(estimator, "classes_", []), dtype=int)
    if not np.array_equal(classes, np.arange(len(CLASSES))):
        raise ValueError(
            f"Estimator class order {classes.tolist()} does not match core.LABELS[{TASK!r}]."
        )
    return _validate_probabilities(
        estimator.predict_proba(np.asarray(texts, dtype=str)),
        len(texts),
    )


def evaluate_probabilities(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame, np.ndarray, np.ndarray]:
    y_true = np.asarray(labels, dtype=int)
    probs = _validate_probabilities(probabilities, len(y_true))
    predictions = probs.argmax(axis=1)
    label_ids = np.arange(len(CLASSES))
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        predictions,
        labels=label_ids,
        zero_division=0,
    )
    metrics = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "macro_f1": float(f1_score(y_true, predictions, labels=label_ids, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, predictions, labels=label_ids, average="weighted", zero_division=0)),
        "macro_precision": float(precision_score(y_true, predictions, labels=label_ids, average="macro", zero_division=0)),
        "macro_recall": float(recall_score(y_true, predictions, labels=label_ids, average="macro", zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predictions)),
    }
    per_class = pd.DataFrame(
        {
            "label_id": label_ids,
            "label": CLASSES,
            "precision": precision.astype(float),
            "recall": recall.astype(float),
            "f1": f1.astype(float),
            "support": support.astype(int),
        }
    )
    return metrics, per_class, confusion_matrix(y_true, predictions, labels=label_ids), predictions


def _hash_file(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def _baseline_artifact_paths(root: Path) -> list[Path]:
    model_dir = root / "models" / TASK / "tfidf_svm"
    paths = [
        model_dir / "model.joblib",
        model_dir / "validated.json",
        model_dir / "validation_probabilities.npz",
    ]
    missing = [str(path) for path in paths[:2] if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The original tfidf_svm baseline is incomplete; required read-only "
            f"artifacts are missing: {missing}"
        )
    return [path for path in paths if path.is_file()]


def load_and_verify_baseline(
    root: Path,
    validation: pd.DataFrame,
    expected_fingerprint: str,
) -> tuple[Any, np.ndarray, dict, list[Path], dict]:
    model_dir = root / "models" / TASK / "tfidf_svm"
    artifact_paths = _baseline_artifact_paths(root)
    signatures_before = {path.name: _hash_file(path) for path in artifact_paths}
    record = _read_json(model_dir / "validated.json")
    metadata = record.get("meta")
    if not isinstance(metadata, dict):
        raise ValueError(f"{model_dir / 'validated.json'} has no meta object.")
    recorded_metrics = record.get("metrics")
    if not isinstance(recorded_metrics, dict):
        raise ValueError(f"{model_dir / 'validated.json'} has no metrics object.")
    if metadata.get("task") != TASK or metadata.get("model") != "tfidf_svm":
        raise ValueError("Saved baseline metadata is not the cyberbullying7 tfidf_svm model.")
    if metadata.get("seed") != SEED:
        raise ValueError(f"Baseline seed must be {SEED}; found {metadata.get('seed')!r}.")
    if metadata.get("split_sha256") != expected_fingerprint:
        raise RuntimeError(
            "Baseline and saved data-audit split fingerprints differ; refusing to compare "
            "validation results."
        )

    model = joblib.load(model_dir / "model.joblib")
    current_probabilities = predict_probabilities(model, validation["text"])
    current_labels = validation["label"].to_numpy(dtype=int)
    prediction_path = model_dir / "validation_probabilities.npz"
    predictions_verified = False
    if prediction_path.is_file():
        with np.load(prediction_path, allow_pickle=False) as archive:
            if not {"probs", "labels"}.issubset(archive.files):
                raise ValueError(f"{prediction_path} must contain probs and labels arrays.")
            saved_probabilities = _validate_probabilities(
                np.asarray(archive["probs"], dtype=float),
                len(validation),
            )
            saved_labels = np.asarray(archive["labels"], dtype=int)
        if not np.array_equal(saved_labels, current_labels):
            raise RuntimeError(
                "Saved baseline validation labels differ in value or order from the "
                "current prepared validation split."
            )
        if not np.allclose(current_probabilities, saved_probabilities, rtol=1e-6, atol=1e-7):
            raise RuntimeError(
                "Loaded baseline predictions do not match saved validation probabilities; "
                "refusing an unverified comparison."
            )
        current_probabilities = saved_probabilities
        predictions_verified = True

    return (
        model,
        current_probabilities,
        {
            "metadata": metadata,
            "recorded_metrics": recorded_metrics,
            "saved_validation_predictions_verified": predictions_verified,
            "saved_validation_row_order_verified": predictions_verified,
            "fingerprint_verified_against_data_audit": True,
            "fingerprint_recomputed": False,
            "artifact_signatures_before": signatures_before,
        },
        artifact_paths,
        signatures_before,
    )


def _check_output_directory(output_dir: Path, overwrite: bool) -> None:
    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=True)
        return
    existing = {path.name for path in output_dir.iterdir()}
    unexpected = sorted(existing - OUTPUT_FILENAMES)
    if unexpected:
        raise FileExistsError(
            f"{output_dir} contains unrecognized files {unexpected}; refusing to replace them."
        )
    if existing and not overwrite:
        raise FileExistsError(
            f"{output_dir} already contains optimization output. Use --overwrite only "
            "if replacing this exact experiment is intentional."
        )


def _reuse_completed_experiment(
    root: Path,
    output_dir: Path,
    split_fingerprint: str,
) -> dict | None:
    report_path = output_dir / "optimization_report.json"
    transitions_path = root / "private_analysis" / PRIVATE_TRANSITIONS_NAME
    if not output_dir.exists() or not any(output_dir.iterdir()):
        return None
    missing = sorted(
        name for name in REUSE_REQUIRED_OUTPUTS if not (output_dir / name).is_file()
    )
    if missing or not transitions_path.is_file():
        raise FileExistsError(
            f"{output_dir} contains an incomplete prior run (missing outputs: {missing}; "
            f"private transitions present={transitions_path.is_file()}). "
            "Use --overwrite only if you intentionally want to replace it."
        )
    report = _read_json(report_path)
    recorded_fingerprint = report.get("reproducibility", {}).get(
        "stored_project_split_sha256"
    )
    if recorded_fingerprint != split_fingerprint:
        raise RuntimeError(
            "Existing optimization outputs use a different split fingerprint. "
            "Refusing to reuse them; use the matching experiment root."
        )
    recorded_baselines = {
        Path(item["path"]).name: item["sha256"]
        for item in report.get("baseline_artifact_preservation", {}).get("artifacts", [])
        if isinstance(item, dict) and {"path", "sha256"}.issubset(item)
    }
    current_baselines = {
        path.name: _hash_file(path)["sha256"] for path in _baseline_artifact_paths(root)
    }
    if recorded_baselines != current_baselines:
        raise RuntimeError(
            "Saved baseline artifacts differ from those used by the existing optimization "
            "report. Refusing to reuse or compare stale outputs."
        )
    print(
        f"REUSE completed SVM optimization from {output_dir}; no CV work repeated. "
        "Set overwrite=True only to intentionally rerun.",
        flush=True,
    )
    return report


def _paired_bootstrap(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    optimized_predictions: np.ndarray,
    repeats: int,
    seed: int = SEED,
) -> tuple[dict, np.ndarray]:
    if repeats < 100:
        raise ValueError("Use at least 100 paired bootstrap repeats.")
    y = np.asarray(labels, dtype=int)
    before = np.asarray(baseline_predictions, dtype=int)
    after = np.asarray(optimized_predictions, dtype=int)
    if not (len(y) == len(before) == len(after)):
        raise ValueError("Paired bootstrap inputs have different validation row counts.")
    label_ids = np.arange(len(CLASSES))
    point_delta = float(
        f1_score(y, after, labels=label_ids, average="macro", zero_division=0)
        - f1_score(y, before, labels=label_ids, average="macro", zero_division=0)
    )
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=np.float64)
    for repeat in range(repeats):
        indices = rng.integers(0, len(y), size=len(y))
        differences[repeat] = (
            f1_score(y[indices], after[indices], labels=label_ids, average="macro", zero_division=0)
            - f1_score(y[indices], before[indices], labels=label_ids, average="macro", zero_division=0)
        )
    low, high = (float(value) for value in np.quantile(differences, [0.025, 0.975]))
    crosses_zero = low <= 0.0 <= high
    return (
        {
            "delta_optimized_minus_baseline_macro_f1": point_delta,
            "confidence_interval_95_percentile": [low, high],
            "interval_crosses_zero": crosses_zero,
            "interpretation": (
                "Exploratory paired interval includes zero; do not claim a clear "
                "validation improvement."
                if crosses_zero
                else "Exploratory paired interval excludes zero; this is not a test-set "
                "or confirmatory significance claim."
            ),
            "validation_rows": int(len(y)),
            "bootstrap_repeats": int(repeats),
            "seed": int(seed),
            "resampling": "paired row bootstrap; same sampled indices for both models",
        },
        differences,
    )


def _error_transitions(
    validation: pd.DataFrame,
    baseline_predictions: np.ndarray,
    optimized_predictions: np.ndarray,
) -> tuple[dict, pd.DataFrame]:
    y = validation["label"].to_numpy(dtype=int)
    before = np.asarray(baseline_predictions, dtype=int)
    after = np.asarray(optimized_predictions, dtype=int)
    before_correct = before == y
    after_correct = after == y
    corrected = np.flatnonzero(~before_correct & after_correct)
    newly_incorrect = np.flatnonzero(before_correct & ~after_correct)
    unchanged_errors = np.flatnonzero(~before_correct & ~after_correct)
    unchanged_correct = np.flatnonzero(before_correct & after_correct)
    corrected_set = set(corrected.tolist())
    newly_incorrect_set = set(newly_incorrect.tolist())
    unchanged_error_set = set(unchanged_errors.tolist())
    by_class = {}
    for label_id, label in enumerate(CLASSES):
        class_rows = y == label_id
        by_class[label] = {
            "validation_support": int(class_rows.sum()),
            "baseline_errors": int(np.sum(class_rows & ~before_correct)),
            "optimized_errors": int(np.sum(class_rows & ~after_correct)),
            "corrected_by_optimization": int(np.sum(class_rows & ~before_correct & after_correct)),
            "new_errors_from_optimization": int(np.sum(class_rows & before_correct & ~after_correct)),
        }
    summary = {
        "validation_rows": int(len(y)),
        "baseline_correct": int(before_correct.sum()),
        "optimized_correct": int(after_correct.sum()),
        "corrected_cases": int(len(corrected)),
        "newly_introduced_errors": int(len(newly_incorrect)),
        "errors_remaining": int(len(unchanged_errors)),
        "correct_cases_unchanged": int(len(unchanged_correct)),
        "by_true_class": by_class,
        "corrected_validation_row_indices": corrected.tolist(),
        "newly_incorrect_validation_row_indices": newly_incorrect.tolist(),
        "test_split_read": False,
    }
    rows = []
    for index in range(len(y)):
        rows.append(
            {
                "validation_row_index": index,
                "true_label_id": int(y[index]),
                "true_label": label_name(y[index]),
                "baseline_prediction_id": int(before[index]),
                "baseline_prediction": label_name(before[index]),
                "optimized_prediction_id": int(after[index]),
                "optimized_prediction": label_name(after[index]),
                "transition": (
                    "corrected"
                    if index in corrected_set
                    else "new_error"
                    if index in newly_incorrect_set
                    else "unchanged_error"
                    if index in unchanged_error_set
                    else "unchanged_correct"
                ),
                "text": validation.iloc[index]["text"],
            }
        )
    return summary, pd.DataFrame(rows)


def _metric_row(
    name: str,
    role: str,
    feature_variant: str,
    c_value: float | None,
    class_weight: str | None,
    cv_score: float | None,
    metrics: dict[str, float | int],
    baseline_metrics: dict[str, float | int],
) -> dict:
    row = {
        "candidate": name,
        "role": role,
        "feature_variant": feature_variant,
        "C": c_value,
        "class_weight": "None" if class_weight is None else class_weight,
        "cv_mean_macro_f1": cv_score,
    }
    for metric in METRIC_NAMES:
        row[f"validation_{metric}"] = metrics[metric]
        row[f"delta_{metric}_vs_baseline"] = (
            float(metrics[metric]) - float(baseline_metrics[metric])
        )
    return row


def _plot_save(fig: plt.Figure, output_dir: Path, name: str) -> None:
    fig.tight_layout()
    fig.savefig(output_dir / f"{name}.png", dpi=180, bbox_inches="tight", facecolor="white")
    fig.savefig(output_dir / f"{name}.svg", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _plot_confusion(
    matrix: np.ndarray,
    title: str,
    output_dir: Path,
    filename: str,
    vmax: int,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 7))
    image = ax.imshow(matrix, cmap="Blues", vmin=0, vmax=max(vmax, 1))
    ax.set_xticks(np.arange(len(CLASSES)), CLASSES, rotation=40, ha="right")
    ax.set_yticks(np.arange(len(CLASSES)), CLASSES)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    for row in range(len(CLASSES)):
        for column in range(len(CLASSES)):
            value = int(matrix[row, column])
            ax.text(
                column,
                row,
                str(value),
                ha="center",
                va="center",
                color="white" if value > vmax / 2 else "black",
                fontsize=8,
            )
    fig.colorbar(image, ax=ax, shrink=0.82, label="Validation samples")
    _plot_save(fig, output_dir, filename)


def _make_figures(
    output_dir: Path,
    cv_results: pd.DataFrame,
    validation_rows: pd.DataFrame,
    per_class: pd.DataFrame,
    baseline_confusion: np.ndarray,
    optimized_confusion: np.ndarray,
    bootstrap_differences: np.ndarray,
    confidence_interval: list[float],
) -> None:
    feature_names = [
        name for name in FEATURE_VARIANTS
        if name in set(cv_results["feature_variant"])
    ]
    c_values = sorted(float(value) for value in cv_results["C"].unique())
    fig, ax = plt.subplots(figsize=(9, 5))
    for feature in feature_names:
        subset = cv_results[cv_results["feature_variant"] == feature]
        grouped = subset.groupby("C")["mean_test_macro_f1"].mean().sort_index()
        ax.plot(grouped.index, grouped.values, marker="o", label=feature)
    ax.set_xscale("log")
    ax.set_xticks(c_values, [str(value) for value in c_values])
    ax.set_xlabel("LinearSVC C (log scale)")
    ax.set_ylabel("Mean five-fold CV macro-F1 (averaged over class weights)")
    ax.set_title("Cross-validation macro-F1 by C")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    _plot_save(fig, output_dir, "cv_macro_f1_by_c")

    feature_summary = (
        cv_results.groupby("feature_variant")["mean_test_macro_f1"].max()
        .reindex(feature_names)
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(feature_summary.index, feature_summary.values, color="#4477aa")
    ax.set_xlabel("Best five-fold CV macro-F1")
    ax.set_title("Best CV macro-F1 by feature configuration")
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    _plot_save(fig, output_dir, "cv_macro_f1_by_feature")

    chosen = validation_rows[validation_rows["role"].isin(["original_baseline", "global_cv_best"])]
    metric_columns = {
        "Accuracy": "validation_accuracy",
        "Macro-F1": "validation_macro_f1",
        "Weighted F1": "validation_weighted_f1",
        "Macro precision": "validation_macro_precision",
        "Macro recall": "validation_macro_recall",
        "Balanced accuracy": "validation_balanced_accuracy",
    }
    x = np.arange(len(metric_columns))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    baseline_row = chosen[chosen["role"] == "original_baseline"].iloc[0]
    optimized_row = chosen[chosen["role"] == "global_cv_best"].iloc[0]
    ax.bar(
        x - width / 2,
        [baseline_row[column] for column in metric_columns.values()],
        width,
        label="Original tfidf_svm",
        color="#777777",
    )
    ax.bar(
        x + width / 2,
        [optimized_row[column] for column in metric_columns.values()],
        width,
        label="CV-selected SVM",
        color="#4477aa",
    )
    ax.set_xticks(x, list(metric_columns), rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Validation score")
    ax.set_title("Original vs CV-selected SVM on the same validation split")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    _plot_save(fig, output_dir, "validation_metrics_comparison")

    common_vmax = int(max(baseline_confusion.max(), optimized_confusion.max()))
    _plot_confusion(
        baseline_confusion,
        "Original tfidf_svm validation confusion matrix",
        output_dir,
        "original_svm_confusion_matrix",
        common_vmax,
    )
    _plot_confusion(
        optimized_confusion,
        "CV-selected SVM validation confusion matrix",
        output_dir,
        "optimized_svm_confusion_matrix",
        common_vmax,
    )

    for metric, title, filename in (
        ("f1", "Per-class validation F1", "per_class_f1_comparison"),
        ("recall", "Per-class validation recall", "per_class_recall_comparison"),
    ):
        fig, ax = plt.subplots(figsize=(9, 5))
        positions = np.arange(len(CLASSES))
        ax.barh(
            positions - width / 2,
            per_class[f"baseline_{metric}"],
            width,
            label="Original tfidf_svm",
            color="#777777",
        )
        ax.barh(
            positions + width / 2,
            per_class[f"optimized_{metric}"],
            width,
            label="CV-selected SVM",
            color="#4477aa",
        )
        ax.set_yticks(positions, CLASSES)
        ax.set_xlim(0, 1)
        ax.set_xlabel(metric.capitalize())
        ax.set_title(title)
        ax.invert_yaxis()
        ax.legend()
        ax.grid(axis="x", alpha=0.25)
        _plot_save(fig, output_dir, filename)

    heatmap = cv_results.pivot_table(
        index="feature_variant",
        columns="C",
        values="mean_test_macro_f1",
        aggfunc="mean",
    ).reindex(index=feature_names, columns=c_values)
    fig, ax = plt.subplots(figsize=(10, 5))
    image = ax.imshow(heatmap.to_numpy(dtype=float), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(c_values)), [str(value) for value in c_values])
    ax.set_yticks(np.arange(len(feature_names)), feature_names)
    ax.set_xlabel("LinearSVC C")
    ax.set_title("Mean five-fold CV macro-F1 by feature configuration and C")
    for row in range(heatmap.shape[0]):
        for column in range(heatmap.shape[1]):
            value = heatmap.iloc[row, column]
            if pd.notna(value):
                ax.text(column, row, f"{value:.3f}", ha="center", va="center", color="white", fontsize=8)
    fig.colorbar(image, ax=ax, label="Macro-F1")
    _plot_save(fig, output_dir, "cv_hyperparameter_heatmap")

    low, high = confidence_interval
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(bootstrap_differences, bins=40, color="#4477aa", edgecolor="white")
    ax.axvline(0, color="black", linestyle="--", label="No difference")
    ax.axvline(low, color="#cc6677", linestyle=":", label="95% percentile interval")
    ax.axvline(high, color="#cc6677", linestyle=":")
    ax.set_xlabel("Optimized minus baseline validation macro-F1")
    ax.set_ylabel("Bootstrap resamples")
    ax.set_title("Exploratory paired validation bootstrap")
    ax.legend()
    _plot_save(fig, output_dir, "paired_bootstrap_difference")


def _safe_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _safe_json(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_json(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.bool_,)):
        return bool(value)
    return value


def run_experiment(
    root: Path,
    *,
    c_values: tuple[float, ...] = C_VALUES,
    class_weights: tuple[str | None, ...] = CLASS_WEIGHTS,
    feature_names: tuple[str, ...] = tuple(FEATURE_VARIANTS),
    cv_folds: int = DEFAULT_CV_FOLDS,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    overwrite: bool = False,
) -> dict:
    experiment_started = time.perf_counter()
    root = Path(root).expanduser().resolve()
    output_dir = root / DEFAULT_OUTPUT_RELATIVE
    if cv_folds < 2:
        raise ValueError("cv_folds must be at least two.")
    if bootstrap_repeats < 100:
        raise ValueError("Use at least 100 paired bootstrap repeats.")
    if not feature_names or any(name not in FEATURE_VARIANTS for name in feature_names):
        raise ValueError(f"feature_names must be selected from {list(FEATURE_VARIANTS)}.")
    if len(set(feature_names)) != len(feature_names):
        raise ValueError("feature_names must not contain duplicates.")
    transitions_path = root / "private_analysis" / PRIVATE_TRANSITIONS_NAME
    print(f"[1/5] Loading prepared train/validation splits from {root}", flush=True)
    train, validation, split_fingerprint, source = load_experiment_data(root)
    if not overwrite:
        cached_report = _reuse_completed_experiment(
            root,
            output_dir,
            split_fingerprint,
        )
        if cached_report is not None:
            return cached_report
    _check_output_directory(output_dir, overwrite)
    if transitions_path.exists() and not overwrite:
        raise FileExistsError(
            f"{transitions_path} already exists without a complete optimization report. "
            "Use --overwrite only if replacing this prior run is intentional."
        )
    print(
        f"[2/5] Loaded {len(train):,} training and {len(validation):,} validation rows; "
        "verifying the saved baseline.",
        flush=True,
    )
    baseline_model, baseline_probabilities, baseline_info, baseline_paths, baseline_before = (
        load_and_verify_baseline(root, validation, split_fingerprint)
    )
    validation_labels = validation["label"].to_numpy(dtype=int)
    baseline_metrics, baseline_per_class, baseline_confusion, baseline_predictions = (
        evaluate_probabilities(validation_labels, baseline_probabilities)
    )
    recorded_metrics = baseline_info["recorded_metrics"]
    for metric in ("accuracy", "macro_f1", "weighted_f1", "macro_precision", "macro_recall", "balanced_accuracy"):
        if metric in recorded_metrics and not np.isclose(
            float(recorded_metrics[metric]), baseline_metrics[metric], atol=1e-6, rtol=0.0
        ):
            raise RuntimeError(
                f"Saved baseline {metric} does not match verified validation probabilities."
            )
    del baseline_model

    search_started = time.perf_counter()
    print(
        f"[3/5] Baseline verified. Starting {len(feature_names)} feature searches; "
        f"each has {len(c_values) * len(class_weights)} candidates and {cv_folds} CV folds.",
        flush=True,
    )
    cv_tables = []
    for feature_index, feature in enumerate(feature_names, start=1):
        print(
            f"[CV {feature_index}/{len(feature_names)}] Starting {feature}. "
            "Per-fold progress follows.",
            flush=True,
        )
        cv_tables.append(
            cross_validate_training(
                train,
                feature,
                c_values=tuple(c_values),
                class_weights=tuple(class_weights),
                cv_folds=cv_folds,
                seed=SEED,
            )
        )
        search_elapsed = time.perf_counter() - search_started
        estimated_remaining = (
            search_elapsed / feature_index * (len(feature_names) - feature_index)
        )
        print(
            f"[CV {feature_index}/{len(feature_names)}] Finished {feature} in "
            f"{search_elapsed/60:.1f} min; estimated CV time remaining "
            f"{estimated_remaining/60:.1f} min.",
            flush=True,
        )
    cv_results = rank_cv_results(pd.concat(cv_tables, ignore_index=True), feature_names)
    selected = cv_results.iloc[0].to_dict()
    feature_order = {name: index for index, name in enumerate(feature_names)}
    feature_winners = (
        cv_results.assign(
            _weight_order=cv_results["class_weight"].map(_class_weight_order),
            _feature_order=cv_results["feature_variant"].map(feature_order),
        )
        .sort_values(
            [
                "_feature_order",
                "feature_cv_rank",
                "std_test_macro_f1",
                "C",
                "_weight_order",
            ],
            ascending=[True, True, True, True, True],
            kind="mergesort",
        )
        .groupby("feature_variant", sort=False)
        .head(1)
        .drop(columns=["_weight_order", "_feature_order"])
        .reset_index(drop=True)
    )
    selected_feature = str(selected["feature_variant"])
    selected_c = float(selected["C"])
    selected_weight = _class_weight_value(selected["class_weight"])
    print(
        f"[4/5] CV complete. Selected {selected_feature}, C={selected_c:g}, "
        f"class_weight={selected_weight!r}; evaluating validation candidates.",
        flush=True,
    )

    x_validation = validation["text"].to_numpy(dtype=str)
    validation_rows = [
        _metric_row(
            "original_tfidf_svm",
            "original_baseline",
            "original_word_char_1_2_3_5",
            1.0,
            "balanced",
            None,
            baseline_metrics,
            baseline_metrics,
        )
    ]
    optimized_model = None
    optimized_probabilities = None
    validation_candidates = feature_winners.to_dict(orient="records")
    for candidate_index, row in enumerate(validation_candidates, start=1):
        print(
            f"[Validation {candidate_index}/{len(validation_candidates)}] Fitting "
            f"{row['feature_variant']} candidate.",
            flush=True,
        )
        class_weight = _class_weight_value(row["class_weight"])
        candidate = make_calibrated_estimator(
            str(row["feature_variant"]),
            c_value=float(row["C"]),
            class_weight=class_weight,
        )
        candidate.fit(train["text"].to_numpy(dtype=str), train["label"].to_numpy(dtype=int))
        probabilities = predict_probabilities(candidate, x_validation)
        metrics, _, _, _ = evaluate_probabilities(validation_labels, probabilities)
        is_global_winner = (
            row["feature_variant"] == selected_feature
            and float(row["C"]) == selected_c
            and class_weight == selected_weight
        )
        role = "global_cv_best" if is_global_winner else "feature_cv_best"
        validation_rows.append(
            _metric_row(
                f"cv_best_{row['feature_variant']}",
                role,
                str(row["feature_variant"]),
                float(row["C"]),
                class_weight,
                float(row["mean_test_macro_f1"]),
                metrics,
                baseline_metrics,
            )
        )
        if is_global_winner:
            optimized_model = candidate
            optimized_probabilities = probabilities
        else:
            del candidate

    if optimized_model is None or optimized_probabilities is None:
        raise RuntimeError("The global CV winner was not among the feature-specific CV winners.")
    optimized_metrics, optimized_per_class, optimized_confusion, optimized_predictions = (
        evaluate_probabilities(validation_labels, optimized_probabilities)
    )
    baseline_per_class = baseline_per_class.rename(
        columns={
            "precision": "baseline_precision",
            "recall": "baseline_recall",
            "f1": "baseline_f1",
            "support": "validation_support",
        }
    )
    optimized_per_class = optimized_per_class.rename(
        columns={
            "precision": "optimized_precision",
            "recall": "optimized_recall",
            "f1": "optimized_f1",
            "support": "optimized_support",
        }
    )
    per_class_comparison = baseline_per_class.merge(
        optimized_per_class[
            [
                "label_id",
                "optimized_precision",
                "optimized_recall",
                "optimized_f1",
                "optimized_support",
            ]
        ],
        on="label_id",
        validate="one_to_one",
    )
    for metric in ("precision", "recall", "f1"):
        per_class_comparison[f"delta_{metric}"] = (
            per_class_comparison[f"optimized_{metric}"]
            - per_class_comparison[f"baseline_{metric}"]
        )

    baseline_file_hashes_after = {path.name: _hash_file(path) for path in baseline_paths}
    if baseline_file_hashes_after != baseline_before:
        raise RuntimeError("An original tfidf_svm baseline artifact changed during optimization.")

    print(
        f"[5/5] Computing paired bootstrap ({bootstrap_repeats:,} resamples) and writing reports.",
        flush=True,
    )
    bootstrap, bootstrap_differences = _paired_bootstrap(
        validation_labels,
        baseline_predictions,
        optimized_predictions,
        bootstrap_repeats,
        seed=SEED,
    )
    transitions, transition_rows = _error_transitions(
        validation,
        baseline_predictions,
        optimized_predictions,
    )
    transitions["baseline_per_class_recall"] = {
        row["label"]: float(row["baseline_recall"])
        for row in per_class_comparison.to_dict(orient="records")
    }
    transitions["optimized_per_class_recall"] = {
        row["label"]: float(row["optimized_recall"])
        for row in per_class_comparison.to_dict(orient="records")
    }
    transition_cases = transition_rows[
        transition_rows["transition"].isin(["corrected", "new_error"])
    ].reset_index(drop=True)

    _check_output_directory(output_dir, overwrite=True)
    cv_results.to_csv(output_dir / "cv_results.csv", index=False)
    validation_comparison = pd.DataFrame(validation_rows)
    validation_comparison.to_csv(output_dir / "validation_comparison.csv", index=False)
    per_class_comparison.to_csv(output_dir / "per_class_comparison.csv", index=False)
    transitions_path.parent.mkdir(parents=True, exist_ok=True)
    transition_cases.to_csv(transitions_path, index=False)
    np.savez_compressed(
        output_dir / "optimized_validation_predictions.npz",
        labels=validation_labels,
        probs=optimized_probabilities,
        predictions=optimized_predictions,
    )
    prediction_frame = pd.DataFrame(
        {
            "validation_row_index": np.arange(len(validation), dtype=int),
            "true_label_id": validation_labels,
            "true_label": [label_name(value) for value in validation_labels],
            "baseline_prediction_id": baseline_predictions,
            "baseline_prediction": [label_name(value) for value in baseline_predictions],
            "optimized_prediction_id": optimized_predictions,
            "optimized_prediction": [label_name(value) for value in optimized_predictions],
        }
    )
    prediction_frame.to_csv(output_dir / "validation_predictions.csv", index=False)
    joblib.dump(optimized_model, output_dir / "optimized_svm.joblib", compress=3)

    best_parameters = {
        "task": TASK,
        "seed": SEED,
        "primary_metric": "macro_f1",
        "selected_by": "highest mean stratified five-fold CV macro-F1 on training split only",
        "selected_candidate": {
            "feature_variant": selected_feature,
            "C": selected_c,
            "class_weight": selected_weight,
            "mean_cv_macro_f1": float(selected["mean_test_macro_f1"]),
            "std_cv_macro_f1": float(selected["std_test_macro_f1"]),
            "global_cv_rank": int(selected["global_cv_rank"]),
        },
        "feature_specific_cv_winners": [
            {
                "feature_variant": str(row["feature_variant"]),
                "C": float(row["C"]),
                "class_weight": _class_weight_value(row["class_weight"]),
                "mean_cv_macro_f1": float(row["mean_test_macro_f1"]),
                "std_cv_macro_f1": float(row["std_test_macro_f1"]),
            }
            for row in feature_winners.to_dict(orient="records")
        ],
        "configuration": {
            "C_values": [float(value) for value in c_values],
            "class_weight_values": [value for value in class_weights],
            "feature_variants": {
                feature: FEATURE_VARIANTS[feature]["description"] for feature in feature_names
            },
            "outer_cv": {
                "class": "StratifiedKFold",
                "folds": int(cv_folds),
                "shuffle": True,
                "random_state": SEED,
                "scoring": "f1_macro",
                "n_jobs": 1,
            },
            "calibration": {
                "class": "CalibratedClassifierCV",
                "method": "sigmoid",
                "folds": CALIBRATION_FOLDS,
                "ensemble": True,
                "inside_fold_preprocessing": True,
            },
            "linear_svc": {
                "loss": "squared_hinge",
                "max_iter": 5000,
                "random_state": SEED,
            },
            "tfidf": {
                "word": {
                    "analyzer": "word",
                    "ngram_range": [1, 2],
                    "min_df": 2,
                    "max_features": 45000,
                    "sublinear_tf": True,
                },
                "character": {
                    "analyzer": "char",
                    "ngram_range_variants": [[2, 5], [3, 5]],
                    "min_df": 2,
                    "max_features": 50000,
                    "sublinear_tf": True,
                },
            },
            "bootstrap": {
                "repeats": int(bootstrap_repeats),
                "seed": SEED,
                "confidence_interval": "percentile 95%",
            },
            "finalize": False,
            "test_split_read_or_evaluated": False,
        },
    }
    (output_dir / "best_parameters.json").write_text(
        json.dumps(_safe_json(best_parameters), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    report = {
        "task": TASK,
        "dataset_source": source,
        "class_mapping": {str(index): name for index, name in enumerate(CLASSES)},
        "sample_counts": {
            "training": int(len(train)),
            "validation": int(len(validation)),
            "test_rows_loaded": False,
            "test_labels_inspected": False,
        },
        "reproducibility": {
            "seed": SEED,
            "sklearn_version": sklearn.__version__,
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "python_version": platform.python_version(),
            "stored_project_split_sha256": split_fingerprint,
            "split_fingerprint_source": "results/data_audit.json, cross-checked against baseline metadata",
            "full_fingerprint_recomputed": False,
            "current_train_validation_sha256": train_validation_fingerprint(train, validation),
            "baseline_validation_predictions_verified": baseline_info[
                "saved_validation_predictions_verified"
            ],
            "baseline_validation_row_order_verified": baseline_info[
                "saved_validation_row_order_verified"
            ],
        },
        "cross_validation": {
            "training_rows_only": True,
            "folds": int(cv_folds),
            "scoring": "macro-F1",
            "calibration_and_tfidf_inside_nested_folds": True,
            "candidate_count": int(len(cv_results)),
            "selected_candidate": best_parameters["selected_candidate"],
        },
        "validation_metrics": {
            "baseline": baseline_metrics,
            "optimized": optimized_metrics,
            "delta_optimized_minus_baseline": {
                metric: float(optimized_metrics[metric]) - float(baseline_metrics[metric])
                for metric in METRIC_NAMES
            },
            "per_class": per_class_comparison.to_dict(orient="records"),
            "baseline_confusion_matrix": baseline_confusion.tolist(),
            "optimized_confusion_matrix": optimized_confusion.tolist(),
            "confusion_matrix_delta": (optimized_confusion - baseline_confusion).tolist(),
            "bootstrap": bootstrap,
        },
        "error_transitions": transitions,
        "baseline_artifact_preservation": {
            "unchanged": True,
            "artifacts": list(baseline_file_hashes_after.values()),
            "optimization_outputs_are_separate": True,
        },
        "interpretation": {
            "validation_macro_f1_point_estimate_higher": (
                optimized_metrics["macro_f1"] > baseline_metrics["macro_f1"]
            ),
            "bootstrap_interval_lower_bound_above_zero": bootstrap[
                "confidence_interval_95_percentile"
            ][0] > 0.0,
            "bootstrap_interval_crosses_zero": bootstrap["interval_crosses_zero"],
            "statistical_significance_claimed": False,
            "notes": [
                "Candidate selection used training-only stratified cross-validation.",
                "Validation was used only for the prespecified final comparisons.",
                "All per-class rows, including offensive, abusive, and mockery, are reported.",
                "The paired bootstrap is exploratory and does not replace independent confirmation.",
                "The test split and its labels were not read or evaluated.",
            ],
        },
        "output_directory": str(output_dir),
        "private_error_transition_file": str(transitions_path),
        "visualizations": [f"{name}.{ext}" for name in FIGURE_NAMES for ext in ("png", "svg")],
    }
    _make_figures(
        output_dir,
        cv_results,
        validation_comparison,
        per_class_comparison,
        baseline_confusion,
        optimized_confusion,
        bootstrap_differences,
        bootstrap["confidence_interval_95_percentile"],
    )
    _write_markdown_report(output_dir, report, validation_comparison, per_class_comparison)
    (output_dir / "optimization_report.json").write_text(
        json.dumps(_safe_json(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    (output_dir / "error_transition_summary.json").write_text(
        json.dumps(_safe_json(transitions), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    baseline_final = {path.name: _hash_file(path) for path in baseline_paths}
    if baseline_final != baseline_before:
        raise RuntimeError("An original tfidf_svm baseline artifact changed during optimization.")

    print(
        f"Optimization experiment complete in {(time.perf_counter()-experiment_started)/60:.1f} min: "
        f"{output_dir}",
        flush=True,
    )
    print(
        f"Selected by CV: {selected_feature}, C={selected_c:g}, class_weight={selected_weight!r}",
        flush=True,
    )
    print(
        f"Validation macro-F1: baseline={baseline_metrics['macro_f1']:.4f}, "
        f"optimized={optimized_metrics['macro_f1']:.4f}",
        flush=True,
    )
    print("Test split read: no; FINALIZE: False", flush=True)
    return report


def _write_markdown_report(
    output_dir: Path,
    report: dict,
    validation_comparison: pd.DataFrame,
    per_class_comparison: pd.DataFrame,
) -> None:
    ci_low, ci_high = report["validation_metrics"]["bootstrap"][
        "confidence_interval_95_percentile"
    ]
    lines = [
        "# Cyberbullying7 TF-IDF + SVM optimization",
        "",
        "## Protocol",
        "",
        "- Hyperparameter selection: stratified five-fold CV on training rows only; macro-F1.",
        "- TF-IDF and sigmoid calibration are fitted inside nested CV folds.",
        "- Validation is used only for the final candidate comparisons.",
        "- Test rows and labels are not loaded or evaluated. `FINALIZE=False`.",
        "- The original `tfidf_svm` artifact remains unchanged.",
        f"- Seed: {SEED}; scikit-learn: `{sklearn.__version__}`.",
        f"- Stored split fingerprint: `{report['reproducibility']['stored_project_split_sha256']}`.",
        "",
        "## CV-selected configuration",
        "",
        f"- Feature variant: `{report['cross_validation']['selected_candidate']['feature_variant']}`",
        f"- C: `{report['cross_validation']['selected_candidate']['C']}`",
        f"- class_weight: `{report['cross_validation']['selected_candidate']['class_weight']}`",
        f"- Mean CV macro-F1: {report['cross_validation']['selected_candidate']['mean_cv_macro_f1']:.4f}",
        "",
        "## Same-validation comparison",
        "",
        "| Candidate | Feature configuration | Accuracy | Macro-F1 | Weighted F1 | Macro precision | Macro recall | Balanced accuracy |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in validation_comparison.to_dict(orient="records"):
        lines.append(
            f"| {row['candidate']} | {row['feature_variant']} | "
            f"{row['validation_accuracy']:.4f} | {row['validation_macro_f1']:.4f} | "
            f"{row['validation_weighted_f1']:.4f} | {row['validation_macro_precision']:.4f} | "
            f"{row['validation_macro_recall']:.4f} | {row['validation_balanced_accuracy']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Priority-class metrics",
            "",
            "| Class | Baseline precision | Optimized precision | Baseline recall | Optimized recall | Baseline F1 | Optimized F1 |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in per_class_comparison.to_dict(orient="records"):
        if row["label"] in ("offensive", "abusive", "mockery"):
            lines.append(
                f"| {row['label']} | {row['baseline_precision']:.4f} | "
                f"{row['optimized_precision']:.4f} | {row['baseline_recall']:.4f} | "
                f"{row['optimized_recall']:.4f} | {row['baseline_f1']:.4f} | "
                f"{row['optimized_f1']:.4f} |"
            )
    lines.extend(
        [
            "",
            "## Error transitions and paired bootstrap",
            "",
            f"- Corrected validation errors: {report['error_transitions']['corrected_cases']}",
            f"- Newly introduced errors: {report['error_transitions']['newly_introduced_errors']}",
            f"- Optimized minus baseline macro-F1: "
            f"{report['validation_metrics']['bootstrap']['delta_optimized_minus_baseline_macro_f1']:.4f}",
            f"- Exploratory paired 95% percentile interval: [{ci_low:.4f}, {ci_high:.4f}]",
            f"- {report['validation_metrics']['bootstrap']['interpretation']}",
            "- Per-row text for error transitions is stored only under `private_analysis/`.",
            "",
            "## Cautions",
            "",
            "This is one validation comparison after training-only CV selection. The bootstrap "
            "interval is exploratory and does not establish generalization beyond this fixed "
            "validation split. A validation result alone is not evidence to change the locked "
            "model or claim a statistically significant improvement.",
            "",
            "Figures are provided as both PNG and SVG in this experiment directory.",
            "",
        ]
    )
    (output_dir / "optimization_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optimize the existing cyberbullying7 TF-IDF + SVM using training-only CV."
    )
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT, help="Existing V5 experiment root.")
    parser.add_argument(
        "--bootstrap-repeats",
        type=int,
        default=DEFAULT_BOOTSTRAP_REPEATS,
        help="Paired validation bootstrap repeats (default: 5000).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace only recognized outputs in this optimization directory.",
    )
    args = parser.parse_args()
    run_experiment(
        args.root,
        bootstrap_repeats=args.bootstrap_repeats,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
