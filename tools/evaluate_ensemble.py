#!/usr/bin/env python3
"""Compare fixed, validation-only ensembles for the cyberbullying7 task."""
from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import mixbhasha_core as core

TASK = "cyberbullying7"
SEED = 42
CLASSES = list(core.LABELS[TASK])
SVM_NAME = "tfidf_svm"
ROBERTA_NAME = "hing_roberta_mixed"
MODEL_NAMES = (SVM_NAME, ROBERTA_NAME)
DEFAULT_ROOT = Path("/content/drive/MyDrive/MixBhasha_V5_seed42")
DEFAULT_OUTPUT_RELATIVE = Path("experiments") / "cyberbullying7_ensemble"
DEFAULT_BOOTSTRAP_REPEATS = 5000
PROBABILITY_RTOL = 1e-4
PROBABILITY_ATOL = 1e-5
METRICS = (
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "macro_precision",
    "macro_recall",
    "balanced_accuracy",
)
CANDIDATE_NAMES = (
    SVM_NAME,
    ROBERTA_NAME,
    "ensemble_equal_50_50",
    "ensemble_svm_75_roberta_25",
    "ensemble_hard_vote_svm_tiebreak",
)
DISPLAY_NAMES = {
    SVM_NAME: "Original TF-IDF + SVM",
    ROBERTA_NAME: "HingRoBERTa-Mixed",
    "ensemble_equal_50_50": "Probability average (50/50)",
    "ensemble_svm_75_roberta_25": "Probability average (75/25)",
    "ensemble_hard_vote_svm_tiebreak": "Hard vote (ties to SVM)",
}
FIGURE_NAMES = (
    "macro_f1_comparison",
    "all_metrics_comparison",
    "per_class_f1_comparison",
    "per_class_recall_comparison",
    "model_agreement",
    "error_recovery_comparison",
    "paired_bootstrap_macro_f1_differences",
    *(f"confusion_matrix_{name}" for name in CANDIDATE_NAMES),
)
OUTPUT_FILENAMES = {
    "ensemble_report.json",
    "ensemble_report.md",
    "ensemble_comparison.csv",
    "per_class_ensemble_comparison.csv",
    "model_agreement.json",
    *(
        f"{name}.{extension}"
        for name in FIGURE_NAMES
        for extension in ("png", "svg")
    ),
}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object in {path}.")
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return _json_safe(value.tolist())
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _validate_fingerprint(value: Any, source: str) -> str:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdefABCDEF" for character in value)
    ):
        raise ValueError(f"{source} does not contain a valid SHA-256 split fingerprint.")
    return value.lower()


def verify_split_fingerprints(
    audit_fingerprint: str,
    metadata_by_model: dict[str, dict[str, Any]],
) -> str:
    """Require the audit record and every model record to share one split hash."""
    expected = _validate_fingerprint(audit_fingerprint, "results/data_audit.json")
    for model_name, metadata in metadata_by_model.items():
        actual = _validate_fingerprint(
            metadata.get("split_sha256"),
            f"{model_name} validated.json metadata",
        )
        if actual != expected:
            raise RuntimeError(
                f"Split fingerprint mismatch for {model_name}: expected {expected}, "
                f"found {actual}; refusing to compare predictions."
            )
    return expected


def _validate_model_metadata(
    model_name: str,
    record: dict[str, Any],
) -> dict[str, Any]:
    metadata = record.get("meta")
    if not isinstance(metadata, dict):
        raise ValueError(f"{model_name}/validated.json has no meta object.")
    if metadata.get("task") != TASK or metadata.get("model") != model_name:
        raise ValueError(
            f"{model_name}/validated.json does not identify task {TASK!r} "
            f"and model {model_name!r}."
        )
    if metadata.get("seed") != SEED:
        raise ValueError(
            f"{model_name} was not recorded with seed {SEED}; "
            f"found {metadata.get('seed')!r}."
        )
    return metadata


def _index_mapping(mapping: Any, source: str) -> list[str]:
    if not isinstance(mapping, dict):
        raise ValueError(f"{source} is missing a verifiable class mapping.")
    labels = []
    for index in range(len(CLASSES)):
        if index in mapping:
            label = mapping[index]
        elif str(index) in mapping:
            label = mapping[str(index)]
        else:
            raise ValueError(f"{source} is missing class index {index}.")
        labels.append(str(label))
    return labels


def verify_model_class_order(model_name: str, loaded_model: tuple) -> list[str]:
    """Check that saved model output columns follow the core's explicit label IDs."""
    expected = list(CLASSES)
    if not isinstance(loaded_model, tuple) or len(loaded_model) != 3:
        raise TypeError(f"Unexpected loaded model structure for {model_name}.")
    kind, model, _ = loaded_model
    if model_name == SVM_NAME:
        if kind != "classical":
            raise TypeError(f"{model_name} must load as a classical model.")
        raw_classes = getattr(model, "classes_", None)
        if raw_classes is None:
            final_estimator = getattr(model, "named_steps", {}).get("classifier")
            raw_classes = getattr(final_estimator, "classes_", None)
        if raw_classes is None:
            raise ValueError(f"{model_name} artifact does not expose its class order.")
        actual_ids = np.asarray(raw_classes)
        if not np.array_equal(actual_ids, np.arange(len(expected))):
            raise ValueError(
                f"{model_name} class IDs {actual_ids.tolist()} do not match "
                f"the required order 0..{len(expected) - 1}."
            )
        return expected

    if model_name == ROBERTA_NAME:
        if kind != "transformer":
            raise TypeError(f"{model_name} must load as a transformer model.")
        config = getattr(model, "config", None)
        if config is None:
            raise ValueError(f"{model_name} artifact has no model config.")
        if int(getattr(config, "num_labels", -1)) != len(expected):
            raise ValueError(f"{model_name} config does not declare seven output labels.")
        id2label = _index_mapping(getattr(config, "id2label", None), "id2label")
        label2id = getattr(config, "label2id", None)
        if not isinstance(label2id, dict):
            raise ValueError("HingRoBERTa config is missing a verifiable label2id mapping.")
        normalized_label2id = {
            str(label): int(index) for label, index in label2id.items()
        }
        expected_label2id = {label: index for index, label in enumerate(expected)}
        if id2label != expected or normalized_label2id != expected_label2id:
            raise ValueError(
                "HingRoBERTa class order does not exactly match "
                f"core.LABELS[{TASK!r}]: {expected!r}; found id2label={id2label!r}."
            )
        return id2label

    raise ValueError(f"Unsupported model name {model_name!r}.")


def validate_probabilities(probabilities: np.ndarray, row_count: int, source: str) -> np.ndarray:
    values = np.asarray(probabilities, dtype=float)
    expected_shape = (row_count, len(CLASSES))
    if values.shape != expected_shape:
        raise ValueError(
            f"{source} must have probability shape {expected_shape}; got {values.shape}."
        )
    if not np.isfinite(values).all() or np.any(values < 0):
        raise ValueError(f"{source} contains non-finite or negative probabilities.")
    if not np.allclose(values.sum(axis=1), 1.0, rtol=0.0, atol=2e-3):
        raise ValueError(f"{source} probability rows do not sum approximately to one.")
    return values


def verify_prediction_alignment(
    model_name: str,
    saved_probabilities: np.ndarray,
    saved_labels: np.ndarray,
    validation: pd.DataFrame,
    reproduced_probabilities: np.ndarray,
) -> dict[str, Any]:
    """Verify labels and saved probability order against current validation rows."""
    if not {"text", "label"}.issubset(validation.columns):
        raise ValueError("Prepared validation split must contain text and label columns.")
    current_labels = pd.to_numeric(validation["label"], errors="raise").to_numpy(dtype=int)
    if current_labels.size == 0:
        raise ValueError("Prepared validation split is empty.")
    if not set(current_labels).issubset(set(range(len(CLASSES)))):
        raise ValueError("Prepared validation split contains an unknown class ID.")
    stored_labels = np.asarray(saved_labels)
    if stored_labels.ndim != 1 or not np.issubdtype(stored_labels.dtype, np.integer):
        raise ValueError(f"{model_name} saved validation labels must be a one-dimensional integer array.")
    if not np.array_equal(stored_labels, current_labels):
        raise RuntimeError(
            f"{model_name} saved labels do not match the current validation labels "
            "in value and row order."
        )

    saved = validate_probabilities(
        saved_probabilities,
        len(validation),
        f"{model_name} saved predictions",
    )
    reproduced = validate_probabilities(
        reproduced_probabilities,
        len(validation),
        f"{model_name} recomputed predictions",
    )
    max_abs_difference = float(np.max(np.abs(saved - reproduced)))
    if not np.allclose(
        saved,
        reproduced,
        rtol=PROBABILITY_RTOL,
        atol=PROBABILITY_ATOL,
    ):
        raise RuntimeError(
            f"{model_name} saved probabilities do not align with the current validation "
            f"row order/model artifact (maximum absolute difference {max_abs_difference:.6g})."
        )
    return {
        "saved_labels_match_validation_in_order": True,
        "saved_probabilities_match_recomputed_rows": True,
        "max_probability_absolute_difference": max_abs_difference,
        "probability_rtol": PROBABILITY_RTOL,
        "probability_atol": PROBABILITY_ATOL,
    }


def _validate_validation_frame(validation: pd.DataFrame) -> pd.DataFrame:
    if not {"text", "label"}.issubset(validation.columns):
        raise ValueError("Prepared validation split must contain text and label columns.")
    frame = validation[["text", "label"]].copy()
    if frame.empty or frame["text"].isna().any() or frame["label"].isna().any():
        raise ValueError("Prepared validation split is empty or contains missing values.")
    frame["text"] = frame["text"].astype(str)
    frame["label"] = pd.to_numeric(frame["label"], errors="raise").astype(int)
    labels = set(frame["label"])
    expected = set(range(len(CLASSES)))
    if labels != expected:
        raise ValueError(
            f"Validation labels must contain all seven core classes; missing "
            f"{sorted(expected - labels)}, unexpected {sorted(labels - expected)}."
        )
    return frame.reset_index(drop=True)


def _load_optimization_context(root: Path, expected_fingerprint: str) -> dict[str, Any]:
    report_path = (
        root
        / "experiments"
        / "cyberbullying7_svm_optimization"
        / "optimization_report.json"
    )
    if not report_path.is_file():
        return {"available": False, "path": str(report_path)}

    report = _read_json(report_path)
    if report.get("task") != TASK:
        raise ValueError(f"{report_path} does not describe task {TASK!r}.")
    reproducibility = report.get("reproducibility")
    if not isinstance(reproducibility, dict):
        raise ValueError(f"{report_path} has no reproducibility section.")
    report_fingerprint = _validate_fingerprint(
        reproducibility.get("stored_project_split_sha256"),
        f"{report_path} reproducibility",
    )
    if report_fingerprint != expected_fingerprint:
        raise RuntimeError(
            "The SVM optimization report has a different dataset fingerprint; "
            "refusing to mix experiment reports."
        )
    mapping = report.get("class_mapping")
    expected_mapping = {str(index): name for index, name in enumerate(CLASSES)}
    if mapping != expected_mapping:
        raise RuntimeError("The SVM optimization report class mapping does not match core labels.")
    validation_metrics = report.get("validation_metrics", {})
    comparison_path = report_path.with_name("validation_comparison.csv")
    char_only_macro_f1 = None
    if comparison_path.is_file():
        comparison = pd.read_csv(comparison_path)
        if {"feature_variant", "validation_macro_f1"}.issubset(comparison.columns):
            char_rows = comparison[comparison["feature_variant"] == "char_3_5"]
            if len(char_rows) > 1:
                raise ValueError(
                    f"{comparison_path} has multiple char_3_5 rows; cannot report one score."
                )
            if not char_rows.empty:
                char_only_macro_f1 = float(char_rows.iloc[0]["validation_macro_f1"])
    return {
        "available": True,
        "path": str(report_path),
        "split_fingerprint_verified": True,
        "original_svm_macro_f1": validation_metrics.get("baseline", {}).get("macro_f1"),
        "optimized_svm_macro_f1": validation_metrics.get("optimized", {}).get("macro_f1"),
        "char_3_5_validation_macro_f1": char_only_macro_f1,
        "optimized_svm_selected_candidate": report.get("cross_validation", {}).get(
            "selected_candidate"
        ),
    }


def load_verified_predictions(
    root: Path,
) -> tuple[pd.DataFrame, dict[str, np.ndarray], dict[str, Any], str, dict[str, Any]]:
    root = Path(root)
    print("[1/4] Loading validation-only data and stored split fingerprint.", flush=True)
    audit_path = root / "results" / "data_audit.json"
    if not audit_path.is_file():
        raise FileNotFoundError(
            f"Missing {audit_path}; prepare the existing V5 experiment before evaluation."
        )
    audit = _read_json(audit_path)
    task_audit = audit.get(TASK)
    if not isinstance(task_audit, dict):
        raise ValueError(f"{audit_path} has no {TASK!r} entry.")
    audit_fingerprint = _validate_fingerprint(
        task_audit.get("split_sha256"),
        f"{audit_path} {TASK}",
    )

    validation_data = core.load_prepared(root, [TASK], splits=("val",))[TASK]
    validation = _validate_validation_frame(validation_data["val"])
    metadata_by_model: dict[str, dict[str, Any]] = {}
    for model_name in MODEL_NAMES:
        model_dir = root / "models" / TASK / model_name
        metadata_path = model_dir / "validated.json"
        prediction_path = model_dir / "validation_probabilities.npz"
        model_path = model_dir / "model.joblib" if model_name == SVM_NAME else model_dir / "final"
        if not metadata_path.is_file():
            raise FileNotFoundError(f"Missing saved model metadata: {metadata_path}")
        if not prediction_path.is_file():
            raise FileNotFoundError(f"Missing saved validation predictions: {prediction_path}")
        if model_name == SVM_NAME and not model_path.is_file():
            raise FileNotFoundError(f"Missing saved SVM model artifact: {model_path}")
        if model_name == ROBERTA_NAME and not (model_path / "config.json").is_file():
            raise FileNotFoundError(
                f"Missing saved HingRoBERTa config: {model_path / 'config.json'}"
            )
        record = _read_json(metadata_path)
        metadata = _validate_model_metadata(model_name, record)
        metadata_by_model[model_name] = metadata

    fingerprint = verify_split_fingerprints(audit_fingerprint, metadata_by_model)
    optimization_context = _load_optimization_context(root, fingerprint)

    probabilities: dict[str, np.ndarray] = {}
    alignment: dict[str, Any] = {}
    for model_index, model_name in enumerate(MODEL_NAMES, start=1):
        print(
            f"[2/4] Verifying saved model {model_index}/{len(MODEL_NAMES)}: {model_name}.",
            flush=True,
        )
        prediction_path = root / "models" / TASK / model_name / "validation_probabilities.npz"
        with np.load(prediction_path, allow_pickle=False) as archive:
            if not {"probs", "labels"}.issubset(archive.files):
                raise ValueError(
                    f"{prediction_path} must contain 'probs' and 'labels' arrays."
                )
            saved_probabilities = np.asarray(archive["probs"], dtype=float)
            saved_labels = np.asarray(archive["labels"])

        loaded_model = core.load_model(root, TASK, model_name)
        verify_model_class_order(model_name, loaded_model)
        max_len = int(metadata_by_model[model_name].get("max_len", 128))
        if max_len < 1:
            raise ValueError(f"{model_name} metadata has an invalid max_len.")
        recomputed = core.predict_loaded(
            loaded_model,
            validation,
            max_len=max_len,
        )
        probabilities[model_name] = validate_probabilities(
            saved_probabilities,
            len(validation),
            f"{model_name} saved predictions",
        )
        alignment[model_name] = verify_prediction_alignment(
            model_name,
            probabilities[model_name],
            saved_labels,
            validation,
            recomputed,
        )
        del loaded_model
        print(
            f"Verified {model_name}: fingerprint, label/class order, and validation row alignment.",
            flush=True,
        )

    return validation, probabilities, alignment, fingerprint, optimization_context


def hard_vote_svm_tiebreak(
    svm_predictions: np.ndarray,
    roberta_predictions: np.ndarray,
) -> np.ndarray:
    svm = np.asarray(svm_predictions, dtype=int)
    roberta = np.asarray(roberta_predictions, dtype=int)
    if svm.ndim != 1 or roberta.ndim != 1 or svm.shape != roberta.shape:
        raise ValueError("Hard-vote predictions must be equally sized one-dimensional arrays.")
    if (
        not set(svm).issubset(set(range(len(CLASSES))))
        or not set(roberta).issubset(set(range(len(CLASSES))))
    ):
        raise ValueError("Hard-vote predictions contain an invalid class ID.")
    votes = np.eye(len(CLASSES), dtype=int)[svm] + np.eye(len(CLASSES), dtype=int)[roberta]
    winning_votes = votes.max(axis=1, keepdims=True)
    tied_winners = votes == winning_votes
    deterministic_winner = tied_winners.argmax(axis=1)
    svm_is_tied_winner = tied_winners[np.arange(len(svm)), svm]
    return np.where(svm_is_tied_winner, svm, deterministic_winner)


def build_candidate_predictions(
    svm_probabilities: np.ndarray,
    roberta_probabilities: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    row_count = len(svm_probabilities)
    svm_probs = validate_probabilities(svm_probabilities, row_count, SVM_NAME)
    roberta_probs = validate_probabilities(roberta_probabilities, row_count, ROBERTA_NAME)
    svm_predictions = svm_probs.argmax(axis=1)
    roberta_predictions = roberta_probs.argmax(axis=1)
    equal_probabilities = 0.5 * svm_probs + 0.5 * roberta_probs
    weighted_probabilities = 0.75 * svm_probs + 0.25 * roberta_probs
    probabilities = {
        SVM_NAME: svm_probs,
        ROBERTA_NAME: roberta_probs,
        "ensemble_equal_50_50": equal_probabilities,
        "ensemble_svm_75_roberta_25": weighted_probabilities,
    }
    predictions = {
        name: values.argmax(axis=1) for name, values in probabilities.items()
    }
    predictions["ensemble_hard_vote_svm_tiebreak"] = hard_vote_svm_tiebreak(
        svm_predictions,
        roberta_predictions,
    )
    definitions = {
        "ensemble_equal_50_50": {
            "method": "multiclass probability average",
            "weights": {SVM_NAME: 0.5, ROBERTA_NAME: 0.5},
        },
        "ensemble_svm_75_roberta_25": {
            "method": "multiclass probability average",
            "weights": {SVM_NAME: 0.75, ROBERTA_NAME: 0.25},
        },
        "ensemble_hard_vote_svm_tiebreak": {
            "method": "two-model hard vote; ties resolved in favor of tfidf_svm",
            "weights": None,
        },
    }
    return probabilities, predictions, definitions


def evaluate_predictions(
    labels: np.ndarray,
    predictions: np.ndarray,
) -> tuple[dict[str, float | int], pd.DataFrame, np.ndarray]:
    y_true = np.asarray(labels, dtype=int)
    y_pred = np.asarray(predictions, dtype=int)
    if y_true.ndim != 1 or y_pred.shape != y_true.shape:
        raise ValueError("True and predicted labels must be aligned one-dimensional arrays.")
    label_ids = np.arange(len(CLASSES))
    if not set(y_true).issubset(set(label_ids)):
        raise ValueError("True labels contain a class ID outside the core seven-class mapping.")
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=label_ids,
        zero_division=0,
    )
    precision_values = np.asarray(precision, dtype=float)
    recall_values = np.asarray(recall, dtype=float)
    f1_values = np.asarray(f1, dtype=float)
    support_values = np.bincount(y_true, minlength=len(CLASSES)).astype(int)
    metrics: dict[str, float | int] = {
        "n": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(
            f1_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(y_true, y_pred, labels=label_ids, average="weighted", zero_division=0)
        ),
        "macro_precision": float(
            precision_score(
                y_true, y_pred, labels=label_ids, average="macro", zero_division=0
            )
        ),
        "macro_recall": float(
            recall_score(y_true, y_pred, labels=label_ids, average="macro", zero_division=0)
        ),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
    }
    per_class = pd.DataFrame(
        {
            "label_id": label_ids,
            "label": CLASSES,
            "precision": precision_values,
            "recall": recall_values,
            "f1": f1_values,
            "support": support_values,
        }
    )
    matrix = confusion_matrix(y_true, y_pred, labels=label_ids)
    return metrics, per_class, matrix


def paired_bootstrap_macro_f1(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
    repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    seed: int = SEED,
    batch_size: int = 128,
) -> tuple[dict[str, Any], np.ndarray]:
    """Paired row bootstrap of macro-F1 differences, with fixed class order."""
    y_true = np.asarray(labels, dtype=int)
    baseline = np.asarray(baseline_predictions, dtype=int)
    candidate = np.asarray(candidate_predictions, dtype=int)
    if y_true.ndim != 1 or baseline.shape != y_true.shape or candidate.shape != y_true.shape:
        raise ValueError("Paired bootstrap inputs must have equal one-dimensional shapes.")
    if repeats < 100:
        raise ValueError("Use at least 100 paired bootstrap resamples.")
    if batch_size < 1:
        raise ValueError("batch_size must be positive.")
    rng = np.random.default_rng(seed)
    differences = np.empty(repeats, dtype=float)
    class_ids = np.arange(len(CLASSES))

    def batch_macro_f1(y_batch: np.ndarray, pred_batch: np.ndarray) -> np.ndarray:
        class_f1 = []
        for class_id in class_ids:
            true_positive = np.sum(
                (y_batch == class_id) & (pred_batch == class_id),
                axis=1,
            )
            true_count = np.sum(y_batch == class_id, axis=1)
            predicted_count = np.sum(pred_batch == class_id, axis=1)
            denominator = true_count + predicted_count
            score = np.divide(
                2.0 * true_positive,
                denominator,
                out=np.zeros_like(true_positive, dtype=float),
                where=denominator != 0,
            )
            class_f1.append(score)
        return np.mean(np.stack(class_f1, axis=1), axis=1)

    for start in range(0, repeats, batch_size):
        stop = min(start + batch_size, repeats)
        indices = rng.integers(0, len(y_true), size=(stop - start, len(y_true)))
        y_batch = y_true[indices]
        baseline_batch = batch_macro_f1(y_batch, baseline[indices])
        candidate_batch = batch_macro_f1(y_batch, candidate[indices])
        differences[start:stop] = candidate_batch - baseline_batch

    low, high = (float(value) for value in np.quantile(differences, [0.025, 0.975]))
    result = {
        "repeats": int(repeats),
        "seed": int(seed),
        "confidence_level": 0.95,
        "interval_method": "paired percentile bootstrap over validation rows",
        "mean_macro_f1_difference": float(np.mean(differences)),
        "median_macro_f1_difference": float(np.median(differences)),
        "confidence_interval_95_percentile": [low, high],
        "interval_crosses_zero": bool(low <= 0.0 <= high),
        "statistical_significance_claimed": False,
    }
    return result, differences


def _per_class_recovery(
    labels: np.ndarray,
    baseline_predictions: np.ndarray,
    candidate_predictions: np.ndarray,
) -> list[dict[str, Any]]:
    y_true = np.asarray(labels, dtype=int)
    baseline_correct = np.asarray(baseline_predictions) == y_true
    candidate_correct = np.asarray(candidate_predictions) == y_true
    rows = []
    for label_id, label_name in enumerate(CLASSES):
        in_class = y_true == label_id
        rows.append(
            {
                "label_id": label_id,
                "label": label_name,
                "baseline_error_count": int(np.sum(in_class & ~baseline_correct)),
                "errors_recovered": int(
                    np.sum(in_class & ~baseline_correct & candidate_correct)
                ),
                "new_errors": int(
                    np.sum(in_class & baseline_correct & ~candidate_correct)
                ),
                "candidate_error_count": int(np.sum(in_class & ~candidate_correct)),
            }
        )
    return rows


def build_model_agreement(
    labels: np.ndarray,
    candidate_predictions: dict[str, np.ndarray],
) -> dict[str, Any]:
    y_true = np.asarray(labels, dtype=int)
    svm_predictions = np.asarray(candidate_predictions[SVM_NAME], dtype=int)
    roberta_predictions = np.asarray(candidate_predictions[ROBERTA_NAME], dtype=int)
    svm_correct = svm_predictions == y_true
    roberta_correct = roberta_predictions == y_true
    disagreement = svm_predictions != roberta_predictions
    status = {
        "both_correct": int(np.sum(svm_correct & roberta_correct)),
        "both_incorrect": int(np.sum(~svm_correct & ~roberta_correct)),
        "svm_correct_hing_roberta_incorrect": int(np.sum(svm_correct & ~roberta_correct)),
        "hing_roberta_correct_svm_incorrect": int(np.sum(~svm_correct & roberta_correct)),
    }
    recovery = {
        model_name: _per_class_recovery(
            y_true,
            svm_predictions,
            np.asarray(candidate_predictions[model_name], dtype=int),
        )
        for model_name in CANDIDATE_NAMES[1:]
    }
    svm_recovery_from_roberta = _per_class_recovery(
        y_true,
        roberta_predictions,
        svm_predictions,
    )
    focused_classes = {}
    for label_name in ("offensive", "abusive", "mockery"):
        focused_classes[label_name] = {
            "recovered_by_hing_roberta_from_svm_errors": next(
                row
                for row in recovery[ROBERTA_NAME]
                if row["label"] == label_name
            ),
            "recovered_by_svm_from_hing_roberta_errors": next(
                row
                for row in svm_recovery_from_roberta
                if row["label"] == label_name
            ),
            "ensembles_vs_svm": {
                model_name: next(
                    row
                    for row in recovery[model_name]
                    if row["label"] == label_name
                )
                for model_name in CANDIDATE_NAMES[2:]
            },
        }
    return {
        "task": TASK,
        "sample_count": int(len(y_true)),
        "correctness_partition": status,
        "prediction_disagreement_count": int(np.sum(disagreement)),
        "prediction_disagreement_percentage": float(100.0 * np.mean(disagreement)),
        "prediction_agreement_count": int(np.sum(~disagreement)),
        "prediction_agreement_percentage": float(100.0 * np.mean(~disagreement)),
        "hard_vote_matches_original_svm_predictions": bool(
            np.array_equal(
                np.asarray(candidate_predictions["ensemble_hard_vote_svm_tiebreak"]),
                svm_predictions,
            )
        ),
        "svm_errors_recovered_by_hing_roberta": int(
            np.sum(~svm_correct & roberta_correct)
        ),
        "hing_roberta_errors_recovered_by_svm": int(
            np.sum(~roberta_correct & svm_correct)
        ),
        "per_class_error_recovery_vs_svm": recovery,
        "per_class_svm_recovery_from_hing_roberta_errors": svm_recovery_from_roberta,
        "focus_classes": focused_classes,
        "raw_text_exported": False,
    }


def _output_dir(root: Path) -> Path:
    return root / DEFAULT_OUTPUT_RELATIVE


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
            f"{output_dir} already contains ensemble outputs. Use --overwrite only "
            "if replacing this exact evaluation is intentional."
        )


def _save_figure(fig: Figure, output_dir: Path, name: str) -> None:
    fig.savefig(
        output_dir / f"{name}.png",
        dpi=180,
        bbox_inches="tight",
        facecolor="white",
    )
    fig.savefig(
        output_dir / f"{name}.svg",
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def _make_figures(
    output_dir: Path,
    metrics_frame: pd.DataFrame,
    per_class_frame: pd.DataFrame,
    confusion_matrices: dict[str, np.ndarray],
    agreement: dict[str, Any],
    bootstrap_differences: dict[str, np.ndarray],
) -> None:
    names = [DISPLAY_NAMES[name] for name in CANDIDATE_NAMES]
    macro = metrics_frame.set_index("candidate").loc[list(CANDIDATE_NAMES)]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(names, macro["macro_f1"], color=["#4477aa", "#cc6677", "#228833", "#66ccee", "#aa3377"])
    ax.set_ylabel("Validation macro-F1")
    ax.set_ylim(0, 1)
    ax.set_title("Fixed ensemble candidates — validation macro-F1")
    ax.tick_params(axis="x", rotation=20)
    for bar, value in zip(bars, macro["macro_f1"]):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.01, f"{value:.3f}", ha="center")
    fig.tight_layout()
    _save_figure(fig, output_dir, "macro_f1_comparison")

    fig, ax = plt.subplots(figsize=(12, 6))
    x = np.arange(len(METRICS))
    width = 0.15
    for index, candidate in enumerate(CANDIDATE_NAMES):
        values = macro.loc[candidate, list(METRICS)].to_numpy(dtype=float)
        ax.bar(x + (index - 2) * width, values, width, label=DISPLAY_NAMES[candidate])
    ax.set_xticks(x, [metric.replace("_", " ") for metric in METRICS], rotation=15, ha="right")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1)
    ax.set_title("Validation metrics by model and fixed ensemble")
    ax.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    _save_figure(fig, output_dir, "all_metrics_comparison")

    for metric, filename, title in (
        ("f1", "per_class_f1_comparison", "Per-class validation F1"),
        ("recall", "per_class_recall_comparison", "Per-class validation recall"),
    ):
        fig, ax = plt.subplots(figsize=(12, 6))
        x = np.arange(len(CLASSES))
        width = 0.15
        for index, candidate in enumerate(CANDIDATE_NAMES):
            column = f"{candidate}_{metric}"
            ax.bar(x + (index - 2) * width, per_class_frame[column], width, label=DISPLAY_NAMES[candidate])
        ax.set_xticks(x, CLASSES, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel(metric.capitalize())
        ax.set_title(title)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        _save_figure(fig, output_dir, filename)

    for candidate in CANDIDATE_NAMES:
        matrix = confusion_matrices[candidate]
        fig, ax = plt.subplots(figsize=(8, 7))
        image = ax.imshow(matrix, cmap="Blues")
        ax.set_xticks(np.arange(len(CLASSES)), CLASSES, rotation=35, ha="right")
        ax.set_yticks(np.arange(len(CLASSES)), CLASSES)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(f"Confusion matrix — {DISPLAY_NAMES[candidate]}")
        threshold = matrix.max() / 2 if matrix.size else 0
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                ax.text(
                    column,
                    row,
                    str(int(matrix[row, column])),
                    ha="center",
                    va="center",
                    color="white" if matrix[row, column] > threshold else "black",
                    fontsize=8,
                )
        fig.colorbar(image, ax=ax)
        fig.tight_layout()
        _save_figure(fig, output_dir, f"confusion_matrix_{candidate}")

    partition = agreement["correctness_partition"]
    categories = [
        "both_correct",
        "both_incorrect",
        "svm_correct_hing_roberta_incorrect",
        "hing_roberta_correct_svm_incorrect",
    ]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.bar(
        [category.replace("_", "\n") for category in categories],
        [partition[category] for category in categories],
        color=["#228833", "#aa3377", "#4477aa", "#cc6677"],
    )
    ax.set_ylabel("Validation examples")
    ax.set_title(
        "SVM and HingRoBERTa agreement outcomes "
        f"(prediction disagreement: {agreement['prediction_disagreement_percentage']:.1f}%)"
    )
    fig.tight_layout()
    _save_figure(fig, output_dir, "model_agreement")

    recovery_candidates = CANDIDATE_NAMES[1:]
    recovered = []
    introduced = []
    for candidate in recovery_candidates:
        rows = agreement["per_class_error_recovery_vs_svm"][candidate]
        recovered.append(sum(row["errors_recovered"] for row in rows))
        introduced.append(sum(row["new_errors"] for row in rows))
    x = np.arange(len(recovery_candidates))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width / 2, recovered, width, label="SVM errors corrected", color="#228833")
    ax.bar(x + width / 2, introduced, width, label="New errors vs SVM", color="#cc6677")
    ax.set_xticks(x, [DISPLAY_NAMES[name] for name in recovery_candidates], rotation=15, ha="right")
    ax.set_ylabel("Validation examples")
    ax.set_title("Error recovery relative to the original SVM")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, output_dir, "error_recovery_comparison")

    fig, ax = plt.subplots(figsize=(10, 5))
    for candidate, differences in bootstrap_differences.items():
        ax.hist(
            differences,
            bins=40,
            alpha=0.45,
            density=True,
            label=DISPLAY_NAMES[candidate],
        )
    ax.axvline(0, color="black", linewidth=1)
    ax.set_xlabel("Paired validation macro-F1 difference vs original SVM")
    ax.set_ylabel("Density")
    ax.set_title("Exploratory paired bootstrap distributions")
    ax.legend(fontsize=8)
    fig.tight_layout()
    _save_figure(fig, output_dir, "paired_bootstrap_macro_f1_differences")


def _comparison_frames(
    labels: np.ndarray,
    predictions: dict[str, np.ndarray],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, np.ndarray]]:
    metric_rows = []
    per_class_by_model: dict[str, pd.DataFrame] = {}
    confusion_matrices: dict[str, np.ndarray] = {}
    for candidate in CANDIDATE_NAMES:
        metrics, per_class, matrix = evaluate_predictions(labels, predictions[candidate])
        metric_rows.append({"candidate": candidate, **metrics})
        per_class_by_model[candidate] = per_class
        confusion_matrices[candidate] = matrix
    metrics_frame = pd.DataFrame(metric_rows)
    baseline_values = {
        metric: float(
            metrics_frame.loc[metrics_frame["candidate"] == SVM_NAME, metric].iloc[0]
        )
        for metric in METRICS
    }
    for metric in METRICS:
        metrics_frame[f"delta_{metric}_vs_original_svm"] = (
            metrics_frame[metric].astype(float) - baseline_values[metric]
        )

    per_class_rows = []
    for label_id, label_name in enumerate(CLASSES):
        row: dict[str, Any] = {
            "label_id": label_id,
            "label": label_name,
            "validation_support": int(
                per_class_by_model[SVM_NAME].iloc[label_id]["support"]
            ),
        }
        for candidate in CANDIDATE_NAMES:
            metrics = per_class_by_model[candidate].iloc[label_id]
            for metric in ("precision", "recall", "f1"):
                row[f"{candidate}_{metric}"] = float(metrics[metric])
            row[f"{candidate}_support"] = int(metrics["support"])
        for metric in ("precision", "recall", "f1"):
            for candidate in CANDIDATE_NAMES:
                row[f"{candidate}_delta_{metric}_vs_original_svm"] = float(
                    per_class_by_model[candidate].iloc[label_id][metric]
                    - per_class_by_model[SVM_NAME].iloc[label_id][metric]
                )
        per_class_rows.append(row)
    per_class_frame = pd.DataFrame(per_class_rows)
    details = {
        candidate: {
            "metrics": {
                key: value
                for key, value in metrics_frame.set_index("candidate")
                .loc[candidate]
                .to_dict()
                .items()
                if not str(key).startswith("delta_")
            },
            "per_class": per_class_by_model[candidate].to_dict(orient="records"),
            "confusion_matrix": confusion_matrices[candidate].tolist(),
        }
        for candidate in CANDIDATE_NAMES
    }
    return metrics_frame, per_class_frame, details, confusion_matrices


def run_evaluation(
    root: Path,
    *,
    bootstrap_repeats: int = DEFAULT_BOOTSTRAP_REPEATS,
    overwrite: bool = False,
) -> dict[str, Any]:
    root = Path(root).expanduser().resolve()
    if bootstrap_repeats < 100:
        raise ValueError("Use at least 100 paired bootstrap resamples.")
    output_dir = _output_dir(root)
    _check_output_directory(output_dir, overwrite)

    validation, probabilities, alignment, fingerprint, optimization_context = (
        load_verified_predictions(root)
    )
    print("[3/4] Computing fixed ensemble predictions and validation metrics.", flush=True)
    labels = validation["label"].to_numpy(dtype=int)
    _, predictions, definitions = build_candidate_predictions(
        probabilities[SVM_NAME],
        probabilities[ROBERTA_NAME],
    )
    metrics_frame, per_class_frame, candidate_details, confusion_matrices = (
        _comparison_frames(labels, predictions)
    )
    prior_baseline_f1 = optimization_context.get("original_svm_macro_f1")
    if optimization_context["available"] and prior_baseline_f1 is not None:
        current_baseline_f1 = float(
            metrics_frame.loc[metrics_frame["candidate"] == SVM_NAME, "macro_f1"].iloc[0]
        )
        if not np.isclose(
            current_baseline_f1,
            float(prior_baseline_f1),
            rtol=0.0,
            atol=1e-6,
        ):
            raise RuntimeError(
                "The original SVM validation macro-F1 differs from the saved SVM "
                "optimization report; refusing to combine inconsistent results."
            )
        optimization_context["original_svm_metric_matches_saved_predictions"] = True
    agreement = build_model_agreement(labels, predictions)

    bootstrap_results: dict[str, Any] = {}
    bootstrap_differences: dict[str, np.ndarray] = {}
    for candidate in CANDIDATE_NAMES[2:]:
        result, differences = paired_bootstrap_macro_f1(
            labels,
            predictions[SVM_NAME],
            predictions[candidate],
            repeats=bootstrap_repeats,
            seed=SEED,
        )
        bootstrap_results[candidate] = result
        bootstrap_differences[candidate] = differences
        print(
            f"Bootstrap {candidate}: 95% CI "
            f"{result['confidence_interval_95_percentile']}",
            flush=True,
        )

    print("[4/4] Writing tables, reports, and visualizations.", flush=True)
    _check_output_directory(output_dir, overwrite=True)
    metrics_frame.to_csv(output_dir / "ensemble_comparison.csv", index=False)
    per_class_frame.to_csv(output_dir / "per_class_ensemble_comparison.csv", index=False)
    (output_dir / "model_agreement.json").write_text(
        json.dumps(_json_safe(agreement), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    report: dict[str, Any] = {
        "task": TASK,
        "dataset_source": _read_json(root / "results" / "data_audit.json")
        .get(TASK, {})
        .get("source", "not recorded"),
        "seed": SEED,
        "class_mapping": {str(index): name for index, name in enumerate(CLASSES)},
        "sample_counts": {
            "validation": int(len(validation)),
            "training_rows_loaded": False,
            "test_rows_loaded": False,
            "test_labels_inspected": False,
        },
        "reproducibility": {
            "numpy_version": np.__version__,
            "pandas_version": pd.__version__,
            "sklearn_version": sklearn.__version__,
            "python_version": platform.python_version(),
            "stored_project_split_sha256": fingerprint,
            "split_fingerprint_source": "results/data_audit.json; matched against both model metadata records",
            "fingerprint_recomputed_from_test_data": False,
            "class_order": CLASSES,
            "validation_alignment": alignment,
        },
        "model_artifacts": {
            "models": list(MODEL_NAMES),
            "predictions_loaded_from": {
                model_name: str(
                    root
                    / "models"
                    / TASK
                    / model_name
                    / "validation_probabilities.npz"
                )
                for model_name in MODEL_NAMES
            },
            "saved_model_class_order_verified": True,
            "saved_predictions_recomputed_against_validation_order": True,
            "prior_svm_optimization_report": optimization_context,
        },
        "ensemble_definitions": definitions,
        "candidate_results": candidate_details,
        "model_agreement": agreement,
        "paired_bootstrap_vs_original_svm": bootstrap_results,
        "interpretation": {
            "weights_were_searched": False,
            "validation_selected_ensemble_independently_confirmed": False,
            "statistical_significance_claimed": False,
            "notes": [
                "Only the three prespecified ensemble rules were evaluated; no ensemble weights were tuned.",
                "All predictions use multiclass argmax or the specified hard-vote rule; no binary threshold was applied.",
                "A paired bootstrap interval containing zero does not establish a reliable improvement or decline.",
                "The hard vote has two voters, so disagreements are ties and are resolved in favor of the original SVM.",
                "The original SVM artifacts were read only and were not modified.",
                "Only validation data and saved model artifacts were accessed; test data and labels were not loaded.",
                "No raw-text example-level records were exported.",
            ],
        },
        "output_directory": str(output_dir),
        "visualizations": [
            f"{name}.{extension}"
            for name in FIGURE_NAMES
            for extension in ("png", "svg")
        ],
    }
    _make_figures(
        output_dir,
        metrics_frame,
        per_class_frame,
        confusion_matrices,
        agreement,
        bootstrap_differences,
    )
    (output_dir / "ensemble_report.json").write_text(
        json.dumps(_json_safe(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _write_markdown_report(output_dir, report, metrics_frame, bootstrap_results)
    print(f"Ensemble evaluation complete: {output_dir}", flush=True)
    return report


def _write_markdown_report(
    output_dir: Path,
    report: dict[str, Any],
    metrics_frame: pd.DataFrame,
    bootstrap_results: dict[str, Any],
) -> None:
    lines = [
        "# Cyberbullying7 fixed ensemble evaluation",
        "",
        "- Dataset: MC-Hinglish1.0",
        f"- Validation examples: {report['sample_counts']['validation']}",
        f"- Seed: {SEED}",
        f"- Split fingerprint: `{report['reproducibility']['stored_project_split_sha256']}`",
        "- Test data/labels loaded: no",
        "- FINALIZE: false",
        "- Raw-text example rows exported: no",
        "",
        "## Validation metrics",
        "",
        "| Candidate | Accuracy | Macro-F1 | Weighted-F1 | Macro precision | Macro recall | Balanced accuracy |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics_frame.to_dict(orient="records"):
        lines.append(
            f"| {DISPLAY_NAMES[row['candidate']]} | {row['accuracy']:.4f} | "
            f"{row['macro_f1']:.4f} | {row['weighted_f1']:.4f} | "
            f"{row['macro_precision']:.4f} | {row['macro_recall']:.4f} | "
            f"{row['balanced_accuracy']:.4f} |"
        )
    prior_optimization = report["model_artifacts"]["prior_svm_optimization_report"]
    if prior_optimization["available"]:
        char_score = prior_optimization["char_3_5_validation_macro_f1"]
        char_score_text = f"{char_score:.4f}" if char_score is not None else "not found"
        lines.extend(
            [
                "",
                "## Prior SVM optimization context",
                "",
                f"- Original SVM macro-F1 in prior report: "
                f"{prior_optimization['original_svm_macro_f1']}.",
                f"- Optimized SVM macro-F1 in prior report: "
                f"{prior_optimization['optimized_svm_macro_f1']}.",
                f"- Character-only (3, 5) candidate validation macro-F1 in prior report: "
                f"{char_score_text}.",
            ]
        )
    else:
        lines.extend(
            [
                "",
                "## Prior SVM optimization context",
                "",
                "No matching SVM optimization JSON report was found; this evaluation uses the verified saved SVM baseline.",
            ]
        )
    lines.extend(["", "## Paired bootstrap vs original SVM", ""])
    for candidate, result in bootstrap_results.items():
        low, high = result["confidence_interval_95_percentile"]
        crossing = "includes zero; no clear direction is supported by this interval" if result[
            "interval_crosses_zero"
        ] else "does not include zero; still exploratory and not independent confirmation"
        lines.append(
            f"- **{DISPLAY_NAMES[candidate]}**: mean macro-F1 difference "
            f"{result['mean_macro_f1_difference']:.4f}; 95% paired percentile interval "
            f"[{low:.4f}, {high:.4f}] ({crossing})."
        )
    lines.extend(
        [
            "",
            "## Complementary errors",
            "",
            f"- Prediction disagreement: {report['model_agreement']['prediction_disagreement_count']} / "
            f"{report['model_agreement']['sample_count']} "
            f"({report['model_agreement']['prediction_disagreement_percentage']:.2f}%).",
            f"- Both correct: {report['model_agreement']['correctness_partition']['both_correct']}.",
            f"- Both incorrect: {report['model_agreement']['correctness_partition']['both_incorrect']}.",
            f"- SVM correct, HingRoBERTa incorrect: "
            f"{report['model_agreement']['correctness_partition']['svm_correct_hing_roberta_incorrect']}.",
            f"- HingRoBERTa correct, SVM incorrect: "
            f"{report['model_agreement']['correctness_partition']['hing_roberta_correct_svm_incorrect']}.",
            "",
            "### Offensive, abusive, and mockery",
            "",
            "| Class | Candidate | Precision | Recall | F1 | Errors recovered vs SVM | New errors vs SVM |",
            "|---|---|---:|---:|---:|---:|---:|",
        ]
    )
    recovery_by_candidate = report["model_agreement"]["per_class_error_recovery_vs_svm"]
    for label_name in ("offensive", "abusive", "mockery"):
        for candidate in CANDIDATE_NAMES:
            class_row = next(
                row
                for row in report["candidate_results"][candidate]["per_class"]
                if row["label"] == label_name
            )
            recovery_row = (
                next(row for row in recovery_by_candidate[candidate] if row["label"] == label_name)
                if candidate in recovery_by_candidate
                else {"errors_recovered": 0, "new_errors": 0}
            )
            lines.append(
                f"| {label_name} | {DISPLAY_NAMES[candidate]} | "
                f"{class_row['precision']:.4f} | {class_row['recall']:.4f} | "
                f"{class_row['f1']:.4f} | {recovery_row['errors_recovered']} | "
                f"{recovery_row['new_errors']} |"
            )
    lines.extend(
        [
            "",
            "All per-class metrics and confusion matrices are also available in the JSON/CSV outputs.",
            "",
            "## Interpretation limits",
            "",
            "These are prespecified exploratory comparisons on the existing validation set. "
            "No mixture weights were searched, and no validation-selected ensemble is "
            "independently confirmed. The paired bootstrap is descriptive, not a significance claim. "
            "The two-model hard vote resolves every disagreement in favor of the SVM, so its "
            "predictions equal the original SVM predictions.",
            "",
            "The original model checkpoints and validation prediction files were read only. "
            "No test data or labels were loaded, and no raw-text example-level exports were created.",
        ]
    )
    (output_dir / "ensemble_report.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--bootstrap-repeats", type=int, default=DEFAULT_BOOTSTRAP_REPEATS)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    run_evaluation(
        args.root,
        bootstrap_repeats=args.bootstrap_repeats,
        overwrite=args.overwrite,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
