#!/usr/bin/env python3
"""Audit prepared MixBhasha v5 cyberbullying training data without editing it.

Aggregate reports omit raw text. Row-level text examples are written only to
the experiment's private_analysis directory.
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import re
import sys
import unicodedata
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
import mixbhasha_core as core

TASK = "cyberbullying7"
CLASSES = list(core.LABELS[TASK])
FOCUS_CLASSES = {"mockery", "abusive", "offensive"}
DEFAULT_ROOT = Path("/content/drive/MyDrive/MixBhasha_V5_seed42")
SIMILARITY_THRESHOLD = 0.80
SIMILARITY_DESCRIPTION = (
    "Unsupervised character n-gram nearest-neighbor comparison between "
    "different focus labels; cosine similarity >= 0.80 is a heuristic only."
)
REPEATED_CHARACTER_RE = re.compile(r"([^\s])\1{3,}", re.UNICODE)
REPEATED_PUNCTUATION_RE = re.compile(r"[!?！？]{2,}", re.UNICODE)
SCRIPT_MARKERS = (
    ("Latin", "LATIN"),
    ("Devanagari", "DEVANAGARI"),
    ("Arabic", "ARABIC"),
    ("Bengali", "BENGALI"),
    ("Gurmukhi", "GURMUKHI"),
    ("Tamil", "TAMIL"),
    ("Telugu", "TELUGU"),
    ("Kannada", "KANNADA"),
    ("Malayalam", "MALAYALAM"),
    ("Gujarati", "GUJARATI"),
    ("Greek", "GREEK"),
    ("Cyrillic", "CYRILLIC"),
    ("Hebrew", "HEBREW"),
    ("Thai", "THAI"),
    ("CJK", "CJK"),
)


def _require_file(path: Path, description: str) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {description}: {path}. Run the existing V5 data-preparation "
            "cell for cyberbullying7 first; this audit will not download, "
            "re-split, or modify source data."
        )


def _read_text_only(path: Path, split: str) -> pd.DataFrame:
    _require_file(path, f"{split} split CSV")
    columns = pd.read_csv(path, nrows=0).columns
    if "text" not in columns:
        raise ValueError(f"{split} split CSV must contain a 'text' column: {path}")
    # Deliberately do not request the label column for either holdout split.
    return pd.read_csv(
        path,
        usecols=["text"],
        dtype={"text": str},
        keep_default_na=False,
    )


def _load_project_fingerprint(root: Path) -> tuple[str, str]:
    path = root / "results" / "data_audit.json"
    _require_file(path, "existing V5 data audit / split fingerprint")
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by task name.")
    task_record = record.get(TASK)
    if not isinstance(task_record, dict):
        raise ValueError(
            f"{path} has no {TASK!r} entry. Run the existing data-preparation "
            "cell for cyberbullying7 first."
        )
    fingerprint = task_record.get("split_sha256")
    if not isinstance(fingerprint, str) or not re.fullmatch(r"[0-9a-fA-F]{64}", fingerprint):
        raise ValueError(f"{path} has no valid stored {TASK} split_sha256.")
    return fingerprint, str(task_record.get("source", "not recorded"))


def _check_model_fingerprints(root: Path, expected: str) -> list[str]:
    model_root = root / "models" / TASK
    if not model_root.is_dir():
        return []
    checked = []
    for metadata_path in sorted(model_root.glob("*/validated.json")):
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        observed = metadata.get("meta", {}).get("split_sha256")
        if observed is not None and observed != expected:
            raise RuntimeError(
                f"Stored split fingerprint mismatch in {metadata_path}: "
                f"{observed} != {expected}. Stop and verify the experiment files."
            )
        if observed is not None:
            checked.append(metadata_path.parent.name)
    return checked


def _stable_digest(values: list[str]) -> str:
    digest = hashlib.sha256()
    for value in values:
        encoded = value.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _text_label_digest(frame: pd.DataFrame) -> str:
    values = []
    for text, label in frame[["text", "label"]].itertuples(index=False, name=None):
        values.extend((str(text), str(int(label))))
    return _stable_digest(values)


def _duplicate_metrics(keys: list[str]) -> dict[str, int]:
    sizes = pd.Series(keys, dtype="object").value_counts(dropna=False)
    repeated = sizes[sizes > 1]
    return {
        "duplicate_groups": int(len(repeated)),
        "affected_rows": int(repeated.sum()),
        "extra_rows": int((repeated - 1).sum()),
    }


def _normalized_keys(texts: list[str]) -> list[str]:
    # Reuse the project's text normalization and split-cleaning key.
    return [core.clean(value).casefold().strip() for value in texts]


def _conflict_distribution(train: pd.DataFrame) -> pd.DataFrame:
    rows = []
    labels = train["label"].astype(int)
    texts = train["text"].astype(str)
    for match_type, keys in (
        ("exact", texts.tolist()),
        ("normalized", _normalized_keys(texts.tolist())),
    ):
        grouped = pd.DataFrame({"key": keys, "label": labels.to_numpy()}).groupby(
            "key", sort=False, dropna=False
        )
        totals: dict[tuple[int, int], dict[str, int]] = {}
        for _, group in grouped:
            group_labels = sorted(int(value) for value in group["label"].unique())
            if len(group_labels) < 2:
                continue
            for label_a, label_b in itertools.combinations(group_labels, 2):
                counts = totals.setdefault((label_a, label_b), {"groups": 0, "rows": 0})
                counts["groups"] += 1
                counts["rows"] += len(group)
        for (label_a, label_b), counts in sorted(totals.items()):
            rows.append(
                {
                    "match_type": match_type,
                    "label_a": CLASSES[label_a],
                    "label_b": CLASSES[label_b],
                    "conflicting_groups": counts["groups"],
                    "affected_rows": counts["rows"],
                }
            )
    return pd.DataFrame(
        rows,
        columns=[
            "match_type",
            "label_a",
            "label_b",
            "conflicting_groups",
            "affected_rows",
        ],
    )


def _overlap_metrics(left: list[str], right: list[str]) -> dict[str, int]:
    left_counts = pd.Series(left, dtype="object").value_counts(dropna=False)
    right_counts = pd.Series(right, dtype="object").value_counts(dropna=False)
    common = set(left_counts.index).intersection(right_counts.index)
    return {
        "shared_unique_texts": len(common),
        "affected_rows_left": int(sum(left_counts[key] for key in common)),
        "affected_rows_right": int(sum(right_counts[key] for key in common)),
        "matching_row_pairs": int(
            sum(left_counts[key] * right_counts[key] for key in common)
        ),
    }


def _text_length_stats(values: list[int]) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.int64)
    if not len(array):
        return {
            "count": 0,
            "min": 0,
            "max": 0,
            "mean": 0.0,
            "median": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p90": 0.0,
        }
    return {
        "count": int(len(array)),
        "min": int(array.min()),
        "max": int(array.max()),
        "mean": round(float(array.mean()), 3),
        "median": round(float(np.median(array)), 3),
        "p25": round(float(np.quantile(array, 0.25)), 3),
        "p75": round(float(np.quantile(array, 0.75)), 3),
        "p90": round(float(np.quantile(array, 0.90)), 3),
    }


def _script_names(text: str) -> list[str]:
    scripts = set()
    for character in text:
        if not character.isalpha():
            continue
        name = unicodedata.name(character, "")
        found = next((label for label, marker in SCRIPT_MARKERS if marker in name), None)
        scripts.add(found or "Other")
    return sorted(scripts)


def _unusual_characters(text: str) -> list[str]:
    unusual = []
    for character in text:
        category = unicodedata.category(character)
        if character == "\ufffd" or (category.startswith("C") and not character.isspace()):
            unusual.append(character)
    return unusual


def _nearest_cross_label_neighbors(train: pd.DataFrame) -> dict[int, tuple[int, float]]:
    """Find cross-label character n-gram neighbors for the three boundary classes."""
    focus_ids = [index for index, label in enumerate(CLASSES) if label in FOCUS_CLASSES]
    source_rows = {
        label_id: np.flatnonzero(train["label"].to_numpy(dtype=int) == label_id)
        for label_id in focus_ids
    }
    nonempty = [rows for rows in source_rows.values() if len(rows)]
    if sum(len(rows) for rows in nonempty) < 2 or len(nonempty) < 2:
        return {}
    if not any(len(str(text)) >= 3 for text in train["text"]):
        return {}

    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(3, 5),
        min_df=1,
        max_features=50000,
        sublinear_tf=True,
        dtype=np.float32,
    )
    matrix = vectorizer.fit_transform(train["text"].astype(str).tolist())
    nearest: dict[int, tuple[int, float]] = {}
    for source_label, source_indices in source_rows.items():
        for target_label, target_indices in source_rows.items():
            if source_label == target_label or not len(target_indices):
                continue
            model = NearestNeighbors(n_neighbors=1, metric="cosine", algorithm="brute")
            model.fit(matrix[target_indices])
            for start in range(0, len(source_indices), 512):
                batch = source_indices[start : start + 512]
                distances, positions = model.kneighbors(matrix[batch])
                for source_index, distance, position in zip(
                    batch, distances[:, 0], positions[:, 0]
                ):
                    similarity = max(0.0, 1.0 - float(distance))
                    target_index = int(target_indices[int(position)])
                    previous = nearest.get(int(source_index))
                    if previous is None or similarity > previous[1]:
                        nearest[int(source_index)] = (target_index, similarity)
    return nearest


def _analyze_training_text(train: pd.DataFrame) -> tuple[dict, list[dict]]:
    texts = train["text"].astype(str).tolist()
    labels = train["label"].astype(int).tolist()
    word_counts = [len(text.split()) for text in texts]
    lengths = [len(text) for text in texts]
    repeated = [bool(REPEATED_CHARACTER_RE.search(text)) for text in texts]
    unusual = [_unusual_characters(text) for text in texts]
    scripts = [_script_names(text) for text in texts]

    quality = {
        "empty_text_rows": sum(not text.strip() for text in texts),
        "near_empty_text_rows_le_2_chars": sum(0 < len(text.strip()) <= 2 for text in texts),
        "unusual_unicode_rows": sum(bool(chars) for chars in unusual),
        "unusual_unicode_characters": sum(len(chars) for chars in unusual),
        "replacement_character_rows": sum("\ufffd" in text for text in texts),
        "potentially_corrupted_rows": sum(
            bool(unusual[index]) or len(text) > 4000
            for index, text in enumerate(texts)
        ),
        "repeated_character_rows_4plus": sum(repeated),
        "mixed_script_rows": sum(len(found) > 1 for found in scripts),
        "script_row_counts": {},
        "character_length": _text_length_stats(lengths),
        "whitespace_word_count": _text_length_stats(word_counts),
        "word_count_bins": {
            "0": sum(count == 0 for count in word_counts),
            "1-3": sum(1 <= count <= 3 for count in word_counts),
            "4-10": sum(4 <= count <= 10 for count in word_counts),
            "11-25": sum(11 <= count <= 25 for count in word_counts),
            "26+": sum(count >= 26 for count in word_counts),
        },
    }
    script_counts: dict[str, int] = {}
    for found in scripts:
        for script in found:
            script_counts[script] = script_counts.get(script, 0) + 1
    quality["script_row_counts"] = dict(sorted(script_counts.items()))

    nearest = _nearest_cross_label_neighbors(train)
    examples = []
    for index, (text, label, word_count) in enumerate(zip(texts, labels, word_counts)):
        reasons = []
        class_name = CLASSES[label]
        if len(text.strip()) <= 2:
            reasons.append("empty_or_near_empty_text")
        if repeated[index]:
            reasons.append("repeated_character_run_4plus")
        if unusual[index]:
            reasons.append("unusual_unicode_control_format_or_replacement")
        if len(scripts[index]) > 1:
            reasons.append("mixed_script")
        if class_name in FOCUS_CLASSES:
            if word_count <= 3:
                reasons.append("short_context_le_3_words")
            if REPEATED_PUNCTUATION_RE.search(text):
                reasons.append("repeated_exclamation_or_question_punctuation")
            neighbor = nearest.get(index)
            if neighbor is not None and neighbor[1] >= SIMILARITY_THRESHOLD:
                neighbor_index, similarity = neighbor
                reasons.append("high_character_similarity_to_other_focus_label")
            else:
                neighbor_index, similarity = None, None
        else:
            neighbor_index, similarity = None, None
        if reasons:
            examples.append(
                {
                    "training_row_index": index,
                    "label_id": label,
                    "label": class_name,
                    "text": text,
                    "character_count": len(text),
                    "word_count": word_count,
                    "scripts": "|".join(scripts[index]),
                    "warning_reasons": "|".join(reasons),
                    "nearest_other_focus_label": (
                        CLASSES[int(train.iloc[neighbor_index]["label"])]
                        if neighbor_index is not None
                        else ""
                    ),
                    "nearest_other_focus_row_index": (
                        neighbor_index if neighbor_index is not None else ""
                    ),
                    "nearest_other_focus_cosine_similarity": (
                        round(similarity, 4) if similarity is not None else ""
                    ),
                    "review_status": "heuristic_warning_not_verified_error",
                }
            )
    return quality, examples


def _save_figure(fig: plt.Figure, path: Path) -> None:
    fig.tight_layout()
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _make_plots(
    report_dir: Path,
    train: pd.DataFrame,
    lengths: list[int],
    duplicate_rows: list[dict],
    overlap: dict,
    conflicts: pd.DataFrame,
    suspicious: list[dict],
) -> None:
    counts = train["label"].value_counts().reindex(range(len(CLASSES)), fill_value=0)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(CLASSES, counts.to_numpy(), color="#4776a8")
    ax.set_xlabel("Training samples")
    ax.set_title(f"Cyberbullying7 training class distribution (n={len(train):,})")
    ax.invert_yaxis()
    _save_figure(fig, report_dir / "class_distribution.png")

    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.hist(lengths, bins=40, color="#4c956c", edgecolor="white")
    ax.set_xlabel("Characters per training text")
    ax.set_ylabel("Samples")
    ax.set_title("Training text length distribution")
    _save_figure(fig, report_dir / "text_length_distribution.png")

    duplicate_labels = [
        f"{row['split']} {row['match_type']}" for row in duplicate_rows
    ]
    duplicate_values = [row["affected_rows"] for row in duplicate_rows]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.barh(duplicate_labels, duplicate_values, color="#9b6aa0")
    ax.set_xlabel("Rows participating in duplicate groups")
    ax.set_title("Exact and normalized duplicate rows by split")
    ax.invert_yaxis()
    _save_figure(fig, report_dir / "duplicate_summary.png")

    pair_names = ["train-validation", "train-test", "validation-test"]
    x = np.arange(len(pair_names))
    width = 0.36
    exact = [overlap[name]["exact"]["shared_unique_texts"] for name in pair_names]
    normalized = [overlap[name]["normalized"]["shared_unique_texts"] for name in pair_names]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.bar(x - width / 2, exact, width, label="Exact text", color="#4477aa")
    ax.bar(x + width / 2, normalized, width, label="Normalized text", color="#ee9944")
    ax.set_xticks(x, pair_names, rotation=15, ha="right")
    ax.set_ylabel("Shared unique text keys")
    ax.set_title("Cross-split text overlap (holdout labels not read)")
    ax.legend()
    _save_figure(fig, report_dir / "train_validation_overlap.png")

    fig, ax = plt.subplots(figsize=(10, 5))
    if conflicts.empty:
        ax.text(0.5, 0.5, "No conflicting duplicate-label groups detected",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
    else:
        conflicts = conflicts.copy()
        conflicts["label_pair"] = conflicts["label_a"] + " / " + conflicts["label_b"]
        pivot = conflicts.pivot_table(
            index="label_pair",
            columns="match_type",
            values="conflicting_groups",
            aggfunc="sum",
            fill_value=0,
        )
        pivot.plot.barh(ax=ax, color={"exact": "#4477aa", "normalized": "#cc6677"})
        ax.set_xlabel("Conflicting duplicate-label groups")
        ax.set_ylabel("Training label pair")
        ax.legend(title="Matching rule")
        ax.invert_yaxis()
    ax.set_title("Potential training label conflicts (not adjudicated)")
    _save_figure(fig, report_dir / "label_conflict_distribution.png")

    suspicious_counts = pd.Series(
        [row["label"] for row in suspicious if row["label"] in FOCUS_CLASSES],
        dtype="object",
    ).value_counts().reindex([name for name in CLASSES if name in FOCUS_CLASSES], fill_value=0)
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.barh(suspicious_counts.index.tolist(), suspicious_counts.to_numpy(), color="#cc6677")
    ax.set_xlabel("Heuristic review candidates")
    ax.set_title("Heuristic warnings in offensive, abusive, and mockery training rows")
    ax.invert_yaxis()
    _save_figure(fig, report_dir / "heuristic_review_candidates.png")


def run(root: Path) -> dict:
    root = Path(root).expanduser()
    split_paths = {
        split: root / "data" / f"{TASK}_{split}.csv"
        for split in ("train", "val", "test")
    }
    for split, path in split_paths.items():
        _require_file(path, f"{split} prepared split")

    train = core.load_prepared(root, [TASK], splits=("train",))[TASK]["train"]
    if not {"text", "label"}.issubset(train.columns):
        raise ValueError("Prepared training split must contain text and integer label columns.")
    if train.empty:
        raise ValueError("Prepared cyberbullying7 training split is empty.")
    if train["text"].isna().any():
        raise ValueError("Prepared training split contains missing text values.")
    invalid_labels = sorted(set(train["label"].astype(int)) - set(range(len(CLASSES))))
    if invalid_labels:
        raise ValueError(
            f"Unexpected training label IDs {invalid_labels}; expected core.LABELS[{TASK!r}] IDs."
        )
    train = train[["text", "label"]].copy()
    train["text"] = train["text"].astype(str)
    train["label"] = train["label"].astype(int)

    # Read only text from both held-out splits. In particular, test labels are
    # never loaded, used for validation, or included in any audit computation.
    validation = _read_text_only(split_paths["val"], "validation")
    test = _read_text_only(split_paths["test"], "test")
    texts = {
        "train": train["text"].tolist(),
        "validation": validation["text"].astype(str).tolist(),
        "test": test["text"].astype(str).tolist(),
    }
    normalized = {name: _normalized_keys(values) for name, values in texts.items()}

    stored_fingerprint, source = _load_project_fingerprint(root)
    fingerprint_models = _check_model_fingerprints(root, stored_fingerprint)

    class_counts = (
        train["label"]
        .value_counts()
        .reindex(range(len(CLASSES)), fill_value=0)
        .astype(int)
    )
    minimum = int(class_counts.min())
    maximum = int(class_counts.max())
    class_distribution = {
        CLASSES[index]: int(class_counts.iloc[index]) for index in range(len(CLASSES))
    }
    lengths = [len(text) for text in texts["train"]]
    word_counts = [len(text.split()) for text in texts["train"]]
    quality, suspicious = _analyze_training_text(train)

    duplicate_rows = []
    for split, values in texts.items():
        for match_type, keys in (
            ("exact", values),
            ("normalized", normalized[split]),
        ):
            duplicate_rows.append(
                {"split": split, "match_type": match_type, **_duplicate_metrics(keys)}
            )
    conflicts = _conflict_distribution(train)
    overlap = {}
    for pair_name, left, right in (
        ("train-validation", "train", "validation"),
        ("train-test", "train", "test"),
        ("validation-test", "validation", "test"),
    ):
        overlap[pair_name] = {
            "exact": _overlap_metrics(texts[left], texts[right]),
            "normalized": _overlap_metrics(normalized[left], normalized[right]),
        }

    warning_counts: dict[str, int] = {}
    warning_by_class: dict[str, int] = {}
    for example in suspicious:
        warning_by_class[example["label"]] = warning_by_class.get(example["label"], 0) + 1
        for reason in example["warning_reasons"].split("|"):
            warning_counts[reason] = warning_counts.get(reason, 0) + 1

    report_dir = root / "reports" / "training_audit"
    private_dir = root / "private_analysis"
    report_dir.mkdir(parents=True, exist_ok=True)
    private_dir.mkdir(parents=True, exist_ok=True)

    duplicate_frame = pd.DataFrame(duplicate_rows)
    duplicate_frame.to_csv(report_dir / "duplicate_summary.csv", index=False)
    conflicts.to_csv(report_dir / "conflicting_labels.csv", index=False)
    pd.DataFrame(
        suspicious,
        columns=[
            "training_row_index",
            "label_id",
            "label",
            "text",
            "character_count",
            "word_count",
            "scripts",
            "warning_reasons",
            "nearest_other_focus_label",
            "nearest_other_focus_row_index",
            "nearest_other_focus_cosine_similarity",
            "review_status",
        ],
    ).to_csv(private_dir / "suspicious_training_examples.csv", index=False)

    summary = {
        "schema_version": 1,
        "task": TASK,
        "source": source,
        "class_mapping": {str(index): name for index, name in enumerate(CLASSES)},
        "dataset_statistics": {
            "training_samples": int(len(train)),
            "samples_per_class": class_distribution,
            "class_imbalance": {
                "max_class": CLASSES[int(class_counts.to_numpy().argmax())],
                "min_class": CLASSES[int(class_counts.to_numpy().argmin())],
                "max_to_min_ratio": round(maximum / minimum, 4) if minimum else None,
            },
            "character_length": quality["character_length"],
            "whitespace_word_count": quality["whitespace_word_count"],
            "word_count_bins": quality["word_count_bins"],
        },
        "duplicates": {
            "within_split": duplicate_rows,
            "conflicting_training_labels": conflicts.to_dict(orient="records"),
        },
        "data_leakage": {
            "matching_rules": {
                "exact": "Stored prepared text strings compared without case or whitespace changes.",
                "normalized": "Existing core clean() preprocessing followed by casefold and strip.",
            },
            "pairwise_text_overlap": overlap,
            "test_labels_inspected": False,
        },
        "text_quality": quality,
        "suspicious_training_labels": {
            "ambiguity_focus_classes": [
                name for name in CLASSES if name in FOCUS_CLASSES
            ],
            "text_quality_flags_apply_to_all_classes": True,
            "heuristic_candidate_count": len(suspicious),
            "by_class": dict(sorted(warning_by_class.items())),
            "warning_reason_counts": dict(sorted(warning_counts.items())),
            "verification_status": "heuristic warnings only; not verified annotation errors",
            "automatic_relabeling_performed": False,
            "similarity_rule": SIMILARITY_DESCRIPTION,
            "similarity_threshold": SIMILARITY_THRESHOLD,
        },
        "reproducibility": {
            "stored_project_split_sha256": stored_fingerprint,
            "fingerprint_source": "results/data_audit.json",
            "fingerprint_recomputed": False,
            "fingerprint_note": (
                "The existing core fingerprint includes test labels. It is reported "
                "from the saved audit record, not recomputed, to honor the no-test-label rule."
            ),
            "training_text_label_sha256": _text_label_digest(train),
            "validation_text_sha256": _stable_digest(texts["validation"]),
            "test_text_sha256": _stable_digest(texts["test"]),
            "validated_model_fingerprints_checked": fingerprint_models,
        },
        "outputs": {
            "aggregate_directory": str(report_dir),
            "private_raw_text_directory": str(private_dir),
            "raw_text_published_in_aggregate_reports": False,
        },
    }
    (report_dir / "training_data_audit.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    markdown = [
        "# MixBhasha v5 — Cyberbullying7 training-data quality audit",
        "",
        f"- Training samples: **{len(train):,}**",
        f"- Stored project split SHA-256: `{stored_fingerprint}`",
        f"- Data source: {source}",
        "- Test labels inspected: **No**",
        "- Automatic relabeling: **None**",
        "",
        "## Training class distribution",
        "",
        "| Class | Samples | Share |",
        "|---|---:|---:|",
    ]
    for name in CLASSES:
        count = class_distribution[name]
        share = count / len(train) * 100 if len(train) else 0
        markdown.append(f"| {name} | {count} | {share:.2f}% |")
    markdown.extend(
        [
            "",
            f"Maximum/minimum class-count ratio: "
            f"{summary['dataset_statistics']['class_imbalance']['max_to_min_ratio']}.",
            "",
            "## Text quality and duplicates",
            "",
            f"- Empty rows: {quality['empty_text_rows']}",
            f"- Near-empty rows (1–2 characters): {quality['near_empty_text_rows_le_2_chars']}",
            f"- Rows containing unusual control, format, private, unassigned, or replacement characters: "
            f"{quality['unusual_unicode_rows']}",
            f"- Rows with repeated character runs of 4+: {quality['repeated_character_rows_4plus']}",
            f"- Rows with mixed detected scripts: {quality['mixed_script_rows']}",
            "",
            "| Split | Match rule | Duplicate groups | Affected rows | Extra rows |",
            "|---|---|---:|---:|---:|",
        ]
    )
    for row in duplicate_rows:
        markdown.append(
            f"| {row['split']} | {row['match_type']} | {row['duplicate_groups']} | "
            f"{row['affected_rows']} | {row['extra_rows']} |"
        )
    markdown.extend(
        [
            "",
            "## Cross-split text overlap",
            "",
            "Counts are computed from text columns only; no validation or test labels are read.",
            "",
            "| Split pair | Match rule | Shared keys | Rows in left | Rows in right | Matching row pairs |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for pair_name, rules in overlap.items():
        for match_type, values in rules.items():
            markdown.append(
                f"| {pair_name} | {match_type} | {values['shared_unique_texts']} | "
                f"{values['affected_rows_left']} | {values['affected_rows_right']} | "
                f"{values['matching_row_pairs']} |"
            )
    markdown.extend(
        [
            "",
            "## Training-label review candidates",
            "",
            f"Flagged training rows: **{len(suspicious)}**. These are **heuristic warnings, "
            "not verified label errors**; no labels were changed.",
            "",
            f"Similarity rule: {SIMILARITY_DESCRIPTION}",
            "",
            "Candidate text rows are saved only in "
            "`private_analysis/suspicious_training_examples.csv`. The aggregate reports "
            "contain no raw-text examples.",
            "",
            "## Reproducibility and cautions",
            "",
            "The stored core split fingerprint is read from `results/data_audit.json`. "
            "It is not recomputed because the existing core fingerprint includes test labels. "
            "Separate digests cover training text+labels and validation/test text only.",
            "",
            "Text length is measured in Unicode code points; word count is whitespace-delimited. "
            "Duplicate normalization follows core.clean() plus the core casefold-and-strip key. Similarity and "
            "text-quality flags are review prompts only, not adjudications.",
            "",
            "![Training class distribution](class_distribution.png)",
            "",
            "![Training text length distribution](text_length_distribution.png)",
            "",
            "![Duplicate summary](duplicate_summary.png)",
            "",
            "![Train-validation and other text overlap](train_validation_overlap.png)",
            "",
            "![Potential label conflicts](label_conflict_distribution.png)",
            "",
            "![Heuristic review candidates](heuristic_review_candidates.png)",
            "",
        ]
    )
    (report_dir / "training_data_audit.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    _make_plots(
        report_dir,
        train,
        lengths,
        duplicate_rows,
        overlap,
        conflicts,
        suspicious,
    )

    print(f"Training-data audit complete: {report_dir}")
    print(f"Private raw-text review candidates: {private_dir / 'suspicious_training_examples.csv'}")
    print(f"Training rows: {len(train):,}; heuristic review candidates: {len(suspicious):,}")
    print("Test labels inspected: no")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit prepared MixBhasha v5 cyberbullying7 training data."
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Existing experiment directory (default: {DEFAULT_ROOT})",
    )
    args = parser.parse_args()
    run(args.root)


if __name__ == "__main__":
    main()
