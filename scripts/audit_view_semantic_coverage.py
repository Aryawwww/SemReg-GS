"""Audit and select donor/train/held-out views by semantic pixel coverage."""

from __future__ import annotations

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--donor-candidates", type=Path, required=True)
    parser.add_argument("--target-candidates", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--donor-count", type=int, default=3)
    parser.add_argument("--heldout-count", type=int, default=2)
    parser.add_argument("--minimum-pixels", type=int, default=1000)
    parser.add_argument("--palette-tolerance", type=float, default=40.0)
    parser.add_argument("--allow-incomplete", action="store_true")
    return parser.parse_args()


def load_classes(path: Path) -> tuple[list[str], np.ndarray]:
    mapping = json.loads(path.resolve().read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    palette = np.asarray([spec["color"][:3] for _, spec in ordered], dtype=np.float32) * 255.0
    return names, palette


def audit_directory(root: Path, names: list[str], palette: np.ndarray, tolerance: float) -> list[dict]:
    records = []
    for directory in sorted(path for path in root.resolve().iterdir() if path.is_dir()):
        semantic_path = directory / "semantic.png"
        camera_path = directory / "camera.json"
        if not semantic_path.is_file() or not camera_path.is_file():
            raise RuntimeError(f"{directory}: missing semantic.png or camera.json")
        image = np.asarray(Image.open(semantic_path).convert("RGB"), dtype=np.float32)
        distance = np.linalg.norm(image[..., None, :] - palette[None, None, :, :], axis=-1)
        labels = distance.argmin(axis=-1)
        valid = distance.min(axis=-1) <= tolerance
        counts = {name: int((valid & (labels == class_id)).sum()) for class_id, name in enumerate(names)}
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        records.append({
            "name": directory.name,
            "room": camera.get("room"),
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "valid_semantic_pixels": int(valid.sum()),
            "class_pixels": counts,
        })
    if not records:
        raise RuntimeError(f"No candidate view directories found in {root}")
    return records


def combined_counts(records: list[dict], indices: tuple[int, ...], names: list[str]) -> dict[str, int]:
    return {name: sum(records[index]["class_pixels"][name] for index in indices) for name in names}


def subset_score(counts: dict[str, int], minimum: int) -> tuple[int, float, float]:
    covered = sum(value >= minimum for value in counts.values())
    minimum_ratio = min(value / minimum for value in counts.values())
    balanced_total = sum(math.log1p(value) for value in counts.values())
    return covered, minimum_ratio, balanced_total


def best_subset(records: list[dict], count: int, names: list[str], minimum: int) -> tuple[tuple[int, ...], dict[str, int], tuple[int, float, float]]:
    if count <= 0 or count > len(records):
        raise ValueError(f"Requested {count} views from {len(records)} candidates")
    best = None
    for indices in itertools.combinations(range(len(records)), count):
        counts = combined_counts(records, indices, names)
        candidate = (subset_score(counts, minimum), indices, counts)
        if best is None or candidate[0] > best[0]:
            best = candidate
    score, indices, counts = best
    return indices, counts, score


def main() -> None:
    args = arguments()
    names, palette = load_classes(args.semantic_mapping)
    donor_records = audit_directory(args.donor_candidates, names, palette, args.palette_tolerance)
    target_records = audit_directory(args.target_candidates, names, palette, args.palette_tolerance)
    donor_indices, donor_counts, donor_score = best_subset(donor_records, args.donor_count, names, args.minimum_pixels)
    heldout_indices, heldout_counts, heldout_score = best_subset(target_records, args.heldout_count, names, args.minimum_pixels)
    heldout_set = set(heldout_indices)
    train_indices = tuple(index for index in range(len(target_records)) if index not in heldout_set)
    train_counts = combined_counts(target_records, train_indices, names)
    donor_missing = [name for name in names if donor_counts[name] < args.minimum_pixels]
    train_missing = [name for name in names if train_counts[name] < args.minimum_pixels]
    heldout_missing = [name for name in names if heldout_counts[name] < args.minimum_pixels]
    errors = []
    if donor_missing:
        errors.append("donor below threshold: " + ", ".join(donor_missing))
    if train_missing:
        errors.append("train below threshold: " + ", ".join(train_missing))
    if heldout_missing:
        errors.append("heldout below threshold: " + ", ".join(heldout_missing))
    report = {
        "status": "passed" if not errors else "failed",
        "minimum_pixels_per_class": args.minimum_pixels,
        "semantic_classes": names,
        "selection": {
            "donor": [donor_records[index]["name"] for index in donor_indices],
            "train": [target_records[index]["name"] for index in train_indices],
            "heldout": [target_records[index]["name"] for index in heldout_indices],
        },
        "combined_class_pixels": {
            "donor": donor_counts,
            "train": train_counts,
            "heldout": heldout_counts,
        },
        "selection_scores": {
            "donor": {"covered_classes": donor_score[0], "minimum_threshold_ratio": donor_score[1], "balanced_total": donor_score[2]},
            "heldout": {"covered_classes": heldout_score[0], "minimum_threshold_ratio": heldout_score[1], "balanced_total": heldout_score[2]},
        },
        "missing_or_under_threshold": {
            "donor": donor_missing,
            "train": train_missing,
            "heldout": heldout_missing,
        },
        "donor_candidates": donor_records,
        "target_candidates": target_records,
        "errors": errors,
        "next_action": "Add or reposition cameras for missing classes, rerender, and rerun this audit." if errors else "Freeze this split manifest before training.",
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "donor": report["selection"]["donor"],
        "train": report["selection"]["train"],
        "heldout": report["selection"]["heldout"],
        "missing": report["missing_or_under_threshold"],
    }))
    if errors and not args.allow_incomplete:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
