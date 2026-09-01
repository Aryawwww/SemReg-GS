"""Audit rendered Gaussian semantic IDs against frozen held-out semantic proxies."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--target-views", type=Path, required=True)
    parser.add_argument("--renders", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--visualizations", type=Path, required=True)
    parser.add_argument("--palette-tolerance", type=float, default=40.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(image: np.ndarray, palette: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(image[..., None, :].astype(np.float32) - palette[None, None, :, :], axis=-1)
    return distance.argmin(axis=-1), distance.min(axis=-1) <= tolerance


def class_metrics(confusion: np.ndarray, names: list[str]) -> dict:
    result = {}
    for class_id, name in enumerate(names):
        true_positive = int(confusion[class_id, class_id])
        target_total = int(confusion[class_id].sum())
        rendered_total = int(confusion[:, class_id].sum())
        union = target_total + rendered_total - true_positive
        result[name] = {
            "target_pixels": target_total,
            "rendered_pixels": rendered_total,
            "true_positive_pixels": true_positive,
            "precision": true_positive / rendered_total if rendered_total else None,
            "recall": true_positive / target_total if target_total else None,
            "iou": true_positive / union if union else None,
        }
    return result


def main() -> None:
    args = arguments()
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("protocol is not frozen")
    mapping_path = args.semantic_mapping.resolve()
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    palette = np.asarray([spec["color"][:3] for _, spec in ordered], dtype=np.float32) * 255.0
    target_root = args.target_views.resolve()
    render_root = args.renders.resolve()
    visual_root = args.visualizations.resolve()
    visual_root.mkdir(parents=True, exist_ok=True)
    total_confusion = np.zeros((len(names), len(names)), dtype=np.int64)
    views = []
    for view in protocol["selection"]["target_heldout"]:
        target_path = target_root / view / "semantic.png"
        rendered_path = render_root / view / "semantic.png"
        coverage_path = render_root / view / "coverage.png"
        target = np.asarray(Image.open(target_path).convert("RGB"), dtype=np.uint8)
        rendered = np.asarray(Image.open(rendered_path).convert("RGB"), dtype=np.uint8)
        coverage = np.asarray(Image.open(coverage_path).convert("L"), dtype=np.uint8) > 0
        target_label, target_valid = decode(target, palette, args.palette_tolerance)
        rendered_label, rendered_valid = decode(rendered, palette, args.palette_tolerance)
        valid = coverage & target_valid & rendered_valid
        confusion = np.zeros_like(total_confusion)
        np.add.at(confusion, (target_label[valid], rendered_label[valid]), 1)
        total_confusion += confusion
        mismatch = valid & (target_label != rendered_label)
        image = np.zeros((*valid.shape, 3), dtype=np.uint8)
        image[valid & ~mismatch] = (40, 180, 70)
        image[mismatch] = (230, 45, 45)
        Image.fromarray(image).save(visual_root / f"{view}_semantic_alignment.png")
        views.append({
            "view": view,
            "evaluated_pixels": int(valid.sum()),
            "agreement": float((~mismatch & valid).sum() / valid.sum()) if valid.any() else None,
            "confusion_matrix_rows_target_columns_rendered": confusion.tolist(),
            "classes": class_metrics(confusion, names),
            "target_semantic_sha256": sha256(target_path),
            "rendered_semantic_sha256": sha256(rendered_path),
        })
    off_diagonal = []
    for target_id, target_name in enumerate(names):
        for rendered_id, rendered_name in enumerate(names):
            if target_id != rendered_id and total_confusion[target_id, rendered_id]:
                off_diagonal.append({
                    "target_class": target_name,
                    "rendered_class": rendered_name,
                    "pixels": int(total_confusion[target_id, rendered_id]),
                })
    off_diagonal.sort(key=lambda entry: entry["pixels"], reverse=True)
    total = int(total_confusion.sum())
    report = {
        "schema_version": 1,
        "status": "passed",
        "pair_id": protocol["pair_id"],
        "definition": "Rows are frozen held-out proxy semantics; columns are rendered target-Gaussian semantic IDs.",
        "protocol_manifest_sha256": sha256(protocol_path),
        "semantic_mapping_sha256": sha256(mapping_path),
        "evaluated_pixels": total,
        "overall_agreement": float(np.trace(total_confusion) / total) if total else None,
        "confusion_matrix_rows_target_columns_rendered": total_confusion.tolist(),
        "classes": class_metrics(total_confusion, names),
        "largest_mismatches": off_diagonal[:20],
        "views": views,
        "visualizations": str(visual_root),
        "visualization_legend": {"agreement": [40, 180, 70], "mismatch": [230, 45, 45], "not_evaluated": [0, 0, 0]},
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "agreement": report["overall_agreement"], "largest_mismatches": report["largest_mismatches"][:5]}))


if __name__ == "__main__":
    main()
