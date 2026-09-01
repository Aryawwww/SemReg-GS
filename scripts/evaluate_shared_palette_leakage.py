"""Evaluate all methods against one frozen, method-independent donor palette."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


METHODS = ("global", "semantic_2d", "global_dino", "semantic_dino")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--target-views", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--reference-palette", type=Path, required=True)
    parser.add_argument("--renders-2d", type=Path, required=True)
    parser.add_argument("--renders-dino", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-tolerance", type=float, default=40.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_views(protocol: dict, root: Path) -> list[Path]:
    records = {
        (entry["view"], entry["modality"]): entry
        for entry in protocol["files"]
        if entry["split"] == "target_heldout"
    }
    result = []
    for view in protocol["selection"]["target_heldout"]:
        directory = root / view
        for modality in ("semantic.png", "camera.json"):
            path = directory / modality
            record = records.get((view, modality))
            if record is None or not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise RuntimeError(f"held-out protocol mismatch: {view}/{modality}")
        result.append(directory)
    return result


def main() -> None:
    args = arguments()
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("protocol is not frozen")
    palette_path = args.reference_palette.resolve()
    palette_report = json.loads(palette_path.read_text(encoding="utf-8"))
    if palette_report.get("status") != "frozen" or palette_report.get("pair_id") != protocol["pair_id"]:
        raise RuntimeError("reference palette is not frozen for this pair")
    if palette_report.get("protocol_manifest_sha256") != sha256(protocol_path):
        raise RuntimeError("reference palette was built from a different protocol")
    if palette_report.get("target_rgb_accessed") is not False:
        raise RuntimeError("reference palette does not prove target_rgb_accessed=false")

    mapping = json.loads(args.semantic_mapping.resolve().read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    semantic_palette = np.asarray([spec["color"][:3] for _, spec in ordered], dtype=np.float64) * 255.0
    reference_rgb = np.asarray([palette_report["class_mean_rgb"][name] for name in names], dtype=np.float64) * 255.0
    views = checked_views(protocol, args.target_views.resolve())
    roots = {
        "global": args.renders_2d.resolve() / "global",
        "semantic_2d": args.renders_2d.resolve() / "semantic_2d",
        "global_dino": args.renders_dino.resolve() / "global_dino",
        "semantic_dino": args.renders_dino.resolve() / "semantic_dino",
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "pair_id": protocol["pair_id"],
        "definition": "Nearest class in one shared frozen donor RGB palette on covered held-out pixels.",
        "protocol_manifest_sha256": sha256(protocol_path),
        "reference_palette": str(palette_path),
        "reference_palette_sha256": sha256(palette_path),
        "methods": {},
    }
    for method in METHODS:
        root = roots[method]
        render_report_path = root / "render_report.json"
        render_report = json.loads(render_report_path.read_text(encoding="utf-8"))
        if render_report.get("pair_id") != protocol["pair_id"] or render_report.get("protocol_split") != "target_heldout":
            raise RuntimeError(f"{method} render is not protocol-locked held-out output")
        confusion = np.zeros((len(names), len(names)), dtype=np.int64)
        view_reports = []
        for view in views:
            prediction = np.asarray(Image.open(root / view.name / "appearance.png").convert("RGB"), dtype=np.float64)
            coverage = np.asarray(Image.open(root / view.name / "coverage.png").convert("L"), dtype=np.uint8) > 0
            semantic = np.asarray(Image.open(view / "semantic.png").convert("RGB"), dtype=np.float64)
            semantic_distance = np.linalg.norm(semantic[..., None, :] - semantic_palette[None, None, :, :], axis=-1)
            labels = semantic_distance.argmin(axis=-1)
            valid = coverage & (semantic_distance.min(axis=-1) <= args.semantic_tolerance)
            predicted = np.linalg.norm(prediction[..., None, :] - reference_rgb[None, None, :, :], axis=-1).argmin(axis=-1)
            np.add.at(confusion, (labels[valid], predicted[valid]), 1)
            pixels = int(valid.sum())
            leaks = int((predicted[valid] != labels[valid]).sum())
            view_reports.append({"view": view.name, "pixels": pixels, "leakage_rate": leaks / pixels if pixels else None})
        row_sum = confusion.sum(axis=1)
        correct = np.diag(confusion)
        per_class = {
            name: {
                "pixels": int(row_sum[index]),
                "leakage_rate": float(1.0 - correct[index] / row_sum[index]) if row_sum[index] else None,
            }
            for index, name in enumerate(names)
        }
        valid_rates = [entry["leakage_rate"] for entry in per_class.values() if entry["leakage_rate"] is not None]
        total = int(confusion.sum())
        report["methods"][method] = {
            "micro_leakage_rate": float(1.0 - np.trace(confusion) / total),
            "macro_leakage_rate": float(np.mean(valid_rates)),
            "evaluated_pixels": total,
            "per_class": per_class,
            "confusion_matrix_rows_true_columns_predicted": confusion.tolist(),
            "views": view_reports,
            "render_report_sha256": sha256(render_report_path),
        }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "methods": {name: values["micro_leakage_rate"] for name, values in report["methods"].items()}}))


if __name__ == "__main__":
    main()
