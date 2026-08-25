"""Evaluate rendered Step 7 baselines against held-out target-view RGB."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--methods-root", type=Path, required=True)
    parser.add_argument("--target-views", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--palette-tolerance", type=float, default=40.0)
    parser.add_argument("--view-names", nargs="*", default=None)
    return parser.parse_args()


def main() -> None:
    args = arguments()
    mapping = json.loads(args.semantic_mapping.resolve().read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    palette = np.asarray([spec["color"][:3] for _, spec in ordered], dtype=np.float32) * 255.0
    methods = sorted(path for path in args.methods_root.resolve().iterdir() if (path / "renders").is_dir())
    if not methods:
        raise RuntimeError("No rendered methods were found")
    report = {"metric_scope": "covered rendered pixels in held-out target views", "methods": {}}
    for method in methods:
        baseline = json.loads((method / "baseline_report.json").read_text(encoding="utf-8"))
        prototypes = np.asarray(list(baseline["class_mean_rgb"].values()), dtype=np.float32) * 255.0
        totals = {"absolute": 0.0, "squared": 0.0, "values": 0, "leaks": 0, "pixels": 0}
        region = {name: {"absolute": 0.0, "squared": 0.0, "values": 0, "pixels": 0} for name in names}
        views = []
        target_directories = sorted(path for path in args.target_views.resolve().iterdir() if path.is_dir())
        if args.view_names:
            requested = set(args.view_names)
            target_directories = [path for path in target_directories if path.name in requested]
            if {path.name for path in target_directories} != requested:
                raise RuntimeError("One or more requested target views were not found")
        for target_directory in target_directories:
            prediction_directory = method / "renders" / target_directory.name
            prediction = np.asarray(Image.open(prediction_directory / "appearance.png").convert("RGB"), dtype=np.float32)
            coverage = np.asarray(Image.open(prediction_directory / "coverage.png").convert("L"), dtype=np.uint8) > 0
            target = np.asarray(Image.open(target_directory / "rgb.png").convert("RGB"), dtype=np.float32)
            semantic = np.asarray(Image.open(target_directory / "semantic.png").convert("RGB"), dtype=np.float32)
            distances = np.linalg.norm(semantic[..., None, :] - palette[None, None, :, :], axis=-1)
            labels = distances.argmin(axis=-1)
            valid = coverage & (distances.min(axis=-1) <= args.palette_tolerance)
            error = prediction - target
            totals["absolute"] += float(np.abs(error[valid]).sum())
            totals["squared"] += float((error[valid] ** 2).sum())
            totals["values"] += int(valid.sum()) * 3
            totals["pixels"] += int(valid.sum())
            predicted_class = np.linalg.norm(prediction[..., None, :] - prototypes[None, None, :, :], axis=-1).argmin(axis=-1)
            totals["leaks"] += int(((predicted_class != labels) & valid).sum())
            for class_id, name in enumerate(names):
                mask = valid & (labels == class_id)
                region[name]["absolute"] += float(np.abs(error[mask]).sum())
                region[name]["squared"] += float((error[mask] ** 2).sum())
                region[name]["values"] += int(mask.sum()) * 3
                region[name]["pixels"] += int(mask.sum())
            views.append({"view": target_directory.name, "evaluated_pixels": int(valid.sum()), "coverage": float(coverage.mean())})
        if totals["values"] == 0:
            raise RuntimeError(f"No valid evaluation pixels for {method.name}")
        mse = totals["squared"] / totals["values"] / (255.0 ** 2)
        summary = {
            "l1": totals["absolute"] / totals["values"] / 255.0,
            "mse": mse,
            "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
            "semantic_color_leakage_rate": totals["leaks"] / totals["pixels"],
            "evaluated_pixels": totals["pixels"],
        }
        regions = {}
        for name, values in region.items():
            if values["values"]:
                class_mse = values["squared"] / values["values"] / (255.0 ** 2)
                regions[name] = {"l1": values["absolute"] / values["values"] / 255.0, "psnr_db": -10.0 * math.log10(max(class_mse, 1e-12)), "pixels": values["pixels"]}
            else:
                regions[name] = {"l1": None, "psnr_db": None, "pixels": 0}
        report["methods"][method.name] = {"summary": summary, "regions": regions, "views": views}
    report["leakage_definition"] = "Fraction of covered target pixels whose rendered RGB is nearest to a donor class-mean prototype different from the target semantic class."
    args.output.resolve().parent.mkdir(parents=True, exist_ok=True)
    args.output.resolve().write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({name: values["summary"] for name, values in report["methods"].items()}))


if __name__ == "__main__":
    main()
