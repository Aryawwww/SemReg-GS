"""Protocol-locked held-out evaluation for cross-geometry Global/B_sem-2D renders."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--target-views", type=Path, required=True)
    parser.add_argument("--renders-root", type=Path, required=True)
    parser.add_argument("--appearance-root", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--gaussian-validation", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--palette-tolerance", type=float, default=40.0)
    parser.add_argument("--methods", nargs="+", default=["global", "semantic_2d"])
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def checked_protocol_views(protocol: dict, root: Path) -> list[Path]:
    names = protocol["selection"]["target_heldout"]
    records = {
        (record["view"], record["modality"]): record
        for record in protocol["files"]
        if record["split"] == "target_heldout"
    }
    directories = []
    for name in names:
        directory = root / name
        for modality in ("rgb.png", "semantic.png", "camera.json"):
            path = directory / modality
            record = records.get((name, modality))
            if record is None or not path.is_file():
                raise RuntimeError(f"held-out protocol file is missing: {name}/{modality}")
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise RuntimeError(f"held-out protocol file hash mismatch: {name}/{modality}")
        directories.append(directory)
    return directories


def metrics(absolute: float, squared: float, values: int) -> dict:
    if values == 0:
        return {"l1": None, "mse": None, "psnr_db": None}
    mse = squared / values / (255.0**2)
    return {
        "l1": absolute / values / 255.0,
        "mse": mse,
        "psnr_db": -10.0 * math.log10(max(mse, 1e-12)),
    }


def main() -> None:
    args = arguments()
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("cross-geometry protocol is not frozen")
    heldout = checked_protocol_views(protocol, args.target_views.resolve())

    mapping = json.loads(args.semantic_mapping.resolve().read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    palette = np.asarray([spec["color"][:3] for _, spec in ordered], dtype=np.float32) * 255.0

    validation_path = args.gaussian_validation.resolve()
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    if validation.get("status") != "passed" or validation.get("errors"):
        raise RuntimeError("target Gaussian geometry validation has not passed")
    if not all(validation.get("freeze", {}).get(field) for field in ("position", "rotation", "scale")):
        raise RuntimeError("target Gaussian geometry fields are not all frozen")

    report = {
        "schema_version": 1,
        "status": "passed",
        "pair_id": protocol["pair_id"],
        "protocol_manifest": str(protocol_path),
        "protocol_manifest_sha256": sha256(protocol_path),
        "metric_scope": "point-renderer-covered pixels in protocol-frozen target held-out views",
        "heldout_views": [path.name for path in heldout],
        "methods": {},
    }
    for method in args.methods:
        render_root = args.renders_root.resolve() / method
        render_report = json.loads((render_root / "render_report.json").read_text(encoding="utf-8"))
        if render_report.get("pair_id") != protocol["pair_id"] or render_report.get("protocol_split") != "target_heldout":
            raise RuntimeError(f"{method} render report is not protocol-locked held-out output")
        if [entry["view"] for entry in render_report["views"]] != [path.name for path in heldout]:
            raise RuntimeError(f"{method} rendered view list differs from frozen held-out selection")

        baseline_path = args.appearance_root.resolve() / method / "baseline_report.json"
        baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
        if baseline.get("pair_id") != protocol["pair_id"] or baseline.get("donor_view_count") != len(protocol["selection"]["source_donor"]):
            raise RuntimeError(f"{method} appearance report differs from frozen protocol")
        prototypes = np.asarray([baseline["class_mean_rgb"][name] for name in names], dtype=np.float32) * 255.0

        total = {"absolute": 0.0, "squared": 0.0, "values": 0, "pixels": 0, "leaks": 0}
        regions = {name: {"absolute": 0.0, "squared": 0.0, "values": 0, "pixels": 0} for name in names}
        view_results = []
        for target_directory in heldout:
            prediction_directory = render_root / target_directory.name
            prediction = np.asarray(Image.open(prediction_directory / "appearance.png").convert("RGB"), dtype=np.float32)
            coverage = np.asarray(Image.open(prediction_directory / "coverage.png").convert("L"), dtype=np.uint8) > 0
            target = np.asarray(Image.open(target_directory / "rgb.png").convert("RGB"), dtype=np.float32)
            semantic = np.asarray(Image.open(target_directory / "semantic.png").convert("RGB"), dtype=np.float32)
            if prediction.shape != target.shape or target.shape != semantic.shape:
                raise RuntimeError(f"image shape mismatch for {method}/{target_directory.name}")
            distance = np.linalg.norm(semantic[..., None, :] - palette[None, None, :, :], axis=-1)
            labels = distance.argmin(axis=-1)
            valid = coverage & (distance.min(axis=-1) <= args.palette_tolerance)
            error = prediction - target
            absolute = float(np.abs(error[valid]).sum())
            squared = float((error[valid] ** 2).sum())
            values = int(valid.sum()) * 3
            predicted_class = np.linalg.norm(prediction[..., None, :] - prototypes[None, None, :, :], axis=-1).argmin(axis=-1)
            leaks = int(((predicted_class != labels) & valid).sum())
            total["absolute"] += absolute
            total["squared"] += squared
            total["values"] += values
            total["pixels"] += int(valid.sum())
            total["leaks"] += leaks
            view_metric = metrics(absolute, squared, values)
            view_results.append({
                "view": target_directory.name,
                **view_metric,
                "evaluated_pixels": int(valid.sum()),
                "coverage_fraction": float(coverage.mean()),
                "semantic_color_leakage_rate": leaks / int(valid.sum()) if valid.any() else None,
            })
            for class_id, name in enumerate(names):
                mask = valid & (labels == class_id)
                regions[name]["absolute"] += float(np.abs(error[mask]).sum())
                regions[name]["squared"] += float((error[mask] ** 2).sum())
                regions[name]["values"] += int(mask.sum()) * 3
                regions[name]["pixels"] += int(mask.sum())
        if total["values"] == 0:
            raise RuntimeError(f"no covered held-out evaluation pixels for {method}")
        region_report = {}
        for name, values in regions.items():
            region_report[name] = {**metrics(values["absolute"], values["squared"], values["values"]), "pixels": values["pixels"]}
        report["methods"][method] = {
            "summary": {
                **metrics(total["absolute"], total["squared"], total["values"]),
                "evaluated_pixels": total["pixels"],
                "semantic_color_leakage_rate": total["leaks"] / total["pixels"],
            },
            "regions": region_report,
            "views": view_results,
            "appearance_report_sha256": sha256(baseline_path),
            "render_report_sha256": sha256(render_root / "render_report.json"),
        }

    if len(args.methods) == 2:
        first, second = (report["methods"][name]["summary"] for name in args.methods)
        report["paired_delta_second_minus_first"] = {
            "methods": args.methods,
            "l1": second["l1"] - first["l1"],
            "psnr_db": second["psnr_db"] - first["psnr_db"],
            "semantic_color_leakage_rate": second["semantic_color_leakage_rate"] - first["semantic_color_leakage_rate"],
        }
    report["geometry_preservation"] = {
        "status": validation["status"],
        "gaussian_count": validation["gaussian_count"],
        "maximum_center_to_source_triangle_m": validation["center_to_source_triangle_m"]["maximum"],
        "threshold_m": validation["center_to_source_triangle_m"]["threshold"],
        "frozen_fields": validation["freeze"],
        "validation_sha256": sha256(validation_path),
    }
    report["unavailable_metrics"] = {
        "region_lpips": "LPIPS model/package and a frozen evaluation configuration are not present.",
        "region_dino": "A frozen evaluation-only DINO model/weights configuration is not present.",
        "multi_view_warp_error": "No target-view correspondence/warp maps or implemented depth reprojection evaluator are present.",
    }
    report["leakage_definition"] = "Fraction of covered pixels whose rendered RGB is nearest to a donor class prototype different from target semantic class."
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": report["status"],
        "methods": {name: values["summary"] for name, values in report["methods"].items()},
        "delta": report.get("paired_delta_second_minus_first"),
    }))


if __name__ == "__main__":
    main()
