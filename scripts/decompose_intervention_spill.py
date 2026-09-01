"""Decompose semantic-edit spill into intrinsic leakage and proxy-label disagreement."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


METHODS = ("semantic_2d", "semantic_dino")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--target-views", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--base-renders-2d", type=Path, required=True)
    parser.add_argument("--base-renders-dino", type=Path, required=True)
    parser.add_argument("--edited-renders", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--palette-tolerance", type=float, default=40.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def decode(image: np.ndarray, palette: np.ndarray, tolerance: float) -> tuple[np.ndarray, np.ndarray]:
    distance = np.linalg.norm(image[..., None, :].astype(np.float32) - palette[None, None, :, :], axis=-1)
    return distance.argmin(axis=-1), distance.min(axis=-1) <= tolerance


def mean_change(change: np.ndarray, mask: np.ndarray) -> tuple[float | None, int, float]:
    pixels = int(mask.sum())
    total = float(change[mask].sum())
    return (total / pixels if pixels else None), pixels, total


def main() -> None:
    args = arguments()
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("protocol is not frozen")
    mapping = json.loads(args.semantic_mapping.resolve().read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    palette = np.asarray([spec["color"][:3] for _, spec in ordered], dtype=np.float32) * 255.0
    roots = {
        "semantic_2d": args.base_renders_2d.resolve() / "semantic_2d",
        "semantic_dino": args.base_renders_dino.resolve() / "semantic_dino",
    }
    report = {
        "schema_version": 1,
        "status": "passed",
        "pair_id": protocol["pair_id"],
        "protocol_manifest_sha256": sha256(protocol_path),
        "definitions": {
            "intrinsic_spill": "RGB change on pixels whose rendered Gaussian semantic ID is not the edited class.",
            "proxy_apparent_spill": "RGB change on pixels whose held-out proxy semantic is not the edited class.",
            "explained_apparent_spill": "Proxy-apparent spill occurring where rendered Gaussian semantic equals the edited class.",
        },
        "methods": {},
    }
    for method in METHODS:
        per_class = {}
        for class_id, class_name in enumerate(names):
            totals = {
                key: {"change": 0.0, "pixels": 0}
                for key in ("rendered_target", "intrinsic_spill", "proxy_target", "proxy_apparent_spill", "explained_apparent_spill")
            }
            per_view = []
            for view in protocol["selection"]["target_heldout"]:
                base_dir = roots[method] / view
                edit_dir = args.edited_renders.resolve() / method / class_name / view
                base = np.asarray(Image.open(base_dir / "appearance.png").convert("RGB"), dtype=np.float32) / 255.0
                edited = np.asarray(Image.open(edit_dir / "appearance.png").convert("RGB"), dtype=np.float32) / 255.0
                rendered_semantic = np.asarray(Image.open(base_dir / "semantic.png").convert("RGB"), dtype=np.uint8)
                proxy_semantic = np.asarray(Image.open(args.target_views.resolve() / view / "semantic.png").convert("RGB"), dtype=np.uint8)
                coverage = (
                    (np.asarray(Image.open(base_dir / "coverage.png").convert("L"), dtype=np.uint8) > 0)
                    & (np.asarray(Image.open(edit_dir / "coverage.png").convert("L"), dtype=np.uint8) > 0)
                )
                rendered_label, rendered_valid = decode(rendered_semantic, palette, args.palette_tolerance)
                proxy_label, proxy_valid = decode(proxy_semantic, palette, args.palette_tolerance)
                valid = coverage & rendered_valid & proxy_valid
                change = np.abs(edited - base).mean(axis=-1)
                masks = {
                    "rendered_target": valid & (rendered_label == class_id),
                    "intrinsic_spill": valid & (rendered_label != class_id),
                    "proxy_target": valid & (proxy_label == class_id),
                    "proxy_apparent_spill": valid & (proxy_label != class_id),
                    "explained_apparent_spill": valid & (proxy_label != class_id) & (rendered_label == class_id),
                }
                view_values = {"view": view}
                for key, mask in masks.items():
                    mean, pixels, total = mean_change(change, mask)
                    totals[key]["change"] += total
                    totals[key]["pixels"] += pixels
                    view_values[key + "_l1"] = mean
                per_view.append(view_values)
            values = {}
            for key, total in totals.items():
                values[key + "_l1"] = total["change"] / total["pixels"] if total["pixels"] else None
                values[key + "_pixels"] = total["pixels"]
                values[key + "_total_change"] = total["change"]
            apparent_total = totals["proxy_apparent_spill"]["change"]
            values["apparent_spill_change_explained_fraction"] = (
                totals["explained_apparent_spill"]["change"] / apparent_total if apparent_total else None
            )
            rendered_response = values["rendered_target_l1"]
            intrinsic = values["intrinsic_spill_l1"]
            values["intrinsic_selectivity"] = (
                rendered_response / max(rendered_response + intrinsic, 1e-12)
                if rendered_response is not None and intrinsic is not None else None
            )
            per_class[class_name] = {**values, "views": per_view}
        report["methods"][method] = {
            "per_class": per_class,
            "macro_intrinsic_spill_l1": float(np.mean([entry["intrinsic_spill_l1"] for entry in per_class.values()])),
            "macro_intrinsic_selectivity": float(np.mean([entry["intrinsic_selectivity"] for entry in per_class.values()])),
            "macro_apparent_spill_explained_fraction": float(np.mean([entry["apparent_spill_change_explained_fraction"] for entry in per_class.values()])),
        }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "methods": {method: {key: value for key, value in result.items() if key != "per_class"} for method, result in report["methods"].items()},
    }))


if __name__ == "__main__":
    main()
