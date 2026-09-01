"""Measure intervention spill after excluding pixels near semantic boundaries."""

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
    parser.add_argument("--margins", nargs="+", type=int, default=(0, 2, 4, 8, 16))
    parser.add_argument("--semantic-tolerance", type=float, default=40.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def erode(mask: np.ndarray, radius: int) -> np.ndarray:
    """Square erosion without scipy; border pixels outside the image are false."""
    if radius == 0:
        return mask.copy()
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    result = np.ones_like(mask)
    for dy in range(2 * radius + 1):
        for dx in range(2 * radius + 1):
            result &= padded[dy : dy + height, dx : dx + width]
    return result


def main() -> None:
    args = arguments()
    if any(margin < 0 for margin in args.margins) or sorted(set(args.margins)) != list(args.margins):
        raise ValueError("--margins must be unique non-negative values in ascending order")
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("protocol is not frozen")
    mapping = json.loads(args.semantic_mapping.resolve().read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    palette = np.asarray([spec["color"][:3] for _, spec in ordered], dtype=np.float32) * 255.0
    base_roots = {
        "semantic_2d": args.base_renders_2d.resolve() / "semantic_2d",
        "semantic_dino": args.base_renders_dino.resolve() / "semantic_dino",
    }
    view_names = list(protocol["selection"]["target_heldout"])
    report = {
        "schema_version": 1,
        "status": "passed",
        "pair_id": protocol["pair_id"],
        "definition": "Intervention response after separately eroding target and non-target semantic masks by a square pixel radius.",
        "protocol_manifest_sha256": sha256(protocol_path),
        "margins_px": list(args.margins),
        "methods": {},
    }
    for method in METHODS:
        per_class = {}
        for class_id, class_name in enumerate(names):
            accumulators = {
                margin: {"target_change": 0.0, "spill_change": 0.0, "target_pixels": 0, "spill_pixels": 0}
                for margin in args.margins
            }
            for view in view_names:
                base_dir = base_roots[method] / view
                edit_dir = args.edited_renders.resolve() / method / class_name / view
                base = np.asarray(Image.open(base_dir / "appearance.png").convert("RGB"), dtype=np.float32) / 255.0
                edited = np.asarray(Image.open(edit_dir / "appearance.png").convert("RGB"), dtype=np.float32) / 255.0
                coverage = (
                    (np.asarray(Image.open(base_dir / "coverage.png").convert("L"), dtype=np.uint8) > 0)
                    & (np.asarray(Image.open(edit_dir / "coverage.png").convert("L"), dtype=np.uint8) > 0)
                )
                semantic = np.asarray(Image.open(args.target_views.resolve() / view / "semantic.png").convert("RGB"), dtype=np.float32)
                distance = np.linalg.norm(semantic[..., None, :] - palette[None, None, :, :], axis=-1)
                labels = distance.argmin(axis=-1)
                valid_semantic = distance.min(axis=-1) <= args.semantic_tolerance
                target_raw = valid_semantic & (labels == class_id)
                spill_raw = valid_semantic & (labels != class_id)
                change = np.abs(edited - base).mean(axis=-1)
                for margin, values in accumulators.items():
                    target = coverage & erode(target_raw, margin)
                    spill = coverage & erode(spill_raw, margin)
                    values["target_change"] += float(change[target].sum())
                    values["spill_change"] += float(change[spill].sum())
                    values["target_pixels"] += int(target.sum())
                    values["spill_pixels"] += int(spill.sum())
            curve = []
            for margin, values in accumulators.items():
                target_response = values["target_change"] / values["target_pixels"] if values["target_pixels"] else None
                spill = values["spill_change"] / values["spill_pixels"] if values["spill_pixels"] else None
                selectivity = target_response / max(target_response + spill, 1e-12) if target_response is not None and spill is not None else None
                curve.append({
                    "margin_px": margin,
                    "target_response_l1": target_response,
                    "non_target_spill_l1": spill,
                    "selectivity": selectivity,
                    "target_pixels": values["target_pixels"],
                    "non_target_pixels": values["spill_pixels"],
                })
            baseline_spill = curve[0]["non_target_spill_l1"]
            final_spill = curve[-1]["non_target_spill_l1"]
            per_class[class_name] = {
                "curve": curve,
                "spill_reduction_at_max_margin": (
                    1.0 - final_spill / baseline_spill if baseline_spill and final_spill is not None else None
                ),
            }
        report["methods"][method] = {"per_class": per_class}
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "passed",
        "wall": {method: report["methods"][method]["per_class"]["wall"] for method in METHODS},
    }))


if __name__ == "__main__":
    main()
