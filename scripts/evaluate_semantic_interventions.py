"""Measure target response and non-target spill for single-class appearance edits."""

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
    parser.add_argument("--base-renders-2d", type=Path, required=True)
    parser.add_argument("--base-renders-dino", type=Path, required=True)
    parser.add_argument("--edited-renders", type=Path, required=True)
    parser.add_argument("--intervention-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantic-tolerance", type=float, default=40.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = arguments()
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    manifest_path = args.intervention_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen" or manifest.get("pair_id") != protocol.get("pair_id"):
        raise RuntimeError("protocol/intervention manifest mismatch")
    if manifest.get("protocol_manifest_sha256") != sha256(protocol_path):
        raise RuntimeError("interventions were built from a different protocol")
    mapping = json.loads(args.semantic_mapping.resolve().read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    palette = np.asarray([spec["color"][:3] for _, spec in ordered], dtype=np.float32) * 255.0
    base_roots = {
        "global": args.base_renders_2d.resolve() / "global",
        "semantic_2d": args.base_renders_2d.resolve() / "semantic_2d",
        "global_dino": args.base_renders_dino.resolve() / "global_dino",
        "semantic_dino": args.base_renders_dino.resolve() / "semantic_dino",
    }
    view_names = list(protocol["selection"]["target_heldout"])
    results = {}
    for method in METHODS:
        method_results = {}
        for class_id, class_name in enumerate(names):
            target_change = spill_change = 0.0
            target_pixels = spill_pixels = 0
            for view in view_names:
                base_dir = base_roots[method] / view
                edit_dir = args.edited_renders.resolve() / method / class_name / view
                base = np.asarray(Image.open(base_dir / "appearance.png").convert("RGB"), dtype=np.float32) / 255.0
                edited = np.asarray(Image.open(edit_dir / "appearance.png").convert("RGB"), dtype=np.float32) / 255.0
                base_coverage = np.asarray(Image.open(base_dir / "coverage.png").convert("L"), dtype=np.uint8) > 0
                edit_coverage = np.asarray(Image.open(edit_dir / "coverage.png").convert("L"), dtype=np.uint8) > 0
                semantic = np.asarray(Image.open(args.target_views.resolve() / view / "semantic.png").convert("RGB"), dtype=np.float32)
                distance = np.linalg.norm(semantic[..., None, :] - palette[None, None, :, :], axis=-1)
                labels = distance.argmin(axis=-1)
                valid = base_coverage & edit_coverage & (distance.min(axis=-1) <= args.semantic_tolerance)
                change = np.abs(edited - base).mean(axis=-1)
                target = valid & (labels == class_id)
                spill = valid & (labels != class_id)
                target_change += float(change[target].sum())
                spill_change += float(change[spill].sum())
                target_pixels += int(target.sum())
                spill_pixels += int(spill.sum())
            target_response = target_change / target_pixels if target_pixels else None
            non_target_spill = spill_change / spill_pixels if spill_pixels else None
            selectivity = None
            if target_response is not None and non_target_spill is not None:
                selectivity = target_response / max(target_response + non_target_spill, 1e-12)
            method_results[class_name] = {
                "target_response_l1": target_response,
                "non_target_spill_l1": non_target_spill,
                "selectivity": selectivity,
                "target_pixels": target_pixels,
                "non_target_pixels": spill_pixels,
            }
        results[method] = {
            "per_class": method_results,
            "macro_target_response_l1": float(np.mean([v["target_response_l1"] for v in method_results.values()])),
            "macro_non_target_spill_l1": float(np.mean([v["non_target_spill_l1"] for v in method_results.values()])),
            "macro_selectivity": float(np.mean([v["selectivity"] for v in method_results.values()])),
        }
    report = {
        "schema_version": 1,
        "status": "passed",
        "pair_id": protocol["pair_id"],
        "definition": "Change in rendered RGB after requesting one semantic-class edit; global methods edit all Gaussians because they have no class selector.",
        "protocol_manifest_sha256": sha256(protocol_path),
        "intervention_manifest_sha256": sha256(manifest_path),
        "methods": results,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "methods": {m: {k: v for k, v in r.items() if k != "per_class"} for m, r in results.items()}}))


if __name__ == "__main__":
    main()
