"""Validate target-scene audit products and advance the pair manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--asset-audit", type=Path, required=True)
    parser.add_argument("--semantic-audit", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--asset-preview", type=Path, required=True)
    parser.add_argument("--semantic-preview", type=Path, required=True)
    args = parser.parse_args()

    paths = {
        "asset_audit": args.asset_audit.resolve(),
        "semantic_audit": args.semantic_audit.resolve(),
        "semantic_mapping": args.semantic_mapping.resolve(),
        "asset_preview": args.asset_preview.resolve(),
        "semantic_preview": args.semantic_preview.resolve(),
    }
    missing = [name for name, path in paths.items() if not path.is_file() or path.stat().st_size == 0]
    if missing:
        raise RuntimeError(f"Missing or empty audit products: {', '.join(missing)}")

    asset = json.loads(paths["asset_audit"].read_text(encoding="utf-8"))
    semantic = json.loads(paths["semantic_audit"].read_text(encoding="utf-8"))
    mapping = json.loads(paths["semantic_mapping"].read_text(encoding="utf-8"))
    if asset.get("mesh_count", 0) <= 0 or asset.get("polygon_count", 0) <= 0:
        raise RuntimeError("The target GLB contains no auditable mesh polygons")
    if semantic.get("instance_count", 0) <= 0:
        raise RuntimeError("The target scene instance file contains no object instances")
    if semantic.get("resolved_instance_count", 0) <= 0:
        raise RuntimeError("No target object instance could be resolved to HSSD metadata")

    required_classes = ["wall", "floor", "ceiling", "door", "window", "other"]
    classes = mapping.get("classes", {})
    if any(name not in classes for name in required_classes):
        raise RuntimeError("Semantic mapping does not define all six experiment classes")
    polygon_counts = mapping.get("polygon_counts", {})
    absent_classes = [name for name in required_classes if polygon_counts.get(name, 0) == 0]
    resolution = semantic["resolved_instance_count"] / semantic["instance_count"]

    manifest_path = args.pair_manifest.resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update({
        "status": "audited_pending_multiview_render",
        "target_audit": {
            "mesh_count": asset["mesh_count"],
            "polygon_count": asset["polygon_count"],
            "instance_count": semantic["instance_count"],
            "resolved_instance_count": semantic["resolved_instance_count"],
            "instance_resolution_fraction": resolution,
            "semantic_polygon_counts": polygon_counts,
            "absent_semantic_classes": absent_classes,
            "products": {
                name: {"path": str(path), "bytes": path.stat().st_size, "sha256": sha256(path)}
                for name, path in paths.items()
            },
        },
        "next_required_steps": [
            "multiview_render", "coverage_audit", "semantic_gaussian_initialization"
        ],
    })
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": manifest["status"],
        "target_scene_id": manifest["target_scene_id"],
        "mesh_count": asset["mesh_count"],
        "polygon_count": asset["polygon_count"],
        "instance_resolution_fraction": resolution,
        "absent_semantic_classes": absent_classes,
    }))


if __name__ == "__main__":
    main()
