"""Freeze a passed semantic coverage audit into an immutable split manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--audit", type=Path, required=True)
    parser.add_argument("--multiview", type=Path, required=True)
    parser.add_argument("--window-manifest", type=Path)
    parser.add_argument("--scene-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    audit_path = args.audit.resolve()
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != "passed" or audit.get("errors"):
        raise RuntimeError("Coverage audit has not passed")
    multiview = args.multiview.resolve()
    group_for_split = {"donor": "donor_reference", "train": "target_views", "heldout": "target_views"}
    files = []
    for split_name in ("donor", "train", "heldout"):
        for view_name in audit["selection"][split_name]:
            view_directory = multiview / group_for_split[split_name] / view_name
            required = ("rgb.png", "semantic.png", "depth.exr", "normal.exr", "camera.json")
            missing = [name for name in required if not (view_directory / name).is_file()]
            if missing:
                raise RuntimeError(f"{view_directory} is missing: {', '.join(missing)}")
            for name in required:
                path = view_directory / name
                files.append({
                    "split": split_name,
                    "view": view_name,
                    "modality": name,
                    "relative_path": path.relative_to(multiview.parent.parent.parent.parent).as_posix(),
                    "bytes": path.stat().st_size,
                    "sha256": sha256(path),
                })
    source_manifests = [{"path": str(audit_path), "sha256": sha256(audit_path)}]
    if args.window_manifest:
        window_path = args.window_manifest.resolve()
        if not window_path.is_file():
            raise RuntimeError(f"Window manifest does not exist: {window_path}")
        source_manifests.append({"path": str(window_path), "sha256": sha256(window_path)})
    manifest = {
        "schema_version": 1,
        "status": "frozen",
        "scene_id": args.scene_id,
        "frozen_on": date.today().isoformat(),
        "minimum_pixels_per_class": audit["minimum_pixels_per_class"],
        "semantic_classes": audit["semantic_classes"],
        "selection": audit["selection"],
        "combined_class_pixels": audit["combined_class_pixels"],
        "source_manifests": source_manifests,
        "files": files,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(manifest, indent=2) + "\n"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        comparable_existing = {key: value for key, value in existing.items() if key != "frozen_on"}
        comparable_new = {key: value for key, value in manifest.items() if key != "frozen_on"}
        if comparable_existing != comparable_new:
            raise RuntimeError(f"Frozen split already exists with different content: {output}")
        print(json.dumps({"status": "unchanged", "output": str(output), "file_count": len(files)}))
        return
    output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"status": "frozen", "output": str(output), "file_count": len(files), "manifest_sha256": sha256(output)}))


if __name__ == "__main__":
    main()
