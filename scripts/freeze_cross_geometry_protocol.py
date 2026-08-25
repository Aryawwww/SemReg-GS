"""Freeze source donors and target train/held-out views for one geometry-transfer pair."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED_MODALITIES = ("rgb.png", "semantic.png", "depth.exr", "normal.exr", "camera.json")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pair-manifest", type=Path, required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--coverage-audit", type=Path, required=True)
    parser.add_argument("--source-multiview", type=Path, required=True)
    parser.add_argument("--target-multiview", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def relative_path(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def file_records(root: Path, group: str, split: str, views: list[str]) -> list[dict]:
    if len(views) != len(set(views)):
        raise RuntimeError(f"{split} contains duplicate view names")
    records = []
    for view in views:
        directory = root / group / view
        missing = [name for name in REQUIRED_MODALITIES if not (directory / name).is_file()]
        if missing:
            raise RuntimeError(f"{directory} is missing: {', '.join(missing)}")
        for modality in REQUIRED_MODALITIES:
            path = (directory / modality).resolve()
            records.append({
                "split": split,
                "scene_role": "source" if split == "source_donor" else "target",
                "view": view,
                "modality": modality,
                "relative_path": relative_path(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })
    return records


def main() -> None:
    args = arguments()
    pair_path = args.pair_manifest.resolve()
    source_split_path = args.source_split.resolve()
    audit_path = args.coverage_audit.resolve()
    pair = load_json(pair_path)
    source_split = load_json(source_split_path)
    audit = load_json(audit_path)

    if source_split.get("status") != "frozen":
        raise RuntimeError("source split is not frozen")
    if pair.get("source_scene_id") != source_split.get("scene_id"):
        raise RuntimeError("pair source_scene_id does not match the frozen source split")
    if pair.get("source_split_sha256") != sha256(source_split_path):
        raise RuntimeError("source split hash differs from the hash recorded by the pair manifest")
    if audit.get("status") != "passed" or audit.get("errors"):
        raise RuntimeError("target coverage audit has not passed")

    source_donors = list(source_split["selection"]["donor"])
    if list(audit["selection"]["donor"]) != source_donors:
        raise RuntimeError("coverage audit donor selection differs from the frozen source donors")
    target_train = list(audit["selection"]["train"])
    target_heldout = list(audit["selection"]["heldout"])
    overlap = sorted(set(target_train) & set(target_heldout))
    if overlap:
        raise RuntimeError(f"target train and held-out overlap: {overlap}")
    if not target_train or not target_heldout:
        raise RuntimeError("target train and held-out must both be non-empty")

    source_root = args.source_multiview.resolve()
    target_root = args.target_multiview.resolve()
    files = []
    files.extend(file_records(source_root, "donor_reference", "source_donor", source_donors))
    files.extend(file_records(target_root, "target_views", "target_train", target_train))
    files.extend(file_records(target_root, "target_views", "target_heldout", target_heldout))

    protocol = {
        "schema_version": 1,
        "status": "frozen",
        "pair_id": pair["pair_id"],
        "source_scene_id": pair["source_scene_id"],
        "target_scene_id": pair["target_scene_id"],
        "frozen_on": date.today().isoformat(),
        "rules": {
            "appearance_source": "source_donor_only",
            "target_heldout_usage": "evaluation_only",
            "target_geometry_frozen": True,
            "minimum_pixels_per_class": audit["minimum_pixels_per_class"],
        },
        "selection": {
            "source_donor": source_donors,
            "target_train": target_train,
            "target_heldout": target_heldout,
        },
        "combined_class_pixels": audit["combined_class_pixels"],
        "source_manifests": [
            {"path": relative_path(pair_path), "sha256": sha256(pair_path)},
            {"path": relative_path(source_split_path), "sha256": sha256(source_split_path)},
            {"path": relative_path(audit_path), "sha256": sha256(audit_path)},
        ],
        "files": files,
    }

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(protocol, indent=2) + "\n"
    if output.exists():
        existing = load_json(output)
        comparable_existing = {key: value for key, value in existing.items() if key != "frozen_on"}
        comparable_new = {key: value for key, value in protocol.items() if key != "frozen_on"}
        if comparable_existing != comparable_new:
            raise RuntimeError(f"frozen protocol already exists with different content: {output}")
        print(json.dumps({"status": "unchanged", "output": str(output), "file_count": len(files)}))
        return
    output.write_text(serialized, encoding="utf-8")
    print(json.dumps({
        "status": "frozen",
        "output": str(output),
        "file_count": len(files),
        "sha256": sha256(output),
        "selection": protocol["selection"],
    }))


if __name__ == "__main__":
    main()
