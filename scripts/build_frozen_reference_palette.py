"""Build one protocol-locked RGB class palette shared by every evaluated method."""

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
    parser.add_argument("--donor-views", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--palette-tolerance", type=float, default=40.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = arguments()
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("protocol is not frozen")
    if protocol.get("rules", {}).get("appearance_source") != "source_donor_only":
        raise RuntimeError("protocol does not require source-donor-only appearance")

    mapping_path = args.semantic_mapping.resolve()
    mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    semantic_palette = np.asarray([spec["color"][:3] for _, spec in ordered], dtype=np.float64) * 255.0
    sums = np.zeros((len(names), 3), dtype=np.float64)
    counts = np.zeros(len(names), dtype=np.int64)

    records = {
        (entry["view"], entry["modality"]): entry
        for entry in protocol["files"]
        if entry["split"] == "source_donor"
    }
    donor_reports = []
    donor_root = args.donor_views.resolve()
    for view in protocol["selection"]["source_donor"]:
        directory = donor_root / view
        paths = {modality: directory / modality for modality in ("rgb.png", "semantic.png")}
        for modality, path in paths.items():
            record = records.get((view, modality))
            if record is None or not path.is_file():
                raise RuntimeError(f"frozen donor is missing: {view}/{modality}")
            if path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise RuntimeError(f"frozen donor hash mismatch: {view}/{modality}")
        rgb = np.asarray(Image.open(paths["rgb.png"]).convert("RGB"), dtype=np.float64)
        semantic = np.asarray(Image.open(paths["semantic.png"]).convert("RGB"), dtype=np.float64)
        distances = np.linalg.norm(semantic[..., None, :] - semantic_palette[None, None, :, :], axis=-1)
        labels = distances.argmin(axis=-1)
        valid = distances.min(axis=-1) <= args.palette_tolerance
        per_view = {}
        for class_id, name in enumerate(names):
            mask = valid & (labels == class_id)
            per_view[name] = int(mask.sum())
            if mask.any():
                sums[class_id] += rgb[mask].sum(axis=0)
                counts[class_id] += int(mask.sum())
        donor_reports.append({"view": view, "class_pixel_count": per_view})

    missing = [name for index, name in enumerate(names) if counts[index] == 0]
    if missing:
        raise RuntimeError(f"frozen donors do not cover classes: {missing}")
    rgb = sums / counts[:, None] / 255.0
    report = {
        "schema_version": 1,
        "status": "frozen",
        "pair_id": protocol["pair_id"],
        "definition": "Per-class mean RGB over protocol-frozen source donor pixels.",
        "usage": "Shared evaluation reference; must not vary by evaluated method.",
        "protocol_manifest": str(protocol_path),
        "protocol_manifest_sha256": sha256(protocol_path),
        "semantic_mapping": str(mapping_path),
        "semantic_mapping_sha256": sha256(mapping_path),
        "palette_tolerance": args.palette_tolerance,
        "source_donor": list(protocol["selection"]["source_donor"]),
        "class_names": names,
        "class_mean_rgb": {name: rgb[index].tolist() for index, name in enumerate(names)},
        "class_pixel_count": {name: int(counts[index]) for index, name in enumerate(names)},
        "donor_reports": donor_reports,
        "target_rgb_accessed": False,
    }
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "class_pixel_count": report["class_pixel_count"]}))


if __name__ == "__main__":
    main()
