"""Build Global and semantic-2D appearance baselines from donor views only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image


SH_C0 = 0.28209479177387814
ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--donor-views", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path)
    parser.add_argument("--palette-tolerance", type=float, default=40.0)
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def protocol_donor_directories(protocol_path: Path, donor_root: Path) -> tuple[list[Path], str]:
    protocol = json.loads(protocol_path.resolve().read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("cross-geometry protocol is not frozen")
    if protocol.get("rules", {}).get("appearance_source") != "source_donor_only":
        raise RuntimeError("protocol does not require source-donor-only appearance")
    names = protocol["selection"]["source_donor"]
    records = {
        (record["view"], record["modality"]): record
        for record in protocol["files"]
        if record["split"] == "source_donor"
    }
    directories = []
    for name in names:
        directory = donor_root / name
        for modality in ("rgb.png", "semantic.png"):
            path = directory / modality
            record = records.get((name, modality))
            if record is None or not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise RuntimeError(f"frozen donor file differs or is missing: {name}/{modality}")
        directories.append(directory)
    return directories, protocol["pair_id"]


def load_palette(path: Path) -> tuple[list[str], np.ndarray]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    palette = np.asarray([spec["color"][:3] for _, spec in ordered], dtype=np.float32) * 255.0
    return names, palette


def decode_semantics(image: np.ndarray, palette: np.ndarray, tolerance: float) -> np.ndarray:
    distance = np.linalg.norm(image[..., None, :].astype(np.float32) - palette[None, None, :, :], axis=-1)
    labels = distance.argmin(axis=-1).astype(np.int16)
    labels[distance.min(axis=-1) > tolerance] = -1
    return labels


def save_appearance(path: Path, rgb: np.ndarray, method: str, metadata: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    sh_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
    np.savez_compressed(path / "appearance.npz", rgb=rgb.astype(np.float32), sh_dc=sh_dc)
    report = {"method": method, **metadata}
    (path / "baseline_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    args = arguments()
    gaussians = np.load(args.gaussians.resolve())
    semantic_ids = gaussians["semantic_id"].astype(np.int64)
    names, palette = load_palette(args.semantic_mapping.resolve())
    donor_root = args.donor_views.resolve()
    pair_id = None
    if args.protocol_manifest:
        view_directories, pair_id = protocol_donor_directories(args.protocol_manifest, donor_root)
    else:
        view_directories = sorted(path for path in donor_root.iterdir() if path.is_dir())
    if not view_directories:
        raise RuntimeError("No donor views were found")

    class_sum = np.zeros((len(names), 3), dtype=np.float64)
    class_count = np.zeros(len(names), dtype=np.int64)
    global_sum = np.zeros(3, dtype=np.float64)
    global_count = 0
    view_reports = []
    hashes = set()
    for view_directory in view_directories:
        rgb_path = view_directory / "rgb.png"
        semantic_path = view_directory / "semantic.png"
        rgb = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.uint8)
        semantic = np.asarray(Image.open(semantic_path).convert("RGB"), dtype=np.uint8)
        if rgb.shape != semantic.shape:
            raise RuntimeError(f"RGB/semantic shape mismatch in {view_directory.name}")
        digest = hashlib.sha256(rgb_path.read_bytes()).hexdigest()
        hashes.add(digest)
        labels = decode_semantics(semantic, palette, args.palette_tolerance)
        valid = labels >= 0
        global_sum += rgb[valid].sum(axis=0)
        global_count += int(valid.sum())
        per_view = {}
        for class_id, name in enumerate(names):
            mask = labels == class_id
            count = int(mask.sum())
            per_view[name] = count
            if count:
                class_sum[class_id] += rgb[mask].sum(axis=0)
                class_count[class_id] += count
        view_reports.append({"view": view_directory.name, "sha256": digest, "semantic_pixels": per_view})
    if global_count == 0 or len(hashes) < 2:
        raise RuntimeError("Donor views contain no valid semantic pixels or are all identical")

    global_mean = (global_sum / global_count / 255.0).astype(np.float32)
    class_mean = np.repeat(global_mean[None, :], len(names), axis=0)
    observed_classes = class_count > 0
    class_mean[observed_classes] = (class_sum[observed_classes] / class_count[observed_classes, None] / 255.0).astype(np.float32)
    global_rgb = np.repeat(global_mean[None, :], len(semantic_ids), axis=0)
    semantic_rgb = class_mean[semantic_ids]
    metadata = {
        "source": "donor RGB and donor semantic masks only",
        "pair_id": pair_id,
        "protocol_manifest": str(args.protocol_manifest.resolve()) if args.protocol_manifest else None,
        "gaussian_count": int(len(semantic_ids)),
        "donor_view_count": len(view_directories),
        "distinct_donor_hashes": len(hashes),
        "class_names": names,
        "global_mean_rgb": global_mean.tolist(),
        "class_mean_rgb": {name: class_mean[index].tolist() for index, name in enumerate(names)},
        "class_pixel_count": {name: int(class_count[index]) for index, name in enumerate(names)},
        "missing_classes_fallback_to_global": [name for index, name in enumerate(names) if not observed_classes[index]],
        "views": view_reports,
    }
    output = args.output.resolve()
    save_appearance(output / "global", global_rgb, "Global donor mean RGB", metadata)
    save_appearance(output / "semantic_2d", semantic_rgb, "B_sem-2D donor semantic mean RGB", metadata)
    print(json.dumps({"donor_views": len(view_directories), "global_mean_rgb": global_mean.tolist(), "class_pixel_count": metadata["class_pixel_count"]}))


if __name__ == "__main__":
    main()
