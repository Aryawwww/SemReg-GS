"""Evaluate bidirectional appearance consistency using Blender depth and cameras."""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from PIL import Image


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--target-views", type=Path, required=True)
    parser.add_argument("--renders-root", type=Path, required=True)
    parser.add_argument("--depth-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--occlusion-tolerance", type=float, default=0.05)
    parser.add_argument("--minimum-correspondences", type=int, default=100)
    parser.add_argument("--methods", nargs="+", default=["global", "semantic_2d"])
    parser.add_argument(
        "--protocol-split",
        choices=("target_heldout", "target_consistency"),
        default="target_heldout",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reproject(source_depth: np.ndarray, source_camera: dict, target_camera: dict):
    height, width = source_depth.shape
    yy, xx = np.indices((height, width), dtype=np.float64)
    intrinsics = source_camera["intrinsics"]
    rays = np.stack(
        ((xx - intrinsics["cx"]) / intrinsics["fx"],
         -(yy - intrinsics["cy"]) / intrinsics["fy"],
         -np.ones_like(xx)),
        axis=-1,
    )
    rays /= np.linalg.norm(rays, axis=-1, keepdims=True)
    camera_points = rays * source_depth[..., None]
    homogeneous = np.concatenate((camera_points, np.ones((height, width, 1))), axis=-1)
    world = homogeneous @ np.asarray(source_camera["camera_to_world_blender"], dtype=np.float64).T
    target = world @ np.asarray(target_camera["world_to_camera_blender"], dtype=np.float64).T
    target_depth = np.linalg.norm(target[..., :3], axis=-1)
    forward = -target[..., 2]
    target_intrinsics = target_camera["intrinsics"]
    u = target_intrinsics["fx"] * target[..., 0] / np.maximum(forward, 1e-12) + target_intrinsics["cx"]
    v = target_intrinsics["cy"] - target_intrinsics["fy"] * target[..., 1] / np.maximum(forward, 1e-12)
    return np.rint(u).astype(np.int64), np.rint(v).astype(np.int64), target_depth, forward


def direction_metrics(
    source_name: str,
    target_name: str,
    method_root: Path,
    views_root: Path,
    depths: dict[str, np.ndarray],
    tolerance: float,
) -> dict:
    source_camera = json.loads((views_root / source_name / "camera.json").read_text(encoding="utf-8"))
    target_camera = json.loads((views_root / target_name / "camera.json").read_text(encoding="utf-8"))
    source_image = np.asarray(Image.open(method_root / source_name / "appearance.png").convert("RGB"), dtype=np.float32)
    target_image = np.asarray(Image.open(method_root / target_name / "appearance.png").convert("RGB"), dtype=np.float32)
    source_coverage = np.asarray(Image.open(method_root / source_name / "coverage.png").convert("L"), dtype=np.uint8) > 0
    target_coverage = np.asarray(Image.open(method_root / target_name / "coverage.png").convert("L"), dtype=np.uint8) > 0
    source_depth = depths[source_name]
    target_depth_map = depths[target_name]
    u, v, projected_depth, forward = reproject(source_depth, source_camera, target_camera)
    target_height, target_width = target_depth_map.shape
    source_valid = source_coverage & np.isfinite(source_depth) & (source_depth > 0)
    in_frame = source_valid & (forward > 0.05) & (u >= 0) & (u < target_width) & (v >= 0) & (v < target_height)
    sy, sx = np.nonzero(in_frame)
    if len(sy) == 0:
        return {"source": source_name, "target": target_name, "in_frame": 0, "correspondences": 0, "l1": None, "rmse": None}
    tu, tv = u[sy, sx], v[sy, sx]
    sampled_depth = target_depth_map[tv, tu]
    visible = (
        np.isfinite(sampled_depth)
        & (sampled_depth > 0)
        & target_coverage[tv, tu]
        & (np.abs(sampled_depth - projected_depth[sy, sx]) <= tolerance)
    )
    sy, sx, tu, tv = sy[visible], sx[visible], tu[visible], tv[visible]
    if len(sy) == 0:
        return {"source": source_name, "target": target_name, "in_frame": int(in_frame.sum()), "correspondences": 0, "l1": None, "rmse": None}
    difference = (source_image[sy, sx] - target_image[tv, tu]) / 255.0
    return {
        "source": source_name,
        "target": target_name,
        "in_frame": int(in_frame.sum()),
        "correspondences": int(len(sy)),
        "l1": float(np.abs(difference).mean()),
        "rmse": float(np.sqrt(np.mean(difference**2))),
    }


def main() -> None:
    args = arguments()
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("cross-geometry protocol is not frozen")
    names = list(protocol["selection"][args.protocol_split])
    depth_cache = args.depth_cache.resolve()
    archive = np.load(depth_cache)
    if set(names) - set(archive.files):
        raise RuntimeError("depth cache does not contain every frozen held-out view")
    depths = {name: archive[name].astype(np.float64) for name in names}
    views_root = args.target_views.resolve()
    report = {
        "schema_version": 1,
        "pair_id": protocol["pair_id"],
        "protocol_manifest_sha256": sha256(protocol_path),
        "depth_cache_sha256": sha256(depth_cache),
        "protocol_split": args.protocol_split,
        "evaluation_views": names,
        "depth_definition": "Blender Z pass interpreted as Euclidean camera-to-surface ray distance.",
        "occlusion_tolerance_m": args.occlusion_tolerance,
        "minimum_correspondences": args.minimum_correspondences,
        "methods": {},
    }
    directions = [(a, b) for a, b in itertools.permutations(names, 2)]
    any_evaluable = False
    for method in args.methods:
        method_root = args.renders_root.resolve() / method
        results = [
            direction_metrics(a, b, method_root, views_root, depths, args.occlusion_tolerance)
            for a, b in directions
        ]
        valid = [entry for entry in results if entry["correspondences"] >= args.minimum_correspondences]
        count = sum(entry["correspondences"] for entry in valid)
        if count:
            any_evaluable = True
            l1 = sum(entry["l1"] * entry["correspondences"] for entry in valid) / count
            rmse = sum(entry["rmse"] * entry["correspondences"] for entry in valid) / count
            status = "passed"
            reason = None
        else:
            l1 = rmse = None
            status = "insufficient_overlap"
            reason = "No direction met the minimum visible correspondence count. Add overlapping held-out cameras."
        report["methods"][method] = {
            "status": status,
            "warp_l1": l1,
            "warp_rmse": rmse,
            "correspondences": count,
            "directions": results,
            "reason": reason,
        }
    report["status"] = "passed" if any_evaluable else "insufficient_overlap"
    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": report["status"], "methods": report["methods"]}))


if __name__ == "__main__":
    main()
