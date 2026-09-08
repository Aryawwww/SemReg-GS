"""Render neutral appearance or semantic colors from initialized Gaussians."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--appearance", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("appearance", "semantic", "both"), default="both")
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--depth-cache", type=Path)
    parser.add_argument("--occlusion-tolerance", type=float, default=0.05)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--protocol-manifest", type=Path)
    parser.add_argument(
        "--protocol-split",
        choices=("target_train", "target_heldout", "target_consistency"),
        default="target_heldout",
    )
    return parser.parse_args()


def protocol_view_directories(protocol_path: Path, views_root: Path, split: str) -> tuple[list[Path], str]:
    protocol = json.loads(protocol_path.resolve().read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise RuntimeError("cross-geometry protocol is not frozen")
    names = protocol["selection"][split]
    records = {
        (record["view"], record["modality"]): record
        for record in protocol["files"]
        if record["split"] == split
    }
    directories = []
    for name in names:
        directory = views_root / name
        camera = directory / "camera.json"
        record = records.get((name, "camera.json"))
        if record is None or not camera.is_file():
            raise RuntimeError(f"frozen camera is missing: {name}/camera.json")
        digest = hashlib.sha256(camera.read_bytes()).hexdigest()
        if camera.stat().st_size != record["bytes"] or digest != record["sha256"]:
            raise RuntimeError(f"frozen camera differs from protocol: {name}/camera.json")
        directories.append(directory)
    return directories, protocol["pair_id"]


def project(xyz: torch.Tensor, camera: dict, device: torch.device):
    matrix = torch.tensor(camera["world_to_camera_blender"], dtype=torch.float32, device=device)
    homogeneous = torch.cat((xyz, torch.ones((len(xyz), 1), device=device)), dim=1)
    camera_xyz = homogeneous @ matrix.T
    depth = -camera_xyz[:, 2]
    intrinsics = camera["intrinsics"]
    u = intrinsics["fx"] * camera_xyz[:, 0] / depth.clamp_min(1e-8) + intrinsics["cx"]
    v = intrinsics["cy"] - intrinsics["fy"] * camera_xyz[:, 1] / depth.clamp_min(1e-8)
    return u, v, depth


def rasterize(
    xyz: torch.Tensor,
    colors: torch.Tensor,
    camera: dict,
    radius: int,
    device: torch.device,
    mesh_depth: torch.Tensor | None = None,
    occlusion_tolerance: float = 0.05,
) -> tuple[np.ndarray, np.ndarray, float]:
    width, height = int(camera["width"]), int(camera["height"])
    u, v, depth = project(xyz, camera, device)
    base_x = torch.round(u).long()
    base_y = torch.round(v).long()
    source_ids = torch.arange(len(xyz), device=device)
    offsets = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    px = torch.cat([base_x + dx for dx, _ in offsets])
    py = torch.cat([base_y + dy for _, dy in offsets])
    expanded_depth = depth.repeat(len(offsets))
    camera_matrix = torch.tensor(camera["world_to_camera_blender"], dtype=torch.float32, device=device)
    homogeneous = torch.cat((xyz, torch.ones((len(xyz), 1), device=device)), dim=1)
    camera_xyz = homogeneous @ camera_matrix.T
    ray_depth = torch.linalg.vector_norm(camera_xyz[:, :3], dim=1)
    expanded_ray_depth = ray_depth.repeat(len(offsets))
    expanded_source = source_ids.repeat(len(offsets))
    valid = (
        (expanded_depth > 0.05)
        & (px >= 0) & (px < width)
        & (py >= 0) & (py < height)
    )
    px, py = px[valid], py[valid]
    expanded_depth = expanded_depth[valid]
    expanded_ray_depth = expanded_ray_depth[valid]
    expanded_source = expanded_source[valid]
    if mesh_depth is not None:
        sampled_mesh_depth = mesh_depth[py, px]
        visible = (
            torch.isfinite(sampled_mesh_depth)
            & (sampled_mesh_depth > 0)
            & (expanded_ray_depth <= sampled_mesh_depth + occlusion_tolerance)
        )
        px, py = px[visible], py[visible]
        expanded_depth = expanded_depth[visible]
        expanded_source = expanded_source[visible]
    pixel_ids = py * width + px
    zbuffer = torch.full((height * width,), torch.inf, device=device)
    zbuffer.scatter_reduce_(0, pixel_ids, expanded_depth, reduce="amin", include_self=True)
    nearest = expanded_depth <= zbuffer[pixel_ids] + 1e-6
    pixel_ids = pixel_ids[nearest]
    expanded_source = expanded_source[nearest]
    # For depth ties, the last write is deterministic enough for this smoke renderer.
    image = torch.zeros((height * width, 3), dtype=torch.float32, device=device)
    image[pixel_ids] = colors[expanded_source]
    covered = torch.isfinite(zbuffer)
    result = image.reshape(height, width, 3).clamp(0, 1).cpu().numpy()
    mask = covered.reshape(height, width).cpu().numpy().astype(np.uint8) * 255
    return np.rint(result * 255.0).astype(np.uint8), mask, float(covered.float().mean().cpu())


def main() -> None:
    args = arguments()
    if args.occlusion_tolerance < 0:
        raise ValueError("--occlusion-tolerance must be non-negative")
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    gaussians = np.load(args.gaussians.resolve())
    appearance = np.load(args.appearance.resolve())
    mapping = json.loads(args.semantic_mapping.resolve().read_text(encoding="utf-8"))
    xyz = torch.from_numpy(gaussians["xyz"].astype(np.float32)).to(device)
    appearance_colors = torch.from_numpy(appearance["rgb"].astype(np.float32)).to(device)
    palette = np.zeros((max(spec["id"] for spec in mapping["classes"].values()) + 1, 3), dtype=np.float32)
    for spec in mapping["classes"].values():
        palette[int(spec["id"])] = np.asarray(spec["color"][:3], dtype=np.float32)
    semantic_colors = torch.from_numpy(palette[gaussians["semantic_id"]]).to(device)
    views_root = args.views.resolve()
    pair_id = None
    if args.protocol_manifest:
        view_directories, pair_id = protocol_view_directories(
            args.protocol_manifest, views_root, args.protocol_split
        )
    else:
        view_directories = sorted(path for path in views_root.iterdir() if path.is_dir())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    depth_archive = np.load(args.depth_cache.resolve()) if args.depth_cache else None
    if depth_archive is not None:
        missing_depth = {path.name for path in view_directories} - set(depth_archive.files)
        if missing_depth:
            raise RuntimeError(f"depth cache is missing frozen views: {sorted(missing_depth)}")
    reports = []
    for view_directory in view_directories:
        camera = json.loads((view_directory / "camera.json").read_text(encoding="utf-8"))
        view_output = output / view_directory.name
        view_output.mkdir(parents=True, exist_ok=True)
        entry = {"view": view_directory.name}
        mesh_depth = None
        if depth_archive is not None:
            depth_array = depth_archive[view_directory.name].astype(np.float32)
            if depth_array.shape != (int(camera["height"]), int(camera["width"])):
                raise RuntimeError(f"depth shape mismatch for {view_directory.name}: {depth_array.shape}")
            mesh_depth = torch.from_numpy(depth_array).to(device)
        coverage_mask = None
        if args.mode in ("appearance", "both"):
            image, coverage_mask, coverage = rasterize(
                xyz, appearance_colors, camera, args.point_radius, device, mesh_depth, args.occlusion_tolerance
            )
            Image.fromarray(image).save(view_output / "appearance.png")
            entry["appearance_coverage"] = coverage
        if args.mode in ("semantic", "both"):
            image, semantic_mask, coverage = rasterize(
                xyz, semantic_colors, camera, args.point_radius, device, mesh_depth, args.occlusion_tolerance
            )
            Image.fromarray(image).save(view_output / "semantic.png")
            entry["semantic_coverage"] = coverage
            if coverage_mask is None:
                coverage_mask = semantic_mask
        Image.fromarray(coverage_mask).save(view_output / "coverage.png")
        (view_output / "camera.json").write_text(
            json.dumps(camera, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(entry)
    report = {
        "pair_id": pair_id,
        "protocol_manifest": str(args.protocol_manifest.resolve()) if args.protocol_manifest else None,
        "protocol_split": args.protocol_split if args.protocol_manifest else None,
        "renderer": "portable point-zbuffer smoke renderer",
        "device": str(device),
        "point_radius": args.point_radius,
        "mesh_depth_occlusion_gate": depth_archive is not None,
        "depth_cache": str(args.depth_cache.resolve()) if args.depth_cache else None,
        "occlusion_tolerance_m": args.occlusion_tolerance if depth_archive is not None else None,
        "mode": args.mode,
        "view_count": len(view_directories),
        "views": reports,
        "limitations": [
            "Uses fixed-radius point splats rather than anisotropic Gaussian ellipses.",
            "Intended for semantic/camera/appearance smoke validation only.",
        ],
    }
    (output / "render_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"device": str(device), "view_count": len(view_directories), "mode": args.mode}))


if __name__ == "__main__":
    main()
