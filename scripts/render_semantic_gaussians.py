"""Render neutral appearance or semantic colors from initialized Gaussians."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--appearance", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", choices=("appearance", "semantic", "both"), default="both")
    parser.add_argument("--point-radius", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


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
) -> tuple[np.ndarray, float]:
    width, height = int(camera["width"]), int(camera["height"])
    u, v, depth = project(xyz, camera, device)
    base_x = torch.round(u).long()
    base_y = torch.round(v).long()
    source_ids = torch.arange(len(xyz), device=device)
    offsets = [(dx, dy) for dy in range(-radius, radius + 1) for dx in range(-radius, radius + 1)]
    px = torch.cat([base_x + dx for dx, _ in offsets])
    py = torch.cat([base_y + dy for _, dy in offsets])
    expanded_depth = depth.repeat(len(offsets))
    expanded_source = source_ids.repeat(len(offsets))
    valid = (
        (expanded_depth > 0.05)
        & (px >= 0) & (px < width)
        & (py >= 0) & (py < height)
    )
    px, py = px[valid], py[valid]
    expanded_depth = expanded_depth[valid]
    expanded_source = expanded_source[valid]
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
    return np.rint(result * 255.0).astype(np.uint8), float(covered.float().mean().cpu())


def main() -> None:
    args = arguments()
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
    view_directories = sorted(path for path in args.views.resolve().iterdir() if path.is_dir())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    reports = []
    for view_directory in view_directories:
        camera = json.loads((view_directory / "camera.json").read_text(encoding="utf-8"))
        view_output = output / view_directory.name
        view_output.mkdir(parents=True, exist_ok=True)
        entry = {"view": view_directory.name}
        if args.mode in ("appearance", "both"):
            image, coverage = rasterize(xyz, appearance_colors, camera, args.point_radius, device)
            Image.fromarray(image).save(view_output / "appearance.png")
            entry["appearance_coverage"] = coverage
        if args.mode in ("semantic", "both"):
            image, coverage = rasterize(xyz, semantic_colors, camera, args.point_radius, device)
            Image.fromarray(image).save(view_output / "semantic.png")
            entry["semantic_coverage"] = coverage
        (view_output / "camera.json").write_text(
            json.dumps(camera, indent=2) + "\n", encoding="utf-8"
        )
        reports.append(entry)
    report = {
        "renderer": "portable point-zbuffer smoke renderer",
        "device": str(device),
        "point_radius": args.point_radius,
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
