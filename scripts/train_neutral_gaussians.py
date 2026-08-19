"""Train an appearance-only neutral CAD-GS smoke baseline.

Geometry is immutable. Visibility is estimated with a point z-buffer and only
per-Gaussian RGB logits are optimized. This is intentionally a portable smoke
baseline, not a replacement for the CUDA Gaussian rasterizer used later.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image


SH_C0 = 0.28209479177387814


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--views", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.05)
    parser.add_argument("--visibility-tolerance", type=float, default=0.03)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--minimum-observed-fraction", type=float, default=0.50)
    return parser.parse_args()


def validate_rgb_views(view_directories: list[Path]) -> dict:
    records = []
    hashes = set()
    errors = []
    for view_directory in view_directories:
        image_path = view_directory / "rgb.png"
        camera_path = view_directory / "camera.json"
        if not image_path.is_file() or not camera_path.is_file():
            errors.append(f"{view_directory.name}: missing rgb.png or camera.json")
            continue
        camera = json.loads(camera_path.read_text(encoding="utf-8"))
        image = np.asarray(Image.open(image_path).convert("RGB"), dtype=np.uint8)
        digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
        hashes.add(digest)
        unique_values = int(np.unique(image).size)
        record = {
            "view": view_directory.name,
            "width": int(image.shape[1]),
            "height": int(image.shape[0]),
            "minimum": int(image.min()),
            "maximum": int(image.max()),
            "unique_8bit_values": unique_values,
            "sha256": digest,
        }
        records.append(record)
        if image.shape[1] != int(camera["width"]) or image.shape[0] != int(camera["height"]):
            errors.append(f"{view_directory.name}: RGB and camera dimensions differ")
        if record["maximum"] <= 1 and unique_values <= 4:
            errors.append(f"{view_directory.name}: RGB is effectively a 0/1 image")
    if len(records) != len(view_directories):
        errors.append("one or more view directories are incomplete")
    if len(hashes) < 2:
        errors.append("all RGB views have identical file hashes")
    if records and max(record["unique_8bit_values"] for record in records) < 16:
        errors.append("the RGB view set has insufficient color variation")
    report = {
        "status": "passed" if not errors else "failed",
        "view_count": len(records),
        "distinct_file_hashes": len(hashes),
        "views": records,
        "errors": errors,
    }
    if errors:
        raise RuntimeError("RGB validation failed: " + "; ".join(errors))
    return report


def project(
    xyz: torch.Tensor, camera: dict, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    world_to_camera = torch.tensor(
        camera["world_to_camera_blender"], dtype=torch.float32, device=device
    )
    homogeneous = torch.cat(
        (xyz, torch.ones((len(xyz), 1), dtype=xyz.dtype, device=device)), dim=1
    )
    camera_xyz = homogeneous @ world_to_camera.T
    depth = -camera_xyz[:, 2]
    intrinsics = camera["intrinsics"]
    u = intrinsics["fx"] * camera_xyz[:, 0] / depth.clamp_min(1e-8) + intrinsics["cx"]
    v = intrinsics["cy"] - intrinsics["fy"] * camera_xyz[:, 1] / depth.clamp_min(1e-8)
    return u, v, depth


def gather_observations(
    xyz: torch.Tensor,
    view_directories: list[Path],
    device: torch.device,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, list[dict]]:
    count = len(xyz)
    color_sum = torch.zeros((count, 3), dtype=torch.float32, device=device)
    observation_count = torch.zeros(count, dtype=torch.float32, device=device)
    view_reports = []
    for view_directory in view_directories:
        camera = json.loads((view_directory / "camera.json").read_text(encoding="utf-8"))
        image_np = np.asarray(Image.open(view_directory / "rgb.png").convert("RGB"), dtype=np.float32) / 255.0
        image = torch.from_numpy(image_np).to(device)
        height, width = image.shape[:2]
        u, v, depth = project(xyz, camera, device)
        px = torch.round(u).long()
        py = torch.round(v).long()
        valid = (
            (depth > 0.05)
            & (px >= 0) & (px < width)
            & (py >= 0) & (py < height)
        )
        indices = torch.nonzero(valid, as_tuple=False).squeeze(1)
        pixel_ids = py[indices] * width + px[indices]
        zbuffer = torch.full((height * width,), torch.inf, dtype=torch.float32, device=device)
        zbuffer.scatter_reduce_(0, pixel_ids, depth[indices], reduce="amin", include_self=True)
        visible = depth[indices] <= zbuffer[pixel_ids] + tolerance
        indices = indices[visible]
        sampled = image[py[indices], px[indices]]
        color_sum.index_add_(0, indices, sampled)
        observation_count.index_add_(0, indices, torch.ones(len(indices), device=device))
        view_reports.append(
            {
                "view": view_directory.name,
                "projected": int(valid.sum().item()),
                "visible": int(len(indices)),
            }
        )
    return color_sum, observation_count, view_reports


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    archive = np.load(args.gaussians.resolve())
    xyz = torch.from_numpy(archive["xyz"].astype(np.float32)).to(device)
    xyz.requires_grad_(False)
    view_directories = sorted(path for path in args.views.resolve().iterdir() if path.is_dir())
    if not view_directories:
        raise RuntimeError("No view directories were found")
    rgb_validation = validate_rgb_views(view_directories)

    color_sum, counts, view_reports = gather_observations(
        xyz, view_directories, device, args.visibility_tolerance
    )
    observed = counts > 0
    observed_fraction = float(observed.float().mean().item())
    if not observed.any():
        raise RuntimeError("No Gaussians were observed from the supplied cameras")
    if observed_fraction < args.minimum_observed_fraction:
        raise RuntimeError(
            f"Observed Gaussian fraction {observed_fraction:.4f} is below "
            f"the required {args.minimum_observed_fraction:.4f}"
        )
    target_rgb = torch.full_like(color_sum, 0.5)
    target_rgb[observed] = color_sum[observed] / counts[observed, None]
    rgb_logits = torch.nn.Parameter(torch.zeros_like(target_rgb))
    optimizer = torch.optim.Adam([rgb_logits], lr=args.learning_rate)
    history = []
    with torch.no_grad():
        initial_l1 = torch.mean(torch.abs(torch.sigmoid(rgb_logits)[observed] - target_rgb[observed]))
        initial_mse = torch.mean((torch.sigmoid(rgb_logits)[observed] - target_rgb[observed]) ** 2)
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        rgb = torch.sigmoid(rgb_logits)
        loss = torch.mean(torch.abs(rgb[observed] - target_rgb[observed]))
        loss.backward()
        optimizer.step()
        if step == 0 or (step + 1) % 25 == 0 or step + 1 == args.steps:
            history.append({"step": step + 1, "l1": float(loss.detach().cpu())})

    rgb = torch.sigmoid(rgb_logits).detach().cpu().numpy().astype(np.float32)
    with torch.no_grad():
        final_rgb = torch.sigmoid(rgb_logits)[observed]
        final_l1 = torch.mean(torch.abs(final_rgb - target_rgb[observed]))
        final_mse = torch.mean((final_rgb - target_rgb[observed]) ** 2)
        final_psnr = -10.0 * torch.log10(final_mse.clamp_min(1e-12))
    sh_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "appearance.npz",
        rgb=rgb,
        sh_dc=sh_dc,
        observation_count=counts.detach().cpu().numpy().astype(np.int16),
    )
    torch.save(
        {
            "rgb": torch.from_numpy(rgb),
            "sh_dc": torch.from_numpy(sh_dc),
            "geometry_frozen": True,
            "source_gaussians": str(args.gaussians.resolve()),
        },
        output / "checkpoint.pt",
    )
    report = {
        "baseline": "Neutral CAD-GS portable point-zbuffer smoke baseline",
        "device": str(device),
        "gaussian_count": int(len(xyz)),
        "view_count": len(view_directories),
        "observed_gaussians": int(observed.sum().item()),
        "unobserved_gaussians": int((~observed).sum().item()),
        "observed_fraction": observed_fraction,
        "minimum_observed_fraction": args.minimum_observed_fraction,
        "rgb_validation": rgb_validation,
        "visibility_tolerance_m": args.visibility_tolerance,
        "optimized_parameters": ["rgb", "sh_dc"],
        "frozen_parameters": ["xyz", "normal", "rotation_wxyz", "scale", "semantic_id"],
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "metrics": {
            "initial_l1": float(initial_l1.cpu()),
            "initial_mse": float(initial_mse.cpu()),
            "final_l1": float(final_l1.cpu()),
            "final_mse": float(final_mse.cpu()),
            "final_psnr_db": float(final_psnr.cpu()),
        },
        "loss_history": history,
        "views": view_reports,
        "limitations": [
            "Uses point projection and a z-buffer, not anisotropic CUDA Gaussian rasterization.",
            "Unobserved Gaussians retain neutral gray RGB.",
            "This baseline validates appearance-only optimization and camera alignment only.",
        ],
    }
    (output / "training_report.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({key: report[key] for key in (
        "device", "gaussian_count", "view_count", "observed_gaussians",
        "observed_fraction", "steps"
    )}))


if __name__ == "__main__":
    main()
