"""Train matched Global-DINO or Semantic-DINO Gaussian appearance decoders.

DINOv2 and CAD geometry stay frozen. Six target views supervise the smoke
decoder; the remaining target views are reserved for evaluation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from train_neutral_gaussians import gather_observations, validate_rgb_views


SH_C0 = 0.28209479177387814
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=("global_dino", "semantic_dino"), required=True)
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--donor-views", type=Path, required=True)
    parser.add_argument("--target-views", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dino-model", default="dinov2_vits14")
    parser.add_argument("--dino-image-size", type=int, default=518)
    parser.add_argument("--train-view-count", type=int, default=6)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--visibility-tolerance", type=float, default=0.03)
    parser.add_argument("--palette-tolerance", type=float, default=40.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class AppearanceDecoder(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.network = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim), torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, hidden_dim), torch.nn.SiLU(),
            torch.nn.Linear(hidden_dim, 3),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.network(features))


def position_features(xyz: torch.Tensor, frequencies: int = 4) -> torch.Tensor:
    features = [xyz]
    for exponent in range(frequencies):
        frequency = (2.0 ** exponent) * torch.pi
        features.extend((torch.sin(xyz * frequency), torch.cos(xyz * frequency)))
    return torch.cat(features, dim=-1)


def load_palette(path: Path) -> tuple[list[str], np.ndarray]:
    mapping = json.loads(path.read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    return [name for name, _ in ordered], np.asarray([spec["color"][:3] for _, spec in ordered], dtype=np.float32) * 255.0


@torch.inference_mode()
def extract_dino_codes(
    donor_directories: list[Path], palette: np.ndarray, model_name: str,
    image_size: int, tolerance: float, device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[int]]:
    if image_size % 14:
        raise ValueError("DINO image size must be divisible by patch size 14")
    model = torch.hub.load("facebookresearch/dinov2", model_name, trust_repo=True)
    model.eval().to(device)
    class_sum = None
    class_count = torch.zeros(len(palette), dtype=torch.float64, device=device)
    global_sum = None
    global_count = 0
    mean = torch.tensor(IMAGENET_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, device=device)[None, :, None, None]
    for directory in donor_directories:
        rgb = torch.from_numpy(np.asarray(Image.open(directory / "rgb.png").convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        semantic = torch.from_numpy(np.asarray(Image.open(directory / "semantic.png").convert("RGB"), dtype=np.float32)).permute(2, 0, 1)[None].to(device)
        resized = F.interpolate(rgb, size=(image_size, image_size), mode="bicubic", align_corners=False)
        output = model.forward_features((resized - mean) / std)
        tokens = output["x_norm_patchtokens"][0]
        grid = image_size // 14
        semantic = F.interpolate(semantic, size=(grid, grid), mode="nearest")[0].permute(1, 2, 0).reshape(-1, 3)
        palette_tensor = torch.from_numpy(palette).to(device)
        distance = torch.linalg.vector_norm(semantic[:, None, :] - palette_tensor[None, :, :], dim=-1)
        labels = distance.argmin(dim=-1)
        valid = distance.min(dim=-1).values <= tolerance
        valid_tokens = tokens[valid]
        if global_sum is None:
            global_sum = torch.zeros(tokens.shape[-1], dtype=torch.float64, device=device)
            class_sum = torch.zeros((len(palette), tokens.shape[-1]), dtype=torch.float64, device=device)
        global_sum += valid_tokens.double().sum(dim=0)
        global_count += int(valid.sum())
        for class_id in range(len(palette)):
            mask = valid & (labels == class_id)
            if mask.any():
                class_sum[class_id] += tokens[mask].double().sum(dim=0)
                class_count[class_id] += int(mask.sum())
    if global_count == 0:
        raise RuntimeError("No valid donor semantic patches were available for DINO pooling")
    global_code = (global_sum / global_count).float()
    class_codes = global_code[None].repeat(len(palette), 1)
    observed = class_count > 0
    class_codes[observed] = (class_sum[observed] / class_count[observed, None]).float()
    return global_code, class_codes, class_count.cpu().long().tolist()


def main() -> None:
    args = arguments()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")
    gaussians = np.load(args.gaussians.resolve())
    xyz = torch.from_numpy(gaussians["xyz"].astype(np.float32)).to(device)
    normal = torch.from_numpy(gaussians["normal"].astype(np.float32)).to(device)
    semantic_id = torch.from_numpy(gaussians["semantic_id"].astype(np.int64)).to(device)
    names, palette = load_palette(args.semantic_mapping.resolve())
    donor_directories = sorted(path for path in args.donor_views.resolve().iterdir() if path.is_dir())
    target_directories = sorted(path for path in args.target_views.resolve().iterdir() if path.is_dir())
    if not donor_directories or len(target_directories) <= args.train_view_count:
        raise RuntimeError("Donor views are missing or no target views remain held out")
    donor_validation = validate_rgb_views(donor_directories)
    train_directories = target_directories[:args.train_view_count]
    held_out_names = [path.name for path in target_directories[args.train_view_count:]]
    target_validation = validate_rgb_views(train_directories)

    global_code, class_codes, patch_counts = extract_dino_codes(
        donor_directories, palette, args.dino_model, args.dino_image_size,
        args.palette_tolerance, device,
    )
    color_sum, counts, view_reports = gather_observations(
        xyz, train_directories, device, args.visibility_tolerance
    )
    observed = counts > 0
    observed_indices = torch.nonzero(observed, as_tuple=False).squeeze(1)
    target_rgb = color_sum[observed] / counts[observed, None]

    xyz_center = xyz.mean(dim=0)
    xyz_scale = (xyz - xyz_center).abs().amax(dim=0).clamp_min(1e-6)
    normalized_xyz = (xyz - xyz_center) / xyz_scale
    geometry = torch.cat((position_features(normalized_xyz), normal), dim=-1)
    semantic_slot = F.one_hot(semantic_id, num_classes=len(names)).float()
    if args.method == "global_dino":
        conditioning = global_code[None].expand(len(xyz), -1)
        semantic_slot = torch.zeros_like(semantic_slot)
    else:
        conditioning = class_codes[semantic_id]
    input_features = torch.cat((geometry, semantic_slot, conditioning), dim=-1)
    decoder = AppearanceDecoder(input_features.shape[-1], args.hidden_dim).to(device)
    optimizer = torch.optim.AdamW(decoder.parameters(), lr=args.learning_rate, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
    history = []
    for step in range(args.steps):
        selection = torch.randint(len(observed_indices), (min(args.batch_size, len(observed_indices)),), device=device)
        gaussian_indices = observed_indices[selection]
        batch_target = target_rgb[selection]
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=device.type == "cuda"):
            prediction = decoder(input_features[gaussian_indices])
            loss = F.l1_loss(prediction, batch_target)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        if step == 0 or (step + 1) % 50 == 0 or step + 1 == args.steps:
            history.append({"step": step + 1, "l1": float(loss.detach().cpu())})
    decoder.eval()
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(xyz), args.batch_size):
            chunks.append(decoder(input_features[start:start + args.batch_size]).float().cpu())
    rgb = torch.cat(chunks).numpy().astype(np.float32)
    sh_dc = ((rgb - 0.5) / SH_C0).astype(np.float32)
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "appearance.npz", rgb=rgb, sh_dc=sh_dc, observation_count=counts.cpu().numpy().astype(np.int16))
    torch.save({"decoder": decoder.state_dict(), "method": args.method, "dino_model": args.dino_model, "geometry_frozen": True}, output / "checkpoint.pt")
    report = {
        "method": args.method,
        "device": str(device),
        "dino_model": args.dino_model,
        "dino_image_size": args.dino_image_size,
        "dino_feature_dim": int(global_code.numel()),
        "donor_validation": donor_validation,
        "target_train_validation": target_validation,
        "train_views": [path.name for path in train_directories],
        "held_out_views": held_out_names,
        "class_patch_count": {name: patch_counts[index] for index, name in enumerate(names)},
        "missing_classes_fallback_to_global": [name for index, name in enumerate(names) if patch_counts[index] == 0],
        "gaussian_count": int(len(xyz)),
        "observed_gaussians": int(observed.sum()),
        "optimized_parameters": ["appearance_decoder"],
        "frozen_parameters": ["DINOv2", "xyz", "normal", "rotation_wxyz", "scale", "semantic_id"],
        "steps": args.steps,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "loss_history": history,
        "views": view_reports,
        "warning": "Single-scene supervised smoke decoder; cross-geometry claims require training across donor-target pairs and evaluation on unseen pairs.",
    }
    (output / "training_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    # Keep evaluator compatibility with the non-learning baselines.
    prototype_rgb = {name: rgb[semantic_id.cpu().numpy() == index].mean(axis=0).tolist() for index, name in enumerate(names)}
    (output / "baseline_report.json").write_text(json.dumps({"method": args.method, "class_mean_rgb": prototype_rgb}, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"method": args.method, "observed_gaussians": int(observed.sum()), "held_out_views": held_out_names, "final_l1": history[-1]["l1"]}))


if __name__ == "__main__":
    main()
