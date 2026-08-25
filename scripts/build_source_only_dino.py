"""Build Global-DINO and Semantic-DINO appearances using frozen source donors only."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


SH_C0 = 0.28209479177387814
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--donor-views", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--dino-repo", type=Path, required=True)
    parser.add_argument("--dino-weights", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dino-model", default="dinov2_vits14")
    parser.add_argument("--image-size", type=int, default=518)
    parser.add_argument("--ridge", type=float, default=1e-2)
    parser.add_argument("--palette-tolerance", type=float, default=40.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def frozen_donors(protocol_path: Path, root: Path) -> tuple[dict, list[Path]]:
    protocol = json.loads(protocol_path.resolve().read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen" or protocol.get("rules", {}).get("appearance_source") != "source_donor_only":
        raise RuntimeError("protocol is not frozen source-donor-only")
    records = {
        (record["view"], record["modality"]): record
        for record in protocol["files"] if record["split"] == "source_donor"
    }
    directories = []
    for name in protocol["selection"]["source_donor"]:
        directory = root.resolve() / name
        for modality in ("rgb.png", "semantic.png"):
            path = directory / modality
            record = records.get((name, modality))
            if record is None or not path.is_file() or path.stat().st_size != record["bytes"] or sha256(path) != record["sha256"]:
                raise RuntimeError(f"frozen donor differs: {name}/{modality}")
        directories.append(directory)
    return protocol, directories


def save_appearance(path: Path, rgb: np.ndarray, report: dict) -> None:
    path.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path / "appearance.npz", rgb=rgb, sh_dc=((rgb - 0.5) / SH_C0).astype(np.float32))
    (path / "baseline_report.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")


@torch.inference_mode()
def main() -> None:
    args = arguments()
    if args.image_size % 14:
        raise ValueError("--image-size must be divisible by DINO patch size 14")
    if args.ridge <= 0:
        raise ValueError("--ridge must be positive")
    protocol, donors = frozen_donors(args.protocol_manifest, args.donor_views)
    mapping = json.loads(args.semantic_mapping.resolve().read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    palette = torch.tensor([spec["color"][:3] for _, spec in ordered], dtype=torch.float32) * 255.0
    gaussians = np.load(args.gaussians.resolve())
    semantic_id = gaussians["semantic_id"].astype(np.int64)
    device = torch.device(args.device if args.device != "cuda" or torch.cuda.is_available() else "cpu")

    weights_path = args.dino_weights.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(weights_path)
    model = torch.hub.load(str(args.dino_repo.resolve()), args.dino_model, source="local", pretrained=False)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint, strict=True)
    model.eval().to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device)[None, :, None, None]
    std = torch.tensor(IMAGENET_STD, device=device)[None, :, None, None]
    palette = palette.to(device)
    grid = args.image_size // 14
    all_tokens, all_rgb, all_labels = [], [], []
    donor_reports = []
    for directory in donors:
        rgb = torch.from_numpy(np.asarray(Image.open(directory / "rgb.png").convert("RGB"), dtype=np.float32) / 255.0).permute(2, 0, 1)[None].to(device)
        semantic = torch.from_numpy(np.asarray(Image.open(directory / "semantic.png").convert("RGB"), dtype=np.float32)).permute(2, 0, 1)[None].to(device)
        resized = F.interpolate(rgb, size=(args.image_size, args.image_size), mode="bicubic", align_corners=False)
        tokens = model.forward_features((resized - mean) / std)["x_norm_patchtokens"][0]
        patch_rgb = F.interpolate(rgb, size=(grid, grid), mode="area")[0].permute(1, 2, 0).reshape(-1, 3)
        patch_semantic = F.interpolate(semantic, size=(grid, grid), mode="nearest")[0].permute(1, 2, 0).reshape(-1, 3)
        distance = torch.linalg.vector_norm(patch_semantic[:, None, :] - palette[None, :, :], dim=-1)
        labels = distance.argmin(dim=-1)
        valid = distance.min(dim=-1).values <= args.palette_tolerance
        all_tokens.append(tokens[valid].float().cpu())
        all_rgb.append(patch_rgb[valid].float().cpu())
        all_labels.append(labels[valid].cpu())
        donor_reports.append({"view": directory.name, "valid_patches": int(valid.sum()), "rgb_sha256": sha256(directory / "rgb.png")})
    tokens = torch.cat(all_tokens).double()
    targets = torch.cat(all_rgb).double()
    labels = torch.cat(all_labels).long()
    design = torch.cat((tokens, torch.ones((len(tokens), 1), dtype=torch.float64)), dim=1)
    regularizer = torch.eye(design.shape[1], dtype=torch.float64) * args.ridge
    regularizer[-1, -1] = 0.0
    coefficients = torch.linalg.solve(design.T @ design + regularizer, design.T @ targets)
    global_code = tokens.mean(dim=0)
    class_codes = global_code[None].repeat(len(names), 1)
    patch_counts = []
    for class_id in range(len(names)):
        mask = labels == class_id
        patch_counts.append(int(mask.sum()))
        if mask.any():
            class_codes[class_id] = tokens[mask].mean(dim=0)
    def decode(codes: torch.Tensor) -> np.ndarray:
        augmented = torch.cat((codes.double(), torch.ones((len(codes), 1), dtype=torch.float64)), dim=1)
        return (augmented @ coefficients).clamp(0, 1).float().numpy()
    global_color = decode(global_code[None])[0]
    class_colors = decode(class_codes)
    global_rgb = np.repeat(global_color[None, :], len(semantic_id), axis=0).astype(np.float32)
    semantic_rgb = class_colors[semantic_id].astype(np.float32)
    common = {
        "source": "frozen source donor RGB/semantic and DINO patch tokens only; no target RGB",
        "pair_id": protocol["pair_id"],
        "protocol_manifest": str(args.protocol_manifest.resolve()),
        "donor_view_count": len(donors),
        "donor_views": donor_reports,
        "gaussian_count": int(len(semantic_id)),
        "dino_model": args.dino_model,
        "dino_repo": str(args.dino_repo.resolve()),
        "dino_weights": str(weights_path),
        "dino_weights_sha256": sha256(weights_path),
        "feature_dim": int(tokens.shape[1]),
        "ridge": args.ridge,
        "training_samples": int(len(tokens)),
        "optimized_parameters": ["donor_patch_ridge_rgb_decoder"],
        "target_rgb_accessed": False,
        "class_patch_count": {name: patch_counts[index] for index, name in enumerate(names)},
        "missing_classes_fallback_to_global": [name for index, name in enumerate(names) if patch_counts[index] == 0],
    }
    output = args.output.resolve()
    global_prototypes = {name: global_color.tolist() for name in names}
    semantic_prototypes = {name: class_colors[index].tolist() for index, name in enumerate(names)}
    save_appearance(output / "global_dino", global_rgb, {"method": "Global-DINO source-only", "class_mean_rgb": global_prototypes, **common})
    save_appearance(output / "semantic_dino", semantic_rgb, {"method": "Semantic-DINO source-only", "class_mean_rgb": semantic_prototypes, **common})
    print(json.dumps({"status": "passed", "donor_views": len(donors), "training_samples": len(tokens), "class_patch_count": common["class_patch_count"]}))


if __name__ == "__main__":
    main()
