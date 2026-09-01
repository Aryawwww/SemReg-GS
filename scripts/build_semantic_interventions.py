"""Create controlled single-class appearance edits for controllability evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


SH_C0 = 0.28209479177387814
METHODS = ("global", "semantic_2d", "global_dino", "semantic_dino")
SEMANTIC_METHODS = {"semantic_2d", "semantic_dino"}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol-manifest", type=Path, required=True)
    parser.add_argument("--gaussians", type=Path, required=True)
    parser.add_argument("--semantic-mapping", type=Path, required=True)
    parser.add_argument("--appearance-2d", type=Path, required=True)
    parser.add_argument("--appearance-dino", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--delta", nargs=3, type=float, default=(0.20, -0.15, 0.10))
    return parser.parse_args()


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    args = arguments()
    protocol_path = args.protocol_manifest.resolve()
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen" or protocol.get("rules", {}).get("appearance_source") != "source_donor_only":
        raise RuntimeError("a frozen source-donor-only protocol is required")
    mapping = json.loads(args.semantic_mapping.resolve().read_text(encoding="utf-8"))
    ordered = sorted(mapping["classes"].items(), key=lambda item: int(item[1]["id"]))
    names = [name for name, _ in ordered]
    semantic_id = np.load(args.gaussians.resolve())["semantic_id"].astype(np.int64)
    delta = np.asarray(args.delta, dtype=np.float32)
    if not np.any(delta) or np.max(np.abs(delta)) > 1.0:
        raise ValueError("--delta must be non-zero normalized RGB offsets within [-1, 1]")
    roots = {
        "global": args.appearance_2d.resolve(),
        "semantic_2d": args.appearance_2d.resolve(),
        "global_dino": args.appearance_dino.resolve(),
        "semantic_dino": args.appearance_dino.resolve(),
    }
    output = args.output.resolve()
    entries = []
    for method in METHODS:
        source = roots[method] / method / "appearance.npz"
        archive = np.load(source)
        base = archive["rgb"].astype(np.float32)
        if len(base) != len(semantic_id):
            raise RuntimeError(f"{method} Gaussian count mismatch")
        for class_id, class_name in enumerate(names):
            mask = semantic_id == class_id if method in SEMANTIC_METHODS else np.ones(len(base), dtype=bool)
            edited = base.copy()
            edited[mask] = np.clip(edited[mask] + delta, 0.0, 1.0)
            directory = output / method / class_name
            directory.mkdir(parents=True, exist_ok=True)
            appearance_path = directory / "appearance.npz"
            np.savez_compressed(appearance_path, rgb=edited, sh_dc=((edited - 0.5) / SH_C0).astype(np.float32))
            entry = {
                "method": method,
                "edited_class": class_name,
                "edit_scope": "selected_semantic_gaussians" if method in SEMANTIC_METHODS else "all_gaussians_global_method",
                "edited_gaussians": int(mask.sum()),
                "appearance": str(appearance_path),
                "appearance_sha256": sha256(appearance_path),
            }
            (directory / "intervention_report.json").write_text(
                json.dumps({**entry, "pair_id": protocol["pair_id"], "rgb_delta": delta.tolist()}, indent=2) + "\n",
                encoding="utf-8",
            )
            entries.append(entry)
    manifest = {
        "schema_version": 1,
        "status": "passed",
        "pair_id": protocol["pair_id"],
        "protocol_manifest_sha256": sha256(protocol_path),
        "rgb_delta": delta.tolist(),
        "global_baseline_semantics": "Global methods have no class selector, so the same requested edit changes every Gaussian.",
        "entries": entries,
    }
    (output / "intervention_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": "passed", "interventions": len(entries)}))


if __name__ == "__main__":
    main()
