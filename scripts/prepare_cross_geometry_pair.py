"""Select/download a second HSSD scene and create a cross-geometry pair manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

from huggingface_hub import HfApi, hf_hub_download


ROOT = Path(__file__).resolve().parents[1]


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-scene-id", required=True)
    parser.add_argument("--source-split", type=Path, required=True)
    parser.add_argument("--target-scene-id")
    parser.add_argument("--selection-seed", type=int, default=42)
    parser.add_argument("--output-root", type=Path, default=ROOT / "data" / "raw" / "hssd")
    parser.add_argument("--pair-root", type=Path, default=ROOT / "data" / "processed" / "pairs")
    parser.add_argument("--select-only", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scene_ids(api: HfApi) -> list[str]:
    scene_files = api.list_repo_files("hssd/hssd-scenes", repo_type="dataset")
    hab_files = api.list_repo_files("hssd/hssd-hab", repo_type="dataset")
    glb = {Path(path).stem for path in scene_files if path.startswith("scenes/") and path.endswith(".glb")}
    semantics = {
        Path(path).name.removesuffix(".semantic_config.json")
        for path in hab_files
        if path.startswith("semantics/scenes/") and path.endswith(".semantic_config.json")
    }
    instances = {
        Path(path).name.removesuffix(".scene_instance.json")
        for path in hab_files
        if path.startswith("scenes/") and path.endswith(".scene_instance.json")
    }
    return sorted(glb & semantics & instances)


def deterministic_target(candidates: list[str], source: str, seed: int) -> str:
    ranked = sorted(
        (scene for scene in candidates if scene != source),
        key=lambda scene: hashlib.sha256(f"{seed}:{source}:{scene}".encode()).hexdigest(),
    )
    if not ranked:
        raise RuntimeError("No second complete HSSD scene is available")
    return ranked[0]


def download_files(scene_id: str, destination: Path) -> list[dict]:
    files = (
        ("hssd/hssd-scenes", f"scenes/{scene_id}.glb", "scene.glb"),
        ("hssd/hssd-hab", f"scenes/{scene_id}.scene_instance.json", "scene_instance.json"),
        ("hssd/hssd-hab", f"semantics/scenes/{scene_id}.semantic_config.json", "semantic_config.json"),
        ("hssd/hssd-hab", "hssd-hab.scene_dataset_config.json", "hssd-hab.scene_dataset_config.json"),
        ("hssd/hssd-hab", "metadata/hssd_obj_semantics_condensed.csv", "hssd_obj_semantics_condensed.csv"),
        ("hssd/hssd-hab", "semantics/hssd-hab_semantic_lexicon.json", "semantic_lexicon.json"),
    )
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for repository, remote_path, local_name in files:
        cached = Path(hf_hub_download(repo_id=repository, filename=remote_path, repo_type="dataset"))
        local_path = destination / local_name
        shutil.copy2(cached, local_path)
        records.append({
            "repository": repository,
            "remote_path": remote_path,
            "local_path": local_name,
            "bytes": local_path.stat().st_size,
            "sha256": sha256(local_path),
        })
        print(f"downloaded {local_name}: {local_path.stat().st_size} bytes")
    return records


def main() -> None:
    args = arguments()
    source_split = args.source_split.resolve()
    split = json.loads(source_split.read_text(encoding="utf-8"))
    if split.get("status") != "frozen" or split.get("scene_id") != args.source_scene_id:
        raise RuntimeError("Source split is not frozen or scene ID does not match")
    api = HfApi()
    candidates = scene_ids(api)
    target = args.target_scene_id or deterministic_target(candidates, args.source_scene_id, args.selection_seed)
    if target == args.source_scene_id or target not in candidates:
        raise RuntimeError("Target scene must be different and contain GLB, instance, and semantic config files")
    pair_id = f"{args.source_scene_id}__to__{target}"
    target_directory = args.output_root.resolve() / target
    pair_directory = args.pair_root.resolve() / pair_id
    selection = {
        "schema_version": 1,
        "pair_id": pair_id,
        "source_scene_id": args.source_scene_id,
        "target_scene_id": target,
        "selection_seed": args.selection_seed,
        "complete_scene_candidate_count": len(candidates),
        "source_split": str(source_split),
        "source_split_sha256": sha256(source_split),
        "target_directory": str(target_directory),
    }
    pair_directory.mkdir(parents=True, exist_ok=True)
    if args.select_only:
        selection["status"] = "selected_not_downloaded"
        (pair_directory / "pair_manifest.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
        print(json.dumps(selection))
        return
    records = download_files(target, target_directory)
    raw_manifest = {
        "scene_id": target,
        "downloaded_on": date.today().isoformat(),
        "license": "CC BY-NC 4.0",
        "source": "https://huggingface.co/hssd",
        "files": records,
    }
    raw_manifest_path = target_directory / "manifest.json"
    raw_manifest_path.write_text(json.dumps(raw_manifest, indent=2) + "\n", encoding="utf-8")
    selection.update({
        "status": "downloaded_pending_audit",
        "target_raw_manifest": str(raw_manifest_path),
        "target_raw_manifest_sha256": sha256(raw_manifest_path),
        "next_required_steps": ["asset_audit", "semantic_mapping", "multiview_render", "coverage_audit", "semantic_gaussian_initialization"],
    })
    (pair_directory / "pair_manifest.json").write_text(json.dumps(selection, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"pair_id": pair_id, "target_scene_id": target, "status": selection["status"]}))


if __name__ == "__main__":
    main()
