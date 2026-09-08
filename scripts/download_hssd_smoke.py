"""Download one or more pinned HSSD scenes and write per-scene manifests.

The historical no-argument invocation still downloads the original smoke scene.
For expansion experiments, pass ``--scene-id`` repeatedly or provide a JSON
scene list with ``--scene-list``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

DEFAULT_SCENE_ID = "107734119_175999932"
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DESTINATION_ROOT = ROOT / "data" / "raw" / "hssd"

SHARED_FILES = (
    (
        "hssd/hssd-hab",
        "hssd-hab.scene_dataset_config.json",
        "hssd-hab.scene_dataset_config.json",
    ),
    (
        "hssd/hssd-hab",
        "metadata/hssd_obj_semantics_condensed.csv",
        "hssd_obj_semantics_condensed.csv",
    ),
    (
        "hssd/hssd-hab",
        "semantics/hssd-hab_semantic_lexicon.json",
        "semantic_lexicon.json",
    ),
    ("hssd/hssd-scenes", "README.md", "HSSD_SCENES_README.md"),
    ("hssd/hssd-models", "README.md", "HSSD_MODELS_README.md"),
)


def scene_files(
    scene_id: str, *, include_geometry: bool = True
) -> tuple[tuple[str, str, str], ...]:
    per_scene = (
        (
            "hssd/hssd-hab",
            f"scenes/{scene_id}.scene_instance.json",
            "scene_instance.json",
        ),
        (
            "hssd/hssd-hab",
            f"semantics/scenes/{scene_id}.semantic_config.json",
            "semantic_config.json",
        ),
    )
    geometry = (
        ("hssd/hssd-scenes", f"scenes/{scene_id}.glb", "scene.glb"),
    ) if include_geometry else ()
    return (*geometry, *per_scene, *SHARED_FILES)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scene-id",
        action="append",
        dest="scene_ids",
        help="HSSD scene ID; repeat this option to download multiple scenes.",
    )
    parser.add_argument(
        "--scene-list",
        type=Path,
        help=(
            "JSON file containing either a list of scene IDs or an object with "
            "a 'scene_ids' list. IDs already supplied with --scene-id are merged."
        ),
    )
    parser.add_argument(
        "--destination-root",
        type=Path,
        default=DEFAULT_DESTINATION_ROOT,
        help="Directory that will contain one subdirectory per scene.",
    )
    parser.add_argument(
        "--metadata-only",
        action="store_true",
        help="Download instance/semantic metadata but skip the large scene GLB.",
    )
    return parser.parse_args()


def load_scene_ids(args: argparse.Namespace) -> list[str]:
    scene_ids = list(args.scene_ids or [])
    if args.scene_list:
        payload = json.loads(args.scene_list.resolve().read_text(encoding="utf-8"))
        listed = payload.get("scene_ids") if isinstance(payload, dict) else payload
        if not isinstance(listed, list) or not all(isinstance(item, str) for item in listed):
            raise ValueError("--scene-list must contain a JSON string list or {'scene_ids': [...]}.")
        scene_ids.extend(listed)
    if not scene_ids:
        scene_ids = [DEFAULT_SCENE_ID]

    normalized = []
    for scene_id in scene_ids:
        scene_id = scene_id.strip()
        if not scene_id or any(character in scene_id for character in "/\\"):
            raise ValueError(f"Invalid HSSD scene ID: {scene_id!r}")
        if scene_id not in normalized:
            normalized.append(scene_id)
    return normalized


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_scene(
    scene_id: str, destination_root: Path, *, metadata_only: bool = False
) -> Path:
    try:
        from huggingface_hub import hf_hub_download
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "HSSD download requires huggingface_hub in the active Python environment. "
            "Install it in semreg-gs before running a download."
        ) from error

    destination = destination_root / scene_id
    destination.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema_version": 2,
        "scene_id": scene_id,
        "downloaded_on": date.today().isoformat(),
        "license": "CC BY-NC 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "source": "https://huggingface.co/hssd",
        "download_scope": "metadata_only" if metadata_only else "complete_scene",
        "files": [],
    }

    for repo_id, remote_name, local_name in scene_files(
        scene_id, include_geometry=not metadata_only
    ):
        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_name,
                repo_type="dataset",
            )
        )
        local_path = destination / local_name
        if not local_path.exists() or sha256(local_path) != sha256(cached):
            shutil.copy2(cached, local_path)
        manifest["files"].append(
            {
                "repository": repo_id,
                "remote_path": remote_name,
                "local_path": local_name,
                "bytes": local_path.stat().st_size,
                "sha256": sha256(local_path),
            }
        )
        print(f"ready {scene_id}/{local_name}: {local_path.stat().st_size} bytes")

    manifest_name = "metadata_manifest.json" if metadata_only else "manifest.json"
    manifest_path = destination / manifest_name
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")
    return manifest_path


def main() -> None:
    args = arguments()
    destination_root = args.destination_root.resolve()
    scene_ids = load_scene_ids(args)
    manifests = [
        download_scene(scene_id, destination_root, metadata_only=args.metadata_only)
        for scene_id in scene_ids
    ]
    print(json.dumps({
        "status": "passed",
        "scene_count": len(manifests),
        "download_scope": "metadata_only" if args.metadata_only else "complete_scene",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
