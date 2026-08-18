"""Download the pinned single-scene HSSD smoke-test subset."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import date
from pathlib import Path

from huggingface_hub import hf_hub_download


SCENE_ID = "107734119_175999932"
ROOT = Path(__file__).resolve().parents[1]
DESTINATION = ROOT / "data" / "raw" / "hssd" / SCENE_ID

FILES = (
    ("hssd/hssd-scenes", f"scenes/{SCENE_ID}.glb", "scene.glb"),
    (
        "hssd/hssd-hab",
        f"scenes/{SCENE_ID}.scene_instance.json",
        "scene_instance.json",
    ),
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
        f"semantics/scenes/{SCENE_ID}.semantic_config.json",
        "semantic_config.json",
    ),
    (
        "hssd/hssd-hab",
        "semantics/hssd-hab_semantic_lexicon.json",
        "semantic_lexicon.json",
    ),
    ("hssd/hssd-scenes", "README.md", "HSSD_SCENES_README.md"),
    ("hssd/hssd-models", "README.md", "HSSD_MODELS_README.md"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    DESTINATION.mkdir(parents=True, exist_ok=True)
    manifest = {
        "scene_id": SCENE_ID,
        "downloaded_on": date.today().isoformat(),
        "license": "CC BY-NC 4.0",
        "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
        "source": "https://huggingface.co/hssd",
        "files": [],
    }

    for repo_id, remote_name, local_name in FILES:
        cached = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=remote_name,
                repo_type="dataset",
            )
        )
        destination = DESTINATION / local_name
        shutil.copy2(cached, destination)
        manifest["files"].append(
            {
                "repository": repo_id,
                "remote_path": remote_name,
                "local_path": local_name,
                "bytes": destination.stat().st_size,
                "sha256": sha256(destination),
            }
        )
        print(f"downloaded {local_name}: {destination.stat().st_size} bytes")

    manifest_path = DESTINATION / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()
