"""Extend a frozen pair protocol with separate overlapping consistency views."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REQUIRED = ("rgb.png", "semantic.png", "depth.exr", "normal.exr", "camera.json")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-protocol", type=Path, required=True)
    parser.add_argument("--consistency-multiview", type=Path, required=True)
    parser.add_argument("--render-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def relative(path: Path) -> str:
    try:
        return path.relative_to(ROOT).as_posix()
    except ValueError:
        return str(path)


def main() -> None:
    args = arguments()
    base_path = args.base_protocol.resolve()
    render_manifest_path = args.render_manifest.resolve()
    base = json.loads(base_path.read_text(encoding="utf-8"))
    render_manifest = json.loads(render_manifest_path.read_text(encoding="utf-8"))
    if base.get("status") != "frozen":
        raise RuntimeError("base protocol is not frozen")
    if render_manifest.get("scene_id") != base.get("target_scene_id"):
        raise RuntimeError("consistency render scene differs from protocol target scene")
    if render_manifest.get("donor_reference_count") != 0:
        raise RuntimeError("consistency render must not contain donor references")
    specs = render_manifest.get("target_views", [])
    names = [spec["name"] for spec in specs]
    if len(names) < 2 or len(names) != len(set(names)):
        raise RuntimeError("at least two unique consistency views are required")
    if any(not name.startswith("consistency_") for name in names):
        raise RuntimeError("consistency view names must start with consistency_")
    rooms = {spec["room"] for spec in specs}
    if len(rooms) != 1:
        raise RuntimeError("consistency cameras must be in the same room")

    root = args.consistency_multiview.resolve() / "target_views"
    records = []
    for name in names:
        directory = root / name
        missing = [modality for modality in REQUIRED if not (directory / modality).is_file()]
        if missing:
            raise RuntimeError(f"{directory} is missing: {', '.join(missing)}")
        for modality in REQUIRED:
            path = (directory / modality).resolve()
            records.append({
                "split": "target_consistency",
                "scene_role": "target",
                "view": name,
                "modality": modality,
                "relative_path": relative(path),
                "bytes": path.stat().st_size,
                "sha256": sha256(path),
            })

    protocol = dict(base)
    protocol["schema_version"] = 2
    protocol["base_protocol"] = {"path": relative(base_path), "sha256": sha256(base_path)}
    protocol["selection"] = {**base["selection"], "target_consistency": names}
    protocol["rules"] = {
        **base["rules"],
        "consistency_view_usage": "multi_view_evaluation_only",
        "consistency_views_excluded_from_appearance_training": True,
    }
    protocol["consistency_render_manifest"] = {
        "path": relative(render_manifest_path),
        "sha256": sha256(render_manifest_path),
    }
    protocol["files"] = [*base["files"], *records]

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(protocol, indent=2) + "\n"
    if output.exists():
        existing = json.loads(output.read_text(encoding="utf-8"))
        if existing != protocol:
            raise RuntimeError(f"frozen consistency protocol already differs: {output}")
        print(json.dumps({"status": "unchanged", "output": str(output)}))
        return
    output.write_text(serialized, encoding="utf-8")
    print(json.dumps({"status": "frozen", "views": names, "files_added": len(records), "sha256": sha256(output)}))


if __name__ == "__main__":
    main()
