"""Validate CAD anchoring and fields in a semantic Gaussian initialization."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--source-mesh", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--max-distance", type=float, default=1e-5)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    args = arguments()
    input_directory = args.input.resolve()
    metadata = json.loads((input_directory / "metadata.json").read_text(encoding="utf-8"))
    archive = np.load(input_directory / "gaussians.npz")
    required = {
        "xyz": (3,), "normal": (3,), "rotation_wxyz": (4,), "scale": (3,),
        "opacity": (), "semantic_id": (), "source_face_id": (),
        "source_triangle": (3, 3), "barycentric": (3,),
    }
    errors: list[str] = []
    count = len(archive["xyz"])
    for field, trailing_shape in required.items():
        if field not in archive:
            errors.append(f"missing field: {field}")
            continue
        if archive[field].shape != (count, *trailing_shape):
            errors.append(f"invalid shape for {field}: {archive[field].shape}")
        if not np.isfinite(archive[field]).all():
            errors.append(f"non-finite values in {field}")

    xyz = archive["xyz"].astype(np.float64)
    triangles = archive["source_triangle"].astype(np.float64)
    barycentric = archive["barycentric"].astype(np.float64)
    reconstructed = np.einsum("ni,nij->nj", barycentric, triangles)
    source_distances = np.linalg.norm(xyz - reconstructed, axis=1)
    normal_norm_error = np.abs(np.linalg.norm(archive["normal"], axis=1) - 1.0)
    rotation_norm_error = np.abs(np.linalg.norm(archive["rotation_wxyz"], axis=1) - 1.0)
    barycentric_sum_error = np.abs(barycentric.sum(axis=1) - 1.0)
    barycentric_min = float(barycentric.min())
    allowed_ids = set(int(value) for value in metadata["class_ids"].values())
    observed_ids = set(int(value) for value in np.unique(archive["semantic_id"]))
    if not observed_ids.issubset(allowed_ids):
        errors.append(f"unknown semantic IDs: {sorted(observed_ids - allowed_ids)}")
    if barycentric_min < -1e-6 or float(barycentric.max()) > 1.0 + 1e-6:
        errors.append("sample lies outside its source triangle")
    if float(source_distances.max()) > args.max_distance:
        errors.append("center-to-source-triangle distance exceeds threshold")
    if float(normal_norm_error.max()) > 1e-5:
        errors.append("surface normals are not unit length")
    if float(rotation_norm_error.max()) > 1e-5:
        errors.append("rotation quaternions are not unit length")
    if float(barycentric_sum_error.max()) > 1e-5:
        errors.append("barycentric coordinates do not sum to one")
    if not np.all(archive["scale"] > 0):
        errors.append("scale must be strictly positive")
    if not np.all((archive["opacity"] > 0) & (archive["opacity"] < 1)):
        errors.append("opacity must lie strictly between zero and one")
    mesh_hash = sha256(args.source_mesh.resolve())
    if mesh_hash != metadata["source_mesh_sha256"]:
        errors.append("source mesh SHA-256 differs from conversion metadata")
    if count != int(metadata["gaussian_count"]):
        errors.append("Gaussian count differs from metadata")

    semantic_counts = {
        name: int(np.count_nonzero(archive["semantic_id"] == class_id))
        for name, class_id in metadata["class_ids"].items()
    }
    report = {
        "status": "passed" if not errors else "failed",
        "gaussian_count": count,
        "required_fields": sorted(required),
        "semantic_counts": semantic_counts,
        "observed_semantic_ids": sorted(observed_ids),
        "center_to_source_triangle_m": {
            "mean": float(source_distances.mean()),
            "p95": float(np.percentile(source_distances, 95)),
            "maximum": float(source_distances.max()),
            "threshold": args.max_distance,
        },
        "normal_norm_error_maximum": float(normal_norm_error.max()),
        "rotation_norm_error_maximum": float(rotation_norm_error.max()),
        "barycentric_sum_error_maximum": float(barycentric_sum_error.max()),
        "barycentric_minimum": barycentric_min,
        "source_mesh_sha256_matches": mesh_hash == metadata["source_mesh_sha256"],
        "freeze": metadata["freeze"],
        "errors": errors,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
